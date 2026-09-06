from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from numbers import Real
from typing import Any, Dict, Optional

from .freshness import QUOTE_LIVE_SECONDS, QUOTE_STALE_SECONDS, standard_weekend_window
from .market_metadata import (
    TICK_FUTURE_TOLERANCE_SECONDS,
    build_tick_freshness_context,
)
from .tick_flags import bid_ask_flags

QUOTE_EXECUTION_READINESS_BASIS = (
    "quote_age_market_session_and_positive_spread"
)
QUOTE_EXECUTION_SOURCE_AGREEMENT_BASIS = (
    "quote_age_market_session_spread_and_source_agreement"
)
MAX_BROKER_TICK_CLOCK_RECONCILIATION_SECONDS = 30.0


def tick_clock_reference(
    epoch: Optional[float],
    *,
    wall_clock_epoch: float,
) -> Dict[str, Any]:
    """Choose the clock used for a synchronously acquired trading tick.

    A modest positive lead is normally workstation clock lag, not a forecast
    from the broker. Reconcile that bounded case to the acquired broker tick;
    larger leads remain invalid and continue to fail closed.
    """
    wall_epoch = float(wall_clock_epoch)
    out: Dict[str, Any] = {
        "reference_epoch": wall_epoch,
        "clock_reference": "wall_clock",
        "clock_reconciled": False,
    }
    if epoch is None:
        return out
    lead_seconds = float(epoch - wall_epoch)
    if (
        lead_seconds >= float(TICK_FUTURE_TOLERANCE_SECONDS)
        and lead_seconds
        <= float(MAX_BROKER_TICK_CLOCK_RECONCILIATION_SECONDS)
    ):
        out.update(
            {
                "reference_epoch": epoch,
                "clock_reference": "broker_tick_at_acquisition",
                "clock_reconciled": True,
                "local_clock_lag_seconds": round(lead_seconds, 3),
                "clock_reconciliation_limit_seconds": int(
                    MAX_BROKER_TICK_CLOCK_RECONCILIATION_SECONDS
                ),
            }
        )
    return out

def tick_age_error(age: Optional[float], *, symbol: str, threshold: float = QUOTE_LIVE_SECONDS) -> Optional[Dict[str, Any]]:
    """Apply the submission freshness policy to an age on the chosen clock."""
    if age is None:
        return {
            "error": f"Tick for {symbol} has no usable timestamp; freshness cannot be verified.",
            "tick_age_status": "unknown",
            "tick_max_age_seconds": threshold,
        }
    if age < -float(TICK_FUTURE_TOLERANCE_SECONDS):
        future_skew = -age
        return {
            "error": (
                f"Tick for {symbol} is {future_skew:.1f}s ahead of the wall clock "
                "and is not safe for live trading."
            ),
            "tick_age_status": "future",
            "timestamp_in_future": True,
            "timestamp_skew_seconds": round(future_skew, 2),
            "tick_max_age_seconds": threshold,
        }
    if age <= threshold:
        return None
    return {
        "error": f"Tick for {symbol} is stale ({age:.1f}s old, threshold {threshold:.0f}s).",
        "tick_age_seconds": round(age, 2),
        "tick_max_age_seconds": threshold,
    }


_MATERIAL_QUOTE_MID_DISAGREEMENT_POINTS = 10.0
_MATERIAL_QUOTE_SIDE_DISAGREEMENT_POINTS = 6.0
_SERVER_CLOCK_OFFSET_MATCH_SECONDS = 2.0


def _configured_broker_offset_seconds(at_epoch: float) -> int:
    try:
        from ..bootstrap.settings import mt5_config

        at_time = datetime.fromtimestamp(float(at_epoch), tz=timezone.utc)
        return int(mt5_config.get_time_offset_seconds(at_time))
    except Exception:
        return 0


def _cache_epoch_is_unnormalized_server_clock(
    raw_epoch: Any,
    stream_epoch: Any,
    *,
    now_epoch: float,
) -> bool:
    """True when cached tick looks like a server-clock epoch of the UTC stream."""
    if raw_epoch is None or stream_epoch is None:
        return False
    try:
        raw_value = float(raw_epoch)
        stream_value = float(stream_epoch)
    except (TypeError, ValueError):
        return False
    offset_seconds = _configured_broker_offset_seconds(now_epoch)
    if offset_seconds == 0:
        return False
    return abs((raw_value - stream_value) - float(offset_seconds)) <= (
        _SERVER_CLOCK_OFFSET_MATCH_SECONDS
    )


def _quote_source_conflict_is_material(
    quote_source_conflict: Any,
    *,
    point: Any = None,
) -> bool:
    """Return True when source disagreement is large enough to block execution."""
    if not isinstance(quote_source_conflict, dict):
        return False
    cached = quote_source_conflict.get("symbol_info_tick")
    stream = quote_source_conflict.get("stream_tick")
    if not isinstance(cached, dict) or not isinstance(stream, dict):
        return True
    left_mid = canonical_quote_midpoint(cached.get("bid"), cached.get("ask"))
    right_mid = canonical_quote_midpoint(stream.get("bid"), stream.get("ask"))
    if left_mid is None or right_mid is None:
        return True
    try:
        point_value = float(point) if point is not None else None
    except (TypeError, ValueError):
        point_value = None
    if point_value is None or not math.isfinite(point_value) or point_value <= 0:
        point_value = None
        for pair in (cached, stream):
            try:
                bid_value = float(pair.get("bid"))
                ask_value = float(pair.get("ask"))
            except (TypeError, ValueError):
                continue
            spread = abs(ask_value - bid_value)
            if math.isfinite(spread) and spread > 0:
                point_value = spread / 10.0
                break

    def _price(value: Any) -> Optional[float]:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    left_bid = _price(cached.get("bid"))
    left_ask = _price(cached.get("ask"))
    right_bid = _price(stream.get("bid"))
    right_ask = _price(stream.get("ask"))
    if None in (left_bid, left_ask, right_bid, right_ask):
        return True
    left_spread = abs(left_ask - left_bid)
    right_spread = abs(right_ask - right_bid)
    mid_gap = abs(left_mid - right_mid)
    side_gap = max(abs(left_bid - right_bid), abs(left_ask - right_ask))
    spread_gap = abs(left_spread - right_spread)
    if point_value is None:
        return mid_gap > 0 or side_gap > 0 or spread_gap > 0
    mid_points = mid_gap / point_value
    side_points = side_gap / point_value
    spread_points = spread_gap / point_value
    if mid_points >= _MATERIAL_QUOTE_MID_DISAGREEMENT_POINTS:
        return True
    return (
        side_points >= _MATERIAL_QUOTE_SIDE_DISAGREEMENT_POINTS
        or spread_points >= _MATERIAL_QUOTE_SIDE_DISAGREEMENT_POINTS
    )


def enforce_quote_execution_readiness(
    context: Dict[str, Any],
    *,
    bid: Any,
    ask: Any,
    quote_source_conflict: Any = None,
    point: Any = None,
) -> Dict[str, Any]:
    """Apply the canonical executable-quote gate to a freshness context.

    Tick freshness alone only establishes that a quote is current and the
    market session is active. Live execution additionally requires a positive
    two-sided spread. Small same-timestamp book disagreements are disclosed
    but do not by themselves mark a two-sided quote unusable.
    """
    spread = compute_spread_metrics(bid, ask)
    spread_quality = str(spread["spread_quality"])
    source_conflict = isinstance(quote_source_conflict, dict)
    material_conflict = _quote_source_conflict_is_material(
        quote_source_conflict,
        point=point,
    )
    context["spread_valid"] = bool(spread["spread_valid"])
    context["spread_quality"] = spread_quality
    spread_is_executable = spread["spread_valid"] is True and not material_conflict
    freshness_available = "usable_for_live_trading" in context
    if not spread_is_executable:
        context["usable_for_live_trading"] = False
    elif freshness_available:
        context["usable_for_live_trading"] = (
            context.get("usable_for_live_trading") is True
        )
    if "usable_for_live_trading" in context:
        context["usable_for_live_trading_basis"] = (
            QUOTE_EXECUTION_SOURCE_AGREEMENT_BASIS
            if material_conflict
            else QUOTE_EXECUTION_READINESS_BASIS
        )

    warnings: list[str] = []
    if context.get("send_path_tick_fresh") is False:
        context["usable_for_live_trading"] = False
        context["usable_for_live_trading_basis"] = "submission_tick_freshness_required"
        if not (context.get("data_stale") is True and context.get("warning")):
            warnings.append(str(context.get("send_path_freshness_error") or "Submission tick freshness cannot be verified."))
    if spread_quality == "locked":
        warnings.append(
            "Locked quote (bid equals ask) is not usable for live trading."
        )
    elif spread_quality != "two_sided":
        warnings.append("Quote is not a valid positive two-sided market.")
    if source_conflict:
        warnings.append(
            "Cached and streamed quotes disagree at the same timestamp; "
            "review quote_source_conflict before trading."
        )
    if warnings:
        existing = str(context.get("warning") or "").strip()
        context["warning"] = " ".join(
            part for part in (existing, *warnings) if part
        )
    return context


def tick_value(tick: Any, field: str) -> Any:
    if isinstance(tick, dict):
        return tick.get(field)
    try:
        value = tick[field]
        if type(value).__module__ != "unittest.mock":
            return value
    except Exception:
        pass
    return getattr(tick, field, None)


def tick_epoch(tick: Any) -> Optional[float]:
    def _number(value: Any) -> Optional[float]:
        # Mock placeholders and arbitrary objects may implement ``__float__``;
        # accepting them can turn a missing attribute into a plausible epoch.
        if isinstance(value, bool) or not isinstance(value, (Real, str)):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) and number > 0.0 else None

    time_msc = _number(tick_value(tick, "time_msc"))
    if time_msc is not None:
        return time_msc / 1000.0
    return _number(tick_value(tick, "time"))


def canonical_quote_midpoint(bid: Any, ask: Any) -> Optional[float]:
    """Return the exact decimal midpoint represented by broker quote strings.

    Converting each quote through ``str`` first removes binary-float addition
    artifacts while retaining legitimate half-tick precision.
    """
    try:
        bid_decimal = Decimal(str(bid))
        ask_decimal = Decimal(str(ask))
        midpoint = (bid_decimal + ask_decimal) / Decimal(2)
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not midpoint.is_finite() or midpoint <= 0:
        return None
    return float(midpoint)


def canonical_quote_spread(bid: Any, ask: Any) -> Optional[float]:
    """Return ask minus bid without leaking binary subtraction artifacts."""
    try:
        spread = Decimal(str(ask)) - Decimal(str(bid))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return float(spread) if spread.is_finite() else None


def quote_spread_bps(bid: Any, ask: Any) -> Optional[float]:
    """Return two-sided quoted spread in basis points, or None if invalid."""
    try:
        bid_value = float(bid or 0.0)
        ask_value = float(ask or 0.0)
        mid = (bid_value + ask_value) / 2.0
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (bid_value, ask_value, mid)):
        return None
    if bid_value <= 0.0 or ask_value <= bid_value or mid <= 0.0:
        return None
    return round((ask_value - bid_value) / mid * 10_000.0, 4)


def symbol_info_spread_bps(
    *,
    spread_points: Any,
    point: Any,
    bid: Any,
    ask: Any,
    fallback_mid: Any = None,
) -> Optional[float]:
    """Convert MT5 spread-point metadata to basis points using quote or close mid."""
    try:
        spread_pts = float(spread_points or 0.0)
        point_value = float(point or 0.0)
        bid_value = float(bid or 0.0)
        ask_value = float(ask or 0.0)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(spread_pts) or spread_pts <= 0.0:
        return None
    if not math.isfinite(point_value) or point_value <= 0.0:
        return None
    mid = (bid_value + ask_value) / 2.0 if bid_value > 0.0 and ask_value > bid_value else 0.0
    if mid <= 0.0:
        try:
            close = float(fallback_mid or 0.0)
        except (TypeError, ValueError):
            close = 0.0
        if math.isfinite(close) and close > 0.0:
            mid = close
    if mid <= 0.0:
        return None
    return round(spread_pts * point_value / mid * 10_000.0, 4)


def compute_spread_metrics(
    bid: Any,
    ask: Any,
    *,
    point: Any = None,
    points_per_pip: Any = None,
    tick_size: Any = None,
    tick_value_money: Any = None,
    account_currency: Optional[str] = None,
) -> Dict[str, Any]:
    """Return unrounded spread measurements and a canonical quote-quality label."""

    def _positive(value: Any) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number > 0.0 else None

    bid_value = _positive(bid)
    ask_value = _positive(ask)
    if bid_value is None or ask_value is None:
        quality = "one_sided"
    elif ask_value < bid_value:
        quality = "inverted"
    elif ask_value == bid_value:
        quality = "locked"
    else:
        quality = "two_sided"

    out: Dict[str, Any] = {
        "mid": None,
        "spread": None,
        "spread_points": None,
        "spread_pips": None,
        "spread_pct": None,
        "spread_cost_per_lot": None,
        "spread_valid": quality == "two_sided",
        "spread_quality": quality,
        "pricing_basis": "quote_only",
    }
    if quality not in {"two_sided", "locked"}:
        return out

    spread = canonical_quote_spread(bid_value, ask_value)
    mid = canonical_quote_midpoint(bid_value, ask_value)
    if spread is None or mid is None:
        return out
    point_value = _positive(point)
    pip_points = _positive(points_per_pip)
    tick_size_value = _positive(tick_size)
    tick_money_value = _positive(tick_value_money)
    spread_points = spread / point_value if point_value is not None else None
    spread_pips = (
        spread_points / pip_points
        if spread_points is not None and pip_points is not None
        else None
    )
    spread_cost = (
        (spread / tick_size_value) * tick_money_value
        if tick_size_value is not None
        and tick_money_value is not None
        and account_currency
        else None
    )
    out.update(
        {
            "mid": mid,
            "spread": spread,
            "spread_points": spread_points,
            "spread_pips": spread_pips,
            "spread_pct": (spread / mid) * 100.0,
            "spread_cost_per_lot": spread_cost,
            "pricing_basis": (
                "per_1_lot_estimate" if spread_cost is not None else "quote_only"
            ),
        }
    )
    return out


def _latest_stream_ticks(
    gateway: Any,
    symbol: str,
    *,
    now_epoch: float,
) -> tuple[Any, Any]:
    end = datetime.fromtimestamp(now_epoch, tz=timezone.utc) + timedelta(seconds=5)
    start = end - timedelta(minutes=15, seconds=5)
    closure = standard_weekend_window(datetime.fromtimestamp(now_epoch, tz=timezone.utc))
    if closure is not None:
        # Include only the final active portion of Friday, not an arbitrary
        # multi-day tick history, when reconciling a frozen weekend quote.
        start = closure[0] - timedelta(minutes=15)
    try:
        rows = gateway.copy_ticks_range(
            symbol,
            start,
            end,
            gateway.COPY_TICKS_ALL,
        )
    except Exception:
        return None, None
    if rows is None:
        return None, None
    try:
        candidates = [row for row in rows if tick_epoch(row) is not None]
    except (TypeError, ValueError):
        return None, None
    if not candidates:
        return None, None
    # CopyTicksRange returns ticks from oldest to newest.  Several updates can
    # share a millisecond, so preserve that ordering when timestamps tie: max()
    # alone would retain the first (stale) row with the newest timestamp.
    _, latest_tick = max(
        enumerate(candidates),
        key=lambda item: (float(tick_epoch(item[1]) or 0.0), item[0]),
    )
    two_sided_candidates = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if _quote_pair_quality(candidate) == "two_sided"
    ]
    latest_two_sided = None
    if two_sided_candidates:
        _, latest_two_sided = max(
            two_sided_candidates,
            key=lambda item: (float(tick_epoch(item[1]) or 0.0), item[0]),
        )
    return latest_tick, latest_two_sided


def _quote_pair(tick: Any) -> tuple[Optional[float], Optional[float]]:
    values = []
    for field in ("bid", "ask"):
        try:
            value = float(tick_value(tick, field))
        except (TypeError, ValueError):
            value = float("nan")
        values.append(value if math.isfinite(value) and value > 0.0 else None)
    return values[0], values[1]


def _quote_pair_quality(tick: Any) -> str:
    bid, ask = _quote_pair(tick)
    if bid is None or ask is None:
        return "one_sided"
    if ask < bid:
        return "inverted"
    if ask == bid:
        return "locked"
    return "two_sided"


def _quote_pair_quality_rank(tick: Any) -> int:
    return {"inverted": 0, "one_sided": 1, "locked": 2, "two_sided": 3}[
        _quote_pair_quality(tick)
    ]


def _symbol_point(gateway: Any, symbol: str) -> Optional[float]:
    try:
        info = gateway.symbol_info(symbol)
        point = getattr(info, "point", None)
        value = float(point) if isinstance(point, (Real, str)) else None
    except (AttributeError, TypeError, ValueError):
        return None
    return value if value is not None and math.isfinite(value) and value > 0.0 else None


def _is_one_sided_quote_update(gateway: Any, tick: Any) -> bool:
    try:
        raw_flags = tick_value(tick, "flags")
        flags = int(raw_flags) if isinstance(raw_flags, (Real, str)) else 0
    except (TypeError, ValueError):
        return False
    bid_flag, ask_flag = bid_ask_flags(gateway)
    return bool(flags & bid_flag) != bool(flags & ask_flag)


def _quote_pairs_agree(
    left: Any,
    right: Any,
    *,
    point: Optional[float],
) -> bool:
    left_pair = _quote_pair(left)
    right_pair = _quote_pair(right)
    if left_pair == right_pair:
        return True
    if point is None or any(value is None for value in (*left_pair, *right_pair)):
        return False
    return all(
        abs(float(left_value) - float(right_value)) < point
        for left_value, right_value in zip(left_pair, right_pair, strict=True)
    )


def resolve_quote_tick(
    gateway: Any,
    symbol: str,
    tick: Any = None,
    *,
    now_epoch: float,
    stale_after_seconds: int = QUOTE_STALE_SECONDS,
) -> tuple[Any, Dict[str, Any]]:
    """Reconcile MT5's cached symbol tick with its authoritative tick stream."""
    if tick is not None:
        raw_tick = tick
    else:
        try:
            raw_tick = gateway.symbol_info_tick(symbol)
        except Exception:
            raw_tick = None
    raw_epoch = tick_epoch(raw_tick)
    latest_stream_tick, two_sided_stream_tick = _latest_stream_ticks(
        gateway,
        symbol,
        now_epoch=now_epoch,
    )
    stream_tick = latest_stream_tick
    used_recent_two_sided_stream = False
    if (
        latest_stream_tick is not None
        and _quote_pair_quality(latest_stream_tick) != "two_sided"
        and two_sided_stream_tick is not None
    ):
        two_sided_epoch = tick_epoch(two_sided_stream_tick)
        two_sided_freshness = build_tick_freshness_context(
            symbol,
            tick_epoch=two_sided_epoch,
            now_epoch=now_epoch,
            item="tick",
            stale_after_seconds=stale_after_seconds,
        )
        if two_sided_freshness.get("usable_for_live_trading") is True:
            stream_tick = two_sided_stream_tick
            used_recent_two_sided_stream = True
    stream_epoch = tick_epoch(stream_tick)
    metadata: Dict[str, Any] = {
        "quote_source": "mt5.symbol_info_tick",
        "quote_refresh_attempted": True,
    }

    clock = tick_clock_reference(raw_epoch, wall_clock_epoch=now_epoch)
    send_error = tick_age_error(
        float(clock["reference_epoch"]) - raw_epoch if raw_epoch is not None else None,
        symbol=symbol,
    )
    metadata["send_path_tick_fresh"] = send_error is None
    if send_error:
        metadata["send_path_freshness_error"] = send_error["error"]

    raw_freshness = build_tick_freshness_context(
        symbol,
        tick_epoch=raw_epoch,
        now_epoch=now_epoch,
        item="tick",
        stale_after_seconds=stale_after_seconds,
    )
    raw_live_ready = raw_freshness.get("usable_for_live_trading") is True

    if used_recent_two_sided_stream:
        latest_stream_epoch = tick_epoch(latest_stream_tick)
        metadata["raw_last_stream_event"] = {
            "time_epoch": latest_stream_epoch,
            "bid": _quote_pair(latest_stream_tick)[0],
            "ask": _quote_pair(latest_stream_tick)[1],
            "spread_quality": _quote_pair_quality(latest_stream_tick),
            "one_sided_update": _is_one_sided_quote_update(
                gateway,
                latest_stream_tick,
            ),
        }

    if stream_tick is None or stream_epoch is None:
        metadata["quote_source_state"] = (
            "current" if raw_live_ready else "unverified_stale"
        )
        return raw_tick, metadata

    stream_freshness = build_tick_freshness_context(
        symbol,
        tick_epoch=stream_epoch,
        now_epoch=now_epoch,
        item="tick",
        stale_after_seconds=stale_after_seconds,
    )
    stream_live_ready = stream_freshness.get("usable_for_live_trading") is True
    unnormalized_server_cache = _cache_epoch_is_unnormalized_server_clock(
        raw_epoch,
        stream_epoch,
        now_epoch=now_epoch,
    )
    same_epoch = raw_epoch is not None and (
        abs(stream_epoch - raw_epoch) <= 0.001 or unnormalized_server_cache
    )
    pairs_differ = _quote_pair(raw_tick) != _quote_pair(stream_tick)
    point = _symbol_point(gateway, symbol)
    one_sided_stream_update = _is_one_sided_quote_update(gateway, stream_tick)
    equal_timestamp_one_sided_update = same_epoch and one_sided_stream_update
    within_point = same_epoch and _quote_pairs_agree(
        raw_tick,
        stream_tick,
        point=point,
    )
    benign_reconciliation = pairs_differ and (
        equal_timestamp_one_sided_update or within_point
    )
    quote_conflict = same_epoch and pairs_differ and not benign_reconciliation
    raw_rank = _quote_pair_quality_rank(raw_tick)
    stream_rank = _quote_pair_quality_rank(stream_tick)
    raw_execution_ready = raw_live_ready and raw_rank == 3
    stream_execution_ready = stream_live_ready and stream_rank == 3
    use_stream_for_conflict = quote_conflict and stream_rank >= raw_rank
    raw_ahead_of_observation = (
        raw_epoch is not None
        and raw_epoch > float(now_epoch) + TICK_FUTURE_TOLERANCE_SECONDS
    )
    stream_not_ahead_of_observation = (
        stream_epoch <= float(now_epoch) + TICK_FUTURE_TOLERANCE_SECONDS
    )
    reject_lower_quality_stream = (
        raw_tick is not None
        and raw_epoch is not None
        and stream_epoch > raw_epoch + 0.001
        and raw_live_ready
        and raw_rank > stream_rank
    )
    use_stream = (
        raw_tick is None
        or use_stream_for_conflict
        or unnormalized_server_cache
        or (
            raw_ahead_of_observation
            and stream_not_ahead_of_observation
            and stream_live_ready
        )
        or (
            raw_epoch is not None
            and stream_epoch > raw_epoch + 0.001
            and not reject_lower_quality_stream
        )
        or (
            not raw_execution_ready
            and stream_execution_ready
            and not reject_lower_quality_stream
        )
    )
    if not use_stream:
        metadata["quote_source_state"] = (
            "reconciled_equal_timestamp_conflict"
            if quote_conflict
            else "reconciled_lower_quality_stream_update"
            if reject_lower_quality_stream
            else "reconciled_one_sided_update"
            if one_sided_stream_update and pairs_differ
            else "reconciled_within_point"
            if within_point and pairs_differ
            else ("current" if raw_live_ready else "unverified_stale")
        )
        metadata["stream_tick_time_epoch"] = stream_epoch
        if quote_conflict:
            metadata["quote_source_conflict"] = {
                "reason": "equal_timestamp_bid_ask_disagreement",
                "time_epoch": stream_epoch,
                "selected_source": "mt5.symbol_info_tick",
                "symbol_info_tick": dict(zip(("bid", "ask"), _quote_pair(raw_tick))),
                "stream_tick": dict(zip(("bid", "ask"), _quote_pair(stream_tick))),
            }
        return raw_tick, metadata

    metadata.update(
        {
            "quote_source": "mt5.copy_ticks_range",
            "quote_source_state": (
                "reconciled_equal_timestamp_conflict"
                if quote_conflict
                else "reconciled_recent_two_sided_stream"
                if used_recent_two_sided_stream
                else "refreshed_from_tick_stream"
            ),
            "symbol_info_tick_time_epoch": raw_epoch,
            "stream_tick_time_epoch": stream_epoch,
        }
    )
    if quote_conflict:
        metadata["quote_source_conflict"] = {
            "reason": "equal_timestamp_bid_ask_disagreement",
            "time_epoch": stream_epoch,
            "selected_source": "mt5.copy_ticks_range",
            "symbol_info_tick": dict(zip(("bid", "ask"), _quote_pair(raw_tick))),
            "stream_tick": dict(zip(("bid", "ask"), _quote_pair(stream_tick))),
        }
    return stream_tick, metadata
