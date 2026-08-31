from __future__ import annotations

import math
from typing import Any, Callable, Dict

from .freshness import (
    QUOTE_LIVE_SECONDS,
    QUOTE_RECENT_SECONDS,
    QUOTE_STALE_SECONDS,
    closed_session_context,
    format_age_seconds,
    format_freshness_label,
    round_age_seconds,
)

TICK_VOLUME_UNIT = "bid_update_count"
TICK_VOLUME_SEMANTICS = "tick_volume_is_bid_update_count_not_lots"
TICK_VOLUME_EVENT_BASIS = "mt5_broker_bar_bid_updates"
TICK_VOLUME_TAPE_EQUIVALENT = "bid_update_count"
TICK_VOLUME_COMPARISON_NOTE = (
    "MT5 candle tick_volume follows Bid-chart updates. It can differ from "
    "data_fetch_ticks.tick_count when COPY_TICKS_ALL includes ask-only updates."
)
_TICK_VOLUME_UNIT_ALIASES = frozenset(
    {
        TICK_VOLUME_UNIT,
        "broker_tick_count",
        "mt5_tick_volume",
    }
)

FRESHNESS_ANCHOR_QUERY_EXPECTED_END = "query_expected_end"
FRESHNESS_ANCHOR_WALL_CLOCK = "wall_clock"

FRESHNESS_METRIC_LAST_COMPLETED_BAR_AGE = "last_completed_bar_age_seconds"
FRESHNESS_METRIC_LAST_TICK_AGE = "last_tick_age_seconds"
FRESHNESS_METRIC_REQUESTED_RANGE_END_GAP = "requested_range_end_gap_seconds"
TICK_FUTURE_TOLERANCE_SECONDS = 10.0


def attach_candle_volume_semantics(payload: Dict[str, Any]) -> None:
    volume_type = str(payload.get("volume_type") or "").strip().lower()
    volume_unit = str(payload.get("volume_unit") or "").strip().lower()
    if (
        volume_type in {"tick_count", TICK_VOLUME_UNIT}
        or volume_unit in _TICK_VOLUME_UNIT_ALIASES
    ):
        payload["volume_semantics"] = TICK_VOLUME_SEMANTICS
        payload["tick_volume_event_basis"] = TICK_VOLUME_EVENT_BASIS
        payload["tick_volume_tape_equivalent"] = TICK_VOLUME_TAPE_EQUIVALENT
        payload["tick_volume_comparison_note"] = TICK_VOLUME_COMPARISON_NOTE


def normalize_policy_relaxed(value: Any) -> bool:
    try:
        return bool(value)
    except Exception:
        return False


def tick_freshness_state(
    *,
    data_stale: Any,
    market_status: Any = None,
    market_status_reason: Any = None,
    freshness_policy_relaxed: Any = None,
    age_seconds: Any = None,
) -> str:
    if normalize_policy_relaxed(freshness_policy_relaxed):
        status = str(market_status or "closed_or_idle").strip().lower()
        reason = str(market_status_reason or "").strip().lower()
        parts = [status.replace(" ", "_")] if status else ["closed_or_idle"]
        if reason:
            parts.append(reason.replace(" ", "_"))
        parts.append("snapshot")
        return "_".join(part for part in parts if part)
    if bool(data_stale):
        return "stale"
    try:
        age = max(0.0, float(age_seconds))
    except (TypeError, ValueError):
        return "unknown"
    if age <= float(QUOTE_LIVE_SECONDS):
        return "live"
    if age <= float(QUOTE_RECENT_SECONDS):
        return "recent"
    return "delayed"


def build_tick_freshness_context(
    symbol: Any,
    *,
    tick_epoch: Any,
    now_epoch: Any,
    item: str = "tick",
    stale_after_seconds: Any = QUOTE_STALE_SECONDS,
    age_rounder: Callable[[float], Any] | None = None,
) -> Dict[str, Any]:
    try:
        current_epoch = float(now_epoch)
        latest_tick_epoch = float(tick_epoch)
    except Exception:
        return {}
    if not math.isfinite(current_epoch) or not math.isfinite(latest_tick_epoch):
        return {}

    signed_age_seconds = current_epoch - latest_tick_epoch
    age_seconds = max(0.0, signed_age_seconds)
    future_skew_seconds = max(0.0, -signed_age_seconds)
    timestamp_ahead_of_wall_clock = future_skew_seconds > 0.0
    timestamp_in_future = (
        future_skew_seconds >= TICK_FUTURE_TOLERANCE_SECONDS
    )
    try:
        threshold = max(0.0, float(stale_after_seconds))
    except Exception:
        threshold = float(QUOTE_STALE_SECONDS)

    closed_session = closed_session_context(
        symbol,
        now_epoch=current_epoch,
        item=item,
        data_age_seconds=None if timestamp_in_future else age_seconds,
    )
    data_stale = age_seconds > threshold or timestamp_in_future

    rounded_age = (
        age_rounder(age_seconds)
        if age_rounder is not None
        else round_age_seconds(age_seconds)
    )
    stale_after: int | float
    if float(threshold).is_integer():
        stale_after = int(threshold)
    else:
        stale_after = threshold

    out: Dict[str, Any] = {
        "data_age_seconds": rounded_age,
        "data_age_anchor": FRESHNESS_ANCHOR_WALL_CLOCK,
        "data_age_metric": FRESHNESS_METRIC_LAST_TICK_AGE,
        "stale_after_seconds": stale_after,
        "data_stale": data_stale,
        "live_max_age_seconds": QUOTE_LIVE_SECONDS,
    }
    if timestamp_ahead_of_wall_clock:
        out["timestamp_ahead_of_wall_clock"] = True
        out["timestamp_skew_seconds"] = round(future_skew_seconds, 3)
        out["timestamp_skew_tolerance_seconds"] = int(
            TICK_FUTURE_TOLERANCE_SECONDS
        )
    if timestamp_in_future:
        out["timestamp_in_future"] = True
        out["timestamp_warning"] = (
            "Latest tick timestamp is ahead of the wall clock; the quote is not "
            "safe for live decisions until MT5 time alignment is corrected."
        )
    if closed_session:
        out.update(closed_session)
    out["freshness_basis"] = f"absolute_{stale_after}s"
    execution_fresh = (
        age_seconds <= float(QUOTE_LIVE_SECONDS)
        and not data_stale
        and not timestamp_in_future
    )
    out["usable_for_live_trading"] = execution_fresh and not bool(closed_session)
    out["usable_for_live_trading_basis"] = "quote_age_and_market_session"
    if timestamp_in_future:
        out["freshness_reason"] = "future_timestamp"
    elif timestamp_ahead_of_wall_clock:
        out["freshness_reason"] = "clock_skew_within_tolerance"
    elif closed_session:
        out["freshness_reason"] = "market_closed"
    elif data_stale:
        out["freshness_reason"] = "stale_age"
    elif not execution_fresh:
        out["freshness_reason"] = "quote_age_exceeds_live_threshold"
    else:
        out["freshness_reason"] = "live_quote"

    freshness = None
    if timestamp_in_future:
        freshness = (
            f"clock skew, {item} timestamp {format_age_seconds(future_skew_seconds)} "
            "ahead of wall clock"
        )
    elif timestamp_ahead_of_wall_clock:
        freshness = (
            f"live with tolerated clock skew, {item} timestamp "
            f"{format_age_seconds(future_skew_seconds)} ahead of wall clock"
        )
    else:
        freshness = format_freshness_label(
            data_stale=data_stale,
            market_status=out.get("market_status"),
            market_status_reason=out.get("market_status_reason"),
            age_seconds=age_seconds,
            item=item,
        )
    if freshness:
        out["freshness"] = freshness
    out["freshness_state"] = (
        "clock_skew"
        if timestamp_in_future
        else tick_freshness_state(
            data_stale=data_stale,
            market_status=out.get("market_status"),
            market_status_reason=out.get("market_status_reason"),
            freshness_policy_relaxed=out.get("freshness_policy_relaxed"),
            age_seconds=age_seconds,
        )
    )
    if not closed_session and out["freshness_state"] in {"recent", "delayed"}:
        age_text = format_age_seconds(age_seconds)
        item_label = str(item or "tick").strip() or "tick"
        out["freshness"] = (
            f"{out['freshness_state']}, {item_label} {age_text} ago"
        )
    return out
