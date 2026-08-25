import math
import time
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from numbers import Real
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

from ...shared.constants import (
    DEFAULT_ROW_LIMIT,
    FETCH_RETRY_ATTEMPTS,
    FETCH_RETRY_DELAY,
    SIMPLIFY_DEFAULT_METHOD,
    SIMPLIFY_DEFAULT_MODE,
    TICKS_LOOKBACK_DAYS,
)
from ...shared.market_units import forex_points_per_pip
from ...shared.schema import SimplifySpec
from ...utils.coercion import coerce_finite_float as _finite_or_none
from ...utils.market_metadata import build_tick_freshness_context
from ...utils.mt5 import (
    _mt5_copy_ticks_range,
    _symbol_ready_guard,
    describe_mt5_time_normalization,
    get_symbol_info_cached,
    mt5,
    resolve_broker_symbol_name,
)
from ...utils.mt5 import (
    symbol_path as _symbol_path,
)
from ...utils.mt5 import (
    symbol_price_currency as _symbol_price_currency,
)
from ...utils.mt5 import (
    symbol_price_digits as _symbol_price_digits,
)
from ...utils.mt5 import (
    symbol_price_point as _symbol_price_point,
)
from ...utils.quote import (
    canonical_quote_midpoint,
    canonical_quote_spread,
    enforce_quote_execution_readiness,
    resolve_quote_tick,
    tick_epoch,
)
from ...utils.quote import tick_value as _tick_field_value
from ...utils.simplify import (
    _choose_simplify_points,
    _lttb_select_indices,
    _select_indices_for_timeseries,
    _simplify_dataframe_rows_ext,
)
from ...utils.tick_flags import is_mt5_trade_event
from ...utils.time import _format_time_explicit, _resolve_client_tz
from ...utils.utils import (
    _format_numeric_rows_from_df,
    _iana_timezone_datetime_issue,
    _parse_end_datetime,
    _parse_start_datetime,
    _table_from_rows,
    _utc_epoch_seconds,
)
from .candles import (
    _normalize_simplify_spec,
    _round_price_value,
    _round_row_price_columns,
    _timezone_label,
)
from .errors import _future_start_error, attach_empty_range_weekend_context
from .query import _DATE_FORMAT_HINT

_TICK_COUNT_EVENT_BASIS = "mt5_copy_ticks_all_records"


_QUOTE_UPDATE_COUNT_EVENT_BASIS = "records_with_bid_or_ask_update_flag"


_TICK_SUMMARY_MIN_ANALYTIC_TICKS = 20


_INCOMPLETE_TICK_WARNING_RATIO = 0.50


_TICK_PRICE_COLUMNS = frozenset({"bid", "ask", "mid", "spread", "last"})


_TICK_PRICE_STAT_KEYS = frozenset(
    {
        "first",
        "last",
        "low",
        "high",
        "mean",
        "std",
        "stderr",
        "change",
        "median",
        "q25",
        "q75",
    }
)


_TICK_ROW_UNITS = {
    "time_epoch": "unix_seconds",
    "bid": "absolute_price",
    "ask": "absolute_price",
    "last": "absolute_price",
    "mid": "absolute_price",
    "spread": "absolute_price",
    "spread_points": "broker_points",
    "spread_pips": "pips (forex_only; null when not applicable)",
    "spread_pct": "percent (1.0 = 1%)",
    "tick_gap_ms": "milliseconds",
    "volume": "last_trade_volume",
    "volume_real": "last_trade_volume_real",
}


def _round_tick_price_payload(out: Dict[str, Any], digits: int) -> None:
    if digits <= 0:
        return
    stats = out.get("stats")
    if isinstance(stats, dict):
        for name in ("bid", "ask", "spread", "last"):
            values = stats.get(name)
            if not isinstance(values, dict):
                continue
            for key in _TICK_PRICE_STAT_KEYS:
                if key in values:
                    stat_digits = digits + 2 if name == "spread" else digits
                    values[key] = _round_price_value(values[key], stat_digits)
    last_quote = out.get("last_quote")
    if isinstance(last_quote, dict):
        for key in ("bid", "ask", "spread"):
            if key in last_quote:
                last_quote[key] = _round_price_value(last_quote[key], digits)
    if isinstance(stats, dict):
        for volume_key in ("volume", "volume_real"):
            volume_stats = stats.get(volume_key)
            if isinstance(volume_stats, dict):
                for key in ("vwap_mid", "vwap_last"):
                    if key in volume_stats:
                        volume_stats[key] = _round_price_value(volume_stats[key], digits)


def _tick_units_for_headers(headers: List[str]) -> Dict[str, str]:
    return {
        key: unit
        for key, unit in _TICK_ROW_UNITS.items()
        if key in headers
    }


def _tick_spread_points(spread: Any, price_point: Optional[float]) -> Optional[float]:
    if price_point is None or price_point <= 0.0:
        return None
    spread_value = _finite_or_none(spread)
    if spread_value is None:
        return None
    return round(spread_value / price_point, 4)


def _tick_spread_pct(spread: Any, mid: Any) -> Optional[float]:
    spread_value = _finite_or_none(spread)
    mid_value = _finite_or_none(mid)
    if spread_value is None or mid_value is None or mid_value <= 0.0:
        return None
    return round((spread_value / mid_value) * 100.0, 6)


def _fetch_ticks_range_with_retry(
    symbol: str,
    from_date: datetime,
    to_date: datetime,
) -> Any:
    ticks = None
    for _ in range(FETCH_RETRY_ATTEMPTS):
        ticks = _mt5_copy_ticks_range(symbol, from_date, to_date, mt5.COPY_TICKS_ALL)
        if ticks is not None and len(ticks) > 0:
            break
        time.sleep(FETCH_RETRY_DELAY)
    return ticks


def _fetch_recent_ticks_backwards(
    symbol: str,
    *,
    to_date: datetime,
    limit: int,
    min_from_date: Optional[datetime] = None,
) -> Any:
    """Fetch the most recent ticks in bounded backward ranges to avoid huge queries."""
    if limit <= 0:
        return []
    if min_from_date is not None:
        min_is_aware = min_from_date.tzinfo is not None and min_from_date.utcoffset() is not None
        to_is_aware = to_date.tzinfo is not None and to_date.utcoffset() is not None
        if min_is_aware != to_is_aware:
            to_date = to_date.replace(tzinfo=min_from_date.tzinfo if min_is_aware else None)

    chunk_days = 1
    max_lookback_days = max(max(1, int(TICKS_LOOKBACK_DAYS)), 30)
    budget_floor = to_date - timedelta(days=max_lookback_days)
    effective_floor = (
        max(min_from_date, budget_floor)
        if min_from_date is not None
        else budget_floor
    )
    cursor_end = to_date
    lookback_days_used = 0
    saw_response = False
    collected: List[Any] = []

    while True:
        chunk_from = cursor_end - timedelta(days=chunk_days)
        if chunk_from < effective_floor:
            chunk_from = effective_floor

        overlaps_newer_range = saw_response
        exact_bounded_window = min_from_date is not None
        # MT5 range retrieval is unreliable for fractional datetime bounds.
        # For an explicitly bounded window, query the enclosing whole seconds,
        # then enforce the advertised inclusive millisecond bounds.
        provider_start = (
            chunk_from.replace(microsecond=0)
            if exact_bounded_window
            else chunk_from
        )
        provider_end = (
            cursor_end.replace(microsecond=0)
            if exact_bounded_window
            else cursor_end
        )
        if exact_bounded_window and cursor_end.microsecond:
            provider_end += timedelta(seconds=1)
        ticks_candidate = _fetch_ticks_range_with_retry(
            symbol,
            provider_start,
            provider_end,
        )
        if ticks_candidate is not None:
            saw_response = True
            candidate_rows = list(ticks_candidate)
            if exact_bounded_window:
                chunk_start_epoch = float(_utc_epoch_seconds(chunk_from))
                chunk_end_epoch = float(_utc_epoch_seconds(cursor_end))
                candidate_rows = [
                    tick
                    for tick in candidate_rows
                    if (
                        tick_epoch_value := tick_epoch(tick)
                    )
                    is not None
                    and chunk_start_epoch <= tick_epoch_value <= chunk_end_epoch
                ]
            if overlaps_newer_range and candidate_rows:
                boundary_epoch = float(_utc_epoch_seconds(cursor_end))
                candidate_rows = [
                    tick
                    for tick in candidate_rows
                    if (
                        (tick_epoch_value := tick_epoch(tick)) is None
                        or tick_epoch_value < boundary_epoch
                    )
                ]
            if candidate_rows:
                collected = candidate_rows + collected
                if len(collected) > limit:
                    collected = collected[-limit:]
                if len(collected) >= limit:
                    break

        chunk_span_days = max(
            1,
            int(math.ceil((cursor_end - chunk_from).total_seconds() / 86_400.0)),
        )
        lookback_days_used += chunk_span_days
        if chunk_from <= effective_floor or lookback_days_used >= max_lookback_days:
            break

        cursor_end = chunk_from
        chunk_days = min(
            chunk_days * 2,
            max(1, max_lookback_days - lookback_days_used),
        )

    if collected:
        return collected
    if saw_response:
        return []
    return None


def _fetch_ticks_forward(
    symbol: str,
    *,
    from_date: datetime,
    to_date: datetime,
    limit: int,
) -> Any:
    """Fetch the earliest ticks from an inclusive start without loading a huge range."""
    if limit <= 0 or from_date > to_date:
        return []
    from_is_aware = from_date.tzinfo is not None and from_date.utcoffset() is not None
    to_is_aware = to_date.tzinfo is not None and to_date.utcoffset() is not None
    if from_is_aware != to_is_aware:
        to_date = to_date.replace(tzinfo=from_date.tzinfo if from_is_aware else None)

    requested_end_epoch = float(_utc_epoch_seconds(to_date))
    cursor_start = from_date
    chunk_seconds = 3600.0
    saw_response = False
    collected: List[Any] = []
    while cursor_start <= to_date and len(collected) < limit:
        cursor_end = min(to_date, cursor_start + timedelta(seconds=chunk_seconds))
        # MT5 range retrieval is not reliable for fractional datetime bounds.
        # Query an enclosing whole-second superset, then enforce the exact
        # advertised inclusive bounds from each tick's millisecond epoch.
        provider_start = cursor_start.replace(microsecond=0)
        provider_end = cursor_end.replace(microsecond=0)
        if cursor_end.microsecond:
            provider_end += timedelta(seconds=1)
        ticks_candidate = _fetch_ticks_range_with_retry(
            symbol,
            provider_start,
            provider_end,
        )
        if ticks_candidate is not None:
            saw_response = True
            boundary_epoch = float(_utc_epoch_seconds(cursor_start))
            candidate_rows = [
                tick
                for tick in list(ticks_candidate)
                if (tick_epoch_value := tick_epoch(tick)) is not None
                and boundary_epoch <= tick_epoch_value <= requested_end_epoch
            ]
            collected.extend(candidate_rows)
            if len(collected) >= limit:
                break
        if cursor_end >= to_date:
            break
        # Exclude the already queried boundary on the next chunk while retaining
        # sub-second tick precision.
        cursor_start = cursor_end + timedelta(microseconds=1)
        if not collected:
            chunk_seconds = min(chunk_seconds * 2.0, 7 * 86_400.0)

    if collected:
        collected.sort(
            key=lambda tick: tick_epoch(tick) or float("-inf")
        )
        return collected[:limit]
    if saw_response:
        return []
    return None


def _live_tick_spread_reference(
    symbol: str,
) -> Tuple[Optional[float], Dict[str, Any]]:
    now_epoch = time.time()
    try:
        raw_tick = mt5.symbol_info_tick(symbol)
    except Exception:
        raw_tick = None
    tick, quote_source = resolve_quote_tick(
        mt5,
        symbol,
        raw_tick,
        now_epoch=now_epoch,
    )
    bid = _tick_field_value(tick, "bid") if tick is not None else None
    ask = _tick_field_value(tick, "ask") if tick is not None else None
    try:
        bid_f = float(bid)
        ask_f = float(ask)
    except Exception:
        spread = None
    else:
        spread = (
            ask_f - bid_f
            if math.isfinite(bid_f) and math.isfinite(ask_f) and ask_f > bid_f
            else None
        )
    epoch = tick_epoch(tick) if tick is not None else None
    context = build_tick_freshness_context(
        symbol,
        tick_epoch=epoch,
        now_epoch=time.time(),
        item="spread reference",
        age_rounder=lambda value: round(value, 3),
    )
    live_usable = (
        context.get("usable_for_live_trading") is True
        and spread is not None
        and spread > 0.0
    )
    if not live_usable:
        spread = None
        if context.get("usable_for_live_trading") is True:
            context["freshness_reason"] = "locked_or_invalid_quote"
    freshness = {
        "reference_time": _format_time_explicit(epoch) if epoch is not None else None,
        "reference_time_epoch": epoch,
        "data_age_seconds": context.get("data_age_seconds"),
        "freshness_state": context.get("freshness_state") or "unknown",
        "freshness_reason": context.get("freshness_reason"),
        "usable_for_live_trading": live_usable,
        **quote_source,
    }
    return spread, freshness


def _mt5_tick_flag_value(name: str, default: int) -> int:
    try:
        return int(getattr(mt5, name))
    except (TypeError, ValueError, AttributeError):
        return int(default)


def _tick_flag_definitions() -> tuple[tuple[int, str, str], ...]:
    return (
        (
            _mt5_tick_flag_value("TICK_FLAG_BID", 2),
            "bid",
            "Bid price changed in this snapshot.",
        ),
        (
            _mt5_tick_flag_value("TICK_FLAG_ASK", 4),
            "ask",
            "Ask price changed in this snapshot.",
        ),
        (
            _mt5_tick_flag_value("TICK_FLAG_LAST", 8),
            "last",
            "Last traded price changed in this snapshot.",
        ),
        (
            _mt5_tick_flag_value("TICK_FLAG_VOLUME", 16),
            "volume",
            "Last-trade volume changed in this snapshot.",
        ),
        (
            _mt5_tick_flag_value("TICK_FLAG_BUY", 32),
            "buy",
            "Last trade was buyer-initiated.",
        ),
        (
            _mt5_tick_flag_value("TICK_FLAG_SELL", 64),
            "sell",
            "Last trade was seller-initiated.",
        ),
    )


def _decode_tick_flags(flag_value: int) -> List[str]:
    try:
        remaining = int(flag_value)
    except (TypeError, ValueError):
        return []
    labels: List[str] = []
    for bit, label, _description in _tick_flag_definitions():
        if bit > 0 and remaining & bit:
            labels.append(label)
            remaining &= ~bit
    while remaining > 0:
        bit = remaining & -remaining
        labels.append(f"unknown_{bit}")
        remaining &= ~bit
    return labels


def _observed_tick_flags_decoded(flags: List[int]) -> Dict[str, List[str]]:
    return {
        str(flag): _decode_tick_flags(flag)
        for flag in sorted(set(int(value) for value in flags if int(value) != 0))
    }


def _normalize_tick_missing_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_tick_missing_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_tick_missing_values(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_tick_missing_values(item) for item in value]
    if value is None or isinstance(value, (bool, str, bytes)):
        return value
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            return None
        return value
    try:
        if pd.isna(value) and not isinstance(value, (str, bytes)):
            return None
    except Exception:
        pass
    return value


def _compact_tick_summary(out: Dict[str, Any]) -> Dict[str, Any]:
    spread = out.get("stats", {}).get("spread")
    compact_spread: Dict[str, Any] = {}
    if isinstance(spread, dict):
        available = spread.get("available")
        try:
            spread_unavailable = available is not None and not bool(available)
        except Exception:
            spread_unavailable = False
        if spread_unavailable:
            compact_spread["available"] = False
        else:
            for source_key, target_key in (
                ("low", "low"),
                ("high", "high"),
                ("mean", "mean"),
            ):
                value = spread.get(source_key)
                if value is not None:
                    compact_spread[target_key] = value
    compact: Dict[str, Any] = {
        "success": bool(out.get("success")),
        "symbol": out.get("symbol"),
        "count": out.get("count"),
        "start": out.get("start"),
        "end": out.get("end"),
        "duration_seconds": out.get("duration_seconds"),
        "tick_rate_per_second": out.get("tick_rate_per_second"),
        "tick_count": out.get("tick_count", out.get("count")),
        "tick_count_event_basis": out.get("tick_count_event_basis"),
        "trade_event_count": out.get("trade_event_count"),
        "quote_update_count": out.get("quote_update_count"),
        "quote_update_count_event_basis": out.get(
            "quote_update_count_event_basis"
        ),
        "bid_update_count": out.get("bid_update_count"),
        "ask_update_count": out.get("ask_update_count"),
        "timezone": out.get("timezone"),
        "stats": {"spread": compact_spread},
    }
    if isinstance(out.get("_tick_page"), dict):
        compact["_tick_page"] = dict(out["_tick_page"])
    if out.get("price_precision") is not None:
        compact["price_precision"] = out.get("price_precision")
    if out.get("price_point") is not None:
        compact["price_point"] = out.get("price_point")
    if out.get("price_currency") is not None:
        compact["price_currency"] = out.get("price_currency")
    for key in (
        "time_basis",
        "raw_time_basis",
        "time_normalization",
        "broker_server_tz",
        "session_utc_offset_seconds",
        "spread_statistics_basis",
        "feed_tier",
        "history_window_truncated",
        "history_window_limit_days",
        "history_window_floor",
        "data_window",
        "query_applied",
        "warnings",
    ):
        if out.get(key) is not None:
            compact[key] = out.get(key)
    for key in (
        "freshness",
        "data_age_seconds",
        "data_stale",
        "market_status",
        "market_status_reason",
        "market_status_source",
        "freshness_policy_relaxed",
        "note",
    ):
        if out.get(key) is not None:
            compact[key] = out.get(key)
    if isinstance(out.get("last_quote"), dict):
        compact["last_quote"] = dict(out["last_quote"])
    if isinstance(out.get("data_quality"), dict):
        compact["data_quality"] = dict(out["data_quality"])
    return compact


def fetch_ticks(  # noqa: C901
    symbol: str,
    limit: int = DEFAULT_ROW_LIMIT,
    start: Optional[str] = None,
    end: Optional[str] = None,
    simplify: Optional[SimplifySpec] = None,
    time_as_epoch: bool = False,
    format: Literal["summary", "stats", "rows", "full_rows"] = "summary",
    range_selection: Literal["first_n", "last_n"] = "first_n",
    page_offset: int = 0,
    probe_more: bool = False,
    force_utc: bool = False,
) -> Dict[str, Any]:
    """Fetch tick data and return either a summary (default) or raw rows.

    Parameters
    ----------
    format : {"summary","stats","rows","full_rows"}
        - "summary" (default): compact descriptive statistics over the fetched
          ticks. Samples below 20 ticks report spread stats only with a sample
          adequacy note; larger samples include bid/ask/mid, plus last and
          volume when available.
        - "stats": more detailed stats (includes extra distribution moments and
          quantiles).
        - "rows": return tick rows as structured data.
        - "full_rows": return rows with per-tick epoch, mid, spread, and gap fields.
    """
    try:
        symbol = resolve_broker_symbol_name(symbol)
        effective_limit = int(limit)
        effective_offset = max(0, int(page_offset))
        fetch_limit = effective_limit + effective_offset + int(bool(probe_more))
        normalized_range_selection = str(range_selection or "first_n").strip().lower()
        if normalized_range_selection not in {"first_n", "last_n"}:
            return {"error": "range_selection must be first_n or last_n."}
        history_window_truncated = False
        history_window_floor: Optional[datetime] = None
        if effective_limit <= 0:
            return {"error": "limit must be greater than 0."}
        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = get_symbol_info_cached(symbol)
        with _symbol_ready_guard(symbol, info_before=_info_before) as (err, _info):
            if err:
                return {"error": err}
            price_digits = _symbol_price_digits(_info, _info_before)
            price_currency = _symbol_price_currency(_info, _info_before)
            price_point = _symbol_price_point(_info, _info_before)
            points_per_pip = (
                forex_points_per_pip(
                    symbol,
                    path=_symbol_path(_info, _info_before),
                    point=price_point,
                    digits=price_digits,
                )
                if price_point is not None
                else None
            )
            time_normalization = describe_mt5_time_normalization(symbol=symbol)

            # Normalized params only. This is an output shape selector, not the
            # shared compact/full detail enum.
            output_mode = str(format or "summary").strip().lower()
            output_mode = {
                "raw": "rows",
                "ticks": "rows",
            }.get(output_mode, output_mode)
            if start:
                from_date = _parse_start_datetime(start)
                if not from_date:
                    issue = _iana_timezone_datetime_issue(start)
                    if issue is not None:
                        return {"error": f"{issue['error']} {issue['remediation']}"}
                    return {"error": f"Could not parse start date {start!r}. {_DATE_FORMAT_HINT}"}
                future_error = _future_start_error(start, from_date, 0)
                if future_error:
                    return {"error": future_error}
                if end:
                    to_date = _parse_end_datetime(end)
                    if not to_date:
                        issue = _iana_timezone_datetime_issue(end)
                        if issue is not None:
                            return {"error": f"{issue['error']} {issue['remediation']}"}
                        return {"error": f"Could not parse end date {end!r}. {_DATE_FORMAT_HINT}"}
                    if from_date > to_date:
                        return {"error": "start must be before or equal to end."}
                    max_lookback_days = max(max(1, int(TICKS_LOOKBACK_DAYS)), 30)
                    history_window_floor = to_date - timedelta(days=max_lookback_days)
                    history_window_truncated = from_date < history_window_floor
                    effective_from_date = max(from_date, history_window_floor)
                    if normalized_range_selection == "last_n":
                        ticks = _fetch_recent_ticks_backwards(
                            symbol,
                            to_date=to_date,
                            limit=fetch_limit,
                            min_from_date=effective_from_date,
                        )
                    else:
                        ticks = _fetch_ticks_forward(
                            symbol,
                            from_date=effective_from_date,
                            to_date=to_date,
                            limit=fetch_limit,
                        )
                else:
                    max_lookback_days = max(max(1, int(TICKS_LOOKBACK_DAYS)), 30)
                    history_to_date = datetime.now(dt_timezone.utc)
                    if from_date.tzinfo is None or from_date.utcoffset() is None:
                        history_to_date = history_to_date.replace(tzinfo=None)
                    history_window_floor = history_to_date - timedelta(
                        days=max_lookback_days
                    )
                    history_window_truncated = from_date < history_window_floor
                    effective_from_date = max(from_date, history_window_floor)
                    ticks = _fetch_ticks_forward(
                        symbol,
                        from_date=effective_from_date,
                        to_date=history_to_date,
                        limit=fetch_limit,
                    )
            else:
                # End-only requests are historical backward queries anchored
                # at the supplied endpoint, not aliases for "latest".
                if end:
                    to_date = _parse_end_datetime(end)
                    if not to_date:
                        issue = _iana_timezone_datetime_issue(end)
                        if issue is not None:
                            return {"error": f"{issue['error']} {issue['remediation']}"}
                        return {
                            "error": (
                                f"Could not parse end date {end!r}. "
                                f"{_DATE_FORMAT_HINT}"
                            )
                        }
                else:
                    to_date = datetime.now(dt_timezone.utc)
                ticks = _fetch_recent_ticks_backwards(
                    symbol,
                    to_date=to_date,
                    limit=fetch_limit,
                )
        # visibility handled by _symbol_ready_guard
        
        if ticks is None:
            return {"error": f"Failed to get ticks for {symbol}: {mt5.last_error()}"}
        
        page_has_more = False
        page_source_returned = len(ticks)
        if probe_more:
            fetched_ticks = list(ticks)
            if normalized_range_selection == "last_n":
                page_end = max(0, len(fetched_ticks) - effective_offset)
                page_start = max(0, page_end - effective_limit)
                page_has_more = page_start > 0
                ticks = fetched_ticks[page_start:page_end]
            else:
                page_start = effective_offset
                page_end = page_start + effective_limit
                page_has_more = len(fetched_ticks) > page_end
                ticks = fetched_ticks[page_start:page_end]
            page_source_returned = len(ticks)

        # Generate tabular format with dynamic column filtering
        if len(ticks) == 0:
            empty_payload: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "count": 0,
                "tick_count": 0,
                "tick_count_event_basis": _TICK_COUNT_EVENT_BASIS,
                "quote_update_count": 0,
                "quote_update_count_event_basis": _QUOTE_UPDATE_COUNT_EVENT_BASIS,
                "bid_update_count": 0,
                "ask_update_count": 0,
                "data": [],
                "empty": True,
                "empty_reason": "no_ticks_in_range",
                "timezone": "UTC",
                "query_applied": {
                    "mode": "historical" if start or end else "latest",
                    "start": start,
                    "end": end,
                    "selection": (
                        normalized_range_selection
                        if start and end
                        else "first_n"
                        if start
                        else "last_n"
                    ),
                },
                "_tick_page": {
                    "offset": effective_offset,
                    "source_returned": 0,
                    "has_more": page_has_more,
                },
            }
            if history_window_truncated:
                empty_payload["history_window_truncated"] = True
                empty_payload["history_window_limit_days"] = max(
                    max(1, int(TICKS_LOOKBACK_DAYS)), 30
                )
                if history_window_floor is not None:
                    effective_start = _format_time_explicit(
                        _utc_epoch_seconds(history_window_floor)
                    )
                    empty_payload["history_window_floor"] = effective_start
                    empty_payload["effective_start"] = effective_start
                    empty_payload["query_applied"]["requested_start"] = str(start)
                    empty_payload["query_applied"]["start"] = effective_start
                    empty_payload["query_applied"]["effective_start"] = effective_start
                empty_payload["warnings"] = [
                    "Tick history retrieval stopped at the configured lookback "
                    "budget before reaching the requested start."
                ]
            return attach_empty_range_weekend_context(
                empty_payload,
                symbol=symbol,
                start=start,
                end=end,
                item="ticks",
            )

        if output_mode not in ("summary", "stats", "rows", "full_rows"):
            return {
                "error": (
                    f"Invalid format: {format}. "
                    "Use 'summary', 'stats', 'rows', or 'full_rows'."
                )
            }

        # Extract shared tick columns once so summary/stats, simplification,
        # and row rendering can all reuse the same values.
        _epochs: List[float] = []
        bids: List[float] = []
        asks: List[float] = []
        effective_bids: List[Optional[float]] = []
        effective_asks: List[Optional[float]] = []
        lasts: List[float] = []
        flags: List[int] = []
        volumes: List[float] = []
        volumes_real: List[float] = []
        trade_events: List[bool] = []
        quote_types: List[str] = []
        for tick in ticks:
            tick_time = tick_epoch(tick)
            if tick_time is None:
                raise ValueError("tick timestamp unavailable")
            _epochs.append(tick_time)
            bid_value = _finite_or_none(_tick_field_value(tick, "bid"))
            ask_value = _finite_or_none(_tick_field_value(tick, "ask"))
            bid = float("nan") if bid_value is None else bid_value
            ask = float("nan") if ask_value is None else ask_value
            flag_value = int(_tick_field_value(tick, "flags") or 0)
            bids.append(bid)
            asks.append(ask)
            last_value = _finite_or_none(_tick_field_value(tick, "last"))
            lasts.append(
                float("nan")
                if last_value is None or last_value <= 0.0
                else last_value
            )
            flags.append(flag_value)
            effective_bids.append(None if bid_value is None else bid)
            effective_asks.append(None if ask_value is None else ask)
            if bid_value is None and ask_value is not None:
                quote_types.append("ask_only")
            elif ask_value is None and bid_value is not None:
                quote_types.append("bid_only")
            elif bid_value is None and ask_value is None:
                quote_types.append("no_quote")
            else:
                quote_types.append("bid_ask")
            try:
                volume_value = float(_tick_field_value(tick, "volume"))
            except (TypeError, ValueError):
                volume_value = float("nan")
            try:
                volume_real_value = float(_tick_field_value(tick, "volume_real"))
            except (TypeError, ValueError):
                volume_real_value = float("nan")
            volumes.append(volume_value)
            volumes_real.append(volume_real_value)
            trade_events.append(
                is_mt5_trade_event(flag_value, mt5)
                and (
                    (last_value is not None and last_value > 0.0)
                    or (math.isfinite(volume_value) and volume_value > 0.0)
                    or (
                        math.isfinite(volume_real_value)
                        and volume_real_value > 0.0
                    )
                )
            )

        has_last = any(math.isfinite(value) for value in lasts)
        finite_volumes = [v for v in volumes if math.isfinite(v)]
        has_volume = bool(finite_volumes) and (
            len(set(finite_volumes)) > 1 or any(v != 0.0 for v in finite_volumes)
        )
        has_flags = len(set(flags)) > 1 or any(v != 0 for v in flags)
        has_real_volume = any(math.isfinite(v) and v != 0.0 for v in volumes_real)
        trade_event_count = int(sum(trade_events))
        quote_only_feed = not has_last and trade_event_count == 0
        incomplete_quote_count = sum(
            1
            for bid, ask in zip(effective_bids, effective_asks, strict=False)
            if bid is None or ask is None
        )
        bid_update_flag = _mt5_tick_flag_value("TICK_FLAG_BID", 2)
        ask_update_flag = _mt5_tick_flag_value("TICK_FLAG_ASK", 4)
        quote_update_mask = bid_update_flag | ask_update_flag
        has_quote_update_flags = any(flag & quote_update_mask for flag in flags)
        quote_update_types: List[str] = []
        spread_valid_flags: List[bool] = []
        bid_changed_flags: List[bool] = []
        ask_changed_flags: List[bool] = []
        for flag, quote_type in zip(flags, quote_types, strict=False):
            bid_updated = bool(flag & bid_update_flag)
            ask_updated = bool(flag & ask_update_flag)
            bid_changed_flags.append(bid_updated)
            ask_changed_flags.append(ask_updated)
            if bid_updated and ask_updated:
                update_type = "bid_ask_update"
            elif bid_updated:
                update_type = "bid_only_update"
            elif ask_updated:
                update_type = "ask_only_update"
            else:
                update_type = "update_flags_unavailable"
            quote_update_types.append(update_type)
            spread_valid_flags.append(
                quote_type == "bid_ask"
                and (
                    (bid_updated and ask_updated)
                    if has_quote_update_flags
                    else True
                )
            )
        one_sided_update_count = sum(
            update_type in {"bid_only_update", "ask_only_update"}
            for update_type in quote_update_types
        )
        bid_update_count = int(sum(bid_changed_flags))
        ask_update_count = int(sum(ask_changed_flags))
        quote_update_count = int(
            sum(flag & quote_update_mask != 0 for flag in flags)
        )
        zero_spread_count = sum(
            quote_type == "bid_ask"
            and (
                (bid_updated and ask_updated)
                if has_quote_update_flags
                else True
            )
            and bid is not None
            and ask is not None
            and float(ask) == float(bid)
            for quote_type, bid_updated, ask_updated, bid, ask in zip(
                quote_types,
                bid_changed_flags,
                ask_changed_flags,
                effective_bids,
                effective_asks,
                strict=False,
            )
        )

        full_rows = output_mode == "full_rows"

        # Keep row schemas stable; compact public output prunes unused fields.
        headers = ["time"]
        if full_rows:
            headers.append("time_epoch")
        headers.extend(["bid", "ask"])
        include_quote_type = any(value != "bid_ask" for value in quote_types)
        if include_quote_type:
            headers.append("quote_type")
        if full_rows:
            headers.extend(
                [
                    "quote_update_type",
                    "bid_changed",
                    "ask_changed",
                    "spread_valid",
                    "spread_basis",
                    "spread_sample_eligible",
                ]
            )
            headers.extend(["mid", "spread"])
            if price_point is not None:
                headers.append("spread_points")
            if points_per_pip is not None:
                headers.append("spread_pips")
            headers.extend(["spread_pct", "tick_gap_ms"])
        headers.extend(["last", "volume", "volume_real", "flags", "flags_decoded"])

        # Choose a consistent millisecond time format for tick rows.
        # Low-level tick fetch helpers have already normalized epochs to UTC.
        client_tz = None if force_utc else _resolve_client_tz()
        _use_ctz = client_tz is not None

        def _format_tick_time(epoch: float) -> str:
            if time_as_epoch:
                return int(epoch) if float(epoch).is_integer() else float(epoch)
            try:
                dt = datetime.fromtimestamp(float(epoch), tz=dt_timezone.utc)
            except Exception:
                return str(epoch)
            if _use_ctz:
                try:
                    dt = dt.astimezone(client_tz)
                except Exception:
                    dt = dt.astimezone()
            millis = int(round(float(dt.microsecond) / 1000.0))
            if millis >= 1000:
                dt = dt + timedelta(seconds=1)
                millis = 0
            offset = dt.strftime("%z")
            if offset == "+0000":
                offset = "Z"
            elif len(offset) == 5 and offset[0] in {"+", "-"}:
                offset = f"{offset[:3]}:{offset[3:]}"
            return f"{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{millis:03d}{offset}"

        original_count = len(ticks)
        simplify_eff = _normalize_simplify_spec(simplify, limit=limit, fallback_rows=original_count)
        simplify_present = (simplify_eff is not None) or (simplify is not None)
        simplify_used = simplify_eff if simplify_eff is not None else simplify
        simplify_mode = (
            str((simplify_used or {}).get("mode", SIMPLIFY_DEFAULT_MODE)).lower().strip()
            if simplify_present
            else SIMPLIFY_DEFAULT_MODE
        )
        simplify_target_points: Optional[int] = None
        if simplify_present and simplify_mode != "resample":
            try:
                simplify_target_points = _choose_simplify_points(
                    original_count, simplify_used
                )
            except Exception:
                simplify_target_points = None

        df_ticks = pd.DataFrame({
            "__epoch": _epochs,
            "bid": effective_bids,
            "ask": effective_asks,
        })
        if full_rows:
            tick_gap_ms: List[Optional[float]] = [None]
            for idx in range(1, len(_epochs)):
                tick_gap_ms.append(float((_epochs[idx] - _epochs[idx - 1]) * 1000.0))
            df_ticks["time_epoch"] = [
                int(epoch) if float(epoch).is_integer() else float(epoch)
                for epoch in _epochs
            ]
            df_ticks["mid"] = (
                (df_ticks["bid"] + df_ticks["ask"]) / 2.0
            ).round(price_digits + 1)
            df_ticks["spread"] = df_ticks["ask"] - df_ticks["bid"]
            if price_point is not None:
                df_ticks["spread_points"] = df_ticks["spread"] / price_point
            if price_point is not None and points_per_pip is not None:
                df_ticks["spread_pips"] = (
                    df_ticks["spread"] / price_point / points_per_pip
                )
            df_ticks["spread_pct"] = (df_ticks["spread"] / df_ticks["mid"]) * 100.0
            df_ticks["tick_gap_ms"] = tick_gap_ms
        df_ticks["last"] = lasts
        df_ticks["volume"] = volumes
        df_ticks["volume_real"] = volumes_real
        df_ticks["flags"] = flags
        df_ticks["trade_event"] = trade_events
        df_ticks["spread_valid"] = [
            bid is not None and ask is not None and ask > bid
            for bid, ask in zip(effective_bids, effective_asks, strict=False)
        ]
        spread_sample_eligible_flags = [
            eligible
            and bid is not None
            and ask is not None
            and ask > bid
            for eligible, bid, ask in zip(
                spread_valid_flags,
                effective_bids,
                effective_asks,
                strict=False,
            )
        ]
        df_ticks["spread_sample_eligible"] = spread_sample_eligible_flags
        df_ticks["spread_basis"] = [
            "quote_snapshot" if valid else "unavailable"
            for valid in df_ticks["spread_valid"].tolist()
        ]
        df_ticks["bid_changed"] = bid_changed_flags
        df_ticks["ask_changed"] = ask_changed_flags
        df_ticks["quote_update_type"] = quote_update_types
        if include_quote_type:
            df_ticks["quote_type"] = quote_types
        df_ticks["flags_decoded"] = [
            _decode_tick_flags(flag_value) for flag_value in flags
        ]
        df_ticks["time"] = [_format_tick_time(e) for e in _epochs]

        def _add_tick_data_quality(payload: Dict[str, Any]) -> None:
            if (
                incomplete_quote_count <= 0
                and one_sided_update_count <= 0
                and zero_spread_count <= 0
            ):
                return
            incomplete_ratio = incomplete_quote_count / max(1, original_count)
            quote_type_counts = {
                kind: quote_types.count(kind)
                for kind in sorted(set(quote_types))
            }
            complete_ticks = int(quote_type_counts.get("bid_ask", 0))
            incomplete_ticks = int(original_count - complete_ticks)
            payload["data_quality"] = {
                "incomplete_quote_ticks": int(incomplete_quote_count),
                "complete_ticks": complete_ticks,
                "incomplete_ticks": incomplete_ticks,
                "total_ticks": int(original_count),
                "incomplete_quote_ratio": round(incomplete_ratio, 4),
                "spread_ticks_excluded": int(
                    original_count - sum(spread_sample_eligible_flags)
                ),
                "one_sided_updates": int(one_sided_update_count),
                "valid_spread_ticks": int(sum(spread_sample_eligible_flags)),
                "spread_sample_basis": "coherent_bid_ask_updates",
                "zero_spread_ticks": int(zero_spread_count),
                "incomplete_quote_warning_threshold": _INCOMPLETE_TICK_WARNING_RATIO,
                "quote_type_counts": quote_type_counts,
            }
            if one_sided_update_count > 0:
                payload["data_quality"]["one_sided_update_status"] = "expected"
            if incomplete_ratio < _INCOMPLETE_TICK_WARNING_RATIO:
                payload["data_quality"]["incomplete_quote_status"] = "info"
                return
            payload["data_quality"]["incomplete_quote_status"] = "warning"
            warnings_list = payload.get("warnings")
            if not isinstance(warnings_list, list):
                warnings_list = []
            warning = (
                "Spread statistics exclude incomplete quote snapshots; "
                "zero-spread counts include only coherent two-sided updates."
            )
            if warning not in warnings_list:
                warnings_list.append(warning)
            payload["warnings"] = warnings_list

        def _add_tick_last_quality(payload: Dict[str, Any]) -> None:
            if has_last:
                return
            payload["last_unavailable"] = True
            if quote_only_feed:
                return
            warnings_list = payload.get("warnings")
            if not isinstance(warnings_list, list):
                warnings_list = []
            warning = "Broker tick data did not provide a usable last price; last is null."
            if warning not in warnings_list:
                warnings_list.append(warning)
            payload["warnings"] = warnings_list

        def _add_tick_context_fields(payload: Dict[str, Any]) -> None:
            payload["_tick_page"] = {
                "offset": effective_offset,
                "source_returned": int(page_source_returned),
                "has_more": bool(page_has_more),
            }
            payload["spread_statistics_basis"] = "coherent_bid_ask_updates"
            if quote_only_feed:
                payload["feed_tier"] = "quote_only"
            if start or end:
                query_applied: Dict[str, Any] = {
                    "mode": "historical",
                    "limit": int(effective_limit),
                    "limit_anchor": (
                        "end"
                        if start and end and normalized_range_selection == "last_n"
                        else "start"
                        if start
                        else "end"
                    ),
                    "selection": (
                        normalized_range_selection
                        if start and end
                        else "first_n"
                        if start
                        else "last_n"
                    ),
                    "order": "ascending",
                }
                if start:
                    query_applied["start"] = str(start)
                if end:
                    query_applied["end"] = str(end)
                payload["query_applied"] = query_applied
            else:
                payload["query_applied"] = {
                    "mode": "latest",
                    "limit": int(effective_limit),
                    "limit_anchor": "latest",
                    "selection": "last_n",
                    "order": "ascending",
                }
            if history_window_truncated:
                payload["history_window_truncated"] = True
                payload["history_window_limit_days"] = max(
                    max(1, int(TICKS_LOOKBACK_DAYS)), 30
                )
                if history_window_floor is not None:
                    effective_start = _format_time_explicit(
                        _utc_epoch_seconds(history_window_floor)
                    )
                    payload["history_window_floor"] = effective_start
                    payload["effective_start"] = effective_start
                    if start:
                        query_applied["requested_start"] = str(start)
                        query_applied["start"] = effective_start
                        query_applied["effective_start"] = effective_start
                warning = (
                    "Tick history retrieval stopped at the configured lookback "
                    "budget before reaching the requested start."
                )
                warnings = payload.setdefault("warnings", [])
                if warning not in warnings:
                    warnings.append(warning)
            last_quote = payload.get("last_quote")
            payload["data_window"] = {
                "start": df_ticks["time"].iloc[0],
                "end": df_ticks["time"].iloc[-1],
            }
            if isinstance(last_quote, dict):
                last_quote["time"] = df_ticks["time"].iloc[-1]
                last_quote["quote_scope"] = (
                    "historical_sample" if start or end else "latest_sample"
                )
            execution_quote = None
            if (
                isinstance(last_quote, dict)
                and last_quote.get("spread_valid") is not True
            ):
                execution_quote = _reconciled_execution_quote(df_ticks)
                if execution_quote is not None:
                    payload["execution_quote"] = execution_quote
            if isinstance(last_quote, dict) and price_point is not None:
                spread_value = _finite_or_none(last_quote.get("spread"))
                if spread_value is not None:
                    last_quote["spread_points"] = round(spread_value / price_point, 4)
                    if points_per_pip is not None:
                        last_quote["spread_pips"] = round(
                            spread_value / price_point / points_per_pip,
                            4,
                        )
            if isinstance(last_quote, dict):
                spread_pct = _tick_spread_pct(
                    last_quote.get("spread"),
                    last_quote.get("mid"),
                )
                if spread_pct is not None:
                    last_quote["spread_pct"] = spread_pct
            payload.update(time_normalization)
            if start or end or not _epochs:
                return
            latest_tick_epoch = float(_epochs[-1])
            freshness_context = build_tick_freshness_context(
                symbol,
                tick_epoch=latest_tick_epoch,
                now_epoch=time.time(),
                item="tick",
                age_rounder=lambda value: round(value, 3),
            )
            payload.update(freshness_context)
            quote_for_gate = execution_quote or last_quote
            if isinstance(execution_quote, dict):
                payload["usable_for_live_trading_basis"] = (
                    "quote_age_market_session_and_reconciled_spread"
                )
            elif (
                isinstance(quote_for_gate, dict)
                and quote_for_gate.get("spread_valid") is True
            ):
                enforce_quote_execution_readiness(
                    payload,
                    bid=quote_for_gate.get("bid"),
                    ask=quote_for_gate.get("ask"),
                )
            elif (
                isinstance(quote_for_gate, dict)
                and quote_for_gate.get("spread_valid") is not True
            ):
                payload["usable_for_live_trading"] = False
                payload["usable_for_live_trading_basis"] = (
                    "quote_age_market_session_and_positive_spread"
                )
                blockers = list(payload.get("execution_blockers") or [])
                blocker = f"latest_quote_{last_quote.get('spread_quality') or 'invalid'}"
                if blocker not in blockers:
                    blockers.append(blocker)
                payload["execution_blockers"] = blockers
                warning = (
                    "Latest quote is not executable because it lacks a positive "
                    "two-sided spread."
                )
                warnings_list = list(payload.get("warnings") or [])
                if warning not in warnings_list:
                    warnings_list.append(warning)
                payload["warnings"] = warnings_list

        def _last_snapshot_quote(frame: pd.DataFrame) -> Dict[str, Any]:
            bid = _finite_or_none(frame["bid"].iloc[-1])
            ask = _finite_or_none(frame["ask"].iloc[-1])
            spread_valid = bool(
                bid is not None and ask is not None and float(ask) > float(bid)
            )
            two_sided = bid is not None and ask is not None
            spread = (
                canonical_quote_spread(bid, ask)
                if two_sided and float(ask) >= float(bid)
                else None
            )
            mid = (
                canonical_quote_midpoint(bid, ask)
                if two_sided and float(ask) >= float(bid)
                else None
            )
            spread_quality = (
                "two_sided"
                if spread_valid
                else "locked"
                if two_sided and float(ask) == float(bid)
                else "inverted"
                if two_sided
                else "one_sided"
            )
            return {
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread": spread,
                "spread_valid": spread_valid,
                "spread_quality": spread_quality,
                "spread_basis": (
                    "quote_snapshot"
                    if spread_valid
                    else "quote_snapshot_locked"
                    if spread_quality == "locked"
                    else "unavailable"
                ),
            }

        def _reconciled_execution_quote(
            frame: pd.DataFrame,
        ) -> Optional[Dict[str, Any]]:
            if frame.empty or "quote_update_type" not in frame.columns:
                return None
            update_type = str(frame["quote_update_type"].iloc[-1] or "")
            if update_type not in {"bid_only_update", "ask_only_update"}:
                return None
            prior = frame.iloc[:-1]
            if "spread_sample_eligible" in prior.columns:
                prior = prior[prior["spread_sample_eligible"].astype(bool)]
            if prior.empty:
                return None
            prior_row = prior.iloc[-1]
            latest_row = frame.iloc[-1]
            bid = _finite_or_none(
                latest_row.get("bid")
                if update_type == "bid_only_update"
                else prior_row.get("bid")
            )
            ask = _finite_or_none(
                latest_row.get("ask")
                if update_type == "ask_only_update"
                else prior_row.get("ask")
            )
            if bid is None or ask is None or ask <= bid:
                return None
            spread = canonical_quote_spread(bid, ask)
            mid = canonical_quote_midpoint(bid, ask)
            if spread is None or spread <= 0.0 or mid is None:
                return None
            return {
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread": spread,
                "spread_valid": True,
                "spread_quality": "two_sided",
                "spread_basis": "reconciled_one_sided_update",
                "time": latest_row.get("time"),
            }

        def _compact_summary_from_ticks() -> Dict[str, Any]:
            df_stats = df_ticks.copy()
            df_stats["mid"] = (
                (df_stats["bid"] + df_stats["ask"]) / 2.0
            ).round(price_digits + 1)
            df_stats["spread"] = df_stats["ask"] - df_stats["bid"]
            start_epoch = float(df_stats["__epoch"].iloc[0])
            end_epoch = float(df_stats["__epoch"].iloc[-1])
            duration_seconds = float(max(0.0, end_epoch - start_epoch))
            tick_rate_per_second = (
                float(len(df_stats) / duration_seconds) if duration_seconds > 0 else None
            )
            spread = pd.to_numeric(
                df_stats["spread"].where(df_stats["spread_sample_eligible"]),
                errors="coerce",
            ).dropna()
            out: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "count": int(len(df_stats)),
                "start": df_stats["time"].iloc[0],
                "end": df_stats["time"].iloc[-1],
                "duration_seconds": duration_seconds,
                "tick_rate_per_second": tick_rate_per_second,
                "tick_count": int(len(df_stats)),
                "tick_count_event_basis": _TICK_COUNT_EVENT_BASIS,
                "trade_event_count": int(sum(trade_events)),
                "quote_update_count": quote_update_count,
                "quote_update_count_event_basis": _QUOTE_UPDATE_COUNT_EVENT_BASIS,
                "bid_update_count": bid_update_count,
                "ask_update_count": ask_update_count,
                "timezone": _timezone_label(use_client_tz=_use_ctz, client_tz=client_tz),
                "stats": {
                    "spread": (
                        {
                            "low": float(spread.min()),
                            "high": float(spread.max()),
                            "mean": float(spread.mean()),
                        }
                        if not spread.empty
                        else {"available": False}
                    )
                },
                "last_quote": _last_snapshot_quote(df_stats),
            }
            if price_digits > 0:
                out["price_precision"] = int(price_digits)
            if price_point is not None:
                out["price_point"] = price_point
            if price_currency:
                out["price_currency"] = price_currency
            _add_tick_data_quality(out)
            _add_tick_last_quality(out)
            _add_tick_context_fields(out)
            _round_tick_price_payload(out, price_digits)
            return _normalize_tick_missing_values(_compact_tick_summary(out))

        def _add_tick_summary_fields(payload: Dict[str, Any]) -> None:
            summary = _compact_summary_from_ticks()
            for key, value in summary.items():
                if key not in ("success", "symbol", "count", "timezone"):
                    payload[key] = value

        if output_mode in ("summary", "stats"):
            detailed_stats = output_mode == "stats"

            def _series_stats(s: pd.Series, *, total_count: int) -> Dict[str, Any]:
                vals = pd.to_numeric(s, errors="coerce")
                vals = vals[pd.notna(vals)].astype(float)
                n = int(vals.shape[0])
                if n <= 0:
                    out = {
                        "available": False,
                        "first": float("nan"),
                        "last": float("nan"),
                        "low": float("nan"),
                        "high": float("nan"),
                        "mean": float("nan"),
                        "std": float("nan"),
                        "stderr": float("nan"),
                        "kurtosis": float("nan"),
                        "change": float("nan"),
                        "change_pct": float("nan"),
                    }
                    if detailed_stats:
                        out["median"] = float("nan")
                        out["skew"] = float("nan")
                        out["q25"] = float("nan")
                        out["q75"] = float("nan")
                    if detailed_stats or n != int(total_count):
                        out["count"] = n
                    return _normalize_tick_missing_values(out)
                first = float(vals.iloc[0])
                last = float(vals.iloc[-1])
                low = float(vals.min())
                high = float(vals.max())
                mean = float(vals.mean())
                std = float(vals.std(ddof=0)) if n > 0 else float("nan")
                stderr = float(std / math.sqrt(n)) if n > 0 else float("nan")
                kurtosis = float(vals.kurtosis()) if n >= 4 else None
                change = float(last - first)
                change_pct = float((change / first) * 100.0) if first != 0.0 else float("nan")
                out = {
                    "first": first,
                    "last": last,
                    "low": low,
                    "high": high,
                    "mean": mean,
                    "std": std,
                    "stderr": stderr,
                    "kurtosis": kurtosis,
                    "change": change,
                    "change_pct": change_pct,
                }
                if detailed_stats:
                    out["median"] = float(vals.median())
                    out["skew"] = float(vals.skew()) if n >= 3 else float("nan")
                    out["q25"] = float(vals.quantile(0.25))
                    out["q75"] = float(vals.quantile(0.75))
                if detailed_stats or n != int(total_count):
                    out["count"] = n
                return _normalize_tick_missing_values(out)

            df_stats = df_ticks.copy()
            df_stats["mid"] = (
                (df_stats["bid"] + df_stats["ask"]) / 2.0
            ).round(price_digits + 1)
            df_stats["spread"] = (
                (df_stats["ask"] - df_stats["bid"])
                .where(df_stats["spread_sample_eligible"])
            )

            start_epoch = float(df_stats["__epoch"].iloc[0])
            end_epoch = float(df_stats["__epoch"].iloc[-1])
            duration_seconds = float(max(0.0, end_epoch - start_epoch))
            tick_rate_per_second = (
                float(len(df_stats) / duration_seconds) if duration_seconds > 0 else None
            )

            timezone = _timezone_label(use_client_tz=_use_ctz, client_tz=client_tz)

            out: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "output": "stats" if detailed_stats else "summary",
                "count": int(len(df_stats)),
                "start": df_stats["time"].iloc[0],
                "end": df_stats["time"].iloc[-1],
                "duration_seconds": duration_seconds,
                "tick_rate_per_second": tick_rate_per_second,
                "timezone": timezone,
                "stats": {
                    "bid": _series_stats(df_stats["bid"], total_count=len(df_stats)),
                    "ask": _series_stats(df_stats["ask"], total_count=len(df_stats)),
                    "mid": _series_stats(df_stats["mid"], total_count=len(df_stats)),
                    "spread": _series_stats(df_stats["spread"], total_count=len(df_stats)),
                },
            }
            out["last_quote"] = _last_snapshot_quote(df_stats)
            if duration_seconds <= 0:
                out["tick_rate_note"] = "< 1s window"
            small_summary_sample = (
                not detailed_stats
                and int(len(df_stats)) < _TICK_SUMMARY_MIN_ANALYTIC_TICKS
            )
            if not detailed_stats:
                out["sample_adequacy"] = not small_summary_sample
                if small_summary_sample:
                    out["sample_adequacy_note"] = (
                        f"Small sample ({len(df_stats)} ticks"
                        f" in {duration_seconds:g}s) - spread stats only."
                    )
                    out["sample_min_ticks"] = _TICK_SUMMARY_MIN_ANALYTIC_TICKS
                    out["stats"] = {"spread": out["stats"]["spread"]}
            spread_stats = out.get("stats", {}).get("spread")
            if isinstance(spread_stats, dict):
                try:
                    spread_first = float(spread_stats.get("first"))
                    spread_change_pct = spread_stats.get("change_pct")
                    spread_change_pct_f = float(spread_change_pct) if spread_change_pct is not None else float("nan")
                    if spread_first == 0.0 and not math.isfinite(spread_change_pct_f):
                        spread_stats["change_pct"] = None
                        out["spread_change_pct_note"] = "first spread was zero"
                except Exception:
                    pass
            if price_digits > 0:
                out["price_precision"] = int(price_digits)
            if price_point is not None:
                out["price_point"] = price_point
            if price_currency:
                out["price_currency"] = price_currency
            units = _tick_units_for_headers(headers)
            if units and detailed_stats:
                out["units"] = units
            if has_last and not small_summary_sample:
                out["stats"]["last"] = _series_stats(df_stats["last"], total_count=len(df_stats))

            trade_event_mask = df_stats["trade_event"].astype(bool)
            trade_event_count = int(trade_event_mask.sum())
            out["tick_count"] = int(len(df_stats))
            out["tick_count_event_basis"] = _TICK_COUNT_EVENT_BASIS
            out["trade_event_count"] = trade_event_count
            out["quote_update_count"] = quote_update_count
            out["quote_update_count_event_basis"] = _QUOTE_UPDATE_COUNT_EVENT_BASIS
            out["bid_update_count"] = bid_update_count
            out["ask_update_count"] = ask_update_count
            if detailed_stats:
                out["stats"]["tick_count"] = {
                    "kind": "tick_count",
                    "sum": int(len(df_stats)),
                    "per_second": tick_rate_per_second,
                }

            volume_kind: Optional[str] = None
            vol_vals = pd.Series(index=df_stats.index, dtype=float)
            real_trade_volume = df_stats["volume_real"].where(trade_event_mask)
            snapshot_trade_volume = df_stats["volume"].where(trade_event_mask)
            if bool((real_trade_volume.fillna(0.0) > 0.0).any()):
                volume_kind = "volume_real"
                vol_vals = real_trade_volume
            elif bool((snapshot_trade_volume.fillna(0.0) > 0.0).any()):
                volume_kind = "volume"
                vol_vals = snapshot_trade_volume

            if volume_kind is not None:
                vol_vals_num = pd.to_numeric(vol_vals, errors="coerce").astype(float)
                vol_sum = float(vol_vals_num.fillna(0.0).sum())
                vol_nonzero_count = int((vol_vals_num.fillna(0.0) != 0.0).sum())
                vol_out: Dict[str, Any] = {
                    "kind": volume_kind,
                    "sum": vol_sum,
                    "per_second": (
                        float(vol_sum / duration_seconds) if duration_seconds > 0 else None
                    ),
                    "per_trade_event": float(vol_sum / float(trade_event_count or 1)),
                    "nonzero_share": float(vol_nonzero_count) / float(trade_event_count or 1),
                }
                try:
                    mean_v = float(vol_vals_num.mean())
                    std_v = float(vol_vals_num.std(ddof=0))
                    vol_out["cv"] = (
                        float(std_v / mean_v) if (mean_v != 0.0 and not math.isnan(mean_v)) else float("nan")
                    )
                except Exception:
                    pass

                if vol_sum > 0.0:
                    try:
                        top_n = min(10, int(len(vol_vals_num)))
                        if top_n > 0:
                            vol_top = vol_vals_num.fillna(0.0).sort_values(ascending=False).iloc[:top_n]
                            vol_out["top10_share"] = float(vol_top.sum() / vol_sum)
                    except Exception:
                        pass
                    try:
                        q95 = float(vol_vals_num.quantile(0.95))
                        spikes = vol_vals_num[vol_vals_num >= q95]
                        vol_out["spike95_count"] = int(spikes.shape[0])
                        vol_out["spike95_share"] = float(spikes.fillna(0.0).sum() / vol_sum)
                    except Exception:
                        pass
                    try:
                        w = vol_vals_num.fillna(0.0)
                        vol_out["vwap_mid"] = float((df_stats["mid"] * w).sum() / vol_sum)
                        if has_last:
                            vol_out["vwap_last"] = float((df_stats["last"] * w).sum() / vol_sum)
                    except Exception:
                        pass

                    try:
                        dmid = df_stats["mid"].diff().abs()
                        corr_df = pd.DataFrame(
                            {"volume": vol_vals_num, "abs_mid_change": dmid}
                        ).dropna()
                        if (
                            int(corr_df.shape[0]) >= 3
                            and int(corr_df["volume"].nunique()) > 1
                            and int(corr_df["abs_mid_change"].nunique()) > 1
                        ):
                            vol_out["corr_abs_mid_change"] = float(
                                corr_df["volume"].corr(corr_df["abs_mid_change"])
                            )
                    except Exception:
                        pass

                    try:
                        n_v = int(vol_vals_num.shape[0])
                        if n_v >= 4:
                            half = max(1, int(n_v // 2))
                            first_mean = float(vol_vals_num.iloc[:half].mean())
                            second_mean = float(vol_vals_num.iloc[half:].mean())
                            vol_out["half_ratio"] = (
                                float(second_mean / first_mean) if first_mean != 0.0 else float("nan")
                            )
                    except Exception:
                        pass

                if detailed_stats:
                    vol_out["dist"] = _series_stats(
                        vol_vals_num, total_count=trade_event_count
                    )
                if not small_summary_sample:
                    out["stats"][volume_kind] = vol_out

            _add_tick_data_quality(out)
            _add_tick_last_quality(out)
            _add_tick_context_fields(out)
            _round_tick_price_payload(out, price_digits)
            return _normalize_tick_missing_values(
                out if detailed_stats else _compact_tick_summary(out)
            )

        # If simplify mode requests approximation or resampling, use shared path
        if simplify_present and simplify_mode in ('approximate', 'resample'):
            df_for_simplify = df_ticks.copy()
            if simplify_mode == "resample":
                trade_mask = df_for_simplify["trade_event"].astype(bool)
                for volume_column in ("volume", "volume_real"):
                    if volume_column in df_for_simplify:
                        df_for_simplify[volume_column] = df_for_simplify[
                            volume_column
                        ].where(trade_mask, 0.0)
            df_out, simplify_meta = _simplify_dataframe_rows_ext(
                df_for_simplify, headers, simplify_used
            )
            rows = _format_numeric_rows_from_df(df_out, headers, stringify=False)
            rows = _round_row_price_columns(
                rows,
                headers,
                digits=price_digits,
                price_columns=_TICK_PRICE_COLUMNS,
            )
            table_payload = _table_from_rows(headers, rows)
            payload = {
                "success": True,
                "symbol": symbol,
                "count": len(rows),
            }
            payload.update(table_payload)
            payload["timezone"] = _timezone_label(use_client_tz=_use_ctz, client_tz=client_tz)
            if price_point is not None:
                payload["price_point"] = price_point
            if price_currency:
                payload["price_currency"] = price_currency
            units = _tick_units_for_headers(headers)
            if units:
                payload["units"] = units
            payload["tick_count"] = int(original_count)
            payload["trade_event_count"] = int(sum(trade_events))
            payload["quote_update_count"] = quote_update_count
            _add_tick_summary_fields(payload)
            if has_flags:
                payload["flags_legend"] = _observed_tick_flags_decoded(flags)
            if simplify_meta is not None and original_count > len(rows):
                payload["simplified"] = True
                meta = dict(simplify_meta)
                meta["columns"] = [
                    c
                    for c in ["bid", "ask"]
                    + (["last"] if has_last else [])
                    + (["volume"] if has_volume else [])
                    + (["volume_real"] if has_real_volume else [])
                ]
                meta["original_rows"] = int(original_count)
                meta["returned_rows"] = int(len(rows))
                if simplify_target_points is not None:
                    meta["points"] = int(simplify_target_points)
                    meta["target_points"] = int(simplify_target_points)
                payload["simplify"] = meta
            _add_tick_data_quality(payload)
            _add_tick_last_quality(payload)
            _add_tick_context_fields(payload)
            return _normalize_tick_missing_values(payload)
        # Optional simplification based on a chosen y-series
        select_indices = list(range(original_count))
        _simp_method_used: Optional[str] = None
        _simp_params_meta: Optional[Dict[str, Any]] = None
        if simplify_present and original_count > 3:
            try:
                # Always represent available bid/ask/last and volume columns.
                cols: List[str] = ['bid', 'ask']
                if has_last:
                    cols.append('last')
                if has_volume:
                    cols.append('volume')
                if has_real_volume:
                    cols.append('volume_real')
                n_out = (
                    simplify_target_points
                    if simplify_target_points is not None
                    else _choose_simplify_points(original_count, simplify_used)
                )
                per = max(3, int(round(n_out / max(1, len(cols)))))
                idx_set: set = set([0, original_count - 1])
                params_accum: Dict[str, Any] = {}
                method_used_overall = None
                extracted_columns: Dict[str, List[float]] = {
                    "bid": bids,
                    "ask": asks,
                    "last": lasts,
                    "volume": volumes,
                    "volume_real": volumes_real,
                }
                series_by_col: Dict[str, List[float]] = {c: extracted_columns[c] for c in cols}
                for c in cols:
                    series = series_by_col[c]
                    sub_spec = dict(simplify)
                    sub_spec['points'] = per
                    idxs, method_used, params_meta = _select_indices_for_timeseries(_epochs, series, sub_spec)
                    method_used_overall = method_used
                    for i in idxs:
                        if 0 <= int(i) < original_count:
                            idx_set.add(int(i))
                    try:
                        if params_meta:
                            for k2, v2 in params_meta.items():
                                params_accum.setdefault(k2, v2)
                    except Exception:
                        pass
                union_idxs = sorted(idx_set)
                # Build composite metric for refinement/top-up
                mins: Dict[str, float] = {}
                ranges: Dict[str, float] = {}
                for c in cols:
                    vals = series_by_col[c]
                    if vals:
                        mn, mx = min(vals), max(vals)
                        ranges[c] = max(1e-12, mx - mn)
                        mins[c] = mn
                    else:
                        ranges[c] = 1.0
                        mins[c] = 0.0
                comp: List[float] = []
                for i in range(original_count):
                    s = 0.0
                    for c in cols:
                        vv = (series_by_col[c][i] - mins[c]) / ranges[c]
                        s += abs(vv)
                    comp.append(s)
                if len(union_idxs) > n_out:
                    refined = _lttb_select_indices(_epochs, comp, n_out)
                    select_indices = sorted(set(int(i) for i in refined if 0 <= i < original_count))
                elif len(union_idxs) < n_out:
                    refined = _lttb_select_indices(_epochs, comp, n_out)
                    merged = sorted(set(union_idxs).union(refined))
                    if len(merged) > n_out:
                        keep = set([0, original_count - 1])
                        candidates = [(comp[i], i) for i in merged if i not in keep]
                        candidates.sort(reverse=True)
                        for _, i in candidates:
                            keep.add(i)
                            if len(keep) >= n_out:
                                break
                        select_indices = sorted(keep)
                    else:
                        select_indices = merged
                else:
                    select_indices = union_idxs
                _simp_method_used = method_used_overall or str((simplify_used or {}).get('method', SIMPLIFY_DEFAULT_METHOD)).lower()
                _simp_params_meta = params_accum
            except Exception:
                select_indices = list(range(original_count))

        rows = []
        for i in select_indices:
            time_value = _format_tick_time(_epochs[i])
            values = [time_value]
            if full_rows:
                epoch_value = _epochs[i]
                values.append(int(epoch_value) if float(epoch_value).is_integer() else float(epoch_value))
            values.extend(
                [
                    _round_price_value(effective_bids[i], price_digits),
                    _round_price_value(effective_asks[i], price_digits),
                ]
            )
            if include_quote_type:
                values.append(quote_types[i])
            if full_rows:
                bid_value = effective_bids[i]
                ask_value = effective_asks[i]
                snapshot_spread_valid = bool(
                    bid_value is not None
                    and ask_value is not None
                    and ask_value > bid_value
                )
                values.extend(
                    [
                        quote_update_types[i],
                        bid_changed_flags[i],
                        ask_changed_flags[i],
                        snapshot_spread_valid,
                        "quote_snapshot" if snapshot_spread_valid else "unavailable",
                        spread_sample_eligible_flags[i],
                    ]
                )
                mid = (
                    canonical_quote_midpoint(bid_value, ask_value)
                    if snapshot_spread_valid
                    else None
                )
                spread = (
                    canonical_quote_spread(bid_value, ask_value)
                    if snapshot_spread_valid
                    else None
                )
                spread_points = _tick_spread_points(spread, price_point)
                spread_pct = _tick_spread_pct(spread, mid)
                gap_ms = None if i <= 0 else float((_epochs[i] - _epochs[i - 1]) * 1000.0)
                values.extend(
                    [
                        float(mid) if mid is not None else None,
                        _round_price_value(spread, price_digits),
                    ]
                )
                if price_point is not None:
                    values.append(spread_points)
                if points_per_pip is not None:
                    values.append(
                        round(spread_points / points_per_pip, 4)
                        if spread_points is not None
                        else None
                    )
                values.extend([spread_pct, gap_ms])
            values.append(_round_price_value(_finite_or_none(lasts[i]), price_digits))
            values.append(_finite_or_none(volumes[i]))
            values.append(_finite_or_none(volumes_real[i]))
            values.append(int(flags[i]) if flags[i] is not None else 0)
            values.append(_decode_tick_flags(flags[i]))
            rows.append(values)

        table_payload = _table_from_rows(headers, rows)
        payload = {
            "success": True,
            "symbol": symbol,
            "count": len(rows),
        }
        payload.update(table_payload)
        payload["timezone"] = _timezone_label(use_client_tz=_use_ctz, client_tz=client_tz)
        if price_point is not None:
            payload["price_point"] = price_point
        if price_currency:
            payload["price_currency"] = price_currency
        units = _tick_units_for_headers(headers)
        if units:
            payload["units"] = units
        payload["tick_count"] = int(original_count)
        payload["trade_event_count"] = int(sum(trade_events))
        payload["quote_update_count"] = quote_update_count
        _add_tick_summary_fields(payload)
        if has_flags:
            payload["flags_legend"] = _observed_tick_flags_decoded(flags)
        _add_tick_data_quality(payload)
        _add_tick_last_quality(payload)
        _add_tick_context_fields(payload)
        if simplify_present and original_count > len(rows):
            payload["simplified"] = True
            meta = {
                "method": (_simp_method_used or str((simplify_used or {}).get('method', SIMPLIFY_DEFAULT_METHOD)).lower()),
                "original_rows": original_count,
                "multi_column": True,
                "columns": [
                    c
                    for c in ["bid", "ask"]
                    + (["last"] if has_last else [])
                    + (["volume"] if has_volume else [])
                    + (["volume_real"] if has_real_volume else [])
                ],
            }
            try:
                if _simp_params_meta:
                    meta.update(_simp_params_meta)
                else:
                    # Return key params if present
                    for key in ("epsilon", "max_error", "segments", "points", "ratio"):
                        if key in (simplify or {}):
                            meta[key] = (simplify or {})[key]
            except Exception:
                pass
            meta["returned_rows"] = int(len(rows))
            if simplify_target_points is not None:
                meta["points"] = int(simplify_target_points)
                meta["target_points"] = int(simplify_target_points)
                meta["per_column_target"] = max(
                    3,
                    int(
                        round(
                            simplify_target_points
                            / max(1, len(meta.get("columns") or []))
                        )
                    ),
                )
            payload["simplify"] = meta
        return _normalize_tick_missing_values(payload)
    except Exception as e:
        return {"error": f"Error getting ticks: {str(e)}"}

