from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from ...services.data_service.errors import attach_empty_range_weekend_context
from ...utils.continuation import (
    decode_continuation_cursor,
    encode_continuation_cursor,
)
from ...utils.freshness import (
    QUOTE_STALE_SECONDS,
    format_freshness_label,
)
from ...utils.freshness import (
    format_age_seconds as _format_age_seconds,
)
from ...utils.market_metadata import (
    FRESHNESS_ANCHOR_QUERY_EXPECTED_END,
    FRESHNESS_ANCHOR_WALL_CLOCK,
    FRESHNESS_METRIC_LAST_COMPLETED_BAR_AGE,
    FRESHNESS_METRIC_LAST_TICK_AGE,
    FRESHNESS_METRIC_REQUESTED_RANGE_END_GAP,
    attach_candle_volume_semantics,
    build_tick_freshness_context,
    normalize_policy_relaxed,
)
from ...utils.quote import (
    canonical_quote_midpoint,
    resolve_quote_tick,
    tick_epoch,
)
from ...utils.symbol import (
    find_live_extended_session_symbols,
    symbol_suggestions_from_gateway,
)
from ...utils.time import bar_close_epoch, format_datetime_utc
from ...utils.utils import (
    _iana_timezone_datetime_issue,
    _is_in_progress_calendar_day_end,
    _parse_end_datetime,
    _parse_start_datetime,
)
from ..error_envelope import build_error_payload
from ..execution_logging import run_logged_operation
from ..mt5_gateway import mt5_connection_error
from ..output_contract import attach_collection_contract
from ..runtime_metadata import attach_mt5_source
from .requests import (
    DATA_FETCH_CANDLES_DEFAULT_LIMIT,
    DATA_FETCH_TICKS_DEFAULT_LIMIT,
    DataFetchCandlesRequest,
    DataFetchTicksRequest,
    WaitEventRequest,
)
from .wait_events import run_wait_event_loop

logger = logging.getLogger(__name__)

_TICK_DETAIL_FORMATS = {
    "compact": "rows",
    "summary": "summary",
    "standard": "full_rows",
    "full": "full_rows",
}

_COMPACT_TICK_TOP_LEVEL_FIELDS = (
    "success",
    "symbol",
    "count",
    "feed_tier",
    "data",
    "empty",
    "empty_reason",
    "no_data_reason",
    "data_window",
    "timezone",
    "price_precision",
    "price_point",
    "price_currency",
    "units",
    "freshness",
    "freshness_state",
    "freshness_reason",
    "data_age_seconds",
    "data_age_anchor",
    "data_age_metric",
    "data_stale",
    "timestamp_ahead_of_wall_clock",
    "timestamp_in_future",
    "timestamp_skew_seconds",
    "timestamp_skew_tolerance_seconds",
    "timestamp_warning",
    "usable_for_live_trading",
    "usable_for_live_trading_basis",
    "execution_blockers",
    "last_quote",
    "live_max_age_seconds",
    "market_status",
    "market_status_reason",
    "market_status_source",
    "freshness_policy_relaxed",
    "note",
    "suggestion",
    "simplified",
    "simplify",
    "query_applied",
    "history_window_truncated",
    "history_window_limit_days",
    "history_window_floor",
    "effective_start",
    "warnings",
    "_tick_page",
)

def _ensure_gateway_connection(gateway: Any) -> Dict[str, Any] | None:
    return mt5_connection_error(gateway)


def run_data_fetch_candles(
    request: DataFetchCandlesRequest,
    *,
    gateway: Any,
    fetch_candles_impl: Any,
) -> Dict[str, Any]:
    effective_limit = _effective_candle_limit(request)
    return run_logged_operation(
        logger,
        operation="data_fetch_candles",
        symbol=request.symbol,
        timeframe=request.timeframe,
        limit=effective_limit,
        func=lambda: _run_data_fetch_candles_impl(
            request=request,
            gateway=gateway,
            fetch_candles_impl=fetch_candles_impl,
            effective_limit=effective_limit,
        ),
    )


def _effective_tick_limit(request: DataFetchTicksRequest) -> int:
    try:
        limit = max(1, int(request.limit))
    except Exception:
        limit = DATA_FETCH_TICKS_DEFAULT_LIMIT
    return limit


def run_data_fetch_ticks(
    request: DataFetchTicksRequest,
    *,
    gateway: Any,
    fetch_ticks_impl: Any,
) -> Dict[str, Any]:
    effective_limit = _effective_tick_limit(request)
    return run_logged_operation(
        logger,
        operation="data_fetch_ticks",
        symbol=request.symbol,
        limit=effective_limit,
        detail=request.detail,
        func=lambda: _run_data_fetch_ticks_impl(
            request=request,
            gateway=gateway,
            fetch_ticks_impl=fetch_ticks_impl,
            effective_limit=effective_limit,
        ),
    )


def run_wait_event(
    request: WaitEventRequest,
    *,
    gateway: Any,
    sleep_impl: Any = time.sleep,
    monotonic_impl: Any = time.monotonic,
    now_utc_impl: Any = lambda: datetime.now(timezone.utc),
) -> Dict[str, Any]:
    result = run_logged_operation(
        logger,
        operation="wait_event",
        watch_for=len(request.watch_for or []),
        end_on=len(request.end_on),
        poll_interval_seconds=request.poll_interval_seconds,
        func=lambda: _run_wait_event_impl(
            request=request,
            gateway=gateway,
            sleep_impl=sleep_impl,
            monotonic_impl=monotonic_impl,
            now_utc_impl=now_utc_impl,
        ),
    )
    payload = result
    if not _wait_event_needs_gateway(request):
        return payload
    return attach_mt5_source(payload, gateway=gateway, include_errors=True)


def _run_data_fetch_candles_impl(
    *,
    request: DataFetchCandlesRequest,
    gateway: Any,
    fetch_candles_impl: Any,
    effective_limit: Optional[int] = None,
) -> Dict[str, Any]:
    connection_error = _ensure_gateway_connection(gateway)
    if connection_error is not None:
        return connection_error
    future_bound = _future_bound(request)
    if future_bound is not None:
        field, value = future_bound
        details: Dict[str, Any] = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "timezone": "UTC",
        }
        if request.start is not None:
            details["start"] = str(request.start)
        if request.end is not None:
            details["end"] = str(request.end)
        return build_error_payload(
            f"{field} datetime {value} is in the future; historical candle ranges must have elapsed.",
            code="future_date_range",
            operation="data_fetch_candles",
            details=details,
            remediation="Use start and end timestamps at or before the current time.",
        )
    selection = str(request.selection or "").strip().lower()
    if (
        request.start in (None, "")
        and request.end not in (None, "")
        and selection == "first_n"
    ):
        return build_error_payload(
            "selection=first_n is not supported for end-only candle queries.",
            code="selection_unsupported_for_end_only",
            operation="data_fetch_candles",
            details={"selection": "first_n", "end": str(request.end)},
            remediation=(
                "Omit selection or pass last_n, or supply start to page from "
                "the beginning of a bounded window."
            ),
        )
    fetch_start = request.start
    page_offset = 0
    if request.cursor:
        if not request.start:
            return build_error_payload(
                "cursor requires start for a start-anchored candle query.",
                code="data_fetch_candles_invalid_cursor",
                operation="data_fetch_candles",
                remediation=(
                    "Reuse next_cursor with the original start value and unchanged "
                    "symbol, timeframe, and end."
                ),
            )
        try:
            fetch_start, page_offset = _decode_candle_cursor(request.cursor, request)
        except ValueError as exc:
            return build_error_payload(
                str(exc),
                code="data_fetch_candles_invalid_cursor",
                operation="data_fetch_candles",
                remediation=(
                    "Use next_cursor from the preceding candle page without changing "
                    "symbol, timeframe, start, or end."
                ),
            )
    result = fetch_candles_impl(
        symbol=request.symbol,
        timeframe=request.timeframe,
        limit=effective_limit if effective_limit is not None else request.limit,
        start=fetch_start,
        end=request.end,
        range_selection=request.selection,
        ohlcv=request.ohlcv,
        indicators=request.indicators,
        denoise=request.denoise,
        simplify=request.simplify,
        time_as_epoch=not _iso_timestamp_requested(request.timestamp_format),
        force_utc=_force_utc_timestamps(request.timestamp_format),
        include_spread=request.include_spread,
        include_incomplete=request.include_incomplete,
        allow_stale=request.allow_stale,
    )
    result = _normalize_candle_query_error(
        result,
        request=request,
        gateway=gateway,
    )
    detail_mode = str(request.detail or "compact").strip().lower()
    if isinstance(result, dict):
        _normalize_public_candle_timestamp_mode(
            result,
            include_raw=detail_mode == "full",
        )
        limit_explicit = "limit" in getattr(request, "model_fields_set", set())
        applied_limit = (
            effective_limit if effective_limit is not None else request.limit
        )
        if request.cursor:
            query_applied = result.get("query_applied")
            if isinstance(query_applied, dict):
                query_applied["start"] = request.start
                query_applied["cursor_applied"] = True
        if bool(getattr(request, "explain_indicators", False)):
            _attach_indicator_explanations(result)
        _apply_range_limit_cap(
            result,
            limit=applied_limit,
            limit_explicit=limit_explicit,
            start=request.start,
            end=request.end,
            request=request,
            page_offset=page_offset,
        )
        query_applied = result.get("query_applied")
        if not isinstance(query_applied, dict):
            query_applied = {}
        _disclose_in_progress_end_clamp(query_applied, request.end)
        if query_applied:
            result["query_applied"] = query_applied
        _reconcile_returned_window_completeness(result)
        if request.start or request.end:
            _normalize_range_limit_contract(
                result,
                effective_limit=applied_limit,
                limit_explicit=limit_explicit,
            )
        _annotate_empty_candle_result(result, request=request)
        _normalize_candle_count_field(result)
        _prune_zero_candle_exclusions(result)
        if detail_mode == "compact":
            result = _compact_candles_payload(result)
            _slim_projected_candles_payload(result)
            _drop_redundant_session_gap_warnings(result)
        elif detail_mode == "summary":
            result = _summary_candles_payload(result)
        elif detail_mode == "standard":
            result = _standard_candles_payload(result)
        _attach_candle_machine_freshness(result)
        _attach_latest_candle_quote_freshness(
            result,
            request=request,
            gateway=gateway,
        )
        _attach_forming_candle_update_freshness(
            result,
            request=request,
            gateway=gateway,
        )
        _attach_forming_indicator_warning(result, request=request)
        _attach_candle_data_as_of(result, timeframe=request.timeframe)
        _reconcile_returned_query_end_gap(result)
        result = attach_mt5_source(result, gateway=gateway)
    if isinstance(result, dict) and isinstance(result.get("data"), list):
        out = attach_collection_contract(
            result,
            collection_kind="time_series",
            series=result["data"],
            include_contract_meta=detail_mode == "full",
        )
        if detail_mode == "full" and isinstance(out, dict):
            out.pop("series", None)
            out.pop("canonical_source", None)
        return out
    return result


def _attach_candle_data_as_of(payload: Dict[str, Any], *, timeframe: str) -> None:
    if not isinstance(payload, dict) or payload.get("error"):
        return
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        latest = payload.get("latest_candle")
        data = [latest] if isinstance(latest, dict) else []
    if not data:
        return
    last = data[-1]
    if not isinstance(last, dict):
        return
    raw_time = last.get("time")
    open_epoch: Optional[float] = None
    try:
        open_epoch = float(raw_time)
    except (TypeError, ValueError):
        parsed = _parse_start_datetime(str(raw_time or ""))
        if parsed is not None:
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            open_epoch = parsed.replace(tzinfo=timezone.utc).timestamp()
    if open_epoch is None:
        return
    try:
        close_epoch = bar_close_epoch(open_epoch, timeframe)
    except Exception:
        close_epoch = open_epoch
    scheduled_close = format_datetime_utc(
        datetime.fromtimestamp(float(close_epoch), tz=timezone.utc)
    )
    if last.get("bar_state") == "forming":
        payload["scheduled_bar_close"] = scheduled_close
        retrieval_time = payload.get("as_of")
        if retrieval_time in (None, ""):
            retrieval_time = format_datetime_utc(datetime.now(timezone.utc))
        payload["data_as_of"] = retrieval_time
        payload["data_as_of_basis"] = "retrieval_time_unverified"
    else:
        payload["data_as_of"] = scheduled_close
        payload["data_as_of_basis"] = "completed_bar_close"
    if payload.get("as_of") not in (None, ""):
        payload.setdefault("as_of_basis", "retrieval_time")


def _epoch_from_public_timestamp(value: Any) -> Optional[float]:
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        epoch = float(value)
        return epoch if np.isfinite(epoch) else None
    text = str(value).strip()
    if not text:
        return None
    parsed = _parse_start_datetime(text)
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).timestamp()


def _reconcile_returned_query_end_gap(payload: Dict[str, Any]) -> None:
    """Compute coverage gap from the retained page, not the pre-truncation fetch."""
    if not isinstance(payload, dict) or payload.get("error"):
        return
    if payload.get("data_as_of_basis") not in (None, "", "completed_bar_close"):
        return
    query_applied = payload.get("query_applied")
    if not isinstance(query_applied, dict):
        return
    resolved_end = query_applied.get("resolved_end")
    data_as_of = payload.get("data_as_of")
    end_epoch = _epoch_from_public_timestamp(resolved_end)
    as_of_epoch = _epoch_from_public_timestamp(data_as_of)
    if end_epoch is None or as_of_epoch is None:
        return
    seconds = round(max(0.0, end_epoch - as_of_epoch), 3)
    payload["query_end_gap_seconds"] = seconds
    gap_text = _format_age_seconds(seconds)
    if gap_text is not None:
        payload["query_end_gap"] = gap_text
    if "query_end_gap_anchor" in payload:
        payload["query_end_gap_anchor"] = FRESHNESS_ANCHOR_QUERY_EXPECTED_END
    if "query_end_gap_metric" in payload:
        payload["query_end_gap_metric"] = FRESHNESS_METRIC_REQUESTED_RANGE_END_GAP
    meta = payload.get("meta")
    diagnostics = meta.get("diagnostics") if isinstance(meta, dict) else None
    freshness = diagnostics.get("freshness") if isinstance(diagnostics, dict) else None
    if isinstance(freshness, dict):
        freshness["query_end_gap_seconds"] = seconds


def _attach_forming_indicator_warning(
    payload: Dict[str, Any],
    *,
    request: DataFetchCandlesRequest,
) -> None:
    if payload.get("error") or not bool(getattr(request, "include_incomplete", False)):
        return
    if request.indicators in (None, "", [], {}):
        return
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return
    last = data[-1]
    if not isinstance(last, dict) or last.get("bar_state") != "forming":
        return
    payload["indicators_include_forming_bar"] = True
    warning = (
        "Indicator values on the latest row include the current forming bar "
        "and can change before that bar closes."
    )
    existing = payload.get("warnings")
    if isinstance(existing, list):
        if warning not in existing:
            payload["warnings"] = [*existing, warning]
    elif existing:
        payload["warnings"] = [existing, warning]
    else:
        payload["warnings"] = [warning]


def _attach_forming_candle_update_freshness(
    payload: Dict[str, Any],
    *,
    request: DataFetchCandlesRequest,
    gateway: Any,
) -> None:
    if not request.include_incomplete or payload.get("error"):
        return
    data_window = payload.get("data_window")
    if not isinstance(data_window, dict) or data_window.get("latest_bar_complete") is not False:
        return
    try:
        tick = gateway.symbol_info_tick(request.symbol)
    except Exception:
        return
    tick_msc = getattr(tick, "time_msc", None) if tick is not None else None
    tick_seconds = getattr(tick, "time", None) if tick is not None else None
    try:
        tick_epoch = float(tick_msc) / 1000.0 if tick_msc else float(tick_seconds)
    except (TypeError, ValueError):
        return
    if not np.isfinite(tick_epoch) or tick_epoch <= 0:
        return
    update_age = max(0.0, float(time.time()) - tick_epoch)
    bar_open_age = payload.get("data_age_seconds")
    try:
        bar_open_age_value = max(0.0, float(bar_open_age))
    except (TypeError, ValueError):
        bar_open_age_value = None
    if bar_open_age_value is not None:
        payload["bar_open_age_seconds"] = round(bar_open_age_value, 3)
        data_window["latest_bar_open_age_seconds"] = round(bar_open_age_value, 3)
    payload["market_tick_age_seconds"] = round(update_age, 3)
    data_window["market_tick_age_seconds"] = round(update_age, 3)
    update_text = _format_age_seconds(update_age)
    if bar_open_age_value is not None:
        payload["data_age_seconds"] = round(bar_open_age_value, 3)
        payload["data_age_metric"] = "latest_forming_bar_open_age_seconds"
        payload["freshness"] = (
            f"forming bar open {_format_age_seconds(bar_open_age_value)} ago; "
            f"market tick {update_text} ago; forming-bar update time unverified"
        )
    else:
        payload["freshness"] = (
            f"forming bar; market tick {update_text} ago; "
            "forming-bar update time unverified"
        )
    payload["data_age_anchor"] = FRESHNESS_ANCHOR_WALL_CLOCK
    payload["forming_bar_update_verified"] = False
    if update_age > float(QUOTE_STALE_SECONDS):
        payload["data_stale"] = True
    warning = {
        "code": "forming_bar_unverified",
        "scope": "candles",
        "message": (
            "The forming bar is included; its last update time could not be "
            f"verified, but the market tick is {update_text} old."
        ),
        "market_tick_age_seconds": round(update_age, 3),
    }
    existing = payload.get("warnings")
    if isinstance(existing, list):
        if warning not in existing:
            payload["warnings"] = [*existing, warning]
    elif existing:
        payload["warnings"] = [existing, warning]
    else:
        payload["warnings"] = [warning]


def _forming_candle_present(payload: Dict[str, Any]) -> bool:
    if str(payload.get("forming_candle_status") or "").strip().lower() == "included":
        return True
    data_window = payload.get("data_window")
    if isinstance(data_window, dict) and data_window.get("latest_bar_complete") is False:
        return True
    rows = payload.get("data")
    if isinstance(rows, list) and rows:
        last = rows[-1]
        if isinstance(last, dict) and last.get("bar_state") == "forming":
            return True
    return False


def _iso_timestamp_requested(value: Any) -> bool:
    return str(value or "").strip().lower() in {"iso", "iso_utc"}


def _force_utc_timestamps(value: Any) -> bool:
    return str(value or "").strip().lower() == "iso_utc"


def _attach_latest_candle_quote_freshness(
    payload: Dict[str, Any],
    *,
    request: DataFetchCandlesRequest,
    gateway: Any,
) -> None:
    """Prevent a stale latest quote from being presented as a fresh candle mark."""
    if request.start or request.end or payload.get("error"):
        return
    if request.include_incomplete and _forming_candle_present(payload):
        return
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        return
    resolved_symbol = str(payload.get("symbol") or request.symbol).strip()
    symbol_input = str(request.symbol or "").strip()
    if resolved_symbol and symbol_input and resolved_symbol != symbol_input:
        payload["symbol_input"] = symbol_input
    try:
        now_epoch = time.time()
        tick, _ = resolve_quote_tick(gateway, resolved_symbol, now_epoch=now_epoch)
        quote_context = build_tick_freshness_context(
            resolved_symbol,
            tick_epoch=tick_epoch(tick),
            now_epoch=now_epoch,
            item="tick",
        )
    except Exception:
        return
    quote_age = quote_context.get("data_age_seconds")
    freshness_reason = quote_context.get("freshness_reason")
    quote_stale: Optional[bool]
    if quote_age is None and freshness_reason is None:
        quote_stale = None
    else:
        quote_stale = quote_context.get("data_stale") is True
    payload["latest_quote_stale"] = quote_stale
    payload["latest_quote_age_seconds"] = quote_age
    payload["freshness_reason"] = freshness_reason
    payload["freshness_basis"] = (
        "bar_policy_and_latest_quote" if quote_stale else payload.get("freshness_basis")
    )
    if quote_stale is not True:
        return
    payload["data_stale"] = True
    payload["freshness_basis"] = "bar_policy_and_latest_quote"
    payload["freshness_reason"] = str(
        quote_context.get("freshness_reason") or "latest_quote_stale"
    )
    for key in ("market_status", "market_status_reason", "market_status_source"):
        if quote_context.get(key) is not None:
            payload[key] = quote_context[key]
    freshness_label = format_freshness_label(
        data_stale=True,
        market_status=payload.get("market_status"),
        market_status_reason=payload.get("market_status_reason"),
        age_seconds=payload.get("data_age_seconds"),
        item="bar",
    )
    if freshness_label:
        payload["freshness"] = freshness_label
    if payload.get("history_policy_ok") is False:
        payload["stale_warning"] = (
            "The latest quote is stale, so the last candle must not be treated as a "
            "live mark. Completed-bar history is also outside the freshness policy window."
        )
    else:
        payload["stale_warning"] = (
            "The latest quote is stale, so the last candle must not be treated as a "
            "live mark even though completed-bar history is within policy."
        )


def _normalize_candle_query_error(  # noqa: C901
    result: Any,
    *,
    request: DataFetchCandlesRequest,
    gateway: Any = None,
) -> Any:
    if not isinstance(result, dict) or not result.get("error"):
        return result
    if result.get("error_code") == "data_fetch_candles_no_data":
        details = result.get("details")
        details = dict(details) if isinstance(details, dict) else {}
        empty_reason = str(details.get("no_data_reason") or "no_candles_in_range")
        payload: Dict[str, Any] = {
            "success": True,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "count": 0,
            "data": [],
            "row_key": "data",
            "empty": True,
            "empty_reason": empty_reason,
            "timezone": "UTC",
        }
        if details.get("no_data_reason") is not None:
            payload["no_data_reason"] = details["no_data_reason"]
        for key in (
            "market_status",
            "market_status_reason",
            "note",
            "requested_range",
            "available_range",
        ):
            if details.get(key) is not None:
                payload[key] = details[key]
        for key in ("query_applied", "warnings", "diagnostics"):
            if result.get(key) is not None:
                payload[key] = result[key]
        return payload
    if result.get("error_code"):
        return result

    message = str(result["error"])
    normalized = message.lower()
    error_code: Optional[str] = None
    remediation: Optional[str] = None
    dst_issue = next(
        (
            (field, issue)
            for field in ("start", "end")
            if (value := getattr(request, field, None)) is not None
            if (issue := _iana_timezone_datetime_issue(str(value))) is not None
        ),
        None,
    )

    if dst_issue is not None:
        _, issue = dst_issue
        error_code = str(issue["error_code"])
        message = str(issue["error"])
        remediation = str(issue["remediation"])
    elif "not found" in normalized and "symbol" in normalized:
        error_code = "symbol_not_found"
        message = f"Symbol '{request.symbol}' was not found in MT5."
        remediation = (
            f"Use symbols_list(search_term='{request.symbol}') to find the broker's "
            "exact MT5 symbol name, including any suffixes or aliases."
        )
    elif "could not parse date" in normalized or "invalid date" in normalized:
        error_code = "data_fetch_candles_invalid_date"
        remediation = (
            "Use an ISO 8601 date or timestamp, for example 2026-08-03 or "
            "2026-08-03T14:30:00Z."
        )
    elif (
        "start_datetime must be before end_datetime" in normalized
        or "start must be before or equal to end" in normalized
    ):
        error_code = "invalid_date_range"
        remediation = "Set start to a timestamp earlier than or equal to end."
    elif "in the future" in normalized and "start" in normalized:
        error_code = "future_date_range"
        remediation = "Use a start timestamp at or before the current time."
    elif (
        "before mt5's supported history boundary" in normalized
        or "mt5 rejected the requested candle date range" in normalized
    ):
        error_code = "data_fetch_candles_unsupported_date_range"
        remediation = (
            "Use start and end timestamps on or after 1970-01-01T00:00:00Z. "
            "For an all-history query, use that boundary as start; MT5 will return "
            "the first broker bars available on or after it."
        )
    elif "data appears stale" in normalized:
        error_code = "data_fetch_candles_stale_data"
        remediation = (
            "Confirm the market session and broker feed, or set allow_stale=true "
            "when historical data is intentionally acceptable."
        )
    elif (
        "data_shape_invalid" in normalized
        or ("keyerror" in normalized and "'time'" in normalized)
    ):
        error_code = "data_shape_invalid"
        remediation = (
            "Retry the request. If it persists, the broker history payload is "
            "not in the expected MT5 rate shape."
        )
    elif "invalid ohlcv token" in normalized:
        error_code = "invalid_ohlcv_selector"
        remediation = (
            "Use open, high, low, close, volume (or o,h,l,c,v)."
        )
    elif (
        "indicator 'macd' requires fast < slow" in normalized
        or "requires fast < slow" in normalized
    ):
        error_code = "invalid_indicator_parameters"
        remediation = (
            "Pass MACD as macd(fast,slow,signal) with fast < slow, for example "
            "macd(12,26,9)."
        )

    if error_code is None:
        return result

    details = {
        "symbol": request.symbol,
        "timeframe": request.timeframe,
    }
    if dst_issue is not None:
        field, issue = dst_issue
        details.update(dict(issue.get("details") or {}))
        details["field"] = field
    elif error_code == "symbol_not_found":
        details["did_you_mean"] = symbol_suggestions_from_gateway(
            gateway,
            request.symbol,
        )
    elif error_code == "data_fetch_candles_stale_data":
        related_live_symbols = find_live_extended_session_symbols(
            gateway,
            request.symbol,
        )
        if related_live_symbols:
            details["related_live_symbols"] = related_live_symbols
            live_symbol = related_live_symbols[0]["symbol"]
            remediation = (
                f"A live extended-session contract is available: call market_ticker "
                f"for {live_symbol}, then use that exact symbol for current data. "
                "Set allow_stale=true only when the regular-session history is "
                "intentionally acceptable."
            )
    if request.start is not None:
        details["start"] = str(request.start)
    if request.end is not None:
        details["end"] = str(request.end)

    payload = build_error_payload(
        message,
        code=error_code,
        operation="data_fetch_candles",
        details=details,
        remediation=remediation,
    )
    for key in ("warnings", "diagnostics"):
        if key in result:
            payload[key] = result[key]
    return payload


def _effective_candle_limit(request: DataFetchCandlesRequest) -> int:
    try:
        return max(1, int(request.limit))
    except Exception:
        return DATA_FETCH_CANDLES_DEFAULT_LIMIT


def _annotate_empty_candle_result(
    result: Dict[str, Any], *, request: DataFetchCandlesRequest
) -> None:
    if (
        result.get("error")
        or not isinstance(result.get("data"), list)
        or result["data"]
    ):
        return
    reported_count = result.get("count", result.get("candles"))
    try:
        if reported_count is not None and int(reported_count) > 0:
            return
    except (TypeError, ValueError):
        pass
    result["empty"] = True
    result.setdefault(
        "empty_reason",
        result.get("range_incomplete_reason") or "no_candles_in_range",
    )
    attach_empty_range_weekend_context(
        result,
        symbol=request.symbol,
        start=request.start,
        end=request.end,
        item="candles",
    )
    _attach_empty_candle_schema(result, request=request)


def _attach_empty_candle_schema(
    result: Dict[str, Any], *, request: DataFetchCandlesRequest
) -> None:
    """Keep empty candle successes schema-compatible with nonempty results."""
    result.setdefault("row_key", "data")
    result.setdefault("timezone", "UTC")
    if str(request.timestamp_format or "iso").strip().lower() == "epoch":
        result.setdefault("timestamp_format", "epoch_seconds")
    else:
        result.setdefault("timestamp_format", "iso_utc")
    if result.get("price_basis") in (None, ""):
        try:
            from ...utils.mt5 import symbol_candle_price_basis_for

            result["price_basis"] = symbol_candle_price_basis_for(request.symbol)
        except Exception:
            pass
    if not (request.start or request.end):
        return
    try:
        limit_value = max(1, int(request.limit))
    except Exception:
        limit_value = DATA_FETCH_CANDLES_DEFAULT_LIMIT
    result.setdefault(
        "pagination",
        {
            "total": 0,
            "returned": 0,
            "offset": 0,
            "limit": limit_value,
            "has_more": False,
            "more_available": 0,
        },
    )


def _latest_numeric_row_value(rows: Any, column: str) -> Optional[float]:
    if not isinstance(rows, list):
        return None
    for row in reversed(rows):
        if not isinstance(row, dict) or column not in row:
            continue
        try:
            value = float(row.get(column))
        except Exception:
            continue
        if np.isfinite(value):
            return value
    return None


def _indicator_family(column: str) -> str:
    name = str(column or "").strip().upper()
    if name.startswith("MACD"):
        return "MACD"
    return name.split("_", 1)[0]


def _indicator_reading(column: str, value: float, *, latest_close: Optional[float]) -> str:
    family = _indicator_family(column)
    if family == "RSI":
        if value >= 70.0:
            state = "overbought"
        elif value <= 30.0:
            state = "oversold"
        else:
            state = "neutral"
        return f"RSI {value:.2f}: {state}; common bands are 30/70."
    if family in {"EMA", "SMA", "WMA", "HMA"}:
        if latest_close is None:
            return f"{family} {value:.5g}: moving-average trend reference."
        side = "above" if latest_close > value else "below" if latest_close < value else "at"
        return f"Close is {side} {family} ({value:.5g}); above often supports bullish trend context."
    if family == "MACD":
        if str(column).upper().startswith("MACDH"):
            side = "positive" if value > 0 else "negative" if value < 0 else "flat"
            return f"MACD histogram {value:.5g}: {side} momentum."
        side = "above zero" if value > 0 else "below zero" if value < 0 else "at zero"
        return f"MACD {value:.5g}: {side}; compare line/signal/histogram together."
    if family == "ATR":
        return f"ATR {value:.5g}: volatility/range estimate in price units."
    if family in {"BBL", "BBM", "BBU"}:
        return f"{family} {value:.5g}: Bollinger Band level; compare close to lower/mid/upper bands."
    return f"{column} {value:.5g}: see indicators_describe for detailed interpretation."


def _attach_indicator_explanations(result: Dict[str, Any]) -> None:
    meta = result.get("meta")
    diagnostics = meta.get("diagnostics") if isinstance(meta, dict) else None
    indicators = diagnostics.get("indicators") if isinstance(diagnostics, dict) else None
    added_columns = indicators.get("added_columns") if isinstance(indicators, dict) else None
    if not isinstance(added_columns, list) or not added_columns:
        return
    rows = result.get("data")
    latest_close = _latest_numeric_row_value(rows, "close")
    explanations: List[Dict[str, Any]] = []
    for column in added_columns:
        column_name = str(column or "").strip()
        if not column_name:
            continue
        value = _latest_numeric_row_value(rows, column_name)
        if value is None:
            continue
        explanations.append(
            {
                "column": column_name,
                "family": _indicator_family(column_name),
                "latest": round(float(value), 6),
                "reading": _indicator_reading(column_name, value, latest_close=latest_close),
            }
        )
    if explanations:
        result["indicator_explanations"] = explanations


def _reconcile_returned_window_completeness(result: Dict[str, Any]) -> None:
    """Describe completeness from the returned rows, not the untrimmed fetch."""
    data = result.get("data")
    if not isinstance(data, list) or not data:
        return
    last = data[-1]
    if not isinstance(last, dict):
        return
    last_state = last.get("bar_state")
    if last_state in (None, ""):
        return
    last_is_complete = str(last_state).strip().lower() not in {
        "forming",
        "incomplete",
        "open",
    }
    if not last_is_complete:
        return
    data_window = result.get("data_window")
    if isinstance(data_window, dict):
        data_window["latest_bar_complete"] = True


def _forming_bar_exclusion_affects_range(
    result: Dict[str, Any],
    *,
    end: Optional[str],
) -> bool:
    """Return whether a skipped forming bar belonged to the requested range."""
    if end in (None, ""):
        return True
    query_applied = result.get("query_applied")
    if not isinstance(query_applied, dict):
        return True
    if query_applied.get("end_filter") != "bar_close":
        return True
    resolved_end = query_applied.get("resolved_end") or end
    data = result.get("data")
    if not isinstance(data, list) or not data:
        return True
    latest = data[-1]
    if not isinstance(latest, dict) or latest.get("time") in (None, ""):
        return True
    try:
        end_dt = datetime.fromisoformat(
            str(resolved_end).strip()
        )
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        latest_value = latest["time"]
        if isinstance(latest_value, (int, float)) and not isinstance(
            latest_value, bool
        ):
            latest_open_epoch = float(latest_value)
        else:
            latest_dt = datetime.fromisoformat(
                str(latest_value).strip()
            )
            if latest_dt.tzinfo is None:
                latest_dt = latest_dt.replace(tzinfo=timezone.utc)
            latest_open_epoch = latest_dt.astimezone(timezone.utc).timestamp()
        timeframe = str(
            result.get("timeframe") or query_applied.get("timeframe") or ""
        )
        latest_close_epoch = bar_close_epoch(latest_open_epoch, timeframe)
    except (TypeError, ValueError, OverflowError, OSError):
        return True
    return latest_close_epoch < end_dt.astimezone(timezone.utc).timestamp() - 1e-6


def _disclose_in_progress_end_clamp(
    query_applied: Dict[str, Any],
    end: Optional[str],
    *,
    now: Optional[datetime] = None,
) -> None:
    """Echo when a date-only current-day --end is clamped to now."""
    if end in (None, "") or not isinstance(query_applied, dict):
        return
    parsed = _parse_end_datetime(str(end))
    if parsed is None:
        return
    now_utc = now or datetime.now(timezone.utc)
    now_naive = now_utc.astimezone(timezone.utc).replace(tzinfo=None)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    if not _is_in_progress_calendar_day_end(str(end), parsed, now_naive):
        return
    query_applied["effective_end"] = format_datetime_utc(now_utc)
    query_applied["end_clamped_to"] = "now"
    query_applied.setdefault("end", str(end))


def _apply_range_limit_cap(  # noqa: C901
    result: Dict[str, Any],
    *,
    limit: int,
    limit_explicit: bool,
    start: Optional[str],
    end: Optional[str],
    request: DataFetchCandlesRequest,
    page_offset: int = 0,
) -> None:
    data = result.get("data")
    if not isinstance(data, list):
        return
    meta = result.get("meta")
    diagnostics = meta.get("diagnostics") if isinstance(meta, dict) else None
    query = diagnostics.get("query") if isinstance(diagnostics, dict) else None
    if not isinstance(query, dict) or query.get("mode") != "range":
        return
    query_applied = result.get("query_applied")
    if not isinstance(query_applied, dict):
        query_applied = {}
        result["query_applied"] = query_applied
    query_applied.setdefault("mode", "range")
    if start not in (None, ""):
        query_applied.setdefault("start", str(start))
    if end not in (None, ""):
        query_applied.setdefault("end", str(end))
    requested_selection = str(getattr(request, "selection", None) or "").strip().lower()
    if requested_selection in {"first_n", "last_n"}:
        start_anchored = requested_selection == "first_n"
    else:
        start_anchored = start not in (None, "")
    query_applied["limit_anchor"] = "start" if start_anchored else "end"
    query_applied["selection"] = "first_n" if start_anchored else "last_n"
    query_applied["order"] = "ascending"
    query_applied["limit_source"] = "user" if limit_explicit else "default"
    _disclose_in_progress_end_clamp(query_applied, end)
    try:
        limit_value = max(1, int(limit))
    except Exception:
        return
    available = len(data)
    provider_bounded = bool(query.get("provider_bounded"))
    if available <= limit_value and not provider_bounded:
        spacing_mismatch = bool(result.get("timeframe_spacing_mismatch"))
        forming_bar_excluded = bool(
            result.get("forming_candle_status") == "skipped"
            and (
                result.get("has_forming_candle") is True
                or int(result.get("incomplete_candles_skipped") or 0) > 0
            )
            and _forming_bar_exclusion_affects_range(result, end=end)
        )
        result["range_complete"] = not spacing_mismatch and not forming_bar_excluded
        if forming_bar_excluded:
            result["range_incomplete_reason"] = "forming_bar_excluded"
            data_window = result.get("data_window")
            if isinstance(data_window, dict):
                data_window["latest_bar_complete"] = False
        elif spacing_mismatch:
            result["range_incomplete_reason"] = "timeframe_spacing_mismatch"
        if request.cursor:
            result["pagination"] = {
                "total": page_offset + available,
                "returned": available,
                "offset": page_offset,
                "limit": limit_value,
                "has_more": False,
                "more_available": 0,
            }
        return

    retained = (
        data[:limit_value]
        if start_anchored and available > limit_value
        else data[-limit_value:]
        if available > limit_value
        else data
    )
    result["data"] = retained
    result["count"] = len(retained)
    result["limit_applied"] = limit_value
    result["truncated"] = True
    result["truncation"] = {
        "reason": "limit",
        "retained": "first" if start_anchored else "last",
    }
    result["range_complete"] = False
    if available >= limit_value:
        result["range_incomplete_reason"] = "limit"
    elif query.get("provider_end_bounded"):
        result["range_incomplete_reason"] = (
            "provider_window_ended_before_requested_end"
        )
    else:
        result["range_incomplete_reason"] = "limit"
    if available > limit_value:
        result["truncation"]["excluded_count"] = (
            None
            if query.get("provider_end_bounded")
            else available - len(retained)
        )
        retained_label = "earliest" if start_anchored else "latest"
        if query.get("provider_end_bounded"):
            warning = (
                f"Returned the {retained_label} {len(retained)} bars because "
                f"limit={limit_value}. Increase limit; the "
                "remaining range size is not known from this fetch window."
            )
        else:
            result["available_count"] = available
            warning = (
                f"Fetched range contained {available} bars; returned the {retained_label} "
                f"{len(retained)} because limit={limit_value}."
            )
        pagination: Dict[str, Any] = {
            "total": None if query.get("provider_end_bounded") else page_offset + available,
            "returned": len(retained),
            "offset": page_offset,
            "limit": limit_value,
            "has_more": True,
            "more_available": (
                None
                if query.get("provider_end_bounded")
                else available - len(retained)
            ),
        }
        if query.get("provider_end_bounded"):
            pagination["total_lower_bound"] = page_offset + len(retained) + 1
        result["pagination"] = pagination
    else:
        result["truncation"]["excluded_count"] = None
        if start_anchored and query.get("provider_end_bounded"):
            warning = (
                f"The start-only range reached limit={limit_value} before its "
                "implied end at the current time; returned the earliest matching "
                "bars. Pass --selection last_n to keep the latest bars since start, "
                "or continue from the timestamp after the final returned bar."
            )
        else:
            warning = (
                "The requested range began before the bounded provider window; "
                f"returned up to the latest {limit_value} bars. Increase limit or "
                "move the range start forward to retrieve an earlier page."
            )
        result["pagination"] = {
            "total": None,
            "total_lower_bound": page_offset + len(retained) + 1,
            "returned": len(retained),
            "offset": page_offset,
            "limit": limit_value,
            "has_more": True,
            "more_available": None,
        }
    if start_anchored and retained:
        next_cursor = _next_candle_cursor(
            request,
            retained[-1],
            offset=page_offset + len(retained),
        )
        if next_cursor is not None:
            result["pagination"]["next_cursor"] = next_cursor
    elif retained:
        result["pagination"]["pagination_supported"] = False
        result["pagination"]["continuation_direction"] = "reverse"
    result.setdefault("warnings", []).append(warning)
    data_window = result.get("data_window")
    if isinstance(data_window, dict) and retained:
        first_row = retained[0]
        last_row = retained[-1]
        if isinstance(first_row, dict) and first_row.get("time") is not None:
            data_window["start"] = first_row["time"]
        if isinstance(last_row, dict) and last_row.get("time") is not None:
            data_window["end"] = last_row["time"]
    candle_counts = result.get("candle_counts")
    if isinstance(candle_counts, dict):
        candle_counts["returned"] = len(retained)
        excluded = candle_counts.get("excluded")
        if not isinstance(excluded, dict):
            excluded = {}
            candle_counts["excluded"] = excluded
        excluded_count = max(0, available - len(retained))
        excluded["limit_truncated"] = excluded_count
        excluded["total"] = int(excluded.get("total") or 0) + excluded_count
    query["limit_applied_to_range"] = True
    query["available_rows_before_limit"] = available
    query["returned_rows_after_limit"] = len(retained)


def _encode_candle_cursor(
    request: DataFetchCandlesRequest,
    *,
    resume_start: str,
    offset: int,
) -> str:
    cursor_payload = {
        "v": 1,
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "start": request.start,
        "end": request.end,
        "selection": "first_n",
        "resume_start": resume_start,
        "offset": int(offset),
    }
    return encode_continuation_cursor(cursor_payload)


def _decode_candle_cursor(
    cursor: str,
    request: DataFetchCandlesRequest,
) -> tuple[str, int]:
    decoded = decode_continuation_cursor(
        cursor,
        invalid_message="cursor is not a valid candle continuation token",
        unsupported_version_message="cursor uses an unsupported candle continuation version",
        expected_versions=1,
    )
    for key, expected in (
        ("symbol", request.symbol),
        ("timeframe", request.timeframe),
        ("start", request.start),
        ("end", request.end),
    ):
        if decoded.get(key) != expected:
            raise ValueError(f"cursor does not match the request {key}")
    if decoded.get("selection") != "first_n":
        raise ValueError("cursor has an invalid candle selection direction")
    resume_start = decoded.get("resume_start")
    if not isinstance(resume_start, str) or not resume_start.strip():
        raise ValueError("cursor has an invalid candle resume boundary")
    try:
        offset = int(decoded.get("offset"))
    except (TypeError, ValueError) as exc:
        raise ValueError("cursor has an invalid candle page offset") from exc
    if offset < 0:
        raise ValueError("cursor has an invalid candle page offset")
    return resume_start, offset


def _next_candle_cursor(
    request: DataFetchCandlesRequest,
    last_row: Any,
    *,
    offset: int,
) -> Optional[str]:
    if not isinstance(last_row, dict):
        return None
    value = last_row.get("time")
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            observed = datetime.fromtimestamp(float(value), timezone.utc)
        else:
            observed = datetime.fromisoformat(str(value).strip())
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            else:
                observed = observed.astimezone(timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None
    resume_start = (observed + timedelta(microseconds=1)).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")
    return _encode_candle_cursor(
        request,
        resume_start=resume_start,
        offset=offset,
    )


def _normalize_range_limit_contract(
    result: Dict[str, Any],
    *,
    effective_limit: int,
    limit_explicit: bool,
) -> None:
    query_applied = result.get("query_applied")
    if not isinstance(query_applied, dict):
        return
    result["limit_explicit"] = bool(limit_explicit)
    if limit_explicit:
        return
    result.pop("requested_limit", None)
    result.pop("candles_requested", None)
    result["default_limit"] = int(effective_limit)
    query_applied.pop("limit", None)
    query_applied["default_limit"] = int(effective_limit)
    candle_counts = result.get("candle_counts")
    if isinstance(candle_counts, dict):
        candle_counts.pop("requested", None)
        excluded = candle_counts.get("excluded")
        if isinstance(excluded, dict):
            excluded.pop("window_or_source_shortfall", None)
            excluded["total"] = sum(
                int(value)
                for key, value in excluded.items()
                if key != "total" and isinstance(value, int) and value > 0
            )
            result["candles_excluded"] = int(excluded["total"])


def _normalize_candle_count_field(result: Dict[str, Any]) -> None:
    candles_value = result.pop("candles", None)
    if "count" not in result and candles_value is not None:
        result["count"] = candles_value
    elif "count" not in result:
        data = result.get("data")
        if isinstance(data, list):
            result["count"] = len(data)
    result.pop("returned_count", None)
    data_window = result.get("data_window")
    if isinstance(data_window, dict):
        data_window.pop("requested_limit", None)
        data_window.pop("returned_count", None)


def _compact_candles_payload(
    result: Dict[str, Any],
    *,
    include_forming_booleans: bool = False,
) -> Dict[str, Any]:
    compact = dict(result)
    compact_time_normalization = result.get("time_normalization")
    public_diagnostics = _public_candle_diagnostics(result)
    try:
        requested_count = int(result["candles_requested"])
        returned_count = int(compact["count"])
    except (KeyError, TypeError, ValueError):
        pass
    else:
        query_applied = result.get("query_applied")
        is_range = (
            isinstance(query_applied, dict)
            and query_applied.get("mode") == "range"
        )
        if is_range:
            compact["range_complete"] = bool(result.get("range_complete", False))
            if requested_count >= 0 and returned_count >= 0:
                compact["limit_reached"] = returned_count >= requested_count
        elif requested_count >= 0 and returned_count >= 0:
            # Compact responses omit the detailed exclusion breakdown, but a
            # caller must still be able to distinguish a complete response
            # from one shortened by the source, filters, or a forming bar.
            compact["limit_satisfied"] = returned_count >= requested_count
    for key in (
        "candles_requested",
        "candle_counts",
        "candles_excluded",
        "hint",
        "incomplete_candles_skipped",
        "spread_note",
        "volume_note",
        "bar_time_convention",
        "meta",
        "raw_time_basis",
        "raw_timestamp_mode",
        "time_normalization",
        "broker_server_tz",
        "broker_utc_offset_seconds",
        "timezone_note",
        "volume_semantics",
        "data_age_anchor",
        "data_age_metric",
        "query_end_gap_anchor",
        "query_end_gap_metric",
        "mt5_time_alignment",
        "bar_spacing",
        "source_bar_spacing",
    ):
        compact.pop(key, None)
    if not bool(compact.get("has_forming_candle")):
        compact.pop("has_forming_candle", None)
        compact["forming_candle_status"] = str(
            compact.get("forming_candle_status") or "none"
        )
        compact.pop("forming_candle_included", None)
        compact.pop("forming_candle_skipped", None)
    elif not include_forming_booleans:
        compact.pop("has_forming_candle", None)
        compact.pop("forming_candle_included", None)
        compact.pop("forming_candle_skipped", None)
    if result.get("forming_candle_status") == "skipped" and result.get("hint"):
        compact["hint"] = result["hint"]
    _attach_candle_timestamp_metadata(compact)
    _collapse_compact_timestamp_metadata(compact)
    if compact_time_normalization not in (None, ""):
        compact["time_normalization"] = compact_time_normalization
    for key in (
        "query_type",
        "freshness",
        "freshness_applicability",
        "data_age_seconds",
        "data_stale",
        "history_policy_ok",
        "usable_for_live_trading",
        "usable_for_live_trading_basis",
        "freshness_policy_relaxed",
        "market_status",
        "market_status_reason",
        "market_status_source",
        "note",
        "query_end_gap_seconds",
        "query_end_gap",
        "indicator_warmup_bars",
        "history_bars_fetched",
        "indicator_columns",
        "indicators_spec",
        "indicator_engine",
    ):
        if key in public_diagnostics:
            compact[key] = public_diagnostics[key]
    if "spread_estimate" in public_diagnostics:
        compact["spread_estimate"] = public_diagnostics["spread_estimate"]
    _attach_denoise_disclosure(compact)
    attach_candle_volume_semantics(compact)
    for key in (
        "tick_volume_event_basis",
        "tick_volume_tape_equivalent",
        "tick_volume_comparison_note",
    ):
        compact.pop(key, None)
    return compact


def _attach_candle_timestamp_metadata(payload: Dict[str, Any]) -> None:
    rows = payload.get("data")
    if not isinstance(rows, list):
        latest = payload.get("latest_candle")
        rows = [latest] if isinstance(latest, dict) else []
    for row in rows:
        if not isinstance(row, dict) or "time" not in row:
            continue
        timestamp_value = row.get("time")
        if isinstance(timestamp_value, bool):
            continue
        representation = _timestamp_representation(timestamp_value)
        if representation is not None:
            timestamp_format, timestamp_mode, timestamp_timezone = representation
            payload["timestamp_format"] = timestamp_format
            payload["timestamp_mode"] = timestamp_mode
            payload["public_timestamp_mode"] = timestamp_mode
            if timestamp_timezone == "client_timezone":
                timestamp_timezone = str(payload.get("timezone") or "").strip()
            if timestamp_timezone:
                payload["timestamp_timezone"] = timestamp_timezone
            else:
                payload.pop("timestamp_timezone", None)
            if timestamp_format == "epoch_seconds":
                payload["timezone"] = "UTC"
            payload.pop("timestamp_format_hint", None)
            return


def _timestamp_representation(value: Any) -> Optional[tuple[str, str, str]]:
    """Describe the serialized timestamp, separately from its MT5 provenance."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return "epoch_seconds", "utc", "UTC"
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return "iso_text", "unspecified", ""
    offset = parsed.utcoffset() if parsed.tzinfo is not None else None
    if offset is None:
        return "iso_without_offset", "unspecified", ""
    if offset == timedelta(0):
        return "iso_utc", "utc", "UTC"
    return "iso_offset", "client_timezone", "client_timezone"


def _collapse_compact_timestamp_metadata(payload: Dict[str, Any]) -> None:
    """Keep only timestamp distinctions not implied by the serialized format."""
    timestamp_format = str(payload.get("timestamp_format") or "").strip().lower()
    implied_mode = {
        "epoch_seconds": "utc",
        "iso_utc": "utc",
        "iso_offset": "client_timezone",
    }.get(timestamp_format)
    if implied_mode is None:
        return

    timestamp_mode = str(payload.get("timestamp_mode") or "").strip().lower()
    public_mode = str(payload.get("public_timestamp_mode") or "").strip().lower()
    if timestamp_mode == implied_mode:
        payload.pop("timestamp_mode", None)
    if public_mode == implied_mode:
        payload.pop("public_timestamp_mode", None)

    time_basis = str(payload.get("time_basis") or "").strip().lower()
    if time_basis == implied_mode:
        payload.pop("time_basis", None)

    timestamp_timezone = str(payload.get("timestamp_timezone") or "").strip()
    timezone_name = str(payload.get("timezone") or "").strip()
    implied_timezone = "UTC" if implied_mode == "utc" else timezone_name
    if (
        timestamp_timezone
        and implied_timezone
        and timestamp_timezone.casefold() == implied_timezone.casefold()
    ):
        payload.pop("timestamp_timezone", None)


def _rename_nested_raw_timestamp_mode(payload: Dict[str, Any]) -> None:
    """Keep nested diagnostics from reusing the public timestamp_mode key."""
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return
    diagnostics = meta.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return
    time_norm = diagnostics.get("time_normalization")
    if not isinstance(time_norm, dict):
        return
    raw = time_norm.get("timestamp_mode")
    if raw in (None, ""):
        return
    time_norm["raw_timestamp_mode"] = raw
    time_norm.pop("timestamp_mode", None)


def _normalize_public_candle_timestamp_mode(
    payload: Dict[str, Any],
    *,
    include_raw: bool,
) -> None:
    """Name the clock used by emitted timestamps, not the raw MT5 epoch axis."""
    raw_mode = str(payload.get("timestamp_mode") or "").strip()
    if include_raw and raw_mode:
        payload["raw_timestamp_mode"] = raw_mode
    else:
        payload.pop("raw_timestamp_mode", None)
    _rename_nested_raw_timestamp_mode(payload)
    _attach_candle_timestamp_metadata(payload)
    if payload.get("public_timestamp_mode") is not None:
        return
    time_basis = str(payload.get("time_basis") or "").strip().lower()
    if raw_mode and time_basis == "utc":
        payload["timestamp_mode"] = "utc"
        payload["public_timestamp_mode"] = "utc"


def _attach_denoise_disclosure(payload: Dict[str, Any]) -> None:
    denoise_info = payload.get("denoise")
    applications = denoise_info.get("applications") if isinstance(denoise_info, dict) else None
    if not isinstance(applications, list) or not applications:
        return

    methods: List[str] = []
    overwritten: List[str] = []
    causalities: List[str] = []
    for app in applications:
        if not isinstance(app, dict):
            continue
        added_columns = app.get("added_columns")
        overwritten_columns = app.get("overwrote_columns")
        added = added_columns if isinstance(added_columns, list) else []
        overwritten_for_app = (
            overwritten_columns if isinstance(overwritten_columns, list) else []
        )
        if not added and not overwritten_for_app:
            continue
        method = str(app.get("method") or "").strip().lower()
        if method and method != "none" and method not in methods:
            methods.append(method)
        causality = str(app.get("causality") or "").strip().lower()
        if causality and causality not in causalities:
            causalities.append(causality)
        if bool(app.get("keep_original")):
            continue
        for column in overwritten_for_app:
            column = str(column).strip()
            if column and column not in overwritten:
                overwritten.append(column)

    if not methods and not overwritten:
        return
    payload["denoise_applied"] = True
    payload["denoise_status"] = "applied"
    denoise_columns: List[str] = []
    for app in applications:
        if not isinstance(app, dict):
            continue
        added_columns = app.get("added_columns")
        if not isinstance(added_columns, list):
            continue
        suffix = str(app.get("suffix") or "_dn")
        for column in added_columns:
            name = str(column or "").strip()
            if not name:
                continue
            base = name[: -len(suffix)] if suffix and name.endswith(suffix) else name
            if base and base not in denoise_columns:
                denoise_columns.append(base)
    if denoise_columns:
        payload["denoise_columns"] = denoise_columns
    if methods:
        payload["denoise_method"] = methods[0] if len(methods) == 1 else methods
    if overwritten:
        payload["denoise_overwrote_columns"] = overwritten
        if "close" in overwritten and methods:
            payload["price_column"] = f"close ({methods[0]}-smoothed)"
            payload["price_is_synthetic"] = True
    if "zero_phase" in causalities:
        payload["denoise_live_safe"] = False
        payload.setdefault("warnings", []).append(
            "Zero-phase denoise uses future observations and is not usable for live trading."
        )
    elif causalities:
        payload["denoise_live_safe"] = True
    payload.pop("denoise", None)


def _slim_projected_candles_payload(payload: Dict[str, Any]) -> None:
    if not bool(payload.get("ohlcv_filter_applied")):
        return
    rows = payload.get("data")
    projected_fields: set[str] = set()
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                projected_fields.update(str(key) for key in row if str(key) != "time")
    payload.pop("ohlcv_filter_applied", None)
    if not projected_fields or projected_fields.isdisjoint({"tick_volume", "volume"}):
        for key in (
            "volume_type",
            "volume_unit",
            "volume_semantics",
            "tick_volume_event_basis",
            "tick_volume_tape_equivalent",
            "tick_volume_comparison_note",
        ):
            payload.pop(key, None)
    if not projected_fields or "real_volume" not in projected_fields:
        for key in ("real_volume_type", "real_volume_unit"):
            payload.pop(key, None)
    if projected_fields.isdisjoint({"spread", "spread_points"}):
        payload.pop("spread_estimate", None)
        payload.pop("spread_unavailable", None)
    _filter_candle_units_to_projected_fields(payload, projected_fields)
    if not bool(payload.get("forming_candle_included")):
        payload.pop("has_forming_candle", None)
        payload.pop("forming_candle_included", None)
        payload.pop("forming_candle_skipped", None)


def _filter_candle_units_to_projected_fields(
    payload: Dict[str, Any],
    projected_fields: set[str],
) -> None:
    units = payload.get("units")
    if not isinstance(units, dict):
        return
    allowed_fields = set(projected_fields)
    if "volume" in allowed_fields:
        allowed_fields.update({"tick_volume", "real_volume"})
    filtered_units = {
        key: value
        for key, value in units.items()
        if key in allowed_fields
    }
    if filtered_units:
        payload["units"] = filtered_units
    else:
        payload.pop("units", None)


def _standard_candles_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    standard = _compact_candles_payload(
        result,
        include_forming_booleans=True,
    )
    public_diagnostics = _public_candle_diagnostics(result)
    for key in (
        "query_type",
        "freshness",
        "freshness_applicability",
        "data_stale",
        "history_policy_ok",
        "usable_for_live_trading",
        "usable_for_live_trading_basis",
        "data_age_seconds",
        "data_age_anchor",
        "data_age_metric",
        "freshness_policy_relaxed",
        "market_status",
        "market_status_reason",
        "market_status_source",
        "note",
        "query_end_gap_seconds",
        "query_end_gap",
        "query_end_gap_anchor",
        "query_end_gap_metric",
        "mt5_time_alignment",
        "stale_warning",
        "spread_estimate",
        "indicator_warmup_bars",
        "history_bars_fetched",
        "indicator_columns",
        "indicators_spec",
        "indicator_engine",
    ):
        if key in public_diagnostics:
            standard[key] = public_diagnostics[key]
    return standard


def _attach_candle_machine_freshness(payload: Dict[str, Any]) -> None:
    public_diagnostics = _public_candle_diagnostics(payload)
    for key in (
        "query_type",
        "freshness_applicability",
        "data_age_seconds",
        "data_stale",
        "history_policy_ok",
        "usable_for_live_trading",
        "usable_for_live_trading_basis",
        "freshness_policy_relaxed",
        "query_end_gap_seconds",
        "query_end_gap",
    ):
        if key in public_diagnostics:
            payload.setdefault(key, public_diagnostics[key])


def _summary_candles_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    summary = _compact_candles_payload(
        result,
        include_forming_booleans=True,
    )
    for key, value in _public_candle_diagnostics(result).items():
        summary[key] = value
    summary["output"] = "summary"
    rows = result.get("data")
    if isinstance(rows, list) and rows:
        latest = rows[-1]
        if isinstance(latest, dict):
            summary["latest_candle"] = dict(latest)
        statistics = _candle_summary_statistics(rows)
        if statistics:
            summary["summary_statistics"] = statistics
        _attach_candle_timestamp_metadata(summary)
    summary.pop("data", None)
    summary.pop("row_key", None)
    summary.pop("session_gaps", None)
    for key in (
        "candles_requested",
        "candles_excluded",
        "candle_counts",
        "incomplete_candles_skipped",
    ):
        value = result.get(key)
        if value not in (None, 0, [], {}):
            summary[key] = value
    return summary


def _finite_candle_values(rows: List[Any], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        if not isinstance(row, dict) or key not in row:
            continue
        try:
            value = float(row.get(key))
        except Exception:
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _round_candle_stat(value: float) -> float:
    rounded = round(float(value), 6)
    return 0.0 if rounded == -0.0 else rounded


def _candle_summary_statistics(rows: List[Any]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for field in ("open", "high", "low", "close"):
        values = _finite_candle_values(rows, field)
        if not values:
            continue
        stats[field] = {
            "min": _round_candle_stat(min(values)),
            "max": _round_candle_stat(max(values)),
            "mean": _round_candle_stat(float(np.mean(values))),
        }

    close_values = _finite_candle_values(rows, "close")
    if len(close_values) >= 2:
        first_close = close_values[0]
        last_close = close_values[-1]
        change = last_close - first_close
        close_stats = stats.setdefault("close", {})
        close_stats["change"] = _round_candle_stat(change)
        if first_close:
            close_stats["change_pct"] = _round_candle_stat((change / first_close) * 100.0)

    high_values = _finite_candle_values(rows, "high")
    low_values = _finite_candle_values(rows, "low")
    if high_values and low_values:
        paired_ranges: List[float] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                high = float(row.get("high"))
                low = float(row.get("low"))
            except Exception:
                continue
            if np.isfinite(high) and np.isfinite(low):
                paired_ranges.append(high - low)
        if paired_ranges:
            stats["range"] = {
                "min": _round_candle_stat(min(paired_ranges)),
                "max": _round_candle_stat(max(paired_ranges)),
                "mean": _round_candle_stat(float(np.mean(paired_ranges))),
            }

    for field in ("tick_volume", "real_volume", "volume"):
        values = _finite_candle_values(rows, field)
        if values:
            stats[field] = {
                "min": _round_candle_stat(min(values)),
                "max": _round_candle_stat(max(values)),
                "mean": _round_candle_stat(float(np.mean(values))),
                "sum": _round_candle_stat(float(np.sum(values))),
            }
    return stats


def _is_paginated_historical_candle_page(
    result: Dict[str, Any],
    query_mode: Any,
) -> bool:
    """True for a start-anchored page that is not the live tail."""
    if query_mode != "range":
        return False
    query_applied = result.get("query_applied")
    if not isinstance(query_applied, dict):
        return False
    if query_applied.get("selection") != "first_n":
        return False
    pagination = result.get("pagination")
    has_more = isinstance(pagination, dict) and pagination.get("has_more") is True
    return has_more or bool(result.get("truncated"))


def _public_candle_diagnostics(result: Dict[str, Any]) -> Dict[str, Any]:  # noqa: C901
    meta = result.get("meta")
    diagnostics = meta.get("diagnostics") if isinstance(meta, dict) else None
    if not isinstance(diagnostics, dict):
        return {}

    public: Dict[str, Any] = {}
    query = diagnostics.get("query")
    query_mode = query.get("mode") if isinstance(query, dict) else None
    if query_mode == "range":
        public["query_type"] = "historical"
    elif query_mode == "latest":
        public["query_type"] = "latest"
    if isinstance(query, dict) and query.get("latency_ms") is not None:
        public["latency_ms"] = query["latency_ms"]
    indicators = diagnostics.get("indicators")
    if isinstance(indicators, dict) and indicators.get("requested") is True:
        if isinstance(query, dict) and (
            query.get("indicator_warmup_bars") is not None
            or query.get("warmup_bars") is not None
        ):
            public["indicator_warmup_bars"] = int(
                query.get("indicator_warmup_bars", query.get("warmup_bars"))
            )
        if isinstance(query, dict) and query.get("raw_bars_fetched") is not None:
            public["history_bars_fetched"] = int(query["raw_bars_fetched"])
        added_columns = indicators.get("added_columns")
        if isinstance(added_columns, list) and added_columns:
            public["indicator_columns"] = [
                str(column).strip()
                for column in added_columns
                if str(column).strip()
            ]
        spec_text = str(indicators.get("spec") or "").strip()
        if spec_text:
            public["indicators_spec"] = spec_text
        engine = result.get("indicator_engine")
        if isinstance(engine, dict) and engine:
            public["indicator_engine"] = engine

    spread_estimate = diagnostics.get("spread_estimate")
    if isinstance(spread_estimate, dict):
        value = spread_estimate.get("estimated_mean")
        source = spread_estimate.get("source")
        unit = spread_estimate.get("unit")
        if value is not None or source:
            public_estimate: Dict[str, Any] = {}
            if value is not None:
                public_estimate["value"] = value
            if source:
                public_estimate["source"] = source
            if unit:
                public_estimate["unit"] = unit
            public["spread_estimate"] = public_estimate

    freshness = diagnostics.get("freshness")
    if isinstance(freshness, dict):
        public["freshness_basis"] = "bar_policy"
        historical_page = _is_paginated_historical_candle_page(result, query_mode)
        if historical_page:
            public["freshness_applicability"] = "historical_page"
        within_policy = freshness.get("last_bar_within_policy_window")
        if (
            not historical_page
            and freshness.get("last_bar_within_policy_window") is not None
        ):
            public["last_bar_within_policy_window"] = bool(
                freshness["last_bar_within_policy_window"]
            )
        if "freshness_policy_relaxed" in freshness:
            public["freshness_policy_relaxed"] = normalize_policy_relaxed(
                freshness.get("freshness_policy_relaxed")
            )
        query_gap_value = freshness.get("query_end_gap_seconds")
        if (
            query_gap_value is None
            and query_mode == "range"
            and freshness.get("data_freshness_anchor")
            != FRESHNESS_ANCHOR_WALL_CLOCK
        ):
            query_gap_value = freshness.get("data_freshness_seconds")
        if query_mode == "range" and query_gap_value is not None:
            try:
                seconds = max(0.0, float(query_gap_value))
            except Exception:
                seconds = query_gap_value
            public["query_end_gap_seconds"] = seconds
            public["query_end_gap_anchor"] = (
                freshness.get("query_end_gap_anchor")
                or FRESHNESS_ANCHOR_QUERY_EXPECTED_END
            )
            public["query_end_gap_metric"] = (
                freshness.get("query_end_gap_metric")
                or FRESHNESS_METRIC_REQUESTED_RANGE_END_GAP
            )
            gap_text = _format_age_seconds(seconds)
            if gap_text is not None:
                public["query_end_gap"] = gap_text
        publish_data_age = bool(
            query_mode != "range"
            or freshness.get("data_freshness_anchor") == FRESHNESS_ANCHOR_WALL_CLOCK
        )
        if publish_data_age and freshness.get("data_freshness_seconds") is not None:
            try:
                seconds = max(0.0, float(freshness["data_freshness_seconds"]))
            except Exception:
                seconds = freshness["data_freshness_seconds"]
            public.setdefault("data_age_seconds", seconds)
            public["data_age_anchor"] = (
                freshness.get("data_freshness_anchor")
                or FRESHNESS_ANCHOR_WALL_CLOCK
            )
            public["data_age_metric"] = (
                freshness.get("data_freshness_metric")
                or FRESHNESS_METRIC_LAST_COMPLETED_BAR_AGE
            )
            age_text = _format_age_seconds(seconds)
            if age_text is not None:
                public["data_age"] = age_text
            if not historical_page:
                relaxed_policy = normalize_policy_relaxed(
                    freshness.get("freshness_policy_relaxed")
                )
                if relaxed_policy:
                    public["market_status"] = (
                        freshness.get("market_session_status") or "closed_or_idle"
                    )
                    if freshness.get("market_session_reason"):
                        public["market_status_reason"] = freshness[
                            "market_session_reason"
                        ]
                    if freshness.get("market_session_source"):
                        public["market_status_source"] = freshness[
                            "market_session_source"
                        ]
                    note = freshness.get("freshness_note")
                    if note:
                        public["note"] = note
                stale = (
                    within_policy is not None
                    and not bool(within_policy)
                )
                history_policy_ok = not stale and not relaxed_policy
                public["history_policy_ok"] = history_policy_ok
                public["data_stale"] = stale
                freshness_label = format_freshness_label(
                    data_stale=stale,
                    market_status=public.get("market_status"),
                    market_status_reason=public.get("market_status_reason"),
                    age_seconds=seconds,
                    item="bar",
                )
                if freshness_label:
                    public["freshness"] = freshness_label
                if stale:
                    public["stale_warning"] = (
                        "Latest completed candle is outside the freshness policy window; "
                        "market may be closed or broker data may be stale."
                    )
    mt5_time_alignment = diagnostics.get("mt5_time_alignment")
    if isinstance(mt5_time_alignment, dict):
        status = str(mt5_time_alignment.get("status") or "").strip().lower()
        if status and status != "ok":
            public["mt5_time_alignment"] = {
                key: mt5_time_alignment.get(key)
                for key in (
                    "status",
                    "reason",
                    "warning",
                    "probe_timeframe",
                    "timestamp_contract",
                    "tick_age_seconds",
                    "current_bar_delta_seconds",
                )
                if mt5_time_alignment.get(key) is not None
            }
    return public


def _drop_redundant_session_gap_warnings(result: Dict[str, Any]) -> None:
    if not result.get("session_gaps"):
        return
    warnings = result.get("warnings")
    if not isinstance(warnings, list):
        return
    filtered = [
        warning
        for warning in warnings
        if not (
            isinstance(warning, str)
            and (
                warning.startswith("Detected session gaps larger than expected bar spacing")
                or warning.startswith("Example gap:")
            )
        )
    ]
    if filtered:
        result["warnings"] = filtered
    else:
        result.pop("warnings", None)


def _prune_zero_candle_exclusions(result: Dict[str, Any]) -> None:
    candle_counts = result.get("candle_counts")
    if not isinstance(candle_counts, dict):
        return
    excluded = candle_counts.get("excluded")
    if not isinstance(excluded, dict):
        return
    candle_counts["excluded"] = {
        key: value
        for key, value in excluded.items()
        if key == "total" or value not in (None, 0)
    }


def _freeze_tick_bound(value: Any, *, end: bool) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = _parse_end_datetime(text) if end else _parse_start_datetime(text)
    if parsed is None:
        return None
    return format_datetime_utc(parsed, timespec="microseconds")


def _encode_tick_cursor(
    request: DataFetchTicksRequest,
    *,
    selection: str,
    offset: int,
    resolved_start: Optional[str] = None,
    resolved_end: Optional[str] = None,
) -> str:
    cursor_payload = {
        "v": 2,
        "symbol": request.symbol,
        "start": request.start,
        "end": request.end,
        "resolved_start": resolved_start,
        "resolved_end": resolved_end,
        "selection": selection,
        "offset": int(offset),
    }
    return encode_continuation_cursor(cursor_payload)


def _decode_tick_cursor(
    cursor: str,
    request: DataFetchTicksRequest,
) -> tuple[str, int, Optional[str], Optional[str]]:
    decoded = decode_continuation_cursor(
        cursor,
        invalid_message="cursor is not a valid tick continuation token",
        unsupported_version_message="cursor uses an unsupported tick continuation version",
        expected_versions={1, 2},
    )
    for key, expected in (
        ("symbol", request.symbol),
        ("start", request.start),
        ("end", request.end),
    ):
        if decoded.get(key) != expected:
            raise ValueError(f"cursor does not match the request {key}")
    selection = str(decoded.get("selection") or "")
    if selection not in {"first_n", "last_n"}:
        raise ValueError("cursor has an invalid tick selection direction")
    offset = decoded.get("offset")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("cursor has an invalid tick page offset")
    resolved_start = decoded.get("resolved_start")
    resolved_end = decoded.get("resolved_end")
    if resolved_start is not None:
        resolved_start = str(resolved_start)
    if resolved_end is not None:
        resolved_end = str(resolved_end)
    return selection, offset, resolved_start, resolved_end


def _run_data_fetch_ticks_impl(
    *,
    request: DataFetchTicksRequest,
    gateway: Any,
    fetch_ticks_impl: Any,
    effective_limit: Optional[int] = None,
) -> Dict[str, Any]:
    future_bound = _future_bound(request)
    if future_bound is not None:
        field, value = future_bound
        details: Dict[str, Any] = {
            "symbol": request.symbol,
            "timezone": "UTC",
        }
        if request.start is not None:
            details["start"] = str(request.start)
        if request.end is not None:
            details["end"] = str(request.end)
        return build_error_payload(
            f"{field} datetime {value} is in the future; historical tick ranges must have elapsed.",
            code="future_date_range",
            operation="data_fetch_ticks",
            details=details,
            remediation="Use start and end timestamps at or before the current time.",
        )
    requested_selection = str(
        getattr(request, "selection", None) or ""
    ).strip().lower()
    if (
        request.start in (None, "")
        and request.end not in (None, "")
        and requested_selection == "first_n"
    ):
        return build_error_payload(
            "selection=first_n is not supported for end-only tick queries.",
            code="selection_unsupported_for_end_only",
            operation="data_fetch_ticks",
            details={"selection": "first_n", "end": str(request.end)},
            remediation=(
                "Omit selection or pass last_n, or supply start to page from "
                "the beginning of a bounded window."
            ),
        )
    connection_error = _ensure_gateway_connection(gateway)
    if connection_error is not None:
        return connection_error
    applied_limit = (
        effective_limit if effective_limit is not None else request.limit
    )
    limit_explicit = "limit" in getattr(request, "model_fields_set", set())
    if requested_selection in {"first_n", "last_n"}:
        range_selection = requested_selection
    else:
        range_selection = (
            "last_n"
            if request.end
            else "first_n"
        )
    page_offset = 0
    resolved_start = _freeze_tick_bound(request.start, end=False)
    resolved_end = _freeze_tick_bound(request.end, end=True)
    if request.cursor:
        if not request.start or not request.end:
            return build_error_payload(
                "cursor requires both start and end for a bounded tick query.",
                code="data_fetch_ticks_invalid_cursor",
                operation="data_fetch_ticks",
                remediation="Reuse the cursor with the original start and end values.",
            )
        try:
            range_selection, page_offset, cursor_start, cursor_end = _decode_tick_cursor(
                request.cursor,
                request,
            )
        except ValueError as exc:
            return build_error_payload(
                str(exc),
                code="data_fetch_ticks_invalid_cursor",
                operation="data_fetch_ticks",
                remediation=(
                    "Use next_cursor from the preceding page without changing "
                    "symbol, start, or end."
                ),
            )
        if cursor_start:
            resolved_start = cursor_start
        if cursor_end:
            resolved_end = cursor_end
    result = fetch_ticks_impl(
        symbol=request.symbol,
        limit=applied_limit,
        start=resolved_start or request.start,
        end=resolved_end or request.end,
        simplify=request.simplify,
        time_as_epoch=not _iso_timestamp_requested(request.timestamp_format),
        force_utc=_force_utc_timestamps(request.timestamp_format),
        format=_TICK_DETAIL_FORMATS.get(request.detail, "summary"),
        range_selection=range_selection,
        page_offset=page_offset,
        probe_more=bool(request.start and request.end),
    )
    result = _normalize_tick_query_error(
        result,
        request=request,
        gateway=gateway,
    )
    if isinstance(result, dict) and result.get("empty") is True:
        attach_empty_range_weekend_context(
            result,
            symbol=request.symbol,
            start=str(request.start) if request.start is not None else None,
            end=str(request.end) if request.end is not None else None,
            item="ticks",
        )
    if isinstance(result, dict):
        warnings = result.get("warnings")
        if isinstance(warnings, list):
            result["warnings"] = list(dict.fromkeys(warnings))
    if str(request.detail or "compact").strip().lower() == "compact":
        result = _compact_tick_rows_payload(result)
    if isinstance(result, dict) and not result.get("error"):
        _attach_tick_timestamp_metadata(
            result,
            requested_format=str(request.timestamp_format),
        )
        if str(request.detail or "compact").strip().lower() != "full":
            _collapse_compact_timestamp_metadata(result)
    _attach_tick_freshness_contract(result)
    _attach_tick_pagination(
        result,
        request=request,
        requested_limit=applied_limit,
        limit_explicit=limit_explicit,
        selection=range_selection,
        page_offset=page_offset,
        resolved_start=resolved_start,
        resolved_end=resolved_end,
    )
    return attach_mt5_source(result, gateway=gateway)


def _attach_tick_timestamp_metadata(
    payload: Dict[str, Any],
    *,
    requested_format: str,
) -> None:
    rows = payload.get("data")
    timestamp_value: Any = None
    if isinstance(rows, list):
        timestamp_value = next(
            (
                row.get("time")
                for row in rows
                if isinstance(row, dict) and row.get("time") not in (None, "")
            ),
            None,
        )
    representation = _timestamp_representation(timestamp_value)
    if representation is None:
        if _iso_timestamp_requested(requested_format):
            if _force_utc_timestamps(requested_format):
                representation = ("iso_utc", "utc", "UTC")
            else:
                timezone_label = str(payload.get("timezone") or "UTC").strip()
                representation = (
                    ("iso_utc", "utc", "UTC")
                    if timezone_label.upper() == "UTC"
                    else ("iso_offset", "client_timezone", "client_timezone")
                )
        else:
            representation = ("epoch_seconds", "utc", "UTC")
    timestamp_format, timestamp_mode, timestamp_timezone = representation
    payload["timestamp_format"] = timestamp_format
    payload["timestamp_mode"] = timestamp_mode
    payload["public_timestamp_mode"] = timestamp_mode
    if timestamp_timezone == "client_timezone":
        timestamp_timezone = str(payload.get("timezone") or "").strip()
    if timestamp_timezone:
        payload["timestamp_timezone"] = timestamp_timezone
    else:
        payload.pop("timestamp_timezone", None)
    if timestamp_format == "epoch_seconds":
        payload["timezone"] = "UTC"


def _normalize_tick_query_error(
    result: Any,
    *,
    request: DataFetchTicksRequest,
    gateway: Any = None,
) -> Any:
    if not isinstance(result, dict) or not result.get("error"):
        return result
    if result.get("error_code"):
        return result

    message = str(result["error"])
    normalized = message.lower()
    error_code = "data_fetch_ticks_provider_failure"
    remediation = "Check the MT5 connection and broker data feed, then retry."
    dst_issue = next(
        (
            (field, issue)
            for field in ("start", "end")
            if (value := getattr(request, field, None)) is not None
            if (issue := _iana_timezone_datetime_issue(str(value))) is not None
        ),
        None,
    )

    if dst_issue is not None:
        _, issue = dst_issue
        error_code = str(issue["error_code"])
        message = str(issue["error"])
        remediation = str(issue["remediation"])
    elif (
        ("not found" in normalized and "symbol" in normalized)
        or "failed to select symbol" in normalized
        or "unknown symbol" in normalized
    ):
        error_code = "symbol_not_found"
        message = f"Symbol '{request.symbol}' was not found in MT5."
        remediation = (
            "Use symbols_list to find the broker's exact symbol name, including "
            "any suffix or alias."
        )
    elif "could not parse" in normalized and "date" in normalized:
        error_code = "data_fetch_ticks_invalid_date"
        remediation = "Use an ISO-8601 timestamp such as 2026-07-16T12:00:00Z."
    elif "start must be before or equal to end" in normalized:
        error_code = "data_fetch_ticks_invalid_date_range"
        remediation = "Set start to a timestamp earlier than or equal to end."
    elif "start datetime" in normalized and "in the future" in normalized:
        error_code = "future_date_range"
        remediation = "Use a start timestamp at or before the current time."
    elif "no tick data" in normalized:
        if _tick_request_is_future_only(request):
            error_code = "future_date_range"
            message = (
                f"start datetime {request.start or request.end} is in the future; "
                "no historical tick data is available for future dates."
            )
            remediation = "Use a start and end timestamp at or before the current time."
        elif "was selected" in normalized:
            error_code = "data_fetch_ticks_not_ready"
            remediation = (
                "Ensure the symbol is selected and the broker is streaming ticks, "
                "then retry. An empty range is returned only when MT5 yields no rows."
            )
        elif "failed to get" in normalized:
            error_code = "data_fetch_ticks_provider_failure"
            remediation = "Check the MT5 connection and broker data feed, then retry."
        else:
            empty: Dict[str, Any] = {
                "success": True,
                "symbol": request.symbol,
                "count": 0,
                "tick_count": 0,
                "tick_count_event_basis": "mt5_copy_ticks_all_records",
                "quote_update_count": 0,
                "quote_update_count_event_basis": "records_with_bid_or_ask_update_flag",
                "bid_update_count": 0,
                "ask_update_count": 0,
                "data": [],
                "empty": True,
                "empty_reason": "no_ticks_in_range",
                "timezone": "UTC",
            }
            if request.start is not None:
                empty["start"] = str(request.start)
            if request.end is not None:
                empty["end"] = str(request.end)
            return attach_empty_range_weekend_context(
                empty,
                symbol=request.symbol,
                start=str(request.start) if request.start is not None else None,
                end=str(request.end) if request.end is not None else None,
                item="ticks",
            )

    details: Dict[str, Any] = {
        "symbol": request.symbol,
        "timezone": "UTC",
    }
    if dst_issue is not None:
        field, issue = dst_issue
        details.update(dict(issue.get("details") or {}))
        details["field"] = field
    elif error_code == "symbol_not_found":
        details["did_you_mean"] = symbol_suggestions_from_gateway(
            gateway,
            request.symbol,
        )
    if request.start is not None:
        details["start"] = str(request.start)
    if request.end is not None:
        details["end"] = str(request.end)
    return build_error_payload(
        message,
        code=error_code,
        operation="data_fetch_ticks",
        details=details,
        remediation=remediation,
        related_tools=["symbols_list"] if error_code == "symbol_not_found" else None,
    )


def _future_bound(request: Any) -> Optional[tuple[str, str]]:
    now_utc = datetime.now(timezone.utc)
    now_naive = now_utc.replace(tzinfo=None)
    for field in ("start", "end"):
        value = getattr(request, field, None)
        if value in (None, ""):
            continue
        parsed = (
            _parse_start_datetime(str(value))
            if field == "start"
            else _parse_end_datetime(str(value))
        )
        if parsed is None:
            continue
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        if parsed <= now_naive:
            continue
        if field == "end" and _is_in_progress_calendar_day_end(
            str(value), parsed, now_naive
        ):
            continue
        return field, str(value)
    return None


def _tick_request_is_future_only(request: DataFetchTicksRequest) -> bool:
    return _future_bound(request) is not None


def _attach_tick_pagination(
    payload: Any,
    *,
    request: DataFetchTicksRequest,
    requested_limit: int,
    limit_explicit: bool = True,
    selection: str,
    page_offset: int,
    resolved_start: Optional[str] = None,
    resolved_end: Optional[str] = None,
) -> None:
    """Attach evidence-based pagination for bounded tick queries."""
    if not isinstance(payload, dict) or payload.get("error"):
        return
    page_info = payload.pop("_tick_page", None)
    if not isinstance(page_info, dict):
        page_info = {}
    source_returned = page_info.get(
        "source_returned",
        payload.get("tick_count", payload.get("count")),
    )
    if not isinstance(source_returned, int):
        source_returned = 0
    try:
        limit_value = int(requested_limit)
    except (TypeError, ValueError):
        return
    payload["requested_limit"] = limit_value
    limit_reached = bool(source_returned >= limit_value)
    payload["limit_reached"] = limit_reached
    if request.start or request.end:
        query_applied = payload.get("query_applied")
        if not isinstance(query_applied, dict):
            query_applied = {}
            payload["query_applied"] = query_applied
        query_applied["limit"] = limit_value
        query_applied["limit_source"] = "user" if limit_explicit else "default"
        if resolved_start:
            query_applied["resolved_start"] = resolved_start
        if resolved_end:
            query_applied["resolved_end"] = resolved_end
        if not limit_explicit:
            query_applied["default_limit"] = limit_value
            payload["default_limit"] = limit_value
    if not (request.start and request.end):
        return

    data = payload.get("data")
    returned = len(data) if isinstance(data, list) else payload.get("count", 0)
    if not isinstance(returned, int):
        returned = 0
    has_more = page_info.get("has_more") is True
    offset = page_info.get("offset", page_offset)
    if not isinstance(offset, int) or offset < 0:
        offset = page_offset
    pagination: Dict[str, Any] = {
        "total": None if has_more else offset + source_returned,
        "returned": returned,
        "offset": offset,
        "limit": limit_value,
        "has_more": has_more,
        "more_available": None,
        "selection": selection,
    }
    if source_returned != returned:
        pagination["source_events_returned"] = source_returned
    if has_more:
        pagination["total_lower_bound"] = offset + source_returned + 1
        pagination["next_cursor"] = _encode_tick_cursor(
            request,
            selection=selection,
            offset=offset + source_returned,
            resolved_start=resolved_start,
            resolved_end=resolved_end,
        )
        payload["truncated"] = True
        data_window = payload.get("data_window")
        if isinstance(data_window, dict):
            data_window["truncated"] = True
    payload["pagination"] = pagination


def _attach_tick_freshness_contract(payload: Any) -> None:
    if not isinstance(payload, dict) or payload.get("error"):
        return
    if payload.get("data_age_seconds") is None:
        return
    payload.setdefault("data_age_anchor", FRESHNESS_ANCHOR_WALL_CLOCK)
    payload.setdefault("data_age_metric", FRESHNESS_METRIC_LAST_TICK_AGE)


def _compact_tick_rows_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error"):
        return payload
    compact = {
        key: payload[key]
        for key in _COMPACT_TICK_TOP_LEVEL_FIELDS
        if key in payload
        and (key == "data" or payload[key] not in (None, "", [], {}))
    }
    rows = compact.get("data")
    if isinstance(rows, list):
        compact_rows: List[Any] = []
        last_spread: Optional[float] = None
        for row in rows:
            compact_row, row_spread = _compact_tick_row(
                row,
                last_spread=last_spread,
            )
            if row_spread is not None:
                last_spread = row_spread
            compact_rows.append(compact_row)
        compact["data"] = compact_rows
        compact["count"] = len(compact["data"])
        if compact.get("tick_count") == compact["count"]:
            compact.pop("tick_count", None)
        units = compact.get("units")
        present_fields = {
            key
            for row in compact["data"]
            if isinstance(row, dict)
            for key in row.keys()
        }
        compact_units = (
            {
                key: value
                for key, value in units.items()
                if key in present_fields
            }
            if isinstance(units, dict)
            else {}
        )
        for field in ("bid", "ask", "mid", "spread"):
            if any(isinstance(row, dict) and field in row for row in compact["data"]):
                compact_units.setdefault(field, "absolute_price")
        if compact_units:
            compact["units"] = compact_units
        compact["volume_fields"] = [
            field
            for field in ("volume", "volume_real")
            if field in present_fields
        ]
    quote_completeness = _tick_quote_completeness_pct(payload)
    if quote_completeness is not None:
        compact["quote_completeness_pct"] = quote_completeness
    valid_spread = _tick_valid_spread_sample_pct(payload)
    if valid_spread is not None:
        compact["valid_spread_sample_pct"] = valid_spread
        compact["spread_quality_basis"] = "valid_two_sided_quote_snapshots"
    quality = _compact_tick_quality(payload)
    if quality:
        compact["quality"] = quality
    invalid_spread_rows = 0
    compact_rows = compact.get("data")
    if isinstance(compact_rows, list):
        invalid_spread_rows = sum(
            1
            for row in compact_rows
            if isinstance(row, dict) and row.get("spread_snapshot_valid") is False
        )
    if invalid_spread_rows:
        warning = (
            "Tick spread is unavailable for "
            f"{invalid_spread_rows} of {len(compact_rows)} row(s); those "
            "spread values are null (one-sided update) and must not be treated "
            "as zero cost."
        )
        existing = compact.get("warnings")
        if isinstance(existing, list):
            if warning not in existing:
                compact["warnings"] = [*existing, warning]
        elif existing:
            compact["warnings"] = [existing, warning]
        else:
            compact["warnings"] = [warning]
    return compact


def _tick_quote_completeness_pct(payload: Dict[str, Any]) -> Optional[float]:
    data_quality = payload.get("data_quality")
    if not isinstance(data_quality, dict):
        return None
    complete = _as_nonnegative_int(data_quality.get("complete_ticks"))
    total = _as_nonnegative_int(data_quality.get("total_ticks"))
    if complete is None or not total:
        return None
    return round((float(complete) / float(total)) * 100.0, 2)


def _tick_valid_spread_sample_pct(payload: Dict[str, Any]) -> Optional[float]:
    data_quality = payload.get("data_quality")
    if not isinstance(data_quality, dict):
        return None
    valid = _as_nonnegative_int(data_quality.get("valid_spread_sample_count"))
    total = _as_nonnegative_int(data_quality.get("total_ticks"))
    if valid is None or not total:
        return None
    return round((float(valid) / float(total)) * 100.0, 2)


def _compact_tick_quality(payload: Dict[str, Any]) -> Any:
    notes: List[str] = []
    data_quality = payload.get("data_quality")
    total = None
    valid = None
    if isinstance(data_quality, dict):
        incomplete = _as_nonnegative_int(data_quality.get("incomplete_ticks"))
        total = _as_nonnegative_int(data_quality.get("total_ticks"))
        valid = _as_nonnegative_int(
            data_quality.get("valid_spread_sample_count")
        )
        if total is None:
            total = _as_nonnegative_int(payload.get("count"))
        if incomplete is not None and incomplete > 0 and total:
            notes.append(f"partial_quotes={incomplete}/{total}")
        else:
            status = str(data_quality.get("incomplete_quote_status") or "").strip().lower()
            if status and status not in {"ok", "info"}:
                notes.append(f"quote_quality={status}")
        if valid is not None and total and valid < total:
            notes.append(f"valid_spreads={valid}/{total}")
    quote_only = payload.get("feed_tier") == "quote_only"
    if payload.get("last_unavailable") is True and not quote_only:
        notes.append("last=unavailable")
    warnings = payload.get("warnings")
    if not notes and isinstance(warnings, list) and warnings:
        notes.append(f"warnings={len(warnings)}")
    if valid is not None and total is not None:
        quality: Dict[str, Any] = {
            "valid_spread_ticks": valid,
            "ticks_total": total,
        }
        if notes:
            quality["notes"] = "; ".join(notes)
        return quality
    if notes:
        return "; ".join(notes)
    return "ok" if quote_only else None


def _as_nonnegative_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _compact_tick_row(
    row: Any,
    *,
    last_spread: Optional[float] = None,
) -> tuple[Any, Optional[float]]:
    if not isinstance(row, dict):
        return row, None
    compact = {
        "time": row.get("time"),
        "bid": row.get("bid"),
        "ask": row.get("ask"),
    }
    if row.get("quote_type") not in (None, ""):
        compact["quote_type"] = row.get("quote_type")
    spread = row.get("spread")
    if spread in (None, ""):
        spread = _tick_row_spread(row.get("bid"), row.get("ask"))
    bid = _tick_row_price(row.get("bid"))
    ask = _tick_row_price(row.get("ask"))
    numeric_spread = _tick_row_price(spread)
    spread_valid = bool(
        bid is not None
        and ask is not None
        and numeric_spread is not None
        and ask > bid
        and numeric_spread > 0.0
    )
    compact["spread_snapshot_valid"] = spread_valid
    eligible = _tick_row_spread_sample_eligible(row, snapshot_valid=spread_valid)
    if eligible is not None and eligible != spread_valid:
        compact["spread_sample_eligible"] = eligible
    if spread_valid:
        compact["spread"] = numeric_spread
    if spread_valid:
        midpoint = canonical_quote_midpoint(bid, ask)
        if midpoint is not None:
            compact["mid"] = midpoint
    elif last_spread is not None and bid is not None and ask is None:
        compact["mid"] = round(bid + (last_spread / 2.0), 10)
        compact["mid_inferred"] = True
    elif last_spread is not None and ask is not None and bid is None:
        compact["mid"] = round(ask - (last_spread / 2.0), 10)
        compact["mid_inferred"] = True
    last = _tick_row_price(row.get("last"))
    if last is not None and last > 0.0:
        compact["last"] = last
    for field in ("volume", "volume_real"):
        volume = _tick_row_price(row.get(field))
        if volume is not None and volume != 0.0:
            compact[field] = volume
    decoded = row.get("flags_decoded")
    if isinstance(decoded, list) and decoded:
        quote_flags = {str(value).strip().lower() for value in decoded}
        bid_updated = "bid" in quote_flags
        ask_updated = "ask" in quote_flags
        if bid_updated != ask_updated:
            compact["quote_update_type"] = (
                "bid_only_update" if bid_updated else "ask_only_update"
            )
        elif not spread_valid and bid_updated and ask_updated:
            compact["quote_update_type"] = "bid_ask_update"
    elif str(row.get("quote_update_type") or "") in {
        "bid_only_update",
        "ask_only_update",
        "bid_ask_update",
    } and (not spread_valid or row.get("quote_update_type") != "bid_ask_update"):
        compact["quote_update_type"] = row["quote_update_type"]
    return compact, numeric_spread if spread_valid else None


def _tick_row_spread_sample_eligible(
    row: Dict[str, Any],
    *,
    snapshot_valid: bool,
) -> Optional[bool]:
    """Return eligibility from the complete bid/ask snapshot, not delta flags."""
    return bool(snapshot_valid)


def _tick_row_spread(bid: Any, ask: Any) -> Optional[float]:
    try:
        if bid in (None, "") or ask in (None, ""):
            return None
        return round(float(ask) - float(bid), 10)
    except (TypeError, ValueError):
        return None


def _tick_row_price(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _run_wait_event_impl(
    *,
    request: WaitEventRequest,
    gateway: Any,
    sleep_impl: Any,
    monotonic_impl: Any,
    now_utc_impl: Any,
) -> Dict[str, Any]:
    try:
        return run_wait_event_loop(
            request,
            gateway=gateway,
            sleep_impl=sleep_impl,
            monotonic_impl=monotonic_impl,
            now_utc_impl=now_utc_impl,
        )
    except ValueError as exc:
        return build_error_payload(
            str(exc),
            code="wait_event_error",
            operation="wait_event",
        )


def _wait_event_needs_gateway(request: WaitEventRequest) -> bool:
    if request.watch_for:
        return True
    if request.symbol is not None or bool(request.symbols):
        return True
    return any(
        getattr(item, "type", None) != "candle_close"
        for item in (request.end_on or ())
    )
