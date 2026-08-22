import json
import logging
import statistics
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from ...services.data_service import fetch_candles, fetch_ticks
from ...shared.constants import CALENDAR_TIMEFRAMES, TIMEFRAME_SECONDS
from ...shared.schema import DetailLiteral, TimeframeLiteral
from ...utils.coercion import coerce_finite_float
from ...utils.mt5 import (
    ensure_mt5_connection_or_raise,
    get_symbol_info_cached,
    symbol_candle_price_basis,
)
from .._mcp_instance import mcp
from ..execution_logging import run_logged_operation
from ..mt5_gateway import create_mt5_gateway
from ..pivot import pivot_compute_points, support_resistance_levels
from .requests import (
    WAIT_EVENT_MAX_SYMBOLS,
    DataFetchCandlesRequest,
    DataFetchTicksRequest,
    WaitEventRequest,
)
from .use_cases import (
    run_data_fetch_candles,
    run_data_fetch_ticks,
    run_wait_event,
)
from .wait_events import _WAIT_EVENT_IDENTITY_FIELDS

# Explicitly define what should be exported for '*' imports
__all__ = ['data_fetch_candles', 'data_fetch_ticks', 'wait_event']

logger = logging.getLogger(__name__)

_WAIT_EVENT_BOUNDARY_TYPES = {"candle_close"}
_WAIT_EVENT_SPEC_HINT = (
    'Use event names like order_filled or JSON objects like {"type":"order_filled",'
    '"symbol":"EURUSD"}; use candle_close for candle-boundary waits.'
)
_WAIT_EVENT_MODE_HINT = (
    "Set timeframe for a candle-boundary wait, optionally bounded by "
    "max_wait_seconds, or set max_wait_seconds alone for a duration wait."
)


def _normalize_wait_event_public_specs(
    value: Any,
    *,
    field_name: str,
) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    if value is None:
        return None, None
    if isinstance(value, dict):
        return [dict(value)], None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return [], None
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except Exception as exc:
                return None, f"wait_event {field_name} JSON is invalid: {exc}"
            return _normalize_wait_event_public_specs(parsed, field_name=field_name)
        return [{"type": text}], None
    if isinstance(value, (list, tuple)):
        out: List[Dict[str, Any]] = []
        for item in value:
            parsed, error = _normalize_wait_event_public_specs(item, field_name=field_name)
            if error is not None:
                return None, error
            if parsed:
                out.extend(parsed)
        return out, None
    return None, f"wait_event {field_name} must be event objects or event type strings."


def _normalize_wait_event_symbols(
    value: Any,
) -> tuple[Optional[List[str]], Optional[str]]:
    if value is None:
        return None, None
    if not isinstance(value, (list, tuple)):
        return None, "wait_event symbols must be a list of trading symbols."
    if not value:
        return None, "wait_event symbols must contain at least one symbol."
    if len(value) > WAIT_EVENT_MAX_SYMBOLS:
        return None, (
            f"wait_event symbols accepts at most {WAIT_EVENT_MAX_SYMBOLS} symbols."
        )
    normalized: List[str] = []
    seen: set[str] = set()
    for raw_symbol in value:
        symbol = str(raw_symbol or "").upper().strip()
        if not symbol:
            return None, "wait_event symbols entries must be non-empty strings."
        if symbol in seen:
            return None, (
                "wait_event symbols entries must be unique after normalization; "
                f"received duplicate {symbol}."
            )
        seen.add(symbol)
        normalized.append(symbol)
    return normalized, None


def _move_wait_event_boundary_watchers(
    watch_for: Optional[List[Dict[str, Any]]],
    end_on: Optional[List[Dict[str, Any]]],
) -> tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]], bool]:
    if not watch_for:
        return watch_for, end_on, False

    remaining_watchers: List[Dict[str, Any]] = []
    boundary_watchers: List[Dict[str, Any]] = []
    for item in watch_for:
        event_type = str(item.get("type") or "").strip()
        if event_type in _WAIT_EVENT_BOUNDARY_TYPES:
            boundary_watchers.append(dict(item))
        else:
            remaining_watchers.append(item)
    if not boundary_watchers:
        return watch_for, end_on, False
    resolved_end_on = list(end_on or [])
    resolved_end_on.extend(boundary_watchers)
    return remaining_watchers, resolved_end_on, True


def _wait_event_validation_error(exc: ValidationError) -> tuple[str, str]:
    try:
        errors = exc.errors()
    except Exception:
        return "wait_event request is invalid.", "wait_event_invalid_request"
    messages: List[str] = []
    spec_error = False
    for item in errors:
        loc = ".".join(str(part) for part in item.get("loc", ()))
        msg = str(item.get("msg") or "Invalid value.")
        if loc.split(".", 1)[0] in {"watch_for", "end_on"}:
            spec_error = True
        messages.append(f"{loc}: {msg}" if loc else msg)
    prefix = "Invalid wait_event event spec" if spec_error else "Invalid wait_event request"
    code = "wait_event_invalid_watch_spec" if spec_error else "wait_event_invalid_request"
    return f"{prefix}: {'; '.join(messages)}", code


def _build_default_wait_event_watchers(
    *,
    symbol: str,
    timeframe: TimeframeLiteral,
    watch_tick_count_spike: bool,
) -> List[Dict[str, Any]]:
    watch_for: List[Dict[str, Any]] = [
        {"type": "order_created", "symbol": symbol},
        {"type": "order_filled", "symbol": symbol},
        {"type": "order_cancelled", "symbol": symbol},
        {"type": "position_opened", "symbol": symbol},
        {"type": "position_closed", "symbol": symbol},
        {"type": "tp_hit", "symbol": symbol},
        {"type": "sl_hit", "symbol": symbol},
        {"type": "pending_near_fill", "symbol": symbol},
        {"type": "stop_threat", "symbol": symbol},
        {"type": "price_change", "symbol": symbol},
        {"type": "volume_spike", "symbol": symbol},
        {"type": "spread_spike", "symbol": symbol},
        {"type": "tick_count_drought", "symbol": symbol},
        {"type": "range_expansion", "symbol": symbol},
    ]
    if watch_tick_count_spike:
        watch_for.append({"type": "tick_count_spike", "symbol": symbol})
    return _dedupe_wait_event_watchers(watch_for)


def _build_default_wait_event_basket_watchers(
    *,
    symbols: List[str],
    timeframe: TimeframeLiteral,
    watch_tick_count_spike: bool,
) -> List[Dict[str, Any]]:
    watch_for: List[Dict[str, Any]] = []
    for symbol in symbols:
        watch_for.extend(
            _build_default_wait_event_watchers(
                symbol=symbol,
                timeframe=timeframe,
                watch_tick_count_spike=watch_tick_count_spike,
            )
        )
    return _dedupe_wait_event_watchers(watch_for)


def _support_resistance_watchers(
    *,
    symbol: str,
) -> List[Dict[str, Any]]:
    try:
        raw_tool = getattr(support_resistance_levels, "__wrapped__", support_resistance_levels)
        payload = raw_tool(symbol=symbol, timeframe="auto", detail="compact")
    except Exception:
        return []
    if not isinstance(payload, dict) or payload.get("error"):
        return []
    levels = payload.get("levels")
    if not isinstance(levels, list):
        return []
    watch_for: List[Dict[str, Any]] = []
    price_source = _default_level_price_source(symbol)
    for level in levels:
        if not isinstance(level, dict):
            continue
        level_value = coerce_finite_float(level.get("value"))
        if level_value is None:
            continue
        level_type = str(level.get("type") or "").strip().lower()
        direction = "either"
        if level_type == "support":
            direction = "down"
        elif level_type == "resistance":
            direction = "up"
        watch_for.append(
            {
                "type": "price_touch_level",
                "symbol": symbol,
                "level": level_value,
                "direction": "either",
                "price_source": price_source,
            }
        )
        watch_for.append(
            {
                "type": "price_break_level",
                "symbol": symbol,
                "level": level_value,
                "direction": direction,
                "price_source": price_source,
            }
        )
    return watch_for


def _pivot_zone_watchers(*, symbol: str, timeframe: TimeframeLiteral) -> List[Dict[str, Any]]:
    try:
        raw_tool = getattr(pivot_compute_points, "__wrapped__", pivot_compute_points)
        payload = raw_tool(
            symbol=symbol,
            timeframe=_default_wait_event_pivot_timeframe(timeframe),
            detail="standard",
        )
    except Exception:
        return []
    if not isinstance(payload, dict) or payload.get("error"):
        return []
    levels = _extract_pivot_levels(payload)
    if len(levels) < 2:
        return []
    watch_for: List[Dict[str, Any]] = []
    price_source = _default_level_price_source(symbol)
    for idx in range(len(levels) - 1):
        lower = levels[idx]["value"]
        upper = levels[idx + 1]["value"]
        if upper <= lower:
            continue
        watch_for.append(
            {
                "type": "price_enter_zone",
                "symbol": symbol,
                "lower": lower,
                "upper": upper,
                "direction": "either",
                "price_source": price_source,
            }
        )
    return watch_for


def _default_level_price_source(symbol: str) -> str:
    """Match generated chart levels against the broker's chart price basis."""
    try:
        basis = symbol_candle_price_basis(get_symbol_info_cached(symbol))
    except Exception:
        return "auto"
    if basis == "bid":
        return "bid"
    if basis == "last_trade":
        return "last"
    return "auto"


def _default_wait_event_pivot_timeframe(timeframe: TimeframeLiteral) -> TimeframeLiteral:
    normalized = str(timeframe or "M1").upper().strip()
    if normalized in CALENDAR_TIMEFRAMES:
        return normalized  # type: ignore[return-value]
    return "D1"


def _extract_pivot_levels(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("levels")
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    seen_values: set[float] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("level") or "").strip().upper()
        if not label:
            continue
        values = [
            numeric
            for numeric in (coerce_finite_float(value) for key, value in row.items() if key != "level")
            if numeric is not None
        ]
        if not values:
            continue
        price = round(float(statistics.median(values)), 10)
        if price in seen_values:
            continue
        seen_values.add(price)
        out.append({"label": label, "value": price})
    out.sort(key=lambda item: float(item["value"]))
    return out


def _dedupe_wait_event_watchers(watch_for: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in watch_for:
        key = (
            str(item.get("type") or ""),
            str(item.get("symbol") or "").upper(),
            item.get("order_ticket"),
            item.get("position_ticket"),
            item.get("magic"),
            item.get("side"),
            item.get("direction"),
            item.get("level"),
            item.get("lower"),
            item.get("upper"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(item))
    return out


def _compact_wait_event_criteria(matched_event: Dict[str, Any]) -> Dict[str, Any]:
    criteria = matched_event.get("criteria")
    if not isinstance(criteria, dict):
        return {}
    return {
        field_name: criteria.get(field_name)
        for field_name in (
            "threshold_mode",
            "threshold_value",
            "direction",
            "level",
            "lower",
            "upper",
            "distance",
            "price_source",
            "confirm_ticks",
        )
        if criteria.get(field_name) is not None
    }


def _wait_event_trigger_reason(matched_event: Dict[str, Any]) -> Optional[str]:
    event_type = str(matched_event.get("type") or "").strip()
    if not event_type:
        return None
    criteria = _compact_wait_event_criteria(matched_event)
    reason_parts = [event_type]
    for field_name in (
        "threshold_value",
        "level",
        "lower",
        "upper",
        "distance",
        "direction",
    ):
        value = criteria.get(field_name)
        if value is not None:
            reason_parts.append(f"{field_name}={value}")
    return ", ".join(reason_parts)


def _wait_event_monitored_types(criteria: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(criteria, dict):
        return []
    event_types: set[str] = set()
    for field_name in ("watch_for", "end_on"):
        specs = criteria.get(field_name)
        if not isinstance(specs, list):
            continue
        for spec in specs:
            event_type = str(
                (
                    spec.get("type")
                    if isinstance(spec, dict)
                    else getattr(spec, "type", "")
                )
                or ""
            ).strip()
            if event_type:
                event_types.add(event_type)
    return sorted(event_types)


def _wait_event_next_poll_hint(poll_interval_seconds: Any) -> Optional[str]:
    seconds = coerce_finite_float(poll_interval_seconds)
    if seconds is None or seconds <= 0.0:
        return None
    return f"retry after {seconds:g}s"


def _compact_wait_event_public_result(
    result: Dict[str, Any],
    *,
    explicit_watch_for: bool,
    explicit_end_on: bool,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    out = dict(result)
    max_wait_seconds = out.pop("max_wait_seconds", None)
    elapsed_seconds = out.get("elapsed_seconds")
    poll_interval_seconds = out.get("poll_interval_seconds")
    status = str(out.get("status") or "").strip().lower()

    criteria_in = out.get("criteria")
    criteria = dict(criteria_in) if isinstance(criteria_in, dict) else None
    if criteria is not None:
        watch_specs = criteria.get("watch_for")
        watcher_count = len(watch_specs) if isinstance(watch_specs, list) else 0
        watch_for_inferred = bool(not explicit_watch_for and watcher_count)
        criteria["watch_for_inferred"] = watch_for_inferred
        criteria["end_on_inferred"] = not explicit_end_on
        watcher_types = _wait_event_monitored_types(
            {"watch_for": watch_specs if isinstance(watch_specs, list) else []}
        )
        out["watch_for_inferred"] = watch_for_inferred
        out["watcher_count"] = watcher_count
        out["watcher_types"] = watcher_types

    if str(detail or "compact").strip().lower() == "full":
        if criteria is not None:
            out["criteria"] = criteria
        return out

    for key in (
        "event",
        "criteria",
        "timeframe",
        "started_at_utc",
        "elapsed_seconds",
        "polls",
        "poll_interval_seconds",
        "sleep_seconds",
        "slept",
        "slept_seconds",
    ):
        out.pop(key, None)

    if status != "wait_budget_exceeded":
        out.pop("remaining_seconds", None)

    boundary_event = out.get("boundary_event")
    if isinstance(boundary_event, dict):
        compact_boundary = {
            key: boundary_event.get(key)
            for key in ("type", "timeframe")
            if boundary_event.get(key) is not None
        }
        closed_candle = boundary_event.get("closed_candle")
        if isinstance(closed_candle, dict) and closed_candle:
            compact_boundary["closed_candle"] = dict(closed_candle)
        closed_candles = boundary_event.get("closed_candles")
        if isinstance(closed_candles, list):
            compact_boundary["closed_candles"] = [
                dict(item) for item in closed_candles if isinstance(item, dict)
            ]
        candle_failures = boundary_event.get("candle_failures")
        if isinstance(candle_failures, list) and candle_failures:
            compact_boundary["candle_failures"] = [
                dict(item) for item in candle_failures if isinstance(item, dict)
            ]
        out["boundary_event"] = compact_boundary or None

    matched_event = out.get("matched_event")
    if isinstance(matched_event, dict):
        compact_matched: Dict[str, Any] = {}
        event_type = matched_event.get("type")
        if event_type is not None:
            compact_matched["type"] = event_type
            compact_matched["watcher_type"] = event_type
        trigger_reason = _wait_event_trigger_reason(matched_event)
        if trigger_reason:
            compact_matched["trigger_reason"] = trigger_reason
        compact_criteria = _compact_wait_event_criteria(matched_event)
        if compact_criteria:
            compact_matched["criteria"] = compact_criteria
        for field_name in _WAIT_EVENT_IDENTITY_FIELDS:
            value = matched_event.get(field_name)
            if value is not None:
                compact_matched[field_name] = value
        observed = matched_event.get("observed")
        if isinstance(observed, dict) and observed:
            compact_matched["observed"] = dict(observed)
        out["matched_event"] = compact_matched or None

    if status == "timeout":
        out["timeout"] = True
        out["timed_out"] = True
        out["events"] = []
        mode = str(out.get("wait_mode") or "duration")
        out["wait_mode"] = mode
        if elapsed_seconds is not None:
            out["waited_seconds"] = elapsed_seconds
        if max_wait_seconds is not None:
            out["max_wait_seconds"] = max_wait_seconds
        if poll_interval_seconds is not None:
            out["poll_interval_seconds"] = poll_interval_seconds
        next_poll_hint = _wait_event_next_poll_hint(poll_interval_seconds)
        if next_poll_hint:
            out["next_poll_hint"] = next_poll_hint
        monitored_types = (
            _wait_event_monitored_types(criteria) if explicit_watch_for else []
        )
        if monitored_types:
            out["events_monitored"] = monitored_types
        watcher_types = list(out.get("watcher_types") or [])
        out["details"] = {
            "mode": mode,
            "watch_for": watcher_types,
            "watch_for_inferred": not explicit_watch_for,
            "elapsed_seconds": elapsed_seconds,
            "requested_wait_seconds": max_wait_seconds,
        }
        out.setdefault(
            "remediation",
            "Retry the same wait or increase max_wait_seconds.",
        )
    else:
        wait_policy = {
            key: value
            for key, value in (
                ("elapsed_seconds", elapsed_seconds),
                ("max_wait_seconds", max_wait_seconds),
                ("poll_interval_seconds", poll_interval_seconds),
            )
            if value is not None
        }
        if wait_policy:
            out["wait_policy"] = wait_policy

    return out


@mcp.tool()
def data_fetch_candles(
    request: DataFetchCandlesRequest,
) -> Dict[str, Any]:
    """Fetch historical candle data with optional technical indicators and denoising.
    
    **REQUIRED**: symbol parameter must be provided (e.g., "EURUSD", "BTCUSD")
    
    Features:
    ---------
    - OHLCV data as tabular rows
    - Optional historical candle spread column via include_spread=true
    - Technical indicators (RSI, MACD, EMA, SMA, etc.)
    - Data denoising and smoothing
    - Data simplification for large datasets
    - Defaults to closed candles only; set include_incomplete=true to keep the latest forming candle
    - Set allow_stale=true to return the latest available closed bars even when freshness checks would normally fail; bounded historical ranges do not use the live-feed freshness gate
    - Includes metadata for forming-candle handling (for example has_forming_candle and incomplete_candles_skipped)
    
    Parameters:
    -----------
    symbol : str (REQUIRED)
        Trading symbol (e.g., "EURUSD", "GBPUSD", "BTCUSD")
    
    timeframe : str, optional (default="H1")
        Candle timeframe: "M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"

    detail : {"compact", "standard", "summary", "full"}, optional
        Response detail level. `compact` (default) returns rows plus concise
        freshness when available. `summary` returns metadata and diagnostics
        without candle rows. `standard` also includes latency and policy
        freshness signals with rows. `full` preserves the debug `meta`
        diagnostics block.
    
    limit : int, optional (default=20)
        Maximum number of candles to return
    
    start : str, optional
        Start time (dateparser)

    end : str, optional
        End time (dateparser)

    cursor : str, optional
        Opaque continuation token from a prior start-anchored candle page. Reuse
        it with the original symbol, timeframe, start, and end values.
    
    ohlcv : str, optional
        Candle fields to include. Use "all", "ohlcv", "ohlc", "close"/"price",
        compact letters from o/h/l/c/v, or comma-separated field names such as
        "open,high,low,close,volume".

    include_spread : bool, optional
        Request the historical MT5 per-bar spread column. When unavailable,
        the result reports spread_mode=single_reference with one non-historical
        live/tick reference, or spread_mode=unavailable. Defaults to false.
    
    indicators : list, optional
        Technical indicators list, e.g., [{"name": "rsi", "params": [14]}]
        Or compact string: "rsi(14),ema(20),macd(12,26,9)"
    
    denoise : dict, optional
        Denoising configuration to smooth price data
    
    simplify : dict, optional
        Data reduction options for large datasets. Use a dict such as
        {"method": "lttb", "points": 100} or {"ratio": 0.25}. Passing
        true/"on"/"default" enables default simplification; false/"off"
        disables it.

    include_incomplete : bool, optional
        Keep the latest forming candle instead of trimming it. Defaults to false.

    allow_stale : bool, optional
        Return the latest available closed bars even if they fall outside the normal
        freshness window. This only affects unbounded latest-N queries; requests with
        start or end bounds are historical and bypass live-feed freshness checks.
        Defaults to false.

    explain_indicators : bool, optional
        When true, add compact latest-value interpretation notes for common
        requested indicators. Defaults to false to keep row output lean.
    
    Returns:
    --------
    dict
        - success: bool
        - symbol: str
        - timeframe: str
        - count: int (number of candles returned)
        - has_forming_candle: bool (true when the latest available candle is still forming)
        - forming_candle_status: str ("included", "skipped", "detected", or "none")
        - forming_candle_included: bool (true when the forming candle is present in data)
        - forming_candle_skipped: bool (true when a forming candle was detected but trimmed)
        - incomplete_candles_skipped: int (number of forming candles trimmed because include_incomplete=false)
        - data: list[dict] (tabular candle rows)
    
    Examples:
    ---------
    # Get last 20 H1 candles
    data_fetch_candles(symbol="EURUSD")
    
    # Get 100 M15 candles with RSI indicator
    data_fetch_candles(
        symbol="EURUSD",
        timeframe="M15",
        limit=100,
        indicators="rsi(14)"
    )
    
    # Get date range with multiple indicators
    data_fetch_candles(
        symbol="GBPUSD",
        start="2025-11-01",
        end="2025-11-30",
        indicators="rsi(14),ema(20),macd(12,26,9)"
    )

    # Opt in to historical candle spread output
    data_fetch_candles(symbol="EURUSD", include_spread=True)
    """
    return run_logged_operation(
        logger,
        operation="data_fetch_candles",
        symbol=request.symbol,
        timeframe=request.timeframe,
        detail=request.detail,
        limit=request.limit,
        func=lambda: run_data_fetch_candles(
            request,
            gateway=create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
            fetch_candles_impl=fetch_candles,
        ),
    )

@mcp.tool()
def data_fetch_ticks(
    request: DataFetchTicksRequest,
) -> Dict[str, Any]:
    """Fetch tick data for a symbol.

    By default (`detail="compact"`), returns tick rows plus compact descriptive
    stats over the fetched ticks.

    Use `detail="summary"` or `detail="standard"` for stats-only payloads.
    Use `detail="full"` to return raw tick rows as structured data.
    `simplify` only applies to row output. Use a dict such as
    {"method": "lttb", "points": 100} or pass true/"on"/"default" for
    default simplification; false/"off" disables it.
    """
    return run_logged_operation(
        logger,
        operation="data_fetch_ticks",
        symbol=request.symbol,
        limit=request.limit,
        detail=request.detail,
        func=lambda: run_data_fetch_ticks(
            request,
            gateway=create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
            fetch_ticks_impl=fetch_ticks,
        ),
    )


@mcp.tool()
def wait_event(
    symbol: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    timeframe: Optional[TimeframeLiteral] = None,
    watch_tick_count_spike: bool = True,
    max_wait_seconds: Optional[float] = None,
    poll_interval_seconds: Optional[float] = None,
    accept_preexisting: bool = False,
    watch_for: Optional[List[Dict[str, Any]]] = None,
    end_on: Optional[List[Dict[str, Any]]] = None,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """BLOCKING: Wait for a timeframe boundary and/or a duration deadline.

    Set `timeframe` to stop at the next candle boundary, optionally with
    `max_wait_seconds` as a safety cap. Set `max_wait_seconds` alone to stop after
    a fixed duration. Omitting both is invalid.

    In timeframe mode, omitting `watch_for` waits only for the candle boundary.
    Pass explicit order/position/market watchers when those events should end
    the wait early. `max_wait_seconds` defaults to the timeframe length plus 60
    seconds so a weekend H1 wait cannot block until Sunday reopen.

    In duration mode, omitting `watch_for` creates a timer-only wait. It does not
    connect to MT5 or poll market/account state. Pass explicit watcher objects to
    return early when an event matches; an unmatched explicit event wait fails
    with `wait_event_timeout`.

    Supply either `symbol` for a single instrument or `symbols` for a basket of
    up to 12 instruments; the parameters are mutually exclusive. Basket waits
    return on the first matching event. Explicit watcher specs without a symbol
    are broadcast across the basket, while named watcher symbols must belong to
    it.

    A timeframe wait without `symbol`, `symbols`, or `watch_for` is a pure clock
    boundary wait and returns no candle data. Pass `watch_for=[]` to request the
    same boundary-only behavior while still collecting candle statistics for a
    supplied symbol or basket.

    `max_wait_seconds` alone selects duration mode. With `timeframe`, it bounds
    the candle-boundary wait; when the boundary is beyond that budget, the call
    returns `wait_budget_exceeded` without sleeping. Explicit `end_on`
    timeframes must match the top-level `timeframe`.
    A timer-only duration or a timeframe boundary reached with inferred watchers
    is a successful completion.
    With explicit `watch_for`, a timeout is a failed wait (`success=false`,
    `error_code=wait_event_timeout`) and produces a nonzero CLI exit status. Timeout responses set
    `timed_out=true`, return `events=[]`, identify `wait_mode`, and include the
    requested/elapsed timing context plus a retry remediation. When the watched
    symbol's market is closed for the wait window, the timeout also includes
    `market_status`, `assumed_closure_end`, and a remediation that points at
    reopen instead of a blind retry. For singular
    waits with explicit watchers, reaching an `end_on` boundary before a match
    is also a failed wait
    (`success=false`, `matched=false`,
    `error_code=wait_event_boundary_reached`); `completed=true` distinguishes
    that terminal boundary from a timeout. Basket boundaries complete
    successfully so their candle snapshot can drive the next basket cycle. A
    boundary-only wait
    (`watch_for=[]`, or a symbol-less timeframe wait) succeeds when its boundary
    is reached.
    Set `poll_interval_seconds` to tune polling cadence; omit it to use the
    engine default.

    Set `accept_preexisting=true` to return immediately when a state-style
    watcher is already satisfied during setup. The default false value keeps
    edge-triggered semantics and waits for a new transition after startup.

    Boundary waits belong in `end_on` as `{"type": "candle_close", ...}`.
    `watch_for` is for explicit market/account event objects only; pass
    candle-close boundary objects in `end_on` instead.
    When a candle boundary is reached, a singular call includes a best-effort
    `closed_candle` snapshot. Basket calls include `closed_candles` and any
    per-symbol `candle_failures`; missing basket candles produce partial success.

    Example: `timeframe="H1", end_on=[{"type": "candle_close",
    "timeframe": "H1"}]` or `max_wait_seconds=30,
    watch_for=[{"type": "order_filled", "symbol": "EURUSD"}]`.

    Advanced callers can pass explicit `watch_for` and `end_on` event specs to
    use the richer wait-event engine directly. `watch_tick_count_spike` only
    alters the inferred timeframe watcher list; it does not affect timer-only
    duration waits or explicit `watch_for`.
    Set `detail="full"` to include polling/timing details and the full criteria
    echo in the response.
    """
    symbol_value = str(symbol or "").strip() or None
    symbols_value, symbols_error = _normalize_wait_event_symbols(symbols)
    normalized_watch_for, watch_for_error = _normalize_wait_event_public_specs(
        watch_for,
        field_name="watch_for",
    )
    normalized_end_on, end_on_error = _normalize_wait_event_public_specs(
        end_on,
        field_name="end_on",
    )
    moved_boundary_watchers = False
    if watch_for_error is None and end_on_error is None:
        normalized_watch_for, normalized_end_on, moved_boundary_watchers = (
            _move_wait_event_boundary_watchers(normalized_watch_for, normalized_end_on)
        )
    explicit_watch_for = normalized_watch_for is not None
    explicit_end_on = normalized_end_on is not None
    wait_mode_error: Optional[str] = None
    if timeframe is None and max_wait_seconds is None:
        wait_mode_error = "Provide timeframe and/or max_wait_seconds."
    request_error: Optional[str] = None
    spec_error = watch_for_error or end_on_error
    if symbols_error is not None:
        request_error = symbols_error
    elif symbol_value is not None and symbols_value is not None:
        request_error = "symbol and symbols cannot be combined."

    def _run() -> Dict[str, Any]:
        if spec_error is not None:
            return {
                "error": spec_error,
                "error_code": "wait_event_invalid_watch_spec",
                "hint": _WAIT_EVENT_SPEC_HINT,
            }
        if wait_mode_error is not None:
            return {
                "error": wait_mode_error,
                "error_code": "wait_event_invalid_request",
                "hint": _WAIT_EVENT_MODE_HINT,
            }
        if request_error is not None:
            return {
                "error": request_error,
                "error_code": "wait_event_invalid_request",
            }
        request_kwargs: Dict[str, Any] = {}
        if timeframe is not None:
            request_kwargs["timeframe"] = timeframe
        if symbol_value is not None:
            request_kwargs["symbol"] = symbol_value
        if symbols_value is not None:
            request_kwargs["symbols"] = list(symbols_value)
        if max_wait_seconds is None and timeframe is not None:
            request_kwargs["max_wait_seconds"] = float(
                TIMEFRAME_SECONDS.get(str(timeframe).upper(), 3600)
            ) + 60.0
        else:
            request_kwargs["max_wait_seconds"] = max_wait_seconds
        request_kwargs["accept_preexisting"] = bool(accept_preexisting)
        if poll_interval_seconds is not None:
            request_kwargs["poll_interval_seconds"] = poll_interval_seconds
        if normalized_end_on is not None:
            request_kwargs["end_on"] = list(normalized_end_on)
        elif timeframe is not None and normalized_watch_for is None:
            request_kwargs["end_on"] = [
                {"type": "candle_close", "timeframe": timeframe},
            ]
        if normalized_watch_for is not None:
            resolved_watch_for = list(normalized_watch_for)
        else:
            resolved_watch_for = []
        try:
            request = WaitEventRequest(
                **request_kwargs,
                watch_for=resolved_watch_for,
            )
            request._watch_for_inferred = bool(
                not explicit_watch_for and resolved_watch_for
            )
        except ValidationError as exc:
            error_message, error_code = _wait_event_validation_error(exc)
            return {
                "error": error_message,
                "error_code": error_code,
                "hint": (
                    _WAIT_EVENT_SPEC_HINT
                    if error_code == "wait_event_invalid_watch_spec"
                    else _WAIT_EVENT_MODE_HINT
                ),
            }
        result = run_wait_event(
            request,
            gateway=create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
        )
        if isinstance(result, dict):
            result = _compact_wait_event_public_result(
                result,
                explicit_watch_for=explicit_watch_for,
                explicit_end_on=explicit_end_on,
                detail=detail,
            )
        return result

    return run_logged_operation(
        logger,
        operation="wait_event",
        symbol=symbol_value,
        symbols=symbols_value,
        timeframe=timeframe,
        watch_tick_count_spike=watch_tick_count_spike,
        detail=detail,
        explicit_watch_for=explicit_watch_for,
        moved_boundary_watchers=moved_boundary_watchers,
        end_on_count=len(normalized_end_on or []),
        func=_run,
    )
