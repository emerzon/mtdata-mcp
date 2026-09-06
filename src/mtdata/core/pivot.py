
import logging
import math
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import Field

from ..bootstrap.settings import mt5_config
from ..forecast.common import fetch_history as _fetch_history
from ..shared.constants import CALENDAR_TIMEFRAMES, TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..shared.schema import (
    _PIVOT_METHODS,
    AutoTimeframeLiteral,
    DetailLiteral,
    PivotMethodLiteral,
    TimeframeLiteral,
)
from ..shared.validators import (
    invalid_timeframe_error,
    unsupported_timeframe_seconds_error,
)
from ..utils.coercion import round_finite
from ..utils.freshness import (
    COMPLETED_BAR_FRESHNESS_KEYS,
    completed_bar_freshness_fields,
)
from ..utils.level_confluence import build_level_confluence_payload
from ..utils.market_metadata import build_tick_freshness_context
from ..utils.mt5 import (
    MT5ConnectionError,
    _mt5_copy_rates_from,
    _symbol_ready_guard,
    ensure_mt5_connection_or_raise,
    mt5,
    resolve_public_symbol,
    symbol_price_digits,
    symbol_price_digits_optional,
)
from ..utils.pivot_points import compute_pivot_method_levels, compute_pivot_methods
from ..utils.quote import (
    compute_spread_metrics,
    enforce_quote_execution_readiness,
    resolve_quote_tick,
    tick_value,
)
from ..utils.quote import (
    tick_epoch as quote_tick_epoch,
)
from ..utils.support_resistance import (
    compact_support_resistance_payload,
    compute_support_resistance_levels,
    full_support_resistance_payload,
    get_auto_support_resistance_timeframes,
    merge_support_resistance_results,
    standard_support_resistance_payload,
)
from ..utils.time import (
    _format_time_minimal,
    _format_time_minimal_local,
    _resolve_client_tz,
    _use_client_tz,
    bar_close_epoch,
    display_timezone_label,
    format_datetime_utc,
    format_epoch_utc,
    parse_iso_utc,
)
from ..utils.time import (
    timezone_label as _timezone_object_label,
)
from ..utils.utils import (
    _parse_end_datetime,
    _positive_float_attr,
    validate_historical_range,
)
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .mt5_gateway import create_mt5_gateway
from .output_contract import attach_completed_bar_input_policy
from .runtime_metadata import run_mt5_logged_operation
from .volume_profile import compute_volume_profile_payload

logger = logging.getLogger(__name__)


def _as_warning_object(item: Any, *, default_code: str = "warning") -> Dict[str, Any]:
    """Normalize a warning list item to ``{code, message}``."""
    if isinstance(item, dict):
        message = item.get("message")
        if message in (None, ""):
            message = item.get("warning")
        if message in (None, ""):
            message = str(item)
        out = dict(item)
        out["code"] = str(item.get("code") or default_code)
        out["message"] = str(message)
        return out
    return {"code": default_code, "message": str(item)}


def _normalize_warning_list(warnings: Any) -> List[Dict[str, Any]]:
    if not isinstance(warnings, list):
        return []
    return [_as_warning_object(item) for item in warnings]


def _has_field(row: Any, name: str) -> bool:
    try:
        if isinstance(row, dict):
            return name in row
        dtype = getattr(row, "dtype", None)
        names = getattr(dtype, "names", None) if dtype is not None else None
        return bool(names and name in names)
    except Exception:
        return False


def _attach_pivot_trust_metadata(
    payload: Dict[str, Any],
    *,
    symbol: str,
    timeframe: Any,
    last_bar_epoch: Any,
    now_epoch: Any,
) -> None:
    freshness = completed_bar_freshness_fields(
        symbol,
        timeframe,
        last_bar_epoch,
        now_epoch=now_epoch,
        item="bar",
    )
    for key in COMPLETED_BAR_FRESHNESS_KEYS:
        if key in freshness:
            payload[key] = freshness[key]
    payload["price_basis"] = "bid"
    warnings = _normalize_warning_list(payload.get("warnings"))
    stale_warning = freshness.get("stale_warning")
    if stale_warning:
        stale_item = _as_warning_object(stale_warning, default_code="stale_bar")
        if stale_item not in warnings and stale_item["message"] not in {
            item.get("message") for item in warnings
        }:
            warnings.append(stale_item)
    note = freshness.get("note")
    if note:
        note_item = _as_warning_object(note, default_code="freshness_note")
        if note_item not in warnings and note_item["message"] not in {
            item.get("message") for item in warnings
        }:
            warnings.append(note_item)
    if warnings:
        payload["warnings"] = warnings


_LEVEL_PRICE_FIELD_NAMES = frozenset(
    {
        "value",
        "price",
        "reference_price",
        "current_price",
        "nearest_support",
        "nearest_resistance",
        "low",
        "high",
        "width",
        "distance",
        "range",
    }
)


def _round_level_price(value: Any, *, digits: int) -> Any:
    return round_finite(value, digits, on_invalid="passthrough")


def _round_level_payload_prices(value: Any, *, digits: Optional[int], key: Optional[str] = None) -> Any:
    if digits is None:
        return value
    if key == "level_counts":
        return value
    if isinstance(value, dict):
        return {
            item_key: _round_level_payload_prices(
                item_value,
                digits=digits,
                key=str(item_key),
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [
            _round_level_payload_prices(item, digits=digits, key=key)
            for item in value
        ]
    if key in _LEVEL_PRICE_FIELD_NAMES or (isinstance(key, str) and key.endswith("_price")):
        return _round_level_price(value, digits=digits)
    return value


def _as_of_epoch(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, datetime):
        resolved = (
            value.replace(tzinfo=timezone.utc)
            if value.tzinfo is None
            else value.astimezone(timezone.utc)
        )
        return resolved.timestamp()
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, str) and value.strip():
        try:
            return parse_iso_utc(value).timestamp()
        except (TypeError, ValueError):
            return None
    return None


def _support_resistance_close_epochs(sr_payload: Dict[str, Any]) -> List[float]:
    epochs: List[float] = []
    structure_epoch = _as_of_epoch(sr_payload.get("structure_as_of"))
    if structure_epoch is not None:
        epochs.append(structure_epoch)
    per_timeframe = sr_payload.get("per_timeframe")
    if isinstance(per_timeframe, list):
        for item in per_timeframe:
            if not isinstance(item, dict):
                continue
            item_as_of = _as_of_epoch(item.get("structure_as_of"))
            if item_as_of is not None:
                epochs.append(item_as_of)
                continue
            window = item.get("window")
            if not isinstance(window, dict):
                continue
            open_epoch = _as_of_epoch(window.get("end"))
            if open_epoch is None:
                continue
            tf = str(item.get("timeframe") or "").strip()
            try:
                epochs.append(bar_close_epoch(open_epoch, tf) if tf else open_epoch)
            except (OverflowError, TypeError, ValueError):
                epochs.append(open_epoch)
    if epochs:
        return epochs
    for key in ("window", "scan_window"):
        window = sr_payload.get(key)
        if not isinstance(window, dict):
            continue
        open_epoch = _as_of_epoch(window.get("end"))
        if open_epoch is None:
            continue
        tf = str(sr_payload.get("timeframe") or "").strip().upper()
        if tf and tf != "AUTO":
            try:
                return [bar_close_epoch(open_epoch, tf)]
            except (OverflowError, TypeError, ValueError):
                return [open_epoch]
        return [open_epoch]
    return epochs


def _confluence_volume_profile_window(
    sr_timeframe: str,
    lookback: int,
) -> tuple[str, int]:
    timeframe = str(sr_timeframe or "H1").strip().upper()
    if timeframe == "AUTO":
        timeframe = max(
            get_auto_support_resistance_timeframes(),
            key=lambda value: int(TIMEFRAME_SECONDS.get(value, 0) or 0),
        )
    seconds = int(TIMEFRAME_SECONDS.get(timeframe, 0) or 0)
    minutes_per_bar = max(1, int(math.ceil(seconds / 60.0)))
    return timeframe, max(1, int(lookback) * minutes_per_bar)


_PIVOT_METHOD_INFO: Dict[str, Dict[str, str]] = {
    "classic": {
        "method_description": "PP=(H+L+C)/3; R/S levels extend arithmetically from the prior bar range.",
        "intended_use": (
            "Timeframe-matched classic pivot context from the last completed source bar; "
            "use D1 for conventional daily floor-trader pivots."
        ),
    },
    "fibonacci": {
        "method_description": "PP=(H+L+C)/3; R/S levels use 0.382, 0.618, and 1.000 range multiples.",
        "intended_use": "Traders who align pivot levels with Fibonacci retracement/extension zones.",
    },
    "camarilla": {
        "method_description": "PP is the prior close; R/S levels use 1.1 * prior range fractions around that reference.",
        "intended_use": "Intraday mean-reversion/breakout context; R3/S3 and R4/S4 are commonly watched.",
    },
    "woodie": {
        "method_description": "PP=(H+L+2*C)/4, weighting the close more heavily than classic pivots.",
        "intended_use": "Close-sensitive intraday pivot context.",
    },
    "demark": {
        "method_description": (
            "Uses the open/close relationship to choose X; R1/S1 are canonical "
            "DeMark levels and PP=X/4 is a common retail-platform extension."
        ),
        "intended_use": "Directional single-level pivot context from the prior bar.",
    },
}


def _resolve_reference_quote(
    gateway: Any,
    symbol: str,
    tick: Any,
    *,
    now_epoch: float,
) -> tuple[Any, Optional[float], Dict[str, Any]]:
    """Resolve one canonical, positive-spread quote for level geometry."""
    resolved_tick, source = resolve_quote_tick(
        gateway,
        symbol,
        tick,
        now_epoch=now_epoch,
    )
    freshness = build_tick_freshness_context(
        symbol,
        tick_epoch=quote_tick_epoch(resolved_tick),
        now_epoch=now_epoch,
        item="reference quote",
    )
    spread = compute_spread_metrics(
        tick_value(resolved_tick, "bid"),
        tick_value(resolved_tick, "ask"),
    )
    context: Dict[str, Any] = {**freshness, **source}
    enforce_quote_execution_readiness(
        context,
        bid=tick_value(resolved_tick, "bid"),
        ask=tick_value(resolved_tick, "ask"),
        quote_source_conflict=source.get("quote_source_conflict"),
    )
    blockers: List[str] = []
    if context.get("spread_valid") is not True:
        blockers.append("invalid_spread")
    if (
        context.get("usable_for_live_trading") is not True
        and isinstance(source.get("quote_source_conflict"), dict)
    ):
        blockers.append("quote_source_conflict")
    if blockers:
        context["execution_blockers"] = blockers
    reference = (
        spread.get("mid")
        if context.get("usable_for_live_trading") is True
        else None
    )
    return resolved_tick, reference, context


def _resolve_support_resistance_timeframes(timeframe: Optional[str]) -> tuple[str, List[str]]:
    raw = str(timeframe or "auto").strip()
    if not raw or raw.lower() == "auto":
        return "auto", list(get_auto_support_resistance_timeframes())
    normalized = raw.upper()
    if normalized not in TIMEFRAME_MAP:
        raise RuntimeError(invalid_timeframe_error(normalized, TIMEFRAME_MAP))
    return normalized, [normalized]


def compute_support_resistance_payload(
    *,
    fetch_history_impl,
    symbol: str,
    timeframe: Optional[str],
    limit: int,
    tolerance_pct: float,
    min_touches: int,
    max_levels: int,
    reaction_bars: int,
    adx_period: int,
    decay_half_life_bars: Optional[int],
    max_distance_pct: Optional[float],
    volume_weighting: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    reference_price: Optional[float] = None,
    reference_price_source: Optional[str] = None,
) -> Dict[str, Any]:
    tolerance_fraction = float(tolerance_pct) / 100.0
    requested_timeframe, timeframes = _resolve_support_resistance_timeframes(timeframe)
    multi_timeframe = len(timeframes) > 1
    results: List[Dict[str, Any]] = []
    errors: List[str] = []
    partial_warnings: List[Dict[str, Any]] = []
    per_timeframe_min_touches = 1 if multi_timeframe else int(min_touches)
    per_timeframe_max_levels = max(int(max_levels), 1) if not multi_timeframe else max(int(max_levels) * 2, 6)

    for tf in timeframes:
        try:
            history_kwargs: Dict[str, Any] = {
                "symbol": symbol,
                "timeframe": tf,
                "need": int(limit),
            }
            if start or end:
                history_kwargs.update({"start": start, "end": end})
            frame = fetch_history_impl(**history_kwargs)
            if frame is None:
                raise RuntimeError("No history available")
            if len(frame) > int(limit):
                frame = frame.iloc[-int(limit):].copy()
            result = compute_support_resistance_levels(
                frame,
                symbol=symbol,
                timeframe=tf,
                limit=int(limit),
                tolerance_fraction=tolerance_fraction,
                min_touches=int(per_timeframe_min_touches),
                max_levels=int(per_timeframe_max_levels),
                reaction_bars=int(reaction_bars),
                adx_period=int(adx_period),
                decay_half_life_bars=None if decay_half_life_bars is None else int(decay_half_life_bars),
                max_distance_pct=None if max_distance_pct is None else float(max_distance_pct),
                volume_weighting=str(volume_weighting),
                reference_price=reference_price,
                reference_price_source=reference_price_source,
            )
            if (result.get("levels") or []) or not multi_timeframe:
                results.append(result)
        except Exception as exc:
            error_text = f"{tf}: {exc}"
            errors.append(error_text)
            partial_warnings.append(
                {
                    "code": "timeframe_failed",
                    "timeframe": tf,
                    "message": error_text,
                }
            )
            if not multi_timeframe:
                raise

    if not results:
        if errors:
            raise RuntimeError("; ".join(errors))
        raise RuntimeError("No history available")

    if not multi_timeframe:
        return results[0]

    merged = merge_support_resistance_results(
        results,
        symbol=symbol,
        timeframe=requested_timeframe,
        limit=int(limit),
        tolerance_fraction=tolerance_fraction,
        min_touches=int(min_touches),
        max_levels=int(max_levels),
        reaction_bars=int(reaction_bars),
        adx_period=int(adx_period),
        decay_half_life_bars=None if decay_half_life_bars is None else int(decay_half_life_bars),
        max_distance_pct=None if max_distance_pct is None else float(max_distance_pct),
        volume_weighting=str(volume_weighting),
    )
    if partial_warnings:
        merged["warnings"] = list(merged.get("warnings") or []) + partial_warnings
    return merged


@mcp.tool()
def pivot_compute_points(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "D1",
    method: Optional[PivotMethodLiteral] = None,
    end: Optional[str] = None,
    as_of: Optional[str] = None,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Compute pivot point levels from the last completed bar on `timeframe`.
    Parameters: symbol, timeframe, method, end, as_of, detail

    Defaults to D1 because daily pivots are the common floor-trader convention.
    Compact detail returns classic pivots while standard/full include every
    supported method. Set `method` to return only one pivot method.
    Pass `end` or `as_of` for a historical cutoff so the source bar is the last
    completed bar at or before that instant (no look-ahead).
    Use `support_resistance_levels` for complementary data-driven levels from
    historical retests and reactions.
    """
    def _run() -> Dict[str, Any]:  # noqa: C901
        try:
            mt5 = create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise)
            mt5.ensure_connection()
            method_filter = str(method).strip().lower() if method is not None else None
            if method_filter and method_filter not in _PIVOT_METHODS:
                return {
                    "error": (
                        f"Invalid pivot method: {method_filter}. "
                        f"Valid methods: {', '.join(_PIVOT_METHODS)}"
                    )
                }
            if timeframe not in TIMEFRAME_MAP:
                return {"error": invalid_timeframe_error(timeframe, TIMEFRAME_MAP)}
            mt5_tf = TIMEFRAME_MAP[timeframe]
            tf_secs = TIMEFRAME_SECONDS.get(timeframe)
            if not tf_secs:
                return {"error": unsupported_timeframe_seconds_error(timeframe)}
            if end not in (None, "") and as_of not in (None, ""):
                return build_error_payload(
                    "end and as_of are aliases and cannot be combined.",
                    code="incompatible_parameters",
                    operation="pivot_compute_points",
                    details={"conflicting_parameters": ["end", "as_of"]},
                    remediation="Pass either end or as_of as the historical cutoff.",
                )
            cutoff_raw = end if end not in (None, "") else as_of
            range_error = validate_historical_range(None, cutoff_raw)
            if range_error is not None:
                return range_error
            historical_cutoff = _parse_end_datetime(cutoff_raw) if cutoff_raw else None
            if cutoff_raw and historical_cutoff is None:
                return {"error": "Invalid end time.", "error_code": "invalid_datetime"}

            with _symbol_ready_guard(symbol) as (err, _info_before):
                if err:
                    return {"error": err}
                system_now_dt = datetime.now(timezone.utc)
                system_now_ts = system_now_dt.timestamp()
                pivot_cutoff_dt = historical_cutoff or system_now_dt
                comparable = (
                    pivot_cutoff_dt.replace(tzinfo=timezone.utc)
                    if getattr(pivot_cutoff_dt, "tzinfo", None) is None
                    else pivot_cutoff_dt
                )
                if comparable.timestamp() > system_now_ts:
                    pivot_cutoff_dt = system_now_dt
                rates = _mt5_copy_rates_from(symbol, mt5_tf, pivot_cutoff_dt, 5)

            if rates is None or len(rates) == 0:
                return {"error": f"Failed to get rates for {symbol}: {mt5.last_error()}"}

            cutoff_ts = (
                historical_cutoff.replace(tzinfo=timezone.utc).timestamp()
                if historical_cutoff is not None
                else system_now_ts
            )
            cutoff_ts = min(float(cutoff_ts), float(system_now_ts))
            src = next(
                (
                    row
                    for row in reversed(rates)
                    if bar_close_epoch(row["time"], timeframe) <= cutoff_ts
                ),
                None,
            )
            if src is None:
                return {"error": "No completed bars available to compute pivot points"}

            H = float(src["high"]) if _has_field(src, "high") else float("nan")
            L = float(src["low"]) if _has_field(src, "low") else float("nan")
            C = float(src["close"]) if _has_field(src, "close") else float("nan")
            O = float(src["open"]) if _has_field(src, "open") else C
            if any(not math.isfinite(v) for v in (H, L, C)):
                return {"error": "Pivot calculation requires high, low, and close prices"}

            period_start = float(src["time"]) if _has_field(src, "time") else float("nan")
            period_end = bar_close_epoch(period_start, timeframe)

            digits = symbol_price_digits(_info_before) if _info_before is not None else 0

            def _round_context(v: float) -> float:
                try:
                    return round(float(v), max(int(digits) + 2, 8))
                except Exception:
                    return float(v)

            rng = H - L
            price_increment = _positive_float_attr(
                _info_before,
                "trade_tick_size",
                "point",
            )
            if price_increment is None and digits >= 0:
                price_increment = 10.0 ** (-int(digits))

            def _degenerate_levels_info(levels: Dict[str, float]) -> Dict[str, Any]:
                values = [
                    float(value)
                    for value in levels.values()
                    if isinstance(value, (int, float)) and math.isfinite(float(value))
                ]
                if not values:
                    return {}
                unique_count = len(set(values))
                reasons: List[str] = []
                if len(values) >= 3 and unique_count < 3:
                    reasons.append(
                        f"Only {unique_count} unique rounded level price(s) remain."
                    )
                if (
                    price_increment is not None
                    and math.isfinite(rng)
                    and rng < 2.0 * price_increment
                ):
                    reasons.append(
                        "Source bar range "
                        f"({_round_context(rng)}) is smaller than 2x price increment "
                        f"({_round_context(price_increment)})."
                    )
                if not reasons:
                    return {}
                return {
                    "levels_degenerate": True,
                    "reason": " ".join(reasons)
                    + " Pivot levels may appear identical after rounding.",
                    "source_range": _round_context(rng),
                    "price_increment": _round_context(price_increment)
                    if price_increment is not None
                    else None,
                    "digits": digits,
                    "unique_level_count": unique_count,
                }

            def _compute_method(method_name: str):
                method_info = compute_pivot_method_levels(
                    method_name,
                    open_price=O,
                    high_price=H,
                    low_price=L,
                    close_price=C,
                    digits=digits,
                )
                if not method_info:
                    return None
                return {
                    **method_info,
                    **_PIVOT_METHOD_INFO.get(str(method_info.get("method")), {}),
                }

            methods_out = []
            levels_by_method: Dict[str, Dict[str, float]] = {}
            pivot_values: Dict[str, float] = {}
            for method_name in _PIVOT_METHODS:
                if method_filter and method_name != method_filter:
                    continue
                method_info = _compute_method(method_name)
                if not method_info:
                    continue
                methods_out.append(method_info)
                levels_by_method[method_info["method"]] = method_info["levels"]
                pivot_val = method_info.get('pivot')
                if isinstance(pivot_val, (int, float)):
                    pivot_values[method_info["method"]] = float(pivot_val)

            method_names = [info["method"] for info in methods_out]
            present_levels = set()
            for info in methods_out:
                for lvl in info["levels"].keys():
                    present_levels.add(str(lvl))
            import re as _re
            rs_nums = set()
            for name in list(present_levels):
                m = _re.match(r"^([RS])(\d+)$", str(name))
                if m:
                    try:
                        rs_nums.add(int(m.group(2)))
                    except Exception:
                        pass
            max_n = max(rs_nums) if rs_nums else 0
            include_pivot_row = bool(pivot_values)
            level_sequence: List[str] = []
            for n in range(max_n, 0, -1):
                rn = f"R{n}"
                if rn in present_levels:
                    level_sequence.append(rn)
            if not include_pivot_row and 'PP' in present_levels:
                level_sequence.append('PP')
            for n in range(1, max_n + 1):
                sn = f"S{n}"
                if sn in present_levels:
                    level_sequence.append(sn)
            consumed = set(level_sequence) | ({'PP'} if include_pivot_row else set())
            leftovers = sorted([lv for lv in present_levels if lv not in consumed])
            level_sequence.extend(leftovers)
            levels_table: List[Dict[str, Any]] = []
            for lvl in level_sequence:
                if not str(lvl).startswith('R'):
                    continue
                row: Dict[str, Any] = {"level": lvl}
                for name in method_names:
                    level_map = levels_by_method.get(name, {})
                    val = level_map.get(lvl)
                    if val is not None:
                        row[name] = val
                levels_table.append(row)
            if include_pivot_row:
                pivot_row: Dict[str, Any] = {"level": "PP"}
                for name in method_names:
                    if name in pivot_values:
                        pivot_row[name] = pivot_values.get(name)
                levels_table.append(pivot_row)
            elif 'PP' in level_sequence:
                row: Dict[str, Any] = {"level": 'PP'}
                for name in method_names:
                    level_map = levels_by_method.get(name, {})
                    val = level_map.get('PP')
                    if val is not None:
                        row[name] = val
                levels_table.append(row)
            for lvl in level_sequence:
                if not str(lvl).startswith('S'):
                    continue
                row: Dict[str, Any] = {"level": lvl}
                for name in method_names:
                    level_map = levels_by_method.get(name, {})
                    val = level_map.get(lvl)
                    if val is not None:
                        row[name] = val
                levels_table.append(row)
            for lvl in leftovers:
                row: Dict[str, Any] = {"level": lvl}
                for name in method_names:
                    level_map = levels_by_method.get(name, {})
                    val = level_map.get(lvl)
                    if val is not None:
                        row[name] = val
                levels_table.append(row)

            _use_ctz = _use_client_tz()
            timezone_label = display_timezone_label(
                use_client_tz=_use_ctz,
                fallback="UTC",
                resolve_client_tz=_resolve_client_tz,
            )
            start_str = _format_time_minimal_local(period_start) if _use_ctz else _format_time_minimal(period_start)
            end_str = _format_time_minimal_local(period_end) if _use_ctz else _format_time_minimal(period_end)
            period_note = None
            if str(timeframe).upper() in CALENDAR_TIMEFRAMES:
                period_note = (
                    "MT5 daily/weekly/monthly bar periods follow broker/server "
                    "session boundaries; UTC timestamps may not align to UTC "
                    "calendar midnight."
                )

            detail_value = str(detail).strip().lower()
            if detail_value in {"summary"}:
                detail_value = "compact"
            elif detail_value not in {"compact", "standard", "full"}:
                detail_value = "compact"

            requested_cutoff = None if cutoff_raw in (None, "") else str(cutoff_raw)
            effective_cutoff = end_str
            payload: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "period": {
                    "start": start_str,
                    "end": end_str,
                },
                "historical_cutoff": {
                    "requested": requested_cutoff,
                    "effective": effective_cutoff,
                },
                "analysis_as_of": effective_cutoff,
                "calculation_basis": {
                    "source_bar": (
                        f"last completed {timeframe} bar at or before {requested_cutoff}"
                        if requested_cutoff
                        else f"last completed {timeframe} bar"
                    ),
                    "session_boundary": "MT5 broker/session calendar",
                    "display_timezone": timezone_label,
                },
                "levels_note": (
                    "null cells mean that pivot method does not define that level. "
                    "Each method's PP follows that method's documented reference convention."
                ),
                "method_descriptions": {
                    name: dict(_PIVOT_METHOD_INFO.get(name, {}))
                    for name in method_names
                },
                "levels": levels_table,
            }
            if str(timeframe).upper() == "D1":
                broker_tz = mt5_config.get_server_tz()
                if broker_tz is not None:
                    payload["period"]["broker_trading_day"] = datetime.fromtimestamp(
                        period_start,
                        tz=broker_tz,
                    ).date().isoformat()
                    payload["period"]["broker_timezone"] = _timezone_object_label(broker_tz)
            if period_note:
                payload["period_note"] = period_note
            payload["timezone"] = timezone_label
            _attach_pivot_trust_metadata(
                payload,
                symbol=symbol,
                timeframe=timeframe,
                last_bar_epoch=period_start,
                now_epoch=cutoff_ts,
            )
            payload["freshness_reference"] = (
                "historical_cutoff"
                if historical_cutoff is not None
                else "retrieval_time"
            )
            if detail_value == "compact":
                compact_method_name = method_filter or "classic"
                selected_method = next(
                    (
                        info
                        for info in methods_out
                        if str(info.get("method")).strip().lower() == compact_method_name
                    ),
                    methods_out[0] if methods_out else None,
                )
                compact_levels = (
                    dict(selected_method.get("levels", {}))
                    if isinstance(selected_method, dict)
                    else {}
                )
                compact_payload: Dict[str, Any] = {
                    "success": True,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "period": payload["period"],
                    "historical_cutoff": payload["historical_cutoff"],
                    "analysis_as_of": payload["analysis_as_of"],
                    "method": (
                        selected_method.get("method")
                        if isinstance(selected_method, dict)
                        else "classic"
                    ),
                    "pivot": (
                        selected_method.get("pivot")
                        if isinstance(selected_method, dict)
                        else None
                    ),
                    "levels": compact_levels,
                }
                if isinstance(selected_method, dict) and selected_method.get(
                    "pivot_convention"
                ):
                    compact_payload["pivot_convention"] = selected_method[
                        "pivot_convention"
                    ]
                if period_note:
                    compact_payload["period_note"] = period_note
                compact_payload["timezone"] = timezone_label
                _attach_pivot_trust_metadata(
                    compact_payload,
                    symbol=symbol,
                    timeframe=timeframe,
                    last_bar_epoch=period_start,
                    now_epoch=cutoff_ts,
                )
                compact_payload["freshness_reference"] = payload[
                    "freshness_reference"
                ]
                degenerate_info = _degenerate_levels_info(compact_payload["levels"])
                if degenerate_info:
                    compact_payload.update(degenerate_info)
                return compact_payload
            payload["detail"] = "full" if detail_value == "full" else "standard"
            if detail_value == "full":
                payload["methods"] = methods_out
            return payload
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error computing pivot points: {str(exc)}"}

    return run_mt5_logged_operation(
        logger,
        operation="pivot_compute_points",
        symbol=symbol,
        timeframe=timeframe,
        method=method,
        end=end,
        as_of=as_of,
        detail=detail,
        func=_run,
    )


@mcp.tool()
def confluence_levels(  # noqa: C901
    symbol: str,
    pivot_timeframe: TimeframeLiteral = "D1",
    sr_timeframe: AutoTimeframeLiteral = "auto",
    lookback: Annotated[int, Field(ge=3, le=20_000)] = 200,
    start: Optional[str] = None,
    end: Optional[str] = None,
    tolerance_pct: Annotated[float, Field(ge=0.0)] = 0.15,
    tolerance_points: Annotated[Optional[float], Field(ge=0.0)] = None,
    min_touches: Annotated[int, Field(ge=1)] = 2,
    max_levels: Annotated[int, Field(ge=1)] = 5,
    side: Literal["both", "above", "below", "support", "resistance"] = "both",
    max_distance_pct: Annotated[Optional[float], Field(ge=0.0)] = 5.0,
    min_source_families: Annotated[int, Field(ge=1)] = 2,
    pivot_method: Optional[PivotMethodLiteral] = None,
    volume_weighting: Literal["off", "auto"] = "off",
    reaction_bars: Annotated[int, Field(ge=1)] = 6,
    adx_period: Annotated[int, Field(ge=2)] = 14,
    decay_half_life_bars: Annotated[Optional[int], Field(gt=0)] = None,
    volume_profile_source: Literal["off", "auto", "ticks", "m1_bars"] = "off",
    volume_profile_max_tick_window_days: Annotated[int, Field(ge=1)] = 1,
    volume_profile_max_ticks: Annotated[int, Field(ge=1)] = 50_000,
    volume_profile_max_m1_bars: Annotated[int, Field(ge=1)] = 20_000,
    detail: Literal["compact", "standard", "full"] = "compact",
) -> Dict[str, Any]:
    """Find nearby high-confluence price zones where pivots, support/resistance, and Fibonacci agree.

    Combines formula pivot levels, touch-derived support/resistance, and
    Fibonacci swing levels. Defaults use daily pivots, auto-timeframe S/R, and
    `min_source_families=2` so a zone needs independent agreement. Pass
    `min_source_families=1` to include single-family clusters.
    Use `support_resistance_levels` for structural levels alone; this tool is
    multi-source consensus of those levels with pivots and Fibonacci.
    """

    def _run() -> Dict[str, Any]:  # noqa: C901
        try:
            range_error = validate_historical_range(start, end)
            if range_error is not None:
                return range_error
            gateway = create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise)
            gateway.ensure_connection()

            pivot_tf = str(pivot_timeframe or "D1").strip().upper()
            sr_tf = str(sr_timeframe or "auto").strip()
            method_filter = str(pivot_method).strip().lower() if pivot_method is not None else None
            if method_filter and method_filter not in _PIVOT_METHODS:
                return {
                    "error": (
                        f"Invalid pivot method: {method_filter}. "
                        f"Valid methods: {', '.join(_PIVOT_METHODS)}"
                    )
                }
            if pivot_tf not in TIMEFRAME_MAP:
                return {"error": invalid_timeframe_error(pivot_tf, TIMEFRAME_MAP)}
            mt5_tf = TIMEFRAME_MAP[pivot_tf]
            tf_secs = TIMEFRAME_SECONDS.get(pivot_tf)
            if not tf_secs:
                return {"error": unsupported_timeframe_seconds_error(pivot_tf)}
            if tolerance_points is not None and float(tolerance_points) < 0.0:
                return {"error": "tolerance_points must be non-negative"}
            if float(tolerance_pct) < 0.0:
                return {"error": "tolerance_pct must be non-negative"}
            historical_cutoff = _parse_end_datetime(end) if end else None
            if end and historical_cutoff is None:
                return {"error": "Invalid end time."}

            with _symbol_ready_guard(symbol) as (err, info_before):
                if err:
                    return {"error": err}
                system_now_dt = datetime.now(timezone.utc)
                system_now_ts = system_now_dt.timestamp()
                tick = mt5.symbol_info_tick(symbol)
                tick, live_reference_price, reference_quote_context = (
                    _resolve_reference_quote(
                        gateway,
                        symbol,
                        tick,
                        now_epoch=system_now_ts,
                    )
                )
                pivot_cutoff_dt = historical_cutoff or system_now_dt
                rates = _mt5_copy_rates_from(symbol, mt5_tf, pivot_cutoff_dt, 5)

            if rates is None or len(rates) == 0:
                return {"error": f"Failed to get rates for {symbol}: {mt5.last_error()}"}

            pivot_cutoff_ts = (
                historical_cutoff.replace(tzinfo=timezone.utc).timestamp()
                if historical_cutoff is not None
                else system_now_ts
            )
            source_bar = next(
                (
                    row
                    for row in reversed(rates)
                    if bar_close_epoch(row["time"], pivot_tf) <= pivot_cutoff_ts
                ),
                None,
            )
            if source_bar is None:
                return {"error": "No completed bars available to compute pivot points"}

            high = float(source_bar["high"]) if _has_field(source_bar, "high") else float("nan")
            low = float(source_bar["low"]) if _has_field(source_bar, "low") else float("nan")
            close = float(source_bar["close"]) if _has_field(source_bar, "close") else float("nan")
            open_ = float(source_bar["open"]) if _has_field(source_bar, "open") else close
            if any(not math.isfinite(value) for value in (high, low, close)):
                return {"error": "Pivot calculation requires high, low, and close prices"}

            digits = symbol_price_digits(info_before) if info_before is not None else 0
            price_increment = _positive_float_attr(info_before, "trade_tick_size", "point")
            if price_increment is None and digits >= 0:
                price_increment = 10.0 ** (-int(digits))

            requested_methods = [method_filter] if method_filter else list(_PIVOT_METHODS)
            pivot_methods = compute_pivot_methods(
                open_price=open_,
                high_price=high,
                low_price=low,
                close_price=close,
                digits=digits,
                methods=requested_methods,
            )
            for method_info in pivot_methods:
                method_name = str(method_info.get("method") or "")
                method_info.update(_PIVOT_METHOD_INFO.get(method_name, {}))

            sr_payload = compute_support_resistance_payload(
                fetch_history_impl=_fetch_history,
                symbol=symbol,
                timeframe=sr_tf,
                limit=int(lookback),
                start=start,
                end=end,
                tolerance_pct=float(tolerance_pct),
                min_touches=int(min_touches),
                max_levels=max(1, int(max_levels)),
                max_distance_pct=None if max_distance_pct is None else float(max_distance_pct),
                volume_weighting=str(volume_weighting),
                reaction_bars=int(reaction_bars),
                adx_period=int(adx_period),
                decay_half_life_bars=None if decay_half_life_bars is None else int(decay_half_life_bars),
            )

            reference_price = (
                None if historical_cutoff is not None else live_reference_price
            )
            reference_price_source = (
                "historical_window_close"
                if historical_cutoff is not None
                else "live_tick_mid"
            )
            if reference_price is None:
                if historical_cutoff is None:
                    reference_price_source = "last_completed_bar_close"
                try:
                    sr_current = sr_payload.get("current_price")
                    reference_price = float(sr_current) if sr_current is not None else None
                except Exception:
                    reference_price = None
            if reference_price is None or not math.isfinite(float(reference_price)):
                reference_price = close
                reference_price_source = "last_completed_bar_close"

            detail_value = str(detail).strip().lower()
            if detail_value in {"summary"}:
                detail_value = "compact"
            volume_profile_payload: Optional[Dict[str, Any]]
            vp_timeframe, derived_max_m1_bars = _confluence_volume_profile_window(
                sr_tf,
                int(lookback),
            )
            effective_max_m1_bars = min(
                int(derived_max_m1_bars),
                int(volume_profile_max_m1_bars),
            )
            if str(volume_profile_source).lower() == "off":
                volume_profile_payload = None
            else:
                volume_window = (
                    {"start": start, "end": end}
                    if start is not None
                    else {
                        "end": end,
                        "timeframe": vp_timeframe,
                        "lookback": int(lookback),
                    }
                )
                volume_profile_payload = compute_volume_profile_payload(
                    symbol=symbol,
                    **volume_window,
                    source=volume_profile_source,
                    price_source="mid",
                    volume_source="auto",
                    bucket_points=None,
                    bucket_count=80,
                    max_buckets=120,
                    value_area_pct=70.0,
                    reference_price=float(reference_price),
                    max_tick_window_days=int(volume_profile_max_tick_window_days),
                    max_ticks=int(volume_profile_max_ticks),
                    max_m1_bars=effective_max_m1_bars,
                    detail="compact",
                )

            payload = build_level_confluence_payload(
                symbol=symbol,
                pivot_timeframe=pivot_tf,
                sr_timeframe=str(sr_payload.get("timeframe") or sr_tf),
                pivot_methods=pivot_methods,
                support_resistance_payload=sr_payload,
                reference_price=float(reference_price),
                tolerance_pct=float(tolerance_pct),
                tolerance_points=tolerance_points,
                price_increment=price_increment,
                max_levels=int(max_levels),
                side=str(side),
                max_distance_pct=None if max_distance_pct is None else float(max_distance_pct),
                min_source_families=max(1, int(min_source_families)),
                detail=detail_value,
                volume_profile_payload=volume_profile_payload,
            )
            payload["reference_price_source"] = reference_price_source
            if historical_cutoff is None:
                payload["reference_quote_usable_for_live_trading"] = bool(
                    reference_quote_context.get("usable_for_live_trading")
                )
                for source_key, target_key in (
                    ("freshness_state", "reference_quote_freshness_state"),
                    ("freshness_reason", "reference_quote_freshness_reason"),
                ):
                    value = reference_quote_context.get(source_key)
                    if value not in (None, ""):
                        payload[target_key] = value
                for key in (
                    "quote_source",
                    "quote_source_state",
                    "spread_quality",
                    "execution_blockers",
                    "quote_source_conflict",
                ):
                    if reference_quote_context.get(key) not in (None, [], {}):
                        payload[key] = reference_quote_context[key]
            payload["volume_profile_status"] = {
                "enabled": str(volume_profile_source).lower() != "off",
                "requested_source": str(volume_profile_source).lower(),
                "max_m1_bars": int(volume_profile_max_m1_bars),
                "effective_max_m1_bars": int(effective_max_m1_bars),
                "status": (
                    "disabled"
                    if volume_profile_payload is None
                    else (
                        "available"
                        if volume_profile_payload.get("success")
                        else "unavailable"
                    )
                ),
            }
            period_start = float(source_bar["time"]) if _has_field(source_bar, "time") else float("nan")
            pivot_close_epoch = (
                bar_close_epoch(period_start, pivot_tf)
                if math.isfinite(period_start)
                else None
            )
            if reference_price_source == "live_tick_mid":
                payload["reference_quote_as_of"] = (
                    format_datetime_utc(datetime.now(timezone.utc))
                )
            elif historical_cutoff is not None:
                sr_close_epochs = _support_resistance_close_epochs(sr_payload)
                if sr_close_epochs:
                    payload["reference_price_as_of"] = format_epoch_utc(
                        max(sr_close_epochs)
                    )
                else:
                    reference_as_of = sr_payload.get("structure_as_of")
                    if reference_as_of is None:
                        scan_window = sr_payload.get("scan_window")
                        if not isinstance(scan_window, dict):
                            scan_window = sr_payload.get("window")
                        if isinstance(scan_window, dict):
                            reference_as_of = scan_window.get("end")
                    payload["reference_price_as_of"] = reference_as_of
                analysis_epochs = list(sr_close_epochs)
                if pivot_close_epoch is not None:
                    analysis_epochs.append(float(pivot_close_epoch))
                if analysis_epochs:
                    payload["analysis_as_of"] = format_epoch_utc(max(analysis_epochs))
                else:
                    payload["analysis_as_of"] = payload.get(
                        "reference_price_as_of"
                    ) or format_datetime_utc(
                        historical_cutoff.replace(tzinfo=timezone.utc)
                    )
            else:
                quote_source = str(
                    reference_quote_context.get("quote_source") or ""
                ).strip()
                freshness_state = str(
                    reference_quote_context.get("freshness_state") or ""
                ).strip()
                freshness_reason = str(
                    reference_quote_context.get("freshness_reason") or ""
                ).strip()
                blockers = [
                    str(value)
                    for value in (reference_quote_context.get("execution_blockers") or [])
                    if str(value or "").strip()
                ]
                rejection = " / ".join(
                    value for value in (freshness_state, freshness_reason) if value
                ) or ", ".join(blockers) or "quote readiness checks failed"
                fallback_reason = (
                    f"live quote rejected: {rejection}"
                    if quote_source
                    else "no live quote available"
                )
                payload.setdefault("warnings", []).append(
                    {
                        "code": "reference_price_fallback_last_close",
                        "message": (
                            "reference_price is the latest completed bar close because the "
                            f"{fallback_reason}; "
                            "the proximity of price to support/resistance reflects the "
                            "analysis window, not a live quote."
                        ),
                    }
                )
            if math.isfinite(period_start):
                _use_ctz = _use_client_tz()
                payload["pivot_period"] = {
                    "start": _format_time_minimal_local(period_start) if _use_ctz else _format_time_minimal(period_start),
                    "end": _format_time_minimal_local(pivot_close_epoch)
                    if _use_ctz
                    else _format_time_minimal(pivot_close_epoch),
                }
                payload["timezone"] = display_timezone_label(
                    use_client_tz=_use_ctz,
                    fallback="UTC",
                    resolve_client_tz=_resolve_client_tz,
                )
            else:
                payload["timezone"] = "UTC"
            if detail_value != "compact":
                payload["calculation_basis"] = {
                    "pivot_source_bar": f"last completed {pivot_tf} bar",
                    "support_resistance_timeframe": str(sr_payload.get("timeframe") or sr_tf),
                    "reference_price": (
                        "historical S/R window close"
                        if historical_cutoff is not None
                        else "latest tick midpoint/last when available, else S/R current price or pivot close"
                    ),
                    "volume_profile": (
                        f"{volume_profile_payload.get('source')} source"
                        if isinstance(volume_profile_payload, dict) and volume_profile_payload.get("success")
                        else "unavailable"
                    ),
                }
            warnings = sr_payload.get("warnings")
            if isinstance(warnings, list) and warnings:
                payload["warnings"] = list(warnings)
            digits_value = symbol_price_digits_optional(info_before)
            if digits_value is not None:
                payload["price_precision"] = digits_value
                payload = _round_level_payload_prices(payload, digits=digits_value)
            return attach_completed_bar_input_policy(payload)
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error computing confluence levels: {str(exc)}"}

    return run_mt5_logged_operation(
        logger,
        operation="confluence_levels",
        symbol=symbol,
        pivot_timeframe=pivot_timeframe,
        sr_timeframe=sr_timeframe,
        lookback=lookback,
        start=start,
        end=end,
        tolerance_pct=tolerance_pct,
        tolerance_points=tolerance_points,
        min_touches=min_touches,
        max_levels=max_levels,
        max_distance_pct=max_distance_pct,
        min_source_families=min_source_families,
        pivot_method=pivot_method,
        volume_weighting=volume_weighting,
        reaction_bars=reaction_bars,
        adx_period=adx_period,
        decay_half_life_bars=decay_half_life_bars,
        volume_profile_source=volume_profile_source,
        volume_profile_max_tick_window_days=volume_profile_max_tick_window_days,
        volume_profile_max_ticks=volume_profile_max_ticks,
        volume_profile_max_m1_bars=volume_profile_max_m1_bars,
        detail=detail,
        func=_run,
    )


@mcp.tool()
def support_resistance_levels(  # noqa: C901
    symbol: str,
    timeframe: AutoTimeframeLiteral = "H1",
    lookback: Annotated[int, Field(ge=3, le=20_000)] = 200,
    start: Optional[str] = None,
    end: Optional[str] = None,
    tolerance_pct: Annotated[float, Field(ge=0.0)] = 0.15,
    min_touches: Annotated[int, Field(ge=1)] = 2,
    max_levels: Annotated[int, Field(ge=1)] = 4,
    max_distance_pct: Annotated[Optional[float], Field(ge=0.0)] = 5.0,
    volume_weighting: Literal["off", "auto"] = "off",
    reaction_bars: Annotated[int, Field(ge=1)] = 6,
    adx_period: Annotated[int, Field(ge=2)] = 14,
    decay_half_life_bars: Annotated[Optional[int], Field(gt=0)] = None,
    detail: Literal["compact", "standard", "full"] = "compact",
) -> Dict[str, Any]:
    """Detect support/resistance levels around the current price from historical structure.

    Set `timeframe="auto"` to merge levels from M15, H1, H4, and D1.
    `lookback` caps the historical bars used to detect levels after applying
    any optional `start`/`end` time window.
    Use `detail="compact"` for the nearest-level summary, `detail="standard"`
    for compact actionable supports/resistances/levels plus Fibonacci swing
    levels, and `detail="full"` for the raw diagnostic payload. The default
    `max_distance_pct=5.0` keeps returned levels near current price.
    Level `type` reflects current price geometry; `dominant_source` reflects
    whether historical tests mostly behaved as support or resistance.
    Use `pivot_compute_points` for complementary formula-based PP/R/S levels
    from the last completed OHLC bar.
    Use `confluence_levels` to cluster these levels with pivots and Fibonacci
    into scored consensus zones.

    Score combines:
    - repeated tests of a level
    - bounce strength after each test (normalized by ATR)
    - pre-test ADX trend strength
    - exponential time decay so recent tests matter more
    - ATR-filtered Fibonacci retracement/extension levels from the most relevant completed swing
    """

    def _run() -> Dict[str, Any]:
        try:
            if int(lookback) < 3:
                return build_error_payload(
                    "Need at least 3 bars to compute support/resistance levels.",
                    code="insufficient_data",
                    operation="support_resistance_levels",
                    details={
                        "parameter": "lookback",
                        "received": int(lookback),
                        "required_minimum": 3,
                    },
                    remediation="Increase lookback to at least 3 bars.",
                    example="--lookback 200",
                )
            range_error = validate_historical_range(start, end)
            if range_error is not None:
                return range_error
            gateway = create_mt5_gateway(
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            gateway.ensure_connection()
            resolved_symbol, symbol_input = resolve_public_symbol(
                symbol,
                gateway=gateway,
            )
            symbol_info = gateway.symbol_info(resolved_symbol)
            digits_value = symbol_price_digits_optional(symbol_info)
            reference_price = None
            reference_price_source = None
            reference_quote_as_of = None
            reference_quote_context: Dict[str, Any] = {}
            if not start and not end:
                raw_tick = gateway.symbol_info_tick(resolved_symbol)
                tick, tick_price, reference_quote_context = _resolve_reference_quote(
                    gateway,
                    resolved_symbol,
                    raw_tick,
                    now_epoch=datetime.now(timezone.utc).timestamp(),
                )
                resolved_epoch = quote_tick_epoch(tick)
                if resolved_epoch is not None:
                    reference_quote_as_of = _format_time_minimal(resolved_epoch)
                if (
                    tick_price is not None
                    and reference_quote_context.get("usable_for_live_trading") is True
                ):
                    reference_price = tick_price
                    reference_price_source = "live_tick_mid"
            result = compute_support_resistance_payload(
                fetch_history_impl=_fetch_history,
                symbol=resolved_symbol,
                timeframe=timeframe,
                limit=int(lookback),
                start=start,
                end=end,
                tolerance_pct=float(tolerance_pct),
                min_touches=int(min_touches),
                max_levels=int(max_levels),
                max_distance_pct=None if max_distance_pct is None else float(max_distance_pct),
                volume_weighting=str(volume_weighting),
                reaction_bars=int(reaction_bars),
                adx_period=int(adx_period),
                decay_half_life_bars=None if decay_half_life_bars is None else int(decay_half_life_bars),
                reference_price=reference_price,
                reference_price_source=reference_price_source,
            )
            current_price_source = str(
                result.get("current_price_source")
                or "last_completed_bar_close"
            )
            if current_price_source == "live_tick_mid":
                result["current_price_as_of"] = reference_quote_as_of
                result["current_price_time_basis"] = "quote_time"
            else:
                result["current_price_as_of"] = result.get("structure_as_of")
                result["current_price_time_basis"] = "completed_bar_close_time"
            if reference_quote_as_of is not None:
                result["reference_quote_as_of"] = reference_quote_as_of
            if reference_quote_context:
                result["reference_quote_usable_for_live_trading"] = bool(
                    reference_quote_context.get("usable_for_live_trading")
                )
                for source_key, target_key in (
                    ("freshness_state", "reference_quote_freshness_state"),
                    ("freshness_reason", "reference_quote_freshness_reason"),
                ):
                    value = reference_quote_context.get(source_key)
                    if value is not None:
                        result[target_key] = value
                for key in (
                    "quote_source",
                    "quote_source_state",
                    "spread_quality",
                    "execution_blockers",
                    "quote_source_conflict",
                ):
                    if reference_quote_context.get(key) not in (None, [], {}):
                        result[f"reference_{key}"] = reference_quote_context[key]
                if (
                    reference_price is None
                    and result.get("current_price_source")
                    == "last_completed_bar_close"
                ):
                    result["reference_price_warning_code"] = (
                        "reference_price_fallback_last_close"
                    )
                    result.setdefault("warnings", []).append(
                        {
                            "code": "reference_price_fallback_last_close",
                            "message": (
                                "The latest quote was not usable for live trading, so "
                                "distances and nearest-level ordering use the latest "
                                "completed bar close."
                            ),
                        }
                    )
            result["lookback_bars"] = int(lookback)
            if isinstance(result.get("warnings"), list):
                result["warnings"] = _normalize_warning_list(result["warnings"])
            detail_value = str(detail).strip().lower()
            if detail_value in {"summary"}:
                detail_value = "compact"
            if detail_value == "compact":
                payload = compact_support_resistance_payload(result)
            elif detail_value == "standard":
                payload = standard_support_resistance_payload(result)
            else:
                detail_value = "full"
                payload = full_support_resistance_payload(result)
            if reference_quote_context:
                for key in (
                    "quote_source",
                    "quote_source_state",
                    "spread_quality",
                    "execution_blockers",
                    "quote_source_conflict",
                ):
                    if reference_quote_context.get(key) not in (None, [], {}):
                        payload[f"reference_{key}"] = reference_quote_context[key]
            payload["detail"] = detail_value
            payload["symbol"] = resolved_symbol
            if symbol_input is not None:
                payload["symbol_input"] = symbol_input
            payload.setdefault("timezone", "UTC")
            if digits_value is not None:
                payload["price_precision"] = digits_value
                payload = _round_level_payload_prices(payload, digits=digits_value)
            return attach_completed_bar_input_policy(payload)
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error computing support/resistance levels: {str(exc)}"}

    return run_mt5_logged_operation(
        logger,
        operation="support_resistance_levels",
        symbol=symbol,
        timeframe=timeframe,
        lookback=lookback,
        start=start,
        end=end,
        tolerance_pct=tolerance_pct,
        min_touches=min_touches,
        max_levels=max_levels,
        max_distance_pct=max_distance_pct,
        volume_weighting=volume_weighting,
        reaction_bars=reaction_bars,
        adx_period=adx_period,
        decay_half_life_bars=decay_half_life_bars,
        detail=detail,
        func=_run,
    )
