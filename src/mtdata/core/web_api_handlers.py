"""Endpoint orchestration helpers for the Web API transport."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, NoReturn, Optional

from fastapi import HTTPException

from ..forecast.exceptions import ForecastError
from ..forecast.forecast_methods import get_forecast_methods_payload
from ..shared.constants import DEFAULT_ROW_LIMIT
from ..utils.coercion import UNPARSED_BOOL, parse_bool_like
from ..utils.denoise import DenoiseCausalityError
from ..utils.mt5 import MT5ConnectionError
from ..utils.utils import parse_kv_or_json
from ._mcp_tools import shape_public_tool_output
from .data.requests import DATA_FETCH_CANDLES_DEFAULT_LIMIT, DataFetchCandlesRequest
from .data.use_cases import run_data_fetch_candles
from .error_envelope import build_error_payload, normalize_error_payload
from .mt5_gateway import create_mt5_gateway
from .output_contract import (
    ensure_common_meta,
)
from .tool_calling import resolve_sync_tool_result

logger = logging.getLogger(__name__)
_MAX_DENOISE_PARAMS_CHARS = 4096
_HISTORY_DENOISE_CONTROL_KEYS = frozenset(
    {"columns", "when", "causality", "keep_original"}
)


def _http_status_for_error(payload: Dict[str, Any], *, default: int = 400) -> int:
    """Map a canonical domain error to HTTP without changing its identity."""
    code = str(payload.get("error_code") or "").strip().lower()
    if code == "symbol_not_found":
        return 404
    if code in {
        "market_ticker_mt5_connection",
        "market_ticker_tick_unavailable",
        "market_ticker_quote_unavailable",
        "mt5_connection_error",
    } or (
        "mt5" in code
        and any(token in code for token in ("connection", "unavailable", "quote"))
    ):
        return 503
    return default


def _http_error(
    status_code: int,
    message: Any,
    *,
    code: str,
    operation: str,
    details: Optional[Dict[str, Any]] = None,
) -> HTTPException:
    if isinstance(message, dict) and isinstance(message.get("error"), str):
        payload = normalize_error_payload(
            message,
            default_code=code,
            operation=operation,
        )
        if details and "details" not in payload:
            payload["details"] = dict(details)
    else:
        payload = build_error_payload(
            message,
            code=code,
            operation=operation,
            details=details,
        )
    log_fn = logger.error if status_code >= 500 else logger.warning
    log_fn(
        "transport=web_api operation=%s request_id=%s status=%s error=%s",
        operation,
        payload["request_id"],
        status_code,
        payload["error"],
    )
    return HTTPException(status_code=status_code, detail=payload)


def _raise_tool_error(
    result: Any,
    *,
    operation: str,
    default_code: str,
    default_status: int = 400,
    invalid_message: Optional[str] = None,
) -> None:
    """Raise the shared HTTP error for a structured tool failure."""
    if isinstance(result, dict) and result.get("error"):
        raise _http_error(
            _http_status_for_error(result, default=default_status),
            result,
            code=str(result.get("error_code") or default_code),
            operation=operation,
        )
    if invalid_message is not None and not isinstance(result, dict):
        raise _http_error(
            500,
            invalid_message,
            code=default_code,
            operation=operation,
        )


def _raise_history_fetch_error(exc: Exception) -> NoReturn:
    if isinstance(exc, MT5ConnectionError):
        raise _http_error(
            503,
            str(exc),
            code="history_mt5_unavailable",
            operation="get_history",
        )
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        raise _http_error(
            400,
            f"history fetch failed: {exc}",
            code="history_fetch_failed",
            operation="get_history",
        )
    logger.exception("transport=web_api operation=get_history unhandled_exception")
    raise _http_error(
        500,
        "History fetch failed.",
        code="history_fetch_internal_error",
        operation="get_history",
    )


def _raise_internal_handler_error(*, operation: str, code: str, message: str) -> NoReturn:
    logger.exception("transport=web_api operation=%s unhandled_exception", operation)
    raise _http_error(500, message, code=code, operation=operation)


def _require_mt5_connection() -> None:
    mt5 = create_mt5_gateway()
    try:
        mt5.ensure_connection()
    except MT5ConnectionError as exc:
        raise _http_error(
            503,
            str(exc),
            code="mt5_connection_error",
            operation="require_mt5_connection",
        )


def _history_denoise_bool(value: Any, *, field_name: str) -> bool:
    parsed = parse_bool_like(value)
    if parsed is UNPARSED_BOOL:
        raise _http_error(
            400,
            f"denoise_params.{field_name} must be a boolean value.",
            code="denoise_params_invalid",
            operation="get_history",
        )
    return bool(parsed)


def _history_denoise_choice(value: Any, *, field_name: str, allowed: set[str]) -> str:
    if not isinstance(value, str):
        raise _http_error(
            400,
            f"denoise_params.{field_name} must be a string.",
            code="denoise_params_invalid",
            operation="get_history",
        )
    normalized = value.strip().lower()
    if normalized not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise _http_error(
            400,
            f"denoise_params.{field_name} must be one of: {allowed_text}.",
            code="denoise_params_invalid",
            operation="get_history",
        )
    return normalized


def _history_denoise_params_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _http_error(
            400,
            "denoise_params.params must be a JSON object.",
            code="denoise_params_invalid",
            operation="get_history",
        )
    return dict(value)


def _history_denoise_columns(value: Any) -> List[str]:
    if isinstance(value, str):
        columns = [col.strip() for col in value.split(",") if col.strip()]
    elif isinstance(value, list):
        columns = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise _http_error(
                    400,
                    f"denoise_params.columns[{index}] must be a string.",
                    code="denoise_params_invalid",
                    operation="get_history",
                )
            name = item.strip()
            if name:
                columns.append(name)
    else:
        raise _http_error(
            400,
            "denoise_params.columns must be a string or list of strings.",
            code="denoise_params_invalid",
            operation="get_history",
        )
    if not columns:
        raise _http_error(
            400,
            "denoise_params.columns must contain at least one column name.",
            code="denoise_params_invalid",
            operation="get_history",
        )
    return columns


def _apply_history_denoise_controls(
    spec_input: Dict[str, Any],
    controls: Dict[str, Any],
) -> None:
    if "columns" in controls:
        spec_input["columns"] = _history_denoise_columns(controls["columns"])
    if "when" in controls:
        spec_input["when"] = _history_denoise_choice(
            controls["when"],
            field_name="when",
            allowed={"post_ti", "pre_ti"},
        )
    if "causality" in controls:
        spec_input["causality"] = _history_denoise_choice(
            controls["causality"],
            field_name="causality",
            allowed={"causal", "zero_phase"},
        )
    if "keep_original" in controls:
        spec_input["keep_original"] = _history_denoise_bool(
            controls["keep_original"],
            field_name="keep_original",
        )


def get_instruments_response(
    *,
    search: Optional[str],
    limit: Optional[int],
    symbols_list_tool: Any,
    call_tool_raw: Callable[[Any], Any],
) -> Dict[str, Any]:
    tool = call_tool_raw(symbols_list_tool)
    result = resolve_sync_tool_result(
        tool(
            search_term=search,
            limit=int(limit) if limit is not None else DEFAULT_ROW_LIMIT,
            detail="compact",
        )
    )
    _raise_tool_error(
        result,
        operation="get_instruments",
        default_code="symbols_list_failed",
        invalid_message="Unexpected symbol catalog payload.",
    )
    items = [
        {
            "symbol": row.get("symbol"),
            "group": row.get("group"),
            "description": row.get("description"),
        }
        for row in result.get("data", [])
        if isinstance(row, dict)
    ]
    return {
        "items": items,
        "count": len(items),
        "pagination": result.get("pagination"),
    }


def _compact_forecast_method_definition(method_def: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in (
        "method",
        "available",
        "requires",
        "category",
        "description",
        "supports_ci",
        "params",
    ):
        value = method_def.get(key)
        if value is not None:
            out[key] = value
    return out


def get_methods_response(
    *,
    get_methods_impl: Callable[[], Any],
    detail: str = "compact",
) -> Dict[str, Any]:
    data = get_methods_impl()
    if not isinstance(data, dict) or data.get("methods") is None:
        return {"methods": []}
    methods = data.get("methods")
    if not isinstance(methods, list):
        return {"methods": []}
    try:
        payload = get_forecast_methods_payload(method_data=data)
    except Exception:
        return data
    if detail != "compact":
        return payload
    methods_payload = payload.get("methods")
    if not isinstance(methods_payload, list):
        return {"methods": []}
    out = dict(payload)
    out["methods"] = [
        _compact_forecast_method_definition(method_def)
        for method_def in methods_payload
        if isinstance(method_def, dict)
    ]
    out["detail"] = "compact"
    return out


def get_models_response(
    *,
    get_models_impl: Callable[..., Any],
    method: Optional[str],
    detail: str = "compact",
) -> Dict[str, Any]:
    data = get_models_impl(method=method, detail=detail)
    if not isinstance(data, dict):
        return {"success": True, "detail": detail, "count": 0, "models": []}
    models = data.get("models")
    if not isinstance(models, list):
        return {"success": True, "detail": detail, "count": 0, "models": []}
    return data


def get_vol_methods_response(*, get_vol_methods: Callable[[], Any]) -> Dict[str, Any]:
    data = get_vol_methods()
    if not isinstance(data, dict):
        return {"methods": []}
    return data


def get_denoise_methods_response(*, get_denoise_methods: Callable[[], Any]) -> Dict[str, Any]:
    data = get_denoise_methods()
    if isinstance(data, dict) and data.get("methods") is not None:
        return data
    return {"methods": []}


def get_dimred_methods_response(*, list_dimred_methods: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    base = list_dimred_methods()
    return {
        "methods": [
            {
                "method": name,
                "available": bool(info.get("available")),
                "description": info.get("description"),
                "params": list(info.get("params") or []),
            }
            for name, info in base.items()
        ]
    }


def get_wavelets_response() -> Dict[str, Any]:
    try:
        import pywt  # type: ignore
    except Exception:
        return {"available": False, "families": [], "wavelets": [], "by_family": {}}
    try:
        families = list(pywt.families())  # type: ignore[attr-defined]
    except Exception:
        families = []
    by_family: Dict[str, List[str]] = {}
    flat: List[str] = []
    if families:
        for family in families:
            names: List[str] = []
            try:
                names = list(pywt.wavelist(family))  # type: ignore[attr-defined]
            except Exception:
                try:
                    names = list(pywt.wavelist(family, kind="discrete"))  # type: ignore[attr-defined]
                except Exception:
                    names = []
            by_family[family] = names
            for wavelet in names:
                if wavelet not in flat:
                    flat.append(wavelet)
    else:
        try:
            flat = list(pywt.wavelist(kind="discrete"))  # type: ignore[attr-defined]
        except Exception:
            try:
                flat = list(pywt.wavelist())  # type: ignore[attr-defined]
            except Exception:
                flat = []
    return {"available": True, "families": families, "wavelets": flat, "by_family": by_family}


def get_history_response(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    limit: Optional[int],
    start: Optional[str],
    end: Optional[str],
    ohlcv: Optional[str],
    include_spread: bool,
    include_incomplete: bool,
    allow_stale: bool,
    indicators: Optional[str],
    timestamp_format: str,
    detail: str,
    denoise_method: Optional[str],
    denoise_params: Optional[str],
    fetch_candles_impl: Callable[..., Any],
    get_denoise_methods: Callable[[], Any],
    normalize_denoise_spec: Callable[..., Any],
    gateway: Any,
    mt5_config: Any,
) -> Dict[str, Any]:
    _require_mt5_connection()
    denoise_method_val = denoise_method.strip() if isinstance(denoise_method, str) else None
    denoise_params_val = denoise_params if isinstance(denoise_params, str) else None

    denoise_spec: Optional[Dict[str, Any]] = None
    if denoise_method_val:
        try:
            meta = get_denoise_methods()
            if isinstance(meta, dict):
                methods = {method.get("method"): method for method in (meta.get("methods") or [])}
                method_meta = methods.get(denoise_method_val)
                if not method_meta or not bool(method_meta.get("available", True)):
                    req = method_meta.get("requires") if method_meta else ""
                    suffix = f"Requires {req}" if req else ""
                    raise _http_error(
                        400,
                        f"Denoise method '{denoise_method_val}' is not available. {suffix}".strip(),
                        code="denoise_method_unavailable",
                        operation="get_history",
                    )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning(
                "transport=web_api operation=get_history denoise_validation_failed error=%s",
                exc,
            )
            raise _http_error(
                500,
                "Denoise method validation failed.",
                code="denoise_validation_failed",
                operation="get_history",
            )

        spec_input: Dict[str, Any] = {
            "method": denoise_method_val,
            "when": "post_ti",
            "columns": ["close"],
            "keep_original": True,
            "suffix": "_dn",
            "params": {},
        }
        if denoise_params_val:
            if len(denoise_params_val) > _MAX_DENOISE_PARAMS_CHARS:
                raise _http_error(
                    400,
                    f"denoise_params exceeds {_MAX_DENOISE_PARAMS_CHARS} characters.",
                    code="denoise_params_too_large",
                    operation="get_history",
                    details={"max_chars": _MAX_DENOISE_PARAMS_CHARS},
                )
            stripped_params = denoise_params_val.strip()
            if stripped_params.startswith(("{", "[")):
                try:
                    decoded = json.loads(stripped_params)
                except (TypeError, ValueError):
                    decoded = None
                if not isinstance(decoded, dict):
                    raise _http_error(
                        400,
                        "denoise_params must be a JSON object or key=value pairs.",
                        code="denoise_params_invalid",
                        operation="get_history",
                    )
            try:
                payload = parse_kv_or_json(stripped_params)
            except ValueError as exc:
                raise _http_error(
                    400,
                    str(exc),
                    code="denoise_params_invalid",
                    operation="get_history",
                ) from exc
            if not payload and stripped_params != "{}":
                raise _http_error(
                    400,
                    "denoise_params must be a JSON object or key=value pairs.",
                    code="denoise_params_invalid",
                    operation="get_history",
                )
            if "params" in payload:
                spec_input["params"] = _history_denoise_params_dict(payload.pop("params"))
            else:
                spec_input["params"] = {
                    key: value
                    for key, value in payload.items()
                    if key not in _HISTORY_DENOISE_CONTROL_KEYS
                }
            _apply_history_denoise_controls(spec_input, payload)
        try:
            denoise_spec = normalize_denoise_spec(spec_input, default_when="pre_ti")
        except DenoiseCausalityError as exc:
            # Non-causal filters (e.g. l1_trend) require explicit zero_phase opt-in.
            # Surface as a client error, not an unhandled 500.
            raise _http_error(
                400,
                (
                    f"Denoise method '{exc.method}' requires explicit zero-phase "
                    "opt-in because it uses future bars."
                ),
                code="denoise_non_causal_requires_opt_in",
                operation="get_history",
                details={
                    "method": exc.method,
                    "required_causality": "zero_phase",
                    "uses_future_bars": True,
                    "remediation": (
                        "Pass denoise_params causality=zero_phase for retrospective "
                        "chart analysis, or choose a causal denoise method."
                    ),
                },
            ) from exc

    request_values: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "ohlcv": ohlcv,
        "include_spread": include_spread,
        "indicators": indicators,
        "denoise": denoise_spec,
        "include_incomplete": include_incomplete,
        "allow_stale": allow_stale,
        "timestamp_format": timestamp_format,
        "detail": detail,
    }
    # The Web API deliberately keeps a small default page for every history
    # query. Passing it explicitly prevents the core tool's omitted-limit
    # bounded-range contract from expanding an HTTP request to its safety cap.
    request_values["limit"] = (
        DATA_FETCH_CANDLES_DEFAULT_LIMIT if limit is None else int(limit)
    )
    try:
        request = DataFetchCandlesRequest(**request_values)
        result = run_data_fetch_candles(
            request,
            gateway=gateway,
            fetch_candles_impl=fetch_candles_impl,
        )
    except Exception as exc:
        _raise_history_fetch_error(exc)

    _raise_tool_error(
        result,
        operation="get_history",
        default_code="history_tool_error",
        invalid_message="Unexpected history payload",
    )

    payload = ensure_common_meta(
        result,
        tool_name="data_fetch_candles",
        mt5_config=mt5_config,
    )
    shape_detail = detail
    if shape_detail != "full":
        meta = payload.get("meta")
        if isinstance(meta, dict):
            runtime = meta.get("runtime")
            timezone_meta = runtime.get("timezone") if isinstance(runtime, dict) else None
            server_meta = timezone_meta.get("server") if isinstance(timezone_meta, dict) else None
            if isinstance(server_meta, dict) and server_meta.get("offset_seconds") is not None:
                payload["server_utc_offset_seconds"] = server_meta["offset_seconds"]
            if isinstance(server_meta, dict) and server_meta.get("tz"):
                payload["server_timezone"] = server_meta["tz"]
    shaped = shape_public_tool_output(
        payload,
        detail=shape_detail,
        tool_name="data_fetch_candles",
    )
    if shape_detail != "full" and isinstance(shaped, dict):
        # The chart endpoint has a small, stable DTO of its own.  These fields
        # drive rendering and paging in the Web UI; they are not restored to
        # generic MCP/CLI candle output.
        rows = shaped.get("data")
        if isinstance(rows, list):
            shaped["count"] = len(rows)
        for key in (
            "data_as_of",
            "data_as_of_basis",
            "indicator_columns",
            "indicators_spec",
            "timestamp_format",
        ):
            value = payload.get(key)
            if value not in (None, "", [], {}):
                shaped[key] = value
        forming_status = payload.get("forming_candle_status")
        if forming_status not in (None, ""):
            shaped["forming_candle_status"] = forming_status
    return shaped


def get_pivots_response(
    *,
    symbol: str,
    timeframe: str,
    method: str,
    detail: str,
    pivot_tool: Any,
    call_tool_raw: Callable[[Any], Any],
) -> Dict[str, Any]:
    tool = call_tool_raw(pivot_tool)
    method_key = str(method).lower().strip()
    try:
        result = resolve_sync_tool_result(
            tool(
                symbol=symbol,
                timeframe=timeframe,
                method=method_key,
                detail=detail,
            )
        )
    except TypeError:
        result = resolve_sync_tool_result(
            pivot_tool(
                symbol=symbol,
                timeframe=timeframe,
                method=method_key,
                detail=detail,
            )
        )
    except Exception as exc:
        raise _http_error(
            500,
            f"pivot compute failed: {exc}",
            code="pivot_compute_failed",
            operation="get_pivots",
        )

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            raise _http_error(
                500,
                "Unexpected pivot output format",
                code="pivot_output_invalid",
                operation="get_pivots",
            )

    _raise_tool_error(result, operation="get_pivots", default_code="pivot_tool_error")
    if not isinstance(result, dict):
        raise _http_error(
            500,
            "Pivot tool returned non-JSON payload",
            code="pivot_payload_invalid",
            operation="get_pivots",
        )

    levels = []
    raw_levels = result.get("levels", []) or []
    if isinstance(raw_levels, dict):
        for level_name, value in raw_levels.items():
            try:
                levels.append({"level": str(level_name), "value": float(value)})
            except (TypeError, ValueError):
                continue
    elif isinstance(raw_levels, list):
        for row in raw_levels:
            if not isinstance(row, dict):
                continue
            level_name = row.get("level") or row.get("Level")
            value = row.get(method_key)
            if level_name is None or value is None:
                continue
            try:
                levels.append({"level": str(level_name), "value": float(value)})
            except (TypeError, ValueError):
                continue
    if not levels:
        raise _http_error(
            404,
            f"No pivot levels for method {method}",
            code="pivot_levels_missing",
            operation="get_pivots",
        )
    payload = dict(result)
    payload.update({
        "levels": levels,
        "period": result.get("period"),
        "symbol": result.get("symbol", symbol),
        "timeframe": result.get("timeframe", timeframe),
        "method": method_key,
    })
    return shape_public_tool_output(
        payload,
        detail=detail,
        tool_name="pivot_compute_points",
    )


def get_support_resistance_response(
    *,
    symbol: str,
    timeframe: str,
    lookback: Optional[int],
    tolerance_pct: float,
    min_touches: int,
    max_levels: int,
    max_distance_pct: Optional[float],
    volume_weighting: str,
    reaction_bars: int,
    adx_period: int,
    decay_half_life_bars: Optional[int],
    detail: str,
    support_resistance_tool: Any,
    call_tool_raw: Callable[[Any], Any],
) -> Dict[str, Any]:
    effective_lookback = int(lookback if lookback is not None else 200)
    tool = call_tool_raw(support_resistance_tool)
    try:
        result = resolve_sync_tool_result(tool(
            symbol=symbol,
            timeframe=timeframe,
            lookback=effective_lookback,
            tolerance_pct=float(tolerance_pct),
            min_touches=int(min_touches),
            max_levels=int(max_levels),
            max_distance_pct=None if max_distance_pct is None else float(max_distance_pct),
            volume_weighting=str(volume_weighting),
            reaction_bars=int(reaction_bars),
            adx_period=int(adx_period),
            decay_half_life_bars=None if decay_half_life_bars is None else int(decay_half_life_bars),
            detail=detail,
        ))
    except Exception as exc:
        message = str(exc)
        status_code = 404 if "No history available" in message else 400
        detail = message if status_code == 404 else f"history fetch failed: {message}"
        raise _http_error(
            status_code,
            detail,
            code="support_resistance_history_failed" if status_code != 404 else "support_resistance_history_missing",
            operation="get_support_resistance",
        )

    _raise_tool_error(
        result,
        operation="get_support_resistance",
        default_code="support_resistance_failed",
    )
    if not _support_resistance_has_levels(result):
        raise _http_error(
            404,
            "No support/resistance levels detected",
            code="support_resistance_levels_missing",
            operation="get_support_resistance",
        )
    return result


def _support_resistance_has_levels(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    for key in ("supports", "resistances", "levels"):
        value = result.get(key)
        if isinstance(value, list) and value:
            return True
    counts = result.get("level_counts")
    if isinstance(counts, dict):
        try:
            return int(counts.get("total") or 0) > 0
        except (TypeError, ValueError):
            return False
    return False


def get_tick_response(
    *,
    symbol: str,
    detail: str,
    market_ticker_tool: Any,
    call_tool_raw: Callable[[Any], Any],
) -> Dict[str, Any]:
    tool = call_tool_raw(market_ticker_tool)
    try:
        result = resolve_sync_tool_result(tool(symbol=symbol, detail=detail))
    except Exception as exc:
        raise _http_error(
            500,
            f"Tick lookup failed: {exc}",
            code="tick_lookup_failed",
            operation="get_tick",
        ) from exc
    if not isinstance(result, dict):
        raise _http_error(
            500,
            "Unexpected tick payload",
            code="tick_payload_invalid",
            operation="get_tick",
        )
    _raise_tool_error(result, operation="get_tick", default_code="tick_data_missing")
    shaped = shape_public_tool_output(
        result,
        detail=detail,
        tool_name="market_ticker",
    )
    if detail != "full" and isinstance(shaped, dict):
        # The tick endpoint feeds chart cursors, so retain its single display
        # timestamp without restoring generic quote telemetry.
        for key in ("time", "time_epoch"):
            if result.get(key) not in (None, ""):
                shaped[key] = result[key]
    return shaped


def _post_use_case_response(
    *,
    body: Any,
    use_case: Callable[..., Any],
    operation: str,
    domain_error_code: str,
    mt5_error_code: str,
    internal_error_code: str,
    internal_message: str,
    result_error_code: str,
) -> Dict[str, Any]:
    try:
        result = use_case(body)
    except HTTPException:
        raise
    except ForecastError as exc:
        raise _http_error(
            400,
            str(exc),
            code=domain_error_code,
            operation=operation,
        )
    except MT5ConnectionError as exc:
        raise _http_error(
            503,
            str(exc),
            code=mt5_error_code,
            operation=operation,
        )
    except Exception:
        _raise_internal_handler_error(
            operation=operation,
            code=internal_error_code,
            message=internal_message,
        )
    _raise_tool_error(
        result,
        operation=operation,
        default_code=result_error_code,
    )
    return result


def post_forecast_price_response(*, body: Any, forecast_generate_use_case: Callable[..., Any]) -> Dict[str, Any]:
    return _post_use_case_response(
        body=body,
        use_case=forecast_generate_use_case,
        operation="post_forecast_price",
        domain_error_code="forecast_error",
        mt5_error_code="forecast_mt5_unavailable",
        internal_error_code="forecast_internal_error",
        internal_message="Forecast computation failed.",
        result_error_code="forecast_tool_error",
    )


def post_trade_idea_response(*, body: Any, compose_impl: Callable[..., Any]) -> Dict[str, Any]:
    return _post_use_case_response(
        body=body,
        use_case=compose_impl,
        operation="post_trade_idea",
        domain_error_code="trade_idea_error",
        mt5_error_code="trade_idea_mt5_unavailable",
        internal_error_code="trade_idea_internal_error",
        internal_message="Trade idea composition failed.",
        result_error_code="trade_idea_tool_error",
    )


def post_forecast_volatility_response(*, body: Any, forecast_vol_impl: Callable[..., Any]) -> Dict[str, Any]:
    return _post_use_case_response(
        body=body,
        use_case=forecast_vol_impl,
        operation="post_forecast_volatility",
        domain_error_code="forecast_volatility_error",
        mt5_error_code="forecast_volatility_mt5_unavailable",
        internal_error_code="forecast_volatility_internal_error",
        internal_message="Forecast volatility computation failed.",
        result_error_code="forecast_volatility_error",
    )


def post_backtest_response(*, body: Any, backtest_use_case: Callable[..., Any]) -> Dict[str, Any]:
    return _post_use_case_response(
        body=body,
        use_case=backtest_use_case,
        operation="post_backtest",
        domain_error_code="backtest_error",
        mt5_error_code="backtest_mt5_unavailable",
        internal_error_code="backtest_internal_error",
        internal_message="Backtest computation failed.",
        result_error_code="backtest_error",
    )
