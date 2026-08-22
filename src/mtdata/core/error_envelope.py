"""Shared error-envelope and transport logging helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from uuid import uuid4

from .request_context import current_request_id

_ERROR_GUIDANCE: Dict[str, Dict[str, Any]] = {
    "mt5_connection_error": {
        "remediation": "Ensure MetaTrader 5 is running, logged in, and reachable.",
        "related_tools": ["symbols_list"],
    },
    "symbol_not_found": {
        "remediation": (
            "Use symbols_list with a search term, then retry with a broker-listed "
            "symbol."
        ),
        "related_tools": ["symbols_list"],
    },
    "finviz_unsupported_symbol": {
        "remediation": (
            "Equity-profile tools require a US equity ticker. Use news or MT5 "
            "market-data tools for broker, FX, and crypto symbols; use "
            "asset_performance for delayed forex/crypto/futures context."
        ),
        "related_tools": [
            "news",
            "data_fetch_candles",
            "asset_performance",
        ],
    },
    "news_symbol_unavailable": {
        "remediation": (
            "Use symbols_list to verify a broker FX or crypto symbol, or verify "
            "the standard US equity ticker used by the news provider. Use "
            "screener to discover supported equity tickers."
        ),
        "related_tools": ["symbols_list", "screener", "news"],
    },
    "research_source_unavailable": {
        "remediation": (
            "Pass source=auto to use every available adapter, or pick a name "
            "from valid_values.source."
        ),
        "related_tools": ["news", "calendar"],
    },
    "research_capability_unsupported": {
        "remediation": (
            "Use source=auto, or pin a source listed in valid_values.source."
        ),
        "related_tools": ["news", "calendar"],
    },
    "calendar_invalid_view": {
        "remediation": "Use view=period only with kind=earnings, or switch to view=range.",
        "related_tools": ["calendar", "news"],
    },
    "options_unsupported_symbol": {
        "remediation": (
            "Options-chain tools require a US-listed underlier such as AAPL. Use "
            "market_ticker or data_fetch_candles for broker FX and crypto symbols."
        ),
        "related_tools": [
            "options_provider_status",
            "market_ticker",
            "symbols_list",
        ],
    },
    "dependency_missing": {
        "remediation": (
            "Install the optional dependency group required by this method, then retry."
        ),
    },
    "insufficient_data": {
        "remediation": (
            "Increase the lookback, request more bars, or use a longer timeframe."
        ),
    },
    "invalid_date_range": {
        "remediation": "Set start to a timestamp earlier than or equal to end.",
    },
    "invalid_datetime": {
        "remediation": (
            "Correct the listed start/end value using an ISO 8601 date or timestamp."
        ),
    },
    "forecast_task_not_found": {
        "remediation": (
            "Use forecast_task_list to inspect active and recent forecast tasks."
        ),
        "related_tools": ["forecast_task_list"],
    },
    "forecast_task_cancel_failed": {
        "remediation": (
            "Use forecast_task_status to verify the task state, or forecast_task_list "
            "to inspect active tasks."
        ),
        "related_tools": ["forecast_task_status", "forecast_task_list"],
    },
    "forecast_model_not_found": {
        "remediation": "Use forecast_models_list to inspect stored forecast models.",
        "related_tools": ["forecast_models_list"],
    },
    "indicator_not_found": {
        "remediation": (
            "Use indicators_list to inspect canonical catalog names such as rsi, "
            "then retry indicators_describe with that exact name. Tokens like "
            "rsi_14 are fetch specs / output columns for data_fetch_candles, "
            "not catalog names."
        ),
        "related_tools": ["indicators_list"],
    },
    "ticket_not_found": {
        "remediation": (
            "Use trade_get_open or trade_get_pending to find an active ticket, "
            "then retry with that exact ticket."
        ),
        "related_tools": ["trade_get_open", "trade_get_pending"],
    },
    "close_scope_required": {
        "remediation": (
            "Specify --ticket, --symbol, --magic, or --close-all true, then retry "
            "trade_close."
        ),
        "related_tools": ["trade_get_open", "trade_get_pending"],
    },
}

_CANONICAL_DATE_RANGE_MESSAGE = "start must be before or equal to end."
_GUIDANCE_KEYS = {
    "remediation",
    "related_tools",
    "valid_values",
    "example",
    "documentation",
}
_GENERIC_ERROR_CODES = {
    "",
    "error",
    "internal_error",
    "tool_error",
    "unknown_error",
    "forecast_generate_error",
}
_METHOD_ERROR_CODES = frozenset(
    {"invalid_method", "unsupported_method", "method_unavailable"}
)


def new_request_id() -> str:
    return current_request_id() or uuid4().hex[:12]


def _default_error_guidance(
    *,
    code: str,
    operation: Optional[str],
) -> Dict[str, Any]:
    code_text = str(code or "").strip().lower()
    operation_text = str(operation or "").strip().lower()
    if code_text in {"symbol_not_found", "finviz_symbol_not_found"} and (
        operation_text
        in {"equity_profile", "screener", "asset_performance", "news", "calendar"}
        or operation_text.startswith("finviz_")
        or code_text == "finviz_symbol_not_found"
    ):
        return {
            "remediation": (
                "Verify the standard US equity ticker used by the research "
                "provider; do not use an MT5 broker suffix. Use screener to "
                "discover provider tickers."
            ),
            "related_tools": ["screener"],
        }
    if code_text in _ERROR_GUIDANCE:
        return dict(_ERROR_GUIDANCE[code_text])
    if code_text.endswith("_connection_error"):
        return dict(_ERROR_GUIDANCE["mt5_connection_error"])
    if code_text in _METHOD_ERROR_CODES:
        return {
            "remediation": (
                "Use this operation's --help and choose one of the listed method values."
            )
        }
    if code_text == "cli_missing_required" and operation_text in {
        "forecast_task_cancel",
        "forecast_task_status",
        "forecast_task_wait",
    }:
        return {
            "remediation": (
                "Use forecast_task_list to find a task_id, then retry the task "
                "operation with that identifier."
            ),
            "related_tools": ["forecast_task_list"],
        }
    if operation_text.startswith("forecast_models_"):
        return {
            "remediation": (
                "Use forecast_models_list to inspect stored model IDs, then retry "
                "the model-store operation with the intended scope."
            ),
            "related_tools": ["forecast_models_list"],
        }
    if operation_text in {"forecast_barrier_optimize", "forecast_barrier_prob"}:
        related = (
            ["forecast_barrier_prob"]
            if operation_text == "forecast_barrier_optimize"
            else ["forecast_barrier_optimize"]
        )
        return {
            "remediation": (
                f"Use {operation_text} --help and choose values documented for "
                "that barrier workflow."
            ),
            "related_tools": related,
        }
    if operation_text == "forecast_train":
        return {
            "remediation": (
                "Choose a trainable method with forecast_list_methods "
                "--supports-training true, then retry forecast_train."
            ),
            "related_tools": ["forecast_list_methods"],
        }
    if operation_text.startswith("forecast_") or code_text.startswith("forecast_"):
        return {
            "remediation": (
                "Check forecast inputs and use forecast_list_methods to inspect "
                "available methods."
            ),
            "related_tools": ["forecast_list_methods"],
        }
    if "insufficient" in code_text:
        return dict(_ERROR_GUIDANCE["insufficient_data"])
    return {}


def _apply_error_guidance(
    payload: Dict[str, Any],
    *,
    code: str,
    operation: Optional[str],
    remediation: Optional[str] = None,
    related_tools: Optional[list[str]] = None,
    valid_values: Optional[Dict[str, Any]] = None,
    example: Optional[str] = None,
    documentation: Optional[str] = None,
) -> None:
    guidance = _default_error_guidance(code=code, operation=operation)
    if remediation:
        guidance["remediation"] = str(remediation)
    if related_tools:
        guidance["related_tools"] = list(related_tools)
    if valid_values:
        guidance["valid_values"] = dict(valid_values)
    if example:
        guidance["example"] = str(example)
    if documentation:
        guidance["documentation"] = str(documentation)

    for key, value in guidance.items():
        if key in payload or value in (None, "", [], {}):
            continue
        payload[key] = value


def _error_payload_text(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_error_payload_text(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_error_payload_text(item))
        return out
    return []


def _canonical_error_code(
    payload: Dict[str, Any],
    current_code: str,
    *,
    operation: Optional[str] = None,
) -> str:
    normalized_code = str(current_code or "").strip().lower()
    normalized_operation = str(
        payload.get("operation") or operation or ""
    ).strip().lower()
    is_operation_catch_all = bool(
        normalized_operation
        and normalized_code == f"{normalized_operation}_error"
    )
    if normalized_operation and normalized_code.startswith(f"{normalized_operation}_"):
        suffix = normalized_code[len(normalized_operation) + 1 :]
        if suffix in {"invalid_date_range", "invalid_date", "symbol_not_found"}:
            return suffix
    if normalized_code not in _GENERIC_ERROR_CODES and not is_operation_catch_all:
        return current_code
    evidence = " ".join(
        _error_payload_text(
            {
                "error": payload.get("error"),
                "details": payload.get("details"),
                "warnings": payload.get("warnings"),
            }
        )
    ).lower()
    symbol_failure = (
        "unknown symbol" in evidence
        or "failed to select symbol" in evidence
        or (
            "symbol" in evidence
            and any(
                phrase in evidence
                for phrase in (
                    "not found",
                    "was not found",
                    "could not be fetched",
                )
            )
        )
    )
    if symbol_failure:
        return "symbol_not_found"
    if any(
        phrase in evidence
        for phrase in (
            "start_datetime must be before end_datetime",
            "start must be before end",
            "start must be before or equal to end",
        )
    ):
        return "invalid_date_range"
    return current_code


def _dedupe_error_sequence(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    out: list[Any] = []
    seen: set[str] = set()
    for item in value:
        marker = repr(item)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(item)
    return out


def normalize_error_payload(
    payload: Dict[str, Any],
    *,
    default_code: Optional[str] = None,
    request_id: Optional[str] = None,
    operation: Optional[str] = None,
) -> Dict[str, Any]:
    error_text = payload.get("error")
    if not isinstance(error_text, str) or not error_text.strip():
        return payload

    out = dict(payload)
    original_error_code = str(
        out.get("error_code") or default_code or "tool_error"
    ).strip()
    operation_value = str(out.get("operation") or operation or "").strip()
    error_code = _canonical_error_code(
        out,
        original_error_code,
        operation=operation_value,
    )
    error_code_changed = error_code != original_error_code
    rid = str(out.get("request_id") or "").strip() or (request_id or new_request_id())

    normalized_error = str(error_text)
    if error_code == "invalid_date_range":
        normalized_error = _CANONICAL_DATE_RANGE_MESSAGE

    if not str(out.get("remediation") or "").strip():
        suggestion = out.get("suggestion")
        if isinstance(suggestion, str) and suggestion.strip():
            out["remediation"] = suggestion.strip()
        else:
            alternatives = out.get("alternatives")
            if isinstance(alternatives, list):
                texts = [str(item).strip() for item in alternatives if str(item).strip()]
                if texts:
                    out["remediation"] = " ".join(texts)

    normalized: Dict[str, Any] = {
        "success": False,
        "error": normalized_error,
        "error_code": error_code,
        "request_id": rid,
    }
    if operation_value:
        normalized["operation"] = operation_value
    for key, value in out.items():
        if key in normalized or key in {
            "success",
            "error",
            "error_code",
            "request_id",
            "operation",
        }:
            continue
        if error_code_changed and key in _GUIDANCE_KEYS:
            continue
        if key in {"details", "warnings"}:
            value = _dedupe_error_sequence(value)
        normalized[key] = value
    _apply_error_guidance(
        normalized,
        code=error_code,
        operation=operation_value or None,
    )
    return normalized


def build_error_payload(
    message: Any,
    *,
    code: str,
    request_id: Optional[str] = None,
    operation: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    remediation: Optional[str] = None,
    related_tools: Optional[list[str]] = None,
    valid_values: Optional[Dict[str, Any]] = None,
    example: Optional[str] = None,
    documentation: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "success": False,
        "error": str(message),
        "error_code": str(code),
        "request_id": request_id or new_request_id(),
    }
    if operation:
        payload["operation"] = str(operation)
    _apply_error_guidance(
        payload,
        code=str(code),
        operation=operation,
        remediation=remediation,
        related_tools=related_tools,
        valid_values=valid_values,
        example=example,
        documentation=documentation,
    )
    if details:
        payload["details"] = dict(details)
    return payload


def log_transport_exception(
    logger: logging.Logger,
    *,
    transport: str,
    operation: str,
    request_id: str,
    exc: BaseException,
) -> None:
    logger.exception(
        "transport=%s operation=%s request_id=%s failed: %s",
        transport,
        operation,
        request_id,
        exc,
    )
