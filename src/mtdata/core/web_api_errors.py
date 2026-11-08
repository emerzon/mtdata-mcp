"""Canonical HTTP error mapping shared by Web API endpoint adapters."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import HTTPException

from ..shared.error_envelope import build_error_payload, normalize_error_payload

logger = logging.getLogger(__name__)


def http_status_for_error(payload: Any, *, default: int = 400) -> int:
    """Map a canonical domain error to HTTP without changing its identity."""
    if not isinstance(payload, dict):
        return default
    code = str(payload.get("error_code") or "").strip().lower()
    if (
        "param" in code
        or "validation" in code
        or code in {"invalid_params", "invalid_argument", "tool_param_error"}
    ):
        return 422
    if "not_found" in code:
        return 404
    if "internal" in code:
        return 500
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


def http_error(
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


def raise_tool_error(
    result: Any,
    *,
    operation: str,
    default_code: str,
    default_status: int = 400,
    invalid_message: Optional[str] = None,
) -> None:
    """Raise the shared HTTP error for a structured tool failure."""
    if isinstance(result, dict) and result.get("error"):
        raise http_error(
            http_status_for_error(result, default=default_status),
            result,
            code=str(result.get("error_code") or default_code),
            operation=operation,
        )
    if invalid_message is not None and not isinstance(result, dict):
        raise http_error(
            500,
            invalid_message,
            code=default_code,
            operation=operation,
        )
