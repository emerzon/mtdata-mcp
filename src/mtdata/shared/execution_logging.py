"""Shared execution-path logging helpers."""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from typing import Any, Callable, Optional, TypeVar

from mtdata.shared.request_context import current_request_id

ResultT = TypeVar("ResultT")

_ACTIVE_OPERATIONS: ContextVar[tuple[str, ...]] = ContextVar(
    "mtdata_active_operations",
    default=(),
)


def infer_result_success(result: Any) -> bool:
    if isinstance(result, dict):
        error_text = result.get("error")
        if isinstance(error_text, str) and error_text.strip():
            return False
        if error_text not in (None, False):
            return False
        success = result.get("success")
        if isinstance(success, bool):
            return success
        return True
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                error_text = item.get("error")
                if isinstance(error_text, str) and error_text.strip():
                    return False
                if error_text not in (None, False):
                    return False
        return True
    return result is not None


def log_operation_start(logger: logging.Logger, *, operation: str, **fields: Any) -> None:
    parent_operation = _push_operation(operation)
    if parent_operation == str(operation):
        return
    logger.debug("event=start operation=%s %s", operation, _format_fields(fields))


def log_operation_finish(
    logger: logging.Logger,
    *,
    operation: str,
    started_at: float,
    success: bool,
    **fields: Any,
) -> None:
    parent_operation = _pop_operation(operation)
    if parent_operation == str(operation):
        return
    log = logger.debug if success else logger.warning
    log(
        "event=finish operation=%s success=%s duration_ms=%.3f %s",
        operation,
        bool(success),
        _elapsed_ms(started_at),
        _format_fields(fields),
    )


def log_operation_exception(
    logger: logging.Logger,
    *,
    operation: str,
    started_at: float,
    exc: BaseException,
    **fields: Any,
) -> None:
    parent_operation = _pop_operation(operation)
    if parent_operation == str(operation):
        return
    logger.exception(
        "event=error operation=%s duration_ms=%.3f %s error=%s",
        operation,
        _elapsed_ms(started_at),
        _format_fields(fields),
        exc,
    )


def run_logged_operation(
    logger: logging.Logger,
    *,
    operation: str,
    func: Callable[[], ResultT],
    success_eval: Optional[Callable[[ResultT], bool]] = None,
    **fields: Any,
) -> ResultT:
    started_at = time.perf_counter()
    log_operation_start(logger, operation=operation, **fields)
    try:
        result = func()
    except Exception as exc:
        log_operation_exception(
            logger,
            operation=operation,
            started_at=started_at,
            exc=exc,
            **fields,
        )
        raise
    else:
        success_value = infer_result_success(result) if success_eval is None else bool(success_eval(result))
        finish_fields = dict(fields)
        if not success_value:
            for key, value in _failure_log_fields(result).items():
                finish_fields.setdefault(key, value)
        log_operation_finish(
            logger,
            operation=operation,
            started_at=started_at,
            success=success_value,
            **finish_fields,
        )
        return result


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - float(started_at)) * 1000.0, 3)


def _push_operation(operation: str) -> Optional[str]:
    stack = _ACTIVE_OPERATIONS.get()
    parent = stack[-1] if stack else None
    _ACTIVE_OPERATIONS.set(stack + (str(operation),))
    return parent


def _pop_operation(operation: str) -> Optional[str]:
    stack = _ACTIVE_OPERATIONS.get()
    op_name = str(operation)
    if not stack:
        return None
    if stack[-1] == op_name:
        parent = stack[-2] if len(stack) > 1 else None
        _ACTIVE_OPERATIONS.set(stack[:-1])
        return parent
    for idx in range(len(stack) - 1, -1, -1):
        if stack[idx] == op_name:
            parent = stack[idx - 1] if idx > 0 else None
            _ACTIVE_OPERATIONS.set(stack[:idx] + stack[idx + 1 :])
            return parent
    return None


def _failure_log_fields(result: Any) -> dict[str, Any]:
    """Extract bounded, non-payload diagnostics from a structured failure."""
    if isinstance(result, dict):
        source = result
    elif isinstance(result, list):
        source = next(
            (
                item
                for item in result
                if isinstance(item, dict) and not infer_result_success(item)
            ),
            {},
        )
    else:
        return {}

    fields: dict[str, Any] = {}
    for key in ("error", "error_code", "code", "retcode", "retcode_name", "ambiguous"):
        value = source.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, str):
            value = " ".join(value.split())[:300]
        fields[key] = value
    return fields


def _format_fields(fields: dict[str, Any]) -> str:
    request_id = current_request_id()
    if request_id and "request_id" not in fields:
        fields = {"request_id": request_id, **fields}
    parts: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            parts.append(f"{key}={text}")
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)
