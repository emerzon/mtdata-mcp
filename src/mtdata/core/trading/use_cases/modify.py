"""Trade modification use case."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from mtdata.core.error_envelope import normalize_error_payload
from mtdata.core.execution_logging import (
    infer_result_success,
    log_operation_finish,
    log_operation_start,
)
from mtdata.core.trading import validation
from mtdata.core.trading.requests import TradeModifyRequest
from mtdata.core.trading.use_cases.common import (
    _TRADE_IDEMPOTENCY_STORE,
    TradeIdempotencyStore,
    _attach_live_guardrail_status,
    _attach_trade_attempt_markers,
    _attach_trade_correlation,
    _invalid_pending_expiration_payload,
    _log_trade_correlation,
    _TradeIdempotencyLifecycle,
    logger,
)


def run_trade_modify(
    request: TradeModifyRequest,
    *,
    normalize_pending_expiration: Any,
    modify_pending_order: Any,
    modify_position: Any,
    idempotency_store: Optional[TradeIdempotencyStore] = _TRADE_IDEMPOTENCY_STORE,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    idempotency = _TradeIdempotencyLifecycle(request, idempotency_store)
    log_operation_start(
        logger,
        operation="trade_modify",
        correlation_id=correlation_id,
        ticket=request.ticket,
        dry_run=request.dry_run,
    )

    def _finish(
        result: Dict[str, Any],
        *,
        pending: Optional[bool] = None,
    ) -> Dict[str, Any]:
        result = _attach_trade_attempt_markers(result, dry_run=request.dry_run)
        if (
            request.dry_run
            and result.get("success") is True
            and not str(result.get("error") or "").strip()
            and result.get("preview_ok") is not False
        ):
            result.setdefault("preview_ok", True)
        if correlation_id and str(result.get("error") or "").strip():
            result = normalize_error_payload(
                result,
                default_code="trade_modify_error",
                request_id=correlation_id,
                operation="trade_modify",
            )
        result = _attach_live_guardrail_status(result, dry_run=request.dry_run)
        result = idempotency.annotate(result)
        result = _attach_trade_correlation(result, correlation_id=correlation_id)
        idempotency.settle(result)
        _log_trade_correlation(operation="trade_modify", result=result)
        log_operation_finish(
            logger,
            operation="trade_modify",
            started_at=started_at,
            success=infer_result_success(result),
            correlation_id=correlation_id,
            ticket=request.ticket,
            pending=pending,
            dry_run=request.dry_run,
        )
        return result

    sl_zero = validation._zero_price_requested(request.stop_loss)
    tp_zero = validation._zero_price_requested(request.take_profit)
    if sl_zero and not request.clear_stop_loss:
        return _finish(
            {
                "success": False,
                "preview_ok": False,
                "error_code": "protection_clear_requires_flag",
                "error": (
                    "stop_loss=0 would remove stop-loss protection. Pass "
                    "clear_stop_loss=true to confirm."
                ),
                "remediation": (
                    "Use --clear-stop-loss true to remove the stop, or pass a "
                    "positive protective price."
                ),
                "ticket": request.ticket,
            }
        )
    if tp_zero and not request.clear_take_profit:
        return _finish(
            {
                "success": False,
                "preview_ok": False,
                "error_code": "protection_clear_requires_flag",
                "error": (
                    "take_profit=0 would remove take-profit protection. Pass "
                    "clear_take_profit=true to confirm."
                ),
                "remediation": (
                    "Use --clear-take-profit true to remove the take-profit, or "
                    "pass a positive protective price."
                ),
                "ticket": request.ticket,
            }
        )
    if request.clear_stop_loss and request.stop_loss is not None and not sl_zero:
        return _finish(
            {
                "success": False,
                "error_code": "conflicting_protection_fields",
                "error": "clear_stop_loss cannot be combined with a new stop_loss price.",
                "ticket": request.ticket,
            }
        )
    if request.clear_take_profit and request.take_profit is not None and not tp_zero:
        return _finish(
            {
                "success": False,
                "error_code": "conflicting_protection_fields",
                "error": (
                    "clear_take_profit cannot be combined with a new take_profit price."
                ),
                "ticket": request.ticket,
            }
        )
    mutable_fields = {
        "price",
        "stop_limit_price",
        "stop_loss",
        "take_profit",
        "clear_stop_loss",
        "clear_take_profit",
        "expiration",
    }
    if not (request.model_fields_set & mutable_fields):
        return _finish(
            {
                "success": False,
                "error_code": "no_modification_fields",
                "error": (
                    "trade_modify requires at least one field to change: price, "
                    "stop_limit_price, stop_loss, take_profit, or expiration."
                ),
                "remediation": (
                    "Provide at least one modification field. Price and expiration "
                    "apply only to pending orders."
                ),
                "ticket": request.ticket,
            }
        )

    duplicate_result = idempotency.begin()
    if duplicate_result is not None:
        return _finish(duplicate_result)

    try:
        price_val = request.price
        resolved_sl = 0.0 if request.clear_stop_loss else request.stop_loss
        resolved_tp = 0.0 if request.clear_take_profit else request.take_profit
        try:
            _, expiration_specified = normalize_pending_expiration(request.expiration)
        except (TypeError, ValueError) as ex:
            return _finish(
                _invalid_pending_expiration_payload(
                    ex,
                    dry_run=bool(request.dry_run),
                )
            )

        if (
            price_val is not None
            or request.stop_limit_price is not None
            or expiration_specified
        ):
            result = modify_pending_order(
                ticket=request.ticket,
                price=price_val,
                stop_limit_price=request.stop_limit_price,
                stop_loss=resolved_sl,
                take_profit=resolved_tp,
                expiration=request.expiration,
                dry_run=bool(request.dry_run),
            )
            if validation.is_ticket_not_found(result):
                return _finish(
                    validation.ticket_not_found_error(
                        request.ticket,
                        checked_scopes=["pending_orders"],
                        note="Note: price/expiration only apply to pending orders.",
                    ),
                    pending=True,
                )
            return _finish(result, pending=True)

        position_result = modify_position(
            ticket=request.ticket,
            stop_loss=resolved_sl,
            take_profit=resolved_tp,
            dry_run=bool(request.dry_run),
        )
        if position_result.get("success"):
            return _finish(position_result, pending=False)
        if validation.is_ticket_not_found(position_result):
            pending_result = modify_pending_order(
                ticket=request.ticket,
                price=None,
                stop_limit_price=request.stop_limit_price,
                stop_loss=resolved_sl,
                take_profit=resolved_tp,
                expiration=None,
                dry_run=bool(request.dry_run),
            )
            if validation.is_ticket_not_found(pending_result):
                return _finish(
                    validation.ticket_not_found_error(
                        request.ticket,
                        checked_scopes=["positions", "pending_orders"],
                    ),
                    pending=None,
                )
            return _finish(pending_result, pending=True)
        return _finish(position_result, pending=False)
    finally:
        idempotency.release()
