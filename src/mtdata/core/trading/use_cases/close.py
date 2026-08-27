"""Trade close and cancel use cases."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from mtdata.core.error_envelope import normalize_error_payload
from mtdata.core.execution_logging import (
    infer_result_success,
    log_operation_finish,
    log_operation_start,
)
from mtdata.core.trading import comments
from mtdata.core.trading.requests import TradeCloseRequest
from mtdata.core.trading.use_cases.common import (
    _TRADE_IDEMPOTENCY_STORE,
    TradeIdempotencyStore,
    _annotate_idempotency_scope,
    _attach_trade_attempt_markers,
    _attach_trade_correlation,
    _begin_trade_idempotency,
    _build_trade_request_signature,
    _compact_close_preview_payload,
    _log_trade_correlation,
    _normalize_idempotency_key,
    _record_or_release_idempotency,
    logger,
)


def _run_trade_close_once(  # noqa: C901
    request: TradeCloseRequest,
    *,
    close_positions: Any,
    cancel_pending: Any,
    lookup_ticket_history: Any = None,
    resolve_close_target: Any = None,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="trade_close",
        correlation_id=correlation_id,
        ticket=request.ticket,
        target=request.target,
        close_all=request.close_all,
        symbol=request.symbol,
        volume=request.volume,
        profit_only=request.profit_only,
        loss_only=request.loss_only,
        dry_run=request.dry_run,
        confirm_close_all=request.confirm_close_all,
        magic=request.magic,
    )
    bulk_request = False

    def _finish(
        result: Dict[str, Any],
        *,
        scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = _attach_trade_attempt_markers(result, dry_run=request.dry_run)
        if (
            request.dry_run
            and result.get("success") is True
            and not str(result.get("error") or "").strip()
            and result.get("preview_ok") is not False
        ):
            result.setdefault("preview_ok", True)
            if request.target == "all_exposure":
                result.setdefault(
                    "comment_previews",
                    {
                        "positions": comments._attach_comment_preview_metadata(
                            {},
                            request.comment,
                            default="mtdata close",
                            close=True,
                        ),
                        "pending_orders": comments._attach_comment_preview_metadata(
                            {},
                            request.comment,
                            default="mtdata cancel pending order",
                        ),
                    },
                )
                result.setdefault("requested_comment", request.comment)
            else:
                result = comments._attach_comment_preview_metadata(
                    result,
                    request.comment,
                    default=(
                        "mtdata cancel pending order"
                        if request.target == "pending"
                        else "mtdata close"
                    ),
                    close=request.target != "pending",
                )
        if request.detail == "compact":
            result = _compact_close_preview_payload(result)
        if isinstance(result, dict) and str(result.get("error") or "").strip():
            error_text = str(result.get("error") or "").strip().lower()
            if request.ticket is not None and (
                "not found as position or pending order" in error_text
                or (
                    error_text.startswith(("position ", "pending order "))
                    and " not found" in error_text
                )
                or (
                    request.volume is not None
                    and error_text.startswith("position ")
                    and " not found" in error_text
                )
            ):
                result.setdefault("error_code", "ticket_not_found")
                result.setdefault("ticket", request.ticket)
            result = normalize_error_payload(
                result,
                default_code="trade_close_error",
                request_id=correlation_id,
                operation="trade_close",
            )
        result = _attach_trade_correlation(result, correlation_id=correlation_id)
        log_operation_finish(
            logger,
            operation="trade_close",
            started_at=started_at,
            success=infer_result_success(result),
            correlation_id=correlation_id,
            ticket=request.ticket,
            target=request.target,
            close_all=request.close_all,
            symbol=request.symbol,
            volume=request.volume,
            scope=scope,
            profit_only=request.profit_only,
            loss_only=request.loss_only,
            dry_run=request.dry_run,
            confirm_close_all=request.confirm_close_all,
            magic=request.magic,
        )
        return result

    def _mark_bulk_preview_unconfirmed(payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload)
        preview_failed = (
            out.get("success") is False
            or bool(str(out.get("error") or "").strip())
            or out.get("preview_ok") is False
        )
        if not preview_failed:
            out["success"] = True
            out.setdefault("preview_ok", True)
            out.pop("error", None)
            if out.get("error_code") == "preview_blocked":
                out.pop("error_code", None)
        out["required_confirmation"] = "--confirm-close-all true"
        out["authorization_status"] = "required"
        validation_payload = out.get("validation")
        if not isinstance(validation_payload, dict):
            validation_payload = {}
        validation_payload["live_submission_eligible"] = False
        blockers = [
            str(item)
            for item in list(validation_payload.get("blockers") or [])
            if str(item).strip()
        ]
        if "confirmation_required" not in blockers:
            blockers.append("confirmation_required")
        validation_payload["blockers"] = blockers
        out["validation"] = validation_payload
        return out

    def _with_no_action(
        payload: Optional[Dict[str, Any]] = None,
        *,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(payload or {})
        if message and not str(out.get("message", "")).strip():
            out["message"] = message
        out.setdefault("success", True)
        out["no_action"] = True
        return out

    def _leg_count(result: Any, *keys: str) -> int:
        if not isinstance(result, dict):
            return 0
        for key in keys:
            try:
                if result.get(key) is not None:
                    return max(0, int(result[key]))
            except (TypeError, ValueError):
                continue
        return 0

    def _combine_exposure_legs(
        position_result: Dict[str, Any],
        pending_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        position_error = str(position_result.get("error") or "").strip()
        pending_error = str(pending_result.get("error") or "").strip()
        matched_positions_count = _leg_count(
            position_result, "matched_count", "closed_count"
        )
        matched_pending_count = _leg_count(
            pending_result, "matched_pending_count", "cancelled_count"
        )
        closed_count = 0 if request.dry_run else matched_positions_count
        cancelled_count = 0 if request.dry_run else matched_pending_count
        failed_legs = [
            name
            for name, error in (
                ("positions", position_error),
                ("pending_orders", pending_error),
            )
            if error
        ]
        out: Dict[str, Any] = {
            "success": not failed_legs,
            "target": "all_exposure",
            "dry_run": request.dry_run,
            "closed_count": closed_count,
            "cancelled_count": cancelled_count,
            "closed_positions": position_result,
            "cancelled_pending_orders": pending_result,
            "leg_status": {
                "positions": "error" if position_error else "ok",
                "pending_orders": "error" if pending_error else "ok",
            },
            "partial_failure": len(failed_legs) == 1,
            "failed_legs": failed_legs,
        }
        if request.dry_run:
            out["matched_positions_count"] = matched_positions_count
            out["matched_pending_count"] = matched_pending_count
            out["would_send_orders"] = matched_positions_count
            out["would_cancel_pending_orders"] = matched_pending_count
            out["would_cancel_pending_order"] = matched_pending_count > 0
            out["actionability"] = "preview_only"
            child_preview_ok = True
            for child in (position_result, pending_result):
                if isinstance(child, dict) and child.get("preview_ok") is False:
                    child_preview_ok = False
                    break
            out["preview_ok"] = child_preview_ok
            if not child_preview_ok:
                out["success"] = False
                out.setdefault("error_code", "preview_blocked")
                out.setdefault(
                    "error",
                    "One or more exposure legs are not live-ready.",
                )
        affected_count = (
            matched_positions_count + matched_pending_count
            if request.dry_run
            else closed_count + cancelled_count
        )
        if not failed_legs and affected_count == 0:
            out["no_action"] = True
            out["message"] = "No matching open positions or pending orders."
        elif failed_legs:
            out["error_code"] = (
                "all_exposure_partial_failure"
                if len(failed_legs) == 1
                else "all_exposure_failed"
            )
            out["error"] = "All-exposure operation failed for: " + ", ".join(
                failed_legs
            )
        return out

    magic_kwargs = {"magic": request.magic} if request.magic is not None else {}
    target = request.target

    if request.profit_only and request.loss_only:
        return _finish(
            {"error": "profit_only and loss_only cannot both be true."},
            scope="positions",
        )

    if request.volume is not None:
        if target != "positions":
            return _finish(
                {
                    "error": "volume is valid only with target=positions.",
                    "error_code": "invalid_close_target_options",
                },
                scope=target,
            )
        if request.ticket is None:
            return _finish(
                {
                    "error": (
                        "volume is only supported when closing a specific open position by ticket."
                    )
                },
                scope="positions",
            )
        if request.profit_only or request.loss_only:
            return _finish(
                {
                    "error": (
                        "volume cannot be combined with profit_only or loss_only. "
                        "Use ticket for a specific partial close."
                    )
                },
                scope="positions",
            )

    if target != "positions" and (request.profit_only or request.loss_only):
        return _finish(
            {
                "error": "pnl_filter is valid only with target=positions.",
                "error_code": "invalid_close_target_options",
            },
            scope=target,
        )

    if request.ticket is not None and target == "all_exposure":
        return _finish(
            {
                "error": (
                    "target=all_exposure requires a bulk symbol, magic, or "
                    "account scope; it cannot be combined with ticket."
                ),
                "error_code": "invalid_close_target_options",
            },
            scope="ticket",
        )

    if request.ticket is not None and request.close_all:
        return _finish(
            {
                "error": (
                    "close_all cannot be combined with ticket. "
                    "Use ticket for a specific position or pending order, "
                    "or omit --ticket and pass --close-all true for a bulk close."
                )
            },
            scope="ticket",
        )

    bulk_selector = bool(request.symbol is not None or request.magic is not None)
    bulk_request = request.ticket is None and bool(
        request.close_all or bulk_selector
    )
    if bulk_request and not request.dry_run and not request.confirm_close_all:
        return _finish(
            {
                "error": (
                    "A live bulk operation requires explicit confirmation. Re-run with "
                    "--dry-run true to preview, or pass --confirm-close-all true to "
                    "execute the selected target and scope."
                ),
                "error_code": "confirmation_required",
                "close_all": request.close_all,
                "target": target,
                "dry_run": False,
                "required_confirmation": "--confirm-close-all true",
                "alternatives": [
                    "Pass --dry-run true to preview matching objects",
                    "Pass --ticket <ticket_number> for a specific target object",
                    "Pass --confirm-close-all true only after reviewing exposure",
                ],
                "remediation": (
                    "Pass --dry-run true to preview matching objects, "
                    "--ticket for a specific target, or --confirm-close-all true "
                    "only after reviewing exposure."
                ),
            },
            scope="bulk_confirmation",
        )

    if (
        request.ticket is None
        and request.symbol is None
        and not request.close_all
        and request.magic is None
        and request.dry_run
    ):
        return _finish(
            {
                "error": (
                    "Close preview requires an explicit scope: specify --ticket <ticket>, "
                    "--symbol <symbol>, --magic <number>, or --close-all true."
                ),
                "error_code": "close_scope_required",
                "alternatives": [
                    "Use --ticket <ticket_number> to preview a specific close",
                    "Use --symbol <symbol> to preview positions for one symbol",
                    "Use --magic <number> to preview one strategy's matching objects",
                    "Pass --close-all true to preview the selected target account-wide",
                ],
                "remediation": (
                    "Specify --ticket, --symbol, --magic, or --close-all true, "
                    "then retry trade_close."
                ),
            },
            scope="request",
        )

    if (
        request.ticket is None
        and request.symbol is None
        and request.magic is None
        and not request.close_all
        and not request.dry_run
    ):
        return _finish(
            {
                "error": (
                    "Bulk close requires explicit confirmation: pass --close-all true "
                    "for an account-wide operation, or specify --ticket, --symbol, or --magic."
                ),
                "error_code": "confirmation_required",
                "suggestion": "Review matching positions before closing (irreversible action).",
                "alternatives": [
                    "Use --ticket <ticket_number> for one target object",
                    "Use --symbol <symbol> or --magic <number> for a bounded bulk scope",
                    "Use --close-all true for the selected target account-wide",
                ],
            },
            scope="bulk_confirmation",
        )

    if request.dry_run:
        target_result: Optional[Dict[str, Any]] = None
        if request.ticket is not None and resolve_close_target is not None:
            target_result = resolve_close_target(
                ticket=request.ticket,
                target=target,
                symbol=request.symbol,
                volume=request.volume,
                magic=request.magic,
                profit_only=request.profit_only,
                loss_only=request.loss_only,
                close_priority=request.close_priority,
            )
            if isinstance(target_result, dict) and target_result.get("error"):
                return _finish(
                    target_result,
                    scope=str(target_result.get("target_scope") or "ticket"),
                )

        scope = (
            "ticket"
            if request.ticket is not None
            else f"symbol_{target}"
            if request.symbol is not None
            else target
        )
        operation = (
            "partial_close_position"
            if request.volume is not None
            else f"{target}_ticket"
            if request.ticket is not None
            else "flatten_all_exposure"
            if target == "all_exposure"
            else "cancel_symbol_pending_orders"
            if request.symbol is not None and target == "pending"
            else "close_symbol_positions"
            if request.symbol is not None
            else "cancel_all_pending_orders"
            if target == "pending"
            else "close_all_positions"
        )
        preview: Dict[str, Any] = {
            "success": True,
            "dry_run": True,
            "actionability": "preview_only",
            "operation": operation,
            "scope": scope,
            "would_send_order": False,
            "would_cancel_pending_order": False,
            "preview_scope_summary": (
                "Routing and request validation only; no close or cancel request was sent to MT5."
            ),
            "estimated": [
                "operation",
                "scope",
                "routing",
                "target_resolution" if request.ticket is not None else "filter_scope",
            ],
            "not_estimated": [
                "realized_pnl",
                "slippage",
                "post_close_balance",
                "tax_impact",
            ],
        }
        if request.ticket is None:
            position_preview: Dict[str, Any] = {}
            pending_preview: Dict[str, Any] = {}
            if target in {"positions", "all_exposure"}:
                position_preview = close_positions(
                    symbol=request.symbol,
                    **magic_kwargs,
                    volume=None,
                    profit_only=request.profit_only,
                    loss_only=request.loss_only,
                    close_priority=request.close_priority,
                    comment=request.comment,
                    deviation=request.deviation,
                    dry_run=True,
                )
            if target in {"pending", "all_exposure"}:
                pending_preview = cancel_pending(
                    symbol=request.symbol,
                    **magic_kwargs,
                    comment=request.comment,
                    dry_run=True,
                )
            if target == "all_exposure":
                combined = _combine_exposure_legs(
                    position_preview,
                    pending_preview,
                )
                combined.update(
                    {
                        "operation": operation,
                        "scope": scope,
                        "symbol": request.symbol,
                        "magic": request.magic,
                        "close_all": request.close_all,
                    }
                )
                if bulk_request and not request.confirm_close_all:
                    combined = _mark_bulk_preview_unconfirmed(combined)
                return _finish(
                    {key: value for key, value in combined.items() if value is not None},
                    scope=scope,
                )
            selected_preview = (
                pending_preview if target == "pending" else position_preview
            )
            if isinstance(selected_preview, dict):
                if selected_preview.get("error"):
                    return _finish(selected_preview, scope=target)
                for key in (
                    "matched_count",
                    "matched_positions",
                    "total_volume",
                    "total_profit",
                    "filters_applied",
                    "would_send_orders",
                    "preview_ok",
                    "market_readiness",
                    "matched_pending_count",
                    "matched_pending_orders",
                    "would_cancel_pending_orders",
                    "message",
                ):
                    if key in selected_preview:
                        preview[key] = selected_preview[key]
            if target == "pending":
                matched_pending = _leg_count(
                    preview,
                    "matched_pending_count",
                    "would_cancel_pending_orders",
                )
                preview["would_cancel_pending_orders"] = matched_pending
                preview["would_cancel_pending_order"] = matched_pending > 0
                if matched_pending == 0:
                    preview.setdefault("no_action", True)
                    preview.setdefault("empty", True)
            else:
                matched_positions = _leg_count(
                    preview,
                    "matched_count",
                    "would_send_orders",
                )
                preview["matched_positions_count"] = matched_positions
                preview["empty"] = matched_positions == 0
                if matched_positions == 0:
                    preview.setdefault("no_action", True)
        if request.ticket is not None:
            preview["ticket"] = request.ticket
            preview["ticket_resolution"] = (
                f"Would target only {target}; no object-class fallback is performed."
            )
            if isinstance(target_result, dict):
                for key in (
                    "success",
                    "error",
                    "error_code",
                    "blockers",
                    "target_scope",
                    "target_kind",
                    "resolved_ticket",
                    "target_symbol",
                    "target_volume",
                    "matched_count",
                    "matched_positions",
                    "total_volume",
                    "total_profit",
                    "filters_applied",
                    "would_send_orders",
                    "preview_ok",
                    "market_readiness",
                    "requested_close_volume",
                ):
                    value = target_result.get(key)
                    if value is not None:
                        preview[key] = value
                if preview.get("symbol") in (None, "") and preview.get("target_symbol"):
                    preview["symbol"] = preview["target_symbol"]
                if preview.get("volume") is None and preview.get("target_volume") is not None:
                    preview["volume"] = preview["target_volume"]
        if request.symbol is not None:
            preview["symbol"] = request.symbol
        if request.magic is not None:
            preview["magic"] = request.magic
        preview["target"] = target
        if request.volume is not None:
            preview["volume"] = request.volume
            preview["ticket_resolution"] = (
                "Would target only an open position; partial close does not fall back to pending orders."
            )
        if request.close_all:
            preview["close_all"] = True
        if request.profit_only:
            preview["profit_only"] = True
        if request.loss_only:
            preview["loss_only"] = True
        if request.close_priority:
            preview["close_priority"] = request.close_priority
        if request.deviation != 20:
            preview["deviation"] = request.deviation
        if bulk_request and not request.confirm_close_all:
            preview = _mark_bulk_preview_unconfirmed(preview)
        return _finish(preview, scope=scope)

    if target == "pending":
        pending_result = cancel_pending(
            ticket=request.ticket,
            symbol=request.symbol,
            **magic_kwargs,
            comment=request.comment,
            dry_run=False,
        )
        if isinstance(pending_result, dict):
            message = str(pending_result.get("message") or "").strip().lower()
            if message.startswith("no pending orders"):
                pending_result = _with_no_action(pending_result)
        return _finish(pending_result, scope="pending_orders")

    if target == "all_exposure":
        position_result = close_positions(
            symbol=request.symbol,
            **magic_kwargs,
            volume=None,
            profit_only=False,
            loss_only=False,
            close_priority=request.close_priority,
            comment=request.comment,
            deviation=request.deviation,
            dry_run=False,
        )
        pending_result = cancel_pending(
            symbol=request.symbol,
            **magic_kwargs,
            comment=request.comment,
            dry_run=False,
        )
        return _finish(
            _combine_exposure_legs(position_result, pending_result),
            scope="all_exposure",
        )

    if request.profit_only or request.loss_only:
        result = close_positions(
            ticket=request.ticket,
            symbol=request.symbol,
            **magic_kwargs,
            volume=None,
            profit_only=request.profit_only,
            loss_only=request.loss_only,
            close_priority=request.close_priority,
            comment=request.comment,
            deviation=request.deviation,
            dry_run=False,
        )
        if isinstance(result, dict):
            msg = str(result.get("message", "")).strip().lower()
            if (
                msg.startswith("no open positions")
                or msg == "no positions matched criteria"
            ):
                return _finish(_with_no_action(result), scope="positions")
        return _finish(result, scope="positions")

    if request.ticket is not None:
        position_result = close_positions(
            ticket=request.ticket,
            symbol=request.symbol,
            **magic_kwargs,
            volume=request.volume,
            profit_only=False,
            loss_only=False,
            close_priority=request.close_priority,
            comment=request.comment,
            deviation=request.deviation,
            dry_run=False,
        )
        if (
            request.volume is not None
            and isinstance(position_result, dict)
            and position_result.get("error") == f"Position {request.ticket} not found"
        ):
            return _finish(
                {
                    "error_code": "ticket_not_found",
                    "error": (
                        f"Position {request.ticket} not found. "
                        "Partial close volume only applies to open positions."
                    ),
                    "ticket": request.ticket,
                    "checked_scopes": ["positions"],
                },
                scope="positions",
            )
        if (
            isinstance(position_result, dict)
            and position_result.get("error") == f"Position {request.ticket} not found"
        ):
            history_result = None
            if lookup_ticket_history is not None:
                try:
                    history_result = lookup_ticket_history(request.ticket)
                except Exception:
                    history_result = None
            if isinstance(history_result, dict) and history_result:
                return _finish(history_result, scope="history")
            return _finish(
                {
                    "error_code": "ticket_not_found",
                    "error": f"Position {request.ticket} not found.",
                    "ticket": request.ticket,
                    "checked_scopes": ["positions"],
                    "suggestion": (
                        "Use trade_get_open to find an active position ticket, or "
                        "set target=pending to cancel a pending-order ticket."
                    ),
                },
                scope="positions",
            )
        return _finish(position_result, scope="positions")

    if request.symbol is not None:
        position_result = close_positions(
            symbol=request.symbol,
            **magic_kwargs,
            volume=None,
            profit_only=False,
            loss_only=False,
            close_priority=request.close_priority,
            comment=request.comment,
            deviation=request.deviation,
            dry_run=False,
        )
        if isinstance(position_result, dict):
            msg = str(position_result.get("message", "")).strip().lower()
            if msg.startswith("no open positions for ") or msg == "no positions matched criteria":
                return _finish(_with_no_action(position_result), scope="positions")
        return _finish(position_result, scope="positions")

    position_result = close_positions(
        **magic_kwargs,
        volume=None,
        profit_only=False,
        loss_only=False,
        close_priority=request.close_priority,
        comment=request.comment,
        deviation=request.deviation,
        dry_run=False,
    )
    if isinstance(position_result, dict):
        msg = str(position_result.get("message", "")).strip().lower()
        if msg in {"no open positions", "no positions matched criteria"}:
            return _finish(_with_no_action(position_result), scope="positions")
    return _finish(position_result, scope="positions")


def run_trade_close(
    request: TradeCloseRequest,
    *,
    close_positions: Any,
    cancel_pending: Any,
    lookup_ticket_history: Any = None,
    resolve_close_target: Any = None,
    idempotency_store: Optional[TradeIdempotencyStore] = _TRADE_IDEMPOTENCY_STORE,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a close/cancel operation with the shared durable dedupe lifecycle."""
    idempotency_key = _normalize_idempotency_key(
        getattr(request, "idempotency_key", None)
    )
    idempotency_signature = (
        _build_trade_request_signature(request)
        if idempotency_key is not None
        else None
    )
    duplicate_result, idempotency_reserved = _begin_trade_idempotency(
        idempotency_store=idempotency_store,
        key=idempotency_key,
        request_signature=idempotency_signature,
    )
    idempotency_consumed = False
    if duplicate_result is not None:
        started_at = time.perf_counter()
        log_operation_start(
            logger,
            operation="trade_close",
            correlation_id=correlation_id,
            ticket=request.ticket,
            duplicate=True,
        )
        result = _annotate_idempotency_scope(
            duplicate_result,
            idempotency_key,
            idempotency_store,
        )
        result = _attach_trade_correlation(result, correlation_id=correlation_id)
        _log_trade_correlation(operation="trade_close", result=result)
        log_operation_finish(
            logger,
            operation="trade_close",
            started_at=started_at,
            success=infer_result_success(result),
            correlation_id=correlation_id,
            ticket=request.ticket,
            duplicate=True,
        )
        return result

    try:
        result = _run_trade_close_once(
            request,
            close_positions=close_positions,
            cancel_pending=cancel_pending,
            lookup_ticket_history=lookup_ticket_history,
            resolve_close_target=resolve_close_target,
            correlation_id=correlation_id,
        )
        result = _annotate_idempotency_scope(
            result,
            idempotency_key,
            idempotency_store,
        )
        if _record_or_release_idempotency(
            idempotency_store,
            idempotency_key,
            result,
            request_signature=idempotency_signature,
        ):
            idempotency_consumed = True
        _log_trade_correlation(operation="trade_close", result=result)
        return result
    finally:
        if (
            idempotency_reserved
            and not idempotency_consumed
            and idempotency_store is not None
            and idempotency_key is not None
        ):
            idempotency_store.release(
                idempotency_key,
                request_signature=idempotency_signature,
            )
