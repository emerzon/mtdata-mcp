"""Trade placement use case."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mtdata.bootstrap.settings import mt5_config, trade_guardrails_config
from mtdata.core.execution_logging import (
    infer_result_success,
    log_operation_finish,
    log_operation_start,
)
from mtdata.core.trading import comments, validation
from mtdata.core.trading.requests import TradePlaceRequest
from mtdata.core.trading.safety import (
    evaluate_trade_guardrails,
    guardrails_require_pending_snapshot,
    guardrails_require_position_snapshot,
    preview_trade_guardrails,
)
from mtdata.core.trading.use_cases.common import (
    _TRADE_IDEMPOTENCY_STORE,
    TradeIdempotencyStore,
    _annotate_idempotency_scope,
    _attach_live_guardrail_status,
    _attach_trade_correlation,
    _begin_trade_idempotency,
    _best_effort_trade_guardrail_account_info,
    _best_effort_trade_guardrail_pending_orders,
    _best_effort_trade_guardrail_positions,
    _build_trade_request_signature,
    _coerce_warning_list,
    _dry_run_blocker_error,
    _guardrail_order_side,
    _invalid_order_type_payload,
    _invalid_pending_expiration_payload,
    _log_trade_correlation,
    _normalize_idempotency_key,
    _record_or_release_idempotency,
    _resolve_trade_place_preview_detail,
    _shape_trade_place_preview,
    _sl_tp_result_details,
    _standardize_trade_operation_payload,
    logger,
)
from mtdata.utils.mt5 import mt5_adapter


def run_trade_place(  # noqa: C901
    request: TradePlaceRequest,
    *,
    normalize_order_type_input: Any,
    normalize_pending_expiration: Any,
    prevalidate_trade_place_market_input: Any,
    place_market_order: Any,
    place_pending_order: Any,
    close_positions: Any,
    safe_int_ticket: Any,
    build_dry_run_preview: Any = None,
    idempotency_store: Optional[TradeIdempotencyStore] = _TRADE_IDEMPOTENCY_STORE,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    missing: List[str] = []
    symbol_norm = str(request.symbol).strip() if request.symbol is not None else ""
    idempotency_key = _normalize_idempotency_key(getattr(request, "idempotency_key", None))
    idempotency_signature = (
        _build_trade_request_signature(request)
        if idempotency_key is not None
        else None
    )
    idempotency_consumed = False
    log_operation_start(
        logger,
        operation="trade_place",
        correlation_id=correlation_id,
        symbol=symbol_norm or None,
        requested_order_type=request.order_type,
    )

    def _finish(
        result: Dict[str, Any],
        *,
        order_type: Optional[str] = None,
        pending: Optional[bool] = None,
    ) -> Dict[str, Any]:
        nonlocal idempotency_consumed
        result = _standardize_trade_operation_payload(
            result,
            operation="trade_place",
            default_error_code="trade_place_error",
            request_id=correlation_id,
            dry_run=request.dry_run,
        )
        result = _attach_live_guardrail_status(result, dry_run=request.dry_run)
        result = _annotate_idempotency_scope(result, idempotency_key, idempotency_store)
        result = _attach_trade_correlation(result, correlation_id=correlation_id)
        if not idempotency_consumed:
            if _record_or_release_idempotency(
                idempotency_store,
                idempotency_key,
                result,
                request_signature=idempotency_signature,
            ):
                idempotency_consumed = True
        _log_trade_correlation(operation="trade_place", result=result)
        log_operation_finish(
            logger,
            operation="trade_place",
            started_at=started_at,
            success=infer_result_success(result),
            correlation_id=correlation_id,
            symbol=symbol_norm or None,
            order_type=order_type,
            pending=pending,
        )
        return result

    duplicate_result, idempotency_reserved = _begin_trade_idempotency(
        idempotency_store=idempotency_store,
        key=idempotency_key,
        request_signature=idempotency_signature,
    )
    if duplicate_result is not None:
        idempotency_consumed = True
        return _finish(duplicate_result)

    try:
        dry_run_missing_protection: List[str] = []
        dry_run_protection_error: Optional[Dict[str, Any]] = None

        def _dry_run_preview(  # noqa: C901
            *,
            order_type: str,
            pending: bool,
            normalized_expiration: Any,
            expiration_provided: bool,
            guardrail_preview: Dict[str, Any],
            order_preview: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            preview_detail = _resolve_trade_place_preview_detail(request)
            validation_scope = "local_preview_plus_estimates"
            broker_validation_not_performed = [
                "broker_acceptance",
                "broker_price_distance_enforcement",
                "broker_margin_reservation",
                "broker_fillability",
                "broker_sl_tp_attachment",
            ]
            local_blockers = [
                f"missing_{field_name}"
                for field_name in dry_run_missing_protection
            ]
            if dry_run_protection_error is not None:
                local_blockers.append("invalid_protection_levels")
            if guardrail_preview.get("checks_not_performed"):
                local_blockers.append("guardrail_checks_incomplete")
            if guardrail_preview.get("would_block_live") is True:
                local_blockers.append("guardrails_would_block_live")
            preview: Dict[str, Any] = {
                "success": True,
                "dry_run": True,
                "no_action": True,
                "no_action_reason": "dry_run",
                "would_send_order": False,
                "dry_run_simulated": True,
                "symbol": symbol_norm,
                "order_type": order_type,
                "pending": bool(pending),
                "order_type_category": "pending" if pending else "market",
                "action": "place_pending_order" if pending else "place_market_order",
                "volume": float(request.volume),
                "message": "Dry run only. No order was sent to MT5.",
                "validation_scope": validation_scope,
                "validation_passed": not local_blockers,
                "preview_ok": not local_blockers,
                "validation": {
                    "local_requirements_passed": not local_blockers,
                    "live_submission_eligible": not local_blockers,
                    "blockers": local_blockers,
                    "broker_validation_performed": False,
                },
                "actionability": "preview_only",
                "actionability_reason": (
                    "Dry run did not execute MT5 or broker-side validation. "
                    "Use this preview for request routing only."
                ),
                "preview_scope_summary": (
                    "Routing and local level checks, plus a margin estimate when available."
                ),
                "preview_checks_performed": [
                    "request_routing",
                    "local_safety_requirements",
                    "protection_level_preview",
                ],
                "broker_validation_not_performed": list(
                    broker_validation_not_performed
                ),
                "warnings": [
                    (
                        "Not validated in dry run: broker acceptance/enforcement, margin "
                        "reservation, fillability, and broker-side SL/TP attachment."
                    ),
                ],
                "require_sl_tp": bool(request.require_sl_tp),
                "auto_close_on_sl_tp_fail": True,
                "guardrails_enabled": bool(guardrail_preview.get("enabled")),
                "guardrails_preview": guardrail_preview,
            }
            unprotected_by_request = (
                not bool(request.require_sl_tp)
                and request.stop_loss in (None, 0)
                and request.take_profit in (None, 0)
            )
            if unprotected_by_request:
                preview["protection_status"] = "unprotected_by_request"
                preview["auto_close_on_sl_tp_fail"] = False
                preview["warnings"].append(
                    "Order is unprotected_by_request: require_sl_tp=false and no "
                    "stop_loss/take_profit were supplied. auto_close_on_sl_tp_fail "
                    "does not apply."
                )
            if dry_run_missing_protection:
                preview["dry_run_note"] = (
                    "A live submission with require_sl_tp=true would be rejected. "
                    "Add both stop_loss and take_profit, or explicitly set "
                    "require_sl_tp=false."
                )
                preview["actionability"] = "blocked_by_local_requirements"
            if dry_run_protection_error is not None:
                preview.update(
                    {
                        "sl_tp_valid": False,
                        "sl_tp_error": dry_run_protection_error.get("error"),
                        "validation_error": dry_run_protection_error.get("error"),
                        "validation_code": dry_run_protection_error.get(
                            "error_code",
                            "invalid_protection_levels",
                        ),
                    }
                )
            elif isinstance(order_preview, dict):
                preview.update(order_preview)
            elif callable(build_dry_run_preview):
                preview.update(
                    build_dry_run_preview(
                        symbol=symbol_norm,
                        volume=float(request.volume),
                        order_type=order_type,
                        pending=pending,
                        price=request.price,
                        stop_limit_price=request.stop_limit_price,
                        stop_loss=request.stop_loss,
                        take_profit=request.take_profit,
                    )
                )
            account_blockers = [
                str(blocker)
                for blocker in list(preview.get("account_blockers") or [])
                if str(blocker).strip()
            ]
            if account_blockers:
                validation_payload = preview.get("validation")
                if isinstance(validation_payload, dict):
                    existing_blockers = list(
                        validation_payload.get("blockers") or []
                    )
                    validation_payload["blockers"] = [
                        *account_blockers,
                        *(
                            blocker
                            for blocker in existing_blockers
                            if blocker not in account_blockers
                        ),
                    ]
                    validation_payload["live_submission_eligible"] = False
                preview["validation_passed"] = False
                preview["preview_ok"] = False
                preview["actionability"] = "blocked_by_account_state"
                preview["actionability_reason"] = (
                    "The account execution or margin state blocks a new order."
                )
            margin_estimate = validation.coerce_finite_float(
                preview.get("margin_required")
            )
            if margin_estimate is not None:
                preview["preview_checks_performed"].append("margin_estimate")
                if preview.get("margin_sufficient") is False:
                    validation_payload = preview.get("validation")
                    if isinstance(validation_payload, dict):
                        validation_payload["live_submission_eligible"] = False
                        blockers = validation_payload.setdefault("blockers", [])
                        if "margin_insufficient" not in blockers:
                            blockers.append("margin_insufficient")
                    preview["validation_passed"] = False
                    preview["preview_ok"] = False
                    preview["actionability"] = "blocked_by_margin_estimate"
                    preview["actionability_reason"] = (
                        "The estimated required margin exceeds current free margin."
                    )
            else:
                checks_not_performed = preview.setdefault(
                    "checks_not_performed", []
                )
                if "margin_estimate" not in checks_not_performed:
                    checks_not_performed.append("margin_estimate")
                warnings_out = preview.setdefault("warnings", [])
                warning = (
                    "Margin estimate unavailable from MT5; preview_ok covers local "
                    "request validity only, not affordability."
                )
                if warning not in warnings_out:
                    warnings_out.append(warning)
            quote_context = preview.get("quote_context")
            if (
                isinstance(quote_context, dict)
                and quote_context.get("usable_for_live_trading") is not True
            ):
                validation_payload = preview.get("validation")
                if isinstance(validation_payload, dict):
                    validation_payload["live_submission_eligible"] = False
                    blockers = validation_payload.setdefault("blockers", [])
                    if "quote_not_live_ready" not in blockers:
                        blockers.append("quote_not_live_ready")
                preview["validation_passed"] = False
                preview["preview_ok"] = False
                preview["actionability"] = "blocked_by_quote_freshness"
                preview["actionability_reason"] = (
                    "Quote freshness is not verified as live; refresh the quote "
                    "before using this preview for submission."
                )
                quote_warning = str(
                    quote_context.get("timestamp_warning")
                    or quote_context.get("warning")
                    or "Quote is not usable for live trading."
                )
                warnings_out = preview.setdefault("warnings", [])
                if quote_warning not in warnings_out:
                    warnings_out.append(quote_warning)
            sl_tp_valid = preview.get("sl_tp_valid")
            try:
                sl_tp_invalid = sl_tp_valid is not None and not bool(sl_tp_valid)
            except Exception:
                sl_tp_invalid = False
            if sl_tp_invalid:
                validation_payload = preview.get("validation")
                if isinstance(validation_payload, dict):
                    validation_payload["local_requirements_passed"] = False
                    validation_payload["live_submission_eligible"] = False
                    blockers = validation_payload.setdefault("blockers", [])
                    if "invalid_protection_levels" not in blockers:
                        blockers.append("invalid_protection_levels")
                preview["validation_passed"] = False
                preview["preview_ok"] = False
                sl_tp_error = str(preview.get("sl_tp_error") or "").strip()
                if sl_tp_error:
                    preview["validation_error"] = sl_tp_error
                    preview.setdefault("validation_code", "invalid_protection_levels")
            validation_payload = preview.get("validation")
            validation_blockers = (
                list(validation_payload.get("blockers") or [])
                if isinstance(validation_payload, dict)
                else list(local_blockers)
            )
            preview["blockers"] = validation_blockers
            if validation_blockers:
                preview["status"] = "preview_blocked"
                preview["no_action_reason"] = "dry_run_validation_blocked"
            preview_error = str(preview.get("preview_error") or "").strip()
            if preview_error:
                preview["success"] = False
                preview["error"] = preview_error
                preview.setdefault(
                    "error_code",
                    preview.get("preview_error_code") or "trade_preview_error",
                )
                validation_payload = preview.get("validation")
                if isinstance(validation_payload, dict):
                    validation_payload["live_submission_eligible"] = False
                    blockers = validation_payload.setdefault("blockers", [])
                    blocker = str(preview.get("error_code") or "preview_error")
                    if blocker not in blockers:
                        blockers.append(blocker)
                    if blocker in {
                        "invalid_pending_price",
                        "invalid_pending_order_levels",
                        "invalid_protection_levels",
                    }:
                        validation_payload["local_requirements_passed"] = False
                    preview["blockers"] = list(
                        validation_payload.get("blockers") or []
                    )
                preview["validation_passed"] = False
                preview["preview_ok"] = False
                preview["actionability"] = "preview_failed"
                preview["no_action"] = True
                preview["no_action_reason"] = "dry_run_preview_error"
                if preview.get("error_code") == "symbol_not_found":
                    preview["blockers"] = ["symbol_not_found"]
                    preview.pop("dry_run_note", None)
                    preview["actionability_reason"] = preview.get(
                        "remediation",
                        "Use symbols_list to find the broker's exact symbol name.",
                    )
                    if isinstance(validation_payload, dict):
                        validation_payload["local_requirements_passed"] = False
                        validation_payload["blockers"] = ["symbol_not_found"]
            validation_payload = preview.get("validation")
            local_requirements_passed = bool(
                isinstance(validation_payload, dict)
                and validation_payload.get("local_requirements_passed") is True
            )
            if preview.get("preview_ok") is not True:
                preview["success"] = False
                preview.setdefault("error_code", "preview_blocked")
                preview.setdefault("error", _dry_run_blocker_error(preview.get("blockers")))
                preview.setdefault(
                    "remediation",
                    "Resolve every blocker and run the dry-run preview again before submitting live.",
                )
            if not local_requirements_passed:
                final_safety_warning = (
                    "Dry run only. Local protection validation failed; no order was "
                    "sent and MT5/broker validation was not executed."
                    if "invalid_protection_levels"
                    in list(preview.get("blockers") or [])
                    else (
                        "Dry run only. Local safety requirements failed; no order was "
                        "sent and MT5/broker validation was not executed."
                    )
                )
            elif preview.get("preview_ok") is not True:
                final_safety_warning = (
                    "Dry run only. Local request checks passed, but preview validation "
                    "did not pass; no order was sent and MT5/broker validation was not "
                    "executed."
                )
            else:
                final_safety_warning = (
                    "Dry run only. Routing and local safety checks passed; "
                    "MT5/broker validation was not executed."
                )
            existing_warnings = [
                str(warning)
                for warning in list(preview.get("warnings") or [])
                if not str(warning).startswith("Dry run only.")
            ]
            preview["warnings"] = [final_safety_warning, *existing_warnings]
            if pending:
                preview["requested_price"] = request.price
                if request.stop_limit_price is not None:
                    preview["requested_stop_limit_price"] = request.stop_limit_price
            preview["magic"] = (
                request.magic
                if request.magic is not None
                else int(mt5_config.order_magic)
            )
            preview = comments._attach_comment_preview_metadata(
                preview,
                request.comment,
                default="mtdata pending order" if pending else "mtdata order",
            )
            if request.stop_loss not in (None, 0):
                preview["stop_loss"] = request.stop_loss
            if request.take_profit not in (None, 0):
                preview["take_profit"] = request.take_profit
            if expiration_provided:
                preview["expiration"] = request.expiration
                if normalized_expiration is not None:
                    preview["expiration_policy"] = "expires_at"
                    preview["expiration_normalized"] = normalized_expiration
                    preview["expiration_resolved_utc"] = (
                        datetime.fromtimestamp(
                            normalized_expiration,
                            tz=timezone.utc,
                        )
                        .isoformat()
                        .replace("+00:00", "Z")
                    )
                else:
                    preview["expiration_policy"] = "gtc"
                preview["expiration_explicit"] = True
            elif pending:
                preview["expiration_policy"] = "broker_default_gtc"
                preview["expiration_explicit"] = False
            return _shape_trade_place_preview(preview, detail=preview_detail)

        if not symbol_norm:
            missing.append("symbol")
        if request.volume is None:
            missing.append("volume")
        if request.order_type is None or (
            isinstance(request.order_type, str) and not request.order_type.strip()
        ):
            missing.append("order_type")
        if missing:
            return _finish(
                {
                    "error": (
                        f"Missing required field(s): {', '.join(missing)}. "
                        "Required: symbol, volume, order_type."
                    ),
                    "required": ["symbol", "volume", "order_type"],
                    "hint": (
                        "Example: symbol='BTCUSD', volume=0.03, "
                        "order_type='BUY_LIMIT'."
                    ),
                }
            )

        # Validate volume is positive
        try:
            vol_float = float(request.volume)
        except (TypeError, ValueError):
            return _finish({"error": "volume must be numeric"})
        if not math.isfinite(vol_float):
            return _finish({"error": "volume must be finite"})
        if vol_float <= 0:
            return _finish(
                {
                    "error": "volume must be positive",
                    "volume_received": vol_float,
                    "volume_hint": "volume must be greater than 0",
                }
            )

        order_type_norm, order_type_error = normalize_order_type_input(request.order_type)
        if order_type_error:
            return _finish(
                _invalid_order_type_payload(order_type_error),
                order_type=order_type_norm,
            )
        explicit_pending_types = validation._PENDING_ORDER_TYPES
        stop_limit_types = validation._STOP_LIMIT_ORDER_TYPES
        market_side_types = validation._MARKET_ORDER_TYPES

        price_provided = request.price is not None
        stop_limit_price_provided = request.stop_limit_price is not None
        try:
            normalized_expiration, expiration_provided = normalize_pending_expiration(
                request.expiration
            )
        except (TypeError, ValueError) as ex:
            return _finish(
                _invalid_pending_expiration_payload(
                    ex,
                    dry_run=bool(request.dry_run),
                ),
                order_type=order_type_norm,
            )

        ignore_market_gtc_expiration = (
            order_type_norm in market_side_types
            and not price_provided
            and expiration_provided
            and normalized_expiration is None
        )
        if (
            order_type_norm in market_side_types
            and not price_provided
            and expiration_provided
            and normalized_expiration is not None
        ):
            return _finish(
                {
                    "error": (
                        "expiration only applies to pending orders placed with a price. "
                        "For BUY/SELL market orders, omit expiration. "
                        "For pending orders, use BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP with price."
                    )
                },
                order_type=order_type_norm,
                pending=False,
            )

        if order_type_norm in market_side_types and price_provided:
            explicit_pending = (
                "BUY_LIMIT/BUY_STOP"
                if order_type_norm == "BUY"
                else "SELL_LIMIT/SELL_STOP"
            )
            return _finish(
                {
                    "error": (
                        f"Conflicting arguments: order_type={order_type_norm} is a market order, "
                        "but price was provided. Omit price for a market order, "
                        f"or use {explicit_pending} for a pending order."
                    ),
                    "order_type": order_type_norm,
                    "price": request.price,
                },
                order_type=order_type_norm,
                pending=False,
            )

        is_pending = (
            order_type_norm in explicit_pending_types
            or (expiration_provided and not ignore_market_gtc_expiration)
        )
        if order_type_norm in stop_limit_types and not stop_limit_price_provided:
            return _finish(
                {
                    "success": False,
                    "error": "stop_limit_price is required for stop-limit orders.",
                    "error_code": "invalid_stop_limit_price",
                    "order_type": order_type_norm,
                    "required": ["price", "stop_limit_price"],
                },
                order_type=order_type_norm,
                pending=True,
            )
        if order_type_norm not in stop_limit_types and stop_limit_price_provided:
            return _finish(
                {
                    "success": False,
                    "error": (
                        "stop_limit_price is valid only for BUY_STOP_LIMIT or "
                        "SELL_STOP_LIMIT orders."
                    ),
                    "error_code": "incompatible_parameters",
                    "order_type": order_type_norm,
                },
                order_type=order_type_norm,
                pending=is_pending,
            )
        basic_protection_error = validation._validate_basic_protection_levels(
            side=order_type_norm,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            entry_price=(
                request.stop_limit_price
                if order_type_norm in stop_limit_types
                else request.price
                if is_pending
                else None
            ),
        )
        if basic_protection_error is not None:
            if bool(request.dry_run):
                dry_run_protection_error = basic_protection_error
            else:
                return _finish(
                    basic_protection_error,
                    order_type=order_type_norm,
                    pending=is_pending,
                )
        if bool(request.require_sl_tp):
            missing_protection: List[str] = []
            if request.stop_loss in (None, 0):
                missing_protection.append("stop_loss")
            if request.take_profit in (None, 0):
                missing_protection.append("take_profit")
            if missing_protection:
                if bool(request.dry_run):
                    dry_run_missing_protection = list(missing_protection)
                else:
                    if not is_pending:
                        prevalidation_error = prevalidate_trade_place_market_input(
                            symbol_norm,
                            request.volume,
                        )
                        if prevalidation_error is not None:
                            return _finish(
                                prevalidation_error,
                                order_type=order_type_norm,
                                pending=is_pending,
                            )
                    order_kind = "pending order" if is_pending else "position"
                    return _finish(
                        {
                            "error": (
                                "require_sl_tp=True requires both stop_loss and take_profit. "
                                f"Refusing to place an unprotected {order_kind}."
                            ),
                            "require_sl_tp": True,
                            "missing": missing_protection,
                            "hint": (
                                "Provide both --stop-loss and --take-profit, "
                                "or explicitly set --require-sl-tp false. "
                                "Use trade_risk_analyze for position sizing or "
                                "forecast_barrier_optimize for barrier levels."
                            ),
                            "related_tools": [
                                "trade_risk_analyze",
                                "forecast_barrier_optimize",
                            ],
                        },
                        order_type=order_type_norm,
                        pending=is_pending,
                    )

        if is_pending and request.price is None:
            missing_price_payload: Dict[str, Any] = {
                "success": False,
                "error": "price is required for pending orders.",
                "error_code": "invalid_pending_price",
            }
            if request.dry_run:
                missing_price_payload.update(
                    {
                        "dry_run": True,
                        "no_action": True,
                        "would_send_order": False,
                        "preview_ok": False,
                        "validation_passed": False,
                        "blockers": ["invalid_pending_price"],
                        "validation": {
                            "local_requirements_passed": False,
                            "live_submission_eligible": False,
                            "blockers": ["invalid_pending_price"],
                            "broker_validation_performed": False,
                        },
                    }
                )
            return _finish(
                missing_price_payload,
                order_type=order_type_norm,
                pending=is_pending,
            )

        if bool(request.dry_run):
            order_preview: Optional[Dict[str, Any]] = None
            if dry_run_protection_error is None and callable(build_dry_run_preview):
                order_preview = build_dry_run_preview(
                    symbol=symbol_norm,
                    volume=float(request.volume),
                    order_type=order_type_norm,
                    pending=is_pending,
                    price=request.price,
                    stop_limit_price=request.stop_limit_price,
                    stop_loss=request.stop_loss,
                    take_profit=request.take_profit,
                )
            if str((order_preview or {}).get("preview_error") or "").strip():
                return _finish(
                    _dry_run_preview(
                        order_type=order_type_norm,
                        pending=is_pending,
                        normalized_expiration=normalized_expiration,
                        expiration_provided=expiration_provided,
                        guardrail_preview={
                            "enabled": bool(trade_guardrails_config.is_enabled())
                        },
                        order_preview=order_preview,
                    ),
                    order_type=order_type_norm,
                    pending=is_pending,
                )
            entry_price = validation.coerce_finite_float(
                (order_preview or {}).get("estimated_fill_price")
            )
            guardrail_account_info = _best_effort_trade_guardrail_account_info()
            snapshot_required = guardrails_require_position_snapshot(
                trade_guardrails_config,
                account_info=guardrail_account_info,
                enforce_account_risk=True,
                enforce_wallet_risk=True,
                for_live_projection=True,
            )
            guardrail_positions = (
                _best_effort_trade_guardrail_positions()
                if snapshot_required
                else []
            )
            if guardrail_positions is None:
                return _finish(
                    validation.snapshot_unavailable_error(
                        mt5_adapter,
                        snapshot="positions",
                        context="preview configured trade guardrails",
                        guardrail_blocked=True,
                    ),
                    order_type=order_type_norm,
                    pending=is_pending,
                )
            pending_snapshot_required = guardrails_require_pending_snapshot(
                trade_guardrails_config,
                account_info=guardrail_account_info,
                enforce_account_risk=True,
                enforce_wallet_risk=True,
                for_live_projection=True,
            )
            guardrail_pending_orders = (
                _best_effort_trade_guardrail_pending_orders()
                if pending_snapshot_required
                else []
            )
            if guardrail_pending_orders is None:
                return _finish(
                    validation.snapshot_unavailable_error(
                        mt5_adapter,
                        snapshot="orders",
                        context="preview configured trade guardrails",
                        guardrail_blocked=True,
                    ),
                    order_type=order_type_norm,
                    pending=is_pending,
                )
            guardrail_preview = preview_trade_guardrails(
                trade_guardrails_config,
                symbol=symbol_norm,
                volume=float(request.volume),
                stop_loss=request.stop_loss,
                deviation=request.deviation,
                side=_guardrail_order_side(order_type_norm),
                entry_price=entry_price,
                account_info=guardrail_account_info,
                existing_positions=guardrail_positions,
                existing_pending_orders=guardrail_pending_orders,
                symbol_info_resolver=mt5_adapter.symbol_info,
            )
            if guardrail_preview.get("blocked"):
                violations = list(guardrail_preview.get("violations") or [])
                guardrail_rule = str(guardrail_preview.get("rule") or "").strip()
                error_message = "Trade would be blocked by configured guardrails."
                if violations:
                    prefix = (
                        f"Trade blocked by guardrails ({guardrail_rule})"
                        if guardrail_rule
                        else "Trade blocked by guardrails"
                    )
                    error_message = f"{prefix}: {violations[0]}"
                blocked_payload = {
                    "error": error_message,
                    "guardrail_blocked": True,
                    "dry_run": True,
                    "no_action": True,
                    "actionability": "blocked_by_guardrails",
                    "guardrails_preview": guardrail_preview,
                    "violations": violations,
                }
                for key in (
                    "error_code",
                    "allowed_symbols_sample",
                    "allowed_symbols_count",
                    "suggestion",
                    "guardrail_context",
                ):
                    value = guardrail_preview.get(key)
                    if value not in (None, "", []):
                        blocked_payload[key] = value
                return _finish(
                    blocked_payload,
                    order_type=order_type_norm,
                    pending=is_pending,
                )
            return _finish(
                _dry_run_preview(
                    order_type=order_type_norm,
                    pending=is_pending,
                    normalized_expiration=normalized_expiration,
                    expiration_provided=expiration_provided,
                    guardrail_preview=guardrail_preview,
                    order_preview=order_preview,
                ),
                order_type=order_type_norm,
                pending=is_pending,
            )

        guardrail_account_info = _best_effort_trade_guardrail_account_info()
        snapshot_required = guardrails_require_position_snapshot(
            trade_guardrails_config,
            account_info=guardrail_account_info,
            enforce_account_risk=False,
            enforce_wallet_risk=False,
        )
        guardrail_positions = (
            _best_effort_trade_guardrail_positions() if snapshot_required else []
        )
        if guardrail_positions is None:
            return _finish(
                validation.snapshot_unavailable_error(
                    mt5_adapter,
                    snapshot="positions",
                    context="evaluate configured trade guardrails",
                    guardrail_blocked=True,
                ),
                order_type=order_type_norm,
                pending=is_pending,
            )
        static_guardrail = evaluate_trade_guardrails(
            trade_guardrails_config,
            symbol=symbol_norm,
            volume=float(request.volume),
            stop_loss=request.stop_loss,
            deviation=request.deviation,
            side=_guardrail_order_side(order_type_norm),
            account_info=guardrail_account_info,
            existing_positions=guardrail_positions,
            enforce_account_risk=False,
            enforce_wallet_risk=False,
        )
        if static_guardrail is not None:
            return _finish(static_guardrail, order_type=order_type_norm, pending=is_pending)

        if not is_pending:
            result = place_market_order(
                symbol=symbol_norm,
                volume=float(request.volume),
                order_type=order_type_norm,
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
                comment=request.comment,
                magic=request.magic,
                deviation=request.deviation,
            )
            if isinstance(result, dict):
                sl_tp_requested, sl_tp_status = _sl_tp_result_details(result)
                sl_tp_failed = sl_tp_status == "failed"
                sl_tp_unverified = sl_tp_status not in {"applied", "failed"}
                if sl_tp_requested and (sl_tp_failed or sl_tp_unverified):
                    warnings_out = _coerce_warning_list(result.get("warnings"))
                    pos_ticket = result.get("position_ticket")
                    candidate_tickets = [
                        ticket
                        for ticket in list(result.get("position_ticket_candidates") or [])
                        if ticket is not None
                    ]
                    if sl_tp_failed and pos_ticket is not None:
                        critical = (
                            "CRITICAL: Order executed without applied TP/SL protection. "
                            f"Run trade_modify --ticket {pos_ticket} now, or close the position."
                        )
                    elif sl_tp_failed and candidate_tickets:
                        candidate_list = ", ".join(str(v) for v in candidate_tickets)
                        critical = (
                            "CRITICAL: Order executed without applied TP/SL protection. "
                            f"Try trade_modify --ticket {candidate_tickets[0]} now "
                            f"(candidate tickets: {candidate_list}). "
                            "If that fails, run trade_get_open to confirm the live position ticket, "
                            "or close the position."
                        )
                    elif sl_tp_failed:
                        critical = (
                            "CRITICAL: Order executed without applied TP/SL protection. "
                            "Run trade_get_open to find the live position ticket, then use "
                            "trade_modify --ticket TICKET now, "
                            "or close the position."
                        )
                    else:
                        critical = (
                            "CRITICAL: TP/SL attachment was accepted but could not be verified. "
                            "Confirm the live protection levels before treating this position as protected."
                        )
                    if critical not in warnings_out:
                        warnings_out.append(critical)
                    if warnings_out:
                        result["warnings"] = warnings_out
                    close_ticket = safe_int_ticket(pos_ticket)
                    if close_ticket is None:
                        for candidate_ticket in candidate_tickets:
                            close_ticket = safe_int_ticket(candidate_ticket)
                            if close_ticket is not None:
                                break
                    if close_ticket is None:
                        auto_close_result: Dict[str, Any] = {
                            "error": "Auto-close skipped: position_ticket unavailable."
                        }
                    else:
                        auto_close_result = close_positions(
                            ticket=close_ticket,
                            volume=(
                                validation.coerce_finite_float(
                                    result.get("filled_volume")
                                )
                                or float(request.volume)
                            ),
                            comment="AUTO-CLOSE: TP/SL protection unresolved",
                            deviation=request.deviation,
                        )
                    result["auto_close_on_sl_tp_fail"] = True
                    result["auto_close_result"] = auto_close_result

                    auto_close_ok = bool(
                        isinstance(auto_close_result, dict)
                        and (
                            auto_close_result.get("success") is True
                            or (
                                "success" not in auto_close_result
                                and auto_close_result.get("retcode") == 10009
                            )
                        )
                    )
                    if auto_close_ok:
                        result["protection_status"] = "auto_closed_after_sl_tp_fail"
                        result["success"] = False
                    else:
                        warnings_out = _coerce_warning_list(result.get("warnings"))
                        auto_close_warning = (
                            "AUTO-CLOSE FAILED: position protection remains unresolved; "
                            "reconcile and close immediately."
                        )
                        if auto_close_warning not in warnings_out:
                            warnings_out.append(auto_close_warning)
                        result["warnings"] = warnings_out
                        result["success"] = False

                    if sl_tp_unverified:
                        result.setdefault(
                            "error",
                            "Order was executed, but TP/SL protection could not be verified.",
                        )
                        result.setdefault("error_code", "protection_not_verified")
                        result.setdefault("protection_status", "protection_unverified")
                        result["success"] = False

                if (
                    bool(request.require_sl_tp)
                    and sl_tp_requested
                    and (sl_tp_failed or sl_tp_unverified)
                    and "error" not in result
                ):
                    result["error"] = (
                        "Order was executed, but TP/SL protection could not be applied."
                        if sl_tp_failed
                        else "Order was executed, but TP/SL protection could not be verified."
                    )
                    result["require_sl_tp"] = bool(request.require_sl_tp)
                    result["error_code"] = (
                        "protection_not_applied"
                        if sl_tp_failed
                        else "protection_not_verified"
                    )
                    result["success"] = False
                    result["protection_status"] = (
                        result.get("protection_status")
                        or (
                            "unprotected_position"
                            if sl_tp_failed
                            else "protection_unverified"
                        )
                    )
            return _finish(result, order_type=order_type_norm, pending=is_pending)
        return _finish(
            place_pending_order(
                symbol=symbol_norm,
                volume=float(request.volume),
                order_type=order_type_norm,
                price=request.price,
                stop_limit_price=request.stop_limit_price,
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
                expiration=request.expiration,
                comment=request.comment,
                magic=request.magic,
                deviation=request.deviation,
            ),
            order_type=order_type_norm,
            pending=is_pending,
        )
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
