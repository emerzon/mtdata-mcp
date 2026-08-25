"""Trade modification and closure workflows for MetaTrader integration."""

import logging
import math
import time as _stdlib_time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

from ...bootstrap.settings import trade_guardrails_config
from ...utils.freshness import QUOTE_LIVE_SECONDS
from ...utils.quote import resolve_quote_tick
from ...utils.utils import _utc_epoch_seconds
from . import comments, time, validation
from .common import build_trade_quote_context
from .gateway import MT5TradingGateway, create_trading_gateway, trading_connection_error
from .positions import _resolve_open_position, _resolve_pending_order
from .safety import (
    _build_guardrail_block,
    _resolve_pending_order_side,
    evaluate_trade_guardrails,
    load_guardrail_book_snapshots,
    pending_order_risk_increased,
)
from .time import ExpirationValue
from .validation import (
    _normalize_protection_level,
    _protection_level_tolerance,
    _protection_levels_match,
)

logger = logging.getLogger(__name__)


def _resolve_position_side(position: Any, mt5: Any) -> Optional[str]:
    return validation._resolve_position_side(position, mt5)


def _trading_symbol_or_error(mt5: Any, symbol: Optional[str]) -> Optional[Dict[str, Any]]:
    symbol_value = str(symbol or "").strip()
    if not symbol_value:
        return None
    from mtdata.core.trading.use_cases.common import _validate_trading_symbol

    return _validate_trading_symbol(mt5, symbol_value)


def _position_matches_any_ticket(position: Any, ticket_values: set[int]) -> bool:
    if not ticket_values:
        return False
    for field in ("ticket", "identifier", "position_id", "position", "order", "deal"):
        value = validation._safe_int_ticket(getattr(position, field, None))
        if value is not None and value in ticket_values:
            return True
    return False


def _position_modify_readback(
    mt5: Any,
    *,
    resolved_ticket: Optional[int],
    price_increment: float,
    digits: int,
    price_tol: float,
) -> Optional[Dict[str, Optional[float]]]:
    try:
        row = validation._exact_ticket_row(
            mt5.positions_get(ticket=resolved_ticket),
            resolved_ticket,
        )
    except Exception:
        row = None
    if row is None:
        return None
    actual_sl = validation._normalize_price_for_symbol(
        getattr(row, "sl", None),
        point=price_increment,
        digits=digits,
    )
    actual_tp = validation._normalize_price_for_symbol(
        getattr(row, "tp", None),
        point=price_increment,
        digits=digits,
    )
    return {
        "applied_sl": _normalize_protection_level(actual_sl, tol=price_tol),
        "applied_tp": _normalize_protection_level(actual_tp, tol=price_tol),
    }


def _pending_modify_readback(
    mt5: Any,
    *,
    resolved_ticket: Optional[int],
    price_increment: float,
    digits: int,
    price_tol: float,
) -> Optional[Dict[str, Any]]:
    try:
        row = validation._exact_ticket_row(
            mt5.orders_get(ticket=resolved_ticket),
            resolved_ticket,
        )
    except Exception:
        row = None
    if row is None:
        return None

    def _price(name: str) -> Optional[float]:
        normalized = validation._normalize_price_for_symbol(
            getattr(row, name, None),
            point=price_increment,
            digits=digits,
        )
        return _normalize_protection_level(normalized, tol=price_tol)

    return {
        "applied_price": _price("price_open"),
        "applied_stop_limit_price": _price("price_stoplimit"),
        "applied_sl": _price("sl"),
        "applied_tp": _price("tp"),
        "applied_expiration": validation._safe_int_attr(
            row,
            "time_expiration",
            None,
        ),
        "applied_type_time": validation._safe_int_attr(row, "type_time", None),
    }


def _readback_adjustments(
    requested: Dict[str, Any],
    applied: Dict[str, Any],
    *,
    price_tol: float,
) -> Dict[str, Dict[str, Any]]:
    adjustments: Dict[str, Dict[str, Any]] = {}
    for key, requested_value in requested.items():
        applied_value = applied.get(key)
        if key in {
            "applied_price",
            "applied_stop_limit_price",
            "applied_sl",
            "applied_tp",
        }:
            matches = _protection_levels_match(
                requested_value,
                applied_value,
                tol=price_tol,
            )
        else:
            matches = requested_value == applied_value
        if not matches:
            adjustments[key.removeprefix("applied_")] = {
                "requested": requested_value,
                "applied": applied_value,
            }
    return adjustments


def _trade_rows_excluding_current(
    rows: Optional[List[Any]],
    *,
    resolved_ticket: Optional[int],
    requested_ticket: Optional[int],
) -> List[Any]:
    ticket_values = {
        value
        for value in (
            validation._safe_int_ticket(resolved_ticket),
            validation._safe_int_ticket(requested_ticket),
        )
        if value is not None
    }
    return [
        row
        for row in list(rows or [])
        if not _position_matches_any_ticket(row, ticket_values)
    ]


def _evaluate_position_modify_guardrails(
    mt5: Any,
    *,
    position: Any,
    resolved_ticket: Optional[int],
    requested_ticket: Optional[int],
    side: str,
    symbol_info: Any,
    current_stop_loss: Optional[float],
    candidate_stop_loss: Optional[float],
) -> Optional[Dict[str, Any]]:
    position_volume = validation._safe_float_attr(position, "volume")
    entry_price = validation._safe_float_attr(position, "price_current")
    if entry_price is None or not math.isfinite(float(entry_price)) or entry_price <= 0.0:
        entry_price = validation._safe_float_attr(position, "price_open")
    if not trade_guardrails_config.is_enabled() or position_volume is None:
        return None
    current_has_stop = bool(
        current_stop_loss is not None
        and math.isfinite(float(current_stop_loss))
        and float(current_stop_loss) > 0.0
    )
    candidate_has_stop = bool(
        candidate_stop_loss is not None
        and math.isfinite(float(candidate_stop_loss))
        and float(candidate_stop_loss) > 0.0
    )
    if current_has_stop and not candidate_has_stop:
        risk_increased = True
    elif not current_has_stop or not candidate_has_stop:
        risk_increased = False
    elif side == "BUY":
        risk_increased = float(candidate_stop_loss) < float(current_stop_loss)
    else:
        risk_increased = float(candidate_stop_loss) > float(current_stop_loss)
    stop_loss_required = bool(
        trade_guardrails_config.safety_policy.require_stop_loss
        or trade_guardrails_config.wallet_risk_limits.has_configured_values()
    )
    if not risk_increased and not (stop_loss_required and not candidate_has_stop):
        return None
    if stop_loss_required and not candidate_has_stop:
        block = _build_guardrail_block(
            ["Configured trading safety requires a stop-loss."],
            rule="safety_policy",
            context={"symbol": position.symbol, "side": side},
        )
        block["position_ticket"] = resolved_ticket
        block["ticket_requested"] = requested_ticket
        return block

    try:
        account_info = mt5.account_info()
    except Exception:
        account_info = None
    positions, pending_orders, snapshot_error = load_guardrail_book_snapshots(
        mt5,
        trade_guardrails_config,
        account_info=account_info,
    )
    if snapshot_error is not None:
        return snapshot_error
    guardrail_block = evaluate_trade_guardrails(
        trade_guardrails_config,
        symbol=position.symbol,
        volume=abs(float(position_volume)),
        stop_loss=candidate_stop_loss,
        deviation=None,
        side=side,
        entry_price=entry_price,
        account_info=account_info,
        existing_positions=_trade_rows_excluding_current(
            positions,
            resolved_ticket=resolved_ticket,
            requested_ticket=requested_ticket,
        ),
        existing_pending_orders=pending_orders,
        symbol_info=symbol_info,
        symbol_info_resolver=mt5.symbol_info,
        enforce_symbol_rules=False,
        enforce_safety_policy=False,
    )
    if guardrail_block is not None:
        guardrail_block["position_ticket"] = resolved_ticket
        guardrail_block["ticket_requested"] = requested_ticket
    return guardrail_block


def _position_no_change_result(
    mt5: Any,
    *,
    resolved_ticket: Optional[int],
    requested_ticket: int,
    ticket_resolution: Dict[str, Any],
    desired_sl: Optional[float],
    desired_tp: Optional[float],
    dry_run: bool,
) -> Dict[str, Any]:
    no_change_code = validation._safe_int_attr(mt5, "TRADE_RETCODE_NO_CHANGES", 10025)
    out: Dict[str, Any] = {
        "success": True,
        "retcode": no_change_code,
        "retcode_name": mt5.retcode_name(no_change_code),
        "comment": "No changes",
        "position_ticket": resolved_ticket,
        "ticket_requested": requested_ticket,
        "ticket_resolution": ticket_resolution,
        "applied_sl": desired_sl,
        "applied_tp": desired_tp,
        "no_change": True,
        "message": "Requested SL/TP already match the live position.",
    }
    if dry_run:
        out.update(
            {
                "dry_run": True,
                "actionability": "preview_only",
                "operation": "modify_position",
                "scope": "positions",
                "would_send_order": False,
            }
        )
    return out


def _build_position_modify_request(
    mt5: Any,
    *,
    position: Any,
    resolved_ticket: Optional[int],
    stop_loss: float,
    take_profit: float,
    comment: Optional[str],
) -> Dict[str, Any]:
    request: Dict[str, Any] = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": position.symbol,
        "position": resolved_ticket,
        "sl": stop_loss,
        "tp": take_profit,
        "comment": comments._normalize_trade_comment(
            comment,
            default="mtdata modify position",
        ),
    }
    request_magic = validation._safe_int_magic(getattr(position, "magic", None))
    if request_magic is not None:
        request["magic"] = request_magic
    return request


def _unexpected_operation_error(
    operation: str,
    exc: Exception,
    *,
    mt5: Any,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    logger.error(
        "Unexpected error while %s",
        operation,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    payload: Dict[str, Any] = {
        "error": f"Unexpected error while {operation}",
        "error_code": "internal_execution_error",
    }
    if context:
        payload["context"] = context
    return payload


def _deal_history_sort_key(row: Any) -> float:
    return validation._time_sort_key(
        row,
        ("time_msc", "time", "time_update_msc", "time_update"),
    )


def _deal_history_selection_key(row: Any) -> tuple[float, int]:
    ticket = validation._safe_int_ticket(getattr(row, "ticket", None))
    return _deal_history_sort_key(row), ticket if ticket is not None else -1


def _resolve_closed_deal_from_history(
    mt5: Any,
    *,
    result: Any,
    position: Any,
    closed_at_utc: datetime,
) -> Optional[Any]:
    deal_ticket = validation._safe_int_ticket(getattr(result, "deal", None))
    order_ticket = validation._safe_int_ticket(getattr(result, "order", None))
    position_ticket = validation._safe_int_ticket(getattr(position, "ticket", None))
    if deal_ticket is None and order_ticket is None and position_ticket is None:
        return None
    try:
        rows = mt5.history_deals_get(
            _utc_epoch_seconds(closed_at_utc - timedelta(minutes=5)),
            _utc_epoch_seconds(closed_at_utc + timedelta(minutes=1)),
        )
    except Exception:
        return None
    if not rows:
        return None

    exact_deal_matches: List[Any] = []
    order_matches: List[Any] = []
    position_matches: List[Any] = []
    for row in rows:
        row_ticket = validation._safe_int_ticket(getattr(row, "ticket", None))
        row_order = validation._safe_int_ticket(getattr(row, "order", None))
        row_position_candidates = {
            validation._safe_int_ticket(getattr(row, "position_id", None)),
            validation._safe_int_ticket(getattr(row, "position_by_id", None)),
            validation._safe_int_ticket(getattr(row, "position", None)),
        }
        row_position_candidates.discard(None)
        if deal_ticket is not None and row_ticket == deal_ticket:
            exact_deal_matches.append(row)
            continue
        if order_ticket is not None and row_order == order_ticket:
            order_matches.append(row)
            continue
        if position_ticket is not None and position_ticket in row_position_candidates:
            position_matches.append(row)

    for matches in (exact_deal_matches, order_matches, position_matches):
        if matches:
            return max(matches, key=_deal_history_selection_key)
    return None


def _count_done_results(mt5: Any, results: List[Dict[str, Any]]) -> int:
    success_count = 0
    for item in results:
        if (
            item.get("success") is not False
            and validation._trade_execution_status(mt5, item.get("retcode"))
            == "complete"
        ):
            success_count += 1
    return success_count


def _protection_removed_fields(
    explicit_remove_sl: bool,
    explicit_remove_tp: bool,
) -> List[str]:
    removed: List[str] = []
    if explicit_remove_sl:
        removed.append("stop_loss")
    if explicit_remove_tp:
        removed.append("take_profit")
    return removed


def _modify_freshness_block(
    freshness_error: Dict[str, Any],
    *,
    dry_run: bool,
    ticket: Any,
    symbol: str,
    applied_sl: Any = None,
    applied_tp: Any = None,
    protection_removed: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not dry_run:
        return freshness_error
    out: Dict[str, Any] = {
        "success": False,
        "dry_run": True,
        "preview_ok": False,
        "would_send_order": False,
        "error_code": "preview_blocked",
        "ticket": ticket,
        "symbol": symbol,
        **freshness_error,
    }
    if applied_sl is not None:
        out["applied_sl"] = applied_sl
    if applied_tp is not None:
        out["applied_tp"] = applied_tp
    if protection_removed:
        out["protection_removed"] = list(protection_removed)
    return out


def _modify_position(  # noqa: C901
    ticket: Union[int, str],
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    comment: Optional[str] = None,
    dry_run: bool = False,
    gateway: Optional[MT5TradingGateway] = None,
) -> dict:
    """Internal helper to modify a position by ticket."""
    mt5 = create_trading_gateway(
        gateway=gateway,
    )

    connection_error = trading_connection_error(mt5)
    if connection_error is not None:
        return connection_error

    def _modify_position():  # noqa: C901
        try:
            ticket_id = int(ticket)
            position, resolved_ticket, ticket_resolution = _resolve_open_position(
                mt5,
                ticket_candidates=[ticket_id],
                require_exact_ticket_match=True,
                allow_alternate_ticket_match=True,
            )
            if position is None or resolved_ticket is None:
                out = {"error": f"Position {ticket} not found", "checked_scopes": ["positions"]}
                if isinstance(ticket_resolution, dict):
                    out["ticket_resolution"] = ticket_resolution
                return out

            symbol_info = mt5.symbol_info(position.symbol)
            if symbol_info is None:
                return {"error": f"Failed to get symbol info for {position.symbol}"}

            price_inputs, price_inputs_error = validation._normalize_trade_price_inputs(
                symbol_info=symbol_info,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            if price_inputs_error is not None:
                return {"error": price_inputs_error}
            point = float(price_inputs["point"])
            price_increment = float(price_inputs["price_increment"])
            digits = int(price_inputs["digits"])
            requested_sl = price_inputs["stop_loss"]
            requested_tp = price_inputs["take_profit"]
            explicit_remove_sl = bool(price_inputs["explicit_remove_stop_loss"])
            explicit_remove_tp = bool(price_inputs["explicit_remove_take_profit"])

            # Normalize SL/TP values
            existing_sl = validation._normalize_price_for_symbol(
                getattr(position, "sl", None),
                point=price_increment,
                digits=digits,
            )
            existing_tp = validation._normalize_price_for_symbol(
                getattr(position, "tp", None),
                point=price_increment,
                digits=digits,
            )
            norm_sl = (
                0.0
                if explicit_remove_sl
                else (
                    float(requested_sl)
                    if stop_loss is not None
                    else float(existing_sl or 0.0)
                )
            )
            norm_tp = (
                0.0
                if explicit_remove_tp
                else (
                    float(requested_tp)
                    if take_profit is not None
                    else float(existing_tp or 0.0)
                )
            )
            validate_sl = (
                float(norm_sl)
                if not explicit_remove_sl and float(norm_sl) > 0.0
                else None
            )
            validate_tp = (
                float(norm_tp)
                if not explicit_remove_tp and float(norm_tp) > 0.0
                else None
            )

            price_tol = _protection_level_tolerance(point=point)
            current_sl = _normalize_protection_level(existing_sl, tol=price_tol)
            current_tp = _normalize_protection_level(existing_tp, tol=price_tol)
            desired_sl = _normalize_protection_level(norm_sl, tol=price_tol)
            desired_tp = _normalize_protection_level(norm_tp, tol=price_tol)

            if _protection_levels_match(current_sl, desired_sl, tol=price_tol) and _protection_levels_match(current_tp, desired_tp, tol=price_tol):
                return _position_no_change_result(
                    mt5,
                    resolved_ticket=resolved_ticket,
                    requested_ticket=ticket_id,
                    ticket_resolution=ticket_resolution,
                    desired_sl=desired_sl,
                    desired_tp=desired_tp,
                    dry_run=dry_run,
                )

            side = _resolve_position_side(position, mt5)
            if side is None:
                return {"error": "Unable to determine position side for protection validation."}
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                return {"error": f"Failed to get current price for {position.symbol}"}
            freshness_error = validation._validate_tick_freshness(
                tick,
                symbol=position.symbol,
                max_age_seconds=_CLOSE_TICK_MAX_AGE_SECONDS,
            )
            if freshness_error is not None:
                return _modify_freshness_block(
                    freshness_error,
                    dry_run=dry_run,
                    ticket=ticket_id,
                    symbol=position.symbol,
                    applied_sl=0.0 if explicit_remove_sl else desired_sl,
                    applied_tp=0.0 if explicit_remove_tp else desired_tp,
                    protection_removed=_protection_removed_fields(
                        explicit_remove_sl,
                        explicit_remove_tp,
                    ),
                )
            live_protection_error = validation._validate_live_protection_levels(
                symbol_info=symbol_info,
                tick=tick,
                side=side,
                stop_loss=validate_sl,
                take_profit=validate_tp,
            )
            if live_protection_error is not None:
                return live_protection_error

            guardrail_block = _evaluate_position_modify_guardrails(
                mt5,
                position=position,
                resolved_ticket=resolved_ticket,
                requested_ticket=ticket_id,
                side=side,
                symbol_info=symbol_info,
                current_stop_loss=current_sl,
                candidate_stop_loss=desired_sl,
            )
            if guardrail_block is not None:
                guardrail_block["ticket_resolution"] = ticket_resolution
                return guardrail_block

            request = _build_position_modify_request(
                mt5,
                position=position,
                resolved_ticket=resolved_ticket,
                stop_loss=norm_sl,
                take_profit=norm_tp,
                comment=comment,
            )

            if dry_run:
                preview = {
                    "success": True,
                    "dry_run": True,
                    "actionability": "preview_only",
                    "operation": "modify_position",
                    "scope": "positions",
                    "ticket": ticket_id,
                    "position_ticket": resolved_ticket,
                    "ticket_requested": ticket_id,
                    "ticket_resolution": ticket_resolution,
                    "symbol": position.symbol,
                    "would_send_order": False,
                    "preview_ok": True,
                    "preview_scope_summary": (
                        "Validated position routing and protection levels; no modify request was sent to MT5."
                    ),
                    "applied_sl": 0.0 if explicit_remove_sl else desired_sl,
                    "applied_tp": 0.0 if explicit_remove_tp else desired_tp,
                    "not_estimated": [
                        "broker_acceptance",
                        "execution_latency",
                    ],
                }
                removed = _protection_removed_fields(
                    explicit_remove_sl,
                    explicit_remove_tp,
                )
                if removed:
                    preview["protection_removed"] = removed
                return preview

            result, comment_fallback, last_error = comments._send_order_with_comment_fallback(
                mt5,
                request,
            )
            if result is None:
                out = {
                    "error": (
                        "Position modification outcome is unknown; the broker may "
                        "have applied the requested SL/TP changes. Confirm the live "
                        "position before retrying."
                    ),
                    "error_code": "order_send_ambiguous",
                    "ambiguous": True,
                    "position_ticket": resolved_ticket,
                    "ticket_requested": ticket_id,
                    "last_error": last_error,
                }
                if isinstance(comment_fallback, dict):
                    out["comment_fallback"] = comment_fallback
                return out

            result_retcode = getattr(result, "retcode", None)
            no_change_code = validation._safe_int_attr(mt5, "TRADE_RETCODE_NO_CHANGES", 10025)

            if result_retcode == no_change_code:
                refreshed_position = None
                try:
                    refreshed_rows = mt5.positions_get(ticket=resolved_ticket)
                except Exception:
                    refreshed_rows = None
                if refreshed_rows:
                    try:
                        refreshed_position = list(refreshed_rows)[0]
                    except Exception:
                        refreshed_position = None
                refreshed_sl = validation._normalize_price_for_symbol(
                    getattr(refreshed_position, "sl", None) if refreshed_position is not None else getattr(position, "sl", None),
                    point=price_increment,
                    digits=digits,
                )
                refreshed_tp = validation._normalize_price_for_symbol(
                    getattr(refreshed_position, "tp", None) if refreshed_position is not None else getattr(position, "tp", None),
                    point=price_increment,
                    digits=digits,
                )
                final_sl = _normalize_protection_level(refreshed_sl, tol=price_tol)
                final_tp = _normalize_protection_level(refreshed_tp, tol=price_tol)
                if _protection_levels_match(final_sl, desired_sl, tol=price_tol) and _protection_levels_match(final_tp, desired_tp, tol=price_tol):
                    out = {
                        "success": True,
                        "retcode": result_retcode,
                        "retcode_name": mt5.retcode_name(result_retcode),
                        "deal": result.deal,
                        "order": result.order,
                        "comment": result.comment,
                        "mt5_request_id": result.request_id,
                        "position_ticket": resolved_ticket,
                        "ticket_requested": ticket_id,
                        "ticket_resolution": ticket_resolution,
                        "applied_sl": final_sl,
                        "applied_tp": final_tp,
                        "no_change": True,
                        "message": "Requested SL/TP already match the live position.",
                    }
                    if isinstance(comment_fallback, dict):
                        out["comment_fallback"] = comment_fallback
                    return out

                out = {
                    "error": "Broker reported no changes, but live SL/TP do not match the requested values.",
                    "retcode": result_retcode,
                    "retcode_name": mt5.retcode_name(result_retcode),
                    "comment": result.comment,
                    "mt5_request_id": result.request_id,
                    "position_ticket": resolved_ticket,
                    "ticket_requested": ticket_id,
                    "ticket_resolution": ticket_resolution,
                    "desired_sl": desired_sl,
                    "desired_tp": desired_tp,
                    "actual_sl": final_sl,
                    "actual_tp": final_tp,
                    "no_change": True,
                    "last_error": last_error,
                }
                if isinstance(comment_fallback, dict):
                    out["comment_fallback"] = comment_fallback
                return out

            if result_retcode != mt5.TRADE_RETCODE_DONE:
                out = {
                    "error": "Failed to modify position",
                    "retcode": result_retcode,
                    "retcode_name": mt5.retcode_name(result_retcode),
                    "comment": result.comment,
                    "mt5_request_id": result.request_id,
                    "last_error": last_error,
                }
                if isinstance(comment_fallback, dict):
                    out["comment_fallback"] = comment_fallback
                return out

            readback = _position_modify_readback(
                mt5,
                resolved_ticket=resolved_ticket,
                price_increment=price_increment,
                digits=digits,
                price_tol=price_tol,
            )
            out = {
                "success": True,
                "retcode": result_retcode,
                "retcode_name": mt5.retcode_name(result_retcode),
                "deal": result.deal,
                "order": result.order,
                "comment": result.comment,
                "mt5_request_id": result.request_id,
                "position_ticket": resolved_ticket,
                "ticket_requested": ticket_id,
                "ticket_resolution": ticket_resolution,
                "requested_sl": desired_sl,
                "requested_tp": desired_tp,
            }
            if readback is None:
                out.update(
                    {
                        "applied_sl": None,
                        "applied_tp": None,
                        "verification_status": "unverified",
                        "warnings": [
                            "Broker accepted the position modification, but the live "
                            "SL/TP values could not be read back."
                        ],
                    }
                )
            else:
                out.update(readback)
                out["verification_status"] = "verified"
                adjustments = _readback_adjustments(
                    {
                        "applied_sl": desired_sl,
                        "applied_tp": desired_tp,
                    },
                    readback,
                    price_tol=price_tol,
                )
                if adjustments:
                    out["broker_adjusted"] = True
                    out["adjustment"] = adjustments
                    out["warnings"] = [
                        "Broker-reported SL/TP values differ from the requested modification."
                    ]
            if isinstance(comment_fallback, dict):
                out["comment_fallback"] = comment_fallback
            return out

        except Exception as e:
            return _unexpected_operation_error(
                "modifying position",
                e,
                mt5=mt5,
                context={"ticket": ticket},
            )

    return _modify_position()


def _modify_pending_order(  # noqa: C901
    ticket: Union[int, str],
    price: Optional[Union[int, float]] = None,
    stop_limit_price: Optional[Union[int, float]] = None,
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    expiration: Optional[ExpirationValue] = None,
    comment: Optional[str] = None,
    dry_run: bool = False,
    gateway: Optional[MT5TradingGateway] = None,
) -> dict:
    """Internal helper to modify a pending order by ticket."""
    mt5 = create_trading_gateway(
        gateway=gateway,
    )

    connection_error = trading_connection_error(mt5)
    if connection_error is not None:
        return connection_error

    def _modify_pending_order():  # noqa: C901
        try:
            ticket_id = int(ticket)
            order, resolved_ticket, ticket_resolution = _resolve_pending_order(
                mt5,
                ticket_candidates=[ticket_id],
                require_exact_ticket_match=True,
            )
            if order is None:
                out = {"error": f"Pending order {ticket} not found", "checked_scopes": ["pending_orders"]}
                if isinstance(ticket_resolution, dict):
                    out["ticket_resolution"] = ticket_resolution
                return out
            normalized_expiration, expiration_specified = time._normalize_pending_expiration(expiration)
            symbol_info = mt5.symbol_info(order.symbol)
            if symbol_info is None:
                return {"error": f"Failed to get symbol info for {order.symbol}"}
            tick = mt5.symbol_info_tick(order.symbol)
            if tick is None:
                return {"error": f"Failed to get current price for {order.symbol}"}
            freshness_error = validation._validate_tick_freshness(
                tick,
                symbol=order.symbol,
                max_age_seconds=_CLOSE_TICK_MAX_AGE_SECONDS,
            )
            if freshness_error is not None:
                return _modify_freshness_block(
                    freshness_error,
                    dry_run=dry_run,
                    ticket=ticket_id,
                    symbol=order.symbol,
                )
            order_type_value = validation._safe_int_attr(order, "type", -1)
            stop_limit_types = {
                validation._safe_int_attr(mt5, "ORDER_TYPE_BUY_STOP_LIMIT", 6),
                validation._safe_int_attr(mt5, "ORDER_TYPE_SELL_STOP_LIMIT", 7),
            }
            is_stop_limit = order_type_value in stop_limit_types
            price_inputs, price_inputs_error = validation._normalize_trade_price_inputs(
                symbol_info=symbol_info,
                price=price if price is not None else getattr(order, "price_open", None),
                require_price=True,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            if price_inputs_error is not None:
                return {"error": price_inputs_error}
            price_increment = float(price_inputs["price_increment"])
            digits = int(price_inputs["digits"])
            normalized_price = float(price_inputs["price"])
            requested_sl = price_inputs["stop_loss"]
            requested_tp = price_inputs["take_profit"]
            explicit_remove_sl = bool(price_inputs["explicit_remove_stop_loss"])
            explicit_remove_tp = bool(price_inputs["explicit_remove_take_profit"])
            stop_limit_source = (
                stop_limit_price
                if stop_limit_price is not None
                else getattr(order, "price_stoplimit", None)
            )
            normalized_stop_limit = validation._normalize_price_for_symbol(
                stop_limit_source,
                point=price_increment,
                digits=digits,
            )
            if is_stop_limit and normalized_stop_limit is None:
                return {
                    "error": (
                        f"Pending stop-limit order {ticket_id} has no valid "
                        "price_stoplimit value to preserve."
                    ),
                    "error_code": "invalid_stop_limit_price",
                }
            if not is_stop_limit and stop_limit_price is not None:
                return {
                    "error": "stop_limit_price applies only to stop-limit pending orders.",
                    "error_code": "incompatible_parameters",
                }

            existing_sl = validation._normalize_price_for_symbol(
                getattr(order, "sl", None),
                point=price_increment,
                digits=digits,
            )
            existing_tp = validation._normalize_price_for_symbol(
                getattr(order, "tp", None),
                point=price_increment,
                digits=digits,
            )
            request_sl = (
                0.0
                if explicit_remove_sl
                else (
                    float(requested_sl)
                    if stop_loss is not None
                    else float(existing_sl or 0.0)
                )
            )
            request_tp = (
                0.0
                if explicit_remove_tp
                else (
                    float(requested_tp)
                    if take_profit is not None
                    else float(existing_tp or 0.0)
                )
            )

            pending_level_error = validation._validate_pending_order_levels(
                symbol_info=symbol_info,
                tick=tick,
                order_type_value=order_type_value,
                price=float(normalized_price),
                stop_loss=None if request_sl == 0.0 else float(request_sl),
                take_profit=None if request_tp == 0.0 else float(request_tp),
                mt5=mt5,
                stop_limit_price=(
                    None
                    if normalized_stop_limit is None
                    else float(normalized_stop_limit)
                ),
            )
            if pending_level_error is not None:
                return pending_level_error

            current_volume = validation._safe_float_attr(
                order, "volume_current", None
            )
            order_volume = (
                current_volume
                if current_volume is not None
                else validation._safe_float_attr(order, "volume_initial", None)
            )
            if order_volume is None:
                order_volume = validation._safe_float_attr(order, "volume", None)
            if (
                order_volume is None
                or not math.isfinite(order_volume)
                or order_volume <= 0.0
            ):
                return {
                    "error": (
                        f"Pending order {ticket_id} has no positive remaining volume; "
                        "refusing to modify it."
                    ),
                    "error_code": "invalid_pending_order_volume",
                }
            existing_entry_price = (
                validation._safe_float_attr(order, "price_stoplimit")
                if is_stop_limit
                else validation._safe_float_attr(order, "price_open")
            )
            candidate_stop_loss = None if request_sl == 0.0 else float(request_sl)
            current_stop_loss = existing_sl
            side = _resolve_pending_order_side(order)
            if side is None:
                return {
                    "error": f"Unsupported pending order type {order_type_value}.",
                    "error_code": "unsupported_pending_order_type",
                }
            if (
                trade_guardrails_config.is_enabled()
                and order_volume is not None
                and pending_order_risk_increased(
                    symbol_info=symbol_info,
                    side=side,
                    volume=float(order_volume),
                    existing_entry_price=existing_entry_price,
                    existing_stop_loss=current_stop_loss,
                    candidate_entry_price=float(
                        normalized_stop_limit
                        if normalized_stop_limit is not None
                        else normalized_price
                    ),
                    candidate_stop_loss=candidate_stop_loss,
                )
            ):
                try:
                    account_info = mt5.account_info()
                except Exception:
                    account_info = None
                positions, pending_orders, snapshot_error = load_guardrail_book_snapshots(
                    mt5,
                    trade_guardrails_config,
                    account_info=account_info,
                )
                if snapshot_error is not None:
                    snapshot_error["pending_order_ticket"] = resolved_ticket
                    snapshot_error["ticket_requested"] = ticket_id
                    snapshot_error["ticket_resolution"] = ticket_resolution
                    return snapshot_error
                guardrail_block = evaluate_trade_guardrails(
                    trade_guardrails_config,
                    symbol=order.symbol,
                    volume=float(order_volume),
                    stop_loss=candidate_stop_loss,
                    deviation=None,
                    side=side,
                    entry_price=float(
                        normalized_stop_limit
                        if normalized_stop_limit is not None
                        else normalized_price
                    ),
                    account_info=account_info,
                    existing_positions=positions,
                    existing_pending_orders=_trade_rows_excluding_current(
                        pending_orders,
                        resolved_ticket=resolved_ticket,
                        requested_ticket=ticket_id,
                    ),
                    symbol_info=symbol_info,
                    symbol_info_resolver=mt5.symbol_info,
                    enforce_symbol_rules=False,
                )
                if guardrail_block is not None:
                    guardrail_block["pending_order_ticket"] = resolved_ticket
                    guardrail_block["ticket_requested"] = ticket_id
                    guardrail_block["ticket_resolution"] = ticket_resolution
                    return guardrail_block

            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "order": ticket_id,
                "symbol": order.symbol,
                "volume": order_volume,
                "type": order_type_value,
                "price": float(normalized_price),
                "sl": request_sl,
                "tp": request_tp,
                "comment": comments._normalize_trade_comment(comment, default="mtdata modify pending order"),
            }
            if is_stop_limit and normalized_stop_limit is not None:
                request["stoplimit"] = float(normalized_stop_limit)
            request_magic = validation._safe_int_magic(getattr(order, "magic", None))
            if request_magic is not None:
                request["magic"] = request_magic

            if expiration_specified:
                if normalized_expiration is None:
                    request["type_time"] = mt5.ORDER_TIME_GTC
                else:
                    request["type_time"] = mt5.ORDER_TIME_SPECIFIED
                    request["expiration"] = normalized_expiration
            else:
                current_type_time = getattr(order, "type_time", None)
                current_expiration = getattr(order, "time_expiration", None)
                if current_type_time is not None:
                    request["type_time"] = current_type_time
                    if current_type_time == mt5.ORDER_TIME_SPECIFIED and current_expiration:
                        try:
                            request["expiration"] = int(current_expiration)
                        except Exception:
                            if isinstance(current_expiration, datetime):
                                server_dt = time._to_server_time_naive(current_expiration)
                                request["expiration"] = time._server_time_naive_to_mt5_timestamp(server_dt)

            if dry_run:
                preview = {
                    "success": True,
                    "dry_run": True,
                    "actionability": "preview_only",
                    "operation": "modify_pending_order",
                    "scope": "pending_orders",
                    "ticket": ticket_id,
                    "pending_order_ticket": resolved_ticket,
                    "ticket_requested": ticket_id,
                    "ticket_resolution": ticket_resolution,
                    "symbol": order.symbol,
                    "would_send_order": False,
                    "preview_scope_summary": (
                        "Validated pending-order routing and price levels; no modify request was sent to MT5."
                    ),
                    "applied_price": request.get("price"),
                    "applied_stop_limit_price": request.get("stoplimit"),
                    "applied_sl": request.get("sl"),
                    "applied_tp": request.get("tp"),
                    "applied_expiration": request.get("expiration"),
                    "expiration_explicit": expiration_specified,
                    "expiration_policy": (
                        (
                            "gtc"
                            if normalized_expiration is None
                            else "expires_at"
                        )
                        if expiration_specified
                        else "preserve_existing"
                    ),
                    "expiration_resolved_utc": (
                        time._format_expiration_utc(normalized_expiration)
                        if normalized_expiration is not None
                        else None
                    ),
                    "preview_ok": True,
                    "not_estimated": [
                        "broker_acceptance",
                        "execution_latency",
                    ],
                }
                removed = _protection_removed_fields(
                    explicit_remove_sl,
                    explicit_remove_tp,
                )
                if removed:
                    preview["protection_removed"] = removed
                return preview

            send_tick = mt5.symbol_info_tick(order.symbol)
            if send_tick is None:
                return {"error": f"Failed to refresh current price for {order.symbol}"}
            freshness_error = validation._validate_tick_freshness(
                send_tick,
                symbol=order.symbol,
                max_age_seconds=_CLOSE_TICK_MAX_AGE_SECONDS,
            )
            if freshness_error is not None:
                return freshness_error
            send_level_error = validation._validate_pending_order_levels(
                symbol_info=symbol_info,
                tick=send_tick,
                order_type_value=order_type_value,
                price=float(normalized_price),
                stop_loss=None if request_sl == 0.0 else float(request_sl),
                take_profit=None if request_tp == 0.0 else float(request_tp),
                mt5=mt5,
                stop_limit_price=(
                    None
                    if normalized_stop_limit is None
                    else float(normalized_stop_limit)
                ),
            )
            if send_level_error is not None:
                return send_level_error

            result, comment_fallback, last_error = comments._send_order_with_comment_fallback(
                mt5,
                request,
            )
            if result is None:
                out = {
                    "error": (
                        "Pending-order modification outcome is unknown; the broker "
                        "may have applied the requested changes. Confirm the live "
                        "order before retrying."
                    ),
                    "error_code": "order_send_ambiguous",
                    "ambiguous": True,
                    "pending_order_ticket": resolved_ticket,
                    "ticket_requested": ticket_id,
                    "last_error": last_error,
                }
                if isinstance(comment_fallback, dict):
                    out["comment_fallback"] = comment_fallback
                return out

            result_retcode = getattr(result, "retcode", None)
            no_change_code = validation._safe_int_attr(mt5, "TRADE_RETCODE_NO_CHANGES", 10025)
            pending_readback = None
            pending_requested: Dict[str, Any] = {}
            pending_adjustments: Dict[str, Dict[str, Any]] = {}
            if result_retcode in {no_change_code, mt5.TRADE_RETCODE_DONE}:
                pending_readback = _pending_modify_readback(
                    mt5,
                    resolved_ticket=resolved_ticket,
                    price_increment=price_increment,
                    digits=digits,
                    price_tol=_protection_level_tolerance(point=price_increment),
                )
                pending_requested = {
                    "applied_price": float(normalized_price),
                    "applied_stop_limit_price": (
                        float(normalized_stop_limit)
                        if normalized_stop_limit is not None
                        else None
                    ),
                    "applied_sl": _normalize_protection_level(
                        request_sl,
                        tol=_protection_level_tolerance(point=price_increment),
                    ),
                    "applied_tp": _normalize_protection_level(
                        request_tp,
                        tol=_protection_level_tolerance(point=price_increment),
                    ),
                }
                if "type_time" in request:
                    pending_requested["applied_type_time"] = validation._safe_int_attr(
                        request,
                        "type_time",
                        request.get("type_time"),
                    )
                if "expiration" in request:
                    pending_requested["applied_expiration"] = validation._safe_int_ticket(
                        request.get("expiration")
                    )
                if pending_readback is not None:
                    pending_adjustments = _readback_adjustments(
                        pending_requested,
                        pending_readback,
                        price_tol=_protection_level_tolerance(point=price_increment),
                    )
            if result_retcode == no_change_code:
                if pending_readback is not None and pending_adjustments:
                    return {
                        "error": (
                            "Broker reported no changes, but the live pending order "
                            "does not match the requested values."
                        ),
                        "retcode": result_retcode,
                        "retcode_name": mt5.retcode_name(result_retcode),
                        "pending_order_ticket": resolved_ticket,
                        "ticket_requested": ticket_id,
                        "ticket_resolution": ticket_resolution,
                        "requested": pending_requested,
                        "actual": pending_readback,
                        "adjustment": pending_adjustments,
                        "no_change": True,
                    }
                out = {
                    "success": True,
                    "no_change": True,
                    "retcode": result_retcode,
                    "retcode_name": mt5.retcode_name(result_retcode),
                    "comment": getattr(result, "comment", None),
                    "mt5_request_id": getattr(result, "request_id", None),
                    "pending_order_ticket": resolved_ticket,
                    "ticket_requested": ticket_id,
                    "ticket_resolution": ticket_resolution,
                }
                if pending_readback is None:
                    out.update(
                        {
                            "applied_price": None,
                            "applied_stop_limit_price": None,
                            "applied_sl": None,
                            "applied_tp": None,
                            "applied_expiration": None,
                            "verification_status": "unverified",
                            "warnings": [
                                "Broker reported no changes, but the pending order "
                                "could not be read back."
                            ],
                        }
                    )
                else:
                    out.update(pending_readback)
                    out["verification_status"] = "verified"
                if isinstance(comment_fallback, dict):
                    out["comment_fallback"] = comment_fallback
                return out

            if result_retcode != mt5.TRADE_RETCODE_DONE:
                out = {
                    "error": "Failed to modify pending order",
                    "retcode": result_retcode,
                    "retcode_name": mt5.retcode_name(result_retcode),
                    "comment": result.comment,
                    "mt5_request_id": result.request_id,
                    "last_error": last_error,
                }
                if isinstance(comment_fallback, dict):
                    out["comment_fallback"] = comment_fallback
                return out

            out = {
                "success": True,
                "retcode": result.retcode,
                "retcode_name": mt5.retcode_name(result.retcode),
                "deal": result.deal,
                "order": result.order,
                "comment": result.comment,
                "mt5_request_id": result.request_id,
                "pending_order_ticket": resolved_ticket,
                "ticket_requested": ticket_id,
                "ticket_resolution": ticket_resolution,
            }
            if pending_readback is None:
                out.update(
                    {
                        "applied_price": None,
                        "applied_stop_limit_price": None,
                        "applied_sl": None,
                        "applied_tp": None,
                        "applied_expiration": None,
                        "verification_status": "unverified",
                        "warnings": [
                            "Broker accepted the pending-order modification, but the "
                            "live values could not be read back."
                        ],
                    }
                )
            else:
                out.update(pending_readback)
                out["verification_status"] = "verified"
                if pending_adjustments:
                    out["broker_adjusted"] = True
                    out["adjustment"] = pending_adjustments
                    out["warnings"] = [
                        "Broker-reported pending-order values differ from the requested modification."
                    ]
            if isinstance(comment_fallback, dict):
                out["comment_fallback"] = comment_fallback
                if comment_fallback.get("used"):
                    out["warnings"] = [
                        "Broker rejected the comment field; pending order was retried with a minimal MT5-safe comment."
                    ]
            return out

        except time.PendingExpirationValidationError as e:
            return {
                "error": str(e),
                "error_code": e.error_code,
                "expiration_context": dict(e.context),
            }
        except Exception as e:
            return _unexpected_operation_error(
                "modifying pending order",
                e,
                mt5=mt5,
                context={"ticket": ticket},
            )

    return _modify_pending_order()


_CLOSE_TICK_MAX_AGE_SECONDS = float(QUOTE_LIVE_SECONDS)


def _execute_single_close(  # noqa: C901
    mt5: Any,
    position: Any,
    *,
    requested_volume: Optional[float],
    position_volume_before: Optional[float],
    remaining_volume_estimate: Optional[float],
    deviation: int,
    comment: Optional[str],
    fill_modes: List[Any],
) -> Dict[str, Any]:
    """Execute a close for a single position with fill-mode retry.

    Returns a result dict suitable for inclusion in the bulk-close results list.
    Handles tick refresh on requote/price-change and comment fallback.
    """
    position_side = _resolve_position_side(position, mt5)
    if position_side is None:
        return {
            "ticket": getattr(position, "ticket", None),
            "error": "Unable to determine position side for close request.",
        }
    is_buy_position = position_side == "BUY"

    if requested_volume is not None:
        try:
            close_volume = float(requested_volume)
        except (TypeError, ValueError):
            close_volume = float("nan")
    else:
        close_volume = validation._safe_float_attr(
            position,
            "volume",
            default=float("nan"),
        )
    if not math.isfinite(close_volume) or close_volume <= 0:
        return {
            "ticket": position.ticket,
            "error": f"Position has invalid volume ({getattr(position, 'volume', None)}) for close operation.",
        }

    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None:
        tick_error = validation._safe_last_error(mt5)
        return {
            "ticket": position.ticket,
            "error": f"Failed to get tick data for {position.symbol}",
            "last_error": tick_error,
            "attempts": [
                {
                    "error": f"Failed to get tick data for {position.symbol}",
                    "last_error": tick_error,
                }
            ],
        }
    tick_freshness_error = validation._validate_tick_freshness(
        tick,
        symbol=position.symbol,
        max_age_seconds=_CLOSE_TICK_MAX_AGE_SECONDS,
    )
    if tick_freshness_error is not None:
        return {
            "ticket": position.ticket,
            "error_code": "tick_stale",
            **tick_freshness_error,
            "attempts": [{"error_code": "tick_stale", **tick_freshness_error}],
        }

    close_type_buy = getattr(mt5, "ORDER_TYPE_BUY", 0)
    close_type_sell = getattr(mt5, "ORDER_TYPE_SELL", 1)
    result = None
    request = None
    last_send_error = None
    close_comment_fallback = None
    applied_close_comment = None
    attempts: List[Dict[str, Any]] = []

    price_changed_codes = {
        validation._safe_int_attr(mt5, "TRADE_RETCODE_PRICE_CHANGED", 10020),
        validation._safe_int_attr(mt5, "TRADE_RETCODE_REQUOTE", 10004),
    }
    invalid_fill_codes = {
        validation._safe_int_attr(mt5, "TRADE_RETCODE_INVALID_FILL", 10030),
    }
    stop_retries = False
    for fill_mode in fill_modes:
        price_retry_count = 0
        while True:
            close_price = (
                validation._safe_float_attr(tick, "bid")
                if is_buy_position
                else validation._safe_float_attr(tick, "ask")
            )
            close_type = close_type_sell if is_buy_position else close_type_buy
            close_comment = comments._normalize_close_trade_comment(comment, default="mtdata close")
            applied_close_comment = close_comment

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": position.ticket,
                "symbol": position.symbol,
                "volume": close_volume,
                "type": close_type,
                "price": close_price,
                "deviation": deviation,
                "comment": close_comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": int(fill_mode),
            }
            request_magic = validation._safe_int_magic(getattr(position, "magic", None))
            if request_magic is not None:
                request["magic"] = request_magic

            result, attempt_comment_fallback, last_send_error = comments._send_order_with_comment_fallback(
                mt5,
                request,
            )
            if isinstance(attempt_comment_fallback, dict):
                close_comment_fallback = attempt_comment_fallback
            if result is None:
                attempts.append(
                    {
                        "type_filling": int(fill_mode),
                        "error": "Failed to send close order",
                        "last_error": last_send_error,
                    }
                )
                if isinstance(attempt_comment_fallback, dict):
                    attempts[-1]["comment_fallback"] = attempt_comment_fallback
                # A missing response does not prove that an irreversible close
                # failed. Do not send the close again with another fill mode.
                stop_retries = True
                break

            retcode_val = getattr(result, "retcode", None)
            attempt: Dict[str, Any] = {
                "type_filling": int(fill_mode),
                "retcode": retcode_val,
                "retcode_name": mt5.retcode_name(retcode_val),
                "comment": getattr(result, "comment", None),
            }
            if price_retry_count > 0:
                attempt["price_retry"] = int(price_retry_count)
            attempts.append(attempt)
            if isinstance(attempt_comment_fallback, dict):
                attempts[-1]["comment_fallback"] = attempt_comment_fallback
            if validation._retcode_is_accepted(mt5, retcode_val):
                break
            if retcode_val in invalid_fill_codes:
                break
            if retcode_val not in price_changed_codes or price_retry_count >= 3:
                stop_retries = True
                break
            refreshed_tick = mt5.symbol_info_tick(position.symbol)
            if refreshed_tick is None:
                stop_retries = True
                break
            freshness_error = validation._validate_tick_freshness(
                refreshed_tick,
                symbol=position.symbol,
                max_age_seconds=_CLOSE_TICK_MAX_AGE_SECONDS,
            )
            if freshness_error is not None:
                attempts.append({"error_code": "tick_stale", **freshness_error})
                stop_retries = True
                break
            tick = refreshed_tick
            price_retry_count += 1
            _stdlib_time.sleep(0.15)
        if validation._retcode_is_accepted(mt5, getattr(result, "retcode", None) if result is not None else None):
            break
        if stop_retries:
            break

    close_accepted = (
        validation._retcode_is_accepted(mt5, getattr(result, "retcode", None))
        if result is not None
        else False
    )

    if not close_accepted:
        tick_failures = [
            a for a in attempts if "tick data" in str(a.get("error", "")).lower()
        ]
        error_msg = (
            f"Failed to get tick data for {position.symbol}"
            if attempts and len(tick_failures) == len(attempts)
            else "Failed to send close order"
        )
        fail_result: Dict[str, Any] = {
            "ticket": position.ticket,
            "error": error_msg,
            "attempts": attempts,
            "last_error": last_send_error,
        }
        if result is None and attempts:
            fail_result.update(
                {
                    "error": (
                        "Close outcome is unknown; the broker may have already "
                        "closed the requested position. Confirm live positions "
                        "before retrying."
                    ),
                    "error_code": "order_send_ambiguous",
                    "ambiguous": True,
                }
            )
        if isinstance(close_comment_fallback, dict):
            fail_result["comment_fallback"] = close_comment_fallback
        return fail_result

    # Build success result with PnL metadata
    open_price = getattr(position, "price_open", None)
    try:
        open_price = float(open_price) if open_price is not None else None
    except Exception:
        open_price = None
    close_exec_price = getattr(result, "price", None)
    try:
        close_exec_price = float(close_exec_price) if close_exec_price is not None else None
    except Exception:
        close_exec_price = None
    close_observed_at_utc = datetime.now(timezone.utc)
    open_epoch = getattr(position, "time", None)
    try:
        open_epoch_utc = float(open_epoch) if open_epoch is not None else None
    except Exception:
        open_epoch_utc = None
    duration_seconds = None
    if open_epoch_utc is not None:
        try:
            duration_seconds = int(
                max(0.0, close_observed_at_utc.timestamp() - float(open_epoch_utc))
            )
        except Exception:
            duration_seconds = None

    realized_pnl = getattr(result, "profit", None)
    try:
        realized_pnl = float(realized_pnl) if realized_pnl is not None else None
    except Exception:
        realized_pnl = None
    if realized_pnl is None:
        history_deal = _resolve_closed_deal_from_history(
            mt5,
            result=result,
            position=position,
            closed_at_utc=close_observed_at_utc,
        )
        if history_deal is not None:
            try:
                realized_pnl = float(getattr(history_deal, "profit", None))
            except Exception:
                realized_pnl = None

    pnl_price_delta = None
    if open_price is not None and close_exec_price is not None:
        try:
            if is_buy_position:
                pnl_price_delta = close_exec_price - open_price
            else:
                pnl_price_delta = open_price - close_exec_price
        except Exception:
            pnl_price_delta = None

    filled_volume = validation._safe_float_attr(result, "volume", default=None)
    post_fill_remaining_estimate = remaining_volume_estimate
    if (
        position_volume_before is not None
        and filled_volume is not None
        and filled_volume >= 0.0
    ):
        post_fill_remaining_estimate = max(
            0.0,
            float(position_volume_before) - float(filled_volume),
        )
    effective_requested_volume = (
        requested_volume if requested_volume is not None else position_volume_before
    )
    partial_fill_retcode = validation._safe_int_attr(
        mt5,
        "TRADE_RETCODE_DONE_PARTIAL",
        10010,
    )
    is_partial_fill = validation._safe_int_attr(result, "retcode", -1) == partial_fill_retcode
    if (
        filled_volume is not None
        and effective_requested_volume is not None
        and filled_volume + 1e-12 < float(effective_requested_volume)
    ):
        is_partial_fill = True
    execution_status = validation._trade_execution_status(mt5, result.retcode)
    if is_partial_fill:
        execution_status = "partial"

    res_dict: Dict[str, Any] = {
        "success": execution_status == "complete",
        "execution_status": execution_status,
        "ticket": position.ticket,
        "retcode": result.retcode,
        "retcode_name": mt5.retcode_name(result.retcode),
        "deal": result.deal,
        "order": result.order,
        "volume": result.volume,
        "filled_volume": filled_volume,
        "price": result.price,
        "comment": result.comment,
        "broker_message": result.comment,
        "open_price": open_price,
        "close_price": close_exec_price,
        "pnl": realized_pnl,
        "pnl_price_delta": pnl_price_delta,
        "duration_seconds": duration_seconds,
        "requested_volume": effective_requested_volume,
        "position_volume_before": position_volume_before,
        "position_volume_remaining_estimate": post_fill_remaining_estimate,
        "remaining_volume": (
            max(
                0.0,
                float(effective_requested_volume) - float(filled_volume),
            )
            if effective_requested_volume is not None and filled_volume is not None
            else None
        ),
        "attempts": attempts,
    }
    if applied_close_comment is not None:
        res_dict["comment"] = applied_close_comment
        res_dict["applied_comment"] = applied_close_comment
    if comment is not None:
        try:
            res_dict["requested_comment"] = str(comment).strip()
        except Exception:
            res_dict["requested_comment"] = ""
    if is_partial_fill:
        res_dict["partial_fill"] = True
    if isinstance(close_comment_fallback, dict):
        res_dict["comment_fallback"] = close_comment_fallback
        if close_comment_fallback.get("used"):
            res_dict["warnings"] = [
                "Broker rejected the comment field; close order was retried with a minimal MT5-safe comment."
            ]
    return res_dict


def _sort_close_positions(
    positions: List[Any],
    priority: Optional[str],
) -> List[Any]:
    """Sort positions for close ordering. Returns a new list.

    Supported priorities:
    - None: discovery order (no change)
    - "loss_first": most negative PnL first (close losers first)
    - "profit_first": most positive PnL first (take profits first)
    - "largest_first": largest notional volume first
    """
    if not priority or not positions:
        return list(positions)

    def _safe_profit(pos: Any) -> float:
        return validation._safe_float_attr(pos, "profit")

    def _safe_volume(pos: Any) -> float:
        return validation._safe_float_attr(pos, "volume")

    if priority == "loss_first":
        return sorted(positions, key=_safe_profit)
    elif priority == "profit_first":
        return sorted(positions, key=_safe_profit, reverse=True)
    elif priority == "largest_first":
        return sorted(positions, key=_safe_volume, reverse=True)
    return list(positions)


def _close_position_preview_row(position: Any, mt5: Any) -> Dict[str, Any]:
    return {
        key: value
        for key, value in {
            "ticket": getattr(position, "ticket", None),
            "symbol": getattr(position, "symbol", None),
            "side": _resolve_position_side(position, mt5),
            "volume": getattr(position, "volume", None),
            "profit": getattr(position, "profit", None),
            "price_open": getattr(position, "price_open", None),
            "price_current": getattr(position, "price_current", None),
            "sl": getattr(position, "sl", None),
            "tp": getattr(position, "tp", None),
            "magic": getattr(position, "magic", None),
            "comment": getattr(position, "comment", None),
        }.items()
        if value not in (None, "")
    }


def _close_positions_dry_run_preview(
    positions: List[Any],
    *,
    symbol: Optional[str],
    magic: Optional[int],
    profit_only: bool,
    loss_only: bool,
    close_priority: Optional[str],
    comment: Optional[str] = None,
    mt5: Any,
) -> Dict[str, Any]:
    rows = [_close_position_preview_row(position, mt5) for position in positions]
    quote_contexts: List[Dict[str, Any]] = []
    for row in rows:
        symbol_value = str(row.get("symbol") or "").strip()
        if not symbol_value:
            continue
        try:
            raw_tick = mt5.symbol_info_tick(symbol_value)
        except Exception:
            raw_tick = None
        query_epoch = _stdlib_time.time()
        tick, quote_source = resolve_quote_tick(
            mt5,
            symbol_value,
            raw_tick,
            now_epoch=query_epoch,
        )
        context = build_trade_quote_context(
            symbol_value,
            tick,
            now_epoch=_stdlib_time.time(),
            source_metadata=quote_source,
        )
        row["quote_context"] = context
        quote_contexts.append(context)
    live_ready = bool(quote_contexts) and all(
        context.get("usable_for_live_trading") is True
        for context in quote_contexts
    )
    total_volume = 0.0
    total_profit = 0.0
    for position in positions:
        try:
            total_volume += float(getattr(position, "volume", 0.0) or 0.0)
        except Exception:
            pass
        try:
            total_profit += float(getattr(position, "profit", 0.0) or 0.0)
        except Exception:
            pass
    filters: Dict[str, Any] = {}
    if symbol not in (None, ""):
        filters["symbol"] = symbol
    if magic is not None:
        filters["magic"] = magic
    if profit_only:
        filters["profit_only"] = True
    if loss_only:
        filters["loss_only"] = True
    if close_priority not in (None, ""):
        filters["close_priority"] = close_priority
    preview = {
        "success": live_ready,
        "dry_run": True,
        "actionability": "preview_only",
        "preview_ok": live_ready,
        "market_readiness": {
            "symbols_checked": len(quote_contexts),
            "usable_for_live_trading": live_ready,
            "stale_or_unverified": sum(
                context.get("usable_for_live_trading") is not True
                for context in quote_contexts
            ),
        },
        "matched_count": len(rows),
        "matched_positions": rows,
        "total_volume": round(total_volume, 8),
        "total_profit": round(total_profit, 2),
        "filters_applied": filters,
        "would_send_orders": len(rows),
    }
    if not live_ready:
        preview.update(
            {
                "error_code": "preview_blocked",
                "error": "One or more position quotes are not live-ready.",
                "blockers": ["market_not_live_ready"],
            }
        )
    return comments._attach_comment_preview_metadata(
        preview,
        comment,
        default="mtdata close",
        close=True,
    )


def _pending_order_preview_row(order: Any) -> Dict[str, Any]:
    return {
        key: value
        for key, value in {
            "ticket": getattr(order, "ticket", None),
            "symbol": getattr(order, "symbol", None),
            "type": getattr(order, "type", None),
            "volume_current": getattr(order, "volume_current", None),
            "price_open": getattr(order, "price_open", None),
            "sl": getattr(order, "sl", None),
            "tp": getattr(order, "tp", None),
            "magic": getattr(order, "magic", None),
            "comment": getattr(order, "comment", None),
        }.items()
        if value not in (None, "")
    }


def _cancel_pending_dry_run_preview(
    orders: List[Any],
    *,
    symbol: Optional[str],
    magic: Optional[int],
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    rows = [_pending_order_preview_row(order) for order in orders]
    filters = {
        key: value
        for key, value in {
            "symbol": symbol,
            "magic": magic,
        }.items()
        if value not in (None, "")
    }
    preview = {
        "success": True,
        "dry_run": True,
        "actionability": "preview_only",
        "matched_pending_count": len(rows),
        "matched_pending_orders": rows,
        "filters_applied": filters,
        "would_cancel_pending_orders": len(rows),
    }
    return comments._attach_comment_preview_metadata(
        preview,
        comment,
        default="mtdata cancel pending order",
    )


def _close_positions(  # noqa: C901
    ticket: Optional[Union[int, str]] = None,
    symbol: Optional[str] = None,
    magic: Optional[int] = None,
    volume: Optional[Union[int, float]] = None,
    profit_only: bool = False,
    loss_only: bool = False,
    close_priority: Optional[str] = None,
    comment: Optional[str] = None,
    deviation: int = 20,
    dry_run: bool = False,
    gateway: Optional[MT5TradingGateway] = None,
) -> dict:
    """Internal helper to close open positions."""
    mt5 = create_trading_gateway(
        gateway=gateway,
    )

    connection_error = trading_connection_error(mt5)
    if connection_error is not None:
        return connection_error

    def _close_positions():  # noqa: C901
        results: list[dict[str, Any]] = []
        position = None
        try:
            if profit_only and loss_only:
                return {"error": "profit_only and loss_only cannot both be true."}
            magic_filter = (
                validation._safe_int_magic(magic) if magic is not None else None
            )
            if magic is not None and magic_filter is None:
                return {
                    "error": (
                        "magic must be between 0 and "
                        f"{validation.MT5_UINT64_MAX}, inclusive."
                    )
                }
            symbol_error = _trading_symbol_or_error(mt5, symbol)
            if symbol_error is not None:
                return symbol_error

            # 1. Fetch positions based on criteria
            requested_ticket = None
            resolved_ticket = None
            ticket_resolution = None
            if ticket is not None:
                t_int = int(ticket)
                requested_ticket = t_int
                position, resolved_ticket, ticket_resolution = _resolve_open_position(
                    mt5,
                    ticket_candidates=[t_int],
                    require_exact_ticket_match=True,
                )
                if position is None:
                    if (
                        isinstance(ticket_resolution, dict)
                        and ticket_resolution.get("snapshot_unavailable")
                    ):
                        return validation.snapshot_unavailable_error(
                            mt5,
                            snapshot="positions",
                            context=f"close position {ticket}",
                        )
                    out = {"error": f"Position {ticket} not found", "checked_scopes": ["positions"]}
                    if isinstance(ticket_resolution, dict):
                        out["ticket_resolution"] = ticket_resolution
                    return out
                if (
                    symbol is not None
                    and str(getattr(position, "symbol", "") or "").upper()
                    != str(symbol).upper()
                ):
                    return {
                        "error": (
                            f"Position {ticket} belongs to {position.symbol}, "
                            f"not requested symbol {symbol}."
                        ),
                        "checked_scopes": ["positions"],
                    }
                if (
                    magic_filter is not None
                    and validation._safe_int_magic(getattr(position, "magic", None))
                    != magic_filter
                ):
                    return {
                        "error": f"Position {ticket} does not match magic={magic_filter}",
                        "checked_scopes": ["positions"],
                    }
                positions = [position]
            elif symbol is not None:
                positions = mt5.positions_get(symbol=symbol)
                if positions is None:
                    return validation.snapshot_unavailable_error(
                        mt5,
                        snapshot="positions",
                        context=f"close positions for {symbol}",
                    )
                if len(positions) == 0:
                    return {"message": f"No open positions for {symbol}"}
            else:
                positions = mt5.positions_get()
                if positions is None:
                    return validation.snapshot_unavailable_error(
                        mt5,
                        snapshot="positions",
                        context="close positions",
                    )
                if len(positions) == 0:
                    return {"message": "No open positions"}

            # 2. Filter positions
            to_close = []
            for pos in positions:
                if (
                    magic_filter is not None
                    and validation._safe_int_magic(getattr(pos, "magic", None))
                    != magic_filter
                ):
                    continue
                position_profit = validation._safe_float_attr(pos, "profit")
                if profit_only and position_profit <= 0.0:
                    continue
                if loss_only and position_profit >= 0.0:
                    continue
                to_close.append(pos)

            if not to_close:
                return {"message": "No positions matched criteria"}

            to_close = _sort_close_positions(to_close, close_priority)

            deviation_validated, deviation_error = validation._validate_deviation(deviation)
            if deviation_error:
                return {"error": deviation_error}

            if dry_run:
                return _close_positions_dry_run_preview(
                    to_close,
                    symbol=symbol,
                    magic=magic_filter,
                    profit_only=profit_only,
                    loss_only=loss_only,
                    close_priority=close_priority,
                    comment=comment,
                    mt5=mt5,
                )

            # 3. Close positions
            results.clear()
            for position in to_close:
                # Re-read position to check it still exists
                try:
                    fresh_positions = mt5.positions_get(ticket=position.ticket)
                except Exception:
                    fresh_positions = None
                if fresh_positions is None:
                    error_result = {
                        "ticket": position.ticket,
                        "error": "Failed to refresh open position before close.",
                    }
                    last_error = validation._safe_last_error(mt5)
                    if last_error is not None:
                        error_result["last_error"] = last_error
                    results.append(error_result)
                    continue
                if len(fresh_positions) == 0:
                    results.append(
                        {
                            "ticket": position.ticket,
                            "error": "Position no longer open.",
                        }
                    )
                    # Disappearing position is not a transport failure
                    continue
                expected_ticket = validation._safe_int_ticket(
                    getattr(position, "ticket", None)
                )
                refreshed_position = next(
                    (
                        candidate
                        for candidate in fresh_positions
                        if validation._safe_int_ticket(
                            getattr(candidate, "ticket", None)
                        )
                        == expected_ticket
                    ),
                    None,
                )
                if refreshed_position is None:
                    results.append(
                        {
                            "ticket": getattr(position, "ticket", None),
                            "error": (
                                "Exact position ticket was not returned during "
                                "pre-close refresh; close was not sent."
                            ),
                        }
                    )
                    continue
                position = refreshed_position

                symbol_info = None
                try:
                    symbol_info = mt5.symbol_info(position.symbol)
                except Exception:
                    symbol_info = None

                # Volume validation for partial closes
                requested_volume = None
                position_volume_before = None
                remaining_volume_estimate = None
                position_volume_before = validation._safe_float_attr(position, "volume") or None
                if volume is not None:
                    if symbol_info is None:
                        results.append({
                            "ticket": position.ticket,
                            "error": f"Failed to get symbol info for {position.symbol}",
                        })
                        continue
                    requested_volume, requested_volume_error = validation._validate_volume(
                        volume,
                        symbol_info,
                    )
                    if requested_volume_error:
                        results.append({
                            "ticket": position.ticket,
                            "error": requested_volume_error,
                            "requested_volume": volume,
                        })
                        continue
                    if (
                        position_volume_before is None
                        or not math.isfinite(position_volume_before)
                        or position_volume_before <= 0
                    ):
                        results.append({
                            "ticket": position.ticket,
                            "error": "Open position volume is invalid for partial close.",
                            "requested_volume": requested_volume,
                        })
                        continue
                    if requested_volume > (position_volume_before + 1e-12):
                        results.append({
                            "ticket": position.ticket,
                            "error": (
                                f"volume must be <= open position volume ({position_volume_before:g})"
                            ),
                            "requested_volume": requested_volume,
                            "position_volume": position_volume_before,
                        })
                        continue
                    remaining_volume_estimate = max(0.0, position_volume_before - requested_volume)
                    if remaining_volume_estimate > 1e-12:
                        _, remaining_error = validation._validate_volume(
                            remaining_volume_estimate,
                            symbol_info,
                        )
                        if remaining_error:
                            results.append({
                                "ticket": position.ticket,
                                "error": (
                                    "remaining position volume would be invalid after partial close: "
                                    f"{remaining_error}"
                                ),
                                "requested_volume": requested_volume,
                                "position_volume": position_volume_before,
                                "remaining_volume": remaining_volume_estimate,
                            })
                            continue

                fill_modes = validation._candidate_fill_modes(mt5, symbol_info)
                close_result = _execute_single_close(
                    mt5,
                    position,
                    requested_volume=requested_volume,
                    position_volume_before=position_volume_before,
                    remaining_volume_estimate=remaining_volume_estimate,
                    deviation=deviation_validated,
                    comment=comment,
                    fill_modes=fill_modes,
                )
                if requested_ticket is not None and isinstance(close_result, dict):
                    close_result.setdefault("ticket_requested", requested_ticket)
                    if resolved_ticket is not None:
                        close_result.setdefault("ticket_resolved", resolved_ticket)
                    if isinstance(ticket_resolution, dict):
                        close_result.setdefault("ticket_resolution", ticket_resolution)
                results.append(close_result)

            # If only one position was targeted by ticket, return single result
            if ticket is not None and len(results) == 1:
                return results[0]
            success_count = _count_done_results(mt5, results)
            bulk_result: Dict[str, Any] = {
                "success": bool(results) and success_count == len(results),
                "closed_count": success_count,
                "attempted_count": len(results),
                "results": results,
            }
            if success_count < len(results):
                bulk_result["partial_failure"] = success_count > 0
                if success_count == 0:
                    bulk_result["error"] = "Failed to close any targeted positions."
            if close_priority:
                bulk_result["close_priority"] = close_priority
            return bulk_result

        except Exception as e:
            error_result: Dict[str, Any] = {"error": str(e)}
            current_ticket = validation._safe_int_ticket(getattr(position, "ticket", None))
            if current_ticket is None and position is not None:
                current_ticket = getattr(position, "ticket", None)
            if current_ticket is not None:
                error_result["ticket"] = current_ticket

            last_error = validation._safe_last_error(mt5)
            if last_error is not None:
                error_result["last_error"] = last_error

            if not results:
                return error_result

            if current_ticket is None or not any(
                isinstance(result_item, dict) and result_item.get("ticket") == current_ticket
                for result_item in results
            ):
                results.append(error_result)

            if ticket is not None and len(results) == 1:
                return results[0]

            return {
                "closed_count": _count_done_results(mt5, results),
                "attempted_count": len(results),
                "partial_failure": True,
                "error": str(e),
                "results": results,
            }

    return _close_positions()


def _resolve_close_dry_run_target(
    *,
    ticket: Union[int, str],
    target: str = "positions",
    symbol: Optional[str] = None,
    volume: Optional[Union[int, float]] = None,
    magic: Optional[int] = None,
    profit_only: bool = False,
    loss_only: bool = False,
    close_priority: Optional[str] = None,
    gateway: Optional[MT5TradingGateway] = None,
) -> dict:
    """Resolve a close target for dry-run validation without sending trade requests."""
    mt5 = create_trading_gateway(
        gateway=gateway,
    )

    connection_error = trading_connection_error(mt5)
    if connection_error is not None:
        return connection_error

    requested_ticket = validation._safe_int_ticket(ticket)
    if requested_ticket is None:
        return {"error": f"Invalid ticket {ticket!r}"}
    target_value = str(target or "positions").strip().lower()
    if target_value not in {"positions", "pending"}:
        return {"error": f"Unsupported ticket target {target!r}"}
    magic_filter = validation._safe_int_magic(magic) if magic is not None else None
    if magic is not None and magic_filter is None:
        return {
            "error": (
                "magic must be between 0 and "
                f"{validation.MT5_UINT64_MAX}, inclusive."
            )
        }

    position = None
    resolved_ticket = None
    position_resolution = None
    if target_value == "positions":
        position, resolved_ticket, position_resolution = _resolve_open_position(
            mt5,
            ticket_candidates=[requested_ticket],
            symbol=symbol,
            require_exact_ticket_match=True,
        )
    if position is not None:
        if (
            magic_filter is not None
            and validation._safe_int_magic(getattr(position, "magic", None))
            != magic_filter
        ):
            return {"error": f"Position {ticket} does not match magic={magic_filter}"}
        position_profit = validation._safe_float_attr(position, "profit", 0.0)
        if profit_only and position_profit <= 0.0:
            return {"error": f"Position {ticket} does not match profit_only=true"}
        if loss_only and position_profit >= 0.0:
            return {"error": f"Position {ticket} does not match loss_only=true"}
        position_volume = validation._safe_float_attr(position, "volume") or None
        if volume is not None:
            symbol_info = mt5.symbol_info(getattr(position, "symbol", ""))
            if symbol_info is None:
                return {"error": f"Failed to get symbol info for {position.symbol}"}
            requested_volume, volume_error = validation._validate_volume(
                volume, symbol_info
            )
            if volume_error:
                return {"error": volume_error, "requested_volume": volume}
            if position_volume is None or requested_volume is None:
                return {"error": "Open position volume is invalid for partial close."}
            if requested_volume > position_volume + 1e-12:
                return {
                    "error": f"volume must be <= open position volume ({position_volume:g})",
                    "requested_volume": requested_volume,
                    "position_volume": position_volume,
                }
            remaining_volume = max(0.0, position_volume - requested_volume)
            if remaining_volume > 1e-12:
                _, remaining_error = validation._validate_volume(
                    remaining_volume, symbol_info
                )
                if remaining_error:
                    return {
                        "error": "Partial close would leave an invalid remaining volume: "
                        + remaining_error,
                        "requested_volume": requested_volume,
                        "position_volume": position_volume,
                    }
        preview = _close_positions_dry_run_preview(
            [position],
            symbol=symbol,
            magic=magic_filter,
            profit_only=profit_only,
            loss_only=loss_only,
            close_priority=close_priority,
            mt5=mt5,
        )
        preview.update({
            "success": True,
            "target_scope": "positions",
            "target_kind": "open_position",
            "resolved_ticket": resolved_ticket,
            "target_symbol": getattr(position, "symbol", None),
            "target_volume": position_volume,
            "ticket_resolution": position_resolution,
        })
        if volume is not None:
            preview["requested_close_volume"] = float(requested_volume)
            preview["would_send_orders"] = 1
        return preview

    if (
        isinstance(position_resolution, dict)
        and position_resolution.get("snapshot_unavailable")
    ):
        return validation.snapshot_unavailable_error(
            mt5,
            snapshot="positions",
            context=f"preview close target {ticket}",
        )

    if volume is not None:
        return {
            "error": (
                f"Position {ticket} not found. "
                "Partial close volume only applies to open positions."
            ),
            "checked_scopes": ["positions"],
        }

    if target_value == "positions":
        return {
            "error_code": "ticket_not_found",
            "error": f"Position {ticket} not found.",
            "ticket": requested_ticket,
            "checked_scopes": ["positions"],
            "suggestion": "Use trade_get_open to find an active position ticket.",
            "remediation": "Use trade_get_open to find an active position ticket.",
        }

    pending_order, resolved_ticket, pending_resolution = _resolve_pending_order(
        mt5,
        ticket_candidates=[requested_ticket],
        symbol=symbol,
        require_exact_ticket_match=True,
    )
    if pending_order is not None:
        if (
            magic_filter is not None
            and validation._safe_int_magic(getattr(pending_order, "magic", None))
            != magic_filter
        ):
            return {"error": f"Pending order {ticket} does not match magic={magic_filter}"}
        if profit_only or loss_only:
            return {
                "error": (
                    "profit_only and loss_only filters apply only to open positions; "
                    f"ticket {ticket} is a pending order."
                )
            }
        return {
            "success": True,
            "target_scope": "pending_orders",
            "target_kind": "pending_order",
            "resolved_ticket": resolved_ticket,
            "target_symbol": getattr(pending_order, "symbol", None),
            "target_volume": getattr(pending_order, "volume_current", None),
            "ticket_resolution": pending_resolution,
        }

    if (
        isinstance(pending_resolution, dict)
        and pending_resolution.get("snapshot_unavailable")
    ):
        return validation.snapshot_unavailable_error(
            mt5,
            snapshot="orders",
            context=f"preview pending cancellation {ticket}",
        )

    return {
        "error_code": "ticket_not_found",
        "error": f"Pending order {ticket} not found.",
        "ticket": requested_ticket,
        "checked_scopes": ["pending_orders"],
        "suggestion": (
            "Use trade_get_pending to find an active pending-order ticket "
            "before retrying trade_close with target=pending."
        ),
        "remediation": (
            "Use trade_get_pending to find an active pending-order ticket "
            "before retrying trade_close with target=pending."
        ),
    }


def _cancel_pending(  # noqa: C901
    ticket: Optional[Union[int, str]] = None,
    symbol: Optional[str] = None,
    magic: Optional[int] = None,
    comment: Optional[str] = None,
    dry_run: bool = False,
    gateway: Optional[MT5TradingGateway] = None,
) -> dict:
    """Internal helper to cancel pending orders."""
    mt5 = create_trading_gateway(
        gateway=gateway,
    )

    connection_error = trading_connection_error(mt5)
    if connection_error is not None:
        return connection_error

    def _cancel_pending():  # noqa: C901
        try:
            magic_filter = (
                validation._safe_int_magic(magic) if magic is not None else None
            )
            if magic is not None and magic_filter is None:
                return {
                    "error": (
                        "magic must be between 0 and "
                        f"{validation.MT5_UINT64_MAX}, inclusive."
                    )
                }
            symbol_error = _trading_symbol_or_error(mt5, symbol)
            if symbol_error is not None:
                return symbol_error
            # 1. Fetch orders based on criteria
            if ticket is not None:
                t_int = int(ticket)
                requested_ticket = t_int
                order, resolved_ticket, ticket_resolution = _resolve_pending_order(
                    mt5,
                    ticket_candidates=[t_int],
                    require_exact_ticket_match=True,
                )
                if order is None:
                    if (
                        isinstance(ticket_resolution, dict)
                        and ticket_resolution.get("snapshot_unavailable")
                    ):
                        return validation.snapshot_unavailable_error(
                            mt5,
                            snapshot="orders",
                            context=f"cancel pending order {ticket}",
                        )
                    out = {"error": f"Pending order {ticket} not found", "checked_scopes": ["pending_orders"]}
                    if isinstance(ticket_resolution, dict):
                        out["ticket_resolution"] = ticket_resolution
                    return out
                if (
                    symbol is not None
                    and str(getattr(order, "symbol", "") or "").upper()
                    != str(symbol).upper()
                ):
                    return {
                        "error": (
                            f"Pending order {ticket} belongs to {order.symbol}, "
                            f"not requested symbol {symbol}."
                        ),
                        "checked_scopes": ["pending_orders"],
                    }
                if (
                    magic_filter is not None
                    and validation._safe_int_magic(getattr(order, "magic", None))
                    != magic_filter
                ):
                    return {
                        "error": f"Pending order {ticket} does not match magic={magic_filter}",
                        "checked_scopes": ["pending_orders"],
                    }
                orders = [order]
            elif symbol is not None:
                orders = mt5.orders_get(symbol=symbol)
                if orders is None:
                    return validation.snapshot_unavailable_error(
                        mt5,
                        snapshot="orders",
                        context=f"cancel pending orders for {symbol}",
                    )
                if len(orders) == 0:
                    if dry_run:
                        return _cancel_pending_dry_run_preview(
                            [],
                            symbol=symbol,
                            magic=magic_filter,
                            comment=comment,
                        )
                    return {"message": f"No pending orders for {symbol}"}
            else:
                orders = mt5.orders_get()
                if orders is None:
                    return validation.snapshot_unavailable_error(
                        mt5,
                        snapshot="orders",
                        context="cancel pending orders",
                    )
                if len(orders) == 0:
                    if dry_run:
                        return _cancel_pending_dry_run_preview(
                            [],
                            symbol=symbol,
                            magic=magic_filter,
                            comment=comment,
                        )
                    return {"message": "No pending orders"}
            if magic_filter is not None:
                orders = [
                    order
                    for order in orders
                    if validation._safe_int_magic(getattr(order, "magic", None))
                    == magic_filter
                ]
                if not orders:
                    if dry_run:
                        return _cancel_pending_dry_run_preview(
                            [],
                            symbol=symbol,
                            magic=magic_filter,
                            comment=comment,
                        )
                    return {"message": "No pending orders matched criteria"}

            if dry_run:
                return _cancel_pending_dry_run_preview(
                    list(orders),
                    symbol=symbol,
                    magic=magic_filter,
                    comment=comment,
                )

            # 2. Cancel orders
            results = []
            for order in orders:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                    "comment": comments._normalize_trade_comment(comment, default="mtdata cancel pending order"),
                }
                request_magic = validation._safe_int_magic(getattr(order, "magic", None))
                if request_magic is not None:
                    request["magic"] = request_magic

                result, comment_fallback, last_error = comments._send_order_with_comment_fallback(
                    mt5,
                    request,
                )
                if result is None:
                    result_entry = {
                        "ticket": order.ticket,
                        "error": (
                            "Cancel outcome is unknown; the broker may have already "
                            "cancelled the pending order. Confirm live pending orders "
                            "before retrying."
                        ),
                        "error_code": "order_send_ambiguous",
                        "ambiguous": True,
                        "last_error": last_error,
                    }
                    if isinstance(comment_fallback, dict):
                        result_entry["comment_fallback"] = comment_fallback
                    results.append(result_entry)
                else:
                    execution_status = validation._trade_execution_status(
                        mt5, result.retcode
                    )
                    result_entry = {
                        "success": execution_status == "complete",
                        "execution_status": execution_status,
                        "ticket": order.ticket,
                        "retcode": result.retcode,
                        "retcode_name": mt5.retcode_name(result.retcode),
                        "deal": result.deal,
                        "order": result.order,
                        "comment": result.comment,
                    }
                    if execution_status != "complete":
                        result_entry["error"] = (
                            "Pending order cancellation was not completed by MT5."
                        )
                    if isinstance(comment_fallback, dict):
                        result_entry["comment_fallback"] = comment_fallback
                    if ticket is not None:
                        result_entry.setdefault("ticket_requested", requested_ticket)
                        if resolved_ticket is not None:
                            result_entry.setdefault("ticket_resolved", resolved_ticket)
                        if isinstance(ticket_resolution, dict):
                            result_entry.setdefault("ticket_resolution", ticket_resolution)
                    results.append(result_entry)

            # If only one order was targeted by ticket, return single result
            if ticket is not None and len(results) == 1:
                return results[0]
            success_count = _count_done_results(mt5, results)
            bulk_result: Dict[str, Any] = {
                "success": bool(results) and success_count == len(results),
                "cancelled_count": success_count,
                "attempted_count": len(results),
                "results": results,
            }
            if success_count < len(results):
                bulk_result["partial_failure"] = success_count > 0
                bulk_result["error"] = (
                    "One or more pending orders were not cancelled."
                    if success_count > 0
                    else "Failed to cancel any targeted pending orders."
                )
            return bulk_result

        except Exception as e:
            return {"error": str(e)}

    return _cancel_pending()
