"""Order placement workflows for MetaTrader integration."""

import logging
import math
import threading
import time as _stdlib_time
from datetime import datetime, timedelta, timezone
from numbers import Real
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

from ...bootstrap.settings import mt5_config, trade_guardrails_config
from ...shared.market_units import (
    UNIT_BROKER_POINTS,
    UNIT_PERCENT,
    UNIT_PIPS,
    forex_points_per_pip,
)
from ...utils.coercion import coerce_optional_bool, round_finite
from ...utils.freshness import QUOTE_LIVE_SECONDS
from ...utils.quote import compute_spread_metrics, resolve_quote_tick, tick_value
from ...utils.utils import _utc_epoch_seconds
from . import comments, common, time, validation
from .gateway import MT5TradingGateway, create_trading_gateway, trading_connection_error
from .positions import _resolve_open_position
from .safety import (
    _account_uses_netting,
    _estimate_order_risk_currency,
    _guardrail_snapshot,
    _resolve_existing_symbol_net,
    assess_margin_stress,
    assess_new_exposure_allowed,
    evaluate_trade_guardrails,
    load_guardrail_book_snapshots,
)
from .time import ExpirationValue
from .validation import MarketOrderTypeInput, OrderTypeInput


class _OrderSymbolContext(TypedDict):
    symbol_info: Any
    volume: float
    account_info: Any
    account_state_block: Optional[Dict[str, Any]]


class _OrderSubmitOutcome(TypedDict):
    result: Any
    fill_mode_attempts: List[Dict[str, Any]]
    used_request: Dict[str, Any]


_POSITION_RESOLUTION_WAIT_SCHEDULE_SECONDS = (0.15, 0.3, 0.6, 1.2, 2.4)
_TRADE_TICK_MAX_AGE_SECONDS = float(QUOTE_LIVE_SECONDS)
_POSITION_DEAL_LOOKUP_WINDOW_SECONDS = 30
_TRADE_DECISION_LOCK = threading.RLock()
logger = logging.getLogger(__name__)


def _expand_position_ticket_candidates_from_deals(
    mt5: Any,
    candidates: List[int],
) -> List[int]:
    """Augment ticket candidates with position IDs from recent deal history.

    After a fill, ``result.deal`` / ``result.order`` may not equal the open
    position ticket. Looking up the deal's ``position_id`` avoids attaching
    SL/TP to a pre-existing same-symbol position via heuristic matching.
    """
    expanded: List[int] = []
    for raw in candidates:
        ticket = validation._safe_int_ticket(raw)
        if ticket is not None and ticket not in expanded:
            expanded.append(ticket)
    if not expanded:
        return expanded

    now = datetime.now(timezone.utc)
    try:
        history_from = _utc_epoch_seconds(
            now - timedelta(seconds=_POSITION_DEAL_LOOKUP_WINDOW_SECONDS)
        )
        history_to = _utc_epoch_seconds(now + timedelta(seconds=5))
        rows = mt5.history_deals_get(history_from, history_to)
    except Exception:
        rows = None
    if not rows:
        return expanded

    candidate_set = set(expanded)
    for row in rows:
        row_ticket = validation._safe_int_ticket(getattr(row, "ticket", None))
        row_order = validation._safe_int_ticket(getattr(row, "order", None))
        if row_ticket not in candidate_set and row_order not in candidate_set:
            continue
        for field in ("position_id", "position", "position_by_id"):
            pos_id = validation._safe_int_ticket(getattr(row, field, None))
            if pos_id is not None and pos_id not in expanded:
                # Prefer position id at the front so exact ticket match wins.
                expanded.insert(0, pos_id)
    return expanded


def _configured_order_magic() -> int:
    configured = validation._safe_int_magic(getattr(mt5_config, "order_magic", None))
    if configured is None:
        raise ValueError(
            "Configured MTDATA_ORDER_MAGIC must be between 0 and "
            f"{validation.MT5_UINT64_MAX}, inclusive."
        )
    return int(configured)


def _compact_sl_tp_levels(
    *,
    sl: Optional[float],
    tp: Optional[float],
) -> Optional[Dict[str, float]]:
    levels: Dict[str, float] = {}
    if sl is not None:
        levels["sl"] = float(sl)
    if tp is not None:
        levels["tp"] = float(tp)
    return levels or None


def _build_sl_tp_result(
    *,
    requested_sl: Optional[float],
    requested_tp: Optional[float],
    applied_sl: Optional[float],
    applied_tp: Optional[float],
    status: str,
    error: Optional[str],
    broker_adjusted: bool,
    adjustment: Optional[Dict[str, Any]],
    attempts: int,
    last_retcode: Any,
    last_comment: Any,
    verification_failed: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": status}
    requested = _compact_sl_tp_levels(sl=requested_sl, tp=requested_tp)
    applied = _compact_sl_tp_levels(sl=applied_sl, tp=applied_tp)
    if requested is not None:
        out["requested"] = requested
    if applied is not None:
        out["applied"] = applied
    if error is not None:
        out["error"] = error
    if broker_adjusted:
        out["broker_adjusted"] = True
    if adjustment:
        out["adjustment"] = adjustment
    if attempts > 0:
        out["attempts"] = int(attempts)
    if last_retcode is not None:
        out["last_retcode"] = last_retcode
    if last_comment is not None:
        out["last_comment"] = last_comment
    if verification_failed:
        out["verification_failed"] = True
    return out


def _evaluate_requested_protection(
    *,
    requested_sl: Optional[float],
    requested_tp: Optional[float],
    applied_sl: Optional[float],
    applied_tp: Optional[float],
    symbol_info: Any,
) -> tuple[bool, bool, Optional[Dict[str, Dict[str, float]]]]:
    point = validation._safe_float_attr(symbol_info, "point", default=0.0) or 0.0
    price_tol = validation._protection_level_tolerance(point=point)
    all_requested_present = True
    adjustment: Dict[str, Dict[str, float]] = {}
    for label, requested, applied in (
        ("sl", requested_sl, applied_sl),
        ("tp", requested_tp, applied_tp),
    ):
        if requested is None:
            continue
        if applied is None:
            all_requested_present = False
            continue
        if not validation._protection_levels_match(
            float(applied),
            float(requested),
            tol=price_tol,
        ):
            adjustment[label] = {
                "requested": float(requested),
                "applied": float(applied),
            }
    return all_requested_present, bool(adjustment), adjustment or None


class _ProtectionOutcome(TypedDict, total=False):
    position_ticket: Optional[int]
    position_ticket_candidates: List[int]
    position_ticket_resolution: Optional[Dict[str, Any]]
    protection_status: str
    sl_tp_result: Dict[str, Any]
    warnings: List[str]


def _attach_post_fill_protection(  # noqa: C901
    mt5: Any,
    *,
    symbol: str,
    side: str,
    volume: float,
    position_ticket_candidates: List[int],
    stop_loss: Optional[float],
    take_profit: Optional[float],
    symbol_info: Any,
    comment: Optional[str],
    request_comment: str,
    magic: Optional[int] = None,
) -> _ProtectionOutcome:
    """Resolve the filled position and attach SL/TP protection.

    Returns a structured outcome with position resolution info, the SL/TP
    result dict, an optional protection_status key, and accumulated warnings.
    """
    warnings_out: List[str] = []
    sl_tp_requested = bool(stop_loss is not None or take_profit is not None)

    if not sl_tp_requested:
        return {
            "position_ticket": None,
            "position_ticket_candidates": position_ticket_candidates or None,
            "position_ticket_resolution": None,
            "sl_tp_result": _build_sl_tp_result(
                requested_sl=None,
                requested_tp=None,
                applied_sl=None,
                applied_tp=None,
                status="not_requested",
                error=None,
                broker_adjusted=False,
                adjustment=None,
                attempts=0,
                last_retcode=None,
                last_comment=None,
            ),
            "warnings": [],
        }

    # --- State variables for the protection flow ---
    position_ticket: Optional[int] = None
    position_ticket_resolution: Optional[Dict[str, Any]] = None
    sl_tp_error: Optional[str] = None
    sl_tp_apply_status = "not_requested"
    sl_applied: Optional[float] = None
    tp_applied: Optional[float] = None
    sl_tp_verification_failed = False
    sl_tp_broker_adjusted = False
    sl_tp_adjustment: Dict[str, Any] = {}
    sl_tp_attempts = 0
    sl_tp_last_retcode = None
    sl_tp_last_comment = None
    sl_tp_last_error = None

    try:
        # --- Phase 1: Resolve the open position ---
        position_obj = None
        lookup_wait_schedule = _POSITION_RESOLUTION_WAIT_SCHEDULE_SECONDS
        lookup_attempts = len(lookup_wait_schedule) + 1
        last_resolve_info: Optional[Dict[str, Any]] = None
        resolved_candidates = _expand_position_ticket_candidates_from_deals(
            mt5, list(position_ticket_candidates or [])
        )
        if resolved_candidates:
            position_ticket_candidates = resolved_candidates
        for attempt_idx in range(lookup_attempts):
            pos, resolved_ticket, resolve_info = _resolve_open_position(
                mt5,
                ticket_candidates=position_ticket_candidates,
                symbol=symbol,
                side=side,
                volume=volume,
                magic=magic,
                require_exact_ticket_match=True,
                allow_alternate_ticket_match=True,
            )
            if isinstance(resolve_info, dict):
                last_resolve_info = dict(resolve_info)
            if pos is not None and resolved_ticket is not None:
                # Guard against attaching protection to a pre-existing position
                # when heuristics matched without an exact ticket hit.
                open_time = validation._safe_float_attr(pos, "time", default=None)
                if open_time is not None and open_time > 0:
                    age_seconds = _stdlib_time.time() - float(open_time)
                    method = str((resolve_info or {}).get("method") or "")
                    if age_seconds > 30 and "heuristic" in method:
                        last_resolve_info = {
                            **dict(resolve_info or {}),
                            "matched": False,
                            "rejected_reason": "pre_existing_position",
                            "position_age_seconds": round(age_seconds, 2),
                        }
                        pos = None
                        resolved_ticket = None
            if pos is not None and resolved_ticket is not None:
                position_obj = pos
                position_ticket = resolved_ticket
                position_ticket_resolution = {
                    **dict(resolve_info),
                    "attempts": int(attempt_idx + 1),
                    "matched": True,
                }
                break
            if attempt_idx + 1 < lookup_attempts:
                # Re-expand from deals between retries in case history lagged.
                resolved_candidates = _expand_position_ticket_candidates_from_deals(
                    mt5, list(position_ticket_candidates or [])
                )
                if resolved_candidates:
                    position_ticket_candidates = resolved_candidates
                _stdlib_time.sleep(float(lookup_wait_schedule[attempt_idx]))
        if position_ticket_resolution is None and last_resolve_info is not None:
            position_ticket_resolution = {
                **last_resolve_info,
                "attempts": int(lookup_attempts),
                "matched": False,
            }

        if position_obj is not None and position_ticket is not None:
            sl_applied, tp_applied = validation._position_protection_levels(position_obj)
            initial_confirmed, initial_adjusted, initial_adjustment = (
                _evaluate_requested_protection(
                    requested_sl=stop_loss,
                    requested_tp=take_profit,
                    applied_sl=sl_applied,
                    applied_tp=tp_applied,
                    symbol_info=symbol_info,
                )
            )
            if initial_confirmed:
                sl_tp_apply_status = "applied"
                sl_tp_broker_adjusted = initial_adjusted
                if initial_adjustment:
                    sl_tp_adjustment = dict(initial_adjustment)
            else:
                # --- Phase 2: Attach SL/TP via TRADE_ACTION_SLTP ---
                modify_request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": position_ticket,
                    "sl": (
                        0.0
                        if stop_loss is None and sl_applied is None
                        else float(sl_applied if stop_loss is None else stop_loss)
                    ),
                    "tp": (
                        0.0
                        if take_profit is None and tp_applied is None
                        else float(
                            tp_applied if take_profit is None else take_profit
                        )
                    ),
                    "comment": comments._normalize_trade_comment(
                        comment,
                        default=request_comment,
                        suffix=" - set TP/SL",
                    ),
                }
                modify_magic = validation._safe_int_magic(getattr(position_obj, "magic", None))
                if modify_magic is not None:
                    modify_request["magic"] = modify_magic
                modify_result = None
                max_modify_attempts = 5
                invalid_stops_code = validation._safe_int_attr(mt5, "TRADE_RETCODE_INVALID_STOPS", 10016)
                for modify_try in range(max_modify_attempts):
                    sl_tp_attempts = int(modify_try + 1)
                    try:
                        modify_result = mt5.order_send(modify_request)
                        sl_tp_last_error = validation._safe_last_error(mt5)
                    except Exception as ex:
                        modify_result = None
                        sl_tp_last_error = str(ex)
                        sl_tp_error = f"Error setting TP/SL: {str(ex)}"
                    if modify_result is None:
                        if sl_tp_error is None:
                            sl_tp_error = (
                                "Ambiguous broker response while setting TP/SL; "
                                "the request was not retried."
                            )
                        sl_tp_apply_status = "unverified"
                        break
                    if modify_result is not None:
                        sl_tp_last_retcode = getattr(modify_result, "retcode", None)
                        sl_tp_last_comment = getattr(modify_result, "comment", None)
                        # Only the current attempt's retcode may terminate the loop;
                        # a None return from order_send must not let a prior attempt's
                        # retcode (e.g. INVALID_STOPS) bail out this iteration.
                        if sl_tp_last_retcode == mt5.TRADE_RETCODE_DONE:
                            break
                        if sl_tp_last_retcode == invalid_stops_code:
                            break
                    if modify_try + 1 < max_modify_attempts:
                        _stdlib_time.sleep(0.35)

                if modify_result and getattr(modify_result, "retcode", None) == mt5.TRADE_RETCODE_DONE:
                    # --- Phase 3: Verify SL/TP readback ---
                    positions_after = None
                    try:
                        positions_after = mt5.positions_get(ticket=position_ticket)
                        pos_after = validation._exact_ticket_row(
                            positions_after,
                            position_ticket,
                        )
                        if pos_after is not None:
                            sl_applied, tp_applied = validation._position_protection_levels(
                                pos_after
                            )
                            confirmed, adjusted, adjustment = _evaluate_requested_protection(
                                requested_sl=stop_loss,
                                requested_tp=take_profit,
                                applied_sl=sl_applied,
                                applied_tp=tp_applied,
                                symbol_info=symbol_info,
                            )
                            if confirmed:
                                sl_tp_apply_status = "applied"
                                sl_tp_broker_adjusted = adjusted
                                sl_tp_adjustment = dict(adjustment or {})
                            else:
                                sl_tp_error = (
                                    "Broker accepted the TP/SL update, but the live "
                                    "position did not report the requested protection."
                                )
                                sl_tp_apply_status = "failed"
                        else:
                            sl_tp_verification_failed = True
                    except Exception as verify_exc:
                        sl_tp_verification_failed = True
                        logger.warning(
                            "SL/TP verification failed for ticket %s: %s",
                            position_ticket,
                            verify_exc,
                        )

                    if sl_tp_verification_failed:
                        sl_tp_error = (
                            "Broker accepted the TP/SL update, but live protection "
                            "could not be verified."
                        )
                        sl_tp_apply_status = "unverified"
                else:
                    if sl_tp_error is None:
                        detail_bits = [f"after {sl_tp_attempts} attempt(s)"]
                        if sl_tp_last_retcode is not None:
                            detail_bits.append(f"retcode={sl_tp_last_retcode}")
                        if sl_tp_last_comment:
                            detail_bits.append(f"comment={sl_tp_last_comment!r}")
                        if sl_tp_last_error:
                            detail_bits.append(f"broker_error={sl_tp_last_error!r}")
                        sl_tp_error = "Failed to set TP/SL (" + ", ".join(detail_bits) + ")"
                    if sl_tp_apply_status != "unverified":
                        sl_tp_apply_status = "failed"
        else:
            checked = ", ".join(str(v) for v in position_ticket_candidates) or "none"
            sl_tp_error = (
                "Unable to resolve the filled position for TP/SL verification "
                f"(ticket candidates: {checked})"
            )
            sl_tp_apply_status = "unverified"
    except Exception as e:
        sl_tp_error = f"Error setting TP/SL: {str(e)}"
        sl_tp_apply_status = "failed"

    # --- Build warnings ---
    if sl_tp_apply_status == "failed":
        if position_ticket is not None:
            action_text = f"Run trade_modify --ticket {position_ticket} immediately"
        elif position_ticket_candidates:
            candidate_list = ", ".join(str(v) for v in position_ticket_candidates)
            primary_candidate = position_ticket_candidates[0]
            action_text = (
                f"Try trade_modify --ticket {primary_candidate} immediately "
                f"(candidate tickets: {candidate_list})"
            )
        else:
            action_text = (
                "Run trade_get_open immediately to find the live position ticket, "
                "then use trade_modify --ticket TICKET"
            )
        warnings_out.append(
            "CRITICAL: Order filled but TP/SL could not be applied. "
            f"{action_text}, or close the position."
        )
    elif sl_tp_apply_status == "unverified":
        if position_ticket is not None:
            action_text = (
                f"Run trade_get_open --ticket {position_ticket} immediately and confirm "
                "the live protection levels."
            )
        elif position_ticket_candidates:
            candidate_list = ", ".join(str(v) for v in position_ticket_candidates)
            action_text = (
                "Run trade_get_open immediately and confirm the live protection "
                f"levels (candidate tickets: {candidate_list})."
            )
        else:
            action_text = (
                "Run trade_get_open immediately to find the live position and confirm "
                "its protection levels."
            )
        warnings_out.append(
            "WARNING: Order filled but TP/SL protection could not be verified. "
            f"{action_text}"
        )
    if sl_tp_verification_failed:
        warnings_out.append(
            "SL/TP verification readback failed after broker acceptance. Verify the live position protection directly."
        )

    # --- Build outcome ---
    outcome: _ProtectionOutcome = {
        "position_ticket": position_ticket,
        "position_ticket_candidates": position_ticket_candidates or None,
        "position_ticket_resolution": position_ticket_resolution,
        "sl_tp_result": _build_sl_tp_result(
            requested_sl=stop_loss,
            requested_tp=take_profit,
            applied_sl=sl_applied,
            applied_tp=tp_applied,
            status=sl_tp_apply_status,
            error=sl_tp_error,
            broker_adjusted=sl_tp_broker_adjusted,
            adjustment=sl_tp_adjustment or None,
            attempts=sl_tp_attempts,
            last_retcode=sl_tp_last_retcode,
            last_comment=sl_tp_last_comment,
            verification_failed=sl_tp_verification_failed,
        ),
        "warnings": warnings_out,
    }
    if sl_tp_apply_status == "applied":
        outcome["protection_status"] = "protected"
    elif sl_tp_apply_status == "failed":
        outcome["protection_status"] = "unprotected_position"
    elif sl_tp_apply_status == "unverified":
        outcome["protection_status"] = "protection_unverified"
    return outcome


def _send_order_with_fill_mode_retry(
    mt5: Any,
    request: Dict[str, Any],
    *,
    symbol_info: Any = None,
    price_refresh: Optional[Callable[[Dict[str, Any]], bool]] = None,
    max_price_retries: int = 3,
) -> tuple[Any, Any, List[Dict[str, Any]], Dict[str, Any]]:
    attempts: List[Dict[str, Any]] = []
    last_result = None
    last_error = None
    last_request = dict(request)
    price_changed_codes = {
        validation._safe_int_attr(mt5, "TRADE_RETCODE_PRICE_CHANGED", 10020),
        validation._safe_int_attr(mt5, "TRADE_RETCODE_REQUOTE", 10004),
    }
    invalid_fill_codes = {
        validation._safe_int_attr(mt5, "TRADE_RETCODE_INVALID_FILL", 10030),
    }
    fill_modes: List[int] = []
    preferred_fill_mode = request.get("type_filling")
    try:
        if preferred_fill_mode is not None:
            fill_modes.append(int(preferred_fill_mode))
    except Exception:
        pass
    for candidate in validation._candidate_fill_modes(mt5, symbol_info):
        fill_mode = int(candidate)
        if fill_mode not in fill_modes:
            fill_modes.append(fill_mode)

    for fill_mode in fill_modes:
        attempt_request = dict(request)
        attempt_request["type_filling"] = int(fill_mode)
        price_retry_count = 0
        while True:
            request_to_send = dict(attempt_request)
            result = mt5.order_send(request_to_send)
            last_error = validation._safe_last_error(mt5)
            attempt: Dict[str, Any] = {"type_filling": int(fill_mode)}
            if price_retry_count > 0:
                attempt["price_retry"] = int(price_retry_count)
            if result is None:
                attempt["error"] = "order_send returned None"
                if last_error is not None:
                    attempt["last_error"] = last_error
                attempts.append(attempt)
                last_result = result
                last_request = request_to_send
                # A missing response is ambiguous: the broker may already have
                # accepted the order. Never retry with another fill mode.
                return result, last_error, attempts, last_request
            retcode = getattr(result, "retcode", None)
            attempt["retcode"] = retcode
            attempt["retcode_name"] = mt5.retcode_name(retcode)
            attempt["comment"] = getattr(result, "comment", None)
            attempts.append(attempt)
            last_result = result
            last_request = request_to_send
            try:
                if validation._retcode_is_accepted(mt5, getattr(result, "retcode", -1)):
                    return result, last_error, attempts, last_request
            except Exception:
                pass

            # Trying another fill mode is safe only when the broker explicitly
            # rejected the requested mode. Other failures, especially timeout,
            # may have accepted the order despite the missing/negative reply.
            if retcode not in invalid_fill_codes and retcode not in price_changed_codes:
                return result, last_error, attempts, last_request
            should_retry_same_fill = (
                retcode in price_changed_codes
                and price_retry_count < max(0, int(max_price_retries))
            )
            if not should_retry_same_fill:
                break

            refreshed = True
            if price_refresh is not None:
                try:
                    refreshed = bool(price_refresh(attempt_request))
                except Exception:
                    refreshed = False
            if not refreshed:
                break

            price_retry_count += 1
            _stdlib_time.sleep(0.15)
    return last_result, last_error, attempts, last_request


def _prepare_order_gateway(
    gateway: Optional[MT5TradingGateway] = None,
) -> tuple[Optional[MT5TradingGateway], Optional[Dict[str, Any]]]:
    mt5 = create_trading_gateway(
        gateway=gateway,
    )
    connection_error = trading_connection_error(mt5)
    if connection_error is not None:
        return None, connection_error
    return mt5, None


def _assess_order_account_state(
    mt5: Any,
    *,
    preflight: Optional[Dict[str, Any]] = None,
) -> tuple[Any, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return one canonical account snapshot and any new-order blockers."""
    try:
        account_info = mt5.account_info()
    except Exception:
        account_info = None
    if account_info is None:
        block = validation.snapshot_unavailable_error(
            mt5,
            snapshot="account",
            context="evaluate the current account execution and margin state",
        )
        block["blockers"] = ["account_snapshot_unavailable"]
        if preflight is not None:
            block["preflight"] = preflight
        return (
            None,
            {
                "margin_stress": {"status": "unknown", "reasons": []},
                "blockers": ["account_snapshot_unavailable"],
            },
            block,
        )

    numeric_margin_fields = {
        field: getattr(account_info, field, None)
        for field in ("equity", "margin", "margin_free", "margin_level")
    }
    has_concrete_margin_data = any(
        isinstance(value, Real) and not isinstance(value, bool)
        for value in numeric_margin_fields.values()
    )
    margin_stress = (
        assess_margin_stress(account_info)
        if has_concrete_margin_data
        else {"status": "unknown", "reasons": []}
    )
    margin_free = (
        validation._safe_float_attr(account_info, "margin_free", None)
        if isinstance(numeric_margin_fields["margin_free"], Real)
        and not isinstance(numeric_margin_fields["margin_free"], bool)
        else None
    )
    broker_trade_allowed = coerce_optional_bool(
        getattr(account_info, "trade_allowed", None)
    )
    if preflight is None:
        build_preflight = getattr(mt5, "build_trade_preflight", None)
        if callable(build_preflight):
            try:
                preflight = build_preflight(account_info=account_info)
            except TypeError:
                try:
                    preflight = build_preflight()
                except Exception:
                    preflight = None
            except Exception:
                preflight = None
    if not isinstance(preflight, dict):
        preflight = {}
    actionable_trade_allowed, trade_allowed_basis, _ = assess_new_exposure_allowed(
        broker_trade_allowed=broker_trade_allowed,
        margin_status=margin_stress.get("status"),
        execution_ready_strict=preflight.get("execution_ready_strict"),
        execution_ready_fallback=preflight.get("execution_ready"),
    )
    blockers: List[str] = []
    if broker_trade_allowed is False:
        blockers.append("account_trading_disabled")
    if margin_free is not None and margin_free <= 0:
        blockers.append("no_free_margin")
    if margin_stress.get("status") == "critical":
        blockers.append("critical_margin_stress")
    blockers = list(dict.fromkeys(blockers))

    account_state: Dict[str, Any] = {
        "margin_free": margin_free,
        "margin_stress": margin_stress,
        "blockers": blockers,
        "trade_allowed": actionable_trade_allowed,
        "new_exposure_allowed": actionable_trade_allowed,
        "trade_allowed_basis": trade_allowed_basis,
    }
    if broker_trade_allowed is not None:
        account_state["broker_trade_allowed"] = broker_trade_allowed
    account_state = {
        key: value for key, value in account_state.items() if value is not None
    }

    if not blockers:
        return account_info, account_state, None
    block: Dict[str, Any] = {
        "error": "Trading blocked by the current account execution or margin state.",
        "error_code": "trade_blocked_account_state",
        "blockers": blockers,
        "margin_free": margin_free,
        "margin_stress": margin_stress,
        "remediation": (
            "Enable account trading if disabled, reduce or close exposure, "
            "restore free margin, and re-check trade_account_info before "
            "placing a new order."
        ),
    }
    if preflight:
        block["preflight"] = preflight
    return account_info, account_state, block


def _prepare_order_symbol_context(
    mt5: Any,
    *,
    symbol: str,
    volume: float,
    defer_margin_block: bool = False,
) -> tuple[Optional[_OrderSymbolContext], Optional[Dict[str, Any]]]:
    preflight = mt5.build_trade_preflight()
    if not preflight.get("execution_ready_strict", preflight.get("execution_ready", True)):
        guidance = common._build_trade_preflight_guidance(preflight)
        return None, {
            "error": "Trading not ready in MT5 terminal/account preflight.",
            "preflight": preflight,
            "hint": guidance[0] if guidance else None,
            "next_steps": guidance,
        }

    account_info, _account_state, account_state_block = _assess_order_account_state(
        mt5,
        preflight=preflight,
    )
    if account_state_block is not None and not defer_margin_block:
        return None, account_state_block

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return None, {"error": f"Symbol {symbol} not found"}

    if not symbol_info.visible and not mt5.symbol_select(symbol, True):
        return None, {"error": f"Failed to select symbol {symbol}"}

    volume_validated, volume_error = validation._validate_volume(volume, symbol_info)
    if volume_error:
        return None, {"error": volume_error}

    return {
        "symbol_info": symbol_info,
        "volume": volume_validated,
        "account_info": account_info,
        "account_state_block": account_state_block,
    }, None


def _reduce_only_margin_exemption(
    mt5: Any,
    *,
    symbol: str,
    side: str,
    volume: float,
    account_info: Any,
) -> tuple[bool, Optional[Dict[str, Any]]]:
    """Prove that a netting market deal can only reduce current exposure."""
    if account_info is None or not _account_uses_netting(account_info):
        return False, None
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return False, validation.snapshot_unavailable_error(
            mt5,
            snapshot="positions",
            context="evaluate a reduce-only margin exemption",
        )
    net_side, net_volume = _resolve_existing_symbol_net(
        symbol=symbol,
        existing_positions=list(positions),
    )
    opposite = {"BUY": "SELL", "SELL": "BUY"}
    proven = bool(
        net_side is not None
        and side == opposite.get(net_side)
        and float(volume) <= float(net_volume) + 1e-12
    )
    return proven, None


def _evaluate_live_trade_guardrails(
    mt5: Any,
    *,
    symbol: str,
    volume: float,
    stop_loss: Optional[float],
    deviation: Optional[int],
    side: str,
    entry_price: float,
    symbol_info: Any,
) -> Optional[Dict[str, Any]]:
    guardrail_config = _guardrail_snapshot(trade_guardrails_config)
    if not guardrail_config.is_enabled():
        return None
    try:
        account_info = mt5.account_info()
    except Exception:
        account_info = None
    positions, pending_orders, snapshot_error = load_guardrail_book_snapshots(
        mt5,
        guardrail_config,
        account_info=account_info,
    )
    if snapshot_error is not None:
        return snapshot_error
    return evaluate_trade_guardrails(
        guardrail_config,
        symbol=symbol,
        volume=volume,
        stop_loss=stop_loss,
        deviation=deviation,
        side=side,
        entry_price=entry_price,
        account_info=account_info,
        existing_positions=positions,
        existing_pending_orders=pending_orders,
        symbol_info=symbol_info,
        symbol_info_resolver=mt5.symbol_info,
    )


def _build_order_comment_payload(
    comment: Optional[str],
    *,
    default: str,
) -> tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    request_comment = comments._normalize_trade_comment(comment, default=default)
    return (
        request_comment,
        comments._comment_sanitization_info(comment, request_comment),
        comments._comment_truncation_info(comment, request_comment),
    )


def _order_result_value(result: Any, field: str) -> Any:
    try:
        return getattr(result, field)
    except Exception:
        return None


def _meaningful_last_error(value: Any) -> Any:
    """Drop MT5's success sentinel when an order result itself is a failure."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and value:
        try:
            code = int(value[0])
        except (TypeError, ValueError):
            code = None
        message = str(value[1] if len(value) > 1 else "").strip().lower()
        if code in {0, 1} and message in {"", "success", "ok", "no error"}:
            return None
    return value


def _submit_order_request(
    mt5: Any,
    request: Dict[str, Any],
    *,
    base_error: str,
    invalid_comment_error: str,
    symbol_info: Any = None,
    price_refresh: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> tuple[Optional[_OrderSubmitOutcome], Optional[Dict[str, Any]]]:
    result, last_error, fill_mode_attempts, used_request = _send_order_with_fill_mode_retry(
        mt5,
        request,
        symbol_info=symbol_info,
        price_refresh=price_refresh,
    )
    diagnostic_last_error = _meaningful_last_error(last_error)
    if result is None:
        failure = {
            "error": (
                "Order submission outcome is unknown; the broker may have accepted "
                "the request. Do not retry without reconciling open orders and positions."
            ),
            "error_code": "order_send_ambiguous",
            "ambiguous": True,
            "fill_mode_attempts": fill_mode_attempts,
        }
        if diagnostic_last_error is not None:
            failure["last_error"] = diagnostic_last_error
        return None, failure

    retcode = _order_result_value(result, "retcode")
    if not validation._retcode_is_accepted(mt5, retcode):
        error_message = base_error
        ambiguous = retcode == validation._safe_int_attr(
            mt5, "TRADE_RETCODE_TIMEOUT", 10012
        )
        if ambiguous:
            error_message = (
                "Order submission timed out; the broker may have accepted the request. "
                "Do not retry without reconciling open orders and positions."
            )
        invalid_comment = comments._invalid_comment_error_text(result, last_error)
        if invalid_comment is not None:
            error_message = invalid_comment_error
        failure = {
            "error": error_message,
            "retcode": retcode,
            "retcode_name": mt5.retcode_name(retcode),
            "comment": _order_result_value(result, "comment"),
            "mt5_request_id": _order_result_value(result, "request_id"),
            "fill_mode_attempts": fill_mode_attempts,
        }
        if diagnostic_last_error is not None:
            failure["last_error"] = diagnostic_last_error
        if ambiguous:
            failure.update(error_code="order_send_ambiguous", ambiguous=True)
        return None, failure

    return {
        "result": result,
        "fill_mode_attempts": fill_mode_attempts,
        "used_request": used_request,
    }, None


def _attach_comment_response_metadata(
    out: Dict[str, Any],
    warnings_out: List[str],
    *,
    comment_sanitization: Optional[Dict[str, Any]],
    comment_truncation: Optional[Dict[str, Any]],
    applied_comment: Optional[str] = None,
    requested_comment: Optional[str] = None,
) -> None:
    if "comment" in out:
        out["broker_message"] = out.get("comment")
    if applied_comment is not None:
        out["comment"] = applied_comment
        out["applied_comment"] = applied_comment
    if requested_comment is not None:
        out["requested_comment"] = requested_comment
    if comment_sanitization:
        out["comment_sanitization"] = comment_sanitization
        warnings_out.append(
            f"Comment sanitized for broker compatibility: '{comment_sanitization['applied']}'"
        )
    if comment_truncation:
        out["comment_truncation"] = comment_truncation
        warnings_out.append(
            f"Comment truncated to {comment_truncation['max_length']} characters: '{comment_truncation['applied']}'"
        )


def _round_preview_price(value: Optional[float], *, digits: int) -> Optional[float]:
    return round_finite(value, digits, on_invalid="none")


def _order_type_constant(mt5: Any, order_type: str) -> Optional[int]:
    constant_name = f"ORDER_TYPE_{str(order_type).upper().strip()}"
    return validation._safe_int_attr(mt5, constant_name, None)


def _level_preview_fields(
    prefix: str,
    level: Optional[Union[int, float]],
    *,
    entry_price: float,
    point: float,
    points_per_pip: Optional[float],
) -> Dict[str, Any]:
    try:
        level_value = float(level) if level not in (None, 0) else None
    except (TypeError, ValueError):
        level_value = None
    if level_value is None or not math.isfinite(level_value):
        return {}

    out: Dict[str, Any] = {}
    if point > 0:
        distance_points = abs(level_value - entry_price) / point
        out[f"{prefix}_distance_points"] = round(distance_points, 2)
        if points_per_pip is not None and points_per_pip > 0:
            out[f"{prefix}_distance_pips"] = round(
                distance_points / points_per_pip,
                2,
            )
    if entry_price:
        out[f"{prefix}_distance_pct"] = round(
            (abs(level_value - entry_price) / abs(entry_price)) * 100.0,
            6,
        )
    return out


def _margin_preview_fields(
    mt5: Any,
    *,
    order_type_value: Optional[int],
    symbol: str,
    volume: float,
    entry_price: float,
    account_info: Any = None,
) -> Dict[str, Any]:
    if order_type_value is None:
        return {}
    order_calc_margin = getattr(getattr(mt5, "adapter", None), "order_calc_margin", None)
    if not callable(order_calc_margin):
        return {}

    margin = validation.coerce_finite_float(
        order_calc_margin(order_type_value, symbol, volume, entry_price)
    )
    if margin is None:
        return {}

    out: Dict[str, Any] = {"margin_required": round(float(margin), 2)}
    margin_free = validation._safe_float_attr(account_info, "margin_free")
    if margin_free is not None and math.isfinite(margin_free):
        out["margin_free"] = round(float(margin_free), 2)
        out["margin_sufficient"] = float(margin_free) >= float(margin)
    return out


def _candidate_risk_preview_fields(
    *,
    symbol_info: Any,
    account_info: Any,
    volume: float,
    entry_price: float,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    side: str,
) -> Dict[str, Any]:
    candidate_risk: Dict[str, Any] = {
        "status": "unavailable",
        "basis": "tick_value_linear_sensitivity",
    }
    risk_currency, risk_error = _estimate_order_risk_currency(
        symbol_info=symbol_info,
        volume=volume,
        entry_price=entry_price,
        stop_loss=stop_loss,
        side=side,
    )
    if risk_currency is not None:
        candidate_risk["risk_currency"] = round(float(risk_currency), 2)
        candidate_risk["status"] = "ok"
    elif risk_error:
        candidate_risk["risk_error"] = risk_error
    reward_side = "SELL" if side == "BUY" else "BUY"
    reward_currency, reward_error = _estimate_order_risk_currency(
        symbol_info=symbol_info,
        volume=volume,
        entry_price=entry_price,
        stop_loss=take_profit,
        side=reward_side,
    )
    if reward_currency is not None:
        candidate_risk["reward_currency"] = round(float(reward_currency), 2)
    elif reward_error:
        candidate_risk["reward_error"] = reward_error
    if (
        risk_currency is not None
        and reward_currency is not None
        and float(risk_currency) > 0.0
    ):
        candidate_risk["reward_risk_ratio"] = round(
            float(reward_currency) / float(risk_currency), 4
        )
    equity = validation._safe_float_attr(account_info, "equity")
    if (
        risk_currency is not None
        and equity is not None
        and math.isfinite(equity)
        and equity > 0.0
    ):
        candidate_risk["risk_pct_of_equity"] = round(
            float(risk_currency) / float(equity) * 100.0, 4
        )
        candidate_risk["equity"] = round(float(equity), 2)
    if candidate_risk.get("status") != "ok":
        candidate_risk["status"] = str(
            candidate_risk.get("risk_error") or "unavailable"
        )
    return candidate_risk


def _split_pending_validation_error(
    validation_error: Optional[Dict[str, Any]],
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Separate pending-entry failures from protective-level failures."""
    if validation_error is None:
        return None, None
    error_text = str(validation_error.get("error") or "").lower()
    if error_text.startswith(("stop_loss", "take_profit")):
        return None, validation_error
    return validation_error, None


def build_trade_place_dry_run_preview(  # noqa: C901
    *,
    symbol: str,
    volume: float,
    order_type: str,
    pending: bool,
    price: Optional[Union[int, float]],
    stop_limit_price: Optional[Union[int, float]] = None,
    stop_loss: Optional[Union[int, float]],
    take_profit: Optional[Union[int, float]],
    gateway: Optional[MT5TradingGateway] = None,
) -> Dict[str, Any]:
    """Build a quote-based order preview without sending or checking an order."""
    mt5 = create_trading_gateway(gateway=gateway)
    connection_error = trading_connection_error(mt5)
    if connection_error is not None:
        return {
            "preview_error": connection_error.get("error"),
            "preview_error_code": connection_error.get(
                "error_code",
                "mt5_connection_error",
            ),
        }

    account_info, account_state, account_state_block = _assess_order_account_state(mt5)

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return {
            "preview_error": f"Symbol {symbol} not found",
            "preview_error_code": "symbol_not_found",
            "remediation": (
                "Use symbols_list to find the broker's exact symbol name and suffix."
            ),
            "related_tools": ["symbols_list"],
        }
    if not getattr(symbol_info, "visible", True) and not mt5.symbol_select(symbol, True):
        return {"preview_error": f"Failed to select symbol {symbol}"}

    volume_validated, volume_error = validation._validate_volume(volume, symbol_info)
    if volume_error:
        return {"preview_error": volume_error}

    price_context = validation._symbol_price_context(symbol_info)
    price_inputs, price_inputs_error = validation._normalize_trade_price_inputs(
        symbol_info=symbol_info,
        price=price,
        require_price=pending,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
    if price_inputs_error is not None:
        invalid_pending_price = pending and validation._normalize_price_for_symbol(
            price,
            point=float(price_context["price_increment"]),
            digits=int(price_context["digits"]),
        ) is None
        return {
            "preview_error": price_inputs_error,
            "preview_error_code": (
                "invalid_pending_price"
                if invalid_pending_price
                else "invalid_protection_levels"
            ),
        }
    normalized_price = price_inputs["price"]
    normalized_sl = price_inputs["stop_loss"]
    normalized_tp = price_inputs["take_profit"]
    order_type_norm = str(order_type).upper().strip()
    is_stop_limit = order_type_norm in {"BUY_STOP_LIMIT", "SELL_STOP_LIMIT"}
    normalized_stop_limit = validation._normalize_price_for_symbol(
        stop_limit_price,
        point=float(price_context["price_increment"]),
        digits=int(price_context["digits"]),
    )
    if is_stop_limit and normalized_stop_limit is None:
        return {
            "preview_error": "stop_limit_price is required for stop-limit orders.",
            "preview_error_code": "invalid_stop_limit_price",
        }

    wall_clock_now = _stdlib_time.time()
    raw_tick = mt5.symbol_info_tick(symbol)
    raw_tick_clock = validation._tick_clock_reference(
        raw_tick,
        wall_clock_epoch=wall_clock_now,
    )
    quote_now = float(raw_tick_clock["reference_epoch"])
    tick, quote_source = resolve_quote_tick(
        mt5,
        symbol,
        raw_tick,
        now_epoch=quote_now,
    )
    if tick is None:
        return {"preview_error": f"Failed to get current price for {symbol}"}
    send_path_freshness_error = (
        validation._validate_tick_freshness(
            raw_tick,
            symbol=symbol,
            max_age_seconds=_TRADE_TICK_MAX_AGE_SECONDS,
        )
        if raw_tick is not None
        else {
            "error": f"Failed to get current price for {symbol}",
            "tick_age_status": "missing",
        }
    )

    bid = validation.coerce_finite_float(tick_value(tick, "bid"))
    ask = validation.coerce_finite_float(tick_value(tick, "ask"))
    if bid is None or ask is None or not math.isfinite(bid) or not math.isfinite(ask):
        return {"preview_error": f"Failed to get valid bid/ask for {symbol}"}

    digits = validation._safe_int_attr(symbol_info, "digits", 6)
    point = validation._safe_float_attr(symbol_info, "point") or 0.0
    side = "BUY" if str(order_type).upper().startswith("BUY") else "SELL"
    order_type_value = _order_type_constant(mt5, order_type)
    trigger_price = float(normalized_price) if pending else None
    entry_price = (
        float(normalized_stop_limit)
        if pending and is_stop_limit and normalized_stop_limit is not None
        else float(normalized_price)
        if pending
        else (ask if side == "BUY" else bid)
    )
    points_per_pip = forex_points_per_pip(
        symbol,
        path=str(getattr(symbol_info, "path", "") or ""),
        point=point,
        digits=digits,
    )
    spread_metrics = compute_spread_metrics(
        bid,
        ask,
        point=point,
        points_per_pip=points_per_pip,
    )

    quote_context = common.build_trade_quote_context(
        symbol,
        tick,
        now_epoch=quote_now,
        source_metadata=quote_source,
    )
    if raw_tick_clock.get("clock_reconciled") is True:
        quote_context.update(
            {
                "clock_reconciled": True,
                "quote_clock_reference": raw_tick_clock["clock_reference"],
                "local_clock_lag_seconds": raw_tick_clock[
                    "local_clock_lag_seconds"
                ],
                "clock_reconciliation_limit_seconds": raw_tick_clock[
                    "clock_reconciliation_limit_seconds"
                ],
                "data_age_anchor": "broker_tick_reconciled_clock",
                "clock_reconciliation_note": (
                    "The workstation clock trails the synchronously acquired "
                    "broker tick; quote age was evaluated on the bounded broker "
                    "clock reference."
                ),
            }
        )
    quote_context["send_path_tick_fresh"] = send_path_freshness_error is None
    if send_path_freshness_error is not None:
        quote_context["usable_for_live_trading"] = False
        quote_context["usable_for_live_trading_basis"] = "submission_tick_freshness_required"
        quote_context["send_path_freshness_error"] = send_path_freshness_error.get(
            "error"
        )
        if send_path_freshness_error.get("tick_age_status") is not None:
            quote_context["send_path_tick_age_status"] = send_path_freshness_error[
                "tick_age_status"
            ]
        if send_path_freshness_error.get("timestamp_skew_seconds") is not None:
            quote_context["send_path_timestamp_skew_seconds"] = (
                send_path_freshness_error["timestamp_skew_seconds"]
            )
    out: Dict[str, Any] = {
        "bid": _round_preview_price(bid, digits=digits),
        "ask": _round_preview_price(ask, digits=digits),
        "estimated_fill_price": _round_preview_price(entry_price, digits=digits),
        "quote_context": quote_context,
    }
    if account_state is not None:
        out["account_state"] = account_state
    if account_state_block is not None:
        out["account_blockers"] = list(account_state_block["blockers"])
    if pending:
        out["entry_price"] = _round_preview_price(entry_price, digits=digits)
        out["trigger_price"] = _round_preview_price(trigger_price, digits=digits)
        if normalized_stop_limit is not None:
            out["stop_limit_price"] = _round_preview_price(
                normalized_stop_limit,
                digits=digits,
            )

    if point > 0:
        if spread_metrics["spread_points"] is not None:
            out["spread_points"] = round(spread_metrics["spread_points"], 2)
        if spread_metrics["spread_pips"] is not None:
            out["spread_pips"] = round(spread_metrics["spread_pips"], 2)
        if spread_metrics["spread_pct"] is not None:
            out["spread_pct"] = round(spread_metrics["spread_pct"], 6)
        broker_distance = validation._broker_distance_metadata(symbol_info)
        out["min_distance_points"] = int(broker_distance["min_distance_points"])

    out.update(
        _level_preview_fields(
            "sl",
            normalized_sl,
            entry_price=entry_price,
            point=point,
            points_per_pip=points_per_pip,
        )
    )
    out.update(
        _level_preview_fields(
            "tp",
            normalized_tp,
            entry_price=entry_price,
            point=point,
            points_per_pip=points_per_pip,
        )
    )

    out["candidate_risk"] = _candidate_risk_preview_fields(
        symbol_info=symbol_info,
        account_info=account_info,
        volume=float(volume_validated),
        entry_price=float(entry_price),
        stop_loss=None if normalized_sl is None else float(normalized_sl),
        take_profit=None if normalized_tp is None else float(normalized_tp),
        side=side,
    )

    out["units"] = {
        "spread_points": UNIT_BROKER_POINTS,
        "spread_pips": UNIT_PIPS,
        "sl_distance_points": UNIT_BROKER_POINTS,
        "sl_distance_pips": UNIT_PIPS,
        "sl_distance_pct": UNIT_PERCENT,
        "tp_distance_points": UNIT_BROKER_POINTS,
        "tp_distance_pips": UNIT_PIPS,
        "tp_distance_pct": UNIT_PERCENT,
        "risk_currency": "account_currency",
    }

    validation_error: Optional[Dict[str, Any]] = None
    if pending:
        if order_type_value is not None:
            validation_error = validation._validate_pending_order_levels(
                symbol_info=symbol_info,
                tick=tick,
                order_type_value=order_type_value,
                price=float(trigger_price),
                stop_loss=None if normalized_sl is None else float(normalized_sl),
                take_profit=None if normalized_tp is None else float(normalized_tp),
                mt5=mt5,
                stop_limit_price=(
                    None
                    if normalized_stop_limit is None
                    else float(normalized_stop_limit)
                ),
            )
    else:
        validation_error = validation._validate_live_protection_levels(
            symbol_info=symbol_info,
            tick=tick,
            side=side,
            stop_loss=None if normalized_sl is None else float(normalized_sl),
            take_profit=None if normalized_tp is None else float(normalized_tp),
        )
    protection_error = validation_error
    if pending:
        pending_entry_error, protection_error = _split_pending_validation_error(
            validation_error
        )
        out["pending_levels_valid"] = validation_error is None
        out["pending_entry_valid"] = pending_entry_error is None
        out["pending_entry_error"] = (
            pending_entry_error.get("error")
            if pending_entry_error is not None
            else None
        )
        if validation_error is not None:
            out["pending_levels_error"] = validation_error.get("error")
            out["preview_error"] = validation_error.get("error")
            out["preview_error_code"] = "invalid_pending_order_levels"
    if normalized_sl is not None or normalized_tp is not None:
        out["sl_tp_valid"] = protection_error is None
        if protection_error is not None:
            out["sl_tp_error"] = protection_error.get("error")

    # MT5's margin calculator accepts the active BUY/SELL action, even when
    # previewing a pending order.  Passing BUY_LIMIT/SELL_STOP constants can
    # produce a misleading zero on some brokers.
    margin_action_value = _order_type_constant(mt5, side)
    margin_fields = _margin_preview_fields(
        mt5,
        order_type_value=margin_action_value,
        symbol=symbol,
        volume=float(volume_validated),
        entry_price=float(entry_price),
        account_info=account_info,
    )
    if margin_fields:
        margin_fields["margin_action"] = side
        margin_fields["margin_estimate_basis"] = (
            "pending_fill_side_at_entry_price"
            if pending
            else "market_fill_side_at_estimated_price"
        )
        if pending and "margin_required" in margin_fields:
            margin_fields["margin_required_when_filled"] = margin_fields[
                "margin_required"
            ]
    out.update(margin_fields)
    return {key: value for key, value in out.items() if value is not None}


def _place_market_order(  # noqa: C901
    symbol: str,
    volume: float,
    order_type: MarketOrderTypeInput,
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    comment: Optional[str] = None,
    magic: Optional[int] = None,
    deviation: int = 20,
    gateway: Optional[MT5TradingGateway] = None,
) -> dict:
    """Internal helper to place a market order."""
    mt5, connection_error = _prepare_order_gateway(gateway)
    if connection_error is not None:
        return connection_error

    def _place_market_order():  # noqa: C901
        try:
            order_context, order_context_error = _prepare_order_symbol_context(
                mt5,
                symbol=symbol,
                volume=volume,
                defer_margin_block=True,
            )
            if order_context_error is not None:
                return order_context_error
            symbol_info = order_context["symbol_info"]
            volume_validated = order_context["volume"]

            # Normalize and validate requested order type
            t, order_type_error = validation._normalize_order_type_input(order_type)
            if order_type_error:
                return {"error": order_type_error}
            if t not in {"BUY", "SELL"}:
                return {"error": f"Unsupported order_type '{order_type}'. Use BUY or SELL for market orders."}
            side = t

            account_state_block = order_context.get("account_state_block")
            if account_state_block is not None:
                blockers = set(account_state_block.get("blockers") or [])
                margin_only = blockers and blockers <= {
                    "no_free_margin",
                    "critical_margin_stress",
                }
                exempt = False
                exemption_error = None
                if margin_only:
                    exempt, exemption_error = _reduce_only_margin_exemption(
                        mt5,
                        symbol=symbol,
                        side=side,
                        volume=volume_validated,
                        account_info=order_context.get("account_info"),
                    )
                if exemption_error is not None:
                    return exemption_error
                if not exempt:
                    return account_state_block

            deviation_validated, deviation_error = validation._validate_deviation(deviation)
            if deviation_error:
                return {"error": deviation_error}

            price_inputs, price_inputs_error = validation._normalize_trade_price_inputs(
                symbol_info=symbol_info,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            if price_inputs_error is not None:
                return {
                    "error": price_inputs_error,
                    "error_code": "invalid_protection_levels",
                }
            norm_sl = price_inputs["stop_loss"]
            norm_tp = price_inputs["take_profit"]
            if (
                (norm_sl is not None or norm_tp is not None)
                and order_context.get("account_info") is not None
                and _account_uses_netting(order_context["account_info"])
            ):
                existing_symbol_positions = mt5.positions_get(symbol=symbol)
                if existing_symbol_positions is None:
                    return validation.snapshot_unavailable_error(
                        mt5,
                        snapshot="positions",
                        context="validate netting-account protection scope",
                    )
                if list(existing_symbol_positions):
                    return {
                        "success": False,
                        "error": (
                            "Cannot attach order-specific SL/TP while a netting-account "
                            f"position already exists for {symbol}; MT5 protection would "
                            "replace levels on the entire aggregate position."
                        ),
                        "error_code": "netting_position_protection_conflict",
                        "remediation": (
                            "Modify the existing aggregate position explicitly, or close/reduce "
                            "it before placing a newly protected order."
                        ),
                    }

            # Validate against a recent quote, then refresh again right before send.
            validate_tick = mt5.symbol_info_tick(symbol)
            if validate_tick is None:
                return {"error": f"Failed to get current price for {symbol}"}
            tick_freshness_error = validation._validate_tick_freshness(
                validate_tick,
                symbol=symbol,
                max_age_seconds=_TRADE_TICK_MAX_AGE_SECONDS,
            )
            if tick_freshness_error is not None:
                return tick_freshness_error
            live_protection_error = validation._validate_live_protection_levels(
                symbol_info=symbol_info,
                tick=validate_tick,
                side=side,
                stop_loss=norm_sl,
                take_profit=norm_tp,
            )
            if live_protection_error is not None:
                return live_protection_error

            send_tick = mt5.symbol_info_tick(symbol)
            if send_tick is None:
                return {"error": f"Failed to get fresh price for {symbol}"}
            tick_freshness_error = validation._validate_tick_freshness(
                send_tick,
                symbol=symbol,
                max_age_seconds=_TRADE_TICK_MAX_AGE_SECONDS,
            )
            if tick_freshness_error is not None:
                return tick_freshness_error
            price = send_tick.ask if side == "BUY" else send_tick.bid
            live_protection_error = validation._validate_live_protection_levels(
                symbol_info=symbol_info,
                tick=send_tick,
                side=side,
                stop_loss=norm_sl,
                take_profit=norm_tp,
            )
            if live_protection_error is not None:
                return live_protection_error
            guardrail_block = _evaluate_live_trade_guardrails(
                mt5,
                symbol=symbol,
                volume=volume_validated,
                stop_loss=norm_sl,
                deviation=deviation_validated,
                side=side,
                entry_price=price,
                symbol_info=symbol_info,
            )
            if guardrail_block is not None:
                return guardrail_block
            request_comment, comment_sanitization, comment_truncation = _build_order_comment_payload(
                comment,
                default="mtdata order",
            )
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume_validated,
                "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "deviation": deviation_validated,
                "magic": magic if magic is not None else _configured_order_magic(),
                "comment": request_comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": validation._safe_int_attr(mt5, "ORDER_FILLING_IOC", 1),
            }
            if norm_sl is not None:
                request["sl"] = float(norm_sl)
            if norm_tp is not None:
                request["tp"] = float(norm_tp)

            def _refresh_market_order_price(attempt_request: Dict[str, Any]) -> bool:
                refreshed_tick = mt5.symbol_info_tick(symbol)
                if refreshed_tick is None:
                    return False
                freshness_error = validation._validate_tick_freshness(
                    refreshed_tick,
                    symbol=symbol,
                    max_age_seconds=_TRADE_TICK_MAX_AGE_SECONDS,
                )
                if freshness_error is not None:
                    logger.warning(
                        "Refreshed tick for %s is stale; aborting price retry: %s",
                        symbol,
                        freshness_error.get("error"),
                    )
                    return False
                refreshed_price = refreshed_tick.ask if side == "BUY" else refreshed_tick.bid
                try:
                    refreshed_price = float(refreshed_price)
                except Exception:
                    return False
                if not math.isfinite(refreshed_price) or refreshed_price <= 0.0:
                    return False
                attempt_request["price"] = refreshed_price
                return True

            send_outcome, send_error = _submit_order_request(
                mt5,
                request,
                base_error="Failed to send order",
                invalid_comment_error=(
                    "Failed to send order: broker rejected the comment field. "
                    "Comments are sanitized to letters/numbers/spaces/_/./-; "
                    "try a simpler comment or omit --comment."
                ),
                symbol_info=symbol_info,
                price_refresh=_refresh_market_order_price,
            )
            if send_error is not None:
                return send_error
            result = send_outcome["result"]
            fill_mode_attempts = send_outcome["fill_mode_attempts"]
            used_request = send_outcome["used_request"]
            retcode = _order_result_value(result, "retcode")
            execution_status = validation._trade_execution_status(mt5, retcode)
            filled_volume = validation._safe_float_attr(
                result, "volume", default=None
            )
            if (
                execution_status == "complete"
                and filled_volume is not None
                and filled_volume > 0.0
                and filled_volume + 1e-12 < float(volume_validated)
            ):
                execution_status = "partial"
            remaining_volume = (
                max(0.0, float(volume_validated) - float(filled_volume))
                if filled_volume is not None
                else None
            )

            order_ticket = validation._safe_int_ticket(getattr(result, "order", None))
            deal_ticket = validation._safe_int_ticket(getattr(result, "deal", None))
            position_ticket_candidates: List[int] = []
            for cand in (order_ticket, deal_ticket):
                if cand is not None and cand not in position_ticket_candidates:
                    position_ticket_candidates.append(cand)

            sl_tp_requested = bool(norm_sl is not None or norm_tp is not None)
            position_ticket = None
            position_ticket_resolution = None
            position_ticket_candidates_out: Optional[List[int]] = (
                position_ticket_candidates or None
            )
            protection_status = None
            warnings_out: List[str] = []
            if sl_tp_requested and execution_status != "queued":
                protection_outcome = _attach_post_fill_protection(
                    mt5,
                    symbol=symbol,
                    side=side,
                    volume=(
                        filled_volume
                        if filled_volume is not None and filled_volume > 0.0
                        else volume_validated
                    ),
                    position_ticket_candidates=position_ticket_candidates,
                    stop_loss=norm_sl,
                    take_profit=norm_tp,
                    symbol_info=symbol_info,
                    comment=comment,
                    request_comment="trade_place_market",
                    magic=validation._safe_int_magic(request.get("magic")),
                )
                position_ticket = protection_outcome.get("position_ticket")
                position_ticket_resolution = protection_outcome.get(
                    "position_ticket_resolution"
                )
                position_ticket_candidates_out = protection_outcome.get(
                    "position_ticket_candidates"
                ) or position_ticket_candidates_out
                sl_tp_result = protection_outcome["sl_tp_result"]
                protection_status = protection_outcome.get("protection_status")
                for warning in list(protection_outcome.get("warnings") or []):
                    warning_text = str(warning).strip()
                    if warning_text:
                        warnings_out.append(warning_text)
            elif not sl_tp_requested:
                if position_ticket_candidates:
                    position_obj, resolved_ticket, resolve_info = _resolve_open_position(
                        mt5,
                        ticket_candidates=position_ticket_candidates,
                        symbol=symbol,
                        side=side,
                        volume=volume_validated,
                        magic=validation._safe_int_magic(request.get("magic")),
                    )
                    position_ticket = resolved_ticket
                    position_ticket_resolution = {
                        **dict(resolve_info),
                        "matched": position_obj is not None and resolved_ticket is not None,
                    }
                sl_tp_result = _build_sl_tp_result(
                    requested_sl=None,
                    requested_tp=None,
                    applied_sl=None,
                    applied_tp=None,
                    status="not_requested",
                    error=None,
                    broker_adjusted=False,
                    adjustment=None,
                    attempts=0,
                    last_retcode=None,
                    last_comment=None,
                )
            else:
                sl_tp_result = _build_sl_tp_result(
                    requested_sl=norm_sl,
                    requested_tp=norm_tp,
                    applied_sl=None,
                    applied_tp=None,
                    status="pending_fill",
                    error=(
                        "Order is queued; protection cannot be verified until a "
                        "position fill exists."
                    ),
                    broker_adjusted=False,
                    adjustment=None,
                    attempts=0,
                    last_retcode=None,
                    last_comment=None,
                )
                protection_status = "pending_fill"
                warnings_out.append(
                    "Order is queued, not filled; reconcile the resulting position "
                    "and attach SL/TP after execution."
                )
            out: Dict[str, Any] = {
                "success": execution_status == "complete",
                "execution_status": execution_status,
                "retcode": retcode,
                "retcode_name": mt5.retcode_name(retcode),
                "deal": _order_result_value(result, "deal"),
                "order": _order_result_value(result, "order"),
                "volume": _order_result_value(result, "volume"),
                "requested_volume": volume_validated,
                "filled_volume": filled_volume,
                "remaining_volume": remaining_volume,
                "price": _order_result_value(result, "price"),
                "bid": _order_result_value(result, "bid"),
                "ask": _order_result_value(result, "ask"),
                "comment": _order_result_value(result, "comment"),
                "mt5_request_id": _order_result_value(result, "request_id"),
                "position_ticket": position_ticket,
                "position_ticket_candidates": position_ticket_candidates_out,
                "position_ticket_resolution": position_ticket_resolution,
                "type_filling_used": used_request.get("type_filling"),
                "sl_tp_result": sl_tp_result,
            }
            _attach_comment_response_metadata(
                out,
                warnings_out,
                comment_sanitization=comment_sanitization,
                comment_truncation=comment_truncation,
                applied_comment=request_comment,
                requested_comment=comment,
            )
            if protection_status is not None:
                out["protection_status"] = protection_status
            if fill_mode_attempts:
                out["fill_mode_attempts"] = fill_mode_attempts
            if warnings_out:
                out["warnings"] = warnings_out
            return out

        except Exception as e:
            return {"error": str(e)}

    # Keep the portfolio snapshot, guardrail decision, and broker submission in
    # one process-wide critical section so concurrent tool calls cannot spend
    # the same exposure or wallet-risk headroom.
    with _TRADE_DECISION_LOCK:
        return _place_market_order()


def _place_pending_order(  # noqa: C901
    symbol: str,
    volume: float,
    order_type: OrderTypeInput,
    price: Union[int, float],
    stop_limit_price: Optional[Union[int, float]] = None,
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    expiration: Optional[ExpirationValue] = None,
    comment: Optional[str] = None,
    magic: Optional[int] = None,
    deviation: int = 20,
    gateway: Optional[MT5TradingGateway] = None,
) -> dict:
    """Internal helper to place a pending order."""
    mt5, connection_error = _prepare_order_gateway(gateway)
    if connection_error is not None:
        return connection_error

    def _place_pending_order():  # noqa: C901
        try:
            order_context, order_context_error = _prepare_order_symbol_context(
                mt5,
                symbol=symbol,
                volume=volume,
            )
            if order_context_error is not None:
                return order_context_error
            symbol_info = order_context["symbol_info"]
            volume_validated = order_context["volume"]

            initial_tick = mt5.symbol_info_tick(symbol)
            if initial_tick is None:
                return {"error": f"Failed to get current price for {symbol}"}

            deviation_validated, deviation_error = validation._validate_deviation(deviation)
            if deviation_error:
                return {"error": deviation_error}

            # Normalize and validate requested order type
            t, order_type_error = validation._normalize_order_type_input(order_type)
            if order_type_error:
                return {"error": order_type_error}
            explicit_map = {
                "BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT,
                "BUY_STOP": mt5.ORDER_TYPE_BUY_STOP,
                "BUY_STOP_LIMIT": validation._safe_int_attr(
                    mt5, "ORDER_TYPE_BUY_STOP_LIMIT", 6
                ),
                "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
                "SELL_STOP": mt5.ORDER_TYPE_SELL_STOP,
                "SELL_STOP_LIMIT": validation._safe_int_attr(
                    mt5, "ORDER_TYPE_SELL_STOP_LIMIT", 7
                ),
            }

            price_inputs, price_inputs_error = validation._normalize_trade_price_inputs(
                symbol_info=symbol_info,
                price=price,
                require_price=True,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            if price_inputs_error is not None:
                price_context = validation._symbol_price_context(symbol_info)
                invalid_pending_price = validation._normalize_price_for_symbol(
                    price,
                    point=float(price_context["price_increment"]),
                    digits=int(price_context["digits"]),
                ) is None
                return {
                    "error": price_inputs_error,
                    "error_code": (
                        "invalid_pending_price"
                        if invalid_pending_price
                        else "invalid_protection_levels"
                    ),
                }
            point = float(price_inputs["point"])
            norm_price = float(price_inputs["price"])
            norm_sl = price_inputs["stop_loss"]
            norm_tp = price_inputs["take_profit"]
            is_stop_limit = t in {"BUY_STOP_LIMIT", "SELL_STOP_LIMIT"}
            norm_stop_limit = validation._normalize_price_for_symbol(
                stop_limit_price,
                point=float(price_inputs["price_increment"]),
                digits=int(price_inputs["digits"]),
            )
            if is_stop_limit and norm_stop_limit is None:
                return {
                    "error": "stop_limit_price is required for stop-limit orders.",
                    "error_code": "invalid_stop_limit_price",
                }

            normalized_expiration, expiration_specified = time._normalize_pending_expiration(expiration)
            live_tick = mt5.symbol_info_tick(symbol) or initial_tick
            if live_tick is None:
                return {"error": f"Failed to refresh current price for {symbol}"}
            bid = validation._safe_float_attr(live_tick, "bid")
            ask = validation._safe_float_attr(live_tick, "ask")
            price_tol = point * 0.1 if point > 0 else 1e-9

            if t == "BUY" and abs(norm_price - ask) <= price_tol:
                return {
                    "error": (
                        "price is at market for BUY pending order. "
                        f"price={norm_price}, ask={ask}. Use a market order or move price away from ask."
                    )
                }
            if t == "SELL" and abs(norm_price - bid) <= price_tol:
                return {
                    "error": (
                        "price is at market for SELL pending order. "
                        f"price={norm_price}, bid={bid}. Use a market order or move price away from bid."
                    )
                }

            order_type_value = None
            if t in explicit_map:
                order_type_value = explicit_map[t]
            elif t == "BUY":
                order_type_value = mt5.ORDER_TYPE_BUY_LIMIT if norm_price < ask else mt5.ORDER_TYPE_BUY_STOP
            elif t == "SELL":
                order_type_value = mt5.ORDER_TYPE_SELL_LIMIT if norm_price > bid else mt5.ORDER_TYPE_SELL_STOP
            else:
                return {
                    "error": (
                        f"Unsupported order_type '{order_type}'. "
                        "Use a market, limit, stop, or stop-limit order type."
                    )
                }

            pending_level_error = validation._validate_pending_order_levels(
                symbol_info=symbol_info,
                tick=live_tick,
                order_type_value=order_type_value,
                price=float(norm_price),
                stop_loss=None if norm_sl is None else float(norm_sl),
                take_profit=None if norm_tp is None else float(norm_tp),
                mt5=mt5,
                stop_limit_price=(
                    None if norm_stop_limit is None else float(norm_stop_limit)
                ),
            )
            if pending_level_error is not None:
                return pending_level_error
            guardrail_block = _evaluate_live_trade_guardrails(
                mt5,
                symbol=symbol,
                volume=volume_validated,
                stop_loss=None if norm_sl is None else float(norm_sl),
                deviation=deviation_validated,
                side="BUY" if "BUY" in str(t) else "SELL",
                entry_price=float(
                    norm_stop_limit if norm_stop_limit is not None else norm_price
                ),
                symbol_info=symbol_info,
            )
            if guardrail_block is not None:
                return guardrail_block

            request_comment, comment_sanitization, comment_truncation = _build_order_comment_payload(
                comment,
                default="mtdata pending order",
            )
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume_validated,
                "type": order_type_value,
                "price": norm_price,
                "sl": 0.0 if norm_sl is None else float(norm_sl),
                "tp": 0.0 if norm_tp is None else float(norm_tp),
                "deviation": deviation_validated,
                "magic": magic if magic is not None else _configured_order_magic(),
                "comment": request_comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": validation._safe_int_attr(mt5, "ORDER_FILLING_RETURN", 2),
            }
            if is_stop_limit and norm_stop_limit is not None:
                request["stoplimit"] = float(norm_stop_limit)

            if expiration_specified:
                if normalized_expiration is None:
                    request["type_time"] = mt5.ORDER_TIME_GTC
                else:
                    request["type_time"] = mt5.ORDER_TIME_SPECIFIED
                    request["expiration"] = normalized_expiration

            send_tick = mt5.symbol_info_tick(symbol)
            if send_tick is None:
                return {"error": f"Failed to refresh current price for {symbol}"}
            send_level_error = validation._validate_pending_order_levels(
                symbol_info=symbol_info,
                tick=send_tick,
                order_type_value=order_type_value,
                price=float(norm_price),
                stop_loss=None if norm_sl is None else float(norm_sl),
                take_profit=None if norm_tp is None else float(norm_tp),
                mt5=mt5,
                stop_limit_price=(
                    None if norm_stop_limit is None else float(norm_stop_limit)
                ),
            )
            if send_level_error is not None:
                return send_level_error

            send_outcome, send_error = _submit_order_request(
                mt5,
                request,
                base_error="Failed to send pending order",
                invalid_comment_error=(
                    "Failed to send pending order: broker rejected the comment field. "
                    "Comments are sanitized to letters/numbers/spaces/_/./-; "
                    "try a simpler comment or omit --comment."
                ),
                symbol_info=symbol_info,
            )
            if send_error is not None:
                return send_error
            result = send_outcome["result"]
            fill_mode_attempts = send_outcome["fill_mode_attempts"]
            used_request = send_outcome["used_request"]

            retcode = _order_result_value(result, "retcode")
            submission_status = validation._trade_execution_status(mt5, retcode)
            pending_created = submission_status in {"complete", "queued"}
            deal = _order_result_value(result, "deal")
            raw_price = _order_result_value(result, "price")
            raw_bid = _order_result_value(result, "bid")
            raw_ask = _order_result_value(result, "ask")
            unfilled = pending_created and deal in (0, 0.0, None)
            out: Dict[str, Any] = {
                "success": pending_created,
                "execution_status": "accepted" if pending_created else submission_status,
                "submission_status": submission_status,
                "fill_status": "not_filled" if pending_created else None,
                "order_status": "working" if pending_created else None,
                "retcode": retcode,
                "retcode_name": mt5.retcode_name(retcode),
                "deal": deal,
                "order": _order_result_value(result, "order"),
                "volume": _order_result_value(result, "volume"),
                "price": None if unfilled and raw_price in (0, 0.0) else raw_price,
                "bid": None if unfilled and raw_bid in (0, 0.0) else raw_bid,
                "ask": None if unfilled and raw_ask in (0, 0.0) else raw_ask,
                "requested_price": float(norm_price),
                "requested_sl": float(norm_sl) if norm_sl is not None else None,
                "requested_tp": float(norm_tp) if norm_tp is not None else None,
                "comment": _order_result_value(result, "comment"),
                "mt5_request_id": _order_result_value(result, "request_id"),
                "type_filling_used": used_request.get("type_filling"),
            }
            if norm_stop_limit is not None:
                out["requested_stop_limit_price"] = float(norm_stop_limit)
            if expiration_specified:
                out["requested_expiration"] = normalized_expiration
            warnings_out: List[str] = []
            _attach_comment_response_metadata(
                out,
                warnings_out,
                comment_sanitization=comment_sanitization,
                comment_truncation=comment_truncation,
                applied_comment=request_comment,
                requested_comment=comment,
            )
            if fill_mode_attempts:
                out["fill_mode_attempts"] = fill_mode_attempts
            if warnings_out:
                out["warnings"] = warnings_out
            return out

        except time.PendingExpirationValidationError as e:
            return {
                "error": str(e),
                "error_code": e.error_code,
                "expiration_context": dict(e.context),
            }
        except Exception as e:
            return {"error": str(e)}

    with _TRADE_DECISION_LOCK:
        return _place_pending_order()
