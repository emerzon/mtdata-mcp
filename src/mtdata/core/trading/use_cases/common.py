"""Shared trade-use-case helpers for correlation, idempotency, and payloads."""

from __future__ import annotations

import copy
import json
import logging
import math
from typing import Any, Dict, List, Optional

from mtdata.bootstrap.settings import trade_guardrails_config
from mtdata.core.error_envelope import normalize_error_payload
from mtdata.core.execution_logging import infer_result_success
from mtdata.core.output_contract import resolve_output_contract
from mtdata.core.trading import safety, validation
from mtdata.core.trading.idempotency import (
    IdempotencyStore,
    SQLiteIdempotencyStore,
    create_default_idempotency_store,
)
from mtdata.core.trading.requests import TradePlaceRequest
from mtdata.utils.coercion import round_finite
from mtdata.utils.mt5 import mt5_adapter

logger = logging.getLogger("mtdata.core.trading.use_cases")
TradeIdempotencyStore = IdempotencyStore | SQLiteIdempotencyStore
_TRADE_IDEMPOTENCY_STORE = create_default_idempotency_store()


def _invalid_order_type_payload(message: str) -> Dict[str, Any]:
    return {
        "error": message,
        "error_code": "invalid_order_type",
        "valid_values": {"order_type": list(validation._SUPPORTED_ORDER_TYPE_ORDER)},
        "remediation": "Choose a market side or an explicit pending-order type.",
        "example": "mtdata-cli trade_place EURUSD --order-type BUY --volume 0.01",
    }


def _dry_run_blocker_error(blockers: Any) -> str:
    blocker_list = [str(item).strip() for item in list(blockers or []) if str(item).strip()]
    blocker_set = set(blocker_list)
    if {"missing_stop_loss", "missing_take_profit"}.issubset(blocker_set):
        return (
            "stop_loss and take_profit are required when require_sl_tp=true. "
            "Provide both levels or explicitly set require_sl_tp=false."
        )
    if "missing_stop_loss" in blocker_set:
        return "stop_loss is required when require_sl_tp=true."
    if "missing_take_profit" in blocker_set:
        return "take_profit is required when require_sl_tp=true."
    if "margin_insufficient" in blocker_set:
        return "Estimated free margin is insufficient for this order."
    if "quote_not_live_ready" in blocker_set:
        return "The current quote is not usable for live submission; refresh it and retry."
    if blocker_list:
        return "Dry-run preview blocked by: " + ", ".join(blocker_list) + "."
    return "Dry-run preview is not eligible for live submission."


def _invalid_pending_expiration_payload(
    exc: Exception,
    *,
    dry_run: bool,
) -> Dict[str, Any]:
    error_code = str(
        getattr(exc, "error_code", "invalid_pending_expiration")
    )
    payload: Dict[str, Any] = {
        "success": False,
        "error": str(exc),
        "error_code": error_code,
    }
    context = getattr(exc, "context", None)
    if isinstance(context, dict):
        payload["expiration_context"] = dict(context)
    if dry_run:
        payload.update(
            {
                "dry_run": True,
                "no_action": True,
                "would_send_order": False,
                "preview_ok": False,
                "validation_passed": False,
                "blockers": [error_code],
                "validation": {
                    "local_requirements_passed": False,
                    "live_submission_eligible": False,
                    "blockers": [error_code],
                    "broker_validation_performed": False,
                },
            }
        )
    return payload


_TRADE_PLACE_PREVIEW_KEYS = (
    "success",
    "status",
    "error",
    "error_code",
    "remediation",
    "related_tools",
    "dry_run",
    "no_action",
    "no_action_reason",
    "would_send_order",
    "symbol",
    "order_type",
    "pending",
    "order_type_category",
    "action",
    "volume",
    "bid",
    "ask",
    "spread_points",
    "spread_pips",
    "spread_pct",
    "estimated_fill_price",
    "entry_price",
    "trigger_price",
    "stop_limit_price",
    "margin_required",
    "margin_required_when_filled",
    "margin_free",
    "margin_sufficient",
    "margin_action",
    "margin_estimate_basis",
    "account_state",
    "account_blockers",
    "sl_distance_points",
    "sl_distance_pips",
    "sl_distance_pct",
    "tp_distance_points",
    "tp_distance_pips",
    "tp_distance_pct",
    "candidate_risk",
    "min_distance_points",
    "sl_tp_valid",
    "sl_tp_error",
    "validation_error",
    "validation_code",
    "blockers",
    "preview_error",
    "preview_error_code",
    "units",
    "message",
    "dry_run_note",
    "preview_ok",
    "validation_passed",
    "validation",
    "validation_scope",
    "preview_checks_performed",
    "checks_not_performed",
    "broker_validation_not_performed",
    "warnings",
    "require_sl_tp",
    "auto_close_on_sl_tp_fail",
    "protection_status",
    "guardrails_enabled",
    "guardrails_preview",
    "magic",
    "comment",
    "requested_comment",
    "applied_comment",
    "comment_max_length",
    "comment_changed",
    "comment_sanitization",
    "comment_truncation",
    "requested_price",
    "requested_stop_limit_price",
    "stop_loss",
    "take_profit",
    "expiration",
    "expiration_policy",
    "expiration_explicit",
    "expiration_normalized",
    "expiration_resolved_utc",
    "expiration_context",
    "quote_context",
)


def _linearized_account_currency_notional(
    *,
    volume: float,
    price: float,
    symbol_info: Any,
) -> Optional[float]:
    """Approximate account-currency exposure from broker tick economics."""
    tick_size = validation._safe_float_attr(symbol_info, "trade_tick_size", 0.0)
    tick_values = [
        validation._safe_float_attr(symbol_info, "trade_tick_value", 0.0),
        validation._safe_float_attr(symbol_info, "trade_tick_value_profit", 0.0),
        validation._safe_float_attr(symbol_info, "trade_tick_value_loss", 0.0),
    ]
    tick_value = next(
        (value for value in tick_values if math.isfinite(value) and value > 0.0),
        0.0,
    )
    if (
        not math.isfinite(volume)
        or not math.isfinite(price)
        or not math.isfinite(tick_size)
        or volume < 0.0
        or price < 0.0
        or tick_size <= 0.0
        or tick_value <= 0.0
    ):
        return None
    return abs(float(volume)) * float(price) * tick_value / tick_size
_TRADE_PLACE_BASIC_KEYS = _TRADE_PLACE_PREVIEW_KEYS + (
    "no_action",
    "would_send_order",
    "dry_run_simulated",
    "validation_passed",
    "actionability",
    "actionability_reason",
    "validation",
    "preview_checks_performed",
    "broker_validation_not_performed",
    "preview_scope_summary",
    "checks_not_performed",
    "validation_not_performed",
    "warnings",
)


def _human_join(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _round_optional_number(value: Any, digits: int) -> Optional[float]:
    return round_finite(value, digits, on_invalid="none")


def _coerce_warning_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return [str(value)]


def _trade_row_to_dict(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "_asdict"):
        return dict(row._asdict())
    try:
        return dict(vars(row))
    except TypeError as exc:
        raise TypeError(f"Unsupported trade row type: {type(row).__name__}") from exc


def _trade_rows_to_dataframe(rows: Any, *, pd_module: Any) -> Any:
    row_dicts = [_trade_row_to_dict(row) for row in list(rows)]
    if not row_dicts:
        return pd_module.DataFrame()
    return pd_module.DataFrame.from_records(row_dicts)


def _resolve_trade_place_preview_detail(request: TradePlaceRequest) -> str:
    contract = resolve_output_contract(
        request,
        detail=request.detail,
        default_detail="compact",
    )
    if contract.shape_detail == "full":
        return "full"
    if contract.detail in {"standard", "summary"}:
        return "basic"
    return "preview"


def _shape_trade_place_preview(
    payload: Dict[str, Any], *, detail: str
) -> Dict[str, Any]:
    if detail == "full":
        return dict(payload)
    keys = _TRADE_PLACE_BASIC_KEYS if detail == "basic" else _TRADE_PLACE_PREVIEW_KEYS
    out = {key: payload[key] for key in keys if key in payload}
    if detail == "preview" and isinstance(out.get("guardrails_preview"), dict):
        preview = out["guardrails_preview"]
        out["guardrails_preview"] = {
            key: preview[key]
            for key in (
                "enabled",
                "blocked",
                "ignored_for_demo",
                "would_block_live",
                "live_projection",
                "checks_not_performed",
            )
            if key in preview
        }
    if detail == "preview" and isinstance(out.get("quote_context"), dict):
        quote = out["quote_context"]
        out["quote_context"] = {
            key: quote[key]
            for key in (
                "usable_for_live_trading",
                "freshness_state",
                "quote_time",
            )
            if key in quote
        }
    if detail == "preview" and isinstance(out.get("validation"), dict):
        validation_payload = out["validation"]
        out["validation"] = {
            key: validation_payload[key]
            for key in (
                "local_requirements_passed",
                "live_submission_eligible",
                "blockers",
            )
            if key in validation_payload
        }
    if detail == "preview" and isinstance(out.get("candidate_risk"), dict):
        risk = out["candidate_risk"]
        out["candidate_risk"] = {
            key: risk[key]
            for key in (
                "status",
                "risk_currency",
                "risk_pct_of_equity",
                "reward_currency",
                "reward_risk_ratio",
            )
            if key in risk
        }
    return out


def _attach_trade_attempt_markers(
    result: Dict[str, Any],
    *,
    dry_run: bool,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result
    out = dict(result)
    out["dry_run"] = bool(dry_run)
    if dry_run:
        out.setdefault("would_send_order", False)
        out.setdefault("order_sent", False)
        return out
    if out.get("order_sent") is None:
        if out.get("no_action") is True or out.get("success") is False:
            sent = False
        elif out.get("ambiguous") is True:
            sent = True
        else:
            sent = out.get("success") is True or any(
                out.get(key) is not None
                for key in ("retcode", "deal", "order", "position_ticket")
            )
        out["order_sent"] = bool(sent)
    out.setdefault("would_send_order", bool(out.get("order_sent")))
    return out


def _standardize_trade_operation_payload(
    result: Dict[str, Any],
    *,
    operation: str,
    default_error_code: str,
    request_id: Optional[str] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result
    if str(result.get("error") or "").strip():
        out = normalize_error_payload(
            result,
            default_code=default_error_code,
            request_id=request_id,
            operation=operation,
        )
        if dry_run is not None:
            return _attach_trade_attempt_markers(out, dry_run=dry_run)
        return out
    out = dict(result)
    out.setdefault("success", True)
    if dry_run is not None:
        return _attach_trade_attempt_markers(out, dry_run=dry_run)
    return out


def _attach_live_guardrail_status(
    result: Dict[str, Any],
    *,
    dry_run: bool,
) -> Dict[str, Any]:
    """Make the configured safety state explicit on submitted trade outcomes."""
    if dry_run or not isinstance(result, dict):
        return result
    if not (
        infer_result_success(result)
        or result.get("ambiguous") is True
        or result.get("error_code") == "order_send_ambiguous"
    ):
        return result
    out = dict(result)
    enabled = bool(trade_guardrails_config.is_enabled())
    out["guardrails_enabled"] = enabled
    if enabled:
        return out
    warning = (
        "Live trade submitted without configured trade guardrails. Set "
        "MTDATA_TRADE_GUARDRAILS_ENABLED=1 and configure symbol, volume, or "
        "risk limits to enable pre-trade protection."
    )
    warnings_out = (
        [str(item).strip() for item in out.get("warnings", []) if str(item).strip()]
        if isinstance(out.get("warnings"), list)
        else []
    )
    if warning not in warnings_out:
        warnings_out.append(warning)
    out["warnings"] = warnings_out
    return out


def _attach_trade_correlation(
    result: Dict[str, Any],
    *,
    correlation_id: Optional[str],
) -> Dict[str, Any]:
    """Attach the invocation ID and link an idempotent replay to its source."""
    correlation_value = str(correlation_id or "").strip()
    if not correlation_value or not isinstance(result, dict):
        return result
    out = dict(result)
    broker_request_id = out.get("mt5_request_id")
    legacy_request_id = out.get("request_id")
    if (
        broker_request_id is None
        and isinstance(legacy_request_id, int)
        and not isinstance(legacy_request_id, bool)
    ):
        out["mt5_request_id"] = legacy_request_id
    out["request_id"] = correlation_value
    out["correlation_id"] = correlation_value
    original_outcome = out.get("original_outcome")
    if isinstance(original_outcome, dict):
        original_correlation_id = str(
            original_outcome.get("correlation_id") or ""
        ).strip()
        if original_correlation_id:
            out["original_correlation_id"] = original_correlation_id
    return out


def _log_trade_correlation(
    *,
    operation: str,
    result: Dict[str, Any],
) -> None:
    """Log bounded identifiers that join an invocation to its MT5 result."""
    correlation_id = str(result.get("correlation_id") or "").strip()
    if not correlation_id:
        return
    fields = [f"correlation_id={correlation_id}"]
    original_correlation_id = str(
        result.get("original_correlation_id") or ""
    ).strip()
    if original_correlation_id:
        fields.append(f"original_correlation_id={original_correlation_id}")

    original_outcome = result.get("original_outcome")
    identifier_source = (
        original_outcome if isinstance(original_outcome, dict) else result
    )
    mt5_request_id = identifier_source.get("mt5_request_id")
    if isinstance(mt5_request_id, int) and not isinstance(mt5_request_id, bool):
        fields.append(f"mt5_request_id={mt5_request_id}")
    for key in ("order", "deal", "position_ticket", "ticket"):
        value = identifier_source.get(key)
        if value not in (None, "", 0, "0"):
            fields.append(f"{key}={value}")
    idempotency_key = result.get("idempotency_key")
    if idempotency_key not in (None, ""):
        fields.append(f"idempotency_key={idempotency_key}")
    if result.get("duplicate") is True:
        fields.append("duplicate=True")
    logger.info("event=trade_result operation=%s %s", operation, " ".join(fields))


def _sl_tp_result_details(result: Dict[str, Any]) -> tuple[bool, str]:
    sl_tp_result = result.get("sl_tp_result")
    if isinstance(sl_tp_result, dict):
        requested = sl_tp_result.get("requested")
        requested_bool = isinstance(requested, dict) and bool(requested)
        status = str(sl_tp_result.get("status") or "").lower()
        return requested_bool, status
    return False, ""


def _guardrail_order_side(order_type: Optional[str]) -> Optional[str]:
    side = safety._normalize_side(order_type)
    return side if side in {"BUY", "SELL"} else None


def _best_effort_trade_guardrail_account_info() -> Any:
    if not trade_guardrails_config.is_enabled():
        return None
    try:
        return mt5_adapter.account_info()
    except Exception:
        return None


def _best_effort_trade_guardrail_positions() -> Optional[List[Any]]:
    if not trade_guardrails_config.is_enabled():
        return []
    try:
        positions = mt5_adapter.positions_get()
    except Exception:
        return None
    return None if positions is None else list(positions)


def _best_effort_trade_guardrail_pending_orders() -> Optional[List[Any]]:
    if not trade_guardrails_config.is_enabled():
        return []
    try:
        pending_orders = mt5_adapter.orders_get()
    except Exception:
        return None
    return None if pending_orders is None else list(pending_orders)


def _normalize_idempotency_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip()
    return key or None


def _annotate_idempotency_scope(
    result: Dict[str, Any],
    key: Optional[str],
    store: Optional[TradeIdempotencyStore],
) -> Dict[str, Any]:
    if key is not None:
        result.setdefault("idempotency_key", key)
        result.setdefault("idempotency_scope", getattr(store, "scope", "unknown"))
        if result.get("dry_run") or result.get("no_action"):
            result["idempotency_durable"] = False
            result["idempotency_applies"] = "live_only"
        else:
            result.setdefault(
                "idempotency_durable", bool(getattr(store, "durable", False))
            )
    return result


def _build_trade_request_signature(request: Any) -> Optional[str]:
    if request is None:
        return None
    try:
        payload = request.model_dump(mode="json")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    payload.pop("idempotency_key", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _idempotency_duplicate_response(
    *,
    key: str,
    original_outcome: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "success": infer_result_success(original_outcome),
        "duplicate": True,
        "idempotency_key": key,
        "message": "Duplicate request suppressed by idempotency key.",
        "original_outcome": original_outcome,
    }


def _should_persist_idempotency_outcome(result: Any) -> bool:
    """Return True when *result* is safe to cache for idempotent retries.

    Transient preflight failures must not stick for the TTL. Ambiguous live
    submissions are deliberately retained so the same key cannot submit a
    second order while the broker outcome is unknown.
    """
    if not isinstance(result, dict):
        return False
    if result.get("dry_run") or result.get("no_action"):
        return False
    if result.get("duplicate"):
        return True
    if result.get("ambiguous") or result.get("error_code") == "order_send_ambiguous":
        return True
    if infer_result_success(result):
        return True
    if result.get("error_code") == "ticket_not_found":
        # The ticket is request context, not evidence of a broker-side effect.
        # Keep the reservation retryable in case terminal state was stale.
        return False
    for count_key in ("closed_count", "cancelled_count"):
        try:
            if int(result.get(count_key) or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
    nested_results = result.get("results")
    if isinstance(nested_results, list):
        for nested in nested_results:
            if not isinstance(nested, dict):
                continue
            if (
                nested.get("ambiguous") is True
                or nested.get("error_code") == "order_send_ambiguous"
                or infer_result_success(nested)
            ):
                return True
    for key in (
        "deal",
        "order",
        "position_ticket",
        "ticket",
        "order_ticket",
        "deal_ticket",
    ):
        value = result.get(key)
        if value in (None, "", 0, "0"):
            continue
        try:
            if int(value) != 0:
                return True
        except (TypeError, ValueError):
            return True
    # Nested auto-close after a partial fill is also a durable side effect.
    if isinstance(result.get("auto_close_result"), dict):
        nested = result["auto_close_result"]
        for key in ("deal", "order", "ticket"):
            if nested.get(key) not in (None, "", 0, "0"):
                return True
    return False


def _record_or_release_idempotency(
    store: Optional[TradeIdempotencyStore],
    key: Optional[str],
    result: Any,
    *,
    request_signature: Optional[str],
) -> bool:
    """Persist or release an idempotency reservation. Returns True if handled."""
    if store is None or key is None:
        return False
    if _should_persist_idempotency_outcome(result):
        store.record(
            key,
            copy.deepcopy(result) if isinstance(result, dict) else result,
            request_signature=request_signature,
        )
    else:
        store.release(key, request_signature=request_signature)
    return True


def _begin_trade_idempotency(
    *,
    idempotency_store: Optional[TradeIdempotencyStore],
    key: Optional[str],
    request_signature: Optional[str],
) -> tuple[Optional[Dict[str, Any]], bool]:
    if idempotency_store is None or key is None:
        return None, False
    duplicate = idempotency_store.reserve(
        key,
        request_signature=request_signature,
    )
    if duplicate is None:
        return None, True
    stored_signature = duplicate.get("request_signature")
    if (
        stored_signature is not None
        and request_signature is not None
        and stored_signature != request_signature
    ):
        return {
            "error": (
                "Idempotency key was already used for a different trade request. "
                "Use a new idempotency_key when changing parameters."
            ),
            "idempotency_key": key,
            "idempotency_conflict": True,
        }, False
    if duplicate.get("in_progress"):
        return {
            "error": (
                "An earlier request with this idempotency key is still unresolved. "
                "The retry was suppressed to avoid a duplicate live trade."
            ),
            "error_code": "idempotency_request_in_progress",
            "idempotency_key": key,
            "idempotency_in_progress": True,
        }, False
    original_outcome = duplicate.get("original_outcome")
    if not isinstance(original_outcome, dict):
        return {
            "error": "Stored idempotency outcome is invalid; use a new idempotency_key.",
            "idempotency_key": key,
            "idempotency_conflict": True,
        }, False
    return _idempotency_duplicate_response(
        key=key,
        original_outcome=copy.deepcopy(original_outcome),
    ), False


def _compact_close_preview_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove nested quote diagnostics from compact close previews."""
    out = dict(payload)
    rows = out.get("matched_positions")
    if isinstance(rows, list):
        compact_rows: List[Any] = []
        allowed = {
            "ticket",
            "symbol",
            "side",
            "volume",
            "profit",
            "price_open",
            "price_current",
            "sl",
            "tp",
            "magic",
        }
        for row in rows:
            if not isinstance(row, dict):
                compact_rows.append(row)
                continue
            compact = {key: row[key] for key in allowed if key in row}
            context = row.get("quote_context")
            if isinstance(context, dict):
                compact["quote_usable"] = (
                    context.get("usable_for_live_trading") is True
                )
                if context.get("usable_for_live_trading") is not True:
                    compact["quote_readiness_reason"] = (
                        "quote_source_conflict"
                        if isinstance(context.get("quote_source_conflict"), dict)
                        else context.get("freshness_reason")
                        or context.get("spread_quality")
                        or "quote_not_live_ready"
                    )
            compact_rows.append(compact)
        out["matched_positions"] = compact_rows
    closed_positions = out.get("closed_positions")
    if isinstance(closed_positions, dict):
        out["closed_positions"] = _compact_close_preview_payload(closed_positions)
    return out


def _validate_trading_symbol(gateway: Any, symbol: Optional[str]) -> Optional[Dict[str, Any]]:
    symbol_value = str(symbol or "").strip()
    if not symbol_value:
        return None
    symbol_info = getattr(gateway, "symbol_info", None)
    if not callable(symbol_info):
        return None
    try:
        info = symbol_info(symbol_value)
        if info is None:
            symbol_select = getattr(gateway, "symbol_select", None)
            if callable(symbol_select) and symbol_select(symbol_value, True):
                info = symbol_info(symbol_value)
    except Exception:
        return None
    if info is not None:
        return None
    return {
        "success": False,
        "error": f"Symbol '{symbol_value}' was not found by MT5.",
        "error_code": "symbol_not_found",
        "symbol": symbol_value,
        "remediation": (
            "Use symbols_list to find the broker's exact symbol name and suffix."
        ),
        "related_tools": ["symbols_list"],
    }


def _epoch_series_to_utc_and_text(
    raw_series: Any,
    *,
    pd_module: Any,
    mt5_epoch_to_utc: Any,
    fmt_time: Any,
    require_positive: bool = False,
) -> tuple[Any, Any]:
    numeric = pd_module.to_numeric(raw_series, errors="coerce")
    utc_values: List[float] = []
    text_values: List[Optional[str]] = []
    for raw_value in numeric.tolist():
        if pd_module.isna(raw_value):
            utc_values.append(float("nan"))
            text_values.append(None)
            continue
        epoch_value = float(raw_value)
        if require_positive and epoch_value <= 0.0:
            utc_values.append(float("nan"))
            text_values.append(None)
            continue
        utc_value = float(mt5_epoch_to_utc(epoch_value))
        utc_values.append(utc_value)
        text_values.append(fmt_time(utc_value))
    return (
        pd_module.Series(utc_values, index=numeric.index),
        pd_module.Series(text_values, index=numeric.index),
    )
