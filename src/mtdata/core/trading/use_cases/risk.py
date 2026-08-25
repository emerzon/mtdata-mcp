"""Trade risk analysis, stress test, and VaR/CVaR use cases."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mtdata.bootstrap.settings import trade_guardrails_config
from mtdata.core.execution_logging import (
    infer_result_success,
    log_operation_finish,
    log_operation_start,
)
from mtdata.core.trading import validation
from mtdata.core.trading.common import account_context_id, build_trade_quote_context
from mtdata.core.trading.requests import (
    TradeRiskAnalyzeRequest,
    TradeStressTestRequest,
    TradeVarCvarRequest,
)
from mtdata.core.trading.safety import assess_margin_stress, resolve_volume_guardrail
from mtdata.core.trading.sizing import (
    _floor_volume_steps,
    _resolve_risk_tick_value,
    compute_kelly_sizing_context,
)
from mtdata.core.trading.use_cases.common import (
    _human_join,
    _linearized_account_currency_notional,
    _round_optional_number,
    _validate_trading_symbol,
    logger,
)
from mtdata.services.data_service.candles import _is_last_bar_forming
from mtdata.shared.constants import BROKER_VOLUME_UNIT, TIMEFRAME_MAP
from mtdata.shared.market_units import price_delta_ticks
from mtdata.shared.validators import invalid_timeframe_error
from mtdata.utils.barriers import normalize_trade_direction
from mtdata.utils.mt5 import MT5ConnectionError, _normalize_times_in_struct
from mtdata.utils.quote import resolve_quote_tick, tick_value


def _resolve_trade_risk_direction(
    *,
    direction: Any,
    entry: float,
    stop_loss: float,
    take_profit: float | None = None,
) -> tuple[str | None, str | None, str]:
    direction_text = str(direction).strip() if direction is not None else ""
    if direction_text:
        direction_norm, direction_error = normalize_trade_direction(direction_text)
        return direction_norm, direction_error, "explicit"
    if stop_loss < entry:
        return "long", None, "inferred_from_stop_loss"
    if stop_loss > entry:
        return "short", None, "inferred_from_stop_loss"
    if take_profit is not None:
        if take_profit > entry:
            return "long", None, "inferred_from_take_profit"
        if take_profit < entry:
            return "short", None, "inferred_from_take_profit"
    return (
        None,
        "Unable to infer trade direction when stop_loss equals entry "
        "and take_profit is missing or also equals entry. "
        "Provide direction='long' or direction='short'.",
        "unable_to_infer",
    )


def _build_position_sizing_error(
    *,
    code: str,
    reason: str,
    field: Optional[str] = None,
    entry: Optional[float] = None,
    constraint: Optional[str] = None,
    remediation: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    error: Dict[str, Any] = {
        "code": code,
        "reason": reason,
        "message": reason,
    }
    if field:
        error["field"] = field
    if entry is not None:
        error["entry"] = entry
    if constraint:
        error["constraint"] = constraint
    if remediation:
        error["remediation"] = remediation
    for key, value in (details or {}).items():
        if value is not None:
            error[key] = value
    return error


def _positive_trade_price(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(price) and price > 0.0:
        return price
    return None


def _resolve_live_trade_risk_entry(
    *,
    gateway: Any,
    symbol: str,
    direction: Any,
) -> tuple[float | None, str | None, Dict[str, Any]]:
    try:
        raw_tick = gateway.symbol_info_tick(symbol)
    except Exception:
        return None, None, {}
    if raw_tick is None:
        return None, None, {}
    tick, quote_source = resolve_quote_tick(
        gateway,
        symbol,
        raw_tick,
        now_epoch=time.time(),
    )
    if tick is None:
        return None, None, quote_source

    quote_context = build_trade_quote_context(
        symbol,
        tick,
        source_metadata=quote_source,
    )
    live_quote = quote_context.get("usable_for_live_trading") is True
    source_prefix = "live_tick" if live_quote else "last_available_tick"
    if not live_quote:
        quote_context["sizing_reference_only"] = True
        quote_context["sizing_warning"] = (
            "The last available non-live quote is retained as a geometry reference; "
            "refresh the quote before requesting a sizing recommendation."
        )

    bid = _positive_trade_price(tick_value(tick, "bid"))
    ask = _positive_trade_price(tick_value(tick, "ask"))
    if bid is not None:
        quote_context["bid"] = bid
    if ask is not None:
        quote_context["ask"] = ask
    direction_norm = None
    if direction is not None:
        direction_norm, direction_error = normalize_trade_direction(str(direction))
        if direction_error:
            direction_norm = None

    if direction_norm == "long":
        if ask is not None:
            return ask, f"{source_prefix}_ask", quote_context
        if live_quote:
            quote_context["required_quote_side"] = "ask"
            quote_context["quote_side_missing"] = True
            return None, None, quote_context
        if bid is not None:
            return bid, f"{source_prefix}_bid_fallback", quote_context
    elif direction_norm == "short":
        if bid is not None:
            return bid, f"{source_prefix}_bid", quote_context
        if live_quote:
            quote_context["required_quote_side"] = "bid"
            quote_context["quote_side_missing"] = True
            return None, None, quote_context
        if ask is not None:
            return ask, f"{source_prefix}_ask_fallback", quote_context

    if bid is not None and ask is not None:
        return (bid + ask) / 2.0, f"{source_prefix}_mid", quote_context
    if bid is not None:
        return bid, f"{source_prefix}_bid_only", quote_context
    if ask is not None:
        return ask, f"{source_prefix}_ask_only", quote_context
    return None, None, quote_context


def _validate_trade_risk_levels(
    *,
    direction: str,
    entry: float,
    stop_loss: float,
    take_profit: float | None,
) -> Dict[str, Any] | None:
    def _error(
        *,
        code: str,
        field: str,
        reason: str,
        constraint: str,
        value: float,
    ) -> Dict[str, Any]:
        return _build_position_sizing_error(
            code=code,
            field=field,
            reason=reason,
            entry=entry,
            constraint=constraint,
            details={field: value},
        )

    if direction == "long":
        if stop_loss >= entry:
            return _error(
                code="invalid_sl_for_direction",
                field="stop_loss",
                reason="For long trades, stop_loss must be below entry.",
                constraint="stop_loss < entry",
                value=stop_loss,
            )
        if take_profit is not None and take_profit <= entry:
            return _error(
                code="invalid_tp_for_direction",
                field="take_profit",
                reason="For long trades, take_profit must be above entry.",
                constraint="take_profit > entry",
                value=take_profit,
            )
        return None
    if stop_loss <= entry:
        return _error(
            code="invalid_sl_for_direction",
            field="stop_loss",
            reason="For short trades, stop_loss must be above entry.",
            constraint="stop_loss > entry",
            value=stop_loss,
        )
    if take_profit is not None and take_profit >= entry:
        return _error(
            code="invalid_tp_for_direction",
            field="take_profit",
            reason="For short trades, take_profit must be below entry.",
            constraint="take_profit < entry",
            value=take_profit,
        )
    return None


def _build_trade_evaluation(
    *,
    symbol: Optional[str],
    direction: Any,
    entry: float,
    stop_loss: float,
    take_profit: Optional[float],
    sym_info: Any = None,
    entry_source: str | None = None,
) -> Dict[str, Any]:
    direction_norm, direction_error, direction_source = _resolve_trade_risk_direction(
        direction=direction,
        entry=float(entry),
        stop_loss=float(stop_loss),
        take_profit=float(take_profit) if take_profit is not None else None,
    )
    out: Dict[str, Any] = {
        "status": "invalid" if direction_error else "valid",
        "symbol": symbol,
        "direction": direction_norm,
        "direction_source": direction_source,
        "entry": float(entry),
        "sl": float(stop_loss),
        "tp": float(take_profit) if take_profit is not None else None,
    }
    if entry_source:
        out["entry_source"] = entry_source
    if direction_error or direction_norm is None:
        out["error"] = direction_error or "Unable to resolve trade direction."
        return out

    level_error = _validate_trade_risk_levels(
        direction=direction_norm,
        entry=float(entry),
        stop_loss=float(stop_loss),
        take_profit=float(take_profit) if take_profit is not None else None,
    )
    if level_error:
        out["status"] = "invalid"
        out["error"] = level_error
        return out

    sl_distance = abs(float(entry) - float(stop_loss))
    out["sl_distance_price"] = round(sl_distance, 10)
    if entry:
        out["sl_distance_pct"] = round((sl_distance / abs(float(entry))) * 100.0, 4)

    tick_size = validation._safe_float_attr(sym_info, "trade_tick_size")
    tick_value = validation._safe_float_attr(sym_info, "trade_tick_value")
    tick_value_loss = validation._safe_float_attr(sym_info, "trade_tick_value_loss")
    risk_tick_value = _resolve_risk_tick_value(
        tick_value=tick_value,
        tick_value_loss=tick_value_loss,
    )
    if math.isfinite(tick_size) and tick_size > 0:
        sl_distance_ticks = abs(
            price_delta_ticks(float(entry), float(stop_loss), tick_size) or 0
        )
        out["tick_size"] = tick_size
        out["sl_distance_ticks"] = round(sl_distance_ticks, 4)
        if math.isfinite(risk_tick_value) and risk_tick_value > 0:
            out["risk_tick_value"] = round(risk_tick_value, 8)
            out["risk_per_lot"] = round(sl_distance_ticks * risk_tick_value, 2)
    elif sym_info is not None:
        out["tick_metadata_warning"] = "Symbol tick size is unavailable or invalid."

    if take_profit is not None:
        tp_distance = abs(float(take_profit) - float(entry))
        out["tp_distance_price"] = round(tp_distance, 10)
        if entry:
            out["tp_distance_pct"] = round((tp_distance / abs(float(entry))) * 100.0, 4)
        if math.isfinite(tick_size) and tick_size > 0:
            out["tp_distance_ticks"] = round(tp_distance / tick_size, 4)
        if sl_distance > 0:
            out["reward_risk_ratio"] = round(tp_distance / sl_distance, 4)
    units = {
        key: value
        for key, value in {
            "sl_distance_price": "price",
            "sl_distance_pct": "percent",
            "sl_distance_ticks": "ticks",
            "risk_per_lot": "account_currency_per_lot",
            "tp_distance_price": "price",
            "tp_distance_pct": "percent",
            "tp_distance_ticks": "ticks",
            "reward_risk_ratio": "scalar",
        }.items()
        if key in out
    }
    if units:
        out["units"] = units
    return out


_COMPACT_POSITION_SIZING_FIELDS = (
    "status",
    "recommendation_status",
    "sizing_method",
    "suggested_volume",
    "unconstrained_volume",
    "guardrail_capped_volume",
    "guardrail_max_volume",
    "guardrail_rule",
    "requested_risk_currency",
    "requested_risk_pct",
    "risk_currency",
    "risk_pct",
    "risk_shortfall_currency",
    "risk_shortfall_pct",
    "risk_compliance",
    "volume_rounding",
    "min_viable_volume",
    "min_viable_risk_currency",
    "min_viable_risk_pct",
    "entry",
    "entry_source",
    "sl",
    "tp",
    "rr_ratio",
    "kelly",
)


def _compact_trade_risk_position_sizing(
    position_sizing: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(position_sizing, dict):
        return position_sizing
    if position_sizing.get("status") == "parameters_missing":
        return {
            key: position_sizing[key]
            for key in ("status", "message", "missing", "note", "related_tools")
            if key in position_sizing
        }
    compact = {
        key: position_sizing[key]
        for key in _COMPACT_POSITION_SIZING_FIELDS
        if key in position_sizing and position_sizing[key] is not None
    }
    if position_sizing.get("status") == "risk_too_small_for_min_lot":
        for key in (
            "volume_min",
            "volume_step",
            "volume_max",
            "strict_risk_hint",
        ):
            if key in position_sizing and position_sizing[key] is not None:
                compact[key] = position_sizing[key]
    return compact


def _compact_unconfigured_flat_trade_risk_payload(
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return a direct next-step response for a flat book with no sizing request."""
    position_sizing = result.get("position_sizing")
    if not isinstance(position_sizing, dict):
        return None
    if position_sizing.get("status") != "parameters_missing":
        return None
    if position_sizing.get("provided"):
        return None
    required = position_sizing.get("required_for_sizing")
    missing = position_sizing.get("missing")
    if not isinstance(required, list) or not isinstance(missing, list):
        return None
    if set(required) != set(missing):
        return None

    scope = result.get("scope")
    if not isinstance(scope, dict):
        return None
    scope_mode = str(scope.get("mode") or "")
    if scope_mode == "symbol" and scope.get("other_positions") != 0:
        return None
    if scope_mode not in {"symbol", "portfolio"}:
        return None

    risk = result.get("scoped_risk") or result.get("portfolio_risk")
    if not isinstance(risk, dict):
        return None
    if int(risk.get("positions_count") or 0) != 0:
        return None
    if int(risk.get("pending_orders_count") or 0) != 0:
        return None
    if not isinstance(result.get("positions"), list) or result.get("positions"):
        return None
    if not isinstance(result.get("pending_orders"), list) or result.get("pending_orders"):
        return None
    if result.get("risk_calculation_failures") or result.get("scope_warning"):
        return None

    return {
        key: result[key]
        for key in ("success", "account", "scope", "risk_visibility")
        if key in result
    } | {
        "book_state": "flat",
        "book_state_scope": scope_mode,
        "message": "No open positions or pending orders in the analyzed scope.",
        "position_sizing": _compact_trade_risk_position_sizing(position_sizing),
    }


def _strip_compact_trade_risk_login(payload: Dict[str, Any]) -> None:
    account = payload.get("account")
    if isinstance(account, dict) and "login" in account:
        compact_account = dict(account)
        compact_account.pop("login", None)
        payload["account"] = compact_account


def _shape_trade_risk_analyze_payload(
    result: Dict[str, Any],
    *,
    detail: str,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result
    if str(detail).strip().lower() != "compact":
        return result
    if result.get("error"):
        shaped = dict(result)
        _strip_compact_trade_risk_login(shaped)
        return shaped
    flat_unconfigured = _compact_unconfigured_flat_trade_risk_payload(result)
    if flat_unconfigured is not None:
        shaped = flat_unconfigured
    else:
        shaped = dict(result)
        position_sizing = shaped.get("position_sizing")
        if isinstance(position_sizing, dict):
            shaped["position_sizing"] = _compact_trade_risk_position_sizing(
                position_sizing
            )
    _strip_compact_trade_risk_login(shaped)
    for risk_key in ("scoped_risk", "portfolio_risk"):
        risk = shaped.get(risk_key)
        if (
            isinstance(risk, dict)
            and risk.get("positions_count") == 0
            and risk.get("pending_orders_count") == 0
            and not shaped.get("risk_calculation_failures")
        ):
            shaped.pop(risk_key, None)
    if isinstance(shaped.get("positions"), list) and not shaped["positions"]:
        shaped.pop("positions", None)
    if isinstance(shaped.get("pending_orders"), list) and not shaped["pending_orders"]:
        shaped.pop("pending_orders", None)
    return shaped


def _promote_position_sizing_block(
    result: Dict[str, Any],
    *,
    geometry_valid: bool,
    candidate_status: str = "blocked",
) -> Dict[str, Any]:
    sizing_error = result.get("position_sizing_error")
    if not isinstance(sizing_error, dict):
        return result
    error_code = str(sizing_error.get("code") or "position_sizing_blocked")
    error_message = str(
        sizing_error.get("reason")
        or sizing_error.get("message")
        or "Position sizing is blocked."
    )
    update: Dict[str, Any] = {
        "success": False,
        "candidate_valid": False,
        "candidate_status": candidate_status,
        "geometry_valid": geometry_valid,
        "sizing_eligible": False,
        "error_code": error_code,
        "error": error_message,
        "portfolio_snapshot_status": "available",
    }
    missing_fields = sizing_error.get("missing_fields")
    if missing_fields:
        update["missing_fields"] = missing_fields
    remediation = sizing_error.get("remediation")
    if remediation:
        update["remediation"] = remediation
    result.update(update)
    return result


def _apply_trade_candidate_outcome(result: Dict[str, Any]) -> Dict[str, Any]:
    """Promote proposed-trade validity to the operation-level contract."""
    evaluation = result.get("trade_evaluation")
    sizing_error = result.get("position_sizing_error")
    sizing_blocked = isinstance(sizing_error, dict)
    if not isinstance(evaluation, dict):
        if (
            sizing_blocked
            and str(sizing_error.get("code") or "") == "position_sizing_inputs_missing"
        ):
            return _promote_position_sizing_block(result, geometry_valid=False)
        return result

    candidate_status = str(evaluation.get("status") or "").strip().lower()
    geometry_valid = candidate_status == "valid"
    position_sizing = result.get("position_sizing")
    suggested_volume = None
    if isinstance(position_sizing, dict):
        suggested_volume = position_sizing.get("suggested_volume")
    try:
        suggested_volume_value = float(suggested_volume)
    except (TypeError, ValueError):
        suggested_volume_value = 0.0
    sizing_eligible = (
        geometry_valid
        and not sizing_blocked
        and isinstance(position_sizing, dict)
        and str(position_sizing.get("recommendation_status") or "").strip().lower()
        == "proposed"
        and math.isfinite(suggested_volume_value)
        and suggested_volume_value > 0.0
    )
    result["geometry_valid"] = geometry_valid
    result["sizing_eligible"] = sizing_eligible

    if candidate_status == "valid":
        quote_context = result.get("quote_context")
        if (
            isinstance(quote_context, dict)
            and quote_context.get("sizing_reference_only") is True
            and quote_context.get("usable_for_live_trading") is not True
        ):
            reason = str(
                quote_context.get("warning")
                or quote_context.get("sizing_warning")
                or "The resolved entry quote is not usable for live trading."
            )
            result.update(
                {
                    "success": False,
                    "candidate_valid": False,
                    "candidate_status": "blocked",
                    "geometry_valid": True,
                    "sizing_eligible": False,
                    "error_code": "quote_not_live_ready",
                    "error": reason,
                    "portfolio_snapshot_status": "available",
                    "position_sizing_error": _build_position_sizing_error(
                        code="quote_not_live_ready",
                        reason=reason,
                        entry=evaluation.get("entry"),
                        remediation=(
                            "Refresh the quote and rerun trade_risk_analyze, or provide "
                            "an explicit entry for research-only geometry."
                        ),
                    ),
                }
            )
            result.pop("position_sizing", None)
            return result
        if sizing_blocked:
            return _promote_position_sizing_block(result, geometry_valid=True)
        result["candidate_valid"] = True
        result["candidate_status"] = "valid"
        return result
    if candidate_status != "invalid":
        return result

    candidate_error = evaluation.get("error")
    sizing_error = result.get("position_sizing_error")
    candidate_error_codes = {
        "direction_inference_ambiguous",
        "direction_unable_to_infer",
        "invalid_direction",
        "invalid_sl_for_direction",
        "invalid_tp_for_direction",
        "non_positive_sl_distance",
    }
    if (
        not isinstance(candidate_error, dict)
        and isinstance(sizing_error, dict)
        and sizing_error.get("code") in candidate_error_codes
    ):
        candidate_error = sizing_error

    error_code = "invalid_trade_candidate"
    error_message = "The proposed trade is invalid."
    if isinstance(candidate_error, dict):
        error_code = str(candidate_error.get("code") or error_code)
        error_message = str(
            candidate_error.get("reason")
            or candidate_error.get("message")
            or error_message
        )
    elif candidate_error:
        error_message = str(candidate_error)
        direction_source = str(evaluation.get("direction_source") or "")
        if direction_source == "unable_to_infer":
            error_code = "direction_unable_to_infer"
        elif evaluation.get("direction") is None:
            error_code = "invalid_direction"

    result.update(
        {
            "success": False,
            "candidate_valid": False,
            "candidate_status": "invalid",
            "geometry_valid": False,
            "sizing_eligible": False,
            "error_code": error_code,
            "error": error_message,
            "portfolio_snapshot_status": "available",
        }
    )
    result.pop("position_sizing", None)
    return result


def _shape_trade_var_cvar_payload(
    result: Dict[str, Any],
    *,
    detail: str,
) -> Dict[str, Any]:
    if not isinstance(result, dict) or result.get("error"):
        return result
    if str(detail).strip().lower() != "compact":
        return result
    return {
        key: result[key]
        for key in (
            "success",
            "empty",
            "status",
            "message",
            "scope",
            "symbol",
            "portfolio_hint",
            "summary",
            "equity",
            "currency",
            "history_policy",
            "forming_candle_status",
            "forming_candle_status_by_symbol",
            "history_failures",
            "warnings",
            "mark_freshness_status",
            "mark_usability_status",
            "data_stale",
            "valuation_time",
            "valuation_basis",
            "valuation_warning",
            "entry_price_fallback_positions",
            "market_status",
            "market_status_reason",
            "marks_evaluated",
            "unusable_marks",
        )
        if key in result
    }


def _trade_risk_sizing_field_label(field_name: str) -> str:
    return {
        "desired_risk_pct": "the risk_pct field in --sizing",
        "entry": "--entry",
        "stop_loss": "--stop-loss",
        "kelly_win_rate": "the win_rate field in --sizing",
        "kelly_avg_win": "the avg_win field in --sizing",
        "kelly_avg_loss": "the avg_loss field in --sizing",
    }.get(field_name, field_name)


def _normalize_trade_risk_sizing_method(
    value: Any,
) -> tuple[Optional[str], Optional[str]]:
    method = str(value or "fixed_fraction").strip().lower().replace("-", "_")
    if method in {"fixed", "fixed_fraction"}:
        return "fixed_fraction", None
    if method == "kelly":
        return "kelly", None
    return None, "Invalid sizing_method. Valid options: fixed_fraction, kelly"


def _extract_trade_risk_kelly_inputs(
    request: TradeRiskAnalyzeRequest,
) -> tuple[Dict[str, Any], List[str], Optional[str]]:
    sizing = request.sizing
    inputs: Dict[str, Any] = {
        "win_rate": getattr(sizing, "win_rate", None),
        "avg_win": getattr(sizing, "avg_win", None),
        "avg_loss": getattr(sizing, "avg_loss", None),
    }
    missing = [
        field_name
        for field_name, value in (
            ("kelly_win_rate", inputs.get("win_rate")),
            ("kelly_avg_win", inputs.get("avg_win")),
            ("kelly_avg_loss", inputs.get("avg_loss")),
        )
        if value is None
    ]
    return inputs, missing, "sizing" if sizing is not None else None


def _normalize_var_cvar_method(method: Any) -> tuple[Optional[str], Optional[str]]:
    method_text = str(method or "historical").strip().lower()
    if method_text in {"historical", "hist"}:
        return "historical", None
    if method_text in {"gaussian", "normal", "parametric"}:
        return "parametric", None
    if method_text in {"cornish_fisher", "cornish-fisher", "cf"}:
        return "cornish_fisher", None
    if method_text in {"ewma", "ewma_historical"}:
        return "ewma", None
    return None, (
        "Invalid method. Valid options: historical, parametric, cornish_fisher, ewma"
    )


def _normalize_var_cvar_transform(
    transform: Any,
) -> tuple[Optional[str], Optional[str]]:
    transform_text = str(transform or "log_return").strip().lower()
    if transform_text in {"log_return", "log_returns", "log"}:
        return "log_return", None
    if transform_text in {
        "pct",
        "pct_return",
        "pct_returns",
        "percent",
        "percent_return",
        "percent_returns",
        "simple_return",
        "simple_returns",
    }:
        return "pct", None
    return None, "Invalid transform. Valid options: log_return, pct"


def _normalize_var_cvar_confidence(
    confidence: Any,
) -> tuple[Optional[float], Optional[str]]:
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        return None, "confidence must be numeric"
    if not math.isfinite(confidence_value):
        return None, "confidence must be finite"
    if confidence_value > 1.0:
        confidence_value /= 100.0
    if confidence_value <= 0.0 or confidence_value >= 1.0:
        return (
            None,
            "confidence must be between 0 and 1, or between 0 and 100 as a percentage",
        )
    return confidence_value, None


def _historical_var_cvar_tail(
    pnl_values: List[float], confidence: float
) -> tuple[float, float, float]:
    ordered = sorted(float(value) for value in pnl_values)
    if not ordered:
        return 0.0, 0.0, 0.0
    alpha = 1.0 - confidence
    index = max(0, min(len(ordered) - 1, int(math.floor(alpha * (len(ordered) - 1)))))
    threshold = float(ordered[index])
    tail_values = [float(value) for value in ordered[: index + 1]]
    tail_mean = float(sum(tail_values) / len(tail_values)) if tail_values else threshold
    var_value = max(0.0, -threshold)
    cvar_value = max(0.0, -tail_mean)
    return var_value, cvar_value, threshold


def _gaussian_var_cvar_tail(
    pnl_values: List[float], confidence: float
) -> tuple[float, float, float]:
    from scipy.stats import norm

    ordered = [float(value) for value in pnl_values]
    if not ordered:
        return 0.0, 0.0, 0.0
    mean_pnl = float(sum(ordered) / len(ordered))
    if len(ordered) == 1:
        threshold = mean_pnl
        var_value = max(0.0, -threshold)
        return var_value, var_value, threshold
    variance = sum((value - mean_pnl) ** 2 for value in ordered) / float(
        len(ordered) - 1
    )
    std_pnl = math.sqrt(max(0.0, variance))
    if std_pnl <= 0.0:
        threshold = mean_pnl
        var_value = max(0.0, -threshold)
        return var_value, var_value, threshold
    alpha = 1.0 - confidence
    z_score = float(norm.ppf(alpha))
    threshold = mean_pnl + (std_pnl * z_score)
    tail_mean = mean_pnl - (std_pnl * float(norm.pdf(z_score)) / alpha)
    var_value = max(0.0, -threshold)
    cvar_value = max(0.0, -tail_mean)
    return var_value, cvar_value, threshold


def _cornish_fisher_var_cvar_tail(
    pnl_values: List[float], confidence: float
) -> tuple[float, float, float]:
    import numpy as np
    from scipy.stats import norm

    ordered = np.asarray([float(value) for value in pnl_values], dtype=float)
    if ordered.size == 0:
        return 0.0, 0.0, 0.0
    if ordered.size < 3:
        return _gaussian_var_cvar_tail(pnl_values, confidence)

    mean_pnl = float(np.mean(ordered))
    centered = ordered - mean_pnl
    std_pnl = float(np.std(ordered, ddof=1))
    if not math.isfinite(std_pnl) or std_pnl <= 0.0:
        threshold = mean_pnl
        var_value = max(0.0, -threshold)
        return var_value, var_value, threshold

    z_score = float(norm.ppf(1.0 - confidence))
    standardized = centered / std_pnl
    skewness = float(np.mean(standardized ** 3))
    excess_kurtosis = float(np.mean(standardized ** 4) - 3.0)
    z_cf = (
        z_score
        + ((z_score**2 - 1.0) * skewness / 6.0)
        + ((z_score**3 - (3.0 * z_score)) * excess_kurtosis / 24.0)
        - (((2.0 * (z_score**3)) - (5.0 * z_score)) * (skewness**2) / 36.0)
    )
    threshold = mean_pnl + (std_pnl * z_cf)

    tail_values = ordered[ordered <= threshold]
    tail_mean = float(np.mean(tail_values)) if tail_values.size else float(threshold)
    var_value = max(0.0, -float(threshold))
    cvar_value = max(0.0, -tail_mean)
    return var_value, cvar_value, float(threshold)


def _ewma_var_cvar_tail(
    pnl_values: List[float], confidence: float, *, decay: float = 0.94
) -> tuple[float, float, float]:
    import numpy as np

    ordered = np.asarray([float(value) for value in pnl_values], dtype=float)
    if ordered.size == 0:
        return 0.0, 0.0, 0.0
    if ordered.size == 1:
        threshold = float(ordered[0])
        var_value = max(0.0, -threshold)
        return var_value, var_value, threshold

    lam = min(max(float(decay), 0.0), 0.999999)
    ages = np.arange(ordered.size - 1, -1, -1, dtype=float)
    weights = (1.0 - lam) * np.power(lam, ages)
    total_weight = float(np.sum(weights))
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        return _historical_var_cvar_tail(pnl_values, confidence)
    weights = weights / total_weight

    sort_idx = np.argsort(ordered)
    sorted_pnl = ordered[sort_idx]
    sorted_weights = weights[sort_idx]
    alpha = 1.0 - confidence
    cumulative = np.cumsum(sorted_weights)
    threshold_idx = int(np.searchsorted(cumulative, alpha, side="left"))
    threshold_idx = max(0, min(threshold_idx, sorted_pnl.size - 1))
    threshold = float(sorted_pnl[threshold_idx])
    tail_mask = sorted_pnl <= threshold
    tail_weights = sorted_weights[tail_mask]
    tail_total = float(np.sum(tail_weights))
    if tail_total <= 0.0:
        tail_mean = threshold
    else:
        tail_mean = float(np.dot(sorted_pnl[tail_mask], tail_weights) / tail_total)
    var_value = max(0.0, -threshold)
    cvar_value = max(0.0, -tail_mean)
    return var_value, cvar_value, threshold


def _calculate_var_cvar_from_pnl(
    pnl_values: List[float],
    *,
    confidence: float,
    method: str,
) -> tuple[float, float, float]:
    if method == "historical":
        return _historical_var_cvar_tail(pnl_values, confidence)
    if method in {"parametric", "gaussian"}:
        return _gaussian_var_cvar_tail(pnl_values, confidence)
    if method == "cornish_fisher":
        return _cornish_fisher_var_cvar_tail(pnl_values, confidence)
    if method == "ewma":
        return _ewma_var_cvar_tail(pnl_values, confidence)
    raise ValueError(f"Unsupported VaR/CVaR method: {method}")


def _extract_var_cvar_return_series(
    *,
    symbol: str,
    rates: Any,
    transform: str,
    pd_module: Any,
    np_module: Any,
) -> tuple[Any, Optional[str]]:
    frame = pd_module.DataFrame(rates)
    if frame.empty:
        return None, f"No candle history returned for {symbol}"
    if "time" not in frame.columns or "close" not in frame.columns:
        return None, f"Candle history for {symbol} is missing time/close columns"
    close = pd_module.to_numeric(frame["close"], errors="coerce")
    timestamps = pd_module.to_datetime(
        frame["time"], unit="s", utc=True, errors="coerce"
    )
    series = pd_module.Series(close.to_numpy(), index=timestamps, name=symbol)
    series = series[~series.index.isna()]
    series = series.replace([np_module.inf, -np_module.inf], np_module.nan).dropna()
    series = series[~series.index.duplicated(keep="last")]
    if len(series) < 2:
        return None, f"Not enough candle history for {symbol}"
    if transform == "log_return":
        returns = np_module.log(series / series.shift(1))
    else:
        returns = series.pct_change()
    returns = returns.replace([np_module.inf, -np_module.inf], np_module.nan).dropna()
    if returns.empty:
        return None, f"No usable returns produced for {symbol}"
    return returns, None


def _format_var_cvar_timestamp(value: Any) -> str:
    try:
        text = value.isoformat()
    except Exception:
        return str(value)
    return text.replace("+00:00", "Z")


def _format_var_cvar_observation_error(
    *,
    observation_name: str,
    available: int,
    required: int,
    lookback: int,
) -> str:
    message = (
        f"Not enough {observation_name} observations for VaR/CVaR calculation: "
        f"lookback={int(lookback)} yielded {int(available)}, need {int(required)}. "
        "Increase lookback"
    )
    if int(available) >= 2:
        return f"{message} or lower min_observations to <= {int(available)}."
    return f"{message}."


def run_trade_risk_analyze(  # noqa: C901
    request: TradeRiskAnalyzeRequest,
    *,
    gateway: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    log_operation_start(
        logger,
        operation="trade_risk_analyze",
        symbol=request.symbol,
        desired_risk_pct=request.desired_risk_pct,
    )

    def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
        result = _apply_trade_candidate_outcome(result)
        result = _shape_trade_risk_analyze_payload(
            result,
            detail=str(getattr(request, "detail", "compact")),
        )
        if not str(result.get("error") or "").strip():
            result.setdefault("as_of", observed_at)
        log_operation_finish(
            logger,
            operation="trade_risk_analyze",
            started_at=started_at,
            success=infer_result_success(result),
            symbol=request.symbol,
            desired_risk_pct=request.desired_risk_pct,
        )
        return result

    try:
        gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return _finish({"error": str(exc)})
    symbol_error = _validate_trading_symbol(gateway, request.symbol)
    if symbol_error is not None:
        return _finish(symbol_error)

    def _analyze_risk():  # noqa: C901
        try:
            entry_was_omitted = request.entry is None
            account = gateway.account_info()
            if account is None:
                return {"error": "Failed to get account info"}

            equity = validation._safe_float_attr(account, "equity", 0.0)
            margin_stress = assess_margin_stress(account)
            currency = getattr(account, "currency", None)
            try:
                positions = (
                    gateway.positions_get(symbol=request.symbol)
                    if request.symbol
                    else gateway.positions_get()
                )
            except Exception:
                positions = None
            if positions is None:
                return validation.snapshot_unavailable_error(
                    gateway,
                    snapshot="positions",
                    context="analyze open-position risk",
                )
            portfolio_positions_total: Optional[int] = None
            portfolio_snapshot_available = True
            if request.symbol:
                try:
                    portfolio_positions = gateway.positions_get()
                except Exception:
                    portfolio_positions = None
                if portfolio_positions is None:
                    portfolio_snapshot_available = False
                else:
                    portfolio_positions_total = len(list(portfolio_positions))

            position_risks: List[Dict[str, Any]] = []
            pending_order_risks: List[Dict[str, Any]] = []
            risk_calculation_failures: List[Dict[str, Any]] = []
            total_risk_currency = 0.0
            total_pending_risk_currency = 0.0
            open_risk_items_total = 0
            open_risk_items_quantified = 0
            pending_risk_items_total = 0
            pending_risk_items_quantified = 0
            positions_without_sl = 0
            positions_with_breached_stops = 0
            total_stop_overrun_currency = 0.0
            pending_orders_without_sl = 0
            total_notional_exposure = 0.0
            total_pending_notional_exposure = 0.0
            notional_items_total = 0
            notional_items_included = 0
            symbol_info_cache: Dict[str, Any] = {}

            for pos in positions:
                open_risk_items_total += 1
                try:
                    symbol_key = str(getattr(pos, "symbol", ""))
                    if symbol_key not in symbol_info_cache:
                        symbol_info_cache[symbol_key] = gateway.symbol_info(pos.symbol)
                    sym_info = symbol_info_cache[symbol_key]
                    if sym_info is None:
                        risk_calculation_failures.append(
                            {
                                "ticket": getattr(pos, "ticket", None),
                                "symbol": getattr(pos, "symbol", None),
                                "error": f"Failed to get symbol info for {getattr(pos, 'symbol', None)}",
                                "error_type": "SymbolInfoUnavailable",
                            }
                        )
                        continue

                    entry_price = float(pos.price_open)
                    mark_price = float(pos.price_current)
                    if not math.isfinite(mark_price) or mark_price <= 0.0:
                        raise ValueError(
                            "Current mark price is unavailable; remaining stop risk "
                            "cannot be calculated."
                        )
                    sl_price = float(pos.sl) if pos.sl and pos.sl > 0 else None
                    tp_price = float(pos.tp) if pos.tp and pos.tp > 0 else None
                    volume = float(pos.volume)

                    contract_size = float(sym_info.trade_contract_size)
                    tick_value = validation._safe_float_attr(
                        sym_info, "trade_tick_value"
                    )
                    tick_value_loss = validation._safe_float_attr(
                        sym_info, "trade_tick_value_loss"
                    )
                    tick_size = validation._safe_float_attr(sym_info, "trade_tick_size")
                    risk_tick_value = _resolve_risk_tick_value(
                        tick_value=tick_value,
                        tick_value_loss=tick_value_loss,
                    )
                    if not math.isfinite(tick_size) or tick_size <= 0:
                        tick_size = 0.0
                    tick_value_valid = (
                        math.isfinite(risk_tick_value) and risk_tick_value > 0
                    )
                    if not math.isfinite(contract_size) or contract_size <= 0:
                        contract_size = 1.0

                    contract_price_product = abs(volume) * contract_size * mark_price
                    notional_value = _linearized_account_currency_notional(
                        volume=volume,
                        price=mark_price,
                        symbol_info=sym_info,
                    )
                    notional_items_total += 1
                    if notional_value is not None:
                        total_notional_exposure += notional_value
                        notional_items_included += 1

                    risk_currency = None
                    stop_overrun_currency = None
                    risk_pct = None
                    reward_currency = None
                    rr_ratio = None
                    reward_status = "undefined"
                    risk_status = "undefined"
                    is_buy_position = (
                        validation._resolve_position_side(pos, gateway) or "SELL"
                    ) == "BUY"

                    if sl_price and tick_size > 0 and tick_value_valid:
                        risk_ticks = (
                            price_delta_ticks(mark_price, sl_price, tick_size)
                            if is_buy_position
                            else price_delta_ticks(sl_price, mark_price, tick_size)
                        )
                        stop_breached = risk_ticks is not None and risk_ticks < 0
                        if stop_breached:
                            positions_with_breached_stops += 1
                            stop_overrun_currency = (
                                abs(risk_ticks or 0)
                                * risk_tick_value
                                * abs(volume)
                            )
                            total_stop_overrun_currency += stop_overrun_currency
                            risk_status = "breached"
                        else:
                            risk_ticks = abs(risk_ticks or 0)
                            risk_currency = risk_ticks * risk_tick_value * abs(volume)
                            risk_pct = (
                                (risk_currency / equity) * 100.0
                                if equity > 0
                                else 0.0
                            )
                            total_risk_currency += risk_currency
                            risk_status = "defined"

                        if tp_price:
                            reward_ticks = (
                                price_delta_ticks(tp_price, mark_price, tick_size)
                                if is_buy_position
                                else price_delta_ticks(mark_price, tp_price, tick_size)
                            )
                            if reward_ticks is not None and reward_ticks > 0:
                                reward_currency = (
                                    reward_ticks * tick_value * abs(volume)
                                )
                                reward_status = "defined"
                                if risk_currency is not None and risk_currency > 0:
                                    rr_ratio = reward_currency / risk_currency
                            else:
                                reward_status = "invalid"
                    elif sl_price:
                        risk_status = "undefined"
                        risk_calculation_failures.append(
                            {
                                "ticket": getattr(pos, "ticket", None),
                                "symbol": getattr(pos, "symbol", None),
                                "error": "Stop-loss is set but symbol tick metadata is invalid.",
                                "error_type": "InvalidTickConfiguration",
                            }
                        )
                    else:
                        positions_without_sl += 1
                        risk_status = "unlimited"

                    if risk_status == "defined":
                        open_risk_items_quantified += 1

                    position_risks.append(
                        {
                            "ticket": pos.ticket,
                            "symbol": pos.symbol,
                            "type": "BUY" if is_buy_position else "SELL",
                            "volume": volume,
                            "volume_unit": "broker_lot",
                            "contract_size": round(contract_size, 6),
                            "lot_definition": (
                                "1 broker lot equals contract_size contract units."
                            ),
                            "entry": entry_price,
                            "current_mark": mark_price,
                            "risk_reference_price": mark_price,
                            "risk_reference_basis": "current_mark",
                            "sl": sl_price,
                            "tp": tp_price,
                            "risk_currency": _round_optional_number(
                                risk_currency, 2
                            ),
                            "risk_pct": _round_optional_number(risk_pct, 2),
                            "risk_status": risk_status,
                            "stop_overrun_currency": _round_optional_number(
                                stop_overrun_currency, 2
                            ),
                            "notional_value": _round_optional_number(
                                notional_value, 2
                            ),
                            "contract_price_product": round(
                                contract_price_product, 2
                            ),
                            "reward_currency": _round_optional_number(
                                reward_currency, 2
                            ),
                            "reward_status": reward_status,
                            "rr_ratio": _round_optional_number(rr_ratio, 2),
                        }
                    )
                except Exception as exc:
                    risk_calculation_failures.append(
                        {
                            "ticket": getattr(pos, "ticket", None),
                            "symbol": getattr(pos, "symbol", None),
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        }
                    )
                    continue

            quantified_open_position_risk_currency = total_risk_currency
            open_position_risk_complete = (
                open_risk_items_quantified == open_risk_items_total
            )
            open_position_risk_currency = (
                quantified_open_position_risk_currency
                if open_position_risk_complete
                else None
            )
            open_position_notional_exposure = total_notional_exposure

            if getattr(request, "include_pending", True):
                pending_orders = (
                    gateway.orders_get(symbol=request.symbol)
                    if request.symbol
                    else gateway.orders_get()
                )
                if pending_orders is None:
                    return validation.snapshot_unavailable_error(
                        gateway,
                        snapshot="orders",
                        context="include pending-order risk",
                    )
                pending_buy_types = {
                    validation._safe_int_attr(gateway, "ORDER_TYPE_BUY_LIMIT", 2),
                    validation._safe_int_attr(gateway, "ORDER_TYPE_BUY_STOP", 4),
                    validation._safe_int_attr(gateway, "ORDER_TYPE_BUY_STOP_LIMIT", 6),
                }
                pending_sell_types = {
                    validation._safe_int_attr(gateway, "ORDER_TYPE_SELL_LIMIT", 3),
                    validation._safe_int_attr(gateway, "ORDER_TYPE_SELL_STOP", 5),
                    validation._safe_int_attr(gateway, "ORDER_TYPE_SELL_STOP_LIMIT", 7),
                }
                for order in pending_orders:
                    pending_risk_items_total += 1
                    try:
                        symbol_key = str(getattr(order, "symbol", ""))
                        if symbol_key not in symbol_info_cache:
                            symbol_info_cache[symbol_key] = gateway.symbol_info(symbol_key)
                        sym_info = symbol_info_cache[symbol_key]
                        if sym_info is None:
                            risk_calculation_failures.append(
                                {
                                    "scope": "pending_order",
                                    "ticket": getattr(order, "ticket", None),
                                    "symbol": getattr(order, "symbol", None),
                                    "error": f"Failed to get symbol info for {getattr(order, 'symbol', None)}",
                                    "error_type": "SymbolInfoUnavailable",
                                }
                            )
                            continue

                        entry_price = float(getattr(order, "price_open", 0.0) or 0.0)
                        sl_raw = getattr(order, "sl", None)
                        tp_raw = getattr(order, "tp", None)
                        sl_price = float(sl_raw) if sl_raw and float(sl_raw) > 0 else None
                        tp_price = float(tp_raw) if tp_raw and float(tp_raw) > 0 else None
                        volume = float(
                            getattr(
                                order,
                                "volume_current",
                                getattr(order, "volume_initial", getattr(order, "volume", 0.0)),
                            )
                            or 0.0
                        )

                        contract_size = float(sym_info.trade_contract_size)
                        tick_value = validation._safe_float_attr(sym_info, "trade_tick_value")
                        tick_value_loss = validation._safe_float_attr(sym_info, "trade_tick_value_loss")
                        tick_size = validation._safe_float_attr(sym_info, "trade_tick_size")
                        risk_tick_value = _resolve_risk_tick_value(
                            tick_value=tick_value,
                            tick_value_loss=tick_value_loss,
                        )
                        if not math.isfinite(tick_size) or tick_size <= 0:
                            tick_size = 0.0
                        tick_value_valid = math.isfinite(risk_tick_value) and risk_tick_value > 0
                        if not math.isfinite(contract_size) or contract_size <= 0:
                            contract_size = 1.0

                        contract_price_product = abs(volume) * contract_size * entry_price
                        notional_value = _linearized_account_currency_notional(
                            volume=volume,
                            price=entry_price,
                            symbol_info=sym_info,
                        )
                        notional_items_total += 1
                        if notional_value is not None:
                            total_pending_notional_exposure += notional_value
                            notional_items_included += 1

                        order_type = validation._safe_int_attr(order, "type", -1)
                        is_buy_order = int(order_type) in pending_buy_types
                        is_sell_order = int(order_type) in pending_sell_types
                        direction_label = "BUY" if is_buy_order else "SELL" if is_sell_order else "UNKNOWN"

                        risk_currency = None
                        risk_pct = None
                        reward_currency = None
                        rr_ratio = None
                        reward_status = "undefined"
                        risk_status = "undefined"
                        if entry_price > 0 and sl_price and tick_size > 0 and tick_value_valid and direction_label != "UNKNOWN":
                            risk_ticks = (
                                price_delta_ticks(entry_price, sl_price, tick_size)
                                if is_buy_order
                                else price_delta_ticks(sl_price, entry_price, tick_size)
                            )
                            risk_currency = abs((risk_ticks or 0) * risk_tick_value * volume)
                            risk_pct = (risk_currency / equity) * 100.0 if equity > 0 else 0.0
                            total_pending_risk_currency += risk_currency
                            risk_status = "defined"
                            if tp_price:
                                reward_ticks = (
                                    price_delta_ticks(tp_price, entry_price, tick_size)
                                    if is_buy_order
                                    else price_delta_ticks(entry_price, tp_price, tick_size)
                                )
                                if reward_ticks is not None and reward_ticks > 0:
                                    reward_currency = (
                                        reward_ticks * tick_value * abs(volume)
                                    )
                                    reward_status = "defined"
                                    if risk_currency > 0:
                                        rr_ratio = reward_currency / risk_currency
                                else:
                                    reward_status = "invalid"
                        elif sl_price:
                            risk_calculation_failures.append(
                                {
                                    "scope": "pending_order",
                                    "ticket": getattr(order, "ticket", None),
                                    "symbol": getattr(order, "symbol", None),
                                    "error": "Pending order has stop-loss but entry, direction, or symbol tick metadata is invalid.",
                                    "error_type": "InvalidPendingRiskMetadata",
                                }
                            )
                        else:
                            pending_orders_without_sl += 1
                            risk_status = "unlimited"

                        if risk_status == "defined":
                            pending_risk_items_quantified += 1

                        pending_order_risks.append(
                            {
                                "ticket": getattr(order, "ticket", None),
                                "symbol": getattr(order, "symbol", None),
                                "type": direction_label,
                                "volume": volume,
                                "volume_unit": "broker_lot",
                                "contract_size": round(contract_size, 6),
                                "lot_definition": (
                                    "1 broker lot equals contract_size contract units."
                                ),
                                "entry": entry_price,
                                "sl": sl_price,
                                "tp": tp_price,
                                "risk_currency": _round_optional_number(risk_currency, 2),
                                "risk_pct": _round_optional_number(risk_pct, 2),
                                "risk_status": risk_status,
                                "notional_value": _round_optional_number(
                                    notional_value, 2
                                ),
                                "contract_price_product": round(
                                    contract_price_product, 2
                                ),
                                "reward_currency": _round_optional_number(reward_currency, 2),
                                "reward_status": reward_status,
                                "rr_ratio": _round_optional_number(rr_ratio, 2),
                            }
                        )
                    except Exception as exc:
                        risk_calculation_failures.append(
                            {
                                "scope": "pending_order",
                                "ticket": getattr(order, "ticket", None),
                                "symbol": getattr(order, "symbol", None),
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                            }
                        )
                        continue

            quantified_risk_currency = (
                total_risk_currency + total_pending_risk_currency
            )
            risk_items_total = open_risk_items_total + pending_risk_items_total
            risk_items_quantified = (
                open_risk_items_quantified + pending_risk_items_quantified
            )
            risk_total_complete = risk_items_quantified == risk_items_total
            total_risk_currency = (
                quantified_risk_currency if risk_total_complete else None
            )
            pending_risk_complete = (
                pending_risk_items_quantified == pending_risk_items_total
            )
            contingent_pending_risk_currency = (
                total_pending_risk_currency if pending_risk_complete else None
            )
            total_notional_exposure += total_pending_notional_exposure
            quantified_risk_pct = (
                (quantified_risk_currency / equity) * 100.0 if equity > 0 else 0.0
            )
            total_risk_pct = (
                quantified_risk_pct if risk_total_complete else None
            )
            notional_exposure_pct = (
                (total_notional_exposure / equity) * 100.0 if equity > 0 else 0.0
            )

            if (
                positions_without_sl > 0
                or pending_orders_without_sl > 0
                or positions_with_breached_stops > 0
            ):
                stop_risk_level = "unlimited"
            else:
                stop_risk_level = (
                    "high"
                    if quantified_risk_pct > 10
                    else "moderate"
                    if quantified_risk_pct > 5
                    else "low"
                )
            margin_risk_level = (
                "high" if margin_stress["status"] == "critical"
                else "moderate" if margin_stress["status"] == "stressed"
                else "low" if margin_stress["status"] == "healthy" else "unknown"
            )
            notional_risk_level = (
                "high" if notional_exposure_pct >= 400.0
                else "moderate" if notional_exposure_pct >= 200.0 else "low"
            )
            risk_rank = {
                "unknown": -1,
                "low": 0,
                "moderate": 1,
                "high": 2,
                "unlimited": 3,
            }
            quantified_risk_level = max(
                (stop_risk_level, margin_risk_level, notional_risk_level),
                key=lambda value: risk_rank[value],
            )

            if (
                positions_without_sl > 0
                or pending_orders_without_sl > 0
                or positions_with_breached_stops > 0
            ):
                overall_risk_status = "unlimited"
            elif risk_calculation_failures:
                overall_risk_status = "incomplete"
            else:
                overall_risk_status = "defined"

            account_payload: Dict[str, Any] = {
                "equity": round(equity, 2),
                "currency": currency,
            }
            leverage = validation._safe_float_attr(account, "leverage", 0.0)
            margin_used = validation._safe_float_attr(account, "margin", 0.0)
            margin_free = validation._safe_float_attr(account, "margin_free", 0.0)
            if leverage > 0:
                account_payload["leverage"] = round(leverage, 2)
            account_payload["margin_used"] = round(margin_used, 2)
            account_payload["margin_free"] = round(margin_free, 2)
            account_login = getattr(account, "login", None)
            account_server = getattr(account, "server", None)
            context_id = account_context_id(account_login, account_server)
            if context_id is not None:
                account_payload = {
                    "account_context_id": context_id,
                    **account_payload,
                }
            if account_login is not None:
                account_payload = {"login": account_login, **account_payload}

            result: Dict[str, Any] = {
                "success": True,
                "account": account_payload,
                "portfolio_risk": {
                    "overall_risk_status": overall_risk_status,
                    "quantified_risk_level": quantified_risk_level,
                    "stop_risk_level": stop_risk_level,
                    "margin_risk_level": margin_risk_level,
                    "notional_risk_level": notional_risk_level,
                    "margin_stress": margin_stress,
                    "risk_total_complete": risk_total_complete,
                    "risk_items_quantified": risk_items_quantified,
                    "risk_items_total": risk_items_total,
                    "quantified_risk_currency": round(quantified_risk_currency, 2),
                    "quantified_risk_pct": round(quantified_risk_pct, 2),
                    "total_risk_currency": _round_optional_number(
                        total_risk_currency, 2
                    ),
                    "total_risk_pct": _round_optional_number(total_risk_pct, 2),
                    "open_position_risk_complete": open_position_risk_complete,
                    "quantified_open_position_risk_currency": round(
                        quantified_open_position_risk_currency, 2
                    ),
                    "open_position_risk_currency": _round_optional_number(
                        open_position_risk_currency, 2
                    ),
                    "pending_risk_complete": pending_risk_complete,
                    "quantified_pending_risk_currency": round(
                        total_pending_risk_currency, 2
                    ),
                    "contingent_pending_risk_currency": _round_optional_number(
                        contingent_pending_risk_currency, 2
                    ),
                    "positions_count": len(position_risks),
                    "pending_orders_included": bool(getattr(request, "include_pending", True)),
                    "pending_orders_count": len(pending_order_risks),
                    "positions_without_sl": positions_without_sl,
                    "positions_with_breached_stops": positions_with_breached_stops,
                    "stop_overrun_currency": round(total_stop_overrun_currency, 2),
                    "pending_orders_without_sl": pending_orders_without_sl,
                    "positions_with_risk_calculation_failures": len(
                        risk_calculation_failures
                    ),
                    "notional_exposure": round(total_notional_exposure, 2),
                    "notional_exposure_pct": round(notional_exposure_pct, 2),
                    "notional_to_equity": round(
                        total_notional_exposure / equity, 4
                    ) if equity > 0 else None,
                    "account_leverage": round(leverage, 2) if leverage > 0 else None,
                    "margin_used": round(margin_used, 2),
                    "margin_free": round(margin_free, 2),
                    "open_position_notional_exposure": round(open_position_notional_exposure, 2),
                    "contingent_pending_notional_exposure": round(total_pending_notional_exposure, 2),
                    "notional_exposure_complete": (
                        notional_items_included == notional_items_total
                    ),
                    "notional_positions_included": notional_items_included,
                    "notional_positions_total": notional_items_total,
                    "notional_model": "tick_value_linear_sensitivity",
                },
                "positions": position_risks,
                "units": {
                    "risk_currency": "account_currency",
                    "risk_pct": "percent_of_equity",
                    "notional_value": "account_currency_linearized",
                    "notional_exposure": "account_currency_linearized",
                    "notional_to_equity": "ratio",
                    "volume": "broker_lot",
                    "contract_size": "contract_units_per_lot",
                    "contract_price_product": "contract_size_times_price",
                },
            }
            other_positions_count: Optional[int] = None
            if request.symbol:
                if portfolio_positions_total is not None:
                    other_positions_count = max(
                        0,
                        int(portfolio_positions_total) - len(position_risks),
                    )
                result["scope"] = {
                    "mode": "symbol",
                    "symbol": str(request.symbol),
                    "matched_positions": len(position_risks),
                    **(
                        {"portfolio_positions": int(portfolio_positions_total)}
                        if portfolio_positions_total is not None
                        else {}
                    ),
                    **(
                        {"other_positions": int(other_positions_count)}
                        if other_positions_count is not None
                        else {}
                    ),
                }
                scoped_risk = result.pop("portfolio_risk")
                result["scoped_risk"] = scoped_risk
                result["risk_visibility"] = (
                    "partial"
                    if other_positions_count or not portfolio_snapshot_available
                    else "symbol_scope"
                )
                if not portfolio_snapshot_available:
                    scoped_risk["overall_risk_status"] = "incomplete"
                    scoped_risk["quantified_risk_level"] = "unknown"
                    result["scope_warning"] = (
                        "The symbol-scoped analysis succeeded, but the full portfolio "
                        "position snapshot was unavailable. Aggregate portfolio risk "
                        "and other-position counts are unknown."
                    )
                if other_positions_count:
                    scoped_risk["overall_risk_status"] = "partial"
                    scoped_risk["quantified_risk_level"] = "unknown"
                    result["scope_warning"] = (
                        f"No open {request.symbol} positions matched; "
                        f"{int(other_positions_count)} open position(s) exist on other symbols."
                        if not position_risks
                        else (
                            f"This analysis is scoped to {request.symbol}; "
                            f"{int(other_positions_count)} open position(s) exist on other symbols."
                        )
                    )
                result["sizing_risk_policy"] = {
                    "mode": "incremental_candidate_risk",
                    "risk_target_basis": "percent_of_account_equity",
                    "candidate_symbol": str(request.symbol),
                    "account_margin_context_included": True,
                    "existing_portfolio_stop_risk_included": bool(
                        portfolio_snapshot_available and not other_positions_count
                    ),
                    "note": (
                        "Suggested volume limits this candidate trade's stop risk; "
                        "it does not cap aggregate portfolio stop risk."
                    ),
                }
                if portfolio_positions_total is not None:
                    result["sizing_risk_policy"]["portfolio_positions"] = int(
                        portfolio_positions_total
                    )
                if other_positions_count is not None:
                    result["sizing_risk_policy"]["other_positions"] = int(
                        other_positions_count
                    )
            else:
                result["risk_visibility"] = "portfolio"
                result["scope"] = {
                    "mode": "portfolio",
                    "matched_positions": len(position_risks),
                }
            if getattr(request, "include_pending", True):
                result["pending_orders"] = pending_order_risks
            if risk_calculation_failures:
                result["risk_calculation_failures"] = risk_calculation_failures
            if positions_without_sl > 0 or pending_orders_without_sl > 0:
                warning_parts = []
                if positions_without_sl > 0:
                    warning_parts.append(f"{positions_without_sl} position(s) without stop loss")
                if pending_orders_without_sl > 0:
                    warning_parts.append(f"{pending_orders_without_sl} pending order(s) without stop loss")
                result["warning"] = "; ".join(warning_parts) + " - UNLIMITED RISK!"
            elif risk_calculation_failures:
                result["warning"] = (
                    f"{len(risk_calculation_failures)} position(s) could not be evaluated for risk; "
                    "portfolio risk is incomplete."
            )

            entry_source = None
            live_quote_context: Dict[str, Any] = {}
            if (
                request.entry is None
                and request.symbol
                and request.stop_loss is not None
            ):
                (
                    live_entry,
                    live_entry_source,
                    live_quote_context,
                ) = _resolve_live_trade_risk_entry(
                    gateway=gateway,
                    symbol=request.symbol,
                    direction=request.direction,
                )
                if live_quote_context:
                    result["quote_context"] = live_quote_context
                if live_quote_context.get("quote_side_missing"):
                    required_side = str(
                        live_quote_context.get("required_quote_side") or "quote"
                    )
                    result["position_sizing_error"] = _build_position_sizing_error(
                        code="required_quote_side_missing",
                        field="entry",
                        reason=(
                            "Live risk sizing needs the "
                            f"{required_side} price; the quote is one-sided."
                        ),
                        remediation=(
                            "Refresh the quote and retry when both bid and ask "
                            "are available."
                        ),
                        details={"required_quote_side": required_side},
                    )
                    return result
                if live_entry is not None:
                    request.entry = float(live_entry)
                    entry_source = live_entry_source or "live_tick"

            candidate_symbol_info = None
            if (
                request.symbol
                and request.entry is not None
                and request.stop_loss is not None
            ):
                try:
                    candidate_symbol_info = gateway.symbol_info(request.symbol)
                except Exception:
                    candidate_symbol_info = None

            direction_inference_ambiguous = False
            if (
                entry_was_omitted
                and request.direction is None
                and request.stop_loss is not None
            ):
                quote_bid = _positive_trade_price(live_quote_context.get("bid"))
                quote_ask = _positive_trade_price(live_quote_context.get("ask"))
                if (
                    quote_bid is not None
                    and quote_ask is not None
                    and min(quote_bid, quote_ask)
                    <= float(request.stop_loss)
                    <= max(quote_bid, quote_ask)
                ):
                    direction_inference_ambiguous = True
                    ambiguity_reason = (
                        "Direction cannot be inferred from a stop_loss inside the live "
                        "bid/ask spread. Provide direction='long' or direction='short'."
                    )
                    result["trade_evaluation"] = {
                        "status": "invalid",
                        "symbol": request.symbol,
                        "direction": None,
                        "direction_source": "ambiguous_inside_spread",
                        "entry": request.entry,
                        "sl": float(request.stop_loss),
                        "tp": (
                            float(request.take_profit)
                            if request.take_profit is not None
                            else None
                        ),
                        "error": ambiguity_reason,
                    }
                    result["position_sizing_error"] = _build_position_sizing_error(
                        code="direction_inference_ambiguous",
                        field="direction",
                        reason=ambiguity_reason,
                        entry=(
                            float(request.entry)
                            if request.entry is not None
                            else None
                        ),
                        remediation="Provide direction='long' or direction='short'.",
                        details={
                            "bid": quote_bid,
                            "ask": quote_ask,
                            "stop_loss": float(request.stop_loss),
                            "entry_in_spread": True,
                        },
                    )

            if (
                not direction_inference_ambiguous
                and request.entry is not None
                and request.stop_loss is not None
            ):
                result["trade_evaluation"] = _build_trade_evaluation(
                    symbol=request.symbol,
                    direction=request.direction,
                    entry=float(request.entry),
                    stop_loss=float(request.stop_loss),
                    take_profit=float(request.take_profit)
                    if request.take_profit is not None
                    else None,
                    sym_info=candidate_symbol_info,
                    entry_source=entry_source,
                )

            sizing_method, sizing_method_error = _normalize_trade_risk_sizing_method(
                getattr(request, "sizing_method", "fixed_fraction")
            )
            kelly_inputs, kelly_missing, kelly_source = (
                _extract_trade_risk_kelly_inputs(request)
                if sizing_method == "kelly"
                else ({}, [], None)
            )
            if sizing_method_error:
                result["position_sizing_error"] = _build_position_sizing_error(
                    code="invalid_sizing_method",
                    field="sizing_method",
                    reason=sizing_method_error,
                    details={
                        "sizing_method": getattr(request, "sizing_method", None),
                        "valid_options": ["fixed_fraction", "kelly"],
                    },
                )
            else:
                required_pairs: List[tuple[str, Any]] = [
                    ("entry", request.entry),
                    ("stop_loss", request.stop_loss),
                ]
                if sizing_method == "kelly":
                    required_pairs.extend(
                        (
                            ("kelly_win_rate", kelly_inputs.get("win_rate")),
                            ("kelly_avg_win", kelly_inputs.get("avg_win")),
                            ("kelly_avg_loss", kelly_inputs.get("avg_loss")),
                        )
                    )
                else:
                    required_pairs.insert(
                        0,
                        ("desired_risk_pct", request.desired_risk_pct),
                    )
                position_sizing_missing = [
                    field_name
                    for field_name, value in required_pairs
                    if value is None
                ]
                if position_sizing_missing:
                    provided_pairs: List[tuple[str, Any]] = [
                        ("desired_risk_pct", request.desired_risk_pct),
                        ("entry", request.entry),
                        ("stop_loss", request.stop_loss),
                        ("kelly_win_rate", kelly_inputs.get("win_rate")),
                        ("kelly_avg_win", kelly_inputs.get("avg_win")),
                        ("kelly_avg_loss", kelly_inputs.get("avg_loss")),
                    ]
                    position_sizing_provided = [
                        field_name
                        for field_name, value in provided_pairs
                        if value is not None
                    ]
                    _missing_msg = (
                        "Risk analysis completed. Position sizing is "
                        "available when you provide "
                        + _human_join(
                            [
                                _trade_risk_sizing_field_label(field_name)
                                for field_name in position_sizing_missing
                            ]
                        )
                        + "."
                    )
                    required_for_sizing = [field_name for field_name, _ in required_pairs]
                    position_sizing: Dict[str, Any] = {
                        "status": "parameters_missing",
                        "message": _missing_msg,
                        "missing": position_sizing_missing,
                        "required_for_sizing": required_for_sizing,
                        "note": (
                            "Add --sizing "
                            "'{\"method\":\"fixed_fraction\",\"risk_pct\":1}' "
                            "to risk 1% of equity on the proposed trade."
                        )
                        if sizing_method == "fixed_fraction"
                        else (
                            "Kelly sizing needs win rate and stake-normalized average "
                            "win/loss returns (for example, R-multiples); "
                            "desired_risk_pct is optional and acts as a cap. Raw "
                            "account-currency PnL averages are not valid inputs."
                        ),
                    }
                    if sizing_method == "kelly":
                        position_sizing["sizing_method"] = sizing_method
                    if position_sizing_provided:
                        proposed_context = {
                            key: value
                            for key, value in (
                                ("desired_risk_pct", request.desired_risk_pct),
                                ("entry", request.entry),
                                ("stop_loss", request.stop_loss),
                                ("take_profit", request.take_profit),
                                ("direction", request.direction),
                                ("kelly_win_rate", kelly_inputs.get("win_rate")),
                                ("kelly_avg_win", kelly_inputs.get("avg_win")),
                                ("kelly_avg_loss", kelly_inputs.get("avg_loss")),
                                ("kelly_source", kelly_source),
                            )
                            if value is not None
                        }
                        position_sizing.update(
                            {
                                "provided": position_sizing_provided,
                                "proposed_trade_context": proposed_context,
                                "sizing_not_calculated_reason": (
                                    "Position sizing requires "
                                    + _human_join(
                                        [
                                            _trade_risk_sizing_field_label(field_name)
                                            for field_name in position_sizing_missing
                                        ]
                                    )
                                    + "."
                                ),
                            }
                        )
                    result["position_sizing"] = position_sizing
                    if request.sizing is not None:
                        missing_labels = _human_join(
                            [
                                _trade_risk_sizing_field_label(field_name)
                                for field_name in position_sizing_missing
                            ]
                        )
                        result["position_sizing_error"] = _build_position_sizing_error(
                            code="position_sizing_inputs_missing",
                            reason=(
                                "Position sizing was requested but required inputs "
                                f"are missing: {missing_labels}."
                            ),
                            remediation=(
                                f"Provide {missing_labels} and rerun "
                                "trade_risk_analyze."
                            ),
                            details={"missing_fields": list(position_sizing_missing)},
                        )

            sizing_ready = bool(
                sizing_method_error is None
                and not direction_inference_ambiguous
                and request.entry is not None
                and request.stop_loss is not None
                and (
                    (
                        sizing_method == "fixed_fraction"
                        and request.desired_risk_pct is not None
                    )
                    or (
                        sizing_method == "kelly"
                        and not kelly_missing
                    )
                )
            )
            if sizing_ready and margin_stress["status"] == "critical":
                block_reason = "Account margin stress is critical."
                result["position_sizing_error"] = _build_position_sizing_error(
                    code="portfolio_safety_block",
                    reason=block_reason,
                    remediation=(
                        "Review the full portfolio and reduce margin pressure "
                        "before sizing a new trade."
                    ),
                    details={
                        "margin_stress": margin_stress,
                    },
                )
                result.pop("position_sizing", None)
                sizing_ready = False
            if sizing_ready:
                if not request.symbol:
                    return {"error": "symbol is required for position sizing"}

                sym_info = candidate_symbol_info or gateway.symbol_info(request.symbol)
                if sym_info is None:
                    return {"error": f"Symbol {request.symbol} not found"}

                contract_size = float(sym_info.trade_contract_size)
                tick_value = validation._safe_float_attr(sym_info, "trade_tick_value")
                tick_value_loss = validation._safe_float_attr(
                    sym_info, "trade_tick_value_loss"
                )
                tick_size = validation._safe_float_attr(sym_info, "trade_tick_size")
                risk_tick_value = _resolve_risk_tick_value(
                    tick_value=tick_value,
                    tick_value_loss=tick_value_loss,
                )
                if not math.isfinite(tick_size) or tick_size <= 0:
                    tick_size = 0.0
                min_volume = float(sym_info.volume_min)
                max_volume = float(sym_info.volume_max)
                volume_step = float(sym_info.volume_step)
                if not (
                    math.isfinite(risk_tick_value)
                    and risk_tick_value > 0
                    and math.isfinite(tick_size)
                    and tick_size > 0
                ):
                    result["position_sizing_error"] = _build_position_sizing_error(
                        code="invalid_tick_configuration",
                        reason="Symbol tick configuration is invalid for risk sizing",
                        details={"symbol": request.symbol},
                    )
                    return result
                if not (math.isfinite(volume_step) and volume_step > 0):
                    volume_step = max(min_volume, 0.01)
                if not math.isfinite(contract_size) or contract_size <= 0:
                    contract_size = 1.0

                direction_norm, direction_error, direction_source = (
                    _resolve_trade_risk_direction(
                        direction=request.direction,
                        entry=float(request.entry),
                        stop_loss=float(request.stop_loss),
                        take_profit=float(request.take_profit)
                        if request.take_profit is not None
                        else None,
                    )
                )
                if direction_error or direction_norm is None:
                    result["position_sizing_error"] = _build_position_sizing_error(
                        code=(
                            "direction_unable_to_infer"
                            if direction_source == "unable_to_infer"
                            else "invalid_direction"
                        ),
                        field="direction",
                        reason=(
                            direction_error
                            or "Unable to resolve trade direction for position sizing."
                        ),
                        entry=float(request.entry),
                        remediation=(
                            "Provide direction='long' or direction='short'."
                            if direction_source == "unable_to_infer"
                            else None
                        ),
                        details={
                            "requested_direction": request.direction,
                            "stop_loss": float(request.stop_loss),
                            "take_profit": (
                                float(request.take_profit)
                                if request.take_profit is not None
                                else None
                            ),
                            "direction_source": direction_source,
                        },
                    )
                    return result

                if entry_was_omitted:
                    (
                        directional_entry,
                        directional_source,
                        directional_quote_context,
                    ) = (
                        _resolve_live_trade_risk_entry(
                            gateway=gateway,
                            symbol=request.symbol,
                            direction=direction_norm,
                        )
                    )
                    if directional_quote_context:
                        live_quote_context = directional_quote_context
                        result["quote_context"] = directional_quote_context
                    if directional_quote_context.get("quote_side_missing"):
                        required_side = str(
                            directional_quote_context.get("required_quote_side")
                            or "quote"
                        )
                        result["position_sizing_error"] = _build_position_sizing_error(
                            code="required_quote_side_missing",
                            field="entry",
                            reason=(
                                "Live risk sizing needs the "
                                f"{required_side} price; the quote is one-sided."
                            ),
                            remediation=(
                                "Refresh the quote and retry when both bid and "
                                "ask are available."
                            ),
                            details={"required_quote_side": required_side},
                        )
                        return result
                    if directional_entry is not None:
                        request.entry = float(directional_entry)
                        entry_source = directional_source or "live_tick"
                        result["trade_evaluation"] = _build_trade_evaluation(
                            symbol=request.symbol,
                            direction=direction_norm,
                            entry=float(request.entry),
                            stop_loss=float(request.stop_loss),
                            take_profit=(
                                float(request.take_profit)
                                if request.take_profit is not None
                                else None
                            ),
                            sym_info=sym_info,
                            entry_source=entry_source,
                        )
                level_error = _validate_trade_risk_levels(
                    direction=direction_norm,
                    entry=float(request.entry),
                    stop_loss=float(request.stop_loss),
                    take_profit=float(request.take_profit)
                    if request.take_profit is not None
                    else None,
                )
                if level_error:
                    result["position_sizing_error"] = level_error
                    return result

                if direction_norm == "long":
                    sl_distance_ticks = price_delta_ticks(
                        request.entry,
                        request.stop_loss,
                        tick_size,
                    )
                else:
                    sl_distance_ticks = price_delta_ticks(
                        request.stop_loss,
                        request.entry,
                        tick_size,
                    )
                if sl_distance_ticks is not None and sl_distance_ticks > 0:
                    kelly_context = None
                    if sizing_method == "kelly":
                        effective_risk_pct_raw, kelly_context = (
                            compute_kelly_sizing_context(
                                win_rate=kelly_inputs.get("win_rate"),
                                avg_win=kelly_inputs.get("avg_win"),
                                avg_loss=kelly_inputs.get("avg_loss"),
                                fraction_multiplier=(
                                    request.kelly_fraction_multiplier
                                ),
                                max_risk_pct=request.kelly_max_risk_pct,
                                desired_risk_pct=request.desired_risk_pct,
                                source=kelly_source,
                            )
                        )
                        if effective_risk_pct_raw is None:
                            result["position_sizing_error"] = _build_position_sizing_error(
                                code="invalid_kelly_inputs",
                                reason=(
                                    kelly_context.get("error")
                                    if isinstance(kelly_context, dict)
                                    else "Invalid Kelly sizing inputs"
                                ),
                                remediation=(
                                    "Provide win rate and stake-normalized average "
                                    "win/loss returns from a consistently risk-sized "
                                    "out-of-sample track record."
                                ),
                                details={
                                    "kelly_win_rate": kelly_inputs.get("win_rate"),
                                    "kelly_avg_win": kelly_inputs.get("avg_win"),
                                    "kelly_avg_loss": kelly_inputs.get("avg_loss"),
                                },
                            )
                            return result
                        effective_risk_pct = float(effective_risk_pct_raw)
                    else:
                        effective_risk_pct = float(request.desired_risk_pct)

                    if sizing_method == "kelly" and effective_risk_pct <= 0.0:
                        result["position_sizing"] = {
                            "symbol": request.symbol,
                            "direction": direction_norm,
                            "direction_source": direction_source,
                            "status": "kelly_no_edge",
                            "sizing_method": "kelly",
                            "suggested_volume": 0.0,
                            "volume_lots": 0.0,
                            "requested_risk_currency": 0.0,
                            "risk_amount_account_currency": 0.0,
                            "requested_risk_pct": 0.0,
                            "strict_risk": bool(
                                getattr(request, "strict_risk", True)
                            ),
                            "risk_mode": "strict"
                            if bool(getattr(request, "strict_risk", True))
                            else "flexible",
                            "entry": request.entry,
                            **(
                                {"entry_source": entry_source}
                                if entry_source
                                else {}
                            ),
                            "sl": request.stop_loss,
                            "tp": request.take_profit,
                            "risk_currency": 0.0,
                            "risk_pct": 0.0,
                            "risk_pct_diff": 0.0,
                            "risk_over_target": False,
                            "risk_compliance": "kelly_no_positive_edge",
                            "risk_overshoot_pct": 0.0,
                            "risk_overshoot_currency": 0.0,
                            "raw_volume": 0.0,
                            "volume_step": volume_step,
                            "volume_min": min_volume,
                            "volume_max": max_volume,
                            "volume_rounding": "kelly_no_edge",
                            "notional_value": 0.0,
                            "units": {
                                "account_currency": currency,
                                "volume": BROKER_VOLUME_UNIT,
                                "risk_currency": "account_currency",
                                "risk_pct": "percent_of_equity",
                                "price": "symbol_price",
                                "notional_value": "account_currency_linearized",
                                "tick_value": "account_currency_per_tick_per_lot",
                                "kelly_fraction": "fraction",
                            },
                            "sizing_context": {
                                "equity": round(equity, 2),
                                "account_currency": currency,
                                "contract_size": contract_size,
                                "tick_size": tick_size,
                                "risk_tick_value": round(risk_tick_value, 8),
                                "volume_step": volume_step,
                                "volume_min": min_volume,
                                "volume_max": max_volume,
                            },
                            "sizing_notes": [
                                "Kelly sizing produced no positive edge; suggested volume is 0.0."
                            ],
                            "kelly": kelly_context,
                        }
                        return result

                    risk_amount = equity * (effective_risk_pct / 100.0)
                    raw_volume = risk_amount / (sl_distance_ticks * risk_tick_value)
                    if not math.isfinite(raw_volume) or raw_volume <= 0:
                        result["position_sizing_error"] = _build_position_sizing_error(
                            code="invalid_calculated_volume",
                            reason="Calculated volume is invalid",
                            details={"raw_volume": raw_volume},
                        )
                        return result

                    volume_steps = _floor_volume_steps(raw_volume, volume_step)
                    suggested_volume = volume_steps * volume_step
                    rounding_mode = "rounded_down_to_step"
                    sizing_notes: List[str] = []

                    if suggested_volume < min_volume:
                        suggested_volume = min_volume
                        rounding_mode = "clamped_to_min_volume"
                        sizing_notes.append(
                            "Minimum trade volume forces the size up to the broker minimum."
                        )
                    elif suggested_volume > max_volume:
                        suggested_volume = max_volume
                        rounding_mode = "clamped_to_max_volume"
                        sizing_notes.append(
                            "Maximum trade volume caps the size below the unconstrained target."
                        )
                    elif suggested_volume < raw_volume:
                        sizing_notes.append(
                            "Volume was rounded down to the nearest broker step to avoid exceeding requested risk."
                        )
                    unconstrained_volume = suggested_volume
                    guardrail = resolve_volume_guardrail(
                        trade_guardrails_config,
                        symbol=str(request.symbol or ""),
                        account_info=account,
                    )
                    guardrail_capped_volume = None
                    guardrail_max_volume = guardrail.get("max_volume")
                    guardrail_rule = guardrail.get("binding_rule")
                    guardrail_blocked = False
                    if guardrail.get("blocked"):
                        guardrail_blocked = True
                        suggested_volume = 0.0
                        rounding_mode = "blocked_by_volume_guardrail"
                        sizing_notes.append(
                            str(
                                guardrail.get("reason")
                                or "Volume is blocked by guardrail policy."
                            )
                        )
                    elif (
                        guardrail.get("active")
                        and guardrail_max_volume is not None
                        and math.isfinite(float(guardrail_max_volume))
                        and suggested_volume > float(guardrail_max_volume)
                    ):
                        capped_steps = _floor_volume_steps(
                            float(guardrail_max_volume), volume_step
                        )
                        capped_volume = capped_steps * volume_step
                        if capped_volume < min_volume or capped_volume <= 0:
                            guardrail_blocked = True
                            suggested_volume = 0.0
                            rounding_mode = "blocked_by_volume_guardrail"
                            sizing_notes.append(
                                "No broker-accepted volume fits the configured "
                                "volume guardrail."
                            )
                        else:
                            suggested_volume = capped_volume
                            guardrail_capped_volume = capped_volume
                            rounding_mode = "clamped_to_guardrail_max_volume"
                            sizing_notes.append(
                                "mtdata volume guardrails cap the size below "
                                "the unconstrained target."
                            )
                    if direction_source == "inferred_from_stop_loss":
                        sizing_notes.append(
                            "Direction was inferred from stop-loss placement."
                        )
                    elif direction_source == "inferred_from_take_profit":
                        sizing_notes.append(
                            "Direction was inferred from take-profit placement because stop-loss matched entry."
                        )
                    if sizing_method == "kelly":
                        sizing_notes.append(
                            f"Kelly sizing set effective risk to {effective_risk_pct:.2f}% after multiplier and cap."
                        )

                    step_txt = f"{volume_step:.10f}".rstrip("0")
                    step_decimals = (
                        len(step_txt.split(".")[1]) if "." in step_txt else 0
                    )
                    if step_decimals > 0:
                        suggested_volume = float(
                            f"{suggested_volume:.{step_decimals}f}"
                        )
                        unconstrained_volume = float(
                            f"{unconstrained_volume:.{step_decimals}f}"
                        )
                        if guardrail_capped_volume is not None:
                            guardrail_capped_volume = float(
                                f"{guardrail_capped_volume:.{step_decimals}f}"
                            )
                    else:
                        suggested_volume = float(round(suggested_volume))
                        unconstrained_volume = float(round(unconstrained_volume))
                        if guardrail_capped_volume is not None:
                            guardrail_capped_volume = float(
                                round(guardrail_capped_volume)
                            )

                    actual_risk = sl_distance_ticks * risk_tick_value * suggested_volume
                    actual_risk_pct = (actual_risk / equity) * 100.0
                    risk_pct_diff = actual_risk_pct - effective_risk_pct
                    risk_over_target = actual_risk_pct > (
                        effective_risk_pct + 1e-9
                    )
                    overshoot_pct = max(
                        0.0, float(actual_risk_pct) - effective_risk_pct
                    )
                    overshoot_currency = max(
                        0.0, float(actual_risk) - float(risk_amount)
                    )
                    overshoot_reason = None
                    if risk_over_target:
                        if rounding_mode == "clamped_to_min_volume":
                            overshoot_reason = "min_volume_constraint"
                        elif rounding_mode == "clamped_to_max_volume":
                            overshoot_reason = "max_volume_constraint"
                        elif rounding_mode == "rounded_down_to_step":
                            overshoot_reason = "step_rounding_precision"
                        else:
                            overshoot_reason = "broker_volume_constraints"
                        sizing_notes.append(
                            "Actual risk still exceeds the requested level after broker volume constraints."
                        )

                    strict_risk_blocked = bool(
                        risk_over_target
                        and rounding_mode == "clamped_to_min_volume"
                        and getattr(request, "strict_risk", True)
                    )
                    min_viable_volume = None
                    min_viable_risk_currency = None
                    min_viable_risk_pct = None
                    min_viable_overshoot_pct = None
                    min_viable_overshoot_currency = None
                    min_viable_overshoot_reason = None
                    if strict_risk_blocked:
                        min_viable_volume = suggested_volume
                        min_viable_risk_currency = actual_risk
                        min_viable_risk_pct = actual_risk_pct
                        min_viable_overshoot_pct = overshoot_pct
                        min_viable_overshoot_currency = overshoot_currency
                        min_viable_overshoot_reason = overshoot_reason
                        suggested_volume = 0.0
                        actual_risk = 0.0
                        actual_risk_pct = 0.0
                        risk_pct_diff = -effective_risk_pct
                        risk_over_target = False
                        overshoot_pct = 0.0
                        overshoot_currency = 0.0
                        overshoot_reason = None
                        rounding_mode = "blocked_by_min_volume_risk"
                        sizing_notes.append(
                            "Strict risk is enabled; no broker-accepted volume fits the requested risk."
                        )

                    if guardrail_blocked:
                        result["position_sizing_error"] = _build_position_sizing_error(
                            code="guardrail_volume_block",
                            reason=str(
                                guardrail.get("reason")
                                or "Suggested volume is blocked by mtdata volume guardrails."
                            ),
                            remediation=(
                                "Reduce requested risk or raise the configured "
                                "volume guardrail, then rerun trade_risk_analyze."
                            ),
                            details={
                                "unconstrained_volume": unconstrained_volume,
                                "guardrail_max_volume": guardrail_max_volume,
                                "guardrail_rule": guardrail_rule,
                                "volume_min": min_volume,
                            },
                        )

                    rr_ratio = None
                    reward_currency = None
                    if (
                        request.take_profit is not None
                        and not strict_risk_blocked
                        and not guardrail_blocked
                    ):
                        if direction_norm == "long":
                            tp_distance_ticks = price_delta_ticks(
                                request.take_profit,
                                request.entry,
                                tick_size,
                            )
                        else:
                            tp_distance_ticks = price_delta_ticks(
                                request.entry,
                                request.take_profit,
                                tick_size,
                            )
                        reward_currency = (
                            (tp_distance_ticks or 0) * tick_value * suggested_volume
                        )
                        if actual_risk > 0:
                            rr_ratio = reward_currency / actual_risk

                    notional_value = _linearized_account_currency_notional(
                        volume=abs(suggested_volume),
                        price=float(request.entry),
                        symbol_info=sym_info,
                    )
                    margin_impact = None
                    order_calc_margin = getattr(
                        getattr(gateway, "adapter", None),
                        "order_calc_margin",
                        None,
                    )
                    if callable(order_calc_margin) and suggested_volume > 0:
                        order_type_for_margin = validation._safe_int_attr(
                            gateway,
                            "ORDER_TYPE_BUY" if direction_norm == "long" else "ORDER_TYPE_SELL",
                            0 if direction_norm == "long" else 1,
                        )
                        try:
                            margin_raw = float(
                                order_calc_margin(
                                    order_type_for_margin,
                                    request.symbol,
                                    suggested_volume,
                                    float(request.entry),
                                )
                            )
                        except Exception:
                            margin_raw = math.nan
                        if math.isfinite(margin_raw):
                            margin_impact = {
                                "margin_required": round(margin_raw, 2),
                                "margin_currency": currency or "account_currency",
                            }
                            margin_free = validation._safe_float_attr(account, "margin_free")
                            if margin_free is not None and math.isfinite(margin_free):
                                margin_impact["margin_free"] = round(float(margin_free), 2)
                                margin_impact["margin_sufficient"] = (
                                    float(margin_free) >= float(margin_raw)
                                )

                    risk_compliance = (
                        "blocked_min_volume_exceeds_requested_risk"
                        if strict_risk_blocked
                        else (
                            "exceeds_requested_risk"
                            if risk_over_target
                            else "within_requested_risk"
                        )
                    )
                    shortfall_pct = max(
                        0.0, effective_risk_pct - float(actual_risk_pct)
                    )
                    shortfall_currency = max(
                        0.0, float(risk_amount) - float(actual_risk)
                    )
                    result["position_sizing"] = {
                        "symbol": request.symbol,
                        "direction": direction_norm,
                        "direction_source": direction_source,
                        **(
                            {"status": "risk_too_small_for_min_lot"}
                            if strict_risk_blocked
                            else {}
                        ),
                        "recommendation_status": (
                            "blocked"
                            if strict_risk_blocked or guardrail_blocked
                            else "proposed"
                        ),
                        **(
                            {"sizing_method": "kelly", "kelly": kelly_context}
                            if sizing_method == "kelly"
                            else {}
                        ),
                        "suggested_volume": suggested_volume,
                        "volume_lots": suggested_volume,
                        **(
                            {
                                "unconstrained_volume": unconstrained_volume,
                            }
                            if guardrail_blocked
                            or guardrail_capped_volume is not None
                            else {}
                        ),
                        **(
                            {
                                "guardrail_capped_volume": guardrail_capped_volume,
                            }
                            if guardrail_capped_volume is not None
                            else {}
                        ),
                        **(
                            {
                                "guardrail_max_volume": guardrail_max_volume,
                                "guardrail_rule": guardrail_rule,
                            }
                            if guardrail.get("active") and guardrail_rule
                            else {}
                        ),
                        "requested_risk_currency": round(risk_amount, 2),
                        "risk_amount_account_currency": round(risk_amount, 2),
                        "requested_risk_pct": effective_risk_pct,
                        "strict_risk": bool(getattr(request, "strict_risk", True)),
                        "risk_mode": "strict"
                        if bool(getattr(request, "strict_risk", True))
                        else "flexible",
                        "entry": request.entry,
                        **({"entry_source": entry_source} if entry_source else {}),
                        "sl": request.stop_loss,
                        "tp": request.take_profit,
                        "risk_currency": round(actual_risk, 2),
                        "risk_pct": round(actual_risk_pct, 2),
                        "risk_pct_diff": round(risk_pct_diff, 2),
                        "risk_over_target": risk_over_target,
                        "risk_compliance": risk_compliance,
                        "risk_overshoot_pct": round(overshoot_pct, 2),
                        "risk_overshoot_currency": round(overshoot_currency, 2),
                        "risk_shortfall_pct": round(shortfall_pct, 2),
                        "risk_shortfall_currency": round(shortfall_currency, 2),
                        "risk_over_target_reason": overshoot_reason,
                        "raw_volume": round(raw_volume, 8),
                        "volume_step": volume_step,
                        "volume_min": min_volume,
                        "volume_max": max_volume,
                        "volume_rounding": rounding_mode,
                        "reward_currency": _round_optional_number(
                            reward_currency, 2
                        ),
                        "rr_ratio": _round_optional_number(rr_ratio, 2),
                        "notional_value": _round_optional_number(notional_value, 2),
                        "units": {
                            "account_currency": currency,
                            "volume": BROKER_VOLUME_UNIT,
                            "risk_currency": "account_currency",
                            "risk_pct": "percent_of_equity",
                            "price": "symbol_price",
                            "notional_value": "account_currency_linearized",
                            "tick_value": "account_currency_per_tick_per_lot",
                            **(
                                {"kelly_fraction": "fraction"}
                                if sizing_method == "kelly"
                                else {}
                            ),
                        },
                        "sizing_context": {
                            "equity": round(equity, 2),
                            "account_currency": currency,
                            "contract_size": contract_size,
                            "tick_size": tick_size,
                            "risk_tick_value": round(risk_tick_value, 8),
                            "volume_step": volume_step,
                            "volume_min": min_volume,
                            "volume_max": max_volume,
                        },
                        **({"margin_impact": margin_impact} if margin_impact else {}),
                        "sizing_notes": sizing_notes,
                    }
                    if other_positions_count:
                        result["position_sizing"]["sizing_notes"].append(
                            "Other-symbol positions are present; this is incremental "
                            "candidate sizing, not an aggregate portfolio risk cap."
                        )
                    if strict_risk_blocked:
                        result["position_sizing"].update(
                            {
                                "min_viable_volume": min_viable_volume,
                                "min_viable_risk_currency": round(
                                    float(min_viable_risk_currency or 0.0), 2
                                ),
                                "min_viable_risk_pct": round(
                                    float(min_viable_risk_pct or 0.0), 2
                                ),
                                "min_viable_risk_overshoot_pct": round(
                                    float(min_viable_overshoot_pct or 0.0), 2
                                ),
                                "min_viable_risk_overshoot_currency": round(
                                    float(min_viable_overshoot_currency or 0.0), 2
                                ),
                                "min_viable_risk_over_target": True,
                                "min_viable_risk_over_target_reason": (
                                    min_viable_overshoot_reason
                                ),
                                "strict_risk_hint": (
                                    "Skip trade or set strict_risk=false to accept "
                                    "the minimum-lot risk."
                                ),
                                "nearest_viable": {
                                    "volume": min_viable_volume,
                                    "risk_currency": round(
                                        float(min_viable_risk_currency or 0.0), 2
                                    ),
                                    "risk_pct": round(
                                        float(min_viable_risk_pct or 0.0), 2
                                    ),
                                    "note": (
                                        "Increase desired_risk_pct to this risk_pct "
                                        "or set strict_risk=false to allow the "
                                        "minimum-lot trade."
                                    ),
                                },
                            }
                        )
                    if strict_risk_blocked or risk_over_target:
                        if strict_risk_blocked:
                            result["position_sizing_warning"] = (
                                f"Requested risk {effective_risk_pct:.2f}% but minimum tradable volume risks "
                                f"{float(min_viable_risk_pct or 0.0):.2f}% "
                                f"(+{float(min_viable_overshoot_pct or 0.0):.2f}%); "
                                "suggested_volume is 0.0 because strict_risk is enabled."
                            )
                        else:
                            result["position_sizing_warning"] = (
                                f"Requested risk {effective_risk_pct:.2f}% but actual risk is "
                                f"{float(actual_risk_pct):.2f}% (+{overshoot_pct:.2f}%) after broker volume constraints."
                            )
                        result["risk_alert"] = {
                            "severity": "block" if strict_risk_blocked else "warning",
                            "code": (
                                "min_volume_exceeds_requested_risk"
                                if strict_risk_blocked
                                else "risk_overshoot_after_volume_constraints"
                            ),
                            "reason": (
                                min_viable_overshoot_reason
                                if strict_risk_blocked
                                else overshoot_reason
                            ),
                            "requested_risk_pct": effective_risk_pct,
                            "actual_risk_pct": round(
                                float(min_viable_risk_pct or actual_risk_pct), 2
                            ),
                            "overshoot_pct": round(
                                float(min_viable_overshoot_pct or 0.0)
                                if strict_risk_blocked
                                else overshoot_pct,
                                2,
                            ),
                            "requested_risk_currency": round(risk_amount, 2),
                            "actual_risk_currency": round(
                                float(min_viable_risk_currency or actual_risk), 2
                            ),
                            "overshoot_currency": round(
                                float(min_viable_overshoot_currency or 0.0)
                                if strict_risk_blocked
                                else overshoot_currency,
                                2,
                            ),
                        }
                else:
                    result["position_sizing_error"] = _build_position_sizing_error(
                        code="non_positive_sl_distance",
                        field="stop_loss",
                        reason="SL distance must be greater than 0",
                        entry=float(request.entry),
                        constraint="abs(stop_loss - entry) > 0",
                        details={
                            "direction": direction_norm,
                            "stop_loss": float(request.stop_loss),
                        },
                    )

            return result
        except Exception as exc:
            return {"error": str(exc)}

    return _finish(_analyze_risk())


def run_trade_stress_test(
    request: TradeStressTestRequest,
    *,
    gateway: Any,
) -> Dict[str, Any]:
    """Apply deterministic price shocks to the current open-position snapshot."""
    observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    try:
        gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return {"error": str(exc)}
    try:
        account = gateway.account_info()
        positions = gateway.positions_get()
    except Exception as exc:
        return {"error": str(exc)}
    if positions is None:
        return validation.snapshot_unavailable_error(
            gateway,
            snapshot="positions",
            context="run a portfolio stress test",
        )
    positions = list(positions)
    equity = validation._safe_float_attr(account, "equity", 0.0) if account is not None else 0.0
    currency = str(getattr(account, "currency", "") or "").strip() if account is not None else ""
    rows: List[Dict[str, Any]] = []
    evaluated_positions: List[Any] = []
    warnings_out: List[Dict[str, Any]] = []
    total_pnl = 0.0
    shocked_positions = 0
    for position in positions:
        symbol = str(getattr(position, "symbol", "") or "").strip().upper()
        shock = request.shocks.get(symbol, request.shocks.get("*"))
        if shock is None and not request.include_unshocked:
            continue
        shock_value = float(shock or 0.0)
        symbol_info = gateway.symbol_info(symbol)
        if symbol_info is None:
            warnings_out.append({"symbol": symbol, "warning": "Symbol info unavailable."})
            continue
        current_price = validation._safe_float_attr(position, "price_current", 0.0)
        valuation_basis = "position_current_price"
        if current_price <= 0.0:
            current_price = validation._safe_float_attr(position, "price_open", 0.0)
            valuation_basis = "entry_price_fallback"
        volume = validation._safe_float_attr(position, "volume", 0.0)
        tick_size = validation._safe_float_attr(symbol_info, "trade_tick_size", 0.0)
        if tick_size <= 0.0:
            tick_size = validation._safe_float_attr(symbol_info, "point", 0.0)
        tick_value = validation._safe_float_attr(symbol_info, "trade_tick_value", 0.0)
        tick_value_profit = validation._safe_float_attr(
            symbol_info,
            "trade_tick_value_profit",
            tick_value,
        )
        tick_value_loss = validation._safe_float_attr(
            symbol_info,
            "trade_tick_value_loss",
            tick_value,
        )
        if current_price <= 0.0 or volume <= 0.0 or tick_size <= 0.0:
            warnings_out.append(
                {
                    "ticket": getattr(position, "ticket", None),
                    "symbol": symbol,
                    "warning": "Invalid position price, volume, or symbol tick size.",
                }
            )
            continue
        shocked_price = current_price * (1.0 + shock_value / 100.0)
        side = validation._resolve_position_side(position, gateway) or "SELL"
        side_sign = 1.0 if side == "BUY" else -1.0
        ticks_moved = (shocked_price - current_price) / tick_size
        raw_pnl_sign = side_sign * ticks_moved
        applied_tick_value = tick_value_profit if raw_pnl_sign >= 0.0 else tick_value_loss
        if not math.isfinite(applied_tick_value) or applied_tick_value <= 0.0:
            warnings_out.append(
                {
                    "ticket": getattr(position, "ticket", None),
                    "symbol": symbol,
                    "warning": "Symbol tick value is unavailable; stress P&L cannot be calculated.",
                }
            )
            continue
        pnl_impact = raw_pnl_sign * applied_tick_value * volume
        total_pnl += pnl_impact
        if abs(shock_value) > 0.0:
            shocked_positions += 1
        row: Dict[str, Any] = {
            "ticket": getattr(position, "ticket", None),
            "symbol": symbol,
            "side": side,
            "volume": round(float(volume), 6),
            "shock_pct": round(shock_value, 6),
            "current_price": round(float(current_price), 8),
            "valuation_basis": valuation_basis,
            "shocked_price": round(float(shocked_price), 8),
            "pnl_impact": round(float(pnl_impact), 2),
        }
        if request.detail == "full":
            row.update(
                {
                    "ticks_moved": round(float(ticks_moved), 4),
                    "tick_size": round(float(tick_size), 10),
                    "tick_value_used": round(float(applied_tick_value), 8),
                }
            )
        rows.append(row)
        evaluated_positions.append(position)
    rows.sort(key=lambda row: (float(row.get("pnl_impact") or 0.0), str(row.get("symbol") or "")))
    stressed_equity = float(equity + total_pnl) if equity > 0.0 else None
    result: Dict[str, Any] = {
        "success": True,
        "scope": "open_positions",
        "shocks": dict(request.shocks),
        "positions_total": len(positions),
        "positions_evaluated": len(rows),
        "positions_shocked": int(shocked_positions),
        "total_pnl_impact": round(float(total_pnl), 2),
        "items": rows,
        "count": len(rows),
    }
    if equity > 0.0:
        result.update(
            {
                "equity_before": round(float(equity), 2),
                "equity_after": round(float(stressed_equity), 2),
                "equity_impact_pct": round(float(total_pnl / equity * 100.0), 4),
            }
        )
    if currency:
        result["currency"] = currency
    if not positions:
        result.update(
            {
                "empty": True,
                "status": "no_open_positions",
                "portfolio_status": "no_open_positions",
                "actionability": "informational_no_exposure",
                "message": (
                    "No open positions found; stress metrics reflect zero exposure."
                ),
            }
        )
    elif not rows:
        result.update(
            {
                "success": False,
                "error_code": "stress_no_positions_evaluated",
                "error": (
                    "No open positions matched the requested shocks or had usable "
                    "tick metadata."
                ),
            }
        )
    if warnings_out:
        result["warnings"] = warnings_out
        result["partial_failure"] = True
    result.update(
        _position_mark_freshness(
            gateway,
            evaluated_positions,
            include_contexts=request.detail == "full",
        )
    )
    result.setdefault("valuation_time", observed_at)
    return result


def _position_mark_freshness(
    gateway: Any,
    positions: List[Any],
    *,
    include_contexts: bool = True,
) -> Dict[str, Any]:
    contexts: List[Dict[str, Any]] = []
    symbol_counts: Dict[str, int] = {}
    fallback_counts: Dict[str, int] = {}
    for position in positions:
        symbol = str(getattr(position, "symbol", "") or "").strip()
        if not symbol:
            continue
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        current_price = validation._safe_float_attr(position, "price_current", 0.0)
        open_price = validation._safe_float_attr(position, "price_open", 0.0)
        if current_price <= 0.0 and open_price > 0.0:
            fallback_counts[symbol] = fallback_counts.get(symbol, 0) + 1
    for symbol, position_count in symbol_counts.items():
        try:
            raw_tick = gateway.symbol_info_tick(symbol)
        except Exception:
            raw_tick = None
        query_epoch = time.time()
        tick, quote_source = resolve_quote_tick(
            gateway,
            symbol,
            raw_tick,
            now_epoch=query_epoch,
        )
        context = build_trade_quote_context(
            symbol,
            tick,
            now_epoch=time.time(),
            source_metadata=quote_source,
        )
        context["symbol"] = symbol
        context["positions"] = int(position_count)
        if fallback_counts.get(symbol):
            context["entry_price_fallback_positions"] = fallback_counts[symbol]
            context["valuation_basis"] = "entry_price_fallback"
        contexts.append(context)
    if not contexts:
        return {
            "mark_freshness_status": "not_applicable",
            "data_stale": None,
        }
    def _age(context: Dict[str, Any]) -> float:
        try:
            value = context.get("data_age_seconds")
            return float(value) if value is not None else float("inf")
        except (TypeError, ValueError):
            return float("inf")

    oldest = max(contexts, key=_age)
    fallback_used = bool(fallback_counts)
    live_ready = not fallback_used and all(
        item.get("usable_for_live_trading") is True for item in contexts
    )
    stale_values = [item.get("data_stale") for item in contexts]
    data_stale = (
        True
        if any(value is True for value in stale_values)
        else False
        if all(value is False for value in stale_values)
        else None
    )
    freshness_live = all(
        str(item.get("freshness_state") or "").strip().lower() == "live"
        for item in contexts
    )

    def _unusable_mark(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        symbol = str(item.get("symbol") or "")
        if fallback_counts.get(symbol):
            return {"symbol": symbol, "reason": "entry_price_fallback"}
        if item.get("usable_for_live_trading") is True:
            return None
        if isinstance(item.get("quote_source_conflict"), dict):
            return {"symbol": symbol, "reason": "quote_source_conflict"}
        spread_quality = str(item.get("spread_quality") or "").strip().lower()
        if item.get("spread_valid") is False:
            spread_reasons = {
                "locked": "locked_quote",
                "one_sided": "one_sided_quote",
                "crossed": "crossed_quote",
            }
            mark = {
                "symbol": symbol,
                "reason": spread_reasons.get(spread_quality, "invalid_spread"),
                "spread_quality": spread_quality or "invalid",
            }
            if spread_quality in {"locked", "one_sided", "crossed"}:
                mark["retry_hint"] = "Refresh the quote and retry."
            return mark
        if item.get("data_stale") is True:
            return {
                "symbol": symbol,
                "reason": item.get("freshness_reason") or "quote_too_old",
            }
        market_status = str(item.get("market_status") or "").strip().lower()
        if market_status and market_status not in {"open", "live"}:
            return {"symbol": symbol, "reason": "market_closed"}
        return {
            "symbol": symbol,
            "reason": item.get("freshness_reason") or "not_live_ready",
        }

    unusable_marks = [
        mark
        for item in contexts
        if (mark := _unusable_mark(item)) is not None
    ]
    out: Dict[str, Any] = {
        "mark_freshness_status": (
            "live"
            if freshness_live and data_stale is False
            else "entry_price_fallback"
            if fallback_used
            else "stale_or_unverified"
        ),
        "mark_usability_status": "usable" if live_ready else "not_live_ready",
        "data_stale": data_stale,
        "valuation_time": oldest.get("quote_time"),
        "valuation_basis": (
            "live_position_marks"
            if live_ready
            else "entry_price_fallback"
            if fallback_used
            else "position_marks_quote_not_live_ready"
            if freshness_live and data_stale is False
            else "stale_or_unverified_position_marks"
        ),
        "marks_evaluated": len(contexts),
        "unusable_marks": unusable_marks,
    }
    if include_contexts:
        out["mark_freshness"] = contexts
    if fallback_used:
        out["valuation_warning"] = (
            "One or more positions were valued from entry price because the current "
            "position mark was unavailable; results are not live-mark ready."
        )
        out["entry_price_fallback_positions"] = sum(fallback_counts.values())
    for key in ("market_status", "market_status_reason"):
        values = [item.get(key) for item in contexts if item.get(key) not in (None, "")]
        if values:
            out[key] = values[0]
    return {
        key: value
        for key, value in out.items()
        if value is not None or key == "data_stale"
    }


def run_trade_var_cvar_calculate(  # noqa: C901
    request: TradeVarCvarRequest,
    *,
    gateway: Any,
) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd

    started_at = time.perf_counter()
    observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    log_operation_start(
        logger,
        operation="trade_var_cvar_calculate",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        confidence=request.confidence,
    )

    def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
        if not str(result.get("error") or "").strip():
            result.setdefault("valuation_time", observed_at)
        log_operation_finish(
            logger,
            operation="trade_var_cvar_calculate",
            started_at=started_at,
            success=infer_result_success(result),
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            confidence=request.confidence,
        )
        return result

    try:
        gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return _finish({"error": str(exc)})

    symbol_error = _validate_trading_symbol(gateway, request.symbol)
    if symbol_error is not None:
        return _finish(symbol_error)

    timeframe_value = str(request.timeframe or "").strip().upper()
    if timeframe_value not in TIMEFRAME_MAP:
        return _finish(
            {"error": invalid_timeframe_error(timeframe_value, TIMEFRAME_MAP)}
        )

    method_value, method_error = _normalize_var_cvar_method(request.method)
    if method_error or method_value is None:
        return _finish({"error": method_error})

    transform_value, transform_error = _normalize_var_cvar_transform(request.transform)
    if transform_error or transform_value is None:
        return _finish({"error": transform_error})

    confidence_value, confidence_error = _normalize_var_cvar_confidence(
        request.confidence
    )
    if confidence_error or confidence_value is None:
        return _finish({"error": confidence_error})

    try:
        lookback = int(request.lookback)
    except (TypeError, ValueError):
        return _finish({"error": "lookback must be an integer"})
    if lookback < 2:
        return _finish({"error": "lookback must be at least 2"})
    try:
        horizon_bars = int(request.horizon_bars)
    except (TypeError, ValueError):
        return _finish({"error": "horizon_bars must be an integer"})
    if horizon_bars < 1:
        return _finish({"error": "horizon_bars must be at least 1"})
    history_policy = (
        "includes_current_forming_bar"
        if request.include_incomplete
        else "completed_bars_only"
    )

    try:
        min_observations = int(request.min_observations)
    except (TypeError, ValueError):
        return _finish({"error": "min_observations must be an integer"})
    if min_observations < 2:
        return _finish({"error": "min_observations must be at least 2"})

    try:
        account = gateway.account_info()
    except Exception as exc:
        return _finish(
            {
                "error": (
                    "Failed to get account info for VaR/CVaR calculation: "
                    f"{str(exc)}"
                )
            }
        )
    equity = None
    currency = None
    if account is not None:
        equity_value = validation._safe_float_attr(account, "equity", 0.0)
        if equity_value > 0.0:
            equity = float(equity_value)
        currency_text = str(getattr(account, "currency", "") or "").strip()
        if currency_text:
            currency = currency_text

    try:
        positions = (
            gateway.positions_get(symbol=request.symbol)
            if request.symbol
            else gateway.positions_get()
        )
    except Exception as exc:
        result = validation.snapshot_unavailable_error(
            gateway,
            snapshot="positions",
            context="calculate portfolio VaR/CVaR",
        )
        result["cause"] = str(exc)
        return _finish(result)
    if positions is None:
        return _finish(
            validation.snapshot_unavailable_error(
                gateway,
                snapshot="positions",
                context="calculate portfolio VaR/CVaR",
            )
        )
    if not positions:
        message = (
            f"No open positions found for symbol {request.symbol}"
            if request.symbol
            else "No open positions found for VaR/CVaR calculation."
        )
        summary: Dict[str, Any] = {
            "method": method_value,
            "confidence": round(float(confidence_value), 6),
            "transform": transform_value,
            "timeframe": timeframe_value,
            "horizon_bars": int(horizon_bars),
            "holding_period": (
                f"1 {timeframe_value} bar"
                if horizon_bars == 1
                else f"{horizon_bars} {timeframe_value} bars"
            ),
            "var_interpretation": (
                f"{horizon_bars} {timeframe_value} bar loss on the current position snapshot."
                if horizon_bars == 1
                else (
                    f"{horizon_bars} {timeframe_value} bar overlapping holding-period "
                    "loss on the current position snapshot."
                )
            ),
            "lookback": int(lookback),
            "history_policy": history_policy,
            "forming_candle_status": "not_applicable",
            "min_observations": int(min_observations),
            "observations": 0,
            "positions": 0,
            "symbols": 0,
            "gross_notional": 0.0,
            "net_exposure": 0.0,
            "var": 0.0,
            "cvar": 0.0,
        }
        if equity is not None and equity > 0.0:
            summary["equity"] = round(float(equity), 2)
            summary["var_pct_of_equity"] = 0.0
            summary["cvar_pct_of_equity"] = 0.0
        if currency:
            summary["currency"] = currency
        if request.detail == "full":
            result: Dict[str, Any] = {
                "success": True,
                "message": message,
                "empty": True,
                "status": "no_open_positions",
                "portfolio_status": "no_open_positions",
                "actionability": "informational_no_exposure",
                "summary": summary,
                "symbol_exposures": [],
                "positions": [],
                "worst_observations": [],
            }
        else:
            result = {
                "success": True,
                "empty": True,
                "status": "no_open_positions",
                "portfolio_status": "no_open_positions",
                "actionability": "informational_no_exposure",
                "message": message,
                "positions": 0,
                "history_policy": history_policy,
                "forming_candle_status": "not_applicable",
            }
        if request.symbol:
            result["scope"] = "symbol"
            result["symbol"] = request.symbol
            result["portfolio_hint"] = (
                "Omit symbol to calculate VaR/CVaR for all open positions."
            )
        else:
            result["scope"] = "portfolio"
        if "equity" in summary:
            result["equity"] = summary["equity"]
        if "currency" in summary:
            result["currency"] = summary["currency"]
        return _finish(result)

    mt5_timeframe = TIMEFRAME_MAP[timeframe_value]
    symbol_info_cache: Dict[str, Any] = {}
    history_failures: List[Dict[str, Any]] = []
    valuation_failures: List[Dict[str, Any]] = []
    position_exposures: List[Dict[str, Any]] = []
    symbol_exposures: Dict[str, Dict[str, Any]] = {}

    def _record_valuation_failure(
        position: Any,
        *,
        symbol: Optional[str],
        error: str,
    ) -> None:
        failure: Dict[str, Any] = {
            "ticket": getattr(position, "ticket", None),
            "error": error,
        }
        if symbol:
            failure["symbol"] = symbol
        valuation_failures.append(failure)

    for position in positions:
        symbol = str(getattr(position, "symbol", "") or "").strip()
        if not symbol:
            _record_valuation_failure(
                position,
                symbol=None,
                error="Position has no symbol.",
            )
            continue
        if symbol not in symbol_info_cache:
            symbol_info_cache[symbol] = gateway.symbol_info(symbol)
        symbol_info = symbol_info_cache[symbol]
        if symbol_info is None:
            _record_valuation_failure(
                position,
                symbol=symbol,
                error="Symbol info is unavailable.",
            )
            continue

        volume = validation._safe_float_attr(position, "volume", 0.0)
        if not math.isfinite(volume) or volume <= 0.0:
            _record_valuation_failure(
                position,
                symbol=symbol,
                error="Position volume is invalid.",
            )
            continue

        contract_size = validation._safe_float_attr(
            symbol_info, "trade_contract_size", 1.0
        )
        if not math.isfinite(contract_size) or contract_size <= 0.0:
            contract_size = 1.0
        mark_price = validation._safe_float_attr(position, "price_current", 0.0)
        valuation_basis = "position_current_price"
        if mark_price <= 0.0:
            mark_price = validation._safe_float_attr(position, "price_open", 0.0)
            valuation_basis = "entry_price_fallback"
        if not math.isfinite(mark_price) or mark_price <= 0.0:
            _record_valuation_failure(
                position,
                symbol=symbol,
                error="Position mark price is invalid.",
            )
            continue

        side = validation._resolve_position_side(position, gateway) or "SELL"
        side_sign = 1.0 if side == "BUY" else -1.0
        account_notional = _linearized_account_currency_notional(
            volume=volume,
            price=mark_price,
            symbol_info=symbol_info,
        )
        if account_notional is None:
            _record_valuation_failure(
                position,
                symbol=symbol,
                error=(
                    "Symbol tick value/tick size is unavailable for "
                    "account-currency VaR/CVaR."
                ),
            )
            continue
        signed_notional = side_sign * account_notional
        contract_price_product = abs(volume) * contract_size * mark_price

        position_exposures.append(
            {
                "ticket": getattr(position, "ticket", None),
                "symbol": symbol,
                "side": side,
                "volume": float(volume),
                "mark_price": round(float(mark_price), 6),
                "valuation_basis": valuation_basis,
                "contract_size": round(float(contract_size), 6),
                "signed_notional": round(float(signed_notional), 2),
                "contract_price_product": round(
                    float(contract_price_product), 2
                ),
                "notional_model": "tick_value_linear_sensitivity",
                "unrealized_profit": round(
                    validation._safe_float_attr(position, "profit", 0.0), 2
                ),
            }
        )

        exposure = symbol_exposures.setdefault(
            symbol,
            {
                "symbol": symbol,
                "signed_notional": 0.0,
                "gross_notional": 0.0,
                "positions": 0,
            },
        )
        exposure["signed_notional"] += float(signed_notional)
        exposure["gross_notional"] += abs(float(signed_notional))
        exposure["positions"] += 1

    if valuation_failures:
        return _finish(
            {
                "success": False,
                "error": (
                    "Portfolio VaR/CVaR requires account-currency valuation for "
                    "every open position; refusing a partial calculation."
                ),
                "error_code": "portfolio_var_incomplete",
                "scope": "symbol" if request.symbol else "portfolio",
                "valuation_failures": valuation_failures,
                "omitted_symbols": sorted(
                    {
                        str(item.get("symbol"))
                        for item in valuation_failures
                        if item.get("symbol")
                    }
                ),
                "remediation": (
                    "Restore valid symbol metadata, position volume and mark prices, "
                    "and positive tick-size/tick-value economics for every position."
                ),
            }
        )

    if not position_exposures:
        return _finish(
            {"error": "No usable open positions available for VaR/CVaR calculation."}
        )

    return_series: Dict[str, Any] = {}
    forming_candle_statuses: Dict[str, str] = {}
    for symbol in list(symbol_exposures.keys()):
        try:
            rates = gateway.copy_rates_from_pos(
                symbol,
                mt5_timeframe,
                0,
                lookback + (0 if request.include_incomplete else 1),
            )
            if rates is not None:
                rates = _normalize_times_in_struct(rates)
                forming = _is_last_bar_forming(rates, timeframe_value)
                forming_candle_statuses[symbol] = (
                    "included"
                    if forming and request.include_incomplete
                    else "excluded"
                    if forming
                    else "none_detected"
                )
                if forming and not request.include_incomplete:
                    rates = rates[:-1]
                rates = rates[-lookback:]
        except Exception as exc:
            history_failures.append({"symbol": symbol, "error": str(exc)})
            continue
        returns, history_error = _extract_var_cvar_return_series(
            symbol=symbol,
            rates=rates,
            transform=transform_value,
            pd_module=pd,
            np_module=np,
        )
        if history_error:
            history_failures.append({"symbol": symbol, "error": history_error})
            continue
        return_series[symbol] = returns

    if history_failures:
        return _finish(
            {
                "success": False,
                "error": (
                    "Portfolio VaR/CVaR requires return history for every included "
                    "open-position symbol; refusing a partial calculation."
                ),
                "error_code": "portfolio_var_incomplete",
                "scope": "symbol" if request.symbol else "portfolio",
                "history_failures": history_failures,
                "omitted_symbols": sorted(
                    {
                        str(item.get("symbol"))
                        for item in history_failures
                        if item.get("symbol")
                    }
                ),
                "remediation": (
                    "Restore price history for every omitted symbol or narrow the "
                    "request to a symbol with sufficient history."
                ),
            }
        )

    if not return_series:
        result: Dict[str, Any] = {
            "error": "Unable to build return series for any open-position symbols.",
        }
        if history_failures:
            result["history_failures"] = history_failures
        return _finish(result)

    valid_symbols = set(return_series)
    position_exposures = [
        item for item in position_exposures if item["symbol"] in valid_symbols
    ]
    symbol_exposure_frame = {
        symbol: data
        for symbol, data in symbol_exposures.items()
        if symbol in valid_symbols
    }
    if not position_exposures or not symbol_exposure_frame:
        return _finish(
            {"error": "No open positions remained after filtering unavailable history."}
        )

    aligned_returns = pd.concat(
        [return_series[symbol].rename(symbol) for symbol in symbol_exposure_frame],
        axis=1,
        join="inner",
    ).dropna(how="any")
    if len(aligned_returns) < min_observations:
        result = {
            "error": _format_var_cvar_observation_error(
                observation_name="aligned return",
                available=len(aligned_returns),
                required=min_observations,
                lookback=lookback,
            ),
            "available_observations": int(len(aligned_returns)),
            "min_observations": int(min_observations),
            "lookback": int(lookback),
        }
        if history_failures:
            result["history_failures"] = history_failures
        return _finish(result)

    exposure_vector = pd.Series(
        {
            symbol: float(data["signed_notional"])
            for symbol, data in symbol_exposure_frame.items()
        }
    )
    pnl_returns = aligned_returns[exposure_vector.index]
    if transform_value == "log_return":
        pnl_returns = np.expm1(pnl_returns)
    portfolio_pnl = pnl_returns.mul(exposure_vector, axis=1).sum(axis=1)
    if horizon_bars > 1:
        portfolio_pnl = portfolio_pnl.rolling(window=horizon_bars).sum().dropna()
    pnl_values = [
        float(value) for value in portfolio_pnl.tolist() if math.isfinite(float(value))
    ]
    if len(pnl_values) < min_observations:
        return _finish(
            {
                "error": _format_var_cvar_observation_error(
                    observation_name="finite portfolio PnL",
                    available=len(pnl_values),
                    required=min_observations,
                    lookback=lookback,
                ),
                "available_observations": int(len(pnl_values)),
                "min_observations": int(min_observations),
                "lookback": int(lookback),
            }
        )

    try:
        var_value, cvar_value, threshold = _calculate_var_cvar_from_pnl(
            pnl_values,
            confidence=confidence_value,
            method=method_value,
        )
    except Exception as exc:
        return _finish({"error": str(exc)})

    total_abs_notional = float(
        sum(abs(float(item["signed_notional"])) for item in position_exposures)
    )
    net_exposure = float(
        sum(float(item["signed_notional"]) for item in position_exposures)
    )
    mean_pnl = float(sum(pnl_values) / len(pnl_values))
    if len(pnl_values) > 1:
        variance = sum((value - mean_pnl) ** 2 for value in pnl_values) / float(
            len(pnl_values) - 1
        )
        volatility_pnl = math.sqrt(max(0.0, variance))
    else:
        volatility_pnl = 0.0

    symbol_rows: List[Dict[str, Any]] = []
    for symbol, data in symbol_exposure_frame.items():
        gross_notional = float(data["gross_notional"])
        symbol_rows.append(
            {
                "symbol": symbol,
                "positions": int(data["positions"]),
                "signed_notional": round(float(data["signed_notional"]), 2),
                "gross_notional": round(gross_notional, 2),
                "gross_weight": round((gross_notional / total_abs_notional), 6)
                if total_abs_notional > 0.0
                else 0.0,
            }
        )
    symbol_rows.sort(
        key=lambda item: (-abs(float(item["signed_notional"])), item["symbol"])
    )

    worst_bars = portfolio_pnl.nsmallest(min(5, len(portfolio_pnl)))
    worst_observations = [
        {
            "time": _format_var_cvar_timestamp(timestamp),
            "simulated_pnl": round(float(value), 2),
        }
        for timestamp, value in worst_bars.items()
    ]

    forming_candle_status = (
        "included"
        if "included" in forming_candle_statuses.values()
        else "excluded"
        if "excluded" in forming_candle_statuses.values()
        else "none_detected"
    )
    summary: Dict[str, Any] = {
        "method": method_value,
        "confidence": round(float(confidence_value), 6),
        "tail_probability": round(float(1.0 - confidence_value), 6),
        "confidence_interpretation": (
            f"{confidence_value * 100.0:g}% confidence "
            f"({(1.0 - confidence_value) * 100.0:g}% tail risk)"
        ),
        "transform": transform_value,
        "timeframe": timeframe_value,
        "horizon_bars": int(horizon_bars),
        "holding_period": (
            f"1 {timeframe_value} bar"
            if horizon_bars == 1
            else f"{horizon_bars} {timeframe_value} bars"
        ),
        "var_interpretation": (
            f"One {timeframe_value} bar loss on the current position snapshot."
            if horizon_bars == 1
            else (
                f"{horizon_bars} {timeframe_value} bar overlapping holding-period "
                "loss on the current position snapshot."
            )
        ),
        "lookback": int(lookback),
        "history_policy": history_policy,
        "forming_candle_status": forming_candle_status,
        "min_observations": int(min_observations),
        "observations": int(len(pnl_values)),
        "positions": int(len(position_exposures)),
        "symbols": int(len(symbol_rows)),
        "gross_notional": round(total_abs_notional, 2),
        "net_exposure": round(net_exposure, 2),
        "pnl_model": "tick_value_linear_sensitivity",
        "pnl_unit": "account_currency",
        "var": round(float(var_value), 2),
        "cvar": round(float(cvar_value), 2),
        "tail_threshold": round(float(threshold), 2),
        "mean_pnl": round(mean_pnl, 2),
        "volatility_pnl": round(float(volatility_pnl), 2),
        "worst_observed_pnl": round(min(pnl_values), 2),
        "best_observed_pnl": round(max(pnl_values), 2),
    }
    if equity is not None and equity > 0.0:
        summary["equity"] = round(float(equity), 2)
        summary["var_pct_of_equity"] = round((float(var_value) / equity) * 100.0, 4)
        summary["cvar_pct_of_equity"] = round((float(cvar_value) / equity) * 100.0, 4)
    if currency:
        summary["currency"] = currency

    result = {
        "success": True,
        "scope": "symbol" if request.symbol else "portfolio",
        "history_policy": history_policy,
        "forming_candle_status": forming_candle_status,
        "forming_candle_status_by_symbol": forming_candle_statuses,
        "summary": summary,
        "symbol_exposures": symbol_rows,
        "positions": position_exposures,
        "worst_observations": worst_observations,
    }
    if request.symbol:
        result["symbol"] = request.symbol
        result["portfolio_hint"] = (
            "Omit symbol to calculate VaR/CVaR for all open positions."
        )
    if history_failures:
        result["history_failures"] = history_failures
    result.update(
        _position_mark_freshness(
            gateway,
            positions,
            include_contexts=request.detail == "full",
        )
    )
    return _finish(
        _shape_trade_var_cvar_payload(result, detail=request.detail)
    )
