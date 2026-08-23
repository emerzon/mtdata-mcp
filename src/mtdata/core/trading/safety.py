"""Pre-trade guardrails and safety rails."""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ...shared.market_units import price_delta_ticks
from ...shared.schema import normalize_optional_symbol
from ...utils.coercion import coerce_finite_float
from . import validation
from .sizing import _resolve_risk_tick_value


class TradeSafetyPolicy(BaseModel):
    """Configurable pre-trade safety rails."""

    max_volume: Optional[float] = None
    require_stop_loss: bool = False
    max_deviation: Optional[int] = None
    reduce_only: bool = False


class AccountRiskLimits(BaseModel):
    """Configurable account-level risk thresholds."""

    min_margin_level_pct: Optional[float] = None
    max_floating_loss: Optional[float] = None
    max_total_exposure_lots: Optional[float] = None


class WalletRiskLimits(BaseModel):
    """Portfolio risk caps measured against account wallet metrics."""

    max_risk_pct_of_equity: Optional[float] = None
    max_risk_pct_of_balance: Optional[float] = None
    max_risk_pct_of_free_margin: Optional[float] = None


class TradeGuardrailsConfig(BaseModel):
    """Top-level trade guardrail configuration."""

    enabled: bool = False
    ignore_on_demo: bool = False
    trading_enabled: bool = True
    allowed_symbols: List[str] = Field(default_factory=list)
    blocked_symbols: List[str] = Field(default_factory=list)
    max_volume: Optional[float] = None
    max_volume_by_symbol: Dict[str, float] = Field(default_factory=dict)
    safety_policy: TradeSafetyPolicy = Field(default_factory=TradeSafetyPolicy)
    account_risk_limits: AccountRiskLimits = Field(default_factory=AccountRiskLimits)
    wallet_risk_limits: WalletRiskLimits = Field(default_factory=WalletRiskLimits)

    def is_enabled(self) -> bool:
        return _guardrails_active(self)


def _safe_float_attr(obj: Any, name: str) -> Optional[float]:
    """Extract a float attribute safely, returning None on failure.

    Thin wrapper over the canonical validation helper that pins the
    missing-value default to ``None`` so guardrail checks can distinguish
    absent fields from genuine zeros.
    """
    return validation._safe_float_attr(obj, name, None)


def _normalize_stop_loss_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or math.isclose(numeric, 0.0, abs_tol=1e-12):
        return None
    return numeric


def _normalize_symbol(value: Any) -> str:
    return normalize_optional_symbol(value) or ""


def assess_margin_stress(account: Any) -> Dict[str, Any]:
    """Classify account margin pressure using conservative broker-agnostic bands."""
    getter = account.get if isinstance(account, dict) else lambda key, default=None: getattr(account, key, default)

    def _number(key: str) -> Optional[float]:
        try:
            value = float(getter(key, None))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    equity = _number("equity")
    margin = _number("margin")
    margin_free = _number("margin_free")
    margin_level = _number("margin_level")
    utilization = (margin / equity * 100.0) if equity and equity > 0 and margin is not None else None
    free_ratio = (margin_free / equity * 100.0) if equity and equity > 0 and margin_free is not None else None

    reasons: List[str] = []
    if equity is not None and equity <= 0:
        status = "critical"
        reasons.append("equity_at_or_below_zero")
    elif margin is not None and margin <= 0:
        status = "healthy"
    elif (
        (margin_level is not None and 0 < margin_level <= 120.0)
        or (utilization is not None and utilization >= 90.0)
        or (free_ratio is not None and free_ratio <= 10.0)
    ):
        status = "critical"
        if margin_level is not None and 0 < margin_level <= 120.0:
            reasons.append("margin_level_at_or_below_120_pct")
        if utilization is not None and utilization >= 90.0:
            reasons.append("margin_utilization_at_or_above_90_pct")
        if free_ratio is not None and free_ratio <= 10.0:
            reasons.append("free_margin_at_or_below_10_pct_of_equity")
    elif (
        (margin_level is not None and 0 < margin_level <= 150.0)
        or (utilization is not None and utilization >= 75.0)
        or (free_ratio is not None and free_ratio <= 25.0)
    ):
        status = "stressed"
        if margin_level is not None and 0 < margin_level <= 150.0:
            reasons.append("margin_level_at_or_below_150_pct")
        if utilization is not None and utilization >= 75.0:
            reasons.append("margin_utilization_at_or_above_75_pct")
        if free_ratio is not None and free_ratio <= 25.0:
            reasons.append("free_margin_at_or_below_25_pct_of_equity")
    elif any(value is not None for value in (margin_level, utilization, free_ratio)):
        status = "healthy"
    else:
        status = "unknown"
    margin_level_applicable = not (margin is not None and margin <= 0)
    return {
        "status": status,
        "margin_level_pct": (
            round(margin_level, 2)
            if margin_level is not None and margin_level_applicable
            else None
        ),
        "margin_level_status": (
            "applicable" if margin_level_applicable else "not_applicable"
        ),
        "margin_utilization_pct": round(utilization, 2) if utilization is not None else None,
        "free_margin_pct_of_equity": round(free_ratio, 2) if free_ratio is not None else None,
        "reasons": reasons,
    }


def _normalize_side(value: Any) -> Optional[str]:
    text = str(value or "").strip().upper()
    if not text:
        return None
    if text.startswith("BUY"):
        return "BUY"
    if text.startswith("SELL"):
        return "SELL"
    return text


def _model_has_values(model: Any) -> bool:
    if hasattr(model, "model_dump"):
        values = model.model_dump().values()
    elif isinstance(model, dict):
        values = model.values()
    else:
        values = vars(model).values()
    for value in values:
        if isinstance(value, bool):
            if value:
                return True
            continue
        if isinstance(value, dict):
            if value:
                return True
            continue
        if isinstance(value, list):
            if value:
                return True
            continue
        if value is not None:
            return True
    return False


def _guardrails_active(config: Optional[Any]) -> bool:
    if config is None:
        return False
    if config.enabled:
        return True
    if not config.trading_enabled:
        return True
    return any(
        (
            bool(config.allowed_symbols),
            bool(config.blocked_symbols),
            config.max_volume is not None,
            bool(config.max_volume_by_symbol),
            _model_has_values(config.safety_policy),
            _model_has_values(config.account_risk_limits),
            _model_has_values(config.wallet_risk_limits),
        )
    )


def _wallet_limits_active(limits: Optional[WalletRiskLimits]) -> bool:
    return limits is not None and _model_has_values(limits)


def guardrails_require_position_snapshot(
    config: Optional[TradeGuardrailsConfig],
    *,
    account_info: Any = None,
    enforce_safety_policy: bool = True,
    enforce_account_risk: bool = True,
    enforce_wallet_risk: bool = True,
    for_live_projection: bool = False,
) -> bool:
    """Return whether enabled guardrails depend on the current position book."""
    if not _guardrails_active(config):
        return False
    if (
        not for_live_projection
        and _guardrails_ignored_for_demo(config, account_info=account_info)
    ):
        return False
    if (
        enforce_safety_policy
        and config is not None
        and config.safety_policy.reduce_only
    ):
        return True
    if (
        enforce_account_risk
        and config is not None
        and config.account_risk_limits.max_total_exposure_lots is not None
    ):
        return True
    return bool(
        enforce_wallet_risk
        and config is not None
        and _wallet_limits_active(config.wallet_risk_limits)
    )


def guardrails_require_pending_snapshot(
    config: Optional[TradeGuardrailsConfig],
    *,
    account_info: Any = None,
    enforce_account_risk: bool = True,
    enforce_wallet_risk: bool = True,
    for_live_projection: bool = False,
) -> bool:
    """Return whether enabled portfolio guardrails depend on pending orders."""
    if not _guardrails_active(config):
        return False
    if (
        not for_live_projection
        and _guardrails_ignored_for_demo(config, account_info=account_info)
    ):
        return False
    if (
        enforce_account_risk
        and config is not None
        and config.account_risk_limits.max_total_exposure_lots is not None
    ):
        return True
    return bool(
        enforce_wallet_risk
        and config is not None
        and _wallet_limits_active(config.wallet_risk_limits)
    )


def load_guardrail_book_snapshots(
    gateway: Any,
    config: Optional[TradeGuardrailsConfig],
    *,
    account_info: Any = None,
) -> tuple[List[Any], List[Any], Optional[Dict[str, Any]]]:
    """Load every book component required by configured portfolio limits."""
    try:
        positions = gateway.positions_get()
    except Exception:
        positions = None
    if positions is None and guardrails_require_position_snapshot(
        config,
        account_info=account_info,
    ):
        return [], [], validation.snapshot_unavailable_error(
            gateway,
            snapshot="positions",
            context="evaluate configured trade guardrails",
            guardrail_blocked=True,
        )

    try:
        pending_orders = gateway.orders_get()
    except Exception:
        pending_orders = None
    if pending_orders is None and guardrails_require_pending_snapshot(
        config,
        account_info=account_info,
    ):
        return [], [], validation.snapshot_unavailable_error(
            gateway,
            snapshot="orders",
            context="evaluate configured trade guardrails",
            guardrail_blocked=True,
        )
    return list(positions or []), list(pending_orders or []), None


def _account_is_demo(account_info: Any) -> bool:
    if account_info is None:
        return False

    is_demo = getattr(account_info, "is_demo", None)
    if isinstance(is_demo, bool):
        return is_demo

    account_type = str(getattr(account_info, "account_type", "") or "").strip().lower()
    if account_type == "demo":
        return True
    if account_type:
        return False

    trade_mode = getattr(account_info, "trade_mode", None)
    if isinstance(trade_mode, str):
        return trade_mode.strip().lower() == "demo"
    try:
        return int(trade_mode) == 0
    except Exception:
        return False


def _guardrails_ignored_for_demo(
    config: Optional[TradeGuardrailsConfig],
    *,
    account_info: Any,
) -> bool:
    return bool(
        config is not None
        and getattr(config, "ignore_on_demo", False)
        and _account_is_demo(account_info)
    )


def _build_guardrail_block(
    violations: List[str],
    *,
    rule: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "error": "Trade blocked by guardrails.",
        "guardrail_blocked": True,
        "guardrail_rule": rule,
        "violations": violations,
    }
    if context:
        payload["guardrail_context"] = context
    return payload


def _evaluate_safety_policy(
    policy: Optional[TradeSafetyPolicy],
    *,
    volume: Optional[float] = None,
    stop_loss: Optional[float] = None,
    deviation: Optional[int] = None,
    side: Optional[str] = None,
    existing_side: Optional[str] = None,
    existing_net_volume: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Evaluate *policy* against the given order parameters."""
    if policy is None:
        return None

    violations: List[str] = []

    if policy.max_volume is not None and volume is not None:
        try:
            if math.isfinite(volume) and volume > policy.max_volume:
                violations.append(
                    f"Volume {volume} exceeds safety limit of {policy.max_volume}."
                )
        except (TypeError, ValueError):
            pass

    if policy.require_stop_loss:
        if stop_loss is None or not math.isfinite(stop_loss) or stop_loss <= 0:
            violations.append("Safety policy requires a stop-loss on every order.")

    if policy.max_deviation is not None and deviation is not None:
        try:
            if int(deviation) > policy.max_deviation:
                violations.append(
                    f"Deviation {deviation} exceeds safety limit of {policy.max_deviation}."
                )
        except (TypeError, ValueError):
            pass

    if policy.reduce_only and side is not None:
        opposite = {"BUY": "SELL", "SELL": "BUY"}
        normalized_existing_side = _normalize_side(existing_side)
        normalized_side = _normalize_side(side)
        if normalized_existing_side is None:
            violations.append("Reduce-only policy: no existing position to reduce.")
        elif normalized_side != opposite.get(normalized_existing_side, ""):
            violations.append(
                f"Reduce-only policy: order side {side} does not reduce "
                f"existing {existing_side} position."
            )
        elif volume is not None:
            try:
                order_volume = float(volume)
                net_volume = (
                    abs(float(existing_net_volume))
                    if existing_net_volume is not None
                    else None
                )
            except (TypeError, ValueError):
                order_volume = None
                net_volume = None
            if (
                order_volume is not None
                and math.isfinite(order_volume)
                and net_volume is not None
                and math.isfinite(net_volume)
                and order_volume > net_volume + 1e-9
            ):
                violations.append(
                    f"Reduce-only policy: volume {order_volume} exceeds net "
                    f"position {net_volume}."
                )

    if not violations:
        return None

    return {
        "error": "Order blocked by safety policy.",
        "violations": violations,
    }


def _evaluate_account_risk_gate(
    limits: Optional[AccountRiskLimits],
    *,
    account_info: Any = None,
    projected_exposure_lots: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Evaluate *limits* against current and projected account state."""
    if limits is None:
        return None

    violations: List[str] = []
    snapshot_error_code: Optional[str] = None

    needs_margin_snapshot = limits.min_margin_level_pct is not None
    needs_profit_snapshot = limits.max_floating_loss is not None
    if account_info is None and (needs_margin_snapshot or needs_profit_snapshot):
        required_checks = []
        if needs_margin_snapshot:
            required_checks.append("margin-level")
        if needs_profit_snapshot:
            required_checks.append("floating-loss")
        violations.append(
            "Account snapshot is unavailable; cannot evaluate configured "
            f"{' and '.join(required_checks)} limits."
        )
        snapshot_error_code = "account_snapshot_unavailable"

    if limits.min_margin_level_pct is not None and account_info is not None:
        margin = _safe_float_attr(account_info, "margin")
        margin_level = _safe_float_attr(account_info, "margin_level")
        # MT5 reports margin_level=0 when used margin is zero (no positions).
        # That is undefined / N/A, not a distressed level — match trade_account_info.
        if margin is not None and margin <= 0:
            margin_level = None
        elif margin_level is None or (
            margin is None and math.isclose(margin_level, 0.0, abs_tol=1e-12)
        ):
            violations.append(
                "Account snapshot is incomplete; cannot evaluate the configured "
                "margin-level limit."
            )
            snapshot_error_code = "account_snapshot_incomplete"
        if (
            margin_level is not None
            and math.isfinite(margin_level)
            and margin_level < limits.min_margin_level_pct
        ):
            violations.append(
                f"Margin level {margin_level:.1f}% is below the "
                f"minimum threshold of {limits.min_margin_level_pct:.1f}%."
            )

    if limits.max_floating_loss is not None and account_info is not None:
        profit = _safe_float_attr(account_info, "profit")
        if profit is None:
            violations.append(
                "Account snapshot is incomplete; cannot evaluate the configured "
                "floating-loss limit."
            )
            snapshot_error_code = "account_snapshot_incomplete"
        elif profit < 0 and abs(profit) > limits.max_floating_loss:
            violations.append(
                f"Floating loss ${abs(profit):.2f} exceeds the "
                f"limit of ${limits.max_floating_loss:.2f}."
            )

    if limits.max_total_exposure_lots is not None:
        if projected_exposure_lots > limits.max_total_exposure_lots:
            violations.append(
                f"Total exposure {projected_exposure_lots:.2f} lots would exceed the "
                f"limit of {limits.max_total_exposure_lots:.2f} lots."
            )

    if not violations:
        return None

    result = {
        "error": "Order blocked by account risk gate.",
        "violations": violations,
    }
    if snapshot_error_code is not None:
        result["error_code"] = snapshot_error_code
    return result


def _evaluate_symbol_guardrails(
    config: TradeGuardrailsConfig,
    *,
    symbol: str,
    volume: Optional[float],
) -> Optional[Dict[str, Any]]:
    violations: List[str] = []
    normalized_symbol = _normalize_symbol(symbol)
    allowlist_details: Optional[Dict[str, Any]] = None

    if normalized_symbol:
        blocked = {_normalize_symbol(item) for item in config.blocked_symbols}
        allowed = {_normalize_symbol(item) for item in config.allowed_symbols}
        if normalized_symbol in blocked:
            violations.append(
                f"Symbol {normalized_symbol} is blocked by guardrail policy."
            )
        elif allowed and normalized_symbol not in allowed:
            violations.append(
                f"Symbol {normalized_symbol} is not in the configured allowlist."
            )
            allowlist_details = {
                "error_code": "symbol_not_in_allowlist",
                "allowed_symbols_sample": sorted(item for item in allowed if item)[:5],
                "allowed_symbols_count": len([item for item in allowed if item]),
                "suggestion": "Use one of the allowed symbols or update the trade guardrail allowlist configuration.",
            }

    if volume is not None and math.isfinite(float(volume)):
        limits = []
        if config.max_volume is not None:
            limits.append(("global", float(config.max_volume)))

        # If max_volume_by_symbol is configured, it acts as a whitelist
        if config.max_volume_by_symbol:
            symbol_limit = config.max_volume_by_symbol.get(normalized_symbol)
            if symbol_limit is None:
                violations.append(
                    f"Symbol {normalized_symbol} is not in the configured max_volume_by_symbol allowlist."
                )
                allowlist_keys = [
                    _normalize_symbol(item)
                    for item in config.max_volume_by_symbol.keys()
                    if _normalize_symbol(item)
                ]
                allowlist_details = {
                    "error_code": "symbol_not_in_allowlist",
                    "allowed_symbols_sample": sorted(allowlist_keys)[:5],
                    "allowed_symbols_count": len(allowlist_keys),
                    "suggestion": "Use one of the configured max_volume_by_symbol entries or update the trade guardrail configuration.",
                }
            else:
                limits.append((normalized_symbol, float(symbol_limit)))

        for scope, cap in limits:
            if float(volume) > cap:
                scope_label = (
                    "global max volume"
                    if scope == "global"
                    else f"max volume for {normalized_symbol}"
                )
                violations.append(
                    f"Volume {float(volume)} exceeds the configured {scope_label} of {cap}."
                )

    if not violations:
        return None
    payload = _build_guardrail_block(
        violations,
        rule="symbol_policy",
        context={"symbol": normalized_symbol or symbol, "volume": volume},
    )
    if allowlist_details:
        payload.update(allowlist_details)
    return payload


def _estimate_order_risk_currency(
    *,
    symbol_info: Any,
    volume: float,
    entry_price: float,
    stop_loss: Optional[float],
    side: str,
    wrong_side_policy: Literal["reject", "secured", "overrun"] = "reject",
) -> tuple[Optional[float], Optional[str]]:
    normalized_stop_loss = _normalize_stop_loss_value(stop_loss)
    if normalized_stop_loss is None:
        return None, "stop_loss_missing"

    tick_size = _safe_float_attr(symbol_info, "trade_tick_size")
    tick_value = _safe_float_attr(symbol_info, "trade_tick_value")
    tick_value_loss = _safe_float_attr(symbol_info, "trade_tick_value_loss")
    risk_tick_value = _resolve_risk_tick_value(
        tick_value=tick_value or float("nan"),
        tick_value_loss=tick_value_loss,
    )
    if tick_size is None or tick_size <= 0:
        return None, "tick_size_invalid"
    if not math.isfinite(risk_tick_value) or risk_tick_value <= 0:
        return None, "tick_value_invalid"

    normalized_side = _normalize_side(side)
    if normalized_side == "BUY":
        risk_ticks = price_delta_ticks(float(entry_price), normalized_stop_loss, tick_size)
    else:
        risk_ticks = price_delta_ticks(normalized_stop_loss, float(entry_price), tick_size)
    if risk_ticks is None:
        return None, "stop_loss_wrong_side"
    if risk_ticks <= 0:
        if wrong_side_policy == "secured":
            risk_ticks = 0.0
        elif wrong_side_policy == "overrun":
            risk_ticks = abs(risk_ticks)
        else:
            return None, "stop_loss_wrong_side"

    risk_currency = abs(float(volume) * risk_ticks * risk_tick_value)
    if not math.isfinite(risk_currency) or risk_currency < 0:
        return None, "risk_invalid"
    return risk_currency, None


def _pending_order_volume(order: Any) -> Optional[float]:
    for field in ("volume_current", "volume_initial", "volume"):
        volume = _safe_float_attr(order, field)
        if volume is not None and math.isfinite(volume) and volume > 0.0:
            return abs(volume)
    return None


def _resolve_pending_order_side(order: Any) -> Optional[str]:
    order_type = _safe_float_attr(order, "type")
    if order_type is None:
        return None
    type_value = int(order_type)
    if type_value in {2, 4, 6}:
        return "BUY"
    if type_value in {3, 5, 7}:
        return "SELL"
    return None


def _sum_existing_exposure_lots(
    existing_positions: Optional[List[Any]],
    existing_pending_orders: Optional[List[Any]] = None,
) -> float:
    total = 0.0
    for position in list(existing_positions or []):
        volume = _safe_float_attr(position, "volume")
        if volume is not None:
            total += abs(volume)
    for order in list(existing_pending_orders or []):
        volume = _pending_order_volume(order)
        if volume is not None:
            total += volume
    return total


def _resolve_existing_symbol_net(
    *,
    symbol: str,
    existing_positions: Optional[List[Any]],
) -> tuple[Optional[str], float]:
    """Return (net_side, abs_net_volume) for *symbol* across open positions."""
    normalized_symbol = _normalize_symbol(symbol)
    net_volume = 0.0
    for position in list(existing_positions or []):
        if _normalize_symbol(getattr(position, "symbol", None)) != normalized_symbol:
            continue
        volume = _safe_float_attr(position, "volume")
        side = validation._resolve_position_side(position)
        if volume is None or side is None:
            continue
        net_volume += float(volume) if side == "BUY" else -float(volume)
    if net_volume > 0:
        return "BUY", abs(net_volume)
    if net_volume < 0:
        return "SELL", abs(net_volume)
    return None, 0.0


def _account_uses_hedging(account_info: Any) -> bool:
    """Return whether MT5 reports retail hedging rather than netting mode."""
    margin_mode = getattr(account_info, "margin_mode", None)
    if "HEDGING" in str(margin_mode or "").upper():
        return True
    try:
        return int(margin_mode) == 2
    except (TypeError, ValueError):
        return False


def _account_uses_netting(account_info: Any) -> bool:
    """Return True only when MT5 explicitly reports a netting account mode."""
    margin_mode = getattr(account_info, "margin_mode", None)
    text = str(margin_mode or "").upper()
    if "NETTING" in text or "EXCHANGE" in text:
        return True
    if isinstance(margin_mode, bool) or not isinstance(margin_mode, (int, float)):
        return False
    try:
        return int(margin_mode) in {0, 1}
    except (TypeError, ValueError):
        return False


def _candidate_reduces_existing_net(
    account_info: Any,
    *,
    side: Optional[str],
    net_side: Optional[str],
    net_volume: float,
) -> bool:
    return bool(
        _account_uses_netting(account_info)
        and net_side is not None
        and net_volume > 0
        and side == {"BUY": "SELL", "SELL": "BUY"}.get(net_side)
    )


def _projected_exposure_lots(
    *,
    existing_positions: Optional[List[Any]],
    existing_pending_orders: Optional[List[Any]],
    symbol: str,
    side: Optional[str],
    volume: Optional[float],
    account_info: Any = None,
) -> float:
    """Project gross exposure after applying a candidate order.

    Opposite-side volume up to the current net size reduces exposure; any
    residual that would flip direction is counted as new exposure.
    """
    existing = _sum_existing_exposure_lots(
        existing_positions,
        existing_pending_orders,
    )
    try:
        new_volume = abs(float(volume or 0.0))
    except (TypeError, ValueError):
        new_volume = 0.0
    if new_volume <= 0 or not math.isfinite(new_volume):
        return existing
    if not _account_uses_netting(account_info):
        return existing + new_volume

    normalized_side = _normalize_side(side)
    if normalized_side is None:
        return existing + new_volume

    net_side, net_volume = _resolve_existing_symbol_net(
        symbol=symbol, existing_positions=existing_positions
    )
    opposite = {"BUY": "SELL", "SELL": "BUY"}
    if net_side is None or net_volume <= 0:
        return existing + new_volume
    if normalized_side == net_side:
        return existing + new_volume
    if normalized_side == opposite.get(net_side):
        reduce_by = min(new_volume, net_volume)
        flip_by = max(0.0, new_volume - net_volume)
        return max(0.0, existing - reduce_by + flip_by)
    return existing + new_volume


def _total_portfolio_risk_currency(
    *,
    existing_positions: Optional[List[Any]],
    existing_pending_orders: Optional[List[Any]],
    symbol_info_resolver: Optional[Callable[[str], Any]],
) -> tuple[Optional[float], List[str]]:
    total = 0.0
    issues: List[str] = []
    if not existing_positions and not existing_pending_orders:
        return 0.0, issues
    if symbol_info_resolver is None:
        return None, ["symbol metadata resolver unavailable"]

    for position in list(existing_positions):
        symbol = _normalize_symbol(getattr(position, "symbol", None))
        if not symbol:
            issues.append("position symbol missing")
            continue
        side = validation._resolve_position_side(position)
        if side is None:
            issues.append(f"{symbol}: unable to determine position side")
            continue
        volume = _safe_float_attr(position, "volume")
        mark_price = _safe_float_attr(position, "price_current")
        stop_loss = _safe_float_attr(position, "sl")
        if volume is None or mark_price is None or mark_price <= 0:
            issues.append(f"{symbol}: position metadata is incomplete")
            continue
        if stop_loss is None or math.isclose(stop_loss, 0.0, abs_tol=1e-12):
            issues.append(
                f"{symbol}: existing position risk cannot be quantified without a stop-loss."
            )
            continue
        symbol_info = symbol_info_resolver(symbol)
        if symbol_info is None:
            issues.append(f"{symbol}: symbol metadata unavailable")
            continue
        risk_currency, risk_error = _estimate_order_risk_currency(
            symbol_info=symbol_info,
            volume=abs(volume),
            entry_price=mark_price,
            stop_loss=stop_loss,
            side=side,
            wrong_side_policy="overrun",
        )
        if risk_currency is None:
            issues.append(f"{symbol}: unable to quantify risk ({risk_error}).")
            continue
        total += risk_currency

    for order in list(existing_pending_orders or []):
        symbol = _normalize_symbol(getattr(order, "symbol", None))
        if not symbol:
            issues.append("pending order symbol missing")
            continue
        side = _resolve_pending_order_side(order)
        if side is None:
            issues.append(f"{symbol}: unable to determine pending order side")
            continue
        volume = _pending_order_volume(order)
        entry_price = _safe_float_attr(order, "price_open")
        stop_loss = _safe_float_attr(order, "sl")
        if volume is None or entry_price is None or entry_price <= 0:
            issues.append(f"{symbol}: pending order metadata is incomplete")
            continue
        if stop_loss is None or math.isclose(stop_loss, 0.0, abs_tol=1e-12):
            issues.append(
                f"{symbol}: pending order risk cannot be quantified without a stop-loss."
            )
            continue
        symbol_info = symbol_info_resolver(symbol)
        if symbol_info is None:
            issues.append(f"{symbol}: symbol metadata unavailable")
            continue
        risk_currency, risk_error = _estimate_order_risk_currency(
            symbol_info=symbol_info,
            volume=volume,
            entry_price=entry_price,
            stop_loss=stop_loss,
            side=side,
            wrong_side_policy="reject",
        )
        if risk_currency is None:
            issues.append(
                f"{symbol}: unable to quantify pending order risk ({risk_error})."
            )
            continue
        total += risk_currency

    return total, issues


def _released_position_risk_currency(
    *,
    existing_positions: Optional[List[Any]],
    symbol_info: Any,
    symbol: str,
    net_side: Optional[str],
    reduce_volume: float,
) -> float:
    """Return the quantified risk released by reducing an existing net position."""
    released = 0.0
    normalized_symbol = _normalize_symbol(symbol)
    for position in list(existing_positions or []):
        if _normalize_symbol(getattr(position, "symbol", None)) != normalized_symbol:
            continue
        pos_side = validation._resolve_position_side(position)
        if pos_side != net_side:
            continue
        pos_volume = _safe_float_attr(position, "volume")
        pos_mark = _safe_float_attr(position, "price_current")
        pos_sl = _safe_float_attr(position, "sl")
        if pos_volume is None or pos_mark is None or pos_mark <= 0 or pos_sl is None:
            continue
        if math.isclose(pos_sl, 0.0, abs_tol=1e-12):
            continue
        pos_risk, _err = _estimate_order_risk_currency(
            symbol_info=symbol_info,
            volume=abs(pos_volume),
            entry_price=pos_mark,
            stop_loss=pos_sl,
            side=pos_side,
            wrong_side_policy="overrun",
        )
        if pos_risk is None or pos_volume <= 0:
            continue
        share = min(1.0, reduce_volume / abs(pos_volume)) if pos_volume else 0.0
        released += float(pos_risk) * share
        reduce_volume -= abs(pos_volume) * share
        if reduce_volume <= 1e-12:
            break
    return released


def _evaluate_wallet_risk_limits(
    limits: Optional[WalletRiskLimits],
    *,
    account_info: Any,
    existing_positions: Optional[List[Any]],
    existing_pending_orders: Optional[List[Any]],
    symbol_info: Any,
    symbol_info_resolver: Optional[Callable[[str], Any]],
    symbol: str,
    volume: Optional[float],
    entry_price: Optional[float],
    stop_loss: Optional[float],
    side: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not _wallet_limits_active(limits):
        return None

    violations: List[str] = []
    normalized_side = _normalize_side(side)
    if account_info is None:
        violations.append(
            "Account information is required to enforce wallet risk guardrails."
        )
    if symbol_info is None:
        violations.append(
            f"Symbol metadata is required to enforce wallet risk for {symbol}."
        )
    if volume is None or not math.isfinite(float(volume)) or float(volume) <= 0:
        violations.append(
            "Volume must be finite and positive to enforce wallet risk guardrails."
        )
    if entry_price is None or not math.isfinite(float(entry_price)):
        violations.append("Entry price is required to enforce wallet risk guardrails.")
    if stop_loss is None or not math.isfinite(float(stop_loss)):
        violations.append("Wallet risk guardrails require a stop-loss on the order.")
    if normalized_side is None:
        violations.append("Order side is required to enforce wallet risk guardrails.")
    if violations:
        return _build_guardrail_block(
            violations,
            rule="wallet_risk",
            context={"symbol": _normalize_symbol(symbol)},
        )

    existing_total_risk, existing_issues = _total_portfolio_risk_currency(
        existing_positions=existing_positions,
        existing_pending_orders=existing_pending_orders,
        symbol_info_resolver=symbol_info_resolver,
    )
    if existing_total_risk is None:
        existing_total_risk = 0.0
    if existing_issues:
        return _build_guardrail_block(
            existing_issues,
            rule="wallet_risk",
            context={"symbol": _normalize_symbol(symbol)},
        )

    candidate_risk, candidate_error = _estimate_order_risk_currency(
        symbol_info=symbol_info,
        volume=float(volume),
        entry_price=float(entry_price),
        stop_loss=float(stop_loss),
        side=normalized_side,
        wrong_side_policy="reject",
    )
    if candidate_risk is None:
        return _build_guardrail_block(
            [f"Unable to quantify candidate order risk ({candidate_error})."],
            rule="wallet_risk",
            context={"symbol": _normalize_symbol(symbol)},
        )

    # Project portfolio risk after a reduce/flip rather than always adding.
    net_side, net_volume = _resolve_existing_symbol_net(
        symbol=symbol, existing_positions=existing_positions
    )
    order_volume = float(volume)
    if _candidate_reduces_existing_net(
        account_info,
        side=normalized_side,
        net_side=net_side,
        net_volume=net_volume,
    ):
        reduce_volume = min(order_volume, net_volume)
        flip_volume = max(0.0, order_volume - net_volume)
        # Risk released proportional to volume closed on the existing book.
        released = _released_position_risk_currency(
            existing_positions=existing_positions,
            symbol_info=symbol_info,
            symbol=symbol,
            net_side=net_side,
            reduce_volume=reduce_volume,
        )
        flip_risk = 0.0
        if flip_volume > 1e-12:
            flip_risk_val, flip_err = _estimate_order_risk_currency(
                symbol_info=symbol_info,
                volume=flip_volume,
                entry_price=float(entry_price),
                stop_loss=float(stop_loss),
                side=normalized_side,
            )
            if flip_risk_val is None:
                return _build_guardrail_block(
                    [f"Unable to quantify flip residual risk ({flip_err})."],
                    rule="wallet_risk",
                    context={"symbol": _normalize_symbol(symbol)},
                )
            flip_risk = float(flip_risk_val)
        total_after = max(0.0, float(existing_total_risk) - released + flip_risk)
        candidate_risk = flip_risk  # residual risk introduced by this order
    else:
        total_after = float(existing_total_risk) + float(candidate_risk)
    thresholds = (
        (
            "equity",
            _safe_float_attr(account_info, "equity"),
            limits.max_risk_pct_of_equity,
        ),
        (
            "balance",
            _safe_float_attr(account_info, "balance"),
            limits.max_risk_pct_of_balance,
        ),
        (
            "free_margin",
            _safe_float_attr(account_info, "margin_free"),
            limits.max_risk_pct_of_free_margin,
        ),
    )
    for basis_name, basis_value, limit_pct in thresholds:
        if limit_pct is None:
            continue
        if basis_value is None or basis_value <= 0:
            violations.append(
                f"Cannot enforce max risk pct of {basis_name}: account {basis_name} is unavailable."
            )
            continue
        risk_pct = (total_after / basis_value) * 100.0
        if risk_pct > float(limit_pct):
            violations.append(
                f"Total risk after this trade would be {risk_pct:.2f}% of {basis_name}, "
                f"exceeding the configured limit of {float(limit_pct):.2f}%."
            )

    if not violations:
        return None
    return _build_guardrail_block(
        violations,
        rule="wallet_risk",
        context={
            "symbol": _normalize_symbol(symbol),
            "candidate_risk": round(float(candidate_risk), 2),
            "portfolio_risk_after": round(float(total_after), 2),
        },
    )


def evaluate_trade_guardrails(
    config: Optional[TradeGuardrailsConfig],
    *,
    symbol: str,
    volume: Optional[float],
    stop_loss: Optional[float] = None,
    deviation: Optional[int] = None,
    side: Optional[str] = None,
    entry_price: Optional[float] = None,
    account_info: Any = None,
    existing_positions: Optional[List[Any]] = None,
    existing_pending_orders: Optional[List[Any]] = None,
    symbol_info: Any = None,
    symbol_info_resolver: Optional[Callable[[str], Any]] = None,
    enforce_symbol_rules: bool = True,
    enforce_safety_policy: bool = True,
    enforce_account_risk: bool = True,
    enforce_wallet_risk: bool = True,
) -> Optional[Dict[str, Any]]:
    """Evaluate configured guardrails against an order request."""
    if not _guardrails_active(config):
        return None
    if config is not None and not config.trading_enabled:
        return _build_guardrail_block(
            ["Trading is disabled by guardrail configuration."],
            rule="trading_disabled",
            context={"symbol": _normalize_symbol(symbol)},
        )
    if _guardrails_ignored_for_demo(config, account_info=account_info):
        return None

    normalized_symbol = _normalize_symbol(symbol)
    normalized_side = _normalize_side(side)

    if enforce_symbol_rules:
        symbol_result = _evaluate_symbol_guardrails(
            config,
            symbol=normalized_symbol,
            volume=volume,
        )
        if symbol_result is not None:
            return symbol_result

    existing_side, existing_net_volume = _resolve_existing_symbol_net(
        symbol=normalized_symbol,
        existing_positions=existing_positions,
    )
    if enforce_safety_policy:
        if (
            config.safety_policy is not None
            and config.safety_policy.reduce_only
            and not _account_uses_netting(account_info)
        ):
            block = _build_guardrail_block(
                [
                    "Reduce-only trade_place requires a verified netting account; "
                    "use trade_close with a position ticket on hedging accounts, "
                    "or refresh account information when margin mode is unavailable."
                ],
                rule="safety_policy",
                context={"symbol": normalized_symbol, "side": normalized_side},
            )
            if not _account_uses_hedging(account_info):
                block["error_code"] = "margin_mode_unknown"
            return block
        safety_result = _evaluate_safety_policy(
            config.safety_policy,
            volume=volume,
            stop_loss=stop_loss,
            deviation=deviation,
            side=normalized_side,
            existing_side=existing_side,
            existing_net_volume=existing_net_volume,
        )
        if safety_result is not None:
            return _build_guardrail_block(
                list(safety_result.get("violations") or []),
                rule="safety_policy",
                context={"symbol": normalized_symbol, "side": normalized_side},
            )

    if enforce_account_risk:
        projected_total = _projected_exposure_lots(
            existing_positions=existing_positions,
            existing_pending_orders=existing_pending_orders,
            symbol=normalized_symbol,
            side=normalized_side,
            volume=volume,
            account_info=account_info,
        )
        account_result = _evaluate_account_risk_gate(
            config.account_risk_limits,
            account_info=account_info,
            projected_exposure_lots=projected_total,
        )
        if account_result is not None:
            block = _build_guardrail_block(
                list(account_result.get("violations") or []),
                rule="account_risk",
                context={
                    "symbol": normalized_symbol,
                    "volume": volume,
                    "projected_exposure_lots": round(projected_total, 4),
                },
            )
            error_code = account_result.get("error_code")
            if error_code is not None:
                block["error_code"] = error_code
            return block

    if enforce_wallet_risk:
        wallet_result = _evaluate_wallet_risk_limits(
            config.wallet_risk_limits,
            account_info=account_info,
            existing_positions=existing_positions,
            existing_pending_orders=existing_pending_orders,
            symbol_info=symbol_info,
            symbol_info_resolver=symbol_info_resolver,
            symbol=normalized_symbol,
            volume=volume,
            entry_price=entry_price,
            stop_loss=stop_loss,
            side=normalized_side,
        )
        if wallet_result is not None:
            return wallet_result

    return None


def preview_trade_guardrails(
    config: Optional[TradeGuardrailsConfig],
    *,
    symbol: str,
    volume: Optional[float],
    stop_loss: Optional[float] = None,
    deviation: Optional[int] = None,
    side: Optional[str] = None,
    entry_price: Optional[float] = None,
    account_info: Any = None,
    existing_positions: Optional[List[Any]] = None,
    existing_pending_orders: Optional[List[Any]] = None,
    symbol_info: Any = None,
    symbol_info_resolver: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    """Produce a dry-run friendly preview of guardrail checks."""
    enabled = _guardrails_active(config)
    if not enabled:
        return {
            "enabled": False,
            "blocked": False,
            "checks_not_performed": [],
            "message": "No trade guardrails are configured.",
        }
    ignored_for_demo = bool(
        config is not None
        and config.trading_enabled
        and _guardrails_ignored_for_demo(config, account_info=account_info)
    )
    evaluation_config = config
    if ignored_for_demo and config is not None:
        evaluation_config = copy.copy(config)
        evaluation_config.ignore_on_demo = False

    wallet_risk_active = _wallet_limits_active(evaluation_config.wallet_risk_limits)
    resolved_symbol_info = symbol_info
    if wallet_risk_active and resolved_symbol_info is None and symbol_info_resolver:
        try:
            resolved_symbol_info = symbol_info_resolver(symbol)
        except Exception:
            resolved_symbol_info = None
    wallet_risk_ready = bool(
        wallet_risk_active
        and resolved_symbol_info is not None
        and entry_price is not None
        and math.isfinite(float(entry_price))
    )
    static_result = evaluate_trade_guardrails(
        evaluation_config,
        symbol=symbol,
        volume=volume,
        stop_loss=stop_loss,
        deviation=deviation,
        side=side,
        entry_price=entry_price,
        account_info=account_info,
        existing_positions=existing_positions,
        existing_pending_orders=existing_pending_orders,
        symbol_info=resolved_symbol_info,
        symbol_info_resolver=symbol_info_resolver,
        enforce_account_risk=True,
        enforce_wallet_risk=wallet_risk_ready,
    )
    checks_not_performed: List[str] = []
    if wallet_risk_active and not wallet_risk_ready:
        checks_not_performed.append("wallet_risk")
    preview = {
        "enabled": True,
        "blocked": static_result is not None,
        "violations": list(static_result.get("violations") or [])
        if static_result
        else [],
        "rule": static_result.get("guardrail_rule") if static_result else None,
        "checks_not_performed": checks_not_performed,
    }
    if static_result:
        for key in (
            "error_code",
            "suggestion",
            "allowed_symbols_sample",
            "allowed_symbols_count",
            "guardrail_context",
        ):
            value = static_result.get(key)
            if value not in (None, "", []):
                preview[key] = value
    if ignored_for_demo:
        projection = dict(preview)
        projection.pop("enabled", None)
        return {
            "enabled": True,
            "blocked": False,
            "ignored_for_demo": True,
            "would_block_live": bool(preview["blocked"]),
            "checks_not_performed": list(preview["checks_not_performed"]),
            "live_projection": projection,
            "message": (
                "Trade guardrails are not enforced on this demo account; "
                "live_projection shows the decision for a live account."
            ),
        }
    return preview


def pending_order_risk_increased(
    *,
    symbol_info: Any,
    side: str,
    volume: float,
    existing_entry_price: Optional[float],
    existing_stop_loss: Optional[float],
    candidate_entry_price: Optional[float],
    candidate_stop_loss: Optional[float],
) -> bool:
    """Return True when a pending-order modification increases downside risk."""
    current_sl = _normalize_stop_loss_value(existing_stop_loss)
    next_sl = _normalize_stop_loss_value(candidate_stop_loss)
    current_entry = coerce_finite_float(existing_entry_price)
    next_entry = coerce_finite_float(candidate_entry_price)

    if current_sl is None and next_sl is None:
        return False
    if current_sl is not None and next_sl is None:
        return True
    if next_sl is None or next_entry is None:
        return False

    next_risk, next_error = _estimate_order_risk_currency(
        symbol_info=symbol_info,
        volume=volume,
        entry_price=next_entry,
        stop_loss=next_sl,
        side=side,
        wrong_side_policy="reject",
    )
    if next_risk is None:
        return next_error in {
            "stop_loss_wrong_side",
            "tick_size_invalid",
            "tick_value_invalid",
        }

    if current_sl is None or current_entry is None:
        return True
    current_risk, current_error = _estimate_order_risk_currency(
        symbol_info=symbol_info,
        volume=volume,
        entry_price=current_entry,
        stop_loss=current_sl,
        side=side,
        wrong_side_policy="secured",
    )
    if current_risk is None:
        return True if current_error else False
    return next_risk > (current_risk + 1e-9)
