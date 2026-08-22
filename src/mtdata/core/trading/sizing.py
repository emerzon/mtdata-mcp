"""Shared position-sizing calculations."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ...utils.coercion import coerce_finite_float as _finite_float

DEFAULT_KELLY_FRACTION_MULTIPLIER = 0.5
DEFAULT_KELLY_MAX_RISK_PCT = 2.0
MAX_KELLY_R_MULTIPLE = 10.0


def _floor_volume_steps(raw: float, step: float) -> int:
    """Floor to nearest volume step count."""
    if step <= 0 or not math.isfinite(raw):
        return 0
    step_ratio = raw / step
    step_count = math.floor(step_ratio)
    if step_count < 0:
        return 0

    next_step_count = step_count + 1
    next_volume = float(next_step_count) * float(step)
    if next_volume >= raw:
        snap_tolerance = max(
            math.ulp(float(raw)) * 256.0,
            math.ulp(next_volume) * 256.0,
        )
        if next_volume - float(raw) <= snap_tolerance:
            step_count = next_step_count

    return int(step_count)


def _resolve_risk_tick_value(
    *,
    tick_value: float,
    tick_value_loss: Optional[float] = None,
) -> float:
    """Prefer the broker-reported loss tick value for downside-risk math."""
    try:
        loss_tick_value = float(tick_value_loss)  # type: ignore[arg-type]
    except Exception:
        loss_tick_value = float("nan")
    if math.isfinite(loss_tick_value) and loss_tick_value > 0:
        return loss_tick_value
    return float(tick_value)


def compute_kelly_sizing_context(
    *,
    win_rate: Any,
    avg_win: Any,
    avg_loss: Any,
    fraction_multiplier: Any = DEFAULT_KELLY_FRACTION_MULTIPLIER,
    max_risk_pct: Any = DEFAULT_KELLY_MAX_RISK_PCT,
    desired_risk_pct: Any = None,
    source: Optional[str] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Compute the effective risk percent implied by Kelly inputs.

    ``avg_win`` and ``avg_loss`` must use the same stake-normalized return
    basis (for example, R-multiples), not raw account-currency PnL. Returns
    ``(effective_risk_pct, metadata)``. Non-positive Kelly edge is a valid
    outcome and returns ``0.0`` with ``status="kelly_no_edge"``.
    """
    errors: List[str] = []
    win_rate_f = _finite_float(win_rate)
    avg_win_f = _finite_float(avg_win)
    avg_loss_f = _finite_float(avg_loss)
    multiplier_f = _finite_float(fraction_multiplier)
    max_risk_pct_f = _finite_float(max_risk_pct)
    desired_risk_pct_f = _finite_float(desired_risk_pct)

    if win_rate_f is None or not 0.0 <= win_rate_f <= 1.0:
        errors.append("kelly_win_rate must be finite and between 0 and 1.")
    if avg_win_f is None or avg_win_f <= 0:
        errors.append("kelly_avg_win must be positive and finite.")
    elif avg_win_f > MAX_KELLY_R_MULTIPLE:
        errors.append(
            "kelly_avg_win must be a stake-normalized R-multiple "
            f"(<= {MAX_KELLY_R_MULTIPLE:g}), not account-currency PnL."
        )
    if avg_loss_f is None or avg_loss_f == 0:
        errors.append("kelly_avg_loss must be non-zero and finite.")
    elif abs(avg_loss_f) > MAX_KELLY_R_MULTIPLE:
        errors.append(
            "kelly_avg_loss must be a stake-normalized R-multiple "
            f"(<= {MAX_KELLY_R_MULTIPLE:g}), not account-currency PnL."
        )
    if multiplier_f is None or multiplier_f < 0:
        errors.append("kelly_fraction_multiplier must be non-negative and finite.")
    if max_risk_pct_f is None or max_risk_pct_f <= 0:
        errors.append("kelly_max_risk_pct must be positive and finite.")
    if desired_risk_pct is not None and (
        desired_risk_pct_f is None or desired_risk_pct_f <= 0
    ):
        errors.append("desired_risk_pct must be positive and finite when supplied.")
    if errors:
        return None, {"error": "; ".join(errors)}

    loss_magnitude = abs(float(avg_loss_f))
    if loss_magnitude <= 0:
        return None, {"error": "kelly_avg_loss must be non-zero and finite."}

    odds = float(avg_win_f) / loss_magnitude
    if not math.isfinite(odds) or odds <= 0:
        return None, {"error": "Kelly odds must be positive and finite."}

    probability = float(win_rate_f)
    loss_probability = 1.0 - probability
    kelly_fraction = probability - (loss_probability / odds)
    half_kelly_fraction = kelly_fraction * 0.5
    applied_fraction = kelly_fraction * float(multiplier_f)
    uncapped_risk_pct = max(0.0, applied_fraction * 100.0)
    cap_risk_pct = float(max_risk_pct_f)
    if desired_risk_pct_f is not None:
        cap_risk_pct = min(cap_risk_pct, float(desired_risk_pct_f))
    effective_risk_pct = min(uncapped_risk_pct, cap_risk_pct)

    context: Dict[str, Any] = {
        "win_rate": probability,
        "avg_win_return": float(avg_win_f),
        "avg_loss_return": loss_magnitude,
        "avg_win_loss_ratio": odds,
        "kelly_fraction": kelly_fraction,
        "half_kelly_fraction": half_kelly_fraction,
        "kelly_fraction_multiplier": float(multiplier_f),
        "applied_kelly_fraction": applied_fraction,
        "uncapped_risk_pct": uncapped_risk_pct,
        "cap_risk_pct": cap_risk_pct,
        "effective_risk_pct": effective_risk_pct,
    }
    if source:
        context["source"] = source
    if desired_risk_pct_f is not None:
        context["desired_risk_pct_cap"] = float(desired_risk_pct_f)
    if kelly_fraction <= 0.0 or effective_risk_pct <= 0.0:
        context["status"] = "kelly_no_edge"
        context["effective_risk_pct"] = 0.0
        return 0.0, context
    return effective_risk_pct, context
