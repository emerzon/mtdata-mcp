import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np

from ..shared.constants import TIMEFRAME_SECONDS
from ..shared.schema import DenoiseSpec, TimeframeLiteral
from ..shared.validators import unknown_mapping_keys_error
from ..utils import denoise as _denoise_api
from ..utils.barriers import get_pip_size as _get_pip_size
from ..utils.barriers import get_tick_size as _get_tick_size
from ..utils.barriers import (
    normalize_same_bar_policy,
    normalize_trade_direction,
    resolve_barrier_prices,
    resolve_same_bar_probabilities,
)
from ..utils.coercion import UNPARSED_BOOL, parse_bool_like
from ..utils.coercion import coerce_finite_float as _optional_finite_float
from ..utils.utils import parse_kv_or_json as _parse_kv_or_json
from .barrier_constants import barrier_simulation_param_keys
from .barrier_outcomes import (
    BarrierPathOutcomes,
    barrier_path_payoffs,
    evaluate_barrier_path_outcomes,
)
from .barrier_stats import (
    bootstrap_metric_uncertainty as _bootstrap_uncertainty,
)
from .barrier_stats import (
    cross_seed_stability as _cross_seed_stability,
)
from .barrier_stats import (
    mc_convergence_diagnostic as _mc_convergence,
)
from .barrier_stats import (
    minimum_simulations_for_ci_width as _min_sims_for_ci,
)
from .barrier_stats import (
    sensitivity_analysis_single_parameter as _sensitivity_analysis,
)
from .barrier_stats import (
    statistical_power_analysis as _power_analysis,
)
from .barriers_shared import (
    BARRIER_GRID_PRESETS,
    BROWNIAN_BRIDGE_DUAL_BARRIER_MODEL,
    BROWNIAN_BRIDGE_DUAL_BARRIER_WARNING,
    DEGENERATE_OBJECTIVE_MIN_RESOLVE,
    LOW_PRACTICAL_WIN_PROB_THRESHOLD,
    _annotate_candidate_metrics,
    _apply_barrier_freshness_contract,
    _auto_barrier_method,
    _barrier_exit_quote_reference,
    _binomial_se,
    _binomial_wilson_95,
    _brownian_bridge_hits,
    _build_actionability_payload,
    _build_selection_diagnostics,
    _candidate_is_viable,
    _candidate_status_reason,
    _get_live_reference_price,
    _history_freshness_context,
    _least_negative_ref,
    _live_reference_time_context,
    _prepare_brownian_bridge_draws,
    _resolve_reference_prices,
    _safe_float,
    _sort_candidate_results,
    _stable_barrier_seed,
    _symbol_price_precision,
    barrier_anchor_is_unbreached,
    barrier_method_error,
    normalize_barrier_method,
    normalize_barrier_seed,
    offset_barrier_seed,
)
from .common import annualization_context as _annualization_context
from .common import fetch_history as _fetch_history
from .common import log_returns_from_prices as _log_returns_from_prices
from .common import resolve_forecast_symbol
from .monte_carlo import (
    simulate_bootstrap_mc as _simulate_bootstrap_mc,
)
from .monte_carlo import (
    simulate_garch_mc as _simulate_garch_mc,
)
from .monte_carlo import (
    simulate_gbm_mc as _simulate_gbm_mc,
)
from .monte_carlo import (
    simulate_heston_mc as _simulate_heston_mc,
)
from .monte_carlo import (
    simulate_hmm_mc as _simulate_hmm_mc,
)
from .monte_carlo import (
    simulate_jump_diffusion_mc as _simulate_jump_diffusion_mc,
)

_BARRIER_SEARCH_PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "n_sims": 1200,
        "n_trials": 24,
        "tp_steps": 4,
        "sl_steps": 4,
        "ratio_steps": 4,
        "vol_steps": 4,
        "refine": False,
    },
    "medium": {
        "n_sims": 4000,
        "n_trials": 63,
        "tp_steps": 7,
        "sl_steps": 9,
        "ratio_steps": 8,
        "vol_steps": 7,
        "refine": False,
    },
    "long": {
        "n_sims": 10000,
        "n_trials": 600,
        "tp_steps": 41,
        "sl_steps": 51,
        "ratio_steps": 24,
        "vol_steps": 18,
        "refine": True,
    },
}

_BARRIER_OPTIMIZER_PARAM_KEYS = {
    "commission_bps",
    "commission_pct",
    "concise",
    "convergence_threshold",
    "convergence_window",
    "drift_stress_multipliers",
    "enable_bootstrap",
    "enable_convergence_check",
    "enable_drift_stress",
    "enable_oos_validation",
    "enable_power_analysis",
    "enable_sensitivity_analysis",
    "fast_defaults",
    "gap_aware_stops",
    "grid_preset",
    "grid_style",
    "live_price",
    "max_median_time",
    "max_prob_no_hit",
    "min_barrier_multiplier",
    "min_barrier_pct",
    "min_barrier_pips",
    "min_barrier_ticks",
    "min_edge",
    "min_ev",
    "min_kelly",
    "min_prob_resolve",
    "min_prob_win",
    "n_bootstrap",
    "n_jobs",
    "n_seeds",
    "n_seeds_stability",
    "n_trials",
    "oos_folds",
    "oos_holdout_bars",
    "oos_n_sims",
    "optimizer",
    "optuna_pareto",
    "optuna_pareto_objectives",
    "pareto_limit",
    "power_effect_size",
    "preset",
    "pruner",
    "ratio_max",
    "ratio_min",
    "ratio_steps",
    "refine",
    "refine_radius",
    "refine_steps",
    "rr_max",
    "rr_min",
    "same_bar_policy",
    "sampler",
    "search_profile",
    "sensitivity_params",
    "sl_max",
    "sl_min",
    "sl_steps",
    "slippage_bps",
    "slippage_pct",
    "slippage_pips",
    "spread_bps",
    "spread_pct",
    "spread_pips",
    "statistical_robustness",
    "target_ci_width",
    "timeout",
    "tp_max",
    "tp_min",
    "tp_steps",
    "tradable_only",
    "use_live_price",
    "viable_only",
    "vol_floor_pct",
    "vol_floor_ticks",
    "vol_max_mult",
    "vol_min_mult",
    "vol_sl_multiplier",
    "vol_sl_steps",
    "vol_steps",
    "vol_window",
}
_BARRIER_ENSEMBLE_PARAM_KEYS = {
    "ensemble_agg",
    "ensemble_methods",
    "ensemble_weights",
}


def _barrier_optimizer_param_keys(method: str) -> set[str]:
    method_keys = barrier_simulation_param_keys(
        "auto" if method == "ensemble" else method
    )
    if method == "ensemble":
        method_keys |= _BARRIER_ENSEMBLE_PARAM_KEYS
    return _BARRIER_OPTIMIZER_PARAM_KEYS | method_keys


# Implicit fixed-grid bounds are percent points (intraday preset). Tick mode
# converts them when the caller did not pass tp_min/tp_max/sl_min/sl_max.
_IMPLICIT_FIXED_GRID_PCT = {
    "tp_min": 0.25,
    "tp_max": 1.5,
    "sl_min": 0.25,
    "sl_max": 2.5,
}


def _pct_points_to_ticks(
    value: float,
    *,
    last_price: float,
    tick_size: float,
) -> float:
    return float(value) * (float(last_price) / float(tick_size)) / 100.0


def _bound_is_implicit_fixed_default(
    *,
    params_dict: Dict[str, Any],
    key: str,
    arg_value: Any,
    default: float,
) -> bool:
    if key in params_dict and params_dict.get(key) is not None:
        return False
    numeric = _optional_finite_float(arg_value)
    if numeric is None:
        return True
    return float(numeric) == float(default)


def _apply_implicit_tick_mode_grid_defaults(
    *,
    mode_val: str,
    params_dict: Dict[str, Any],
    tp_min: Any,
    tp_max: Any,
    sl_min: Any,
    sl_max: Any,
    tp_min_val: float,
    tp_max_val: float,
    sl_min_val: float,
    sl_max_val: float,
    last_price: float,
    tick_size: Optional[float],
    pip_size: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    increment = (
        pip_size if mode_val == "pips" else tick_size if mode_val == "ticks" else None
    )
    if increment is None or increment <= 0 or last_price <= 0:
        return tp_min_val, tp_max_val, sl_min_val, sl_max_val
    converted = {
        "tp_min": float(tp_min_val),
        "tp_max": float(tp_max_val),
        "sl_min": float(sl_min_val),
        "sl_max": float(sl_max_val),
    }
    args = {
        "tp_min": tp_min,
        "tp_max": tp_max,
        "sl_min": sl_min,
        "sl_max": sl_max,
    }
    for key, default in _IMPLICIT_FIXED_GRID_PCT.items():
        if _bound_is_implicit_fixed_default(
            params_dict=params_dict,
            key=key,
            arg_value=args[key],
            default=default,
        ):
            converted[key] = _pct_points_to_ticks(
                converted[key],
                last_price=last_price,
                tick_size=float(increment),
            )
    return converted["tp_min"], converted["tp_max"], converted["sl_min"], converted["sl_max"]


def _tick_grid_below_min_distance_error(
    *,
    min_barrier_distance: float,
    cost_per_trade: float,
    mode: str = "ticks",
) -> Dict[str, Any]:
    min_distance = float(min_barrier_distance)
    cost = float(cost_per_trade)
    unit = str(mode)
    return {
        "success": False,
        "error": (
            "No TP/SL candidates remain: every grid point is below "
            f"min_barrier_distance={min_distance:g} {unit} "
            f"(cost_per_trade={cost:g} {unit}). "
            f'Widen the search with --params "tp_min=… sl_min=…" in {unit}.'
        ),
        "error_code": "invalid_input",
        "status": "no_candidates",
        "min_barrier_distance": min_distance,
        "cost_per_trade": cost,
        "mode": unit,
        "distance_unit": unit,
        "remediation": (
            f'Pass --params "tp_min=… sl_min=…" in {unit} so both bounds exceed '
            f"min_barrier_distance={min_distance:g} "
            f"(cost_per_trade={cost:g})."
        ),
    }


@dataclass(frozen=True)
class _BarrierEvaluationContext:
    mode_val: str
    dir_long: bool
    last_price: float
    tick_size: Optional[float]
    rr_min_val: Optional[float]
    rr_max_val: Optional[float]
    has_trading_costs: bool
    ev_deduct_cost: float
    cost_per_trade: float
    min_prob_win_val: Optional[float]
    max_prob_no_hit_val: Optional[float]
    min_prob_resolve_val: Optional[float]
    max_median_time_val: Optional[float]
    same_bar_policy: str = "sl_first"
    gap_aware_stops: bool = False
    pip_size: Optional[float] = None
    path_anchor_price: Optional[float] = None


@dataclass(frozen=True)
class _BarrierBridgeInputs:
    enabled: bool
    sigma: float
    log_paths: Optional[np.ndarray]
    uniform_tp: Optional[np.ndarray]
    uniform_sl: Optional[np.ndarray]


def _coerce_barrier_bool_flag(value: Any, default: bool = False) -> bool:
    parsed = parse_bool_like(value)
    if parsed is UNPARSED_BOOL:
        return bool(default)
    return bool(parsed)


def _resolve_barrier_search_profile_config(
    params_dict: Dict[str, Any],
    *,
    search_profile: Any,
    fast_defaults: Any,
) -> Tuple[str, Dict[str, Any]]:
    search_profile_requested = str(
        params_dict.get("search_profile", search_profile)
    ).strip().lower()
    if search_profile_requested not in _BARRIER_SEARCH_PROFILE_DEFAULTS:
        valid_profiles = ", ".join(_BARRIER_SEARCH_PROFILE_DEFAULTS)
        raise ValueError(
            f"Invalid search_profile {search_profile_requested!r}. "
            f"Valid profiles: {valid_profiles}."
        )
    fast_defaults_requested = _coerce_barrier_bool_flag(
        params_dict.get("fast_defaults", fast_defaults),
        default=bool(fast_defaults),
    )
    search_profile_val = "fast" if fast_defaults_requested else search_profile_requested
    return search_profile_val, dict(_BARRIER_SEARCH_PROFILE_DEFAULTS[search_profile_val])


def _barrier_search_config(
    *,
    search_profile: str,
    grid_style: str,
    preset: Optional[str],
    mode: str,
    objective: str,
    tp_min: float,
    tp_max: float,
    sl_min: float,
    sl_max: float,
    tp_steps: int,
    sl_steps: int,
    ratio_min: float,
    ratio_max: float,
    ratio_steps: int,
    vol_window: int,
    vol_min_mult: float,
    vol_max_mult: float,
    vol_steps: int,
    candidate_pairs: Optional[List[Tuple[Any, Any]]] = None,
) -> Dict[str, Any]:
    def _range(values: List[float], fallback_min: float, fallback_max: float) -> List[float]:
        finite = [float(value) for value in values if np.isfinite(float(value))]
        if finite:
            return [_safe_float(min(finite)), _safe_float(max(finite))]
        return [_safe_float(fallback_min), _safe_float(fallback_max)]

    pairs = candidate_pairs or []
    tp_values: List[float] = []
    sl_values: List[float] = []
    for tp_value, sl_value in pairs:
        try:
            tp_float = float(tp_value)
            sl_float = float(sl_value)
        except Exception:
            continue
        if np.isfinite(tp_float):
            tp_values.append(tp_float)
        if np.isfinite(sl_float):
            sl_values.append(sl_float)

    out: Dict[str, Any] = {
        "profile": search_profile,
        "grid_style": grid_style,
        "preset": preset if grid_style == "preset" else None,
        "mode": mode,
        "objective": objective,
        "tp_range": _range(tp_values, tp_min, tp_max),
        "sl_range": _range(sl_values, sl_min, sl_max),
        "tp_steps": int(tp_steps),
        "sl_steps": int(sl_steps),
    }
    if grid_style == "ratio":
        out["ratio_range"] = [_safe_float(ratio_min), _safe_float(ratio_max)]
        out["ratio_steps"] = int(ratio_steps)
    if grid_style == "volatility":
        out["vol_window"] = int(vol_window)
        out["vol_mult_range"] = [_safe_float(vol_min_mult), _safe_float(vol_max_mult)]
        out["vol_steps"] = int(vol_steps)
    return {key: value for key, value in out.items() if value is not None}


def _barrier_candidate_filter_config(
    *,
    tradable_only: bool,
    min_ev: Optional[float],
    min_edge: Optional[float],
    min_kelly: Optional[float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if tradable_only:
        out["tradable_only"] = True
    if min_ev is not None:
        out["min_ev"] = _safe_float(min_ev)
    if min_edge is not None:
        out["min_edge"] = _safe_float(min_edge)
        out["min_edge_basis"] = "resolved_edge_vs_breakeven"
    if min_kelly is not None:
        out["min_kelly"] = _safe_float(min_kelly)
    return out


def _candidate_passes_threshold_filters(
    row: Dict[str, Any],
    *,
    cost_per_trade: float,
    tradable_only_val: bool,
    min_ev_val: Optional[float],
    min_edge_val: Optional[float],
    min_kelly_val: Optional[float],
) -> bool:
    _annotate_candidate_metrics(row, cost_per_trade=cost_per_trade)
    if tradable_only_val and not _candidate_is_viable(row, cost_per_trade=cost_per_trade):
        return False
    ev_value = _safe_float(row.get("ev"))
    if min_ev_val is not None and (ev_value is None or ev_value < min_ev_val):
        return False
    edge_value = _safe_float(row.get("edge_vs_breakeven"))
    if edge_value is None:
        edge_value = _safe_float(row.get("edge"))
    if min_edge_val is not None and (edge_value is None or edge_value < min_edge_val):
        return False
    kelly_value = _safe_float(row.get("kelly"))
    if min_kelly_val is not None and (kelly_value is None or kelly_value < min_kelly_val):
        return False
    return True


def _resolve_profile_param(
    params_dict: Dict[str, Any],
    profile_cfg: Dict[str, Any],
    *,
    param_key: str,
    arg_value: Any,
) -> Any:
    if param_key in params_dict:
        return params_dict[param_key]
    if arg_value is not None:
        return arg_value
    return profile_cfg[param_key]


def _post_selection_gate_diagnostics(
    statistical_analysis: Optional[Dict[str, Any]],
    *,
    objective: str,
) -> Dict[str, Any]:
    """Surface debiased post-selection evidence so the trade gate can use it.

    The reported ``best`` metrics are an argmax over many Monte Carlo estimates,
    so the winning value is optimistically biased. When an independent-seed
    re-evaluation or a walk-forward holdout is available, its verdict overrides
    the in-sample one for gating purposes instead of sitting in diagnostics.
    """
    out: Dict[str, Any] = {}
    if not isinstance(statistical_analysis, dict):
        return out

    post_selection = statistical_analysis.get("post_selection_evaluation")
    if isinstance(post_selection, dict):
        optimism = _safe_float(post_selection.get("optimism"))
        holdout_metrics = post_selection.get("metrics")
        holdout_ev = (
            _safe_float(holdout_metrics.get("ev"))
            if isinstance(holdout_metrics, dict)
            else None
        )
        if optimism is not None:
            out["post_selection_optimism"] = float(optimism)
        if holdout_ev is not None:
            out["post_selection_holdout_ev"] = float(holdout_ev)
            if holdout_ev < 0.0:
                out["post_selection_ev_negative"] = True
                out["post_selection_basis"] = "independent_seed_reevaluation"
        holdout_objective = _safe_float(post_selection.get("holdout_estimate"))
        if holdout_objective is not None:
            out["post_selection_holdout_objective"] = float(holdout_objective)
            out["post_selection_objective"] = str(objective)

    walk_forward = statistical_analysis.get("walk_forward_oos")
    if isinstance(walk_forward, dict) and not walk_forward.get("error"):
        mean_realized_ev = _safe_float(walk_forward.get("mean_realized_ev"))
        if mean_realized_ev is not None:
            out["walk_forward_mean_realized_ev"] = float(mean_realized_ev)
            if mean_realized_ev < 0.0:
                out["walk_forward_ev_negative"] = True
    return out


def _candidate_barrier_prices(
    tp_unit: float,
    sl_unit: float,
    *,
    context: _BarrierEvaluationContext,
) -> Tuple[float, float]:
    barrier_kwargs: Dict[str, float]
    if context.mode_val == "pct":
        barrier_kwargs = {"tp_pct": tp_unit, "sl_pct": sl_unit}
    elif context.mode_val == "pips":
        barrier_kwargs = {"tp_pips": tp_unit, "sl_pips": sl_unit}
    else:
        barrier_kwargs = {"tp_ticks": tp_unit, "sl_ticks": sl_unit}
    tp_price, sl_price = resolve_barrier_prices(
        price=context.last_price,
        direction="long" if context.dir_long else "short",
        tick_size=context.tick_size,
        pip_size=context.pip_size,
        **barrier_kwargs,
    )
    if tp_price is None or sl_price is None:
        return float("nan"), float("nan")
    return float(tp_price), float(sl_price)


def _candidate_barrier_geometry_is_valid(
    tp_price: float,
    sl_price: float,
    *,
    context: _BarrierEvaluationContext,
) -> bool:
    if not np.isfinite(tp_price) or not np.isfinite(sl_price):
        return False
    if not np.isfinite(context.last_price) or context.last_price <= 0.0:
        return False
    if tp_price <= 0.0 or sl_price <= 0.0:
        return False
    if context.dir_long:
        if not sl_price < context.last_price < tp_price:
            return False
    elif not tp_price < context.last_price < sl_price:
        return False
    # Barriers are measured from the entry quote, but paths are scored from the
    # executable exit quote. A barrier inside the spread is already breached at
    # t=0, and the path matrix (which starts at bar 1) cannot observe that.
    return barrier_anchor_is_unbreached(
        anchor_price=context.path_anchor_price,
        dir_long=context.dir_long,
        tp_price=tp_price,
        sl_price=sl_price,
    )


def _candidate_hit_arrays(
    eval_paths: np.ndarray,
    *,
    tp_trigger: float,
    sl_trigger: float,
    context: _BarrierEvaluationContext,
    bridge_inputs: _BarrierBridgeInputs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    extra_tp_hits = None
    extra_sl_hits = None
    if (
        bridge_inputs.enabled
        and bridge_inputs.log_paths is not None
        and bridge_inputs.uniform_tp is not None
        and bridge_inputs.uniform_sl is not None
    ):
        tp_dir = "up" if context.dir_long else "down"
        sl_dir = "down" if context.dir_long else "up"
        tp_bridge = _brownian_bridge_hits(
            bridge_inputs.log_paths,
            float(np.log(max(1e-12, tp_trigger))),
            bridge_inputs.sigma,
            direction=tp_dir,
            uniform=bridge_inputs.uniform_tp,
        )
        sl_bridge = _brownian_bridge_hits(
            bridge_inputs.log_paths,
            float(np.log(max(1e-12, sl_trigger))),
            bridge_inputs.sigma,
            direction=sl_dir,
            uniform=bridge_inputs.uniform_sl,
        )
        extra_tp_hits = tp_bridge
        extra_sl_hits = sl_bridge
    outcomes = evaluate_barrier_path_outcomes(
        eval_paths,
        tp_trigger=tp_trigger,
        sl_trigger=sl_trigger,
        direction="long" if context.dir_long else "short",
        extra_tp_hits=extra_tp_hits,
        extra_sl_hits=extra_sl_hits,
    )
    return (
        outcomes.first_tp,
        outcomes.first_sl,
        outcomes.wins,
        outcomes.losses,
        outcomes.ties,
    )


def _candidate_resolution_timing(
    first_tp: np.ndarray,
    first_sl: np.ndarray,
    wins: np.ndarray,
    losses: np.ndarray,
    ties: np.ndarray,
    *,
    horizon: int,
    same_bar_policy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return policy-resolved rows and unconditional bars in trade."""
    resolve_mask = wins | losses
    if same_bar_policy != "neutral":
        resolve_mask |= ties
    time_in_trade = np.minimum(np.minimum(first_tp, first_sl) + 1, horizon)
    if same_bar_policy == "neutral" and np.any(ties):
        time_in_trade = np.array(time_in_trade, copy=True)
        time_in_trade[ties] = horizon
    return resolve_mask, time_in_trade


def _evaluate_barrier_candidate(
    tp_unit: float,
    sl_unit: float,
    eval_paths: np.ndarray,
    *,
    context: _BarrierEvaluationContext,
    bridge_inputs: _BarrierBridgeInputs,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    sims_total, horizon_total = eval_paths.shape
    if sims_total <= 0 or horizon_total <= 0:
        return None, True

    tp_price, sl_price = _candidate_barrier_prices(tp_unit, sl_unit, context=context)
    if not _candidate_barrier_geometry_is_valid(tp_price, sl_price, context=context):
        return None, True

    first_tp, first_sl, wins, losses, ties = _candidate_hit_arrays(
        eval_paths,
        tp_trigger=tp_price,
        sl_trigger=sl_price,
        context=context,
        bridge_inputs=bridge_inputs,
    )
    n_wins = int(wins.sum())
    n_losses = int(losses.sum())
    n_ties = int(ties.sum())

    prob_tp_strict = n_wins / sims_total
    prob_sl_strict = n_losses / sims_total
    prob_same_bar = n_ties / sims_total
    prob_no_hit = max(0.0, 1.0 - prob_tp_strict - prob_sl_strict - prob_same_bar)
    resolved = resolve_same_bar_probabilities(
        tp_strict=prob_tp_strict,
        sl_strict=prob_sl_strict,
        same_bar=prob_same_bar,
        no_hit=prob_no_hit,
        policy=context.same_bar_policy,
    )
    prob_tp_first = resolved["prob_tp_first"]
    prob_sl_first = resolved["prob_sl_first"]
    prob_resolve = resolved["prob_resolve"]
    effective_prob_win = prob_tp_first
    effective_prob_loss = prob_sl_first

    unresolved_mask = ~(wins | losses | ties)
    path_outcomes = BarrierPathOutcomes(
        first_tp=first_tp,
        first_sl=first_sl,
        wins=wins,
        losses=losses,
        ties=ties,
        unresolved=unresolved_mask,
        time_in_trade=np.minimum(
            np.minimum(first_tp, first_sl) + 1,
            horizon_total,
        ),
        horizon=int(horizon_total),
    )

    risk = sl_unit
    reward = tp_unit
    rr = reward / risk if risk > 0 else 0
    if context.rr_min_val and rr < context.rr_min_val:
        return None, False
    if context.rr_max_val and rr > context.rr_max_val:
        return None, False

    net_reward = reward - context.ev_deduct_cost if context.has_trading_costs else reward
    net_risk = risk + context.ev_deduct_cost if context.has_trading_costs else risk
    net_rr = net_reward / net_risk if net_risk > 0 else 0.0

    payoffs = barrier_path_payoffs(
        eval_paths,
        path_outcomes,
        entry_price=context.last_price,
        reward=reward,
        risk=risk,
        direction="long" if context.dir_long else "short",
        mode=context.mode_val,  # type: ignore[arg-type]
        tick_size=context.tick_size,
        pip_size=context.pip_size,
        cost_per_trade=(context.ev_deduct_cost if context.has_trading_costs else 0.0),
        same_bar_policy=context.same_bar_policy,  # type: ignore[arg-type]
        gap_aware_stops=context.gap_aware_stops,
    )
    ev_unresolved = float(np.sum(payoffs.terminal_unresolved) / sims_total)
    ev_unresolved_net = float(
        np.sum(
            np.where(
                unresolved_mask,
                payoffs.terminal_unresolved - (
                    context.ev_deduct_cost if context.has_trading_costs else 0.0
                ),
                0.0,
            )
        ) / sims_total
    )
    ev_gross = float(np.mean(payoffs.gross))
    selected_payoffs = payoffs.net if context.has_trading_costs else payoffs.gross
    ev_val = float(np.mean(selected_payoffs))
    ev_se = (
        float(np.std(selected_payoffs, ddof=1) / np.sqrt(sims_total))
        if sims_total > 1
        else 0.0
    )
    ev_ci95 = {
        "low": float(ev_val - 1.96 * ev_se),
        "high": float(ev_val + 1.96 * ev_se),
    }
    ev_resolved_contribution = float(
        np.mean(np.where(payoffs.active, selected_payoffs, 0.0))
    )
    # Neutral ties are excluded from both active and unresolved but still pay
    # configured costs. Keep them in the published EV split.
    same_bar_contribution = 0.0
    if context.same_bar_policy == "neutral" and np.any(ties):
        same_bar_contribution = float(
            np.mean(np.where(ties, selected_payoffs, 0.0))
        )
    edge = effective_prob_win - effective_prob_loss
    edge_variance = max(
        0.0,
        effective_prob_win + effective_prob_loss - edge**2,
    )
    edge_se = float(np.sqrt(edge_variance / sims_total))
    win_lo, win_hi = _binomial_wilson_95(effective_prob_win, int(sims_total))
    loss_lo, loss_hi = _binomial_wilson_95(effective_prob_loss, int(sims_total))
    tie_lo, tie_hi = _binomial_wilson_95(prob_same_bar, int(sims_total))
    no_hit_lo, no_hit_hi = _binomial_wilson_95(prob_no_hit, int(sims_total))

    kelly_val = 0.0
    if net_rr > 0:
        kelly_val = effective_prob_win - (effective_prob_loss / net_rr)

    active = effective_prob_win + effective_prob_loss
    if active > 0 and np.any(payoffs.active):
        prob_win_c = effective_prob_win / active
        prob_loss_c = effective_prob_loss / active
        ev_cond = float(
            np.mean(
                (payoffs.net if context.has_trading_costs else payoffs.gross)[payoffs.active]
            )
        )
        kelly_cond = prob_win_c - (prob_loss_c / net_rr if net_rr > 0 else 0.0)
    else:
        ev_cond = 0.0
        kelly_cond = 0.0

    resolve_mask, time_in_trade = _candidate_resolution_timing(
        first_tp,
        first_sl,
        wins,
        losses,
        ties,
        horizon=horizon_total,
        same_bar_policy=context.same_bar_policy,
    )
    t_res_mean_all = float(np.mean(time_in_trade)) if time_in_trade.size else None
    t_res_med_all = float(np.median(time_in_trade)) if time_in_trade.size else None
    if np.any(resolve_mask):
        resolve_times = np.minimum(first_tp, first_sl)[resolve_mask] + 1
        t_res_mean = float(np.mean(resolve_times)) if resolve_times.size else None
        t_res_med = float(np.median(resolve_times)) if resolve_times.size else None
    else:
        t_res_mean = None
        t_res_med = None

    ev_per_bar = 0.0
    if t_res_mean_all and t_res_mean_all > 0:
        ev_per_bar = ev_val / t_res_mean_all

    profit_factor: Optional[float] = 0.0
    profit_factor_note: Optional[str] = None
    # Realized resolved payoffs, not prob_win*net_TP / prob_loss*net_SL: this
    # respects gap-aware stop fills and the same-bar policy, and excludes
    # unresolved paths entirely.
    active_payoffs = (
        payoffs.net if context.has_trading_costs else payoffs.gross
    )[payoffs.active]
    gross_profit = float(np.sum(active_payoffs[active_payoffs > 0.0]))
    gross_loss = float(-np.sum(active_payoffs[active_payoffs < 0.0]))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = None
        profit_factor_note = "Undefined: no simulated losses for this barrier pair."

    if context.mode_val == 'pct':
        unit_to_return = 0.01
    elif (
        context.mode_val == "pips"
        and context.pip_size is not None
        and context.pip_size > 0.0
        and context.last_price > 0.0
    ):
        unit_to_return = float(context.pip_size) / float(context.last_price)
    elif (
        context.tick_size is not None
        and context.tick_size > 0.0
        and context.last_price > 0.0
    ):
        unit_to_return = float(context.tick_size) / float(context.last_price)
    else:
        unit_to_return = 0.0
    path_returns = (
        payoffs.net if context.has_trading_costs else payoffs.gross
    ) * unit_to_return
    utility_val = float(np.mean(np.log1p(np.maximum(path_returns, -0.999999))))

    if context.min_prob_win_val is not None and effective_prob_win < context.min_prob_win_val:
        return None, False
    if context.max_prob_no_hit_val is not None and prob_no_hit > context.max_prob_no_hit_val:
        return None, False
    if context.min_prob_resolve_val is not None and prob_resolve < context.min_prob_resolve_val:
        return None, False
    if context.max_median_time_val is not None:
        if t_res_med is None or t_res_med > context.max_median_time_val:
            return None, False

    t_hit_tp = first_tp[wins | ties] + 1
    t_hit_sl = first_sl[losses | ties] + 1
    t_tp_med = float(np.median(t_hit_tp)) if t_hit_tp.size else None
    t_sl_med = float(np.median(t_hit_sl)) if t_hit_sl.size else None

    result = {
        "tp": tp_unit,
        "sl": sl_unit,
        "rr": rr,
        "tp_price": float(tp_price),
        "sl_price": float(sl_price),
        "same_bar_policy": context.same_bar_policy,
        **resolved,
        "prob_win": effective_prob_win,
        "prob_loss": effective_prob_loss,
        "prob_win_se": _binomial_se(effective_prob_win, int(sims_total)),
        "prob_loss_se": _binomial_se(effective_prob_loss, int(sims_total)),
        "prob_same_bar_se": _binomial_se(prob_same_bar, int(sims_total)),
        "prob_no_hit_se": _binomial_se(prob_no_hit, int(sims_total)),
        "prob_win_ci95": {"low": float(win_lo), "high": float(win_hi)},
        "prob_loss_ci95": {"low": float(loss_lo), "high": float(loss_hi)},
        "prob_same_bar_ci95": {"low": float(tie_lo), "high": float(tie_hi)},
        "prob_no_hit_ci95": {"low": float(no_hit_lo), "high": float(no_hit_hi)},
        "prob_resolve": prob_resolve,
        "n_sims": int(sims_total),
        "ev": ev_val,
        "ev_se": ev_se,
        "ev_ci95": ev_ci95,
        "ev_including_timeout": ev_val,
        "ev_resolved_contribution": ev_resolved_contribution,
        "timeout_mtm_contribution": ev_unresolved_net,
        "same_bar_contribution": same_bar_contribution,
        "ev_gross": ev_gross if context.has_trading_costs else None,
        "ev_net": ev_val if context.has_trading_costs else None,
        "ev_unresolved": ev_unresolved_net,
        "ev_unresolved_gross": ev_unresolved,
        "ev_unresolved_net": ev_unresolved_net,
        "realized_loss_mean": None,
        "gap_aware_stops": bool(context.gap_aware_stops),
        "ev_cond": ev_cond,
        "edge": edge,
        "edge_se": edge_se,
        "edge_ci95": {
            "low": float(edge - 1.96 * edge_se),
            "high": float(edge + 1.96 * edge_se),
        },
        "kelly": kelly_val,
        "kelly_cond": kelly_cond,
        "ev_per_bar": ev_per_bar,
        "profit_factor": profit_factor,
        "utility": utility_val,
        "t_hit_tp_median": t_tp_med,
        "t_hit_sl_median": t_sl_med,
        "t_hit_tp_median_cond": t_tp_med,
        "t_hit_sl_median_cond": t_sl_med,
        "t_hit_resolve_mean": t_res_mean,
        "t_hit_resolve_median": t_res_med,
        "t_hit_resolve_mean_all": t_res_mean_all,
        "t_hit_resolve_median_all": t_res_med_all,
    }
    realized_loss_mask = losses | (
        ties if context.same_bar_policy == "sl_first" else np.zeros_like(ties)
    )
    if np.any(realized_loss_mask):
        result["realized_loss_mean"] = float(
            np.mean(payoffs.realized_loss_units[realized_loss_mask])
        )
    if profit_factor_note:
        result["profit_factor_note"] = profit_factor_note
    timeout_dominated = bool(
        ev_val > 0.0
        and ev_unresolved_net > 0.0
        and ev_unresolved_net > max(float(ev_resolved_contribution), 0.0)
    )
    result["ev_timeout_dominated"] = timeout_dominated
    if effective_prob_win <= 0.0:
        result["zero_win_probability"] = True
        result["warning"] = (
            "prob_win is 0: no simulated paths reached TP within horizon; "
            "positive ev_including_timeout is timeout mark-to-market, not a resolved win."
        )
    elif effective_prob_win < LOW_PRACTICAL_WIN_PROB_THRESHOLD:
        result["low_practical_win_probability"] = True
        result["warning"] = (
            f"prob_win is below {LOW_PRACTICAL_WIN_PROB_THRESHOLD:.0%}; "
            "treat positive EV as unresolved-path driven unless confirmed."
        )
    elif timeout_dominated:
        result["warning"] = (
            "Positive EV is dominated by timeout mark-to-market, not resolved TP hits."
        )
    _annotate_candidate_metrics(result, cost_per_trade=context.cost_per_trade)
    return result, False


def _evaluate_barrier_bucket(
    bucket: List[Tuple[float, float]],
    eval_paths: np.ndarray,
    *,
    context: _BarrierEvaluationContext,
    bridge_inputs: _BarrierBridgeInputs,
    count_invalid: bool = True,
) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    invalid_candidates = 0
    for tp_unit, sl_unit in bucket:
        row, is_invalid = _evaluate_barrier_candidate(
            tp_unit,
            sl_unit,
            eval_paths,
            context=context,
            bridge_inputs=bridge_inputs,
        )
        if row is not None:
            rows.append(row)
        elif count_invalid and is_invalid:
            invalid_candidates += 1
    return rows, invalid_candidates


def _rounded_ranked_barrier_value(value: Any, decimals: int = 6) -> Any:
    try:
        if value is None:
            return None
        num = float(value)
        if not np.isfinite(num):
            return str(value)
        return round(num, decimals)
    except Exception:
        return value


def _dedupe_ranked_barrier_candidates(
    ranked_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    deduped_ranked: List[Dict[str, Any]] = []
    seen_ranked: Set[Tuple[Any, ...]] = set()
    for row in ranked_candidates:
        if not isinstance(row, dict):
            continue

        row_key = (
            _rounded_ranked_barrier_value(row.get("tp"), 6),
            _rounded_ranked_barrier_value(row.get("sl"), 6),
            _rounded_ranked_barrier_value(row.get("tp_price"), 6),
            _rounded_ranked_barrier_value(row.get("sl_price"), 6),
            _rounded_ranked_barrier_value(row.get("ev"), 6),
            _rounded_ranked_barrier_value(row.get("edge"), 6),
            _rounded_ranked_barrier_value(row.get("kelly"), 6),
            _rounded_ranked_barrier_value(row.get("prob_tp_first"), 6),
            _rounded_ranked_barrier_value(row.get("prob_sl_first"), 6),
            _rounded_ranked_barrier_value(row.get("prob_no_hit"), 6),
        )
        if row_key in seen_ranked:
            continue
        seen_ranked.add(row_key)
        deduped_ranked.append(row)
    return deduped_ranked


def _select_barrier_candidate_views(
    ranked_candidates: List[Dict[str, Any]],
    *,
    cost_per_trade: float,
    viable_only_val: bool,
    tradable_only_val: bool,
    min_ev_val: Optional[float],
    min_edge_val: Optional[float],
    min_kelly_val: Optional[float],
    concise_val: bool,
    top_k_val: Optional[int],
    return_grid: bool,
    output_mode: str,
) -> Dict[str, Any]:
    ranked_candidates = _dedupe_ranked_barrier_candidates(ranked_candidates)
    ranked_before_thresholds = len(ranked_candidates)
    if tradable_only_val or min_ev_val is not None or min_edge_val is not None or min_kelly_val is not None:
        ranked_candidates = [
            row
            for row in ranked_candidates
            if _candidate_passes_threshold_filters(
                row,
                cost_per_trade=cost_per_trade,
                tradable_only_val=tradable_only_val,
                min_ev_val=min_ev_val,
                min_edge_val=min_edge_val,
                min_kelly_val=min_kelly_val,
            )
        ]
    threshold_filtered_out = bool(ranked_before_thresholds and not ranked_candidates)
    viable_candidates = [
        row
        for row in ranked_candidates
        if _candidate_is_viable(row, cost_per_trade=cost_per_trade)
    ]

    candidates = viable_candidates if viable_only_val else ranked_candidates
    if top_k_val is not None:
        candidates = candidates[:top_k_val]
    elif concise_val and not viable_candidates and len(candidates) > 5:
        candidates = candidates[:5]

    grid_out = candidates if (return_grid and not concise_val) else None
    if output_mode == "summary" and grid_out is not None:
        limit = top_k_val or min(10, len(grid_out))
        grid_out = grid_out[:limit]

    results_limit = min(10, len(candidates))
    if output_mode == "summary":
        if top_k_val is not None:
            results_limit = top_k_val
        elif concise_val:
            results_limit = min(5, len(candidates))
        else:
            results_limit = min(10, len(candidates))
    summary_results = candidates[:results_limit]

    viability_filtered_out = bool(viable_only_val and not viable_candidates and ranked_candidates)
    warning = None
    if not candidates:
        if threshold_filtered_out:
            warning = "No TP/SL candidates satisfied the requested threshold filters."
        elif viability_filtered_out:
            warning = "No viable TP/SL candidates satisfied the viability filter."
        else:
            warning = "No valid TP/SL candidates after applying grid generation and constraints."

    return {
        "ranked_candidates": ranked_candidates,
        "viable_candidates": viable_candidates,
        "candidates": candidates,
        "grid_out": grid_out,
        "summary_results": summary_results,
        "viability_filtered_out": viability_filtered_out,
        "warning": warning,
    }


_BARRIER_CONCISE_DROP_KEYS = frozenset(
    {
        "compute_profile",
        "selection_warnings",
        "ev_edge_conflict",
        "ev_edge_conflict_reason",
        "caution",
        "confidence_warning",
        "low_confidence",
        "min_sims_recommended",
        "statistical_robustness",
        "optuna",
        "pareto_front",
        "pareto_count",
        "ensemble",
        "diagnostics",
    }
)


_BARRIER_CONCISE_CANDIDATE_KEYS = (
    "tp",
    "sl",
    "rr",
    "tp_price",
    "sl_price",
    "prob_win",
    "prob_loss",
    "prob_tp_first",
    "prob_sl_first",
    "prob_no_hit",
    "prob_same_bar",
    "prob_unresolved",
    "prob_resolve",
    "breakeven_win_rate",
    "breakeven_win_rate_net",
    "edge",
    "edge_se",
    "edge_ci95",
    "edge_vs_breakeven",
    "ev",
    "ev_se",
    "ev_ci95",
    "ev_simulation_precision_ci95",
    "ev_simulation_comparisons",
    "ev_model_dispersion",
    "edge_model_dispersion",
    "ev_including_timeout",
    "ev_resolved_contribution",
    "timeout_mtm_contribution",
    "same_bar_contribution",
    "ev_unresolved",
    "ev_timeout_dominated",
    "kelly",
    "profit_factor",
    "profit_factor_note",
    "phantom_profit_risk",
    "low_confidence",
    "low_practical_win_probability",
    "gap_aware_stops",
    "realized_loss_mean",
    "member_method",
    "member_method_used",
    "warning",
)


def _compact_barrier_candidate(row: Any) -> Any:
    if not isinstance(row, dict):
        return row
    compact: Dict[str, Any] = {}
    for key in _BARRIER_CONCISE_CANDIDATE_KEYS:
        value = row.get(key)
        if value is None or value == [] or value == {}:
            continue
        compact[key] = value
    return compact


def _resolved_barrier_output_mode(*, output_mode: str, concise_val: bool) -> str:
    if concise_val:
        return "concise"
    return "full" if output_mode == "full" else "summary"


def _minimal_barrier_diagnostics(
    *,
    ranked_candidates: Optional[List[Dict[str, Any]]],
    viable_candidates: Optional[List[Dict[str, Any]]],
    candidates: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    ranked = ranked_candidates or []
    best_any = ranked[0] if ranked and isinstance(ranked[0], dict) else {}
    return {
        "candidates_evaluated": len(ranked),
        "candidates_viable": len(viable_candidates or []),
        "candidates_returned": len(candidates or []),
        "best_ev": _safe_float(best_any.get("ev")) if best_any else None,
        "best_ev_timeout_dominated": bool(best_any.get("ev_timeout_dominated")),
        "best_edge": _safe_float(best_any.get("edge")) if best_any else None,
    }


def _finalize_barrier_output(
    out: Dict[str, Any],
    *,
    output_mode: str,
    concise_val: bool,
    viable_only_val: bool,
    ranked_candidates: Optional[List[Dict[str, Any]]] = None,
    viable_candidates: Optional[List[Dict[str, Any]]] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    grid_out: Optional[List[Dict[str, Any]]] = None,
    return_grid: bool = False,
    top_k_val: Optional[int] = None,
    selection_diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_mode = _resolved_barrier_output_mode(
        output_mode=output_mode,
        concise_val=concise_val,
    )
    out["output_mode"] = resolved_mode
    if viable_only_val:
        out["viable_only"] = True
    if concise_val:
        out["concise"] = True

    if resolved_mode == "full":
        diagnostics_payload: Dict[str, Any] = {
            "candidate_counts": {
                "candidates_evaluated": len(ranked_candidates or []),
                "candidates_viable": len(viable_candidates or []),
                "candidates_returned": len(candidates or []),
                "grid_total": len(grid_out or []),
            },
            "view": {
                "return_grid_requested": bool(return_grid),
                "grid_returned": grid_out is not None,
                "top_k": top_k_val,
                "viable_only": bool(viable_only_val),
                "concise": bool(concise_val),
            },
        }
        if selection_diagnostics:
            diagnostics_payload["selection"] = dict(selection_diagnostics)
        out["diagnostics"] = diagnostics_payload
        return out

    if resolved_mode == "concise":
        for key in _BARRIER_CONCISE_DROP_KEYS:
            out.pop(key, None)
        if isinstance(out.get("best"), dict):
            out["best"] = _compact_barrier_candidate(out["best"])
        if top_k_val is None:
            out.pop("results", None)
        elif isinstance(out.get("results"), list):
            compacted_results = [
                _compact_barrier_candidate(row)
                for row in out["results"]
            ]
            best_compact = out.get("best")
            if (
                len(compacted_results) == 1
                and isinstance(best_compact, dict)
                and compacted_results[0] == best_compact
            ):
                out.pop("results", None)
            else:
                out["results"] = compacted_results
        for key in ("grid", "least_negative"):
            if out.get(key) is None:
                out.pop(key, None)
    if out.get("status") != "ok":
        out["diagnostics"] = _minimal_barrier_diagnostics(
            ranked_candidates=ranked_candidates,
            viable_candidates=viable_candidates,
            candidates=candidates,
        )
    return out


def _cohere_ensemble_probability_row(
    row: Dict[str, Any],
    *,
    cost_per_trade: float,
) -> Dict[str, Any]:
    """Renormalize independently aggregated probabilities and rebuild derived fields."""
    partition_keys = (
        "prob_tp_strict_first",
        "prob_sl_strict_first",
        "prob_same_bar",
        "prob_no_hit",
    )
    raw_values = [_optional_finite_float(row.get(key)) for key in partition_keys]
    if any(value is None for value in raw_values):
        raw_values = [
            _optional_finite_float(row.get("prob_tp_first")),
            _optional_finite_float(row.get("prob_sl_first")),
            _optional_finite_float(row.get("prob_same_bar")),
            _optional_finite_float(row.get("prob_no_hit")),
        ]
        partition_keys = (
            "prob_tp_first",
            "prob_sl_first",
            "prob_same_bar",
            "prob_no_hit",
        )
    if any(value is None for value in raw_values):
        return row
    total = float(sum(raw_values))  # type: ignore[arg-type]
    if total <= 0.0:
        return row
    if abs(total - 1.0) > 1e-6:
        row["probs_incoherent"] = True
        row["probs_raw_sum"] = total
    normalized = [float(value) / total for value in raw_values]  # type: ignore[arg-type]
    if partition_keys[0] == "prob_tp_strict_first":
        tp_strict, sl_strict, same_bar, no_hit = normalized
    else:
        # Already policy-resolved margins; recover a 4-way partition by
        # keeping same-bar mass and treating the rest as strict-first.
        same_bar = normalized[2]
        no_hit = normalized[3]
        remaining = max(0.0, 1.0 - same_bar - no_hit)
        win_loss = normalized[0] + normalized[1]
        if win_loss > 0.0:
            tp_strict = remaining * (normalized[0] / win_loss)
            sl_strict = remaining * (normalized[1] / win_loss)
        else:
            tp_strict = 0.0
            sl_strict = remaining
    policy = str(row.get("same_bar_policy") or "sl_first")
    resolved = resolve_same_bar_probabilities(
        tp_strict=tp_strict,
        sl_strict=sl_strict,
        same_bar=same_bar,
        no_hit=no_hit,
        policy=policy,
    )
    row.update(resolved)
    row["prob_win"] = resolved["prob_tp_first"]
    row["prob_loss"] = resolved["prob_sl_first"]
    row["edge"] = float(resolved["prob_tp_first"] - resolved["prob_sl_first"])
    tp_unit = _optional_finite_float(row.get("tp"))
    sl_unit = _optional_finite_float(row.get("sl"))
    cost = max(0.0, float(cost_per_trade))
    if tp_unit is not None and sl_unit is not None and sl_unit + cost > 0.0:
        net_rr = (tp_unit - cost) / (sl_unit + cost) if cost > 0.0 else (
            tp_unit / sl_unit if sl_unit > 0.0 else 0.0
        )
        if net_rr > 0.0:
            row["kelly"] = float(
                resolved["prob_tp_first"] - (resolved["prob_sl_first"] / net_rr)
            )
        else:
            row["kelly"] = 0.0
    return row


def _combine_ensemble_standard_error(
    values: List[float],
    standard_errors: List[float],
    *,
    weights: Optional[List[float]] = None,
    aggregation: str = "weighted_mean",
) -> Tuple[Optional[float], Optional[float]]:
    """Combine within-model Monte Carlo error with between-model dispersion."""
    samples: List[Tuple[float, float, float]] = []
    raw_weights = weights if weights is not None else [1.0] * len(values)
    for value, standard_error, weight in zip(values, standard_errors, raw_weights):
        value_float = _optional_finite_float(value)
        se_float = _optional_finite_float(standard_error)
        weight_float = _optional_finite_float(weight)
        if (
            value_float is None
            or se_float is None
            or se_float < 0.0
            or weight_float is None
            or weight_float <= 0.0
        ):
            continue
        samples.append((value_float, se_float, weight_float))
    if not samples:
        return None, None

    sample_values = np.asarray([sample[0] for sample in samples], dtype=float)
    sample_ses = np.asarray([sample[1] for sample in samples], dtype=float)
    if aggregation == "weighted_mean":
        normalized_weights = np.asarray(
            [sample[2] for sample in samples],
            dtype=float,
        )
        normalized_weights /= float(normalized_weights.sum())
        center = float(np.sum(normalized_weights * sample_values))
        between_variance = float(
            np.sum(normalized_weights * np.square(sample_values - center))
        )
        within_variance = float(
            np.sum(np.square(normalized_weights * sample_ses))
        )
        dispersion = float(np.sqrt(max(0.0, between_variance)))
    else:
        center = float(np.median(sample_values))
        dispersion = float(
            1.4826 * np.median(np.abs(sample_values - center))
        )
        # Retain a typical member's simulation error. Model disagreement is
        # epistemic and must not disappear merely because more methods agree.
        within_variance = float(np.median(np.square(sample_ses)))

    combined = float(np.sqrt(max(0.0, within_variance + dispersion**2)))
    return combined, dispersion


def _attach_viability_semantics(
    out: Dict[str, Any],
    *,
    mathematically_viable: bool,
    trade_gate_passed: Any,
) -> None:
    tradable = bool(trade_gate_passed)
    out["mathematically_viable"] = bool(mathematically_viable)
    out["tradable"] = tradable
    if bool(mathematically_viable) != tradable:
        out["viability_note"] = (
            "`viable`/`mathematically_viable` means the selected candidate passed "
            "the optimizer's EV screen; `tradable` and `trade_gate_passed` apply "
            "risk/actionability diagnostics."
        )


def forecast_barrier_optimize(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    method: Literal['mc_gbm','mc_gbm_bb','hmm_mc','garch','bootstrap','heston','jump_diffusion','auto','ensemble'] = 'mc_gbm_bb',
    direction: Literal['long','short'] = 'long',
    mode: Literal['pct','ticks','pips'] = 'pct',
    tp_min: float = 0.25,
    tp_max: float = 1.5,
    tp_steps: Optional[int] = None,
    sl_min: float = 0.25,
    sl_max: float = 2.5,
    sl_steps: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
    denoise: Optional[DenoiseSpec] = None,
    as_of: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    objective: Literal[
        'edge',
        'prob_tp_first',
        'prob_resolve',
        'kelly',
        'kelly_cond',
        'ev',
        'ev_cond',
        'ev_per_bar',
        'profit_factor',
        'min_loss_prob',
        'utility',
    ] = 'ev',
    return_grid: bool = True,
    top_k: Optional[int] = None,
    output_mode: Literal['full','summary'] = 'summary',
    viable_only: bool = True,
    concise: bool = False,
    grid_style: Literal['fixed','volatility','ratio','preset'] = 'fixed',
    preset: Optional[str] = None,
    vol_window: int = 250,
    vol_min_mult: float = 0.5,
    vol_max_mult: float = 4.0,
    vol_steps: Optional[int] = None,
    vol_sl_multiplier: float = 1.8,
    vol_floor_pct: float = 0.15,
    vol_floor_ticks: Optional[float] = None,
    ratio_min: float = 0.5,
    ratio_max: float = 4.0,
    ratio_steps: Optional[int] = None,
    refine: Optional[bool] = None,
    refine_radius: float = 0.3,
    refine_steps: int = 5,
    min_prob_win: Optional[float] = None,
    max_prob_no_hit: Optional[float] = None,
    max_median_time: Optional[float] = None,
    fast_defaults: bool = False,
    search_profile: Literal['fast', 'medium', 'long'] = 'medium',
    statistical_robustness: bool = False,
    target_ci_width: float = 0.05,
    n_seeds_stability: int = 3,
    enable_bootstrap: bool = False,
    n_bootstrap: int = 200,
    enable_convergence_check: bool = True,
    convergence_window: int = 100,
    convergence_threshold: float = 0.01,
    enable_power_analysis: bool = False,
    power_effect_size: float = 0.05,
    enable_sensitivity_analysis: bool = False,
    sensitivity_params: Optional[List[str]] = None,
    _prefetched_history: Optional[Any] = None,
) -> Dict[str, Any]:
    """Optimize TP/SL barriers over a grid of candidate levels.

    Unit conventions:
    - mode="pct": tp/sl are percentage *points* (e.g., tp=0.5 means +0.5%).
    - mode="ticks": tp/sl are ticks (trade_tick_size units). Implicit
      fixed/ratio bounds 0.25/1.5/0.25/2.5 are percent points converted like
      the intraday preset unless tp_min/tp_max/sl_min/sl_max are supplied in
      params.

    Grid styles:
    - fixed/volatility generate tp/sl directly in the selected `mode`.
    - preset ranges are defined in pct terms and converted when `mode="ticks"`.
    - ratio treats `ratio_min/max` as reward/risk = tp/sl (TP distance divided
      by SL distance), with SL sampled from `sl_min/max`.

    Metrics:
    - ev/ev_cond/ev_per_bar are reported in the same units as tp/sl (pct points
      or ticks). `ev_per_bar` divides by unconditional mean time-in-trade
      (bars), counting unresolved paths at horizon expiry.
    
    Statistical Robustness:
    - statistical_robustness: Enable comprehensive statistical analysis
    - target_ci_width: Target confidence interval width (default 0.05 = ±2.5%)
    - n_seeds_stability: Number of seeds for cross-seed stability analysis
    - enable_bootstrap: Enable bootstrap uncertainty estimation
    - n_bootstrap: Number of bootstrap samples
    - enable_convergence_check: Check MC convergence diagnostics
    - convergence_window: Window size for convergence check
    - convergence_threshold: Convergence threshold for probability change
    - enable_power_analysis: Statistical power analysis for probabilities
    - power_effect_size: Minimum detectable effect size for power analysis
    - enable_sensitivity_analysis: Sensitivity analysis for barrier parameters
    - sensitivity_params: List of parameters to analyze (default: ['tp', 'sl'])
    """
    try:
        symbol, symbol_requested = resolve_forecast_symbol(symbol)
        if timeframe not in TIMEFRAME_SECONDS:
            return {"error": f"Invalid timeframe: {timeframe}"}
        try:
            horizon_val = int(horizon)
        except Exception:
            return {"error": f"Invalid horizon: {horizon}. Must be a positive integer."}
        if horizon_val <= 0:
            return {"error": f"Invalid horizon: {horizon_val}. Must be >= 1."}
        direction_norm, direction_error = normalize_trade_direction(direction)
        if direction_error:
            return {"error": direction_error}

        params_dict = _parse_kv_or_json(params)
        method_name = normalize_barrier_method(method, allow_ensemble=True)
        if method_name is None:
            return {"error": barrier_method_error(method, allow_ensemble=True)}
        try:
            same_bar_policy_value = normalize_same_bar_policy(
                params_dict.get("same_bar_policy", "sl_first")
            )
        except ValueError as exc:
            return {"error": str(exc)}
        if "profile" in params_dict:
            return {
                "error": (
                    "params.profile is not supported. Use search_profile either as "
                    "the tool parameter or inside params."
                )
            }
        contract_warnings: List[str] = []
        mode_requested = str(mode).lower().strip()
        if mode_requested not in {'pct', 'ticks', 'pips'}:
            return {"error": f"Invalid mode: {mode}. Use 'pct', 'ticks', or 'pips'."}
        mode_val = mode_requested
        output_mode = str(output_mode).strip().lower()
        if output_mode not in {'full', 'summary'}:
            output_mode = 'summary'

        try:
            search_profile_val, profile_cfg = _resolve_barrier_search_profile_config(
                params_dict,
                search_profile=search_profile,
                fast_defaults=fast_defaults,
            )
        except ValueError as exc:
            return {
                "success": False,
                "error": str(exc),
                "error_code": "invalid_argument",
                "valid_values": {"search_profile": ["fast", "medium", "long"]},
                "remediation": "Set search_profile to fast, medium, or long.",
            }

        viable_only_val = _coerce_barrier_bool_flag(
            params_dict.get('viable_only', viable_only),
            default=bool(viable_only),
        )
        tradable_only_val = _coerce_barrier_bool_flag(
            params_dict.get('tradable_only', False),
            default=False,
        )
        if tradable_only_val:
            viable_only_val = True
        concise_val = _coerce_barrier_bool_flag(
            params_dict.get('concise', concise),
            default=bool(concise),
        )
        if concise_val:
            output_mode = 'summary'
        objective_val = str(objective).lower()
        objective_requested = objective_val
        valid_objectives = {
            'edge',
            'prob_tp_first',
            'prob_resolve',
            'kelly',
            'kelly_cond',
            'ev',
            'ev_cond',
            'ev_per_bar',
            'profit_factor',
            'min_loss_prob',
            'utility',
        }
        if objective_val not in valid_objectives:
            objective_val = 'ev'
        objective_changed = objective_val != objective_requested

        optimizer_val = str(params_dict.get('optimizer', 'grid')).strip().lower()
        if optimizer_val not in {'grid', 'optuna'}:
            optimizer_val = 'grid'
        optuna_default_trials = int(profile_cfg['n_trials'])
        optuna_trials_val = max(1, int(params_dict.get('n_trials', optuna_default_trials)))
        optuna_timeout_raw = params_dict.get('timeout')
        try:
            optuna_timeout_val = float(optuna_timeout_raw) if optuna_timeout_raw is not None else None
            if optuna_timeout_val is not None and optuna_timeout_val <= 0:
                optuna_timeout_val = None
        except Exception:
            optuna_timeout_val = None
        optuna_n_jobs_val = max(1, int(params_dict.get('n_jobs', 1)))
        optuna_sampler_val = str(params_dict.get('sampler', 'tpe')).strip().lower()
        if optuna_sampler_val not in {'tpe', 'random', 'cmaes'}:
            optuna_sampler_val = 'tpe'
        optuna_pruner_val = str(params_dict.get('pruner', 'median')).strip().lower()
        if optuna_pruner_val not in {'median', 'none', 'hyperband', 'percentile'}:
            optuna_pruner_val = 'median'
        optuna_pareto_val = _coerce_barrier_bool_flag(params_dict.get('optuna_pareto', False), default=False)
        try:
            pareto_limit_val = int(params_dict.get('pareto_limit', 20))
        except Exception:
            pareto_limit_val = 20
        if pareto_limit_val <= 0:
            pareto_limit_val = 20
        optuna_pareto_objectives_raw = params_dict.get('optuna_pareto_objectives')

        def _normalize_optuna_direction(value: Any, default: str = 'maximize') -> str:
            v = str(value or default).strip().lower()
            if v in {'max', 'maximize', 'maximise'}:
                return 'maximize'
            if v in {'min', 'minimize', 'minimise'}:
                return 'minimize'
            return str(default)

        pareto_objectives: List[Tuple[str, str]] = [
            ('ev', 'maximize'),
            ('prob_loss', 'minimize'),
            ('t_hit_resolve_median', 'minimize'),
        ]
        if isinstance(optuna_pareto_objectives_raw, dict) and optuna_pareto_objectives_raw:
            tmp: List[Tuple[str, str]] = []
            for mk, mv in optuna_pareto_objectives_raw.items():
                metric_name = str(mk).strip()
                if not metric_name:
                    continue
                tmp.append((metric_name, _normalize_optuna_direction(mv, default='maximize')))
            if tmp:
                pareto_objectives = tmp
        valid_pareto_metrics = {
            'edge', 'prob_tp_first', 'prob_loss', 'prob_resolve', 'kelly',
            'kelly_cond', 'ev', 'ev_cond', 'ev_per_bar', 'profit_factor',
            'utility', 't_hit_resolve_mean', 't_hit_resolve_median',
        }
        invalid_pareto_metrics = [
            metric_name
            for metric_name, _ in pareto_objectives
            if metric_name not in valid_pareto_metrics
        ]
        if invalid_pareto_metrics:
            return {
                "error": (
                    "Unsupported Optuna Pareto metric(s): "
                    f"{', '.join(invalid_pareto_metrics)}."
                )
            }

        if top_k is not None:
            try:
                top_k_val = int(top_k)
            except Exception:
                return {"error": f"Invalid top_k: {top_k}. Must be a positive integer."}
            if top_k_val <= 0:
                return {"error": f"Invalid top_k: {top_k_val}. Must be >= 1."}
        else:
            top_k_val = None

        grid_style_val = str(params_dict.get('grid_style', grid_style)).lower()
        if grid_style_val not in {'fixed', 'volatility', 'ratio', 'preset'}:
            grid_style_val = 'fixed'
        preset_candidate = params_dict.get('grid_preset', params_dict.get('preset', preset))
        preset_val = str(preset_candidate).lower() if isinstance(preset_candidate, str) and preset_candidate else None
        if grid_style_val == 'preset' and preset_val is None:
            return {
                "error": (
                    "Barrier grid preset is required when grid_style='preset'. "
                    "Use one of: intraday, position, scalp, swing."
                )
            }
        parameter_error = unknown_mapping_keys_error(
            params_dict,
            _barrier_optimizer_param_keys(method_name),
            subject=f"barrier optimizer params for method '{method_name}'",
        )
        if parameter_error is not None:
            if "return_grid" in parameter_error["unknown_keys"]:
                parameter_error["remediation"] = (
                    "Pass return_grid as the top-level return_grid option; correct or "
                    "remove every other unknown key and use only valid_keys inside params."
                )
            return parameter_error
        if grid_style_val != 'preset' and preset_val is not None:
            return {
                "error": (
                    "Barrier grid preset is only valid when grid_style='preset'. "
                    "Remove preset/grid_preset or select the preset grid style."
                )
            }
        if grid_style_val == 'preset' and preset_val not in BARRIER_GRID_PRESETS:
            return {
                "error": (
                    f"Invalid barrier grid preset: {preset_val}. Use one of: "
                    f"{', '.join(sorted(BARRIER_GRID_PRESETS))}."
                )
            }

        refine_default = _resolve_profile_param(
            params_dict,
            profile_cfg,
            param_key='refine',
            arg_value=refine,
        )
        refine_flag = _coerce_barrier_bool_flag(
            params_dict.get('refine', refine_default),
            default=bool(refine_default),
        )
        refine_radius_val = max(0.0, float(params_dict.get('refine_radius', refine_radius)))
        refine_steps_val = max(2, int(params_dict.get('refine_steps', refine_steps)))

        ratio_min_val = float(params_dict.get('ratio_min', ratio_min))
        ratio_max_val = float(params_dict.get('ratio_max', ratio_max))
        ratio_steps_default = int(
            _resolve_profile_param(
                params_dict,
                profile_cfg,
                param_key='ratio_steps',
                arg_value=ratio_steps,
            )
        )
        ratio_steps_val = max(2, int(params_dict.get('ratio_steps', ratio_steps_default)))
        if ratio_min_val <= 0:
            ratio_min_val = ratio_min
        if ratio_max_val < ratio_min_val:
            ratio_min_val, ratio_max_val = ratio_max_val, ratio_min_val
            contract_warnings.append(
                "ratio_min and ratio_max were swapped so the search range is low-to-high."
            )

        vol_window_val = int(params_dict.get('vol_window', vol_window))
        vol_min_mult_val = float(params_dict.get('vol_min_mult', vol_min_mult))
        vol_max_mult_val = float(params_dict.get('vol_max_mult', vol_max_mult))
        vol_steps_default = int(
            _resolve_profile_param(
                params_dict,
                profile_cfg,
                param_key='vol_steps',
                arg_value=vol_steps,
            )
        )
        vol_steps_val = max(2, int(params_dict.get('vol_steps', vol_steps_default)))
        vol_sl_multiplier_val = float(params_dict.get('vol_sl_multiplier', vol_sl_multiplier))
        vol_sl_steps_val = max(vol_steps_val, int(params_dict.get('vol_sl_steps', vol_steps_val + 2)))
        vol_floor_pct_val = float(params_dict.get('vol_floor_pct', vol_floor_pct))
        vol_floor_ticks_raw = params_dict.get(
            'vol_floor_ticks',
            vol_floor_ticks if vol_floor_ticks is not None else 8.0,
        )
        vol_floor_ticks_val = float(vol_floor_ticks_raw)

        # Optional risk/reward filter applied across all grid styles
        rr_min_val = params_dict.get('rr_min')
        rr_max_val = params_dict.get('rr_max')
        try:
            rr_min_val = float(rr_min_val) if rr_min_val is not None else None
        except Exception:
            rr_min_val = None
        try:
            rr_max_val = float(rr_max_val) if rr_max_val is not None else None
        except Exception:
            rr_max_val = None
        if rr_min_val is not None and rr_min_val <= 0:
            rr_min_val = None
        if rr_max_val is not None and rr_max_val <= 0:
            rr_max_val = None

        min_ev_val = _optional_finite_float(params_dict.get('min_ev'))
        min_edge_val = _optional_finite_float(params_dict.get('min_edge'))
        min_kelly_val = _optional_finite_float(params_dict.get('min_kelly'))

        min_prob_win_val = params_dict.get('min_prob_win', min_prob_win)
        max_prob_no_hit_val = params_dict.get('max_prob_no_hit', max_prob_no_hit)
        max_median_time_val = params_dict.get('max_median_time', max_median_time)
        min_prob_resolve_val = params_dict.get('min_prob_resolve')
        try:
            min_prob_win_val = float(min_prob_win_val) if min_prob_win_val is not None else None
        except Exception:
            min_prob_win_val = None
        try:
            max_prob_no_hit_val = float(max_prob_no_hit_val) if max_prob_no_hit_val is not None else None
        except Exception:
            max_prob_no_hit_val = None
        try:
            max_median_time_val = float(max_median_time_val) if max_median_time_val is not None else None
        except Exception:
            max_median_time_val = None
        try:
            min_prob_resolve_val = float(min_prob_resolve_val) if min_prob_resolve_val is not None else None
        except Exception:
            min_prob_resolve_val = None
        if min_prob_win_val is not None:
            if not np.isfinite(min_prob_win_val):
                min_prob_win_val = None
            else:
                min_prob_win_val = max(0.0, min(1.0, min_prob_win_val))
        if max_prob_no_hit_val is not None:
            if not np.isfinite(max_prob_no_hit_val):
                max_prob_no_hit_val = None
            else:
                max_prob_no_hit_val = max(0.0, min(1.0, max_prob_no_hit_val))
        if max_median_time_val is not None:
            if not np.isfinite(max_median_time_val) or max_median_time_val <= 0:
                max_median_time_val = None
        if min_prob_resolve_val is not None:
            if not np.isfinite(min_prob_resolve_val):
                min_prob_resolve_val = None
            else:
                min_prob_resolve_val = max(0.0, min(1.0, min_prob_resolve_val))
        elif objective_val in {'profit_factor', 'min_loss_prob'}:
            # Only apply the safety floor when the user did not pass a value at
            # all (None). An explicit min_prob_resolve=0.0 is respected as an
            # intentional override.
            min_prob_resolve_val = DEGENERATE_OBJECTIVE_MIN_RESOLVE

        tp_min_val = float(params_dict.get('tp_min', tp_min))
        tp_max_val = float(params_dict.get('tp_max', tp_max))
        tp_steps_default = int(
            _resolve_profile_param(
                params_dict,
                profile_cfg,
                param_key='tp_steps',
                arg_value=tp_steps,
            )
        )
        tp_steps_val = max(1, int(params_dict.get('tp_steps', tp_steps_default)))
        sl_min_val = float(params_dict.get('sl_min', sl_min))
        sl_max_val = float(params_dict.get('sl_max', sl_max))
        sl_steps_default = int(
            _resolve_profile_param(
                params_dict,
                profile_cfg,
                param_key='sl_steps',
                arg_value=sl_steps,
            )
        )
        sl_steps_val = max(1, int(params_dict.get('sl_steps', sl_steps_default)))
        
        statistical_robustness_requested = _coerce_barrier_bool_flag(
            params_dict.get('statistical_robustness', statistical_robustness),
            default=bool(statistical_robustness),
        )
        target_ci_width_val = float(params_dict.get('target_ci_width', target_ci_width))
        if not 0 < target_ci_width_val < 1:
            target_ci_width_val = 0.05
        n_seeds_stability_val = max(2, int(params_dict.get('n_seeds_stability', n_seeds_stability)))
        enable_bootstrap_val = _coerce_barrier_bool_flag(
            params_dict.get('enable_bootstrap', enable_bootstrap),
            default=bool(enable_bootstrap),
        )
        n_bootstrap_val = max(50, int(params_dict.get('n_bootstrap', n_bootstrap)))
        enable_convergence_check_val = _coerce_barrier_bool_flag(
            params_dict.get('enable_convergence_check', enable_convergence_check),
            default=bool(enable_convergence_check),
        )
        convergence_window_val = max(10, int(params_dict.get('convergence_window', convergence_window)))
        convergence_threshold_val = float(params_dict.get('convergence_threshold', convergence_threshold))
        if convergence_threshold_val <= 0:
            convergence_threshold_val = 0.01
        enable_power_analysis_val = _coerce_barrier_bool_flag(
            params_dict.get('enable_power_analysis', enable_power_analysis),
            default=bool(enable_power_analysis),
        )
        power_effect_size_val = float(params_dict.get('power_effect_size', power_effect_size))
        if power_effect_size_val <= 0:
            power_effect_size_val = 0.05
        enable_sensitivity_analysis_val = _coerce_barrier_bool_flag(
            params_dict.get('enable_sensitivity_analysis', enable_sensitivity_analysis),
            default=bool(enable_sensitivity_analysis),
        )
        sensitivity_params_requested = params_dict.get('sensitivity_params', sensitivity_params)
        if isinstance(sensitivity_params_requested, str):
            sensitivity_params_requested = [
                p.strip().lower() for p in sensitivity_params_requested.split(',') if p.strip()
            ]
        elif not isinstance(sensitivity_params_requested, list):
            sensitivity_params_requested = ['tp', 'sl']
        else:
            sensitivity_params_requested = [
                str(p).strip().lower() for p in sensitivity_params_requested if str(p).strip()
            ]
        enable_drift_stress_val = _coerce_barrier_bool_flag(
            params_dict.get('enable_drift_stress', statistical_robustness_requested),
            default=bool(statistical_robustness_requested),
        )
        drift_multipliers_raw = params_dict.get(
            'drift_stress_multipliers',
            [0.0, 0.5, 1.0, 1.5],
        )
        if not isinstance(drift_multipliers_raw, (list, tuple)):
            drift_multipliers_raw = [0.0, 0.5, 1.0, 1.5]
        drift_multipliers_val = sorted({
            float(value)
            for value in drift_multipliers_raw
            if _optional_finite_float(value) is not None
        })
        enable_oos_validation_val = _coerce_barrier_bool_flag(
            params_dict.get('enable_oos_validation', False),
            default=False,
        )
        if enable_oos_validation_val and grid_style_val == 'volatility':
            return {
                "error": (
                    "Walk-forward OOS validation currently requires fixed, ratio, "
                    "or preset grids so candidate generation cannot use holdout volatility."
                )
            }
        oos_folds_val = max(2, min(10, int(params_dict.get('oos_folds', 5))))
        oos_sims_requested = max(100, int(params_dict.get('oos_n_sims', 1000)))
        oos_holdout_bars_val = max(
            horizon_val * oos_folds_val,
            int(params_dict.get('oos_holdout_bars', 250)),
        )

        if _prefetched_history is not None:
            try:
                df = _prefetched_history.copy()
            except Exception:
                df = _prefetched_history
        else:
            need = int(max(2000, horizon_val + 100))
            history_kwargs: Dict[str, Any] = {"as_of": as_of}
            if not as_of and (start or end):
                history_kwargs.update({"start": start, "end": end})
            df = _fetch_history(symbol, timeframe, need, **history_kwargs)
        if len(df) < 10:
            return {"error": "Insufficient history for simulation"}
        freshness_context = _history_freshness_context(
            df,
            timeframe,
            symbol=symbol,
        )
        use_live_price_raw = params_dict.get('use_live_price', params_dict.get('live_price', True))
        if isinstance(use_live_price_raw, str):
            use_live_price = use_live_price_raw.strip().lower() not in {"0", "false", "no", "off"}
        else:
            use_live_price = bool(use_live_price_raw)
        if as_of or start or end:
            use_live_price = False
        last_price_close, last_price, last_price_source, _price_warning, price_error = _resolve_reference_prices(
            df['close'].astype(float).to_numpy(),
            symbol=symbol,
            direction=direction_norm,
            use_live_price=use_live_price,
            live_price_getter=_get_live_reference_price,
        )
        if price_error:
            return {"error": price_error}
        reference_context = (
            _live_reference_time_context(symbol, timeframe)
            if str(last_price_source or "").startswith("live_tick")
            else {}
        )
        simulation_reference_price = _barrier_exit_quote_reference(
            last_price,
            direction=direction_norm,
            reference_context=reference_context,
        )
        price_precision = _symbol_price_precision(symbol)

        tick_size = _get_tick_size(symbol)
        pip_size = _get_pip_size(symbol)
        if mode_val == 'ticks' and (tick_size is None or tick_size <= 0):
            return {"error": "Tick size unavailable for this symbol; use mode='pct' or provide absolute barriers. ticks is the broker trade tick/point, not FX pips."}
        if mode_val == 'pips' and (pip_size is None or pip_size <= 0):
            return {
                "error": (
                    "Pip size unavailable for this symbol; unit/mode=pips requires "
                    "an identifiable FX pair. Use mode='pct' or mode='ticks'."
                )
            }

        base_col = 'close'
        if denoise:
            try:
                normalized = _denoise_api.normalize_denoise_spec(denoise, default_when='pre_ti') or denoise
                added = _denoise_api.apply_denoise(df, normalized)
                base_col = _denoise_api.effective_denoise_base_col(
                    df,
                    normalized if isinstance(normalized, dict) else denoise,
                    base_col='close',
                    added_columns=added,
                )
            except Exception as ex:
                contract_warnings.append(
                    f"Denoise request failed; using raw close prices instead: {ex}"
                )
        prices = df[base_col].astype(float).to_numpy()

        sims_default = int(profile_cfg['n_sims'])
        raw_sims = params_dict.get('n_sims', params_dict.get('sims', None))
        if raw_sims is None:
            sims = int(sims_default)
        else:
            try:
                sims = int(raw_sims)
            except Exception:
                return {"error": f"Invalid n_sims: {raw_sims}. Must be >= 1."}
        if sims <= 0:
            return {"error": f"Invalid n_sims: {sims}. Must be >= 1."}
        n_seeds = int(params_dict.get('n_seeds', 1) or 1)
        if n_seeds <= 0:
            return {"error": f"Invalid n_seeds: {n_seeds}. Must be >= 1."}
        
        if statistical_robustness_requested:
            min_sims_recommended = _min_sims_for_ci(
                target_width=target_ci_width_val,
                expected_prob=0.5,
                confidence=0.95,
                conservative=True,
            )
            if sims * n_seeds < min_sims_recommended:
                sims = int(np.ceil(min_sims_recommended / n_seeds))
        oos_sims_val = min(int(sims), int(oos_sims_requested))

        def _cost_param_float(key: str) -> float:
            raw = params_dict.get(key, 0.0)
            try:
                value = float(raw or 0.0)
            except Exception as ex:
                raise ValueError(f"{key} must be numeric, finite, and >= 0.") from ex
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{key} must be numeric, finite, and >= 0.")
            return value

        spread_cost_keys = ('spread_pips', 'spread_bps', 'spread_pct')
        commission_cost_keys = ('commission_bps', 'commission_pct')
        slippage_cost_keys = ('slippage_pips', 'slippage_bps', 'slippage_pct')

        def _cost_family_supplied(keys: Tuple[str, ...]) -> bool:
            return any(key in params_dict and params_dict.get(key) is not None for key in keys)

        spread_supplied = _cost_family_supplied(spread_cost_keys)
        commission_supplied = _cost_family_supplied(commission_cost_keys)
        slippage_supplied = _cost_family_supplied(slippage_cost_keys)
        spread_pips_val = _cost_param_float('spread_pips')
        spread_bps_val = _cost_param_float('spread_bps')
        spread_pct_val = _cost_param_float('spread_pct') + spread_bps_val / 100.0
        commission_bps_val = _cost_param_float('commission_bps')
        commission_pct_val = _cost_param_float('commission_pct') + commission_bps_val / 100.0
        slippage_pips_val = _cost_param_float('slippage_pips')
        slippage_bps_val = _cost_param_float('slippage_bps')
        slippage_pct_val = _cost_param_float('slippage_pct') + slippage_bps_val / 100.0
        spread_source = "explicit_params" if spread_supplied else "missing"
        live_spread_pct = _safe_float(reference_context.get("reference_spread_pct"))
        if (
            not spread_supplied
            and str(last_price_source or "").startswith("live_tick")
            and reference_context.get("reference_usable_for_live") is True
            and live_spread_pct is not None
            and live_spread_pct > 0.0
        ):
            spread_pct_val = float(live_spread_pct)
            spread_source = "live_bid_ask"
        spread_available = spread_supplied or spread_source == "live_bid_ask"
        missing_cost_assumptions = [
            name
            for name, supplied in (
                ("spread", spread_available),
                ("commission", commission_supplied),
                ("slippage", slippage_supplied),
            )
            if not supplied
        ]
        cost_model_complete = not missing_cost_assumptions
        cost_pip_size = pip_size
        if (
            spread_pips_val > 0.0 or slippage_pips_val > 0.0
        ) and cost_pip_size is None:
            return {
                "error": (
                    "Pip-denominated trading costs require an identifiable FX symbol; "
                    "use spread_bps/slippage_bps or percentage costs for this instrument."
                )
            }

        pip_to_pct = (
            float(cost_pip_size) / last_price * 100.0
            if cost_pip_size and last_price > 0
            else 0.0
        )
        effective_spread_pct = spread_pct_val + spread_pips_val * pip_to_pct
        effective_slippage_pct = slippage_pct_val + slippage_pips_val * pip_to_pct
        effective_commission_pct = commission_pct_val

        # Explicit spread assumptions are modeled as a payoff deduction from
        # entry-anchored paths. The live bid/ask fallback instead starts paths
        # on the executable exit quote, so that spread must not also be
        # subtracted from every payoff.
        if spread_supplied and str(last_price_source or "").startswith("live_tick"):
            simulation_reference_price = float(last_price)
        spread_embedded_in_path_geometry = bool(
            spread_source == "live_bid_ask"
            and np.isfinite(float(simulation_reference_price))
            and not np.isclose(
                float(simulation_reference_price),
                float(last_price),
                rtol=0.0,
                atol=max(abs(float(last_price)) * 1e-12, 1e-12),
            )
        )

        def _pct_to_pips(value: float) -> Optional[float]:
            if cost_pip_size is None or last_price <= 0.0:
                return None
            return float(value) / 100.0 * last_price / float(cost_pip_size)

        if mode_val == 'pct':
            cost_spread = effective_spread_pct
            cost_slippage = effective_slippage_pct
            cost_commission = effective_commission_pct
        elif mode_val == 'pips':
            cost_spread = _pct_to_pips(effective_spread_pct) or 0.0
            cost_slippage = _pct_to_pips(effective_slippage_pct) or 0.0
            cost_commission = _pct_to_pips(effective_commission_pct) or 0.0
        else:
            # Convert all costs to the tick units used by barrier metrics.
            pct_to_ticks = (last_price / float(tick_size) / 100.0) if (tick_size and tick_size > 0 and last_price > 0) else 0.0
            pips_to_ticks = (
                float(cost_pip_size) / float(tick_size)
                if cost_pip_size and tick_size and tick_size > 0
                else 0.0
            )
            cost_spread = spread_pips_val * pips_to_ticks + spread_pct_val * pct_to_ticks
            cost_slippage = slippage_pips_val * pips_to_ticks + slippage_pct_val * pct_to_ticks
            cost_commission = commission_pct_val * pct_to_ticks
        dir_long = (direction_norm == 'long')

        modeled_cost_per_trade = max(
            0.0,
            cost_spread + cost_slippage + cost_commission,
        )
        payoff_spread_cost = 0.0 if spread_embedded_in_path_geometry else cost_spread
        # Only costs not already represented by executable quote geometry are
        # deducted from path payoffs.
        cost_per_trade = max(
            0.0,
            payoff_spread_cost + cost_slippage + cost_commission,
        )
        ev_deduct_cost = cost_per_trade
        has_trading_costs = bool(
            modeled_cost_per_trade > 0.0
            or spread_supplied
            or commission_supplied
            or slippage_supplied
        )
        trading_costs_payload = {
            "complete": cost_model_complete,
            "missing_assumptions": missing_cost_assumptions,
            "known_costs_applied": has_trading_costs,
            "cost_per_trade": _safe_float(modeled_cost_per_trade),
            "payoff_deduction": _safe_float(cost_per_trade),
            "cost_unit": mode_val,
            "spread_source": spread_source,
            "spread_embedded_in_path_geometry": spread_embedded_in_path_geometry,
            "spread_values_basis": "effective_normalized_total",
            "spread_pips": _safe_float(_pct_to_pips(effective_spread_pct)),
            "spread_bps": _safe_float(effective_spread_pct * 100.0),
            "spread_pct": _safe_float(effective_spread_pct),
            "commission_source": (
                "explicit_params" if commission_supplied else "missing"
            ),
            "commission_values_basis": "effective_normalized_total",
            "commission_bps": _safe_float(effective_commission_pct * 100.0),
            "commission_pct": _safe_float(effective_commission_pct),
            "slippage_source": (
                "explicit_params" if slippage_supplied else "missing"
            ),
            "slippage_values_basis": "effective_normalized_total",
            "slippage_pips": _safe_float(_pct_to_pips(effective_slippage_pct)),
            "slippage_bps": _safe_float(effective_slippage_pct * 100.0),
            "slippage_pct": _safe_float(effective_slippage_pct),
            "explicit_inputs": {
                key: _safe_float(params_dict.get(key))
                for key in (
                    *spread_cost_keys,
                    *commission_cost_keys,
                    *slippage_cost_keys,
                )
                if key in params_dict and params_dict.get(key) is not None
            },
        }

        # Minimum barrier constraints
        min_barrier_multiplier = float(params_dict.get('min_barrier_multiplier', 2.0) if params_dict.get('min_barrier_multiplier') is not None else 2.0)
        min_barrier_key_by_mode = {
            'pct': 'min_barrier_pct',
            'pips': 'min_barrier_pips',
            'ticks': 'min_barrier_ticks',
        }
        min_barrier_key = min_barrier_key_by_mode.get(mode_val, 'min_barrier_ticks')
        # A minimum barrier distance is only comparable to barriers in the same
        # unit. Silently ignoring a mismatched key let callers believe a floor
        # was enforced when it was not.
        mismatched_min_barrier_keys = sorted(
            key
            for key in min_barrier_key_by_mode.values()
            if key != min_barrier_key and params_dict.get(key) is not None
        )
        if mismatched_min_barrier_keys:
            return {
                "success": False,
                "error": (
                    f"{', '.join(mismatched_min_barrier_keys)} cannot be used with "
                    f"mode={mode_val}. Use {min_barrier_key} so the minimum "
                    "barrier distance matches the barrier unit."
                ),
                "error_code": "invalid_input",
                "mode": mode_val,
                "distance_unit": mode_val,
                "expected_param": min_barrier_key,
                "remediation": (
                    f'Pass --params "{min_barrier_key}=…" or switch --mode to match '
                    "the unit you intended."
                ),
            }
        min_barrier_absolute = float(params_dict.get(min_barrier_key, 0.0) or 0.0)
        # Minimum barrier distance must exceed total round-trip cost (spread +
        # slippage + commission), not just spread — otherwise setups that are
        # structurally negative-EV after slippage/commission can slip through.
        # Use the modeled total rather than the payoff deduction: when the live
        # bid/ask fallback embeds spread in the path geometry the deduction
        # excludes spread, which would drop it from this floor entirely.
        min_barrier_distance = max(
            min_barrier_absolute,
            min_barrier_multiplier * modeled_cost_per_trade,
        )
        tp_min_val, tp_max_val, sl_min_val, sl_max_val = (
            _apply_implicit_tick_mode_grid_defaults(
                mode_val=mode_val,
                params_dict=params_dict,
                tp_min=tp_min,
                tp_max=tp_max,
                sl_min=sl_min,
                sl_max=sl_max,
                tp_min_val=tp_min_val,
                tp_max_val=tp_max_val,
                sl_min_val=sl_min_val,
                sl_max_val=sl_max_val,
                last_price=float(last_price),
                tick_size=_optional_finite_float(tick_size),
                pip_size=_optional_finite_float(pip_size),
            )
        )

        method_requested = method_name
        auto_reason = None
        supported_member_methods = ['mc_gbm', 'mc_gbm_bb', 'hmm_mc', 'garch', 'bootstrap', 'heston', 'jump_diffusion', 'auto']

        if method_name == 'ensemble':
            ensemble_methods_raw = params_dict.get('ensemble_methods', ['hmm_mc', 'garch', 'heston', 'jump_diffusion'])
            ensemble_methods: List[str] = []
            if isinstance(ensemble_methods_raw, str):
                ensemble_methods = [p.strip().lower() for p in ensemble_methods_raw.split(',') if p.strip()]
            elif isinstance(ensemble_methods_raw, (list, tuple)):
                for item in ensemble_methods_raw:
                    if isinstance(item, str) and item.strip():
                        ensemble_methods.append(item.strip().lower())
            if not ensemble_methods:
                ensemble_methods = ['hmm_mc', 'garch', 'heston', 'jump_diffusion']
            dedup_members: List[str] = []
            seen_members: Set[str] = set()
            for member_name in ensemble_methods:
                if member_name == 'ensemble':
                    continue
                if member_name not in supported_member_methods:
                    continue
                if member_name in seen_members:
                    continue
                seen_members.add(member_name)
                dedup_members.append(member_name)
            ensemble_methods = dedup_members
            if not ensemble_methods:
                return {"error": "Ensemble requires at least one valid member method."}

            ensemble_agg = str(params_dict.get('ensemble_agg', 'median')).strip().lower()
            if ensemble_agg not in {'median', 'weighted_mean'}:
                ensemble_agg = 'median'

            weight_map_raw = params_dict.get('ensemble_weights')
            ensemble_weight_map: Dict[str, float] = {}
            if isinstance(weight_map_raw, dict):
                for mk, mv in weight_map_raw.items():
                    try:
                        w = float(mv)
                    except Exception:
                        continue
                    if not np.isfinite(w) or w <= 0:
                        continue
                    ensemble_weight_map[str(mk).strip().lower()] = float(w)

            member_params = dict(params_dict)
            for extra_key in (
                'ensemble_methods',
                'ensemble_agg',
                'ensemble_weights',
                'ensemble_top_k',
                'ensemble_vote_metric',
                'tradable_only',
                'min_ev',
                'min_edge',
                'min_kelly',
            ):
                member_params.pop(extra_key, None)

            member_runs: List[Dict[str, Any]] = []
            member_errors: List[Dict[str, Any]] = []
            for member_method in ensemble_methods:
                member_out = forecast_barrier_optimize(
                    symbol=symbol,
                    timeframe=timeframe,
                    horizon=horizon_val,
                    method=member_method,
                    direction=direction_norm,  # type: ignore[arg-type]
                    mode=mode_val,  # type: ignore[arg-type]
                    tp_min=tp_min_val,
                    tp_max=tp_max_val,
                    tp_steps=tp_steps_val,
                    sl_min=sl_min_val,
                    sl_max=sl_max_val,
                    sl_steps=sl_steps_val,
                    params=member_params,
                    denoise=denoise,
                    objective=objective_val,  # type: ignore[arg-type]
                    return_grid=True,
                    top_k=None,
                    output_mode='full',  # type: ignore[arg-type]
                    viable_only=False,
                    concise=False,
                    grid_style=grid_style_val,  # type: ignore[arg-type]
                    preset=preset_val,
                    vol_window=vol_window_val,
                    vol_min_mult=vol_min_mult_val,
                    vol_max_mult=vol_max_mult_val,
                    vol_steps=vol_steps_val,
                    vol_sl_multiplier=vol_sl_multiplier_val,
                    vol_floor_pct=vol_floor_pct_val,
                    vol_floor_ticks=vol_floor_ticks_val,
                    ratio_min=ratio_min_val,
                    ratio_max=ratio_max_val,
                    ratio_steps=ratio_steps_val,
                    refine=False,
                    refine_radius=refine_radius_val,
                    refine_steps=refine_steps_val,
                    min_prob_win=min_prob_win_val,
                    max_prob_no_hit=max_prob_no_hit_val,
                    max_median_time=max_median_time_val,
                    fast_defaults=bool(search_profile_val == 'fast'),
                    search_profile=search_profile_val,  # type: ignore[arg-type]
                    statistical_robustness=False,
                    target_ci_width=target_ci_width_val,
                    n_seeds_stability=n_seeds_stability_val,
                    enable_bootstrap=bool(enable_bootstrap_val),
                    n_bootstrap=n_bootstrap_val,
                    enable_convergence_check=bool(enable_convergence_check_val),
                    convergence_window=convergence_window_val,
                    convergence_threshold=convergence_threshold_val,
                    enable_power_analysis=bool(enable_power_analysis_val),
                    power_effect_size=power_effect_size_val,
                    enable_sensitivity_analysis=bool(enable_sensitivity_analysis_val),
                    sensitivity_params=sensitivity_params_requested,
                    _prefetched_history=df,
                )
                if not isinstance(member_out, dict) or not member_out.get('success'):
                    err_msg = None
                    if isinstance(member_out, dict):
                        err_msg = member_out.get('error')
                    if err_msg is None:
                        err_msg = f"Member method {member_method} failed"
                    member_errors.append({"method": member_method, "error": str(err_msg)})
                    continue
                member_grid = member_out.get('grid')
                if not isinstance(member_grid, list) or not member_grid:
                    member_errors.append({"method": member_method, "error": "No candidate grid returned"})
                    continue
                best_row = member_out.get('best')
                actual_best = dict(best_row) if isinstance(best_row, dict) else {}
                actual_best["member_method"] = str(member_method)
                actual_best["member_method_used"] = str(member_out.get('method', member_method))
                _annotate_candidate_metrics(actual_best, cost_per_trade=cost_per_trade)
                member_runs.append({
                    "method": member_method,
                    "method_used": member_out.get('method', member_method),
                    "best": actual_best,
                    "grid": [dict(row) for row in member_grid if isinstance(row, dict)],
                    "output": member_out,
                })

            if not member_runs:
                return {"error": "Ensemble failed: no successful member methods.", "member_errors": member_errors}

            n_total = len(ensemble_methods)
            n_succeeded = len(member_runs)
            n_failed = len(member_errors)
            ensemble_degraded = n_failed > n_total / 2
            ensemble_confidence = "high" if n_failed == 0 else ("medium" if n_succeeded > n_failed else "low")

            metric_keys = [
                'prob_win', 'prob_loss', 'prob_tp_first', 'prob_sl_first',
                'prob_tp_strict_first', 'prob_sl_strict_first',
                'prob_no_hit', 'prob_same_bar', 'prob_resolve', 'prob_unresolved',
                'ev', 'ev_gross', 'ev_net', 'ev_unresolved',
                'ev_unresolved_gross', 'ev_unresolved_net',
                'ev_resolved_contribution', 'timeout_mtm_contribution',
                'same_bar_contribution', 'ev_cond', 'edge',
                'kelly', 'kelly_cond',
                'ev_per_bar', 'profit_factor', 'utility',
                't_hit_tp_median', 't_hit_sl_median',
                't_hit_resolve_mean', 't_hit_resolve_median',
                't_hit_resolve_mean_all', 't_hit_resolve_median_all',
            ]

            def _member_weight(row: Dict[str, Any]) -> float:
                member_key = str(row.get('method', '')).strip().lower()
                if member_key in ensemble_weight_map:
                    return float(ensemble_weight_map[member_key])
                return 1.0

            def _agg_metric(
                member_rows: List[Tuple[Dict[str, Any], Dict[str, Any]]],
                metric_name: str,
            ) -> Optional[float]:
                vals: List[float] = []
                wts: List[float] = []
                for member_run, candidate_row in member_rows:
                    raw = candidate_row.get(metric_name)
                    try:
                        val = float(raw)
                    except Exception:
                        continue
                    if not np.isfinite(val):
                        continue
                    vals.append(float(val))
                    wts.append(_member_weight(member_run))
                if not vals:
                    return None
                if ensemble_agg == 'weighted_mean':
                    sw = float(sum(wts))
                    if sw > 0:
                        return float(sum(v * w for v, w in zip(vals, wts)) / sw)
                    return float(np.mean(np.asarray(vals, dtype=float)))
                return float(np.median(np.asarray(vals, dtype=float)))

            grouped_candidates: Dict[
                Tuple[int, int],
                List[Tuple[Dict[str, Any], Dict[str, Any]]],
            ] = {}
            for member_run in member_runs:
                for candidate_row in member_run.get('grid', []):
                    try:
                        key = (
                            int(round(float(candidate_row['tp']) * 1e9)),
                            int(round(float(candidate_row['sl']) * 1e9)),
                        )
                    except Exception:
                        continue
                    grouped_candidates.setdefault(key, []).append(
                        (member_run, candidate_row)
                    )

            ranked_candidates: List[Dict[str, Any]] = []
            for member_rows in grouped_candidates.values():
                if len(member_rows) != n_succeeded:
                    continue
                first_row = member_rows[0][1]
                aggregate_row: Dict[str, Any] = {
                    'tp': float(first_row['tp']),
                    'sl': float(first_row['sl']),
                    'rr': float(first_row['rr']),
                    'tp_price': float(first_row['tp_price']),
                    'sl_price': float(first_row['sl_price']),
                    'same_bar_policy': first_row.get('same_bar_policy'),
                    'ensemble_member_count': int(len(member_rows)),
                    'ensemble_methods': [
                        str(member_run.get('method'))
                        for member_run, _ in member_rows
                    ],
                    'member_metrics': {
                        str(member_run.get('method')): {
                            metric_name: candidate_row.get(metric_name)
                            for metric_name in (
                                'ev', 'edge', 'prob_tp_first', 'prob_sl_first',
                                'prob_no_hit', 'kelly', 'utility', 'ev_se', 'edge_se',
                            )
                        }
                        for member_run, candidate_row in member_rows
                    },
                }
                for metric_name in metric_keys:
                    value = _agg_metric(member_rows, metric_name)
                    if value is not None:
                        aggregate_row[metric_name] = value
                _cohere_ensemble_probability_row(
                    aggregate_row,
                    cost_per_trade=cost_per_trade,
                )
                aggregate_ev = _optional_finite_float(aggregate_row.get("ev"))
                aggregate_timeout = _optional_finite_float(
                    aggregate_row.get("timeout_mtm_contribution")
                )
                aggregate_resolved = _optional_finite_float(
                    aggregate_row.get("ev_resolved_contribution")
                )
                aggregate_row["ev_timeout_dominated"] = bool(
                    aggregate_ev is not None
                    and aggregate_ev > 0.0
                    and aggregate_timeout is not None
                    and aggregate_timeout > 0.0
                    and aggregate_timeout > max(float(aggregate_resolved or 0.0), 0.0)
                )
                for metric_name, se_name in (("ev", "ev_se"), ("edge", "edge_se")):
                    metric_values: List[float] = []
                    standard_errors: List[float] = []
                    metric_weights: List[float] = []
                    for member_run, candidate_row in member_rows:
                        metric_value = _optional_finite_float(
                            candidate_row.get(metric_name)
                        )
                        standard_error = _optional_finite_float(
                            candidate_row.get(se_name)
                        )
                        if metric_value is None or standard_error is None:
                            continue
                        metric_values.append(metric_value)
                        standard_errors.append(standard_error)
                        metric_weights.append(_member_weight(member_run))
                    combined_se, model_dispersion = _combine_ensemble_standard_error(
                        metric_values,
                        standard_errors,
                        weights=metric_weights,
                        aggregation=ensemble_agg,
                    )
                    if combined_se is not None:
                        aggregate_row[se_name] = combined_se
                    if model_dispersion is not None:
                        aggregate_row[f"{metric_name}_model_dispersion"] = model_dispersion
                _annotate_candidate_metrics(
                    aggregate_row,
                    cost_per_trade=cost_per_trade,
                )
                ranked_candidates.append(aggregate_row)

            if not ranked_candidates:
                return {
                    "error": "Ensemble failed: member methods produced no common TP/SL candidates.",
                    "member_errors": member_errors,
                }
            ensemble_candidates_evaluated = len(ranked_candidates)
            _sort_candidate_results(ranked_candidates, objective_val)
            ensemble_views = _select_barrier_candidate_views(
                ranked_candidates,
                cost_per_trade=cost_per_trade,
                viable_only_val=viable_only_val,
                tradable_only_val=tradable_only_val,
                min_ev_val=min_ev_val,
                min_edge_val=min_edge_val,
                min_kelly_val=min_kelly_val,
                concise_val=concise_val,
                top_k_val=top_k_val,
                return_grid=return_grid,
                output_mode=output_mode,
            )
            ranked_candidates = ensemble_views['ranked_candidates']
            viable_candidates = ensemble_views['viable_candidates']
            candidates = ensemble_views['candidates']
            grid_out = ensemble_views['grid_out']
            summary_results = ensemble_views['summary_results']

            member_prices = [
                float(r.get('output', {}).get('last_price'))
                for r in member_runs
                if isinstance(r.get('output', {}).get('last_price'), (int, float))
            ]
            member_close_prices = [
                float(r.get('output', {}).get('last_price_close'))
                for r in member_runs
                if isinstance(r.get('output', {}).get('last_price_close'), (int, float))
            ]
            out_last_price = float(np.median(np.asarray(member_prices, dtype=float))) if member_prices else float(last_price)
            out_last_price_close = (
                float(np.median(np.asarray(member_close_prices, dtype=float)))
                if member_close_prices else float(last_price_close)
            )

            viability_filtered_out = bool(ensemble_views['viability_filtered_out'])
            ensemble_warning = ensemble_views.get('warning')
            selected_best = candidates[0] if candidates else None
            if isinstance(selected_best, dict):
                _annotate_candidate_metrics(selected_best, cost_per_trade=cost_per_trade)
            viable = _candidate_is_viable(selected_best, cost_per_trade=cost_per_trade)
            candidates_viable = int(len(viable_candidates))
            status = "ok" if viable else ("non_viable" if viability_filtered_out else ("no_candidates" if not selected_best else "non_viable"))
            status_reason = None
            if viability_filtered_out:
                status_reason = "No viable ensemble candidate satisfied the viability filter."
            elif status == "no_candidates":
                status_reason = "No valid ensemble candidate was produced."
            elif status == "non_viable":
                status_reason = _candidate_status_reason(
                    selected_best,
                    cost_per_trade=cost_per_trade,
                )

            member_summaries: List[Dict[str, Any]] = []
            selected_member_metrics = (
                selected_best.get('member_metrics', {})
                if isinstance(selected_best, dict)
                else {}
            )
            for row in member_runs:
                member_method = row.get('method')
                method_metrics = selected_member_metrics.get(str(member_method), {})
                member_summaries.append({
                    "method": member_method,
                    "method_used": row.get('method_used'),
                    "ev": method_metrics.get('ev'),
                    "prob_win": method_metrics.get('prob_tp_first'),
                    "prob_loss": method_metrics.get('prob_sl_first'),
                    "prob_no_hit": method_metrics.get('prob_no_hit'),
                    "edge": method_metrics.get('edge'),
                    "kelly": method_metrics.get('kelly'),
                    "tp": selected_best.get('tp') if isinstance(selected_best, dict) else None,
                    "sl": selected_best.get('sl') if isinstance(selected_best, dict) else None,
                    "contributed": bool(method_metrics),
                })

            out = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": "ensemble",
                "horizon": horizon_val,
                "direction": direction_norm,
                "mode": mode_val,
                "distance_unit": mode_val,
                "optimizer": optimizer_val,
                "last_price": out_last_price,
                "last_price_close": out_last_price_close,
                "last_price_source": "ensemble_members_median",
                "simulation_reference_price": simulation_reference_price,
                "simulation_quote_side": "bid" if direction_norm == "long" else "ask",
                "objective": objective_val,
                "search_profile": search_profile_val,
                "fast_defaults": bool(search_profile_val == 'fast'),
                "search_config": _barrier_search_config(
                    search_profile=search_profile_val,
                    grid_style=grid_style_val,
                    preset=preset_val,
                    mode=mode_val,
                    objective=objective_val,
                    tp_min=tp_min_val,
                    tp_max=tp_max_val,
                    sl_min=sl_min_val,
                    sl_max=sl_max_val,
                    tp_steps=tp_steps_val,
                    sl_steps=sl_steps_val,
                    ratio_min=ratio_min_val,
                    ratio_max=ratio_max_val,
                    ratio_steps=ratio_steps_val,
                    vol_window=vol_window_val,
                    vol_min_mult=vol_min_mult_val,
                    vol_max_mult=vol_max_mult_val,
                    vol_steps=vol_steps_val,
                    candidate_pairs=[
                        (row.get("tp"), row.get("sl"))
                        for row in candidates
                        if isinstance(row, dict)
                    ],
                ),
                "compute_profile": {
                    "profile": search_profile_val,
                    "n_sims": int(sims),
                    "n_seeds": int(n_seeds),
                    "paths_evaluated_per_member": int(sims * n_seeds),
                    "n_trials": int(optuna_trials_val) if optimizer_val == 'optuna' else None,
                    "tp_steps": int(tp_steps_val),
                    "sl_steps": int(sl_steps_val),
                    "ratio_steps": int(ratio_steps_val),
                    "vol_steps": int(vol_steps_val),
                    "refine": bool(refine_flag),
                    "statistical_robustness": {
                        "enabled": bool(statistical_robustness_requested),
                        "applied": ["method_dispersion"],
                        "unsupported": [
                            "oos_validation",
                            "bootstrap",
                            "drift_stress",
                            "cross_seed_stability",
                            "sensitivity_analysis",
                            "power_analysis",
                        ],
                        "target_ci_width": target_ci_width_val,
                        "n_seeds_stability": n_seeds_stability_val,
                        "bootstrap_enabled": False,
                        "n_bootstrap": n_bootstrap_val,
                        "convergence_check_enabled": False,
                        "convergence_window": convergence_window_val,
                        "convergence_threshold": convergence_threshold_val,
                        "power_analysis_enabled": False,
                        "power_effect_size": power_effect_size_val,
                        "sensitivity_analysis_enabled": False,
                        "drift_stress_enabled": False,
                        "oos_validation_enabled": False,
                    } if statistical_robustness_requested else None,
                },
                "results": summary_results,
                "candidates_evaluated": len(ranked_candidates),
                "candidates_viable": candidates_viable,
                "candidates_returned": len(candidates),
                "best": selected_best,
                "viable": bool(viable),
                "least_negative": _least_negative_ref(selected_best) if (selected_best and not viable) else None,
                "grid": grid_out,
                "no_candidates": not bool(selected_best),
                "status": status,
                "status_reason": status_reason,
                "no_action": status != "ok",
                "ensemble": {
                    "methods": ensemble_methods,
                    "agg": ensemble_agg,
                    "weights": ensemble_weight_map if ensemble_weight_map else None,
                    "members": member_summaries,
                    "member_errors": member_errors,
                    "aggregate_metrics": (
                        {
                            key: value
                            for key, value in selected_best.items()
                            if key not in {'member_metrics'}
                        }
                        if isinstance(selected_best, dict) else None
                    ),
                    "n_total": n_total,
                    "n_succeeded": n_succeeded,
                    "n_failed": n_failed,
                    "degraded": ensemble_degraded,
                    "confidence": ensemble_confidence,
                    "selection_basis": "common_candidate_aggregate",
                },
            }
            if symbol_requested:
                out["symbol_requested"] = symbol_requested
            candidate_filters = _barrier_candidate_filter_config(
                tradable_only=tradable_only_val,
                min_ev=min_ev_val,
                min_edge=min_edge_val,
                min_kelly=min_kelly_val,
            )
            if candidate_filters:
                out["candidate_filters"] = candidate_filters
            if price_precision is not None:
                out["price_precision"] = int(price_precision)
            if objective_changed:
                out["objective_requested"] = objective_requested
                out["objective_used"] = objective_val
            if contract_warnings:
                out["warnings"] = list(contract_warnings)
            if ensemble_warning:
                out["warning"] = str(ensemble_warning)
            if member_errors:
                if ensemble_degraded:
                    out["warning"] = (
                        f"Ensemble degraded: {n_failed}/{n_total} members failed "
                        f"(confidence={ensemble_confidence}). "
                        f"Results based on {n_succeeded} method(s) only — interpret with caution."
                    )
                elif n_succeeded == 1:
                    out["warning"] = (
                        f"{n_failed}/{n_total} ensemble member(s) failed. "
                        f"Only 1 method succeeded — ensemble averaging has no diversification benefit."
                    )
                else:
                    out["warning"] = f"{n_failed} ensemble member(s) failed."
            diagnostics: Dict[str, Any] = {}
            if selected_best:
                diagnostics = _build_selection_diagnostics(
                    selected_best,
                    cost_per_trade=cost_per_trade,
                    candidate_count=ensemble_candidates_evaluated,
                )
                out.update(diagnostics)
            if not cost_model_complete:
                diagnostics["trading_costs_incomplete"] = True
                diagnostics["missing_cost_assumptions"] = list(
                    missing_cost_assumptions
                )
                out.update(diagnostics)
            if statistical_robustness_requested and isinstance(selected_best, dict):
                dispersion_inputs = {
                    index: dict(metrics)
                    for index, metrics in enumerate(selected_member_metrics.values())
                    if isinstance(metrics, dict)
                }
                out["statistical_robustness"] = {
                    "source": "ensemble_common_candidate",
                    "note": (
                        "Ensemble reports member-method dispersion only; "
                        "walk-forward OOS, bootstrap, drift stress, and "
                        "independent-seed holdout are not run for ensemble picks."
                    ),
                    "unsupported": [
                        "oos_validation",
                        "bootstrap",
                        "drift_stress",
                        "cross_seed_stability",
                        "sensitivity_analysis",
                        "power_analysis",
                    ],
                    "method_dispersion": _cross_seed_stability(
                        dispersion_inputs,
                        metric_keys=['prob_tp_first', 'prob_sl_first', 'ev', 'edge', 'kelly'],
                        threshold_cv=0.10,
                    ),
                    "minimum_simulations": {
                        "recommended": int(min_sims_recommended),
                        "used_per_member": int(sims),
                        "target_ci_width": target_ci_width_val,
                    },
                }
                out["min_sims_recommended"] = int(min_sims_recommended)
            if min_prob_resolve_val is not None:
                out["min_prob_resolve"] = float(min_prob_resolve_val)
            out["trading_costs"] = dict(trading_costs_payload)
            if selected_best and not viable:
                out["advice"] = [
                    "Increase horizon to allow more time for barrier resolution.",
                    "Try the opposite direction and compare objective metrics.",
                    "Widen TP/SL search ranges or switch grid_style to volatility/ratio.",
                    "Skip this setup if edge and EV remain unattractive.",
                ]
            actionability_payload = _build_actionability_payload(
                status=status,
                status_reason=status_reason,
                row=selected_best,
                diagnostics=diagnostics,
                warning=out.get("warning"),
                ensemble_degraded=ensemble_degraded,
            )
            out.update(actionability_payload)
            out["no_action"] = not bool(actionability_payload.get("trade_gate_passed"))
            _attach_viability_semantics(
                out,
                mathematically_viable=bool(viable),
                trade_gate_passed=actionability_payload.get("trade_gate_passed"),
            )
            _apply_barrier_freshness_contract(
                out,
                history_context=freshness_context,
                reference_context=reference_context,
                last_price_source=last_price_source,
            )
            return _finalize_barrier_output(
                out,
                output_mode=output_mode,
                concise_val=concise_val,
                viable_only_val=viable_only_val,
                ranked_candidates=ranked_candidates,
                viable_candidates=viable_candidates,
                candidates=candidates,
                grid_out=grid_out,
                return_grid=return_grid,
                top_k_val=top_k_val,
                selection_diagnostics=diagnostics,
            )

        if method_name == 'auto':
            method_name, auto_reason = _auto_barrier_method(
                symbol, timeframe, prices, horizon=horizon_val
            )
        seed_raw = params_dict.get('seed')
        seed_provided = seed_raw is not None
        # The live reference price affects grid scoring but not path generation.
        # Excluding it preserves common random draws across tick-only changes.
        request_seed_base = (
            normalize_barrier_seed(seed_raw)
            if seed_provided
            else _stable_barrier_seed(
                "forecast_barrier_optimize",
                symbol,
                timeframe,
                horizon_val,
                method_name,
                direction_norm,
                mode_val,
                objective_val,
                optimizer_val,
                search_profile_val,
                int(sims),
                int(n_seeds),
                int(len(prices)),
                float(prices[-1]),
                {k: v for k, v in params_dict.items() if k != "seed"},
            )
        )
        optuna_seed = normalize_barrier_seed(request_seed_base)
        hmm_sim_meta_records: List[Dict[str, int]] = []

        def _simulate_paths_for_seed_range(
            seed_base: Optional[int],
            seed_count: int,
            history_prices: Optional[np.ndarray] = None,
            sims_override: Optional[int] = None,
        ) -> Tuple[
            np.ndarray,
            bool,
            float,
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[np.ndarray],
        ]:
            local_paths_list: List[np.ndarray] = []
            calibration_prices = (
                np.asarray(history_prices, dtype=float)
                if history_prices is not None
                else prices
            )
            local_sims = int(sims_override) if sims_override is not None else int(sims)
            local_bb_enabled = method_name == 'mc_gbm_bb'
            effective_seed_count = max(1, int(seed_count))
            local_seed_base = (
                normalize_barrier_seed(seed_base)
                if seed_base is not None
                else normalize_barrier_seed(np.random.default_rng().integers(0, np.iinfo(np.int32).max))
            )

            if method_name in ('mc_gbm', 'mc_gbm_bb'):
                for offset in range(effective_seed_count):
                    sim = _simulate_gbm_mc(
                        calibration_prices,
                        horizon=horizon_val,
                        n_sims=local_sims,
                        seed=offset_barrier_seed(local_seed_base, offset),
                        antithetic=False,
                    )
                    local_paths_list.append(np.asarray(sim['price_paths'], dtype=float))
            elif method_name == 'hmm_mc':
                n_states = int(params_dict.get('n_states', 2) or 2)
                for offset in range(effective_seed_count):
                    sim = _simulate_hmm_mc(
                        calibration_prices,
                        horizon=horizon_val,
                        n_states=int(n_states),
                        n_sims=local_sims,
                        seed=offset_barrier_seed(local_seed_base, offset),
                    )
                    local_paths_list.append(np.asarray(sim['price_paths'], dtype=float))
                    requested_states = sim.get("requested_n_states")
                    fitted_states = sim.get("fitted_n_states")
                    if requested_states is not None and fitted_states is not None:
                        hmm_sim_meta_records.append(
                            {
                                "requested_n_states": int(requested_states),
                                "fitted_n_states": int(fitted_states),
                            }
                        )
            elif method_name == 'garch':
                p_order = int(params_dict.get('p', 1))
                q_order = int(params_dict.get('q', 1))
                for offset in range(effective_seed_count):
                    sim = _simulate_garch_mc(
                        calibration_prices,
                        horizon=horizon_val,
                        n_sims=local_sims,
                        seed=offset_barrier_seed(local_seed_base, offset),
                        p_order=p_order,
                        q_order=q_order,
                    )
                    local_paths_list.append(np.asarray(sim['price_paths'], dtype=float))
            elif method_name == 'bootstrap':
                bs = params_dict.get('block_size')
                if bs:
                    bs = int(bs)
                for offset in range(effective_seed_count):
                    sim = _simulate_bootstrap_mc(
                        calibration_prices,
                        horizon=horizon_val,
                        n_sims=local_sims,
                        seed=offset_barrier_seed(local_seed_base, offset),
                        block_size=bs,
                    )
                    local_paths_list.append(np.asarray(sim['price_paths'], dtype=float))
            elif method_name == 'heston':
                heston_bars_per_year, _ = _annualization_context(timeframe, symbol)
                for offset in range(effective_seed_count):
                    sim = _simulate_heston_mc(
                        calibration_prices,
                        horizon=horizon_val,
                        n_sims=local_sims,
                        seed=offset_barrier_seed(local_seed_base, offset),
                        kappa=params_dict.get('kappa'),
                        theta=params_dict.get('theta'),
                        xi=params_dict.get('xi'),
                        rho=params_dict.get('rho'),
                        v0=params_dict.get('v0'),
                        bars_per_year=heston_bars_per_year,
                    )
                    local_paths_list.append(np.asarray(sim['price_paths'], dtype=float))
            elif method_name == 'jump_diffusion':
                for offset in range(effective_seed_count):
                    sim = _simulate_jump_diffusion_mc(
                        calibration_prices,
                        horizon=horizon_val,
                        n_sims=local_sims,
                        seed=offset_barrier_seed(local_seed_base, offset),
                        jump_lambda=params_dict.get('jump_lambda', params_dict.get('lambda')),
                        jump_mu=params_dict.get('jump_mu'),
                        jump_sigma=params_dict.get('jump_sigma'),
                        jump_threshold=float(params_dict.get('jump_threshold', 3.0)),
                    )
                    local_paths_list.append(np.asarray(sim['price_paths'], dtype=float))
            else:
                raise ValueError(
                    f"Unsupported method: {method}. Use 'mc_gbm', 'mc_gbm_bb', "
                    f"'hmm_mc', 'garch', 'bootstrap', 'heston', 'jump_diffusion', "
                    f"'auto', or 'ensemble'."
                )

            local_paths = (
                np.vstack(local_paths_list)
                if len(local_paths_list) > 1
                else local_paths_list[0]
            )
            (
                local_paths,
                local_bb_enabled,
                local_bb_sigma,
                local_bb_log_paths,
                local_bb_uniform_tp,
                local_bb_uniform_sl,
            ) = _prepare_brownian_bridge_draws(
                local_paths,
                calibration_prices=calibration_prices,
                last_price_close=last_price_close,
                reference_price=simulation_reference_price,
                bb_enabled=local_bb_enabled,
                seed_base=local_seed_base,
            )

            return (
                local_paths,
                local_bb_enabled,
                local_bb_sigma,
                local_bb_log_paths,
                local_bb_uniform_tp,
                local_bb_uniform_sl,
            )

        def _linspace(a: float, b: float, n: int) -> np.ndarray:
            try:
                return np.linspace(float(a), float(b), int(max(1, n)))
            except Exception:
                return np.array([float(a)])

        seen: Set[Tuple[int, int]] = set()
        base_candidates: List[Tuple[float, float]] = []

        def _push(tp_unit: float, sl_unit: float, bucket: List[Tuple[float, float]]) -> None:
            try:
                tp_val = float(tp_unit)
                sl_val = float(sl_unit)
            except (TypeError, ValueError):
                return
            if not np.isfinite(tp_val) or not np.isfinite(sl_val):
                return
            if tp_val <= 0 or sl_val <= 0:
                return
            if tp_val < min_barrier_distance or sl_val < min_barrier_distance:
                return
            key = (int(round(tp_val * 1e6)), int(round(sl_val * 1e6)))
            if key in seen:
                return
            seen.add(key)
            bucket.append((tp_val, sl_val))

        def _add_fixed(bucket: List[Tuple[float, float]], tp_a: float, tp_b: float, tp_n: int, sl_a: float, sl_b: float, sl_n: int) -> None:
            for tp_val in _linspace(tp_a, tp_b, tp_n):
                for sl_val in _linspace(sl_a, sl_b, sl_n):
                    _push(tp_val, sl_val, bucket)

        if grid_style_val == 'preset':
            preset_key = str(preset_val)
            cfg = BARRIER_GRID_PRESETS[preset_key]
            if mode_val == 'pct':
                _add_fixed(base_candidates, cfg['tp_min'], cfg['tp_max'], int(cfg['tp_steps']), cfg['sl_min'], cfg['sl_max'], int(cfg['sl_steps']))
            else:
                increment = float(pip_size) if mode_val == "pips" else float(tick_size)
                scale = (float(last_price) / increment) / 100.0
                _add_fixed(base_candidates, cfg['tp_min'] * scale, cfg['tp_max'] * scale, int(cfg['tp_steps']), cfg['sl_min'] * scale, cfg['sl_max'] * scale, int(cfg['sl_steps']))

        elif grid_style_val == 'volatility':
            # Calculate simple volatility over window
            rets = _log_returns_from_prices(prices)
            rets = rets[np.isfinite(rets)]
            if rets.size > vol_window_val:
                rets = rets[-vol_window_val:]
            vol_per_bar = float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0
            vol_horizon = vol_per_bar * np.sqrt(horizon_val)

            # Convert to percentage space for baseline
            vol_pct = vol_horizon * 100.0

            if mode_val == 'pct':
                tp_start = max(vol_floor_pct_val, vol_pct * vol_min_mult_val)
                tp_end = max(tp_start * 1.1, vol_pct * vol_max_mult_val)
                sl_start = max(vol_floor_pct_val, vol_pct * vol_min_mult_val * 0.8)
                _add_fixed(base_candidates, tp_start, tp_end, vol_steps_val, sl_start, sl_start * vol_sl_multiplier_val, vol_sl_steps_val)
            else:
                increment = float(pip_size) if mode_val == "pips" else float(tick_size)
                vol_distance = (vol_pct / 100.0) * (last_price / increment)
                tp_start = max(vol_floor_ticks_val, vol_distance * vol_min_mult_val)
                tp_end = max(tp_start * 1.1, vol_distance * vol_max_mult_val)
                sl_start = max(vol_floor_ticks_val, vol_distance * vol_min_mult_val * 0.8)
                _add_fixed(base_candidates, tp_start, tp_end, vol_steps_val, sl_start, sl_start * vol_sl_multiplier_val, vol_sl_steps_val)

        elif grid_style_val == 'ratio':
            # Fixed SL grid, TP derived from ratios
            sl_start = sl_min_val
            sl_end = sl_max_val
            for sl_val in _linspace(sl_start, sl_end, sl_steps_val):
                for r in _linspace(ratio_min_val, ratio_max_val, ratio_steps_val):
                    _push(sl_val * r, sl_val, base_candidates)

        else: # fixed
            _add_fixed(base_candidates, tp_min_val, tp_max_val, tp_steps_val, sl_min_val, sl_max_val, sl_steps_val)

        if mode_val in {'ticks', 'pips'} and not base_candidates:
            return _tick_grid_below_min_distance_error(
                min_barrier_distance=min_barrier_distance,
                cost_per_trade=modeled_cost_per_trade,
                mode=mode_val,
            )

        try:
            (
                paths,
                bb_enabled,
                bb_sigma,
                bb_log_paths,
                bb_uniform_tp,
                bb_uniform_sl,
            ) = _simulate_paths_for_seed_range(seed_base=request_seed_base, seed_count=int(n_seeds))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            return {
                "error": f"Simulation failed ({method_name}): {e}",
                "error_type": "simulation_failure",
            }

        S, H = paths.shape

        # Evaluate candidates
        results: List[Dict[str, Any]] = []
        optuna_meta: Optional[Dict[str, Any]] = None
        dir_long = direction_norm == 'long'
        invalid_barrier_candidates = 0
        eval_context = _BarrierEvaluationContext(
            mode_val=mode_val,
            dir_long=dir_long,
            last_price=float(last_price),
            tick_size=_optional_finite_float(tick_size),
            rr_min_val=rr_min_val,
            rr_max_val=rr_max_val,
            has_trading_costs=has_trading_costs,
            ev_deduct_cost=float(ev_deduct_cost),
            cost_per_trade=float(cost_per_trade),
            min_prob_win_val=min_prob_win_val,
            max_prob_no_hit_val=max_prob_no_hit_val,
            min_prob_resolve_val=min_prob_resolve_val,
            max_median_time_val=max_median_time_val,
            same_bar_policy=same_bar_policy_value,
            gap_aware_stops=_coerce_barrier_bool_flag(
                params_dict.get('gap_aware_stops', False),
                default=False,
            ),
            pip_size=_optional_finite_float(pip_size),
            path_anchor_price=_optional_finite_float(simulation_reference_price),
        )

        def _evaluate(
            bucket: List[Tuple[float, float]],
            eval_paths: np.ndarray,
            eval_bb_enabled: bool,
            eval_bb_sigma: float,
            eval_bb_log_paths: Optional[np.ndarray],
            eval_bb_uniform_tp: Optional[np.ndarray],
            eval_bb_uniform_sl: Optional[np.ndarray],
            count_invalid: bool = True,
        ) -> List[Dict[str, Any]]:
            nonlocal invalid_barrier_candidates
            rows, invalid_count = _evaluate_barrier_bucket(
                bucket,
                eval_paths,
                context=eval_context,
                bridge_inputs=_BarrierBridgeInputs(
                    enabled=eval_bb_enabled,
                    sigma=float(eval_bb_sigma),
                    log_paths=eval_bb_log_paths,
                    uniform_tp=eval_bb_uniform_tp,
                    uniform_sl=eval_bb_uniform_sl,
                ),
                count_invalid=count_invalid,
            )
            invalid_barrier_candidates += invalid_count
            return rows

        def _objective_convergence_inputs(
            eval_paths: np.ndarray,
            *,
            best_row: Dict[str, Any],
        ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
            _, horizon_total = eval_paths.shape
            tp_trigger = _safe_float(best_row.get('tp_price'))
            sl_trigger = _safe_float(best_row.get('sl_price'))
            reward = _safe_float(best_row.get('tp'))
            risk = _safe_float(best_row.get('sl'))
            if tp_trigger is None or sl_trigger is None or reward is None or risk is None:
                return None, None, None

            first_tp, first_sl, wins_mask, losses_mask, ties_mask = _candidate_hit_arrays(
                eval_paths,
                tp_trigger=tp_trigger,
                sl_trigger=sl_trigger,
                context=eval_context,
                bridge_inputs=_BarrierBridgeInputs(
                    enabled=bb_enabled,
                    sigma=float(bb_sigma),
                    log_paths=bb_log_paths,
                    uniform_tp=bb_uniform_tp,
                    uniform_sl=bb_uniform_sl,
                ),
            )
            unresolved_mask = ~(wins_mask | losses_mask | ties_mask)

            wins = wins_mask.astype(float)
            losses = losses_mask.astype(float)
            ties = ties_mask.astype(float)

            trials = np.arange(1, eval_paths.shape[0] + 1, dtype=float)
            cum_wins = np.cumsum(wins)
            cum_losses = np.cumsum(losses)
            cum_ties = np.cumsum(ties)

            if same_bar_policy_value == "tp_first":
                prob_tp_first_series = (cum_wins + cum_ties) / trials
                prob_sl_first_series = cum_losses / trials
                active_counts = cum_wins + cum_losses + cum_ties
            elif same_bar_policy_value == "neutral":
                prob_tp_first_series = cum_wins / trials
                prob_sl_first_series = cum_losses / trials
                active_counts = cum_wins + cum_losses
            else:
                prob_tp_first_series = cum_wins / trials
                prob_sl_first_series = (cum_losses + cum_ties) / trials
                active_counts = cum_wins + cum_losses + cum_ties
            prob_win_series = prob_tp_first_series
            prob_loss_series = prob_sl_first_series
            prob_resolve_series = active_counts / trials

            net_reward = reward - cost_per_trade if has_trading_costs else reward
            net_risk = risk + cost_per_trade if has_trading_costs else risk
            net_rr = net_reward / net_risk if net_risk > 0 else 0.0

            convergence_outcomes = BarrierPathOutcomes(
                first_tp=first_tp,
                first_sl=first_sl,
                wins=wins_mask,
                losses=losses_mask,
                ties=ties_mask,
                unresolved=unresolved_mask,
                time_in_trade=np.minimum(
                    np.minimum(first_tp, first_sl) + 1,
                    horizon_total,
                ),
                horizon=int(horizon_total),
            )
            convergence_payoffs = barrier_path_payoffs(
                eval_paths,
                convergence_outcomes,
                entry_price=last_price,
                reward=reward,
                risk=risk,
                direction="long" if dir_long else "short",
                mode=mode_val,  # type: ignore[arg-type]
                tick_size=float(tick_size) if tick_size else 0.0,
                pip_size=float(pip_size) if pip_size else None,
                cost_per_trade=(cost_per_trade if has_trading_costs else 0.0),
                same_bar_policy=same_bar_policy_value,
                gap_aware_stops=eval_context.gap_aware_stops,
            )
            path_payoff = (
                convergence_payoffs.net
                if has_trading_costs
                else convergence_payoffs.gross
            )
            cumulative_payoff = np.cumsum(path_payoff)
            if mode_val == 'pct':
                unit_to_return = 0.01
            elif mode_val == 'pips':
                unit_to_return = float(pip_size) / float(last_price)
            else:
                unit_to_return = float(tick_size) / float(last_price)
            path_utility = np.log1p(
                np.maximum(path_payoff * unit_to_return, -0.999999)
            )

            event = f"selected_objective_{objective_val}"
            if objective_val == 'prob_tp_first':
                estimate_series = prob_tp_first_series
            elif objective_val == 'prob_resolve':
                estimate_series = prob_resolve_series
            elif objective_val == 'min_loss_prob':
                estimate_series = prob_loss_series
            elif objective_val == 'edge':
                estimate_series = prob_win_series - prob_loss_series
            elif objective_val == 'ev':
                estimate_series = cumulative_payoff / trials
            elif objective_val == 'ev_per_bar':
                time_in_trade = np.minimum(np.minimum(first_tp, first_sl) + 1, horizon_total).astype(float)
                mean_time_series = np.cumsum(time_in_trade) / trials
                ev_series = cumulative_payoff / trials
                estimate_series = np.divide(
                    ev_series,
                    mean_time_series,
                    out=np.zeros_like(ev_series),
                    where=mean_time_series > 0,
                )
            elif objective_val == 'kelly':
                estimate_series = (
                    prob_tp_first_series - (prob_sl_first_series / net_rr)
                    if net_rr > 0 else np.zeros_like(prob_tp_first_series)
                )
            elif objective_val == 'ev_cond':
                active_mask = active_counts > 0
                estimate_series = np.zeros_like(prob_tp_first_series)
                if np.any(active_mask):
                    active_payoff = np.where(
                        convergence_payoffs.active,
                        path_payoff,
                        0.0,
                    )
                    estimate_series[active_mask] = (
                        np.cumsum(active_payoff)[active_mask]
                        / active_counts[active_mask]
                    )
            elif objective_val == 'kelly_cond':
                active_mask = active_counts > 0
                estimate_series = np.zeros_like(prob_tp_first_series)
                if np.any(active_mask) and net_rr > 0:
                    win_c = np.divide(
                        prob_tp_first_series * trials,
                        active_counts,
                        out=np.zeros_like(active_counts, dtype=float),
                        where=active_mask,
                    )
                    loss_c = np.divide(
                        prob_sl_first_series * trials,
                        active_counts,
                        out=np.zeros_like(active_counts, dtype=float),
                        where=active_mask,
                    )
                    estimate_series[active_mask] = (
                        win_c[active_mask] - (loss_c[active_mask] / net_rr)
                    )
            elif objective_val == 'profit_factor':
                cumulative_profit = np.cumsum(np.maximum(path_payoff, 0.0))
                denom = np.cumsum(np.maximum(-path_payoff, 0.0))
                estimate_series = np.zeros_like(prob_tp_first_series)
                valid = denom > 0
                estimate_series[valid] = cumulative_profit[valid] / denom[valid]
                positive_no_loss = (~valid) & (cumulative_profit > 0)
                estimate_series[positive_no_loss] = 1e9
            elif objective_val == 'utility':
                estimate_series = np.cumsum(path_utility) / trials
            else:
                estimate_series = prob_resolve_series

            return estimate_series * trials, trials, event

        pareto_front: Optional[List[Dict[str, Any]]] = None
        if optimizer_val == 'optuna':
            try:
                import optuna
                try:
                    from optuna.exceptions import (
                        ExperimentalWarning as _OptunaExperimentalWarning,
                    )
                except Exception:
                    _OptunaExperimentalWarning = Warning
            except Exception as ex:
                return {"error": f"Optuna optimizer requested but unavailable: {ex}"}

            def _suppress_optuna_experimental_warnings() -> None:
                warnings.simplefilter("ignore", _OptunaExperimentalWarning)
                warnings.filterwarnings("ignore", category=_OptunaExperimentalWarning)
                warnings.filterwarnings(
                    "ignore",
                    message=r".*multivariate.*experimental feature.*",
                )

            tp_vals = [float(tp) for tp, _ in base_candidates] if base_candidates else [float(tp_min_val), float(tp_max_val)]
            sl_vals = [float(sl) for _, sl in base_candidates] if base_candidates else [float(sl_min_val), float(sl_max_val)]
            tp_lo = max(1e-9, min_barrier_distance, float(min(tp_vals)))
            tp_hi = max(tp_lo, float(max(tp_vals)))
            sl_lo = max(1e-9, min_barrier_distance, float(min(sl_vals)))
            sl_hi = max(sl_lo, float(max(sl_vals)))
            rr_lo = max(1e-9, float(min(ratio_min_val, ratio_max_val)))
            rr_hi = max(rr_lo, float(max(ratio_min_val, ratio_max_val)))

            sampler_name = optuna_sampler_val
            if sampler_name == 'random':
                sampler_obj = optuna.samplers.RandomSampler(seed=optuna_seed)
            elif sampler_name == 'cmaes':
                if optuna_pareto_val:
                    sampler_name = 'nsga2'
                    sampler_obj = optuna.samplers.NSGAIISampler(seed=optuna_seed)
                else:
                    sampler_obj = optuna.samplers.CmaEsSampler(seed=optuna_seed)
            else:
                sampler_name = 'tpe'
                with warnings.catch_warnings():
                    _suppress_optuna_experimental_warnings()
                    sampler_obj = optuna.samplers.TPESampler(seed=optuna_seed, multivariate=True)

            pruner_name = optuna_pruner_val
            if optuna_pareto_val:
                pruner_name = 'none_multiobjective'
                pruner_obj = optuna.pruners.NopPruner()
            elif pruner_name in {'none'}:
                pruner_obj = optuna.pruners.NopPruner()
            elif pruner_name == 'hyperband':
                pruner_obj = optuna.pruners.HyperbandPruner()
            elif pruner_name == 'percentile':
                pruner_obj = optuna.pruners.PercentilePruner(50.0)
            else:
                pruner_name = 'median'
                pruner_obj = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampled_rows: List[Dict[str, Any]] = []
            trial_rows: Dict[int, Dict[str, Any]] = {}
            pareto_rows_full: List[Dict[str, Any]] = []

            if optuna_pareto_val:
                directions = [d for _, d in pareto_objectives]
                with warnings.catch_warnings():
                    _suppress_optuna_experimental_warnings()
                    study = optuna.create_study(directions=directions, sampler=sampler_obj, pruner=pruner_obj)

                def _bad_values() -> Tuple[float, ...]:
                    vals: List[float] = []
                    for _, d in pareto_objectives:
                        vals.append(-1e18 if d == 'maximize' else 1e18)
                    return tuple(vals)

                def _metric_value(row: Dict[str, Any], metric: str, direction_name: str) -> float:
                    raw = row.get(metric)
                    if metric == 'profit_factor' and raw is None:
                        prob_win = _safe_float(row.get('prob_tp_first'))
                        prob_loss = _safe_float(row.get('prob_sl_first'))
                        if prob_win is not None and prob_win > 0.0 and prob_loss == 0.0:
                            return 1e18 if direction_name == 'maximize' else -1e18
                    try:
                        value = float(raw)
                    except Exception:
                        return -1e18 if direction_name == 'maximize' else 1e18
                    if not np.isfinite(value):
                        return -1e18 if direction_name == 'maximize' else 1e18
                    return float(value)

                def _objective_trial(trial: Any) -> Tuple[float, ...]:
                    if grid_style_val == 'ratio':
                        sl_unit = float(trial.suggest_float('sl', sl_lo, sl_hi))
                        rr_unit = float(trial.suggest_float('rr', rr_lo, rr_hi))
                        tp_unit = sl_unit * rr_unit
                    else:
                        tp_unit = float(trial.suggest_float('tp', tp_lo, tp_hi))
                        sl_unit = float(trial.suggest_float('sl', sl_lo, sl_hi))

                    rows = _evaluate(
                        [(tp_unit, sl_unit)],
                        paths,
                        bb_enabled,
                        bb_sigma,
                        bb_log_paths,
                        bb_uniform_tp,
                        bb_uniform_sl,
                    )
                    if not rows:
                        return _bad_values()
                    row = rows[0]
                    sampled_rows.append(row)
                    trial_rows[int(trial.number)] = row
                    trial.set_user_attr('tp', float(row.get('tp', tp_unit)))
                    trial.set_user_attr('sl', float(row.get('sl', sl_unit)))
                    values = tuple(
                        _metric_value(row, metric_name, direction_name)
                        for metric_name, direction_name in pareto_objectives
                    )
                    trial.set_user_attr('objective_values', {
                        metric_name: float(values[idx]) for idx, (metric_name, _) in enumerate(pareto_objectives)
                    })
                    return values

                try:
                    with warnings.catch_warnings():
                        _suppress_optuna_experimental_warnings()
                        study.optimize(
                            _objective_trial,
                            n_trials=int(optuna_trials_val),
                            timeout=float(optuna_timeout_val) if optuna_timeout_val is not None else None,
                            n_jobs=int(optuna_n_jobs_val),
                        )
                except (ValueError, RuntimeError) as e:
                    return {
                        "error": f"Optuna optimization failed (pareto): {e}",
                        "error_type": "optuna_failure",
                    }
                front: List[Dict[str, Any]] = []
                for trial in study.best_trials:
                    row = trial_rows.get(int(trial.number))
                    if not isinstance(row, dict):
                        continue
                    entry = dict(row)
                    values = list(trial.values) if isinstance(trial.values, (list, tuple)) else []
                    entry['trial'] = int(trial.number)
                    entry['objective_values'] = {
                        metric_name: float(values[idx]) if idx < len(values) else None
                        for idx, (metric_name, _) in enumerate(pareto_objectives)
                    }
                    front.append(entry)

                if front:
                    front.sort(
                        key=lambda row: tuple(
                            -_metric_value(row, metric_name, direction_name)
                            if direction_name == 'maximize'
                            else _metric_value(row, metric_name, direction_name)
                            for metric_name, direction_name in pareto_objectives
                        )
                    )
                pareto_rows_full = list(front)
                pareto_front = front[:int(pareto_limit_val)]
            else:
                maximize = objective_val != 'min_loss_prob'
                direction = 'maximize' if maximize else 'minimize'
                with warnings.catch_warnings():
                    _suppress_optuna_experimental_warnings()
                    study = optuna.create_study(direction=direction, sampler=sampler_obj, pruner=pruner_obj)

                def _objective_trial(trial: Any) -> float:
                    if grid_style_val == 'ratio':
                        sl_unit = float(trial.suggest_float('sl', sl_lo, sl_hi))
                        rr_unit = float(trial.suggest_float('rr', rr_lo, rr_hi))
                        tp_unit = sl_unit * rr_unit
                    else:
                        tp_unit = float(trial.suggest_float('tp', tp_lo, tp_hi))
                        sl_unit = float(trial.suggest_float('sl', sl_lo, sl_hi))

                    rows = _evaluate(
                        [(tp_unit, sl_unit)],
                        paths,
                        bb_enabled,
                        bb_sigma,
                        bb_log_paths,
                        bb_uniform_tp,
                        bb_uniform_sl,
                    )
                    if not rows:
                        return -1e18 if maximize else 1e18
                    row = rows[0]
                    sampled_rows.append(row)
                    trial_rows[int(trial.number)] = row
                    trial.set_user_attr('tp', float(row.get('tp', tp_unit)))
                    trial.set_user_attr('sl', float(row.get('sl', sl_unit)))
                    if objective_val == 'min_loss_prob':
                        objective_value = float(row.get('prob_loss', 1.0))
                    elif objective_val == 'profit_factor' and row.get('profit_factor') is None:
                        prob_win = _safe_float(row.get('prob_tp_first'))
                        prob_loss = _safe_float(row.get('prob_sl_first'))
                        objective_value = (
                            1e18
                            if prob_win is not None and prob_win > 0.0 and prob_loss == 0.0
                            else -1e18
                        )
                    else:
                        objective_value = float(row.get(objective_val, row.get('ev', -1e18)))
                    trial.report(objective_value, step=0)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    return objective_value

                try:
                    with warnings.catch_warnings():
                        _suppress_optuna_experimental_warnings()
                        study.optimize(
                            _objective_trial,
                            n_trials=int(optuna_trials_val),
                            timeout=float(optuna_timeout_val) if optuna_timeout_val is not None else None,
                            n_jobs=int(optuna_n_jobs_val),
                        )
                except (ValueError, RuntimeError) as e:
                    return {
                        "error": f"Optuna optimization failed: {e}",
                        "error_type": "optuna_failure",
                    }

            dedup: Dict[Tuple[int, int], Dict[str, Any]] = {}
            source_rows = pareto_rows_full if optuna_pareto_val else sampled_rows
            for row in source_rows:
                try:
                    key = (int(round(float(row.get('tp', 0.0)) * 1e6)), int(round(float(row.get('sl', 0.0)) * 1e6)))
                except Exception:
                    continue
                cur = dedup.get(key)
                if cur is None:
                    dedup[key] = row
                    continue
                if objective_val == 'min_loss_prob':
                    if float(row.get('prob_loss', 1.0)) < float(cur.get('prob_loss', 1.0)):
                        dedup[key] = row
                else:
                    if float(row.get(objective_val, -1e18)) > float(cur.get(objective_val, -1e18)):
                        dedup[key] = row

            results.extend(dedup.values())
            optuna_meta = {
                "n_trials": int(optuna_trials_val),
                "completed_trials": int(len(study.trials)),
                "sampler": sampler_name,
                "pruner": pruner_name,
                "timeout": float(optuna_timeout_val) if optuna_timeout_val is not None else None,
                "n_jobs": int(optuna_n_jobs_val),
                "pareto": bool(optuna_pareto_val),
            }
            if optuna_pareto_val:
                optuna_meta["pareto_objectives"] = [
                    {"metric": metric_name, "direction": direction_name}
                    for metric_name, direction_name in pareto_objectives
                ]
        else:
            results.extend(
                _evaluate(
                    base_candidates,
                    paths,
                    bb_enabled,
                    bb_sigma,
                    bb_log_paths,
                    bb_uniform_tp,
                    bb_uniform_sl,
                )
            )

        _sort_candidate_results(results, objective_val)

        if refine_flag and results:
            refine_seed_rows = list(results)
            if tradable_only_val or min_ev_val is not None or min_edge_val is not None or min_kelly_val is not None:
                refine_seed_rows = [
                    row
                    for row in refine_seed_rows
                    if _candidate_passes_threshold_filters(
                        row,
                        cost_per_trade=cost_per_trade,
                        tradable_only_val=tradable_only_val,
                        min_ev_val=min_ev_val,
                        min_edge_val=min_edge_val,
                        min_kelly_val=min_kelly_val,
                    )
                ]
            if viable_only_val:
                refine_seed_rows = [
                    row
                    for row in refine_seed_rows
                    if _candidate_is_viable(row, cost_per_trade=cost_per_trade)
                ]
            best_seed = refine_seed_rows[0] if refine_seed_rows else None
        else:
            best_seed = None

        if best_seed is not None:
            tp_c = best_seed['tp']
            sl_c = best_seed['sl']
            refine_candidates: List[Tuple[float, float]] = []
            base_tp_values = [pair[0] for pair in base_candidates]
            base_sl_values = [pair[1] for pair in base_candidates]
            tp_floor = min(base_tp_values) if base_tp_values else tp_c
            tp_ceiling = max(base_tp_values) if base_tp_values else tp_c
            sl_floor = min(base_sl_values) if base_sl_values else sl_c
            sl_ceiling = max(base_sl_values) if base_sl_values else sl_c
            sl_a = max(sl_floor, sl_c * (1.0 - refine_radius_val))
            sl_b = min(sl_ceiling, sl_c * (1.0 + refine_radius_val))
            if grid_style_val == 'ratio':
                ratio_c = tp_c / sl_c
                ratio_a = max(ratio_min_val, ratio_c * (1.0 - refine_radius_val))
                ratio_b = min(ratio_max_val, ratio_c * (1.0 + refine_radius_val))
                for sl_refined in _linspace(sl_a, sl_b, refine_steps_val):
                    for ratio_refined in _linspace(ratio_a, ratio_b, refine_steps_val):
                        _push(sl_refined * ratio_refined, sl_refined, refine_candidates)
            else:
                tp_a = max(tp_floor, tp_c * (1.0 - refine_radius_val))
                tp_b = min(tp_ceiling, tp_c * (1.0 + refine_radius_val))
                _add_fixed(
                    refine_candidates,
                    tp_a,
                    tp_b,
                    refine_steps_val,
                    sl_a,
                    sl_b,
                    refine_steps_val,
                )
            results.extend(
                _evaluate(
                    refine_candidates,
                    paths,
                    bb_enabled,
                    bb_sigma,
                    bb_log_paths,
                    bb_uniform_tp,
                    bb_uniform_sl,
                )
            )
            _sort_candidate_results(results, objective_val)

        candidate_views = _select_barrier_candidate_views(
            list(results),
            cost_per_trade=cost_per_trade,
            viable_only_val=viable_only_val,
            tradable_only_val=tradable_only_val,
            min_ev_val=min_ev_val,
            min_edge_val=min_edge_val,
            min_kelly_val=min_kelly_val,
            concise_val=concise_val,
            top_k_val=top_k_val,
            return_grid=return_grid,
            output_mode=output_mode,
        )
        ranked_candidates = candidate_views["ranked_candidates"]
        viable_candidates = candidate_views["viable_candidates"]
        candidates = candidate_views["candidates"]
        grid_out = candidate_views["grid_out"]
        summary_results = candidate_views["summary_results"]
        viability_filtered_out = candidate_views["viability_filtered_out"]
        no_candidates = len(candidates) == 0
        warning = candidate_views["warning"]
            
        best = candidates[0] if candidates else None
        if isinstance(best, dict):
            _annotate_candidate_metrics(best, cost_per_trade=cost_per_trade)
        viable = _candidate_is_viable(best, cost_per_trade=cost_per_trade)
        candidates_viable = int(len(viable_candidates))
        status = "ok"
        status_reason = None
        if viability_filtered_out:
            status = "non_viable"
            status_reason = warning
        elif no_candidates:
            status = "no_candidates"
            status_reason = warning
        elif not viable:
            status = "non_viable"
            status_reason = _candidate_status_reason(best, cost_per_trade=cost_per_trade)

        out = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "method": method_name,
            "intra_bar_hit_detection": (
                "brownian_bridge" if bb_enabled else "simulated_bar_close"
            ),
            "horizon": horizon_val,
            "direction": direction_norm,
            "mode": mode_val,
            "distance_unit": mode_val,
            "optimizer": optimizer_val,
            "last_price": float(last_price),
            "last_price_close": float(last_price_close),
            "last_price_source": last_price_source,
            "simulation_reference_price": simulation_reference_price,
            "simulation_quote_side": "bid" if direction_norm == "long" else "ask",
            "objective": objective_val,
            "search_profile": search_profile_val,
            "fast_defaults": bool(search_profile_val == 'fast'),
            "search_config": _barrier_search_config(
                search_profile=search_profile_val,
                grid_style=grid_style_val,
                preset=preset_val,
                mode=mode_val,
                objective=objective_val,
                tp_min=tp_min_val,
                tp_max=tp_max_val,
                sl_min=sl_min_val,
                sl_max=sl_max_val,
                tp_steps=tp_steps_val,
                sl_steps=sl_steps_val,
                ratio_min=ratio_min_val,
                ratio_max=ratio_max_val,
                ratio_steps=ratio_steps_val,
                vol_window=vol_window_val,
                vol_min_mult=vol_min_mult_val,
                vol_max_mult=vol_max_mult_val,
                vol_steps=vol_steps_val,
                candidate_pairs=base_candidates,
            ),
            "constraints_applied": {
                "min_prob_win": min_prob_win_val,
                "max_prob_no_hit": max_prob_no_hit_val,
                "max_median_time": max_median_time_val,
                "max_median_time_basis": "t_hit_resolve_median",
                "min_prob_resolve": min_prob_resolve_val,
                "min_prob_resolve_source": (
                    "objective_default"
                    if (
                        params_dict.get('min_prob_resolve') is None
                        and min_prob_resolve_val is not None
                    )
                    else "explicit_params"
                    if params_dict.get('min_prob_resolve') is not None
                    else None
                ),
                "min_barrier_distance": float(min_barrier_distance),
                "min_barrier_distance_basis": (
                    f"max({min_barrier_key}, "
                    f"{min_barrier_multiplier:g} * round_trip_cost)"
                ),
            },
            "compute_profile": {
                "profile": search_profile_val,
                "n_sims": int(sims),
                "n_seeds": int(n_seeds),
                "paths_evaluated": int(S),
                "seed": int(request_seed_base),
                "seed_source": "params" if seed_provided else "derived_from_request",
                "n_trials": int(optuna_trials_val) if optimizer_val == 'optuna' else None,
                "tp_steps": int(tp_steps_val),
                "sl_steps": int(sl_steps_val),
                "ratio_steps": int(ratio_steps_val),
                "vol_steps": int(vol_steps_val),
                "refine": bool(refine_flag),
                "statistical_robustness": {
                    "enabled": bool(statistical_robustness_requested),
                    "target_ci_width": target_ci_width_val,
                    "n_seeds_stability": n_seeds_stability_val,
                    "bootstrap_enabled": bool(enable_bootstrap_val),
                    "n_bootstrap": n_bootstrap_val,
                    "convergence_check_enabled": bool(enable_convergence_check_val),
                    "convergence_window": convergence_window_val,
                    "convergence_threshold": convergence_threshold_val,
                    "power_analysis_enabled": bool(enable_power_analysis_val),
                    "power_effect_size": power_effect_size_val,
                    "sensitivity_analysis_enabled": bool(enable_sensitivity_analysis_val),
                    "drift_stress_enabled": bool(enable_drift_stress_val),
                    "oos_validation_enabled": bool(enable_oos_validation_val),
                } if statistical_robustness_requested else None,
            },
            "simulation_seed": int(request_seed_base),
            "simulation_seed_source": (
                "params" if seed_provided else "derived_from_request"
            ),
            "results": summary_results,
            "candidates_evaluated": len(ranked_candidates),
            "candidates_viable": candidates_viable,
            "candidates_returned": len(candidates),
            "best": best,
            "viable": viable,
            "least_negative": _least_negative_ref(best) if (best is not None and not viable) else None,
            "grid": grid_out,
            "no_candidates": no_candidates,
            "status": status,
            "status_reason": status_reason,
            "no_action": status != "ok",
        }
        candidate_filters = _barrier_candidate_filter_config(
            tradable_only=tradable_only_val,
            min_ev=min_ev_val,
            min_edge=min_edge_val,
            min_kelly=min_kelly_val,
        )
        if symbol_requested:
            out["symbol_requested"] = symbol_requested
        if candidate_filters:
            out["candidate_filters"] = candidate_filters
        if price_precision is not None:
            out["price_precision"] = int(price_precision)
        if optuna_meta is not None:
            out["optuna"] = optuna_meta
        if pareto_front is not None:
            out["pareto_front"] = pareto_front
            out["pareto_count"] = int(len(pareto_front))
        if contract_warnings:
            out["warnings"] = list(contract_warnings)
        
        statistical_analysis: Dict[str, Any] = {}
        if statistical_robustness_requested and isinstance(best, dict) and len(candidates) > 0:
            statistical_analysis = {
                "minimum_simulations": {
                    "recommended": int(min_sims_recommended),
                    "used": int(S),
                    "used_per_seed": int(sims),
                    "target_ci_width": target_ci_width_val,
                    "confidence": 0.95,
                }
            }

            if enable_power_analysis_val and best.get('prob_win') is not None:
                prob_win_val = float(best['prob_win'])
                power_result = _power_analysis(
                    base_prob=prob_win_val,
                    effect_size=power_effect_size_val,
                    n_sims=int(S),
                    alpha=0.05,
                )
                if 'error' not in power_result:
                    statistical_analysis['power_analysis'] = power_result
            
            if enable_convergence_check_val and isinstance(best, dict):
                cumulative_metric, cumulative_trials, convergence_event = _objective_convergence_inputs(
                    paths,
                    best_row=best,
                )
                if (
                    cumulative_metric is not None
                    and cumulative_trials is not None
                    and convergence_event is not None
                ):
                    convergence_result = _mc_convergence(
                        cumulative_metric,
                        cumulative_trials,
                        window_size=convergence_window_val,
                        threshold=convergence_threshold_val,
                    )
                    convergence_result["event"] = convergence_event
                    convergence_result["objective"] = objective_val
                    convergence_result["tp_price"] = float(best.get('tp_price'))
                    convergence_result["sl_price"] = float(best.get('sl_price'))
                    statistical_analysis['convergence_diagnostic'] = convergence_result
            
            if enable_bootstrap_val:
                try:
                    tp_trigger = float(best.get('tp_price', 0))
                    sl_trigger = float(best.get('sl_price', 0))
                    if tp_trigger > 0 and sl_trigger > 0:
                        (
                            bootstrap_first_tp,
                            bootstrap_first_sl,
                            bootstrap_wins,
                            bootstrap_losses,
                            bootstrap_ties,
                        ) = _candidate_hit_arrays(
                            paths,
                            tp_trigger=tp_trigger,
                            sl_trigger=sl_trigger,
                            context=eval_context,
                            bridge_inputs=_BarrierBridgeInputs(
                                enabled=bb_enabled,
                                sigma=float(bb_sigma),
                                log_paths=bb_log_paths,
                                uniform_tp=bb_uniform_tp,
                                uniform_sl=bb_uniform_sl,
                            ),
                        )
                        bootstrap_unresolved = ~(
                            bootstrap_wins | bootstrap_losses | bootstrap_ties
                        )
                        bootstrap_outcomes = BarrierPathOutcomes(
                            first_tp=bootstrap_first_tp,
                            first_sl=bootstrap_first_sl,
                            wins=bootstrap_wins,
                            losses=bootstrap_losses,
                            ties=bootstrap_ties,
                            unresolved=bootstrap_unresolved,
                            time_in_trade=np.minimum(
                                np.minimum(bootstrap_first_tp, bootstrap_first_sl) + 1,
                                paths.shape[1],
                            ),
                            horizon=int(paths.shape[1]),
                        )
                        bootstrap_payoffs = barrier_path_payoffs(
                            paths,
                            bootstrap_outcomes,
                            entry_price=last_price,
                            reward=float(best.get('tp', 0.0)),
                            risk=float(best.get('sl', 0.0)),
                            direction=direction_norm,
                            mode=mode_val,
                            tick_size=float(tick_size) if tick_size else 0.0,
                            pip_size=float(pip_size) if pip_size else None,
                            cost_per_trade=float(cost_per_trade),
                            same_bar_policy=same_bar_policy_value,
                            gap_aware_stops=eval_context.gap_aware_stops,
                        )
                        bootstrap_result = _bootstrap_uncertainty(
                            paths=paths,
                            tp_trigger=tp_trigger,
                            sl_trigger=sl_trigger,
                            direction=direction_norm,
                            entry_price=last_price,
                            reward=float(best.get('tp', 0.0)),
                            risk=float(best.get('sl', 0.0)),
                            cost_per_trade=float(cost_per_trade),
                            same_bar_policy=same_bar_policy_value,
                            n_bootstrap=n_bootstrap_val,
                            seed=request_seed_base,
                            path_outcomes=bootstrap_outcomes,
                            path_payoffs=bootstrap_payoffs,
                        )
                        if bootstrap_result:
                            statistical_analysis['bootstrap_uncertainty'] = bootstrap_result
                except Exception:
                    pass
            
            if n_seeds_stability_val > 1:
                results_by_seed: Dict[int, Dict[str, Any]] = {}
                selection_counts: Dict[Tuple[int, int], int] = {}
                selected_pair_key = (
                    int(round(float(best.get('tp', 0.0)) * 1e9)),
                    int(round(float(best.get('sl', 0.0)) * 1e9)),
                )
                stability_candidates = _dedupe_ranked_barrier_candidates(
                    [dict(row) for row in results if isinstance(row, dict)]
                )
                stability_pairs = [
                    (float(row['tp']), float(row['sl']))
                    for row in stability_candidates
                    if row.get('tp') is not None and row.get('sl') is not None
                ]
                holdout_selected_row: Optional[Dict[str, Any]] = None
                holdout_seed_origin = max(1, int(n_seeds))
                for seed_offset in range(
                    holdout_seed_origin,
                    holdout_seed_origin + min(n_seeds_stability_val, 5),
                ):
                    seed_key = offset_barrier_seed(request_seed_base, seed_offset)
                    try:
                        (
                            stability_paths,
                            stability_bb_enabled,
                            stability_bb_sigma,
                            stability_bb_log_paths,
                            stability_bb_uniform_tp,
                            stability_bb_uniform_sl,
                        ) = _simulate_paths_for_seed_range(seed_base=seed_key, seed_count=1)
                    except (ValueError, RuntimeError, np.linalg.LinAlgError):
                        continue
                    seed_rows = _evaluate(
                        stability_pairs,
                        stability_paths,
                        stability_bb_enabled,
                        stability_bb_sigma,
                        stability_bb_log_paths,
                        stability_bb_uniform_tp,
                        stability_bb_uniform_sl,
                        count_invalid=False,
                    )
                    if seed_rows:
                        _sort_candidate_results(seed_rows, objective_val)
                        seed_views = _select_barrier_candidate_views(
                            seed_rows,
                            cost_per_trade=cost_per_trade,
                            viable_only_val=viable_only_val,
                            tradable_only_val=tradable_only_val,
                            min_ev_val=min_ev_val,
                            min_edge_val=min_edge_val,
                            min_kelly_val=min_kelly_val,
                            concise_val=False,
                            top_k_val=None,
                            return_grid=False,
                            output_mode='full',
                        )
                        seed_candidates = seed_views['candidates']
                        if not seed_candidates:
                            seed_candidates = seed_views['ranked_candidates']
                        if not seed_candidates:
                            continue
                        seed_best = seed_candidates[0]
                        seed_pair_key = (
                            int(round(float(seed_best['tp']) * 1e9)),
                            int(round(float(seed_best['sl']) * 1e9)),
                        )
                        selection_counts[seed_pair_key] = (
                            selection_counts.get(seed_pair_key, 0) + 1
                        )
                        selected_pair_row = next(
                            (
                                row
                                for row in seed_rows
                                if (
                                    int(round(float(row['tp']) * 1e9)),
                                    int(round(float(row['sl']) * 1e9)),
                                ) == selected_pair_key
                            ),
                            None,
                        )
                        # Metric stability must hold the barrier pair fixed.
                        # Using each seed's own winner would mix estimator noise
                        # with selection changes, which `selection_stability`
                        # reports separately.
                        if selected_pair_row is not None:
                            results_by_seed[seed_key] = selected_pair_row
                        if holdout_selected_row is None:
                            holdout_selected_row = selected_pair_row

                if len(results_by_seed) > 1:
                    stability_result = _cross_seed_stability(
                        results_by_seed=results_by_seed,
                        threshold_cv=0.10,
                    )
                    stability_result["seeds_attempted"] = int(min(n_seeds_stability_val, 5))
                    stability_result["seeds_succeeded"] = int(len(results_by_seed))
                    stability_result["metric_basis"] = "selected_barrier_pair_fixed"
                    selections_observed = max(1, sum(selection_counts.values()))
                    selected_frequency = (
                        selection_counts.get(selected_pair_key, 0) / selections_observed
                    )
                    modal_pair, modal_count = max(
                        selection_counts.items(),
                        key=lambda item: item[1],
                    )
                    stability_result["selection_stability"] = {
                        "stable": bool(selected_frequency >= 0.60),
                        "selected_pair_frequency": float(selected_frequency),
                        "seeds_with_selection": int(selections_observed),
                        "unique_pairs_selected": int(len(selection_counts)),
                        "modal_pair": {
                            "tp": float(modal_pair[0] / 1e9),
                            "sl": float(modal_pair[1] / 1e9),
                            "frequency": float(modal_count / selections_observed),
                        },
                        "selection_counts": [
                            {
                                "tp": float(pair[0] / 1e9),
                                "sl": float(pair[1] / 1e9),
                                "count": int(count),
                            }
                            for pair, count in sorted(selection_counts.items())
                        ],
                    }
                    statistical_analysis['cross_seed_stability'] = stability_result
                else:
                    statistical_analysis['cross_seed_stability'] = {
                        "stable": False,
                        "error": "Need at least 2 successful seed re-runs for stability analysis",
                        "seeds_attempted": int(min(n_seeds_stability_val, 5)),
                        "seeds_succeeded": int(len(results_by_seed)),
                        "recommendation": "Retry with a supported stochastic method or fewer failure-prone seeds.",
                    }
                if holdout_selected_row is not None:
                    selected_objective = _safe_float(best.get(objective_val))
                    holdout_objective = _safe_float(
                        holdout_selected_row.get(objective_val)
                    )
                    statistical_analysis['post_selection_evaluation'] = {
                        "source": "independent_seed",
                        "seed": int(offset_barrier_seed(request_seed_base, holdout_seed_origin)),
                        "tp": float(best.get('tp')),
                        "sl": float(best.get('sl')),
                        "objective": objective_val,
                        "selection_estimate": selected_objective,
                        "holdout_estimate": holdout_objective,
                        "optimism": (
                            float(selected_objective - holdout_objective)
                            if selected_objective is not None
                            and holdout_objective is not None
                            else None
                        ),
                        "metrics": {
                            key: holdout_selected_row.get(key)
                            for key in (
                                'ev', 'edge', 'prob_tp_first', 'prob_sl_first',
                                'prob_no_hit', 'kelly', 'utility',
                            )
                        },
                    }

            if enable_drift_stress_val and drift_multipliers_val:
                historical_returns = _log_returns_from_prices(prices)
                historical_returns = historical_returns[np.isfinite(historical_returns)]
                historical_drift = (
                    float(np.mean(historical_returns))
                    if historical_returns.size else 0.0
                )
                steps = np.arange(1, paths.shape[1] + 1, dtype=float)
                drift_scenarios: List[Dict[str, Any]] = []
                for multiplier in drift_multipliers_val:
                    adjustment = np.exp(
                        (float(multiplier) - 1.0) * historical_drift * steps
                    )
                    stressed_paths = paths * adjustment[None, :]
                    stressed_log_paths = None
                    if bb_enabled:
                        stressed_log_paths = np.concatenate(
                            [
                                np.full(
                                    (stressed_paths.shape[0], 1),
                                    np.log(max(last_price, 1e-12)),
                                ),
                                np.log(np.clip(stressed_paths, 1e-12, None)),
                            ],
                            axis=1,
                        )
                    stressed_rows = _evaluate(
                        [(float(best['tp']), float(best['sl']))],
                        stressed_paths,
                        bb_enabled,
                        bb_sigma,
                        stressed_log_paths,
                        bb_uniform_tp,
                        bb_uniform_sl,
                        count_invalid=False,
                    )
                    if stressed_rows:
                        stressed_row = stressed_rows[0]
                        drift_scenarios.append({
                            "multiplier": float(multiplier),
                            "drift_per_bar": float(historical_drift * multiplier),
                            "ev": stressed_row.get('ev'),
                            "edge": stressed_row.get('edge'),
                            "prob_tp_first": stressed_row.get('prob_tp_first'),
                            "prob_sl_first": stressed_row.get('prob_sl_first'),
                            "prob_no_hit": stressed_row.get('prob_no_hit'),
                            "objective_value": stressed_row.get(objective_val),
                        })
                if drift_scenarios:
                    finite_evs = [
                        float(row['ev'])
                        for row in drift_scenarios
                        if _optional_finite_float(row.get('ev')) is not None
                    ]
                    statistical_analysis['drift_stress'] = {
                        "historical_drift_per_bar": historical_drift,
                        "scenarios": drift_scenarios,
                        "worst_ev": min(finite_evs) if finite_evs else None,
                        "best_ev": max(finite_evs) if finite_evs else None,
                    }

            if enable_oos_validation_val:
                last_entry = int(len(prices) - horizon_val - 1)
                first_entry = max(50, int(len(prices) - oos_holdout_bars_val - 1))
                if last_entry >= first_entry:
                    fold_entries = sorted(set(
                        np.linspace(
                            first_entry,
                            last_entry,
                            oos_folds_val,
                            dtype=int,
                        ).tolist()
                    ))
                    walk_forward_pairs = list(base_candidates)
                    fold_results: List[Dict[str, Any]] = []
                    for fold_index, entry_index in enumerate(fold_entries):
                        training_prices = np.asarray(
                            prices[:entry_index + 1],
                            dtype=float,
                        )
                        try:
                            (
                                fold_paths,
                                fold_bb_enabled,
                                fold_bb_sigma,
                                fold_bb_log_paths,
                                fold_bb_uniform_tp,
                                fold_bb_uniform_sl,
                            ) = _simulate_paths_for_seed_range(
                                seed_base=offset_barrier_seed(
                                    request_seed_base,
                                    100 + fold_index,
                                ),
                                seed_count=1,
                                history_prices=training_prices,
                                sims_override=oos_sims_val,
                            )
                        except (ValueError, RuntimeError, np.linalg.LinAlgError):
                            continue
                        fold_rows = _evaluate(
                            walk_forward_pairs,
                            fold_paths,
                            fold_bb_enabled,
                            fold_bb_sigma,
                            fold_bb_log_paths,
                            fold_bb_uniform_tp,
                            fold_bb_uniform_sl,
                            count_invalid=False,
                        )
                        if not fold_rows:
                            continue
                        _sort_candidate_results(fold_rows, objective_val)
                        fold_best = fold_rows[0]
                        entry_price_actual = float(prices[entry_index])
                        future_actual = np.asarray(
                            prices[entry_index + 1:entry_index + 1 + horizon_val],
                            dtype=float,
                        )
                        if future_actual.size != horizon_val or entry_price_actual <= 0.0:
                            continue
                        actual_path = (
                            future_actual / entry_price_actual * float(last_price)
                        )[None, :]
                        actual_rows = _evaluate(
                            [(float(fold_best['tp']), float(fold_best['sl']))],
                            actual_path,
                            False,
                            0.0,
                            None,
                            None,
                            None,
                            count_invalid=False,
                        )
                        if not actual_rows:
                            continue
                        actual_row = actual_rows[0]
                        fold_results.append({
                            "entry_index": int(entry_index),
                            "training_bars": int(entry_index + 1),
                            "tp": float(fold_best['tp']),
                            "sl": float(fold_best['sl']),
                            "selection_objective": fold_best.get(objective_val),
                            "realized_ev": actual_row.get('ev'),
                            "realized_outcome": (
                                "tp" if actual_row.get('prob_tp_first') == 1.0
                                else "sl" if actual_row.get('prob_sl_first') == 1.0
                                else "unresolved"
                            ),
                            "realized_no_hit": actual_row.get('prob_no_hit'),
                        })
                    if fold_results:
                        realized_values = np.asarray(
                            [float(row['realized_ev']) for row in fold_results],
                            dtype=float,
                        )
                        statistical_analysis['walk_forward_oos'] = {
                            "enabled": True,
                            "folds_requested": int(oos_folds_val),
                            "folds_completed": int(len(fold_results)),
                            "holdout_bars": int(oos_holdout_bars_val),
                            "n_sims_per_fold": int(oos_sims_val),
                            "mean_realized_ev": float(np.mean(realized_values)),
                            "positive_fold_rate": float(np.mean(realized_values > 0.0)),
                            "candidate_space": "request_grid_without_refinement",
                            "path_basis": "historical_close_only",
                            "folds": fold_results,
                        }
                    else:
                        statistical_analysis['walk_forward_oos'] = {
                            "enabled": True,
                            "error": "No walk-forward folds completed successfully.",
                        }
                else:
                    statistical_analysis['walk_forward_oos'] = {
                        "enabled": True,
                        "error": "Insufficient history for the requested holdout and horizon.",
                    }

            if enable_sensitivity_analysis_val and sensitivity_params_requested:
                base_result = {"best": best}
                sensitivity_results: Dict[str, Any] = {}

                def _local_sensitivity_values(base_value: float) -> List[float]:
                    values: List[float] = []
                    seen_values: Set[int] = set()
                    for multiplier in (0.8, 0.9, 1.0, 1.1, 1.2):
                        value = max(min_barrier_distance, float(base_value) * multiplier)
                        key = int(round(value * 1e6))
                        if key in seen_values:
                            continue
                        seen_values.add(key)
                        values.append(value)
                    return values

                def _evaluate_sensitivity(override: Dict[str, float]) -> Dict[str, Any]:
                    tp_unit = float(override.get('tp', best.get('tp', 0.0)))
                    sl_unit = float(override.get('sl', best.get('sl', 0.0)))
                    rows = _evaluate(
                        [(tp_unit, sl_unit)],
                        paths,
                        bb_enabled,
                        bb_sigma,
                        bb_log_paths,
                        bb_uniform_tp,
                        bb_uniform_sl,
                        count_invalid=False,
                    )
                    if not rows:
                        return {"success": False}
                    return {"success": True, "best": rows[0]}

                for param_name in sensitivity_params_requested:
                    if param_name not in {'tp', 'sl'}:
                        continue
                    base_value_raw = best.get(param_name)
                    if base_value_raw is None:
                        continue
                    try:
                        parameter_values = _local_sensitivity_values(float(base_value_raw))
                    except (TypeError, ValueError):
                        continue
                    sensitivity_result = _sensitivity_analysis(
                        base_result=base_result,
                        parameter_name=param_name,
                        parameter_values=parameter_values,
                        evaluate_fn=_evaluate_sensitivity,
                    )
                    if sensitivity_result.get("success"):
                        sensitivity_results[param_name] = sensitivity_result

                if sensitivity_results:
                    statistical_analysis["sensitivity_analysis"] = sensitivity_results
            
            if statistical_analysis:
                out['statistical_robustness'] = statistical_analysis
                out['min_sims_recommended'] = int(min_sims_recommended)
        
        diagnostics = {}
        if isinstance(best, dict):
            diagnostics = _build_selection_diagnostics(
                best,
                cost_per_trade=cost_per_trade,
                candidate_count=max(
                    len(results),
                    int(optuna_trials_val) if optimizer_val == 'optuna' else 0,
                ),
            )
            diagnostics.update(
                _post_selection_gate_diagnostics(
                    statistical_analysis,
                    objective=objective_val,
                )
            )
            out.update(diagnostics)
        if not cost_model_complete:
            diagnostics["trading_costs_incomplete"] = True
            diagnostics["missing_cost_assumptions"] = list(
                missing_cost_assumptions
            )
            out.update(diagnostics)
            out["remediation"] = (
                "Pass --spread-bps, --commission-bps-per-side, and --slippage-bps "
                "(use 0 to model a cost as free) so expected value is net of "
                "trading costs. Pip- or percent-denominated values remain available "
                "via --params when needed."
            )
        if warning is not None:
            out["warning"] = warning
        elif best is not None and not viable:
            out["advice"] = [
                "Increase horizon to allow more time for barrier resolution.",
                "Try the opposite direction and compare objective metrics.",
                "Widen TP/SL search ranges or switch grid_style to volatility/ratio.",
                "Skip this setup if edge and EV remain unattractive.",
            ]
        actionability_payload = _build_actionability_payload(
            status=status,
            status_reason=status_reason,
            row=best,
            diagnostics=diagnostics,
            warning=out.get("warning"),
        )
        out.update(actionability_payload)
        out["no_action"] = not bool(actionability_payload.get("trade_gate_passed"))
        _attach_viability_semantics(
            out,
            mathematically_viable=bool(viable),
            trade_gate_passed=actionability_payload.get("trade_gate_passed"),
        )
        _apply_barrier_freshness_contract(
            out,
            history_context=freshness_context,
            reference_context=reference_context,
            last_price_source=last_price_source,
        )
        if invalid_barrier_candidates > 0:
            out["barrier_sanity_filtered"] = int(invalid_barrier_candidates)
        if min_prob_resolve_val is not None:
            out["min_prob_resolve"] = float(min_prob_resolve_val)
        if objective_changed:
            out["objective_requested"] = objective_requested
            out["objective_used"] = objective_val
        if method_requested != method_name:
            out["method_requested"] = method_requested
            out["method_used"] = method_name
            if auto_reason:
                out["auto_reason"] = auto_reason
        if bb_enabled:
            out["bridge_correction"] = True
            out["bridge_dual_barrier_model"] = BROWNIAN_BRIDGE_DUAL_BARRIER_MODEL
            out["bridge_joint_first_passage"] = False
            warnings_out = list(out.get("warnings") or [])
            warnings_out.append(BROWNIAN_BRIDGE_DUAL_BARRIER_WARNING)
            out["warnings"] = warnings_out
        out["trading_costs"] = dict(trading_costs_payload)
        if method_name == "hmm_mc" and hmm_sim_meta_records:
            requested_states = max(
                row["requested_n_states"] for row in hmm_sim_meta_records
            )
            fitted_states_observed = sorted(
                {row["fitted_n_states"] for row in hmm_sim_meta_records}
            )
            collapsed_batches = sum(
                row["fitted_n_states"] < row["requested_n_states"]
                for row in hmm_sim_meta_records
            )
            out["sim_meta"] = {
                "requested_n_states": requested_states,
                "fitted_n_states": min(fitted_states_observed),
                "fitted_n_states_observed": fitted_states_observed,
                "simulation_batches": len(hmm_sim_meta_records),
                "collapsed_batches": int(collapsed_batches),
            }
            if collapsed_batches:
                warnings_out = list(out.get("warnings") or [])
                warnings_out.append(
                    "HMM state collapse detected: requested "
                    f"{requested_states} states but fitted fewer states in "
                    f"{collapsed_batches} of {len(hmm_sim_meta_records)} simulation "
                    "batches; optimization used the reduced-state paths."
                )
                out["warnings"] = warnings_out
            if fitted_states_observed == [1]:
                out["effective_method"] = "single_regime_gaussian_mc"
        return _finalize_barrier_output(
            out,
            output_mode=output_mode,
            concise_val=concise_val,
            viable_only_val=viable_only_val,
            ranked_candidates=ranked_candidates,
            viable_candidates=viable_candidates,
            candidates=candidates,
            grid_out=grid_out,
            return_grid=return_grid,
            top_k_val=top_k_val,
            selection_diagnostics=diagnostics,
        )

    except (KeyError, AttributeError, IndexError):
        raise
    except Exception as e:
        return {
            "error": f"Error optimizing barriers: {str(e)}",
        }

