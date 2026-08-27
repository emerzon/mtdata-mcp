import hashlib
import json
import math
import warnings
from datetime import datetime, timezone
from statistics import NormalDist
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np

from ..shared.constants import TIMEFRAME_SECONDS
from ..shared.schema import DenoiseSpec, TimeframeLiteral
from ..shared.symbols import is_probably_crypto_symbol
from ..utils.barriers import (
    barrier_prices_are_valid as _barrier_prices_are_valid,
)
from ..utils.barriers import (
    normalize_trade_direction,
)
from ..utils.barriers import (
    resolve_barrier_prices as _resolve_barrier_prices,
)
from ..utils.coercion import coerce_finite_float
from ..utils.coercion import safe_float as _safe_float
from ..utils.freshness import (
    completed_bar_freshness_fields,
    format_age_seconds,
)
from ..utils.market_metadata import build_tick_freshness_context
from ..utils.quote import (
    compute_spread_metrics,
    enforce_quote_execution_readiness,
    resolve_quote_tick,
    tick_epoch,
    tick_value,
)
from ..utils.time import (
    _format_time_minimal,
    _format_time_minimal_local,
    _use_client_tz,
)
from ..utils.utils import parse_kv_or_json as _parse_kv_or_json
from .barrier_constants import BARRIER_MONTE_CARLO_METHODS
from .barrier_stats import _confidence_interval_wilson_proportion
from .common import fetch_history as _fetch_history
from .common import log_returns_from_prices as _log_returns_from_prices
from .monte_carlo import gbm_single_barrier_upcross_prob as _gbm_upcross_prob
from .monte_carlo import simulate_bootstrap_mc as _simulate_bootstrap_mc
from .monte_carlo import simulate_garch_mc as _simulate_garch_mc
from .monte_carlo import simulate_gbm_mc as _simulate_gbm_mc
from .monte_carlo import simulate_heston_mc as _simulate_heston_mc
from .monte_carlo import simulate_hmm_mc as _simulate_hmm_mc
from .monte_carlo import simulate_jump_diffusion_mc as _simulate_jump_diffusion_mc

BARRIER_GRID_PRESETS = {
    'scalp': {
        'tp_min': 0.08, 'tp_max': 0.60, 'tp_steps': 7,
        'sl_min': 0.20, 'sl_max': 1.20, 'sl_steps': 7,
    },
    'intraday': {
        'tp_min': 0.25, 'tp_max': 1.50, 'tp_steps': 7,
        'sl_min': 0.25, 'sl_max': 2.50, 'sl_steps': 9,
    },
    'swing': {
        'tp_min': 0.60, 'tp_max': 3.50, 'tp_steps': 7,
        'sl_min': 0.50, 'sl_max': 4.50, 'sl_steps': 8,
    },
    'position': {
        'tp_min': 1.00, 'tp_max': 8.00, 'tp_steps': 8,
        'sl_min': 0.75, 'sl_max': 6.00, 'sl_steps': 8,
    },
}

PHANTOM_PROFIT_NO_HIT_THRESHOLD = 0.50
PHANTOM_PROFIT_LOSS_THRESHOLD = 0.01
DEGENERATE_OBJECTIVE_MIN_RESOLVE = 0.20
LOW_CONFIDENCE_CI_THRESHOLD = 0.10
LOW_PRACTICAL_WIN_PROB_THRESHOLD = 0.05
BARRIER_METRIC_BASIS_NOTE = (
    "ev=mean simulated barrier payoff plus timeout mark-to-market in distance_unit, "
    "net of supplied costs; neutral same-bar ties contribute zero; "
    "edge=prob_win-prob_loss; edge_vs_breakeven="
    "prob_win_resolved-breakeven_win_rate; "
    "profit_factor=resolved reward/loss; higher is better, positive ev/edge is favorable."
)
BARRIER_EDGE_DEFINITION = "prob_win - prob_loss (probability fraction)."
BROWNIAN_BRIDGE_DUAL_BARRIER_MODEL = "independent_single_barrier_approximation"
BROWNIAN_BRIDGE_DUAL_BARRIER_WARNING = (
    "Brownian-bridge TP and SL hits are sampled independently within each "
    "bar; dual-barrier first-passage ordering is approximate and can overstate "
    "same-bar ties when barriers are tight."
)

_RANDOM_SEED_MODULUS = 2**32


def _stable_barrier_seed(*parts: Any) -> int:
    payload = json.dumps(
        parts,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & int(np.iinfo(np.int32).max)


def normalize_barrier_seed(seed: Any) -> int:
    return int(seed) % _RANDOM_SEED_MODULUS


def offset_barrier_seed(seed_base: Any, offset: int = 0) -> int:
    return normalize_barrier_seed(int(seed_base) + int(offset))


def normalize_barrier_method(
    method: Any,
    *,
    allow_closed_form: bool = False,
    allow_ensemble: bool = False,
) -> Optional[str]:
    method_key = str(method or "").strip().lower()
    allowed = set(BARRIER_MONTE_CARLO_METHODS)
    if allow_closed_form:
        allowed.add("closed_form")
    if allow_ensemble:
        allowed.add("ensemble")
    return method_key if method_key in allowed else None


def resolve_barrier_prob_method(
    method: Any,
    barrier_kind: Any,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Resolve omitted/auto barrier-prob methods after inspecting barrier.kind.

    Returns ``(effective_method, method_source, method_requested, auto_reason)``.
    Explicit ``auto`` with ``tp_sl`` keeps ``effective_method='auto'`` so the
    simulation engine can still choose among Monte Carlo models.
    """
    requested = str(method or "").strip().lower() or None
    kind = str(barrier_kind or "").strip().lower()
    if requested is None:
        if kind == "single_price":
            return "closed_form", "auto_for_barrier_kind", None, None
        return "mc_gbm_bb", "auto_for_barrier_kind", None, None
    if requested == "auto":
        if kind == "single_price":
            return (
                "closed_form",
                "auto_selection",
                "auto",
                "auto: single_price barrier; closed_form",
            )
        return "auto", "auto_selection", "auto", None
    return requested, "request", None, None


def barrier_method_error(
    method: Any,
    *,
    allow_closed_form: bool = False,
    allow_ensemble: bool = False,
) -> str:
    allowed = list(BARRIER_MONTE_CARLO_METHODS)
    if allow_closed_form:
        allowed.append("closed_form")
    if allow_ensemble:
        allowed.append("ensemble")
    return f"Unsupported barrier method: {method}. Use one of: {', '.join(allowed)}."


def _binomial_wilson_95(p_hat: float, n: int) -> Tuple[float, float]:
    """Wilson 95% interval for a Bernoulli proportion."""
    return _confidence_interval_wilson_proportion(float(p_hat), int(n), confidence=0.95)


def _binomial_se(p_hat: float, n: int) -> float:
    n_i = int(n)
    if n_i <= 0:
        return float("nan")
    p = float(np.clip(float(p_hat), 0.0, 1.0))
    return float(math.sqrt(max(0.0, p * (1.0 - p) / n_i)))


def _least_negative_ref(best_row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Compact pointer payload for non-viable outputs to avoid duplicating `best`."""
    if not isinstance(best_row, dict):
        return None
    return {
        "ref": "best",
        "ev": best_row.get("ev"),
        "edge": best_row.get("edge"),
        "edge_vs_breakeven": best_row.get("edge_vs_breakeven"),
        "kelly": best_row.get("kelly"),
        "phantom_profit_risk": best_row.get("phantom_profit_risk"),
        "reason": "No viable candidate was found; see `best` for full details.",
    }


def _scale_price_paths_to_reference(
    price_paths: np.ndarray,
    *,
    simulated_anchor_price: Any,
    reference_price: Any,
) -> np.ndarray:
    """Rescale simulated paths so barrier scoring uses the same entry anchor."""
    paths = np.asarray(price_paths, dtype=float)
    anchor = _safe_float(simulated_anchor_price)
    ref = _safe_float(reference_price)
    if anchor is None or anchor <= 0.0 or ref is None or ref <= 0.0:
        return paths
    scale = float(ref / anchor)
    if not np.isfinite(scale) or scale <= 0.0 or abs(scale - 1.0) <= 1e-12:
        return paths
    return paths * scale


def _prepare_brownian_bridge_draws(
    price_paths: np.ndarray,
    *,
    calibration_prices: Any,
    last_price_close: Any,
    reference_price: Any,
    bb_enabled: bool,
    seed_base: Any,
    seed_offset: int = 7,
) -> Tuple[
    np.ndarray,
    bool,
    float,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Scale simulated paths to the live reference and optionally build BB draws.

    Returns ``(paths, bb_enabled, bb_sigma, bb_log_paths, uniform_tp, uniform_sl)``.
    When sigma is non-finite or non-positive, Brownian-bridge extras are disabled
    and the scaled paths are still returned.
    """
    try:
        sim_anchor_price = float(np.asarray(calibration_prices, dtype=float).ravel()[-1])
    except Exception:
        sim_anchor_price = float(last_price_close)
    paths = _scale_price_paths_to_reference(
        price_paths,
        simulated_anchor_price=sim_anchor_price,
        reference_price=reference_price,
    )

    bb_sigma = 0.0
    bb_log_paths = None
    bb_uniform_tp = None
    bb_uniform_sl = None
    enabled = bool(bb_enabled)
    if not enabled:
        return paths, False, bb_sigma, None, None, None

    rets = _log_returns_from_prices(np.asarray(calibration_prices, dtype=float))
    rets = rets[np.isfinite(rets)]
    bb_sigma = float(np.std(rets, ddof=1)) if rets.size else 0.0
    if not np.isfinite(bb_sigma) or bb_sigma <= 0:
        return paths, False, bb_sigma, None, None, None

    sims_total, horizon = paths.shape
    log_paths = np.log(np.clip(paths, 1e-12, None))
    log_s0 = float(np.log(max(float(reference_price), 1e-12)))
    bb_log_paths = np.concatenate(
        [np.full((sims_total, 1), log_s0), log_paths],
        axis=1,
    )
    rng_bb = np.random.RandomState(offset_barrier_seed(seed_base, seed_offset))
    bb_uniform_tp = rng_bb.rand(sims_total, horizon)
    bb_uniform_sl = rng_bb.rand(sims_total, horizon)
    return paths, True, bb_sigma, bb_log_paths, bb_uniform_tp, bb_uniform_sl


def _sort_candidate_results(res_list: List[Dict[str, Any]], objective_val: str) -> None:
    def _candidate_geometry(row: Dict[str, Any]) -> Tuple[float, ...]:
        """Provide a deterministic tie-break without changing grid-order semantics."""

        def _value(key: str) -> float:
            value = _safe_float(row.get(key))
            return float("inf") if value is None else float(value)

        return tuple(_value(key) for key in ("tp", "sl", "tp_price", "sl_price"))

    def _metric(key: str, *, descending: bool) -> Any:
        default = float("-inf") if descending else float("inf")

        def _resolve(row: Dict[str, Any]) -> float:
            value = _safe_float(row.get(key))
            return default if value is None else float(value)

        return _resolve

    # Candidate evaluation is normally generated in ascending TP/SL grid order,
    # but parallel optimizers can return rows in completion order.  Establish the
    # same canonical geometry order first; the stable objective sort below then
    # preserves it whenever objective values tie.
    res_list.sort(key=_candidate_geometry)

    if objective_val == 'edge':
        res_list.sort(key=_metric('edge', descending=True), reverse=True)
    elif objective_val == 'ev':
        res_list.sort(key=_metric('ev', descending=True), reverse=True)
    elif objective_val == 'ev_cond':
        res_list.sort(key=_metric('ev_cond', descending=True), reverse=True)
    elif objective_val == 'ev_per_bar':
        res_list.sort(key=_metric('ev_per_bar', descending=True), reverse=True)
    elif objective_val == 'kelly':
        res_list.sort(key=_metric('kelly', descending=True), reverse=True)
    elif objective_val == 'kelly_cond':
        res_list.sort(key=_metric('kelly_cond', descending=True), reverse=True)
    elif objective_val == 'prob_tp_first':
        res_list.sort(key=_metric('prob_tp_first', descending=True), reverse=True)
    elif objective_val == 'prob_resolve':
        res_list.sort(key=_metric('prob_resolve', descending=True), reverse=True)
    elif objective_val == 'profit_factor':
        def _profit_factor_rank(row: Dict[str, Any]) -> float:
            value = _safe_float(row.get('profit_factor'))
            if value is not None:
                return float(value)
            win_prob = _safe_float(row.get('prob_tp_first'))
            loss_prob = _safe_float(row.get('prob_sl_first'))
            reward = _safe_float(row.get('tp'))
            if win_prob is not None and win_prob > 0.0 and reward is not None and reward > 0.0:
                if loss_prob is not None and loss_prob <= 0.0:
                    return float('inf')
            return float('-inf')

        res_list.sort(key=_profit_factor_rank, reverse=True)
    elif objective_val == 'min_loss_prob':
        res_list.sort(key=_metric('prob_loss', descending=False))
    elif objective_val == 'utility':
        res_list.sort(key=_metric('utility', descending=True), reverse=True)
    else:
        res_list.sort(key=_metric('ev', descending=True), reverse=True)


def _annotate_candidate_metrics(row: Optional[Dict[str, Any]], cost_per_trade: float = 0.0) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return row

    rr = _safe_float(row.get("rr"))
    prob_win = _safe_float(row.get("prob_win"))
    prob_loss = _safe_float(row.get("prob_loss"))
    prob_tp_first = _safe_float(row.get("prob_tp_first"))
    prob_sl_first = _safe_float(row.get("prob_sl_first"))
    prob_no_hit = _safe_float(row.get("prob_no_hit"))
    prob_resolve = _safe_float(row.get("prob_resolve"))
    tp = _safe_float(row.get("tp"))
    sl = _safe_float(row.get("sl"))
    cost = max(0.0, float(cost_per_trade))
    effective_prob_win = prob_tp_first if prob_tp_first is not None else prob_win
    effective_prob_loss = prob_sl_first if prob_sl_first is not None else prob_loss

    if prob_resolve is None and prob_no_hit is not None:
        prob_resolve = float(max(0.0, min(1.0, 1.0 - prob_no_hit)))
        row["prob_resolve"] = prob_resolve
    if prob_no_hit is None and prob_resolve is not None:
        prob_no_hit = float(max(0.0, min(1.0, 1.0 - prob_resolve)))
        row["prob_no_hit"] = prob_no_hit

    breakeven_win_rate: Optional[float] = None
    if rr is not None and rr > 0.0:
        breakeven_win_rate = float(1.0 / (1.0 + rr))
        row["breakeven_win_rate"] = breakeven_win_rate

    breakeven_win_rate_net: Optional[float] = None
    if cost > 0.0 and tp is not None and sl is not None:
        net_reward = tp - cost
        net_risk = sl + cost
        if net_reward > 0.0 and net_risk > 0.0:
            breakeven_win_rate_net = float(net_risk / (net_reward + net_risk))
            row["breakeven_win_rate_net"] = breakeven_win_rate_net
        elif net_reward <= 0.0:
            breakeven_win_rate_net = 1.0
            row["breakeven_win_rate_net"] = breakeven_win_rate_net

    edge_vs_breakeven: Optional[float] = None
    prob_win_resolved: Optional[float] = None
    effective_breakeven = breakeven_win_rate_net if breakeven_win_rate_net is not None else breakeven_win_rate
    if effective_prob_win is not None and effective_prob_loss is not None:
        active_probability = effective_prob_win + effective_prob_loss
        if active_probability > 0.0:
            prob_win_resolved = float(effective_prob_win / active_probability)
            row["prob_win_resolved"] = prob_win_resolved
            row["prob_loss_resolved"] = float(effective_prob_loss / active_probability)
    if prob_win_resolved is not None and effective_breakeven is not None:
        edge_vs_breakeven = float(prob_win_resolved - effective_breakeven)
        row["edge_vs_breakeven"] = edge_vs_breakeven
        row["resolved_edge_vs_breakeven"] = edge_vs_breakeven
        row["edge_vs_breakeven_basis"] = "resolved_trades"

    phantom_profit_risk = bool(
        effective_prob_loss is not None
        and prob_no_hit is not None
        and effective_prob_loss < PHANTOM_PROFIT_LOSS_THRESHOLD
        and prob_no_hit > PHANTOM_PROFIT_NO_HIT_THRESHOLD
    )
    row["phantom_profit_risk"] = phantom_profit_risk

    prob_win_ci = row.get("prob_win_ci95")
    low_confidence = False
    if isinstance(prob_win_ci, dict):
        ci_lo = _safe_float(prob_win_ci.get("low"))
        ci_hi = _safe_float(prob_win_ci.get("high"))
        if ci_lo is not None and ci_hi is not None:
            ci_width = ci_hi - ci_lo
            low_confidence = ci_width > LOW_CONFIDENCE_CI_THRESHOLD
            row["prob_win_ci_width"] = float(ci_width)
    row["low_confidence"] = low_confidence

    return row


def _candidate_is_viable(row: Optional[Dict[str, Any]], cost_per_trade: float = 0.0) -> bool:
    if not isinstance(row, dict):
        return False
    _annotate_candidate_metrics(row, cost_per_trade=cost_per_trade)
    ev_value = _safe_float(row.get("ev"))
    if ev_value is None or ev_value < 0.0:
        return False
    if bool(row.get("ev_timeout_dominated")):
        return False
    if bool(row.get("phantom_profit_risk")):
        return False
    win_rate = _safe_float(row.get("prob_tp_first"))
    if win_rate is None:
        win_rate = _safe_float(row.get("prob_win"))
    if win_rate is not None and win_rate < LOW_PRACTICAL_WIN_PROB_THRESHOLD:
        return False
    return True


def _candidate_status_reason(row: Optional[Dict[str, Any]], cost_per_trade: float = 0.0) -> Optional[str]:
    if not isinstance(row, dict):
        return None
    _annotate_candidate_metrics(row, cost_per_trade=cost_per_trade)
    ev_value = _safe_float(row.get("ev"))
    if ev_value is None:
        return "Selected candidate is missing a finite EV estimate."
    if ev_value < 0.0:
        return "Selected candidate has negative EV."
    if bool(row.get("ev_timeout_dominated")):
        return (
            "Selected candidate's positive EV is dominated by unresolved timeout "
            "mark-to-market rather than resolved barrier outcomes."
        )
    if bool(row.get("phantom_profit_risk")):
        return (
            "Selected candidate relies on unresolved paths and a near-zero loss rate "
            "to appear profitable."
        )
    win_rate = _safe_float(row.get("prob_tp_first"))
    if win_rate is None:
        win_rate = _safe_float(row.get("prob_win"))
    if win_rate is not None and win_rate < LOW_PRACTICAL_WIN_PROB_THRESHOLD:
        return (
            f"Selected candidate win probability is below the "
            f"{LOW_PRACTICAL_WIN_PROB_THRESHOLD:.0%} practical threshold."
        )
    return None


def _build_selection_diagnostics(  # noqa: C901
    row: Optional[Dict[str, Any]],
    cost_per_trade: float = 0.0,
    *,
    candidate_count: int = 1,
) -> Dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    _annotate_candidate_metrics(row, cost_per_trade=cost_per_trade)

    warnings_out: List[str] = []
    ev_edge_conflict = False
    ev_edge_conflict_reason: Optional[str] = None
    caution: Optional[str] = None

    best_ev = _safe_float(row.get("ev"))
    if best_ev is not None and best_ev < 0.0:
        warnings_out.append(
            "Best candidate has negative EV; objective preference may not align with profitability."
        )

    ev_se = _safe_float(row.get("ev_se"))
    statistical_edge_unresolved = False
    selection_adjusted_ci: Optional[Dict[str, float]] = None
    comparisons = max(1, int(candidate_count))
    if best_ev is not None and ev_se is not None and ev_se >= 0.0:
        adjusted_alpha = 0.05 / comparisons
        adjusted_z = NormalDist().inv_cdf(1.0 - adjusted_alpha / 2.0)
        adjusted_low = float(best_ev - adjusted_z * ev_se)
        adjusted_high = float(best_ev + adjusted_z * ev_se)
        selection_adjusted_ci = {
            "low": adjusted_low,
            "high": adjusted_high,
        }
        row["ev_selection_adjusted_ci95"] = selection_adjusted_ci
        row["ev_selection_comparisons"] = comparisons
        statistical_edge_unresolved = adjusted_low <= 0.0
        if statistical_edge_unresolved:
            warnings_out.append(
                "Selected EV is not statistically distinguishable from zero after "
                f"adjusting for {comparisons} searched candidate(s)."
            )

    best_kelly = _safe_float(row.get("kelly"))
    if best_kelly is not None and best_kelly < 0.0:
        warnings_out.append(
            "Best candidate has negative Kelly fraction; sizing should be zero or very conservative."
        )

    best_edge = _safe_float(row.get("edge"))
    win_rate = _safe_float(row.get("prob_tp_first"))
    if win_rate is None:
        win_rate = _safe_float(row.get("prob_win"))
    low_practical_win_probability = bool(
        win_rate is not None and win_rate < LOW_PRACTICAL_WIN_PROB_THRESHOLD
    )
    if low_practical_win_probability:
        warnings_out.append(
            f"Best candidate win probability is {win_rate:.1%}, below the "
            f"{LOW_PRACTICAL_WIN_PROB_THRESHOLD:.0%} practical review threshold."
        )
    if best_edge is not None and best_edge < 0.0:
        if win_rate is not None:
            warnings_out.append(
                "Best candidate has negative raw win/loss edge "
                f"({best_edge:.3f}) with win rate {win_rate:.1%}; "
                "positive EV depends on reward/risk skew."
            )
        else:
            warnings_out.append(
                f"Best candidate has negative raw win/loss edge ({best_edge:.3f}); "
                "positive EV may depend on reward/risk skew."
            )

    breakeven_win_rate = _safe_float(row.get("breakeven_win_rate_net"))
    if breakeven_win_rate is None:
        breakeven_win_rate = _safe_float(row.get("breakeven_win_rate"))
    edge_vs_breakeven = _safe_float(row.get("edge_vs_breakeven"))
    if edge_vs_breakeven is not None and edge_vs_breakeven < 0.0 and breakeven_win_rate is not None:
        if win_rate is not None:
            warnings_out.append(
                f"Win rate {win_rate:.1%} is below the break-even threshold "
                f"{breakeven_win_rate:.1%}; unresolved paths or payoff skew are carrying EV."
            )
        else:
            warnings_out.append(
                f"Break-even win rate is {breakeven_win_rate:.1%}; "
                "selected candidate is below that threshold."
            )

    prob_no_hit = _safe_float(row.get("prob_no_hit"))
    if prob_no_hit is not None and prob_no_hit > PHANTOM_PROFIT_NO_HIT_THRESHOLD:
        warnings_out.append(
            f"More than {PHANTOM_PROFIT_NO_HIT_THRESHOLD:.0%} of simulations did not reach either barrier "
            f"before the horizon ({prob_no_hit:.1%} no-hit)."
        )
        out_metric_note = (
            "EV includes unresolved/no-hit paths at the horizon, while profit_factor "
            "only compares resolved TP-first reward against SL-first loss."
        )
    else:
        out_metric_note = (
            "EV includes all simulated outcomes; profit_factor compares resolved "
            "TP-first reward against SL-first loss."
        )

    # Compare resolved-trade metrics only. Full-path EV includes timeout MTM and
    # can legitimately have a different sign from the resolved-trade edge.
    resolved_ev = _safe_float(row.get("ev_cond"))
    if resolved_ev is not None and edge_vs_breakeven is not None:
        if (resolved_ev > 0.0 and edge_vs_breakeven < 0.0) or (
            resolved_ev < 0.0 and edge_vs_breakeven > 0.0
        ):
            ev_edge_conflict = True
            ev_edge_conflict_reason = "ev_cond and resolved edge_vs_breakeven have opposite signs"
            warnings_out.append(
                "Resolved-trade EV and break-even-adjusted resolved edge disagree; "
                "inspect win probability, reward/risk, and supplied costs."
            )

    if bool(row.get("phantom_profit_risk")):
        ev_edge_conflict = True
        ev_edge_conflict_reason = (
            "positive EV is dominated by unresolved paths with near-zero loss probability"
        )
        warnings_out.append(
            "SL barrier was not reached in most simulations while the observed loss rate "
            "was near zero. Positive EV may reflect horizon boundary effects, not trading edge."
        )
        caution = (
            "Selected candidate is dominated by unresolved paths; inspect `prob_no_hit`, "
            "`prob_resolve`, and `edge_vs_breakeven` before trading."
        )
    elif ev_edge_conflict:
        caution = (
            "EV and break-even-adjusted edge conflict for the selected candidate; inspect win "
            "probability and break-even threshold before trading."
        )

    deduped_warnings: List[str] = []
    seen: Set[str] = set()
    for message in warnings_out:
        msg = str(message).strip()
        if not msg or msg in seen:
            continue
        seen.add(msg)
        deduped_warnings.append(msg)

    out: Dict[str, Any] = {}
    if deduped_warnings:
        out["selection_warnings"] = deduped_warnings
    if ev_edge_conflict:
        out["ev_edge_conflict"] = True
        if ev_edge_conflict_reason:
            out["ev_edge_conflict_reason"] = ev_edge_conflict_reason
    if low_practical_win_probability:
        out["low_practical_win_probability"] = True
        out["low_practical_win_probability_threshold"] = LOW_PRACTICAL_WIN_PROB_THRESHOLD
    if best_ev is not None and row.get("profit_factor") is not None:
        out["metric_interpretation"] = f"{out_metric_note} {BARRIER_METRIC_BASIS_NOTE}"
    if caution:
        out["caution"] = caution
    if bool(row.get("low_confidence")):
        ci_w = _safe_float(row.get("prob_win_ci_width"))
        ci_msg = f" (CI width: {ci_w:.1%})" if ci_w is not None else ""
        out["confidence_warning"] = (
            f"Win probability estimate has wide confidence interval{ci_msg}. "
            "Increase --n-sims or use --search-profile long for tighter estimates."
        )
        out["low_confidence"] = True
        prob_win_val = _safe_float(row.get("prob_win"))
        if prob_win_val is not None:
            # n needed for CI width ≤ 0.05: Wilson CI ≈ 2*z*sqrt(p*(1-p)/n) ≤ 0.05
            target_width = 0.05
            z = 1.96
            pq = prob_win_val * (1.0 - prob_win_val)
            min_sims = int(math.ceil((2.0 * z) ** 2 * pq / (target_width ** 2)))
            out["min_sims_recommended"] = max(min_sims, 2000)
    if selection_adjusted_ci is not None:
        out["ev_selection_adjusted_ci95"] = selection_adjusted_ci
        out["ev_selection_comparisons"] = comparisons
        out["ev_selection_adjustment"] = "bonferroni_two_sided_95pct"
    if statistical_edge_unresolved:
        out["statistical_edge_unresolved"] = True
        out["statistical_edge_basis"] = "selection_adjusted_ev_lower_bound_le_zero"
    return out


def _build_actionability_payload(  # noqa: C901
    *,
    status: str,
    status_reason: Optional[str] = None,
    row: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    warning: Optional[str] = None,
    ensemble_degraded: bool = False,
) -> Dict[str, Any]:
    diag = diagnostics if isinstance(diagnostics, dict) else {}
    flags: List[str] = []

    if status == "no_candidates":
        flags.append("status_no_candidates")
    elif status == "non_viable":
        flags.append("status_non_viable")

    if isinstance(row, dict) and bool(row.get("phantom_profit_risk")):
        flags.append("phantom_profit_risk")
    if bool(diag.get("ev_edge_conflict")):
        flags.append("ev_edge_conflict")
    if bool(diag.get("low_practical_win_probability")):
        flags.append("low_practical_win_probability")
    if bool(diag.get("low_confidence")):
        flags.append("low_confidence")
    if bool(diag.get("statistical_edge_unresolved")):
        flags.append("statistical_edge_unresolved")
    if bool(diag.get("trading_costs_incomplete")):
        flags.append("trading_costs_incomplete")
    if diag.get("selection_warnings"):
        flags.append("selection_warnings")
    if warning:
        flags.append("warning")
    if ensemble_degraded:
        flags.append("ensemble_degraded")

    deduped_flags: List[str] = []
    seen: Set[str] = set()
    for flag in flags:
        key = str(flag).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped_flags.append(key)

    actionability = "actionable"
    trade_gate_passed = True
    actionability_reason = "No blocking diagnostics detected."

    if (
        status != "ok"
        or "phantom_profit_risk" in seen
        or "ev_edge_conflict" in seen
        or "low_practical_win_probability" in seen
    ):
        actionability = "blocked"
        trade_gate_passed = False
        if status != "ok":
            actionability_reason = (
                str(status_reason).strip()
                if status_reason else f"Optimizer status is {status!r}; do not trade this setup."
            )
        elif "phantom_profit_risk" in seen:
            actionability_reason = (
                "Positive EV is dominated by unresolved paths with near-zero observed loss; "
                "skip this setup."
            )
        elif "low_practical_win_probability" in seen:
            actionability_reason = (
                "Win probability is below the practical review threshold; skip or set "
                "min_prob_win explicitly."
            )
        else:
            actionability_reason = (
                "EV and break-even-adjusted edge conflict; manual review is required before trading."
            )
    elif {
        "selection_warnings",
        "low_confidence",
        "statistical_edge_unresolved",
        "trading_costs_incomplete",
        "warning",
        "ensemble_degraded",
    }.intersection(seen):
        actionability = "review"
        trade_gate_passed = False
        if "trading_costs_incomplete" in seen:
            missing_costs = diag.get("missing_cost_assumptions")
            missing_text = (
                ", ".join(str(item) for item in missing_costs)
                if isinstance(missing_costs, list) and missing_costs
                else "one or more cost components"
            )
            actionability_reason = (
                "Trading cost assumptions are incomplete "
                f"({missing_text}); provide explicit zero or positive assumptions."
            )
        elif "statistical_edge_unresolved" in seen:
            actionability_reason = (
                "Selected EV is not statistically distinguishable from zero after "
                "accounting for the searched candidate set."
            )
        elif warning:
            actionability_reason = str(warning).strip()
        elif diag.get("confidence_warning"):
            actionability_reason = str(diag["confidence_warning"]).strip()
        elif diag.get("selection_warnings"):
            warnings_list = diag.get("selection_warnings")
            if isinstance(warnings_list, list) and warnings_list:
                actionability_reason = str(warnings_list[0]).strip()
            else:
                actionability_reason = "Selection diagnostics require manual review before trading."
        else:
            actionability_reason = "Selection diagnostics require manual review before trading."

    recommendation = {
        "actionable": "trade",
        "review": "review",
        "blocked": "avoid",
    }.get(actionability, "review")

    out = {
        "actionability": actionability,
        "recommendation": recommendation,
        "recommendation_reason": actionability_reason,
        "trade_gate_passed": trade_gate_passed,
        "actionability_reason": actionability_reason,
        "actionability_flags": deduped_flags,
    }
    if status != "ok":
        out["remediation"] = {
            "next_steps": [
                "Retry with --search-profile long or a wider TP/SL grid.",
                "Run forecast_barrier_prob with explicit TP/SL levels to inspect hit probabilities.",
                "Use --candidate-filter all only for diagnostics; keep non-tradable setups out of execution.",
            ]
        }
    return out


def _auto_barrier_method(
    symbol: str,
    timeframe: str,
    prices: np.ndarray,
    horizon: Optional[int] = None,
) -> Tuple[str, str]:
    """Heuristically choose a barrier simulation method.

    The thresholds in this selector are pragmatic heuristics (history length,
    volatility clustering, tails/jumps) intended for robust defaults, not a
    universal optimum.
    """
    tf_secs = TIMEFRAME_SECONDS.get(timeframe, 0) or 0
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    horizon_val: Optional[int] = None
    try:
        horizon_val = int(horizon) if horizon is not None else None
    except Exception:
        horizon_val = None
    prefer_bridge = horizon_val is not None and horizon_val <= 12
    if prices.size < 10:
        if prefer_bridge:
            return "mc_gbm_bb", "auto: insufficient history; gbm_bb (short horizon)"
        return "mc_gbm", "auto: insufficient history; gbm baseline"

    rets = _log_returns_from_prices(prices)
    rets = rets[np.isfinite(rets)]
    if rets.size < 5:
        if prefer_bridge:
            return "mc_gbm_bb", "auto: insufficient returns; gbm_bb (short horizon)"
        return "mc_gbm", "auto: insufficient returns; gbm baseline"
    if rets.size < 30:
        if prefer_bridge:
            return "mc_gbm_bb", "auto: limited history; gbm_bb (short horizon)"
        return "mc_gbm", "auto: limited history; gbm baseline"

    mu = float(np.mean(rets))
    sigma = float(np.std(rets, ddof=1)) + 1e-12
    z = (rets - mu) / sigma
    kurt = float(np.mean(z ** 4) - 3.0) if z.size > 0 else 0.0
    jump_ratio = float(np.max(np.abs(rets - mu)) / sigma) if rets.size > 0 else 0.0
    skew = float(np.mean(z ** 3)) if z.size > 0 else 0.0

    r2 = (rets - mu) ** 2
    if r2.size >= 6:
        r2a = r2[1:]
        r2b = r2[:-1]
        denom = float(np.std(r2a) * np.std(r2b)) + 1e-12
        if denom > 0:
            try:
                vol_corr = float(np.corrcoef(r2a, r2b)[0, 1])
                if not np.isfinite(vol_corr):
                    vol_corr = 0.0
            except Exception:
                vol_corr = 0.0
        else:
            vol_corr = 0.0
    else:
        vol_corr = 0.0

    vol_ratio = 1.0
    if rets.size >= 60:
        short_n = min(50, rets.size)
        long_n = min(200, rets.size)
        short_std = float(np.std(rets[-short_n:], ddof=1)) + 1e-12
        long_std = float(np.std(rets[-long_n:], ddof=1)) + 1e-12
        vol_ratio = short_std / long_std

    is_crypto = is_probably_crypto_symbol(symbol)
    is_short_timeframe = bool(tf_secs and tf_secs <= 900)
    if is_crypto or is_short_timeframe:
        if kurt > 2.0 or jump_ratio > 4.0:
            if is_crypto and is_short_timeframe:
                reason = "auto: crypto short-timeframe with jumpy tails"
            elif is_crypto:
                reason = "auto: crypto with jumpy tails"
            else:
                reason = "auto: short-timeframe jumpy tails"
            return "jump_diffusion", reason

    if kurt > 3.5 or jump_ratio > 5.0:
        return "jump_diffusion", "auto: heavy tails/jump risk"

    if vol_ratio >= 1.6 or vol_ratio <= 0.65:
        if rets.size >= 220:
            return "hmm_mc", "auto: regime shift (volatility change)"

    if vol_corr > 0.3 and r2.size >= 150:
        try:
            import arch  # noqa: F401
            return "garch", "auto: strong volatility clustering (garch)"
        except Exception:
            return "heston", "auto: strong volatility clustering (heston fallback)"

    if vol_corr > 0.15:
        return "heston", "auto: volatility clustering"

    if rets.size >= 400 and abs(skew) >= 0.7 and jump_ratio < 4.5:
        return "bootstrap", "auto: non-normal returns; bootstrap"

    if prefer_bridge:
        return "mc_gbm_bb", "auto: mild tails; gbm with bridge correction"
    return "mc_gbm", "auto: mild tails; gbm baseline"


def _brownian_bridge_hits(
    log_paths: np.ndarray,
    barrier_log: float,
    sigma: float,
    *,
    direction: Literal["up", "down"],
    uniform: np.ndarray,
) -> np.ndarray:
    """Single-barrier Brownian bridge intra-step hit detection.

    When used for dual-barrier (TP+SL) scoring, callers use independent
    single-barrier samples. This does not model joint first-passage ordering.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        return np.zeros((log_paths.shape[0], log_paths.shape[1] - 1), dtype=bool)
    x0 = log_paths[:, :-1]
    x1 = log_paths[:, 1:]
    s2 = float(sigma * sigma)
    if direction == "up":
        valid = (x0 < barrier_log) & (x1 < barrier_log)
        dist0 = barrier_log - x0
        dist1 = barrier_log - x1
    else:
        valid = (x0 > barrier_log) & (x1 > barrier_log)
        dist0 = x0 - barrier_log
        dist1 = x1 - barrier_log
    expo = -2.0 * dist0 * dist1 / s2
    p = np.exp(np.clip(expo, -200.0, 50.0))
    hits = (uniform < p) & valid
    return hits


def _format_barrier_epoch(epoch: float) -> str:
    formatter = _format_time_minimal_local if _use_client_tz() else _format_time_minimal
    return formatter(float(epoch))


def _coerce_epoch(value: Any) -> Optional[float]:
    epoch = coerce_finite_float(value)
    if epoch is None or epoch <= 0.0:
        return None
    if epoch > 10_000_000_000:
        epoch /= 1000.0
    return epoch


def _history_freshness_context(
    df: Any,
    timeframe: str,
    *,
    symbol: Optional[str] = None,
    now_epoch: Optional[float] = None,
) -> Dict[str, Any]:
    """Describe whether the model's latest completed bar is execution-current."""
    out: Dict[str, Any] = {"history_bars_used": int(len(df))}
    try:
        first_epoch = _coerce_epoch(df["time"].iloc[0])
        last_epoch = _coerce_epoch(df["time"].iloc[-1])
    except Exception:
        first_epoch = None
        last_epoch = None
    if last_epoch is None:
        out["history_policy_ok"] = False
        return out

    observed_epoch = (
        datetime.now(timezone.utc).timestamp()
        if now_epoch is None
        else float(now_epoch)
    )
    freshness = completed_bar_freshness_fields(
        symbol,
        timeframe,
        last_epoch,
        now_epoch=observed_epoch,
        item="data",
        tolerance_bars=1,
    )
    if not freshness:
        out["history_policy_ok"] = False
        return out
    completed_bar_end = float(freshness["data_as_of_epoch"])
    last_bar_open = _format_barrier_epoch(last_epoch)
    out.update(
        {
            "history_last_bar_open": last_bar_open,
            "history_last_bar_open_epoch": float(last_epoch),
            "last_bar_open": last_bar_open,
            "data_as_of": _format_barrier_epoch(completed_bar_end),
            "data_as_of_epoch": float(completed_bar_end),
            "last_observation_close_time": _format_barrier_epoch(completed_bar_end),
            "last_observation_close_epoch": float(completed_bar_end),
            "data_freshness_seconds": freshness["data_age_seconds"],
            "data_stale": freshness["data_stale"],
            "stale_after_seconds": freshness["stale_after_seconds"],
            "freshness_basis": freshness["freshness_basis"],
            "freshness_age_metric": freshness["freshness_age_metric"],
            "bar_timestamp_basis": "mt5_bar_open",
            "input_bar_policy": "closed_bars_only",
            "timezone": "UTC",
        }
    )
    if first_epoch is not None:
        out["history_window"] = {
            "start": _format_barrier_epoch(first_epoch),
            "end": _format_barrier_epoch(completed_bar_end),
            "bars_used": int(len(df)),
            "timezone": "UTC",
            "input_bar_policy": "closed_bars_only",
        }
    for key, value in freshness.items():
        if key not in {
            "data_as_of",
            "data_as_of_epoch",
            "data_age_seconds",
            "data_stale",
            "stale_after_seconds",
            "freshness_basis",
            "freshness_age_metric",
        }:
            out[key] = value
    return out


def _live_reference_time_context(
    symbol: str,
    timeframe: str,
    *,
    now_epoch: Optional[float] = None,
) -> Dict[str, Any]:
    """Describe freshness of the MT5 tick used as a barrier reference."""
    del timeframe  # Reserved for future per-timeframe quote policy.
    try:
        from ..utils.mt5 import mt5 as _mt5
    except Exception:
        return {"reference_usable_for_live": False}
    try:
        raw_tick = _mt5.symbol_info_tick(symbol)
    except Exception:
        raw_tick = None
    if now_epoch is None:
        now_epoch = datetime.now(timezone.utc).timestamp()
    tick, quote_source = resolve_quote_tick(
        _mt5,
        symbol,
        raw_tick,
        now_epoch=now_epoch,
    )
    if tick is None:
        return {"reference_usable_for_live": False}

    epoch = tick_epoch(tick)
    if epoch is None:
        return {"reference_usable_for_live": False}

    freshness = build_tick_freshness_context(
        symbol,
        tick_epoch=epoch,
        now_epoch=now_epoch,
        item="reference price",
        age_rounder=lambda value: max(0, int(round(value))),
    )
    freshness.update(quote_source)
    bid = tick_value(tick, "bid")
    ask = tick_value(tick, "ask")
    spread = compute_spread_metrics(bid, ask)
    enforce_quote_execution_readiness(
        freshness,
        bid=bid,
        ask=ask,
        quote_source_conflict=freshness.get("quote_source_conflict"),
    )
    age_seconds = freshness.get("data_age_seconds")
    out = {
        "reference_price_time": _format_barrier_epoch(epoch),
        "reference_price_time_epoch": float(epoch),
        "reference_price_age_seconds": age_seconds,
        "reference_price_age": format_age_seconds(age_seconds),
        "reference_price_stale": freshness.get("data_stale"),
        "reference_freshness_state": freshness.get("freshness_state"),
        "reference_live_max_age_seconds": freshness.get("live_max_age_seconds"),
        "reference_usable_for_live": freshness.get("usable_for_live_trading"),
        "reference_spread_quality": freshness.get("spread_quality"),
        "reference_quote_source": freshness.get("quote_source"),
        "reference_quote_source_state": freshness.get("quote_source_state"),
        "reference_bid": _safe_float(bid),
        "reference_ask": _safe_float(ask),
        "reference_spread": _safe_float(spread.get("spread")),
        "reference_spread_pct": _safe_float(spread.get("spread_pct")),
    }
    for source_key, output_key in (
        ("symbol_info_tick_time_epoch", "reference_symbol_info_tick_time_epoch"),
        ("stream_tick_time_epoch", "reference_stream_tick_time_epoch"),
        ("quote_source_conflict", "reference_quote_source_conflict"),
        ("warning", "reference_quote_warning"),
    ):
        if freshness.get(source_key) is not None:
            out[output_key] = freshness[source_key]
    for key in (
        "market_status",
        "market_status_reason",
        "market_status_source",
        "freshness_policy_relaxed",
    ):
        if freshness.get(key) is not None:
            out[key] = freshness[key]
    return out


def _apply_barrier_freshness_contract(
    out: Dict[str, Any],
    *,
    history_context: Dict[str, Any],
    reference_context: Dict[str, Any],
    last_price_source: Any,
) -> None:
    """Attach freshness evidence and fail the execution gate closed."""
    out.update(history_context)
    model_ready = bool(history_context.get("history_policy_ok"))
    out["model_data_usable_for_live"] = model_ready
    blockers: List[str] = []
    if not model_ready:
        blockers.append("model_history_outside_policy")

    live_reference = str(last_price_source or "").startswith("live_tick")
    if live_reference:
        out.update(reference_context)
        reference_ready = bool(reference_context.get("reference_usable_for_live"))
        if not reference_ready:
            blockers.append("reference_quote_not_live")
        if (
            reference_context.get("reference_price_stale") is True
            or reference_context.get("market_status") == "closed"
        ):
            out["last_price_source"] = str(last_price_source).replace(
                "live_tick", "last_tick", 1
            )
    else:
        reference_ready = False
        blockers.append("live_reference_quote_not_used")
        if history_context.get("last_observation_close_time"):
            out["reference_price_time"] = history_context.get(
                "last_observation_close_time"
            )
            out["reference_price_time_epoch"] = history_context.get(
                "last_observation_close_epoch"
            )

    execution_ready = bool(model_ready and reference_ready)
    out["usable_for_live_trading"] = execution_ready
    out["usable_for_live_trading_basis"] = "model_history_and_reference_quote"
    out["execution_blockers"] = blockers
    if execution_ready:
        return

    out["trade_gate_passed"] = False
    out["tradable"] = False
    out["no_action"] = True
    reasons = list(out.get("trade_gate_reasons") or [])
    for blocker in blockers:
        if blocker not in reasons:
            reasons.append(blocker)
    out["trade_gate_reasons"] = reasons
    warnings_out = list(out.get("warnings") or [])
    reference_only_blocker = bool(
        model_ready
        and len(blockers) == 1
        and blockers[0] in {
            "live_reference_quote_not_used",
            "reference_quote_not_live",
        }
        and out.get("status") in (None, "ok")
    )
    if reference_only_blocker:
        warning = (
            "Barrier output is not execution-ready because an executable live "
            "reference quote was unavailable; results remain anchored to research data."
        )
    else:
        warning = (
            "Barrier output is not execution-ready because the model history or "
            "reference quote is outside its live freshness policy."
        )
    if warning not in warnings_out:
        warnings_out.append(warning)
    out["warnings"] = warnings_out
    blocker_reason = f"{warning} Blockers: {', '.join(blockers)}."
    if out.get("actionability") == "actionable":
        out["actionability"] = "blocked"
    out["recommendation"] = "avoid"
    out["recommendation_reason"] = blocker_reason
    out["actionability_reason"] = blocker_reason
    if reference_only_blocker:
        out["remediation"] = {
            "next_steps": [
                "Call market_ticker to confirm a fresh executable quote.",
                "Retry the barrier request after the live quote is available.",
                "Use the close-anchored output only for historical research.",
            ]
        }
    if out.get("status") == "ok":
        out["status_reason"] = blocker_reason
    actionability_flags = list(out.get("actionability_flags") or [])
    if "freshness_gate_failed" not in actionability_flags:
        actionability_flags.append("freshness_gate_failed")
    out["actionability_flags"] = actionability_flags


def _get_live_reference_price(symbol: str, direction: str) -> Tuple[Optional[float], Optional[str]]:
    """Best-effort live tick reference price; returns (price, source) or (None, None)."""
    try:
        from ..utils.mt5 import mt5 as _mt5
    except Exception:
        return None, None

    now_epoch = datetime.now(timezone.utc).timestamp()
    try:
        raw_tick = _mt5.symbol_info_tick(symbol)
    except Exception:
        raw_tick = None
    tick, _quote_source = resolve_quote_tick(
        _mt5,
        symbol,
        raw_tick,
        now_epoch=now_epoch,
    )
    if tick is None:
        return None, None

    def _valid_price(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except Exception:
            return None
        if not np.isfinite(out) or out <= 0.0:
            return None
        return out

    # Reconciled stream ticks are mapping/record-like while symbol_info_tick
    # returns an attribute object. The canonical accessor handles both.
    bid = _valid_price(tick_value(tick, "bid"))
    ask = _valid_price(tick_value(tick, "ask"))
    last = _valid_price(tick_value(tick, "last"))

    direction_norm, _ = normalize_trade_direction(direction)
    if direction_norm == "long":
        if ask is not None:
            return ask, "live_tick_ask"
        return None, None
    if direction_norm == "short":
        if bid is not None:
            return bid, "live_tick_bid"
        return None, None

    if bid is not None and ask is not None:
        return 0.5 * (bid + ask), "live_tick_mid"
    if last is not None:
        return last, "live_tick_last"
    if bid is not None:
        return bid, "live_tick_bid_only"
    if ask is not None:
        return ask, "live_tick_ask_only"
    return None, None


def _symbol_price_precision(symbol: str) -> Optional[int]:
    try:
        from ..utils.mt5 import get_symbol_info_cached

        info = get_symbol_info_cached(symbol)
    except Exception:
        return None
    if info is None:
        return None
    try:
        digits = int(info.digits)
    except Exception:
        return None
    return max(0, digits)


def _resolve_reference_prices(
    close_values: Any,
    *,
    symbol: str,
    direction: str,
    use_live_price: bool = True,
    live_price_getter: Any = None,
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str], Optional[str]]:
    closes = np.asarray(close_values, dtype=float)
    if closes.size == 0:
        return None, None, None, None, "Latest close is non-finite; refresh history or enable a live reference price."

    trailing_close = _safe_float(closes[-1])
    finite_closes = closes[np.isfinite(closes)]
    history_anchor = _safe_float(finite_closes[-1]) if finite_closes.size else None
    if history_anchor is None or history_anchor <= 0.0:
        return None, None, None, None, "Latest close is non-finite; refresh history or enable a live reference price."

    if use_live_price:
        getter = live_price_getter if callable(live_price_getter) else _get_live_reference_price
        live_price, live_source = getter(symbol, direction)
        if live_price is not None and np.isfinite(live_price) and float(live_price) > 0.0:
            warning = None
            if trailing_close is None or trailing_close <= 0.0:
                warning = "Latest close is non-finite; using the last finite historical close as the simulation anchor."
            return float(history_anchor), float(live_price), str(live_source or "live_tick"), warning, None

    if trailing_close is None or trailing_close <= 0.0:
        return None, None, None, None, "Latest close is non-finite; refresh history or enable a live reference price."
    return float(trailing_close), float(trailing_close), "close", None, None


