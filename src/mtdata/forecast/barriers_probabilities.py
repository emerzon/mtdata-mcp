from typing import Any, Dict, List, Literal, Optional

import numpy as np

from ..shared.constants import TIMEFRAME_SECONDS
from ..shared.schema import DenoiseSpec, TimeframeLiteral
from ..shared.validators import (
    unknown_mapping_keys_error,
    unsupported_timeframe_seconds_error,
)
from ..utils import denoise as _denoise_api
from ..utils.barriers import (
    barrier_prices_are_valid as _barrier_prices_are_valid,
)
from ..utils.barriers import get_tick_size as _get_tick_size
from ..utils.barriers import (
    normalize_same_bar_policy,
    normalize_trade_direction,
    resolve_same_bar_probabilities,
    unresolved_barrier_price_error,
    validate_barrier_unit_family_exclusivity,
)
from ..utils.barriers import (
    resolve_barrier_prices as _resolve_barrier_prices,
)
from ..utils.utils import parse_kv_or_json as _parse_kv_or_json
from .barrier_outcomes import evaluate_barrier_path_outcomes
from .barriers_shared import (
    BROWNIAN_BRIDGE_DUAL_BARRIER_MODEL,
    BROWNIAN_BRIDGE_DUAL_BARRIER_WARNING,
    _apply_barrier_freshness_contract,
    _auto_barrier_method,
    _binomial_se,
    _binomial_wilson_95,
    _brownian_bridge_hits,
    _get_live_reference_price,
    _history_freshness_context,
    _live_reference_time_context,
    _prepare_brownian_bridge_draws,
    _resolve_reference_prices,
    _stable_barrier_seed,
    _symbol_price_precision,
    barrier_method_error,
    normalize_barrier_method,
    normalize_barrier_seed,
)
from .common import annualization_context as _annualization_context
from .common import fetch_history as _fetch_history
from .common import log_returns_from_prices as _log_returns_from_prices
from .monte_carlo import gbm_single_barrier_upcross_prob as _gbm_upcross_prob
from .monte_carlo import simulate_bootstrap_mc as _simulate_bootstrap_mc
from .monte_carlo import simulate_garch_mc as _simulate_garch_mc
from .monte_carlo import simulate_gbm_mc as _simulate_gbm_mc
from .monte_carlo import simulate_heston_mc as _simulate_heston_mc
from .monte_carlo import simulate_hmm_mc as _simulate_hmm_mc
from .monte_carlo import simulate_jump_diffusion_mc as _simulate_jump_diffusion_mc

_BARRIER_COMMON_PARAM_KEYS = {"n_sims", "seed", "sims"}
_BARRIER_METHOD_PARAM_KEYS = {
    "mc_gbm": set(),
    "mc_gbm_bb": set(),
    "hmm_mc": {"n_states"},
    "garch": {"p", "q"},
    "bootstrap": {"block_size"},
    "heston": {"kappa", "rho", "theta", "v0", "xi"},
    "jump_diffusion": {
        "jump_lambda",
        "jump_mu",
        "jump_sigma",
        "jump_threshold",
        "lambda",
    },
}


def _barrier_param_keys(method: str) -> set[str]:
    if method == "auto":
        method_keys = set().union(*_BARRIER_METHOD_PARAM_KEYS.values())
    else:
        method_keys = set(_BARRIER_METHOD_PARAM_KEYS.get(method, set()))
    return _BARRIER_COMMON_PARAM_KEYS | method_keys


def _abs_barrier_side_error(
    *,
    price: float,
    direction: str,
    tp_abs: Optional[float],
    sl_abs: Optional[float],
) -> Optional[str]:
    """Reject user-supplied absolute TP/SL levels on the wrong side of price.

    ``tp_abs``/``sl_abs`` are absolute price levels (not offsets). A mis-sided
    level is almost always a unit mistake (e.g. passing an intended offset like
    ``sl_abs=500``), which the silent inversion nudge would otherwise mask with
    meaningless probabilities. Return an actionable error instead.
    """
    try:
        ref = float(price)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(ref):
        return None
    is_long = direction == "long"
    problems: List[str] = []
    for name, value, must_be_above in (
        ("tp_abs", tp_abs, is_long),
        ("sl_abs", sl_abs, not is_long),
    ):
        if value is None:
            continue
        try:
            level = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(level):
            continue
        if must_be_above and level <= ref:
            problems.append(f"{name} ({level:g}) must be above the reference price ({ref:g})")
        elif not must_be_above and level >= ref:
            problems.append(f"{name} ({level:g}) must be below the reference price ({ref:g})")
    if not problems:
        return None
    side = "long" if is_long else "short"
    return (
        f"For a {side} position, " + " and ".join(problems)
        + ". tp_abs/sl_abs are absolute price levels, not offsets; use tp_pct/sl_pct "
        "or tp_ticks/sl_ticks to specify distances from the reference price."
    )


def forecast_barrier_hit_probabilities(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    method: Literal['mc_gbm','mc_gbm_bb','hmm_mc','garch','bootstrap','heston','jump_diffusion','auto'] = 'mc_gbm_bb',
    direction: Literal['long','short'] = 'long',
    same_bar_policy: Literal['sl_first','tp_first','neutral'] = 'sl_first',
    tp_abs: Optional[float] = None,
    sl_abs: Optional[float] = None,
    tp_pct: Optional[float] = None,
    sl_pct: Optional[float] = None,
    tp_ticks: Optional[float] = None,
    sl_ticks: Optional[float] = None,
    params: Optional[Dict[str, Any]] = None,
    denoise: Optional[DenoiseSpec] = None,
    as_of: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    """Monte Carlo barrier analysis: TP/SL hit probabilities within `horizon` bars.

    Notes:
    - Barriers are provided via absolute prices (tp_abs/sl_abs), percentages
      (tp_pct/sl_pct), or ticks (tp_ticks/sl_ticks; uses `trade_tick_size`).
      Use exactly one unit family per request; mixed units are rejected.
    - In discrete time, TP and SL can be hit in the same bar. Resolution is
      controlled explicitly by `same_bar_policy`.
    - The default GBM Brownian-bridge method accounts for barrier touches
      between simulated bar closes. Other path methods disclose close-only
      hit detection in their output.
    """
    try:
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
        try:
            same_bar_policy_value = normalize_same_bar_policy(same_bar_policy)
        except ValueError as exc:
            return {"error": str(exc)}
        p = _parse_kv_or_json(params)
        method_key = normalize_barrier_method(method)
        if method_key is None:
            return {"error": barrier_method_error(method)}
        parameter_error = unknown_mapping_keys_error(
            p,
            _barrier_param_keys(method_key),
            subject=f"barrier params for method '{method_key}'",
        )
        if parameter_error is not None:
            return parameter_error
        warnings_out: List[str] = []
        try:
            barrier_values = validate_barrier_unit_family_exclusivity(
                {
                    "tp_abs": tp_abs,
                    "sl_abs": sl_abs,
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "tp_ticks": tp_ticks,
                    "sl_ticks": sl_ticks,
                }
            )
        except ValueError as exc:
            return {"error": str(exc)}
        # Fetch enough history for calibration
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
        # Current price baseline
        last_price_close, last_price, last_price_source, price_warning, price_error = _resolve_reference_prices(
            df['close'].astype(float).to_numpy(),
            symbol=symbol,
            direction=direction_norm,
            use_live_price=not bool(as_of or start or end),
            live_price_getter=_get_live_reference_price,
        )
        if as_of or start or end:
            last_price_source = "candle_close"
        if price_error:
            return {"error": price_error}
        if price_warning:
            warnings_out.append(price_warning)
        tick_size = _get_tick_size(symbol)

        abs_side_error = _abs_barrier_side_error(
            price=last_price, direction=direction_norm, tp_abs=tp_abs, sl_abs=sl_abs
        )
        if abs_side_error:
            return {"error": abs_side_error}

        # Compute absolute TP/SL prices with explicit trade direction
        dir_long = direction_norm == 'long'
        tp_price, sl_price = _resolve_barrier_prices(
            price=last_price,
            direction=direction_norm,
            tp_abs=tp_abs,
            sl_abs=sl_abs,
            tp_pct=barrier_values.get("tp_pct"),
            sl_pct=barrier_values.get("sl_pct"),
            tp_ticks=barrier_values.get("tp_ticks"),
            sl_ticks=barrier_values.get("sl_ticks"),
            tick_size=tick_size,
        )

        if tp_price is None or sl_price is None:
            return {
                "error": unresolved_barrier_price_error(
                    tp_abs=tp_abs,
                    sl_abs=sl_abs,
                    tp_pct=barrier_values.get("tp_pct"),
                    sl_pct=barrier_values.get("sl_pct"),
                    tp_ticks=barrier_values.get("tp_ticks"),
                    sl_ticks=barrier_values.get("sl_ticks"),
                    tick_size=tick_size,
                )
            }
        if not _barrier_prices_are_valid(
            price=last_price,
            direction=direction_norm,
            tp_price=tp_price,
            sl_price=sl_price,
        ):
            return {"error": "Resolved TP/SL barriers are invalid for the current price."}

        # Build input series (denoise optional)
        base_col = 'close'
        if denoise:
            try:
                normalized = _denoise_api.normalize_denoise_spec(denoise, default_when='pre_ti') or denoise
                added = _denoise_api.apply_denoise(df, normalized, default_when='pre_ti')
                base_col = _denoise_api.effective_denoise_base_col(
                    df,
                    normalized if isinstance(normalized, dict) else denoise,
                    base_col='close',
                    added_columns=added,
                )
            except Exception as ex:
                warnings_out.append(f"Denoise request failed; using raw close prices instead: {ex}")
        prices = df[base_col].astype(float).to_numpy()

        # Simulate paths
        raw_sims = p.get('n_sims', p.get('sims', None))
        if raw_sims is None:
            sims = 2000
        else:
            try:
                sims = int(raw_sims)
            except Exception:
                return {"error": f"Invalid n_sims: {raw_sims}. Must be >= 1."}
        if sims <= 0:
            return {"error": f"Invalid n_sims: {sims}. Must be >= 1."}
        method_requested = method_key
        auto_reason = None
        if method_key == 'auto':
            method_key, auto_reason = _auto_barrier_method(
                symbol, timeframe, prices, horizon=horizon_val
            )
        bb_enabled = method_key == 'mc_gbm_bb'
        seed_raw = p.get('seed')
        seed_provided = seed_raw is not None
        # Live reference prices change the barrier geometry, not the stochastic
        # path generator. Keep common random draws across tick-only changes so
        # nearby barrier results are directly comparable.
        request_seed_base = (
            normalize_barrier_seed(seed_raw)
            if seed_provided
            else _stable_barrier_seed(
                "forecast_barrier_prob",
                symbol,
                timeframe,
                horizon_val,
                method_key,
                direction_norm,
                int(sims),
                int(len(prices)),
                float(prices[-1]),
                {k: v for k, v in p.items() if k != "seed"},
            )
        )
        
        try:
            if method_key in ('mc_gbm', 'mc_gbm_bb'):
                sim = _simulate_gbm_mc(
                    prices,
                    horizon=horizon_val,
                    n_sims=int(sims),
                    seed=normalize_barrier_seed(request_seed_base),
                )
            elif method_key == 'hmm_mc':
                n_states = int(p.get('n_states', 2) or 2)
                sim = _simulate_hmm_mc(
                    prices,
                    horizon=horizon_val,
                    n_states=int(n_states),
                    n_sims=int(sims),
                    seed=normalize_barrier_seed(request_seed_base),
                )
            elif method_key == 'garch':
                p_order = int(p.get('p', 1))
                q_order = int(p.get('q', 1))
                sim = _simulate_garch_mc(
                    prices,
                    horizon=horizon_val,
                    n_sims=int(sims),
                    seed=normalize_barrier_seed(request_seed_base),
                    p_order=p_order,
                    q_order=q_order,
                )
            elif method_key == 'bootstrap':
                bs = p.get('block_size')
                if bs: bs = int(bs)
                sim = _simulate_bootstrap_mc(
                    prices,
                    horizon=horizon_val,
                    n_sims=int(sims),
                    seed=normalize_barrier_seed(request_seed_base),
                    block_size=bs,
                )
            elif method_key == 'heston':
                heston_bars_per_year, _ = _annualization_context(
                    timeframe,
                    symbol,
                    observed_times=df.get("time"),
                    observed_timeframe=timeframe,
                )
                sim = _simulate_heston_mc(
                    prices,
                    horizon=horizon_val,
                    n_sims=int(sims),
                    seed=normalize_barrier_seed(request_seed_base),
                    kappa=p.get('kappa'),
                    theta=p.get('theta'),
                    xi=p.get('xi'),
                    rho=p.get('rho'),
                    v0=p.get('v0'),
                    bars_per_year=heston_bars_per_year,
                )
            elif method_key == 'jump_diffusion':
                sim = _simulate_jump_diffusion_mc(
                    prices,
                    horizon=horizon_val,
                    n_sims=int(sims),
                    seed=normalize_barrier_seed(request_seed_base),
                    jump_lambda=p.get('jump_lambda', p.get('lambda')),
                    jump_mu=p.get('jump_mu'),
                    jump_sigma=p.get('jump_sigma'),
                    jump_threshold=float(p.get('jump_threshold', 3.0)),
                )
            else:
                return {"error": f"Unsupported method: {method}. Use 'mc_gbm', 'mc_gbm_bb', 'hmm_mc', 'garch', 'bootstrap', 'heston', 'jump_diffusion', or 'auto'"}
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            return {
                "error": f"Simulation failed ({method_key}): {e}",
                "error_type": "simulation_failure",
            }

        price_paths = np.asarray(sim['price_paths'], dtype=float)
        (
            price_paths,
            bb_enabled,
            bb_sigma,
            bb_log_paths,
            bb_uniform_tp,
            bb_uniform_sl,
        ) = _prepare_brownian_bridge_draws(
            price_paths,
            calibration_prices=prices,
            last_price_close=last_price_close,
            reference_price=last_price,
            bb_enabled=bb_enabled,
            seed_base=request_seed_base,
        )
        S, H = price_paths.shape
        
        tp_bridge = None
        sl_bridge = None
        if bb_enabled and bb_log_paths is not None and bb_uniform_tp is not None and bb_uniform_sl is not None:
            tp_dir = "up" if dir_long else "down"
            sl_dir = "down" if dir_long else "up"
            tp_bridge = _brownian_bridge_hits(bb_log_paths, float(np.log(tp_price)), bb_sigma, direction=tp_dir, uniform=bb_uniform_tp)
            sl_bridge = _brownian_bridge_hits(bb_log_paths, float(np.log(sl_price)), bb_sigma, direction=sl_dir, uniform=bb_uniform_sl)

        outcomes = evaluate_barrier_path_outcomes(
            price_paths,
            tp_trigger=tp_price,
            sl_trigger=sl_price,
            direction="long" if dir_long else "short",
            extra_tp_hits=tp_bridge,
            extra_sl_hits=sl_bridge,
        )
        idx_tp_val = outcomes.first_tp
        idx_sl_val = outcomes.first_sl
        any_tp = idx_tp_val < outcomes.horizon
        any_sl = idx_sl_val < outcomes.horizon

        tp_first = np.sum(outcomes.wins)
        sl_first = np.sum(outcomes.losses)
        both_tie = np.sum(outcomes.ties)
        no_hit = np.sum(outcomes.unresolved)

        # Collect hit times (1-based) for stats
        # TP stats include strict wins and ties
        t_hit_tp = (idx_tp_val[outcomes.wins | outcomes.ties] + 1).tolist()
        t_hit_sl = (idx_sl_val[outcomes.losses | outcomes.ties] + 1).tolist()

        # Cumulative hit curves (hit at or before t)
        def _compute_cum_curve(indices, valid_mask, length):
            valid_indices = indices[valid_mask]
            if valid_indices.size == 0:
                return np.zeros(length, dtype=float)
            # bincount counts occurrences of each index
            counts = np.bincount(valid_indices, minlength=length)
            if counts.size > length:
                counts = counts[:length]
            return np.cumsum(counts).astype(float)

        tp_any_by_t = _compute_cum_curve(idx_tp_val, any_tp, H)
        sl_any_by_t = _compute_cum_curve(idx_sl_val, any_sl, H)

        S_f = float(S)
        resolved_probabilities = resolve_same_bar_probabilities(
            tp_strict=tp_first / S_f,
            sl_strict=sl_first / S_f,
            same_bar=both_tie / S_f,
            no_hit=no_hit / S_f,
            policy=same_bar_policy_value,
        )
        prob_tp_first = resolved_probabilities["prob_tp_first"]
        prob_sl_first = resolved_probabilities["prob_sl_first"]
        prob_same_bar = resolved_probabilities["prob_same_bar"]
        prob_no_hit = resolved_probabilities["prob_no_hit"]
        tp_any_curve = (tp_any_by_t / S_f).tolist()
        sl_any_curve = (sl_any_by_t / S_f).tolist()
        tp_lo, tp_hi = _binomial_wilson_95(prob_tp_first, int(S))
        sl_lo, sl_hi = _binomial_wilson_95(prob_sl_first, int(S))
        tie_lo, tie_hi = _binomial_wilson_95(prob_same_bar, int(S))
        no_hit_lo, no_hit_hi = _binomial_wilson_95(prob_no_hit, int(S))

        def _stats(arr: list[int]) -> Dict[str, float]:
            if not arr:
                return {"mean": float('nan'), "median": float('nan')}
            a = np.asarray(arr, dtype=float)
            return {"mean": float(a.mean()), "median": float(np.median(a))}

        tp_stats = _stats(t_hit_tp)
        sl_stats = _stats(t_hit_sl)
        def _finite_or_none(x: float) -> Optional[float]:
            try:
                return float(x) if np.isfinite(x) else None
            except Exception:
                return None
        # Directional interpretation:
        # - For long: TP is above last_price, SL is below; prob_tp_first is long win probability.
        # - For short: TP is below last_price, SL is above; prob_tp_first is short win probability.
        probability_edge = float(prob_tp_first - prob_sl_first)
        price_precision = _symbol_price_precision(symbol)
        out = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "method": method_key,
            "intra_bar_hit_detection": (
                "brownian_bridge" if bb_enabled else "simulated_bar_close"
            ),
            "horizon": horizon_val,
            "direction": direction_norm,
            "same_bar_policy": same_bar_policy_value,
            "last_price": last_price,
            "last_price_close": float(last_price_close),
            "last_price_source": last_price_source,
            "tp_price": float(tp_price),
            "sl_price": float(sl_price),
            "n_sims": int(S),
            "seed": int(request_seed_base),
            "seed_source": "params" if seed_provided else "derived_from_request",
            **resolved_probabilities,
            "prob_tp_first_se": _binomial_se(prob_tp_first, int(S)),
            "prob_sl_first_se": _binomial_se(prob_sl_first, int(S)),
            "prob_same_bar_se": _binomial_se(prob_same_bar, int(S)),
            "prob_no_hit_se": _binomial_se(prob_no_hit, int(S)),
            "prob_tp_first_ci95": {"low": float(tp_lo), "high": float(tp_hi)},
            "prob_sl_first_ci95": {"low": float(sl_lo), "high": float(sl_hi)},
            "prob_same_bar_ci95": {"low": float(tie_lo), "high": float(tie_hi)},
            "prob_no_hit_ci95": {"low": float(no_hit_lo), "high": float(no_hit_hi)},
            "probability_edge": probability_edge,
            "tp_hit_prob_by_t": [float(v) for v in tp_any_curve],
            "sl_hit_prob_by_t": [float(v) for v in sl_any_curve],
            "time_to_tp_bars": tp_stats,
            "time_to_sl_bars": sl_stats,
        }
        reference_context: Dict[str, Any] = {}
        if str(last_price_source or "").startswith("live_tick"):
            reference_context = _live_reference_time_context(symbol, timeframe)
        _apply_barrier_freshness_contract(
            out,
            history_context=freshness_context,
            reference_context=reference_context,
            last_price_source=last_price_source,
        )
        out["conditioning_note"] = (
            "Probabilities use closed bars through "
            f"{out.get('data_as_of')}; barriers are measured from "
            f"{out.get('last_price_source')}."
        )
        for warning_text in out.get("warnings") or []:
            if warning_text not in warnings_out:
                warnings_out.append(str(warning_text))
        if price_precision is not None:
            out["price_precision"] = int(price_precision)
        if method_requested != method_key:
            out["method_requested"] = method_requested
            out["method_used"] = method_key
            if auto_reason:
                out["auto_reason"] = auto_reason
        if bb_enabled:
            out["bridge_correction"] = True
            out["bridge_dual_barrier_model"] = BROWNIAN_BRIDGE_DUAL_BARRIER_MODEL
            out["bridge_joint_first_passage"] = False
            warnings_out.append(BROWNIAN_BRIDGE_DUAL_BARRIER_WARNING)
        else:
            warnings_out.append(
                "Barrier hits are evaluated at simulated bar closes; transient "
                "intra-bar touches may be undercounted."
            )
            if method_key in {"jump_diffusion", "bootstrap"}:
                warnings_out.append(
                    "Close-only jump or bootstrap paths can gap through the nearer "
                    "barrier and score the far barrier as first; treat first-hit "
                    "order as discrete-close, not continuous first-passage."
                )
        if 'model_summary' in sim:
            out['model_summary'] = str(sim['model_summary'])
        # Expose simulation model metadata (e.g. HMM fitted vs requested states)
        _meta_keys = ('fitted_n_states', 'requested_n_states', 'model_type')
        sim_meta = {k: sim[k] for k in _meta_keys if k in sim}
        if 'mu' in sim:
            import numpy as _np
            sim_meta['mu'] = [float(v) for v in _np.asarray(sim['mu']).ravel()]
        if 'sigma' in sim:
            import numpy as _np
            sim_meta['sigma'] = [float(v) for v in _np.asarray(sim['sigma']).ravel()]
        sim_params = sim.get("params") if isinstance(sim, dict) else None
        if isinstance(sim_params, dict):
            for key in ("param_time_unit", "bars_per_year", "sigma_total", "diffusion_sigma"):
                if key in sim_params:
                    sim_meta[key] = sim_params[key]
        if sim_meta:
            out['sim_meta'] = sim_meta
            requested_states = sim_meta.get("requested_n_states")
            fitted_states = sim_meta.get("fitted_n_states")
            if (
                isinstance(requested_states, (int, float))
                and isinstance(fitted_states, (int, float))
                and int(fitted_states) < int(requested_states)
            ):
                warnings_out.append(
                    "HMM state collapse: requested "
                    f"{int(requested_states)} states but fitted {int(fitted_states)}; "
                    "probabilities use the reduced-state model."
                )
        if warnings_out:
            out["warnings"] = warnings_out
             
        return out
    except (KeyError, AttributeError, IndexError):
        raise
    except Exception as e:
        return {
            "error": f"Error computing barrier probabilities: {str(e)}",
        }

def forecast_barrier_closed_form(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    direction: Literal['long','short'] = 'long',
    barrier: float = 0.0,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
    denoise: Optional[DenoiseSpec] = None,
    as_of: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    """Closed-form single-barrier hit probability for GBM within horizon.

    Direction semantics:
    - "long": probability of reaching an upper barrier (price >= barrier).
    - "short": probability of reaching a lower barrier (price <= barrier).
    """
    try:
        direction_norm, direction_error = normalize_trade_direction(direction)
        if direction_error:
            return {"error": direction_error}
        need = int(max(2000, horizon + 100))
        history_kwargs: Dict[str, Any] = {"as_of": as_of}
        if not as_of and (start or end):
            history_kwargs.update({"start": start, "end": end})
        df = _fetch_history(symbol, timeframe, need, **history_kwargs)
        if len(df) < 10:
            return {"error": "Insufficient history"}
        freshness_context = _history_freshness_context(
            df,
            timeframe,
            symbol=symbol,
        )
        base_col = 'close'
        denoise_applied = False
        denoise_error: Optional[str] = None
        if denoise:
            try:
                normalized = _denoise_api.normalize_denoise_spec(denoise, default_when='pre_ti') or denoise
                added = _denoise_api.apply_denoise(df, normalized, default_when='pre_ti')
                base_col = _denoise_api.effective_denoise_base_col(
                    df,
                    normalized if isinstance(normalized, dict) else denoise,
                    base_col='close',
                    added_columns=added,
                )
                last_application = df.attrs.get("denoise_last_application")
                overwritten = (
                    list(last_application.get("overwrote_columns") or [])
                    if isinstance(last_application, dict)
                    else []
                )
                denoise_applied = bool(added or overwritten)
            except Exception as exc:
                denoise_error = str(exc)
        prices = np.asarray(df[base_col].astype(float).to_numpy(), dtype=float)
        prices = prices[np.isfinite(prices)]
        if prices.size < 5:
            return {"error": "Insufficient prices"}
        (
            last_price_close,
            s0,
            last_price_source,
            price_warning,
            price_error,
        ) = _resolve_reference_prices(
            prices,
            symbol=symbol,
            direction=direction_norm,
            use_live_price=not bool(as_of or start or end),
            live_price_getter=_get_live_reference_price,
        )
        if price_error:
            return {"error": price_error}
        if as_of or start or end:
            last_price_source = "candle_close"
        if barrier <= 0:
            return {"error": "Provide a positive barrier price"}
        tf_secs = TIMEFRAME_SECONDS.get(timeframe, 0)
        if not tf_secs:
            return {"error": unsupported_timeframe_seconds_error(timeframe)}
        bars_per_year_value, annualization_basis = _annualization_context(
            timeframe,
            symbol,
            observed_times=df.get("time"),
            observed_timeframe=timeframe,
        )
        if not np.isfinite(bars_per_year_value) or bars_per_year_value <= 0:
            return {"error": unsupported_timeframe_seconds_error(timeframe)}
        T = float(int(horizon)) / float(bars_per_year_value)
        if mu is None or sigma is None:
            with np.errstate(divide='ignore', invalid='ignore'):
                r = _log_returns_from_prices(prices)
            r = r[np.isfinite(r)]
            if r.size < 5:
                return {"error": "Insufficient returns for calibration"}
            mu_hat = float(np.mean(r)) * float(bars_per_year_value)
            sigma_hat = float(np.std(r, ddof=1)) * float(bars_per_year_value) ** 0.5
            if mu is None:
                mu = mu_hat
            if sigma is None:
                sigma = sigma_hat
        log_drift = float(mu)
        sigma_val = float(sigma)
        if sigma_val <= 0:
            return {"error": "Sigma must be positive"}
        sigma_sq = sigma_val * sigma_val
        gbm_drift = log_drift + 0.5 * sigma_sq
        if direction_norm == 'short':
            s0_inv = 1.0 / s0
            b_inv = 1.0 / float(barrier)
            inv_drift = sigma_sq - gbm_drift
            prob = _gbm_upcross_prob(s0_inv, b_inv, float(inv_drift), sigma_val, float(T))
        else:
            prob = _gbm_upcross_prob(s0, float(barrier), float(gbm_drift), sigma_val, float(T))
        already_hit = (
            (direction_norm == 'long' and barrier <= s0)
            or (direction_norm == 'short' and barrier >= s0)
        )
        price_precision = _symbol_price_precision(symbol)
        result = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "horizon": int(horizon),
            "direction": direction_norm,
            "last_price": s0,
            "last_price_close": float(last_price_close),
            "last_price_source": last_price_source,
            "barrier": float(barrier),
            "mu_annual": float(gbm_drift),
            "log_drift_annual": float(log_drift),
            "sigma_annual": sigma_val,
            "bars_per_year": float(bars_per_year_value),
            "annualization_basis": annualization_basis,
            "override_units": "annual_decimal_return_fraction",
            "prob_hit": float(prob),
            "analysis_mode": (
                "historical_research"
                if as_of or start or end
                else "live_reference"
                if str(last_price_source or "").startswith("live_tick")
                else "research_close_fallback"
            ),
        }
        result.update(freshness_context)
        if price_warning:
            result["warnings"] = [price_warning]
        reference_context: Dict[str, Any] = {}
        if str(last_price_source or "").startswith("live_tick"):
            reference_context = _live_reference_time_context(symbol, timeframe)
        _apply_barrier_freshness_contract(
            result,
            history_context=freshness_context,
            reference_context=reference_context,
            last_price_source=last_price_source,
        )
        result["conditioning_note"] = (
            "Drift and volatility use closed bars through "
            f"{result.get('data_as_of')}; barrier distance starts from "
            f"{result.get('last_price_source')}."
        )
        if denoise:
            result["denoise_applied"] = denoise_applied
            result["denoise_status"] = (
                "applied"
                if denoise_applied
                else "failed"
                if denoise_error is not None
                else "skipped"
            )
            if denoise_error is not None:
                result["denoise_error"] = denoise_error
                denoise_warning = (
                    "Denoise request failed; using raw close prices instead: "
                    f"{denoise_error}"
                )
                existing_warnings = list(result.get("warnings") or [])
                if denoise_warning not in existing_warnings:
                    existing_warnings.append(denoise_warning)
                result["warnings"] = existing_warnings
        if (
            not str(last_price_source or "").startswith("live_tick")
            and freshness_context.get("last_observation_close_time")
        ):
            result["reference_price_time"] = freshness_context.get(
                "last_observation_close_time"
            )
            result["reference_price_time_epoch"] = freshness_context.get(
                "last_observation_close_epoch"
            )
        if price_precision is not None:
            result["price_precision"] = int(price_precision)
        if already_hit:
            result["already_hit"] = True
        return result
    except (KeyError, AttributeError, IndexError):
        raise
    except Exception as e:
        return {
            "error": f"Error computing closed-form barrier probability: {str(e)}",
        }

