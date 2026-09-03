from __future__ import annotations

import logging
import math
import random
import threading
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

from ..utils.coercion import coerce_finite_float
from ..utils.security import redact_url_credentials
from .backtest import _backtest_units
from .backtest import forecast_backtest as _forecast_backtest
from .tuning_contract import (
    MIN_ANNUALIZED_TUNING_TRADES,
    TRADING_TUNING_METRICS,
    resolve_tuning_mode,
)

_NOISY_FORECAST_TUNE_LOGGERS = (
    "timesfm",
    "timesfm_2p5_torch",
    "torchao",
    "torch._dynamo",
    "torch._inductor",
)


def _tuning_units(metric: Any, quantity: Any) -> Dict[str, str]:
    units = _backtest_units(str(quantity or "price"))
    metric_key = str(metric or "").strip()
    metric_unit = units.get(metric_key, "dimensionless")
    units["best_score"] = metric_unit
    units["score"] = metric_unit
    return units


def _copy_best_backtest_provenance(
    payload: Dict[str, Any],
    best_result: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(best_result, dict):
        return
    window = best_result.get("analysis_time_window")
    if isinstance(window, dict) and window:
        payload["analysis_time_window"] = dict(window)
    plan = best_result.get("backtest_plan")
    if not isinstance(plan, dict):
        return
    if plan.get("model_lookback_bars") is not None:
        payload["model_lookback_bars"] = plan.get("model_lookback_bars")
    if plan.get("history_bars_used") is not None:
        payload["history_bars_used"] = plan.get("history_bars_used")


def _optimization_fitness_unit(metric: Any) -> str:
    metric_key = str(metric or "").strip()
    if metric_key == "composite":
        return "dimensionless"
    return _backtest_units("price").get(metric_key, "dimensionless")


def _suppress_noisy_forecast_tune_loggers() -> None:
    for logger_name in _NOISY_FORECAST_TUNE_LOGGERS:
        noisy_logger = logging.getLogger(logger_name)
        if noisy_logger.level == logging.NOTSET or noisy_logger.getEffectiveLevel() < logging.WARNING:
            noisy_logger.setLevel(logging.WARNING)


def _finite_metric(metrics: Dict[str, Any], key: str) -> Optional[float]:
    return coerce_finite_float(metrics.get(key))


def _has_complete_anchor_coverage(result: Any) -> bool:
    """Reject partial candidate metrics under the current backtest contract."""
    if not isinstance(result, dict) or result.get("success") is False:
        return False

    status = str(result.get("status") or "").strip().lower()
    if status in {"partial", "failed"}:
        return False
    if "complete_success" in result and result.get("complete_success") is not True:
        return False

    failed_tests = coerce_finite_float(result.get("failed_tests"))
    if failed_tests is not None and failed_tests != 0.0:
        return False

    successful_tests = coerce_finite_float(result.get("successful_tests"))
    num_tests = coerce_finite_float(result.get("num_tests"))
    if successful_tests is not None and num_tests is not None:
        if (
            successful_tests < 0.0
            or num_tests <= 0.0
            or successful_tests != num_tests
        ):
            return False

    # No public common-anchor score exists yet, so ad-hoc fields cannot bypass
    # the complete-coverage checks above. Legacy payloads without coverage
    # diagnostics retain their prior behavior.
    return True


def _extract_method_backtest_metrics(
    backtest_res: Dict[str, Any],
    method_name: str,
) -> Dict[str, Any]:
    """Return one canonical metric map across nested and aggregate layouts."""
    if not isinstance(backtest_res, dict):
        return {}
    method_results = backtest_res.get("results", {}).get(method_name, {})
    if not isinstance(method_results, dict):
        return {}
    nested = method_results.get("metrics")
    merged = dict(nested) if isinstance(nested, dict) else {}
    for key, value in method_results.items():
        if key == "metrics" or isinstance(value, (dict, list, tuple, set)):
            continue
        merged.setdefault(str(key), value)
    return merged


def _trading_sample_metadata(metrics: Dict[str, Any]) -> Dict[str, Any]:
    trades_observed = _finite_metric(metrics, "trades_observed")
    reliability = str(metrics.get("metrics_reliability") or "").strip().lower()
    trades_count = int(trades_observed) if trades_observed is not None else 0
    if not reliability:
        reliability = (
            "standard"
            if trades_count >= MIN_ANNUALIZED_TUNING_TRADES
            else "low"
            if trades_count > 0
            else "unavailable"
        )
    reason = metrics.get("metrics_reliability_reason")
    if reason is None and reliability == "low":
        reason = "low_sample"
    sample_notice = metrics.get("sample_notice")
    if sample_notice is None and reliability == "low":
        sample_notice = {
            "code": "trading_fitness_suppressed_low_sample",
            "trades_observed": trades_count,
            "minimum_trades": MIN_ANNUALIZED_TUNING_TRADES,
        }
    sample_warning = metrics.get("sample_warning")
    if sample_warning is None and reliability == "low":
        sample_warning = (
            f"Only {trades_count} trade(s) observed; trading-composite fitness "
            f"requires at least {MIN_ANNUALIZED_TUNING_TRADES}."
        )
    return {
        "trades_observed": trades_count,
        "metrics_reliability": reliability,
        "metrics_reliability_reason": reason,
        "sample_notice": sample_notice,
        "sample_warning": sample_warning,
        "minimum_trades_for_comparable_fitness": MIN_ANNUALIZED_TUNING_TRADES,
    }


def _has_trading_fitness_metrics(metrics: Dict[str, Any]) -> bool:
    sample = _trading_sample_metadata(metrics)
    sample_is_reliable = (
        sample["trades_observed"] >= MIN_ANNUALIZED_TUNING_TRADES
        and sample["metrics_reliability"] not in {"low", "empty", "unavailable"}
    )
    return sample_is_reliable and any(
        _finite_metric(metrics, key) is not None
        for key in (
            "sharpe_ratio",
            "win_rate",
            "max_drawdown",
            "avg_return_per_trade",
        )
    )


def _forecast_accuracy_fitness(metrics: Dict[str, Any]) -> float:
    directional = _finite_metric(metrics, "avg_directional_accuracy")
    error = _finite_metric(metrics, "avg_rmse")
    if error is None:
        error = _finite_metric(metrics, "avg_mae")

    components: List[float] = []
    if directional is not None:
        components.append(max(0.0, min(1.0, directional)))
    if error is not None and error >= 0.0:
        components.append(1.0 / (1.0 + error))
    return float(sum(components) / len(components)) if components else 0.0


# Sensible default search spaces per method (lightweight, CPU-friendly)
# These are intentionally conservative to keep runtime practical.
_DEFAULT_SPACES_METHOD_SCOPED: Dict[str, Dict[str, Any]] = {
    "_shared": {},
    # Classical fast methods
    # Native Theta fits its own smoothing parameters and accepts no method params.
    "theta": {},
    "fourier_ols": {
        # Fourier period (bars) and number of harmonics; allow optional trend toggle
        "seasonality": {"type": "int", "min": 8, "max": 96},
        "terms": {"type": "int", "min": 1, "max": 6},
        "trend": {"type": "categorical", "choices": [True, False]},
    },
    "seasonal_naive": {
        # Period for repeating last seasonal value
        "seasonality": {"type": "int", "min": 5, "max": 96},
    },
    "naive": {},
    "drift": {},
    "ses": {
        # Smoothing level for Simple Exponential Smoothing
        "alpha": {"type": "float", "min": 0.05, "max": 0.95},
    },
    "holt": {
        # Damped trend on/off
        "damped": {"type": "categorical", "choices": [True, False]},
    },
    "holt_winters_add": {
        "seasonality": {"type": "int", "min": 8, "max": 72},
    },
    "holt_winters_mul": {
        "seasonality": {"type": "int", "min": 8, "max": 72},
    },
    "arima": {
        "p": {"type": "int", "min": 0, "max": 3},
        "d": {"type": "int", "min": 0, "max": 2},
        "q": {"type": "int", "min": 0, "max": 3},
    },
    "sarima": {
        "p": {"type": "int", "min": 0, "max": 3},
        "d": {"type": "int", "min": 0, "max": 2},
        "q": {"type": "int", "min": 0, "max": 3},
        "P": {"type": "int", "min": 0, "max": 2},
        "D": {"type": "int", "min": 0, "max": 1},
        "Q": {"type": "int", "min": 0, "max": 2},
        "seasonality": {"type": "int", "min": 4, "max": 48},
    },
    # Monte Carlo
    "mc_gbm": {
        "n_sims": {"type": "int", "min": 200, "max": 1000},
        "seed": {"type": "categorical", "choices": [13, 37, 42, 99]},
    },
    "hmm_mc": {
        "n_sims": {"type": "int", "min": 200, "max": 1000},
        "n_states": {"type": "int", "min": 2, "max": 4},
        "seed": {"type": "categorical", "choices": [13, 37, 42, 99]},
    },
    # NeuralForecast (lightweight ranges)
    "nhits": {
        "input_size": {"type": "int", "min": 64, "max": 256},
        "max_epochs": {"type": "int", "min": 10, "max": 50},
        "batch_size": {"type": "int", "min": 16, "max": 64},
        "learning_rate": {"type": "float", "min": 1e-4, "max": 1e-2, "log": True},
    },
    "nbeatsx": {
        "input_size": {"type": "int", "min": 64, "max": 256},
        "max_epochs": {"type": "int", "min": 10, "max": 50},
        "batch_size": {"type": "int", "min": 16, "max": 64},
        "learning_rate": {"type": "float", "min": 1e-4, "max": 1e-2, "log": True},
    },
    "tft": {
        "input_size": {"type": "int", "min": 64, "max": 256},
        "max_epochs": {"type": "int", "min": 10, "max": 50},
        "batch_size": {"type": "int", "min": 16, "max": 64},
        "learning_rate": {"type": "float", "min": 1e-4, "max": 1e-2, "log": True},
    },
    "patchtst": {
        "input_size": {"type": "int", "min": 64, "max": 256},
        "max_epochs": {"type": "int", "min": 10, "max": 50},
        "batch_size": {"type": "int", "min": 16, "max": 64},
        "learning_rate": {"type": "float", "min": 1e-4, "max": 1e-2, "log": True},
    },
    # StatsForecast
    "sf_autoarima": {
        "seasonality": {"type": "int", "min": 8, "max": 72},
        "stepwise": {"type": "categorical", "choices": [True, False]},
        "d": {"type": "int", "min": 0, "max": 2},
        "D": {"type": "int", "min": 0, "max": 1},
    },
    "sf_theta": {
        "seasonality": {"type": "int", "min": 8, "max": 72},
    },
    "sf_autoets": {
        "seasonality": {"type": "int", "min": 8, "max": 72},
    },
    "sf_seasonalnaive": {
        "seasonality": {"type": "int", "min": 5, "max": 96},
    },
    # MLForecast
    "mlf_rf": {
        "n_estimators": {"type": "int", "min": 100, "max": 500},
        "max_depth": {"type": "categorical", "choices": [None, 5, 10, 15, 20]},
    },
    "mlf_lightgbm": {
        "n_estimators": {"type": "int", "min": 100, "max": 500},
        "learning_rate": {"type": "float", "min": 0.01, "max": 0.2},
        "num_leaves": {"type": "int", "min": 15, "max": 63},
        "max_depth": {"type": "categorical", "choices": [-1, 6, 8, 12, 16]},
    },
    # Transformer family (point forecasts use context length primarily)
    "chronos_bolt": {
        "context_length": {"type": "int", "min": 64, "max": 320},
    },
    "chronos2": {
        "context_length": {"type": "int", "min": 64, "max": 320},
    },
    "timesfm": {
        "context_length": {"type": "int", "min": 64, "max": 320},
    },
    # Ensemble (not implemented): placeholder
    "ensemble": {},
}


def default_search_space(method: Optional[str] = None, methods: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return a sensible default search space.

    - Multiple methods: returns a method-scoped dict with sections for each listed method
      (falling back to shared defaults where available).
    - Single method: returns a flat parameter space for that method.
    - If neither provided, returns method-scoped defaults for a small common set.
    """
    if methods and isinstance(methods, (list, tuple)) and len(methods) > 0:
        out: Dict[str, Any] = {"_shared": dict(_DEFAULT_SPACES_METHOD_SCOPED.get("_shared", {}))}
        for m in methods:
            if m in _DEFAULT_SPACES_METHOD_SCOPED:
                out[m] = dict(_DEFAULT_SPACES_METHOD_SCOPED[m])
        # Ensure at least something is present
        if len(out) == 1:  # only _shared
            # add a couple of common ones if user passed unknowns
            for m in ("theta", "fourier_ols"):
                out[m] = dict(_DEFAULT_SPACES_METHOD_SCOPED[m])
        return out
    if method:
        method_key = str(method)
        if method_key in _DEFAULT_SPACES_METHOD_SCOPED:
            sp = _DEFAULT_SPACES_METHOD_SCOPED.get(method_key)
            if isinstance(sp, dict):
                return dict(sp)
        # Fallback to a generic seasonality search for classical methods
        return {"seasonality": {"type": "int", "min": 8, "max": 48}}
    # Neither provided: return a compact method-scoped default
    return {
        "_shared": {},
        "theta": dict(_DEFAULT_SPACES_METHOD_SCOPED["theta"]),
        "fourier_ols": dict(_DEFAULT_SPACES_METHOD_SCOPED["fourier_ols"]),
    }


Metric = str
_SEARCH_SPACE_SHARED_KEY = "_shared"


def _is_flat_search_space(sp: Dict[str, Any]) -> bool:
    return any(
        isinstance(v, dict) and ('type' in v or 'min' in v or 'max' in v or 'choices' in v)
        for v in sp.values()
    )


def _resolve_method_search_space(
    raw: Dict[str, Any],
    *,
    method_scoped: bool,
    method_name: Optional[str],
) -> Dict[str, Any]:
    """Return the effective parameter space for one forecast method."""
    if not method_scoped:
        space = dict(raw)
        space.pop("method", None)
        return {key: _normalize_param_space(spec) for key, spec in space.items()}

    resolved: Dict[str, Any] = {}
    shared = raw.get(_SEARCH_SPACE_SHARED_KEY)
    if isinstance(shared, dict):
        resolved.update(shared)
    if method_name and isinstance(raw.get(method_name), dict):
        resolved.update(raw[method_name])
    return {key: _normalize_param_space(spec) for key, spec in resolved.items()}


def _try_int_bound(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _try_float_bound(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _normalize_param_space(space: Any) -> Dict[str, Any]:
    """Coerce one search-space mapping into a stable typed structure."""
    spec = dict(space or {}) if isinstance(space, dict) else {}
    kind = str(spec.get("type", "float")).lower()
    if kind == "categorical":
        return {"type": "categorical", "choices": list(spec.get("choices") or [])}
    if kind in {"int", "float"} and "has_min" in spec and "has_max" in spec:
        return spec
    if kind == "int":
        return {
            "type": "int",
            "min": _try_int_bound(spec["min"]) if "min" in spec else None,
            "max": _try_int_bound(spec["max"]) if "max" in spec else None,
            "has_min": "min" in spec,
            "has_max": "max" in spec,
        }
    return {
        "type": "float",
        "min": _try_float_bound(spec["min"]) if "min" in spec else None,
        "max": _try_float_bound(spec["max"]) if "max" in spec else None,
        "has_min": "min" in spec,
        "has_max": "max" in spec,
        "log": bool(spec.get("log", False)),
    }


def _resolved_int_bound(
    space: Dict[str, Any],
    key: str,
    *,
    present_key: str,
    fallback: int,
    invalid: int,
) -> int:
    if space.get(key) is not None:
        return int(space[key])
    if space.get(present_key):
        return int(invalid)
    return int(fallback)


def _resolved_float_bound(
    space: Dict[str, Any],
    key: str,
    *,
    present_key: str,
    fallback: float,
    invalid: float,
) -> float:
    if space.get(key) is not None:
        return float(space[key])
    if space.get(present_key):
        return float(invalid)
    return float(fallback)


def _suggest_optuna_param(trial: Any, name: str, space: Dict[str, Any]) -> Any:
    space = _normalize_param_space(space)
    kind = str(space.get("type", "float")).lower()
    if kind == "categorical":
        choices = list(space.get("choices") or [])
        if not choices:
            return None
        return trial.suggest_categorical(name, choices)
    if kind == "int":
        lo = _resolved_int_bound(space, "min", present_key="has_min", fallback=0, invalid=0)
        hi = _resolved_int_bound(space, "max", present_key="has_max", fallback=lo, invalid=lo)
        if lo > hi:
            lo, hi = hi, lo
        return int(trial.suggest_int(name, lo, hi))
    lo = _resolved_float_bound(space, "min", present_key="has_min", fallback=0.0, invalid=0.0)
    hi = _resolved_float_bound(
        space, "max", present_key="has_max", fallback=max(lo, 1.0), invalid=max(lo, 1.0)
    )
    if lo > hi:
        lo, hi = hi, lo
    if bool(space.get("log", False)) and lo > 0.0 and hi > 0.0:
        return float(trial.suggest_float(name, lo, hi, log=True))
    return float(trial.suggest_float(name, lo, hi))


def _eval_candidate(
    *,
    symbol: str,
    timeframe: str,
    method: Optional[str],
    horizon: int,
    steps: int,
    spacing: int,
    quantity: str = "price",
    as_of: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lookback: Optional[int] = None,
    candidate_params: Dict[str, Any],
    metric: Metric,
    mode: str,
    denoise: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    slippage_bps: float = 0.0,
    spread_bps: Optional[float] = None,
    commission_bps_per_side: Optional[float] = None,
    trade_threshold: float = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    """Run a backtest for a single candidate and return (score, result_dict).

    mode: 'min' or 'max'. score is direction-consistent (lower is better if mode == 'min').
    """
    # Allow method gene inside candidate
    sel_method = str(candidate_params.get('method')) if candidate_params.get('method') else (str(method) if method else None)
    if not sel_method:
        return math.inf, {"error": "No method provided"}
    cand_only = {k: v for k, v in candidate_params.items() if k != 'method'}
    res = _forecast_backtest(
        symbol=symbol,
        timeframe=timeframe,  # type: ignore
        horizon=int(horizon),
        steps=int(steps),
        spacing=int(spacing),
        quantity=str(quantity),
        as_of=as_of,
        start=start,
        end=end,
        lookback=lookback,
        methods=[sel_method],
        params_per_method={sel_method: cand_only},
        denoise=denoise,
        features=features,
        dimred_method=dimred_method,
        dimred_params=dimred_params,
        slippage_bps=float(slippage_bps),
        spread_bps=spread_bps,
        commission_bps_per_side=commission_bps_per_side,
        trade_threshold=float(trade_threshold),
        detail="full",
    )
    # Pull method aggregate
    r = res.get('results', {}).get(sel_method) if isinstance(res, dict) else None
    if not isinstance(r, dict) or not r.get('success'):
        return math.inf, res
    if not _has_complete_anchor_coverage(r):
        result = {'_sel_method': sel_method, **(res or {})}
        result['tuning_error'] = (
            f"Candidate method '{sel_method}' has incomplete anchor coverage; "
            "partial aggregate metrics are not valid tuning fitness."
        )
        result['error_code'] = 'incomplete_anchor_coverage'
        return math.inf, result
    metrics = _extract_method_backtest_metrics(res, sel_method)
    if str(metric) in TRADING_TUNING_METRICS and not _has_trading_fitness_metrics(metrics):
        result = {"_sel_method": sel_method, **(res or {})}
        sample = _trading_sample_metadata(metrics)
        result["tuning_error"] = (
            f"Trading metric '{metric}' requires at least "
            f"{sample['minimum_trades_for_comparable_fitness']} observed trades; "
            f"got {sample['trades_observed']}."
        )
        result["error_code"] = "insufficient_tuning_sample"
        result["trading_sample"] = sample
        return math.inf, result
    score = _finite_metric(metrics, str(metric))
    if score is None:
        result = {'_sel_method': sel_method, **(res or {})}
        result['tuning_error'] = (
            f"Requested metric '{metric}' is missing or non-finite for method "
            f"'{sel_method}'."
        )
        result['metric_requested'] = str(metric)
        return math.inf, result
    return (score if mode == 'min' else -score, {'_sel_method': sel_method, **(res or {})})


def _candidate_failure(res: Any) -> Dict[str, Any]:
    """Extract one actionable failure from a failed candidate evaluation."""
    if not isinstance(res, dict):
        return {"error": "Candidate evaluation returned no structured result."}
    for key in ("tuning_error", "error"):
        if res.get(key):
            out = {"error": str(res[key])}
            if res.get("error_code"):
                out["error_code"] = str(res["error_code"])
            return out
    results = res.get("results")
    if isinstance(results, dict):
        for method_name, method_result in results.items():
            if not isinstance(method_result, dict):
                continue
            for key in ("tuning_error", "error"):
                if method_result.get(key):
                    out = {
                        "method": str(method_name),
                        "error": str(method_result[key]),
                    }
                    if method_result.get("error_code"):
                        out["error_code"] = str(method_result["error_code"])
                    return out
    return {"error": "Candidate did not produce a finite requested metric."}


def _failure_causes(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[tuple[str, str], int] = {}
    for row in history:
        error = str(row.get("failure_reason") or "").strip()
        if not error:
            continue
        code = str(row.get("failure_code") or "candidate_failed")
        counts[(code, error)] = counts.get((code, error), 0) + 1
    return [
        {"error_code": code, "error": error, "count": count}
        for (code, error), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[:5]
    ]


def _sample_param(space: Dict[str, Any], rng: random.Random) -> Any:
    space = _normalize_param_space(space)
    kind = str(space.get("type", "float")).lower()
    if kind == "categorical":
        choices = space.get("choices") or []
        return rng.choice(list(choices)) if choices else None
    if kind == "int":
        lo = _resolved_int_bound(space, "min", present_key="has_min", fallback=0, invalid=0)
        hi = _resolved_int_bound(space, "max", present_key="has_max", fallback=lo, invalid=lo)
        if lo > hi:
            lo, hi = hi, lo
        return rng.randint(lo, hi)
    lo = _resolved_float_bound(space, "min", present_key="has_min", fallback=0.0, invalid=0.0)
    hi = _resolved_float_bound(
        space, "max", present_key="has_max", fallback=max(lo, 1.0), invalid=max(lo, 1.0)
    )
    if lo > hi:
        lo, hi = hi, lo
    if bool(space.get("log", False)) and lo > 0 and hi > 0:
        import math as _m
        a = _m.log(lo)
        b = _m.log(hi)
        x = rng.random() * (b - a) + a
        return float(_m.exp(x))
    return rng.random() * (hi - lo) + lo


def _mutate_value(value: Any, space: Dict[str, Any], rng: random.Random, strength: float = 0.2) -> Any:
    space = _normalize_param_space(space)
    kind = str(space.get("type", "float")).lower()
    if kind == "categorical":
        choices = list(space.get("choices") or [])
        if not choices:
            return value
        if len(choices) == 1:
            return choices[0]
        cand = [c for c in choices if c != value]
        return rng.choice(cand) if cand else value
    if kind == "int":
        value_lo = int(value) if value is not None else 0
        lo = _resolved_int_bound(
            space, "min", present_key="has_min", fallback=value_lo, invalid=0
        )
        hi = _resolved_int_bound(
            space, "max", present_key="has_max", fallback=value_lo, invalid=lo
        )
        if value is None:
            value = int((lo + hi) // 2)
        span = max(1, int(round((hi - lo) * strength)))
        return max(lo, min(hi, int(value) + rng.randint(-span, span)))
    value_lo = float(value) if value is not None else 0.0
    lo = _resolved_float_bound(
        space, "min", present_key="has_min", fallback=value_lo, invalid=0.0
    )
    hi = _resolved_float_bound(
        space, "max", present_key="has_max", fallback=value_lo, invalid=max(lo, 1.0)
    )
    if value is None:
        try:
            value = float((lo + hi) * 0.5)
        except Exception:
            value = lo
    span = (hi - lo) * max(0.01, strength)
    return max(lo, min(hi, float(value) + (rng.random() * 2.0 - 1.0) * span))


def _crossover_for_method(
    a: Dict[str, Any],
    b: Dict[str, Any],
    param_spaces: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    child: Dict[str, Any] = {}
    for k, spec in param_spaces.items():
        av = a.get(k)
        bv = b.get(k)
        child[k] = av if (rng.random() < 0.5) else bv
        if child[k] is None and av is None and bv is None:
            try:
                child[k] = _sample_param(spec or {}, rng)
            except Exception:
                child[k] = None
        # small blend for floats
        t = str(spec.get('type', 'float')).lower()
        if t not in ('categorical', 'int'):
            try:
                fa = float(av if av is not None else _sample_param(spec or {}, rng))
                fb = float(bv if bv is not None else _sample_param(spec or {}, rng))
                child[k] = fa * 0.5 + fb * 0.5
            except Exception:
                pass
    return child


def optuna_search_forecast_params(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    method: Optional[str],
    methods: Optional[List[str]] = None,
    horizon: int = 12,
    steps: int = 5,
    spacing: int = 20,
    quantity: str = "price",
    as_of: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lookback: Optional[int] = None,
    search_space: Optional[Dict[str, Any]] = None,
    metric: Metric = 'avg_rmse',
    mode: str = 'auto',
    n_trials: int = 40,
    timeout: Optional[float] = None,
    n_jobs: int = 1,
    sampler: str = 'tpe',
    seed: int = 42,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    denoise: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    slippage_bps: float = 0.0,
    spread_bps: Optional[float] = None,
    commission_bps_per_side: Optional[float] = None,
    trade_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Optuna search for best params for a forecast method under backtest."""
    import optuna

    mode_val = resolve_tuning_mode(str(metric), str(mode or 'auto'))

    raw = dict(search_space or {})
    method_scoped = not _is_flat_search_space(raw)
    method_names_from_space: List[str] = []
    if method_scoped:
        method_names_from_space = [k for k in raw.keys() if k != _SEARCH_SPACE_SHARED_KEY]

    method_choices: List[str] = []
    if isinstance(methods, (list, tuple)) and methods:
        method_choices = list(methods)
    elif method_scoped and method_names_from_space:
        method_choices = list(method_names_from_space)

    has_method_gene = (
        (not method_scoped)
        and ('method' in raw)
        and isinstance(raw.get('method'), dict)
        and str(raw['method'].get('type', 'categorical')).lower() == 'categorical'
    )

    sampler_name = str(sampler or 'tpe').strip().lower()
    if sampler_name == 'random':
        sampler_obj = optuna.samplers.RandomSampler(seed=int(seed))
    elif sampler_name == 'cmaes':
        sampler_obj = optuna.samplers.CmaEsSampler(seed=int(seed))
    else:
        sampler_name = 'tpe'
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*multivariate.*experimental.*",
                category=Warning,
            )
            sampler_obj = optuna.samplers.TPESampler(seed=int(seed), multivariate=True)

    storage_val: Optional[str]
    if storage is None:
        storage_val = None
    else:
        storage_str = str(storage).strip()
        storage_val = storage_str or None
    study_name_val: Optional[str]
    if study_name is None:
        study_name_val = None
    else:
        name_str = str(study_name).strip()
        study_name_val = name_str or None
    load_if_exists = bool(storage_val and study_name_val)

    direction = 'minimize' if mode_val == 'min' else 'maximize'
    study = optuna.create_study(
        direction=direction,
        sampler=sampler_obj,
        study_name=study_name_val,
        storage=storage_val,
        load_if_exists=load_if_exists,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    history: List[Dict[str, Any]] = []
    lock = threading.Lock()

    best_score = math.inf if mode_val == 'min' else -math.inf
    best_params: Dict[str, Any] = {}
    best_result: Optional[Dict[str, Any]] = None
    successful_evaluations = 0

    def _objective(trial: Any) -> float:
        nonlocal best_score, best_params, best_result, successful_evaluations
        cand: Dict[str, Any] = {}

        sel_method = None
        if has_method_gene:
            try:
                sel_method = _suggest_optuna_param(trial, 'method', raw['method'])
                cand['method'] = sel_method
            except Exception:
                sel_method = None
        elif method_choices:
            sel_method = trial.suggest_categorical('method', list(method_choices))
            cand['method'] = sel_method
        else:
            sel_method = method
            if sel_method is not None:
                cand['method'] = sel_method

        pspaces = _resolve_method_search_space(
            raw,
            method_scoped=method_scoped,
            method_name=str(sel_method) if sel_method else None,
        )
        for k, spec in pspaces.items():
            cand[k] = _suggest_optuna_param(trial, k, spec or {})

        score, res = _eval_candidate(
            symbol=symbol,
            timeframe=timeframe,
            method=method,
            horizon=horizon,
            steps=steps,
            spacing=spacing,
            quantity=quantity,
            as_of=as_of,
            start=start,
            end=end,
            lookback=lookback,
            candidate_params=cand,
            metric=metric,
            mode=mode_val,
            denoise=denoise,
            features=features,
            dimred_method=dimred_method,
            dimred_params=dimred_params,
            slippage_bps=float(slippage_bps),
            spread_bps=spread_bps,
            commission_bps_per_side=commission_bps_per_side,
            trade_threshold=float(trade_threshold),
        )

        true_score = score if mode_val == 'min' else -score
        finite_score = float(true_score) if math.isfinite(true_score) else None
        objective_score = float(true_score)
        if not math.isfinite(objective_score):
            objective_score = 1e18 if mode_val == 'min' else -1e18

        with lock:
            hist_row: Dict[str, Any] = {"trial": int(trial.number), "score": float(objective_score), "params": dict(cand)}
            if isinstance(res, dict) and res.get('_sel_method'):
                hist_row['method'] = res.get('_sel_method')
            if finite_score is None:
                failure = _candidate_failure(res)
                hist_row["failure_reason"] = failure["error"]
                if failure.get("error_code"):
                    hist_row["failure_code"] = failure["error_code"]
            history.append(hist_row)

            if finite_score is not None:
                successful_evaluations += 1
                better = (
                    (mode_val == 'min' and finite_score < best_score)
                    or (mode_val != 'min' and finite_score > best_score)
                )
                if better:
                    best_score = finite_score
                    best_params = dict(cand)
                    best_result = res if isinstance(res, dict) else None

        return float(objective_score)

    timeout_val = None
    if timeout is not None:
        try:
            timeout_float = float(timeout)
            timeout_val = timeout_float if timeout_float > 0 else None
        except Exception:
            timeout_val = None
    n_jobs_val = max(1, int(n_jobs))
    n_trials_val = max(1, int(n_trials))
    study.optimize(_objective, n_trials=n_trials_val, timeout=timeout_val, n_jobs=n_jobs_val)

    if successful_evaluations == 0:
        return {
            "success": False,
            "error": "No candidate produced a finite requested metric.",
            "error_code": "no_successful_trials",
            "metric": metric,
            "mode": mode_val,
            "optimizer": "optuna",
            "n_trials": int(n_trials_val),
            "history_count": len(history),
            "failure_causes": _failure_causes(history),
            "failed_trials": history[:5],
        }

    payload: Dict[str, Any] = {
        "success": True,
        "best_score": float(best_score),
        "best_params": best_params,
        "metric": metric,
        "units": _tuning_units(metric, quantity),
        "mode": mode_val,
        "optimizer": "optuna",
        "n_trials": int(n_trials_val),
        "timeout": float(timeout_val) if timeout_val is not None else None,
        "n_jobs": int(n_jobs_val),
        "sampler": sampler_name,
        "seed": int(seed),
        "history_count": len(history),
    }
    if storage_val:
        payload["storage"] = redact_url_credentials(storage_val)
    if study_name_val:
        payload["study_name"] = study_name_val
    if best_result is not None:
        sel = best_result.get('_sel_method') if isinstance(best_result, dict) else None
        if not sel:
            sel = best_params.get('method') or method
        agg = None
        try:
            agg = best_result.get('results', {}).get(sel) if isinstance(best_result, dict) else None
        except Exception:
            agg = None
        payload["best_method"] = sel
        if isinstance(agg, dict):
            payload["best_result_summary"] = {"horizon": int(horizon), "result": agg}
        _copy_best_backtest_provenance(payload, best_result)
    if history:
        compact_tail = _compact_optuna_history_tail(history, limit=10)
        payload["history_tail"] = compact_tail
        payload["history_tail_limit"] = 10
    return payload


def _compact_optuna_history_tail(
    history: List[Dict[str, Any]],
    *,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    tail = [dict(row) for row in history[-max(1, int(limit)):]]
    methods = {
        str(row.get("method"))
        for row in tail
        if row.get("method") not in (None, "")
    }
    single_method = next(iter(methods)) if len(methods) == 1 else None
    out: List[Dict[str, Any]] = []
    for row in tail:
        params = row.get("params")
        if isinstance(params, dict):
            params = dict(params)
            if single_method is not None and params.get("method") == single_method:
                params.pop("method", None)
            row["params"] = params
        if single_method is not None:
            row.pop("method", None)
        out.append(row)
    return out


def genetic_search_forecast_params(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    method: Optional[str],
    methods: Optional[List[str]] = None,
    horizon: int = 12,
    steps: int = 5,
    spacing: int = 20,
    quantity: str = "price",
    as_of: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lookback: Optional[int] = None,
    search_space: Optional[Dict[str, Any]] = None,
    metric: Metric = 'avg_rmse',
    mode: str = 'auto',
    population: int = 12,
    generations: int = 10,
    crossover_rate: float = 0.6,
    mutation_rate: float = 0.3,
    seed: int = 42,
    max_search_time_seconds: Optional[float] = None,
    denoise: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    slippage_bps: float = 0.0,
    spread_bps: Optional[float] = None,
    commission_bps_per_side: Optional[float] = None,
    trade_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Genetic search for best params for a forecast method under backtest.

    - search_space: {param: {type: 'int'|'float'|'categorical', min, max, choices?, log?}}
    - metric: one of backtest aggregates (e.g., 'avg_rmse', 'avg_mae', 'avg_directional_accuracy')
    - mode: 'auto' (metric-aware), 'min', or 'max' (direction)
    """
    if max_search_time_seconds is not None and float(max_search_time_seconds) <= 0:
        raise ValueError("max_search_time_seconds must be greater than 0")
    search_started_at = time.monotonic()
    deadline = (
        search_started_at + float(max_search_time_seconds)
        if max_search_time_seconds is not None
        else None
    )
    mode = resolve_tuning_mode(str(metric), str(mode or 'auto'))
    rng = random.Random(int(seed))
    raw = dict(search_space or {})

    # Detect method-scoped search space vs flat
    method_scoped = not _is_flat_search_space(raw)
    method_names_from_space: List[str] = []
    if method_scoped:
        method_names_from_space = [k for k in raw.keys() if k != _SEARCH_SPACE_SHARED_KEY]

    # Build method choices if searching over methods
    method_choices: List[str] = []
    if isinstance(methods, (list, tuple)) and methods:
        method_choices = list(methods)
    elif method_scoped and method_names_from_space:
        method_choices = list(method_names_from_space)

    # If there are method choices and no explicit 'method' gene in a flat space, we will sample it explicitly
    has_method_gene = (not method_scoped) and ('method' in raw) and str(raw['method'].get('type', 'categorical')).lower() == 'categorical' if 'method' in raw and isinstance(raw['method'], dict) else False

    # Initialize population
    population_size = int(population)
    if population_size < 2:
        raise ValueError("population must be greater than or equal to 2")
    if population_size > 100:
        raise ValueError("population must be less than or equal to 100")
    generation_count = int(generations)
    if generation_count < 1:
        raise ValueError("generations must be greater than or equal to 1")
    if generation_count > 100:
        raise ValueError("generations must be less than or equal to 100")
    pop: List[Dict[str, Any]] = []
    for _ in range(population_size):
        cand: Dict[str, Any] = {}
        # Choose method if searching across methods
        sel_method = None
        if has_method_gene:
            try:
                sel_method = _sample_param(raw['method'], rng)
                cand['method'] = sel_method
            except Exception:
                sel_method = None
        elif method_choices:
            sel_method = rng.choice(method_choices)
            cand['method'] = sel_method
        else:
            sel_method = method  # fixed
        # Sample params for selected method
        pspaces = _resolve_method_search_space(
            raw,
            method_scoped=method_scoped,
            method_name=str(sel_method) if sel_method else None,
        )
        for k, spec in pspaces.items():
            cand[k] = _sample_param(spec or {}, rng)
        pop.append(cand)

    history: List[Dict[str, Any]] = []
    best_score = math.inf if mode == 'min' else -math.inf
    best_params: Dict[str, Any] = {}
    best_result: Optional[Dict[str, Any]] = None
    successful_evaluations = 0
    evaluations_planned = population_size * generation_count
    generations_completed = 0
    timed_out = False

    for gen in range(generation_count):
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for cand in pop:
            score, res = _eval_candidate(
                symbol=symbol,
                timeframe=timeframe,
                method=method,
                horizon=horizon,
                steps=steps,
                spacing=spacing,
                quantity=quantity,
                as_of=as_of,
                start=start,
                end=end,
                lookback=lookback,
                candidate_params=cand,
                metric=metric,
                mode=mode,
                denoise=denoise,
                features=features,
                dimred_method=dimred_method,
                dimred_params=dimred_params,
                slippage_bps=float(slippage_bps),
                spread_bps=spread_bps,
                commission_bps_per_side=commission_bps_per_side,
                trade_threshold=float(trade_threshold),
            )
            scored.append((score, cand))
            hist_entry = {"generation": gen, "score": float(score), "params": dict(cand)}
            if isinstance(res, dict) and res.get('_sel_method'):
                hist_entry['method'] = res.get('_sel_method')
            if not math.isfinite(score):
                failure = _candidate_failure(res)
                hist_entry["failure_reason"] = failure["error"]
                if failure.get("error_code"):
                    hist_entry["failure_code"] = failure["error_code"]
            history.append(hist_entry)
            # Keep global best in true metric direction
            true_score = score if mode == 'min' else -score
            if math.isfinite(true_score):
                successful_evaluations += 1
            if math.isfinite(true_score) and (
                (mode == 'min' and true_score < best_score)
                or (mode != 'min' and true_score > best_score)
            ):
                best_score = true_score
                best_params = dict(cand)
                best_result = res if isinstance(res, dict) else None

            if (
                deadline is not None
                and len(history) < evaluations_planned
                and time.monotonic() >= deadline
            ):
                timed_out = True
                break

        if len(scored) == len(pop):
            generations_completed += 1
        if timed_out:
            break

        # Selection (elitism: top 2)
        scored.sort(key=lambda t: t[0])  # ascending in adjusted score
        elites = [dict(scored[0][1]), dict(scored[1][1])] if len(scored) >= 2 else [dict(scored[0][1])]

        # Breed new population
        new_pop: List[Dict[str, Any]] = []
        new_pop.extend(elites)
        while len(new_pop) < len(pop):
            # Tournament selection
            a = rng.choice(scored)[1]
            b = rng.choice(scored)[1]
            # Decide child method
            child: Dict[str, Any] = {}
            if has_method_gene:
                # crossover method gene from parents (random pick)
                child_method = rng.choice([a.get('method'), b.get('method')])
                child['method'] = child_method
            elif method_choices:
                child_method = rng.choice([a.get('method'), b.get('method')]) or rng.choice(method_choices)
                child['method'] = child_method
            else:
                child_method = method
                if child_method is not None:
                    child['method'] = child_method
            # Crossover parameters relevant to chosen method
            pspaces = _resolve_method_search_space(
                raw,
                method_scoped=method_scoped,
                method_name=str(child_method) if child_method else None,
            )
            child.update(_crossover_for_method(a, b, pspaces, rng) if rng.random() < crossover_rate else {})
            # Mutation for parameters of chosen method
            for k, spec in pspaces.items():
                if rng.random() < mutation_rate:
                    child[k] = _mutate_value(child.get(k), spec or {}, rng)
            new_pop.append(child)
        pop = new_pop[: len(pop)]

    elapsed_seconds = round(time.monotonic() - search_started_at, 3)
    progress = {
        "timed_out": timed_out,
        "partial_search": timed_out,
        "stop_reason": "timeout" if timed_out else "completed",
        "evaluations_completed": len(history),
        "evaluations_planned": evaluations_planned,
        "generations_completed": generations_completed,
        "elapsed_seconds": elapsed_seconds,
        "max_search_time_seconds": max_search_time_seconds,
    }

    if successful_evaluations == 0:
        return {
            "success": False,
            "error": (
                "Search timed out before any candidate produced a finite requested metric."
                if timed_out
                else "No candidate produced a finite requested metric."
            ),
            "error_code": (
                "search_timeout_no_results" if timed_out else "no_successful_trials"
            ),
            "metric": metric,
            "mode": mode,
            "population": population_size,
            "population_requested": int(population),
            "generations": int(generations),
            "history_count": len(history),
            "failure_causes": _failure_causes(history),
            "failed_candidates": history[:5],
            **progress,
        }

    payload: Dict[str, Any] = {
        "success": True,
        "best_score": float(best_score),
        "best_params": best_params,
        "metric": metric,
        "units": _tuning_units(metric, quantity),
        "mode": mode,
        "population": population_size,
        "population_requested": int(population),
        "generations": int(generations),
        "history_count": len(history),
        **progress,
    }
    if best_result is not None:
        sel = best_result.get('_sel_method') if isinstance(best_result, dict) else None
        if not sel:
            sel = best_params.get('method') or method
        agg = None
        try:
            agg = best_result.get('results', {}).get(sel) if isinstance(best_result, dict) else None
        except Exception:
            agg = None
        payload["best_method"] = sel
        if isinstance(agg, dict):
            payload["best_result_summary"] = {"horizon": int(horizon), "result": agg}
        _copy_best_backtest_provenance(payload, best_result)
    # Optional: compact history preview (keeps payload small)
    try:
        tail_n = 50
        if isinstance(history, list) and history:
            payload["history_tail"] = history[-tail_n:]
    except Exception:
        pass
    return payload


def genetic_search_optimize_hints(  # noqa: C901
    *,
    symbol: str,
    timeframes: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    horizon: int = 12,
    steps: int = 5,
    spacing: int = 20,
    as_of: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lookback: Optional[int] = None,
    search_space: Optional[Dict[str, Any]] = None,
    fitness_metric: str = "composite",
    fitness_weights: Optional[Dict[str, float]] = None,
    population: int = 20,
    generations: int = 15,
    crossover_rate: float = 0.6,
    mutation_rate: float = 0.3,
    seed: int = 42,
    max_search_time_seconds: Optional[float] = None,
    denoise: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    slippage_bps: float = 0.0,
    spread_bps: Optional[float] = None,
    commission_bps_per_side: Optional[float] = None,
    trade_threshold: float = 0.0,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Comprehensive genetic search for optimal forecast settings across timeframes, methods, and parameters.

    Searches across:
    - Timeframes (H1, H4, D1, W1, etc.)
    - Methods (fast classical baselines by default; heavyweight methods are opt-in)
    - Method-specific parameters

    Fitness: Composite score combining Sharpe ratio, win rate, inverse drawdown, and return,
    or single metric if fitness_metric != "composite".

    Args:
        symbol: Symbol to optimize for
        timeframes: Timeframes to search (default: ['H1', 'H4', 'D1', 'W1'])
        methods: Methods to search (default: fast classical baselines)
        horizon, steps, spacing: Backtest parameters
        search_space: Optional pre-built search space. If None, uses default from optimize module.
        fitness_metric: 'composite' or specific metric name ('avg_rmse', 'sharpe_ratio', etc.)
        fitness_weights: Custom weights for composite fitness (dict of metric: weight)
        population: Genetic population size
        generations: Number of generations
        crossover_rate, mutation_rate: Genetic parameters
        seed: Random seed
        max_search_time_seconds: Optional timeout for search
        denoise, features, dimred_method, dimred_params: Preprocessing options
        top_n: Number of top configurations to return

    Returns:
        Dict with 'success', 'hints' (list of top-N configs), 'search_summary', etc.
    """
    _suppress_noisy_forecast_tune_loggers()

    start_time = time.time()
    rng = random.Random(int(seed))
    maximize_metrics = {
        'sharpe_ratio',
        'win_rate',
        'calmar_ratio',
        'annual_return',
        'avg_directional_accuracy',
    }
    metric_mode = 'max' if fitness_metric in maximize_metrics else 'min'
    evaluations_attempted = 0
    timed_out = False

    from .optimize import (
        build_comprehensive_search_space as _build_search_space,
    )
    from .optimize import (
        composite_fitness_score as _composite_fitness,
    )
    from .optimize import (
        extract_method_params_from_genotype as _extract_params,
    )

    # Build search space if not provided
    if search_space is None:
        search_space = _build_search_space(
            timeframes=timeframes,
            methods=methods,
        )
    else:
        # Ensure comprehensive space has required keys
        if '_method_spaces' not in search_space:
            search_space['_method_spaces'] = default_search_space(methods=methods)

    # Extract genetic gene specs
    tf_choices = search_space.get('timeframe', {}).get(
        'choices', timeframes or ['H1', 'H4', 'D1', 'W1']
    )
    method_choices = search_space.get('method', {}).get('choices', methods or ['theta'])
    method_spaces = search_space.get('_method_spaces', {})

    # Helper: create random individual (genotype)
    def _create_individual() -> Dict[str, Any]:
        individual: Dict[str, Any] = {
            'timeframe': rng.choice(list(tf_choices)),
            'method': rng.choice(list(method_choices)),
        }
        # Sample parameters for chosen method
        meth = individual['method']
        params_space = method_spaces.get(str(meth), {})
        for param_name, param_spec in params_space.items():
            individual[param_name] = _sample_param(param_spec or {}, rng)
        return individual

    # Helper: crossover two individuals
    def _crossover(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        child: Dict[str, Any] = {
            'timeframe': rng.choice([a.get('timeframe'), b.get('timeframe')]),
            'method': rng.choice([a.get('method'), b.get('method')]),
        }
        meth = child['method']
        params_space = method_spaces.get(str(meth), {})
        for param_name in params_space.keys():
            av = a.get(param_name)
            bv = b.get(param_name)
            child[param_name] = av if rng.random() < 0.5 else bv
        return child

    # Helper: mutate an individual
    def _mutate(individual: Dict[str, Any]) -> Dict[str, Any]:
        mutant = dict(individual)
        # Mutate timeframe with small probability
        if rng.random() < mutation_rate * 0.3:
            mutant['timeframe'] = rng.choice(list(tf_choices))
        # Mutate method with small probability
        if rng.random() < mutation_rate * 0.3:
            mutant['method'] = rng.choice(list(method_choices))
        # Mutate parameters
        meth = mutant['method']
        params_space = method_spaces.get(str(meth), {})
        for param_name, param_spec in params_space.items():
            if rng.random() < mutation_rate:
                mutant[param_name] = _mutate_value(mutant.get(param_name), param_spec or {}, rng)
        return mutant

    def _evaluate(individual: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        nonlocal evaluations_attempted
        evaluations_attempted += 1
        tf = str(individual.get('timeframe', 'H1'))
        method = str(individual.get('method', 'theta'))
        params = {k: v for k, v in individual.items() if k not in ('timeframe', 'method')}

        # Run backtest with this config
        score, res = _eval_candidate(
            symbol=symbol,
            timeframe=tf,
            method=method,
            horizon=horizon,
            steps=steps,
            spacing=spacing,
            as_of=as_of,
            start=start,
            end=end,
            lookback=lookback,
            candidate_params={'method': method, **params},
            metric=fitness_metric if fitness_metric != 'composite' else 'avg_rmse',
            mode=metric_mode,
            denoise=denoise,
            features=features,
            dimred_method=dimred_method,
            dimred_params=dimred_params,
            slippage_bps=float(slippage_bps),
            spread_bps=spread_bps,
            commission_bps_per_side=commission_bps_per_side,
            trade_threshold=float(trade_threshold),
        )

        if not math.isfinite(float(score)):
            return math.inf, res

        # Compute fitness
        if fitness_metric == 'composite':
            # Extract backtest metrics
            try:
                backtest_metrics = _extract_method_backtest_metrics(res, method)
            except Exception:
                backtest_metrics = {}
            if _has_trading_fitness_metrics(backtest_metrics):
                fitness = _composite_fitness(backtest_metrics, weights=fitness_weights)
                fitness_source = "trading_composite"
                fitness_score = 1.0 - fitness
            else:
                forecast_accuracy = _forecast_accuracy_fitness(backtest_metrics)
                fitness_source = "forecast_accuracy_fallback"
                # Keep accuracy-only candidates below every trading composite.
                # Their score is useful for ordering fallbacks, but it is not a
                # comparable measure of trading edge.
                fitness_score = 3.0 - forecast_accuracy
            if isinstance(res, dict):
                res = dict(res)
                res['_optimization_fitness_source'] = fitness_source
                res['_optimization_trading_sample'] = _trading_sample_metadata(
                    backtest_metrics
                )
                if fitness_source == "forecast_accuracy_fallback":
                    res['_optimization_forecast_accuracy_score'] = forecast_accuracy
        else:
            # Single metric: use raw score from backtest
            fitness_score = float(score)

        return fitness_score, res

    # Initialize population
    population_size = int(population)
    if population_size < 2:
        raise ValueError("population must be greater than or equal to 2")
    pop: List[Tuple[Dict[str, Any], float, Dict[str, Any]]] = []

    for _ in range(population_size):
        ind = _create_individual()
        try:
            fitness, res = _evaluate(ind)
            pop.append((ind, fitness, res))
        except Exception as ex:
            # Failed evaluation; use worst score
            pop.append((ind, math.inf, {'error': str(ex)}))

        # Check timeout
        if (
            max_search_time_seconds
            and (time.time() - start_time) >= max_search_time_seconds
        ):
            timed_out = True
            break

    history: List[Dict[str, Any]] = []
    best_overall = min(pop, key=lambda t: t[1])[1]
    # Generational loop
    for gen in range(max(1, int(generations)) if not timed_out else 0):
        # Sort by fitness
        pop.sort(key=lambda t: t[1])

        # Elitism: keep top 2
        new_pop = [pop[0], pop[1] if len(pop) > 1 else pop[0]]

        # Breed new population
        while len(new_pop) < population_size:
            # Tournament selection
            a_ind, a_fit, _ = rng.choice(pop)
            b_ind, b_fit, _ = rng.choice(pop)

            # Crossover
            if rng.random() < crossover_rate:
                child = _crossover(a_ind, b_ind)
            else:
                child = dict(a_ind if a_fit < b_fit else b_ind)

            # Mutation
            if rng.random() < mutation_rate:
                child = _mutate(child)

            # Evaluate child
            try:
                fitness, res = _evaluate(child)
                new_pop.append((child, fitness, res))
            except Exception as ex:
                new_pop.append((child, math.inf, {'error': str(ex)}))

            # Check timeout
            if (
                max_search_time_seconds
                and (time.time() - start_time) >= max_search_time_seconds
            ):
                timed_out = True
                break

        pop = new_pop[: population_size]

        # Track best
        gen_best = min(pop, key=lambda t: t[1])
        if gen_best[1] < best_overall:
            best_overall = gen_best[1]

        gen_summary = {
            'generation': gen,
            'best_score': float(gen_best[1]),
            'avg_score': float(sum(p[1] for p in pop) / len(pop)) if pop else float('nan'),
        }
        if fitness_metric == 'composite':
            best_source = gen_best[2].get('_optimization_fitness_source')
            gen_summary['best_fitness_score'] = (
                0.0
                if best_source == 'forecast_accuracy_fallback'
                else 1.0 - float(gen_best[1])
            )
            display_scores = [
                0.0
                if item[2].get('_optimization_fitness_source')
                == 'forecast_accuracy_fallback'
                else 1.0 - float(item[1])
                for item in pop
                if math.isfinite(float(item[1]))
            ]
            gen_summary['avg_fitness_score'] = (
                float(sum(display_scores) / len(display_scores))
                if display_scores
                else float('nan')
            )
        history.append(gen_summary)
        if timed_out:
            break

    # Extract top-N candidates
    pop.sort(key=lambda t: t[1])
    finite_pop = [item for item in pop if math.isfinite(float(item[1]))]
    if not finite_pop:
        elapsed = time.time() - start_time
        return {
            'success': False,
            'error': (
                'Search timed out before any candidate produced a finite requested metric.'
                if timed_out
                else 'No candidate produced a finite requested metric.'
            ),
            'error_code': (
                'search_timeout_no_results' if timed_out else 'no_successful_trials'
            ),
            'hints': [],
            'partial': bool(timed_out),
            'stop_reason': 'timeout' if timed_out else 'completed',
            'evaluations_completed': evaluations_attempted,
            'search_summary': {
                'symbol': symbol,
                'population': population_size,
                'population_requested': int(population),
                'generations': int(generations),
                'generations_completed': len(history),
                'elapsed_seconds': round(elapsed, 2),
                'fitness_metric': fitness_metric,
                'total_evaluations': evaluations_attempted,
            },
        }
    top_configs: List[Dict[str, Any]] = []

    seen_configs = set()
    duplicate_results_filtered = 0
    for individual, fitness, backtest_res in finite_pop:
        if len(top_configs) >= int(top_n):
            break
        tf, method, params = _extract_params(individual, search_space)
        config_key = (
            str(tf),
            str(method),
            tuple(sorted((str(key), repr(value)) for key, value in dict(params).items())),
        )
        if config_key in seen_configs:
            duplicate_results_filtered += 1
            continue
        seen_configs.add(config_key)

        # Build hint entry
        fitness_source = (
            backtest_res.get('_optimization_fitness_source')
            if isinstance(backtest_res, dict)
            else None
        )
        fallback_accuracy = (
            backtest_res.get('_optimization_forecast_accuracy_score')
            if isinstance(backtest_res, dict)
            else None
        )
        trading_sample = (
            backtest_res.get('_optimization_trading_sample')
            if isinstance(backtest_res, dict)
            else None
        )
        hint: Dict[str, Any] = {
            'rank': len(top_configs) + 1,
            'timeframe': tf,
            'method': method,
            'method_params': params,
            'fitness_score': (
                0.0
                if fitness_metric == 'composite'
                and fitness_source == 'forecast_accuracy_fallback'
                else 1.0 - fitness
                if fitness_metric == 'composite'
                else (-fitness if metric_mode == 'max' else fitness)
            ),
            'fitness_score_unit': _optimization_fitness_unit(fitness_metric),
        }
        if fitness_source:
            hint['fitness_source'] = fitness_source
        if isinstance(trading_sample, dict):
            for key, value in trading_sample.items():
                if value is not None:
                    hint[key] = value
        if fitness_source == 'forecast_accuracy_fallback':
            hint['fitness_comparable'] = False
            hint['ranking_tier'] = 'forecast_accuracy_fallback'
            hint['forecast_accuracy_score'] = fallback_accuracy
            hint['forecast_accuracy_basis'] = (
                'mean_of_directional_accuracy_and_inverse_one_plus_price_error'
            )
        elif fitness_metric == 'composite':
            hint['fitness_comparable'] = True
            hint['ranking_tier'] = 'trading_composite'

        # Extract backtest metrics
        try:
            method_metrics = _extract_method_backtest_metrics(backtest_res, method)
            if method_metrics:
                hint['backtest_metrics'] = {
                    'avg_rmse': method_metrics.get('avg_rmse'),
                    'avg_mae': method_metrics.get('avg_mae'),
                    'avg_directional_accuracy': method_metrics.get('avg_directional_accuracy'),
                    'sharpe_ratio': method_metrics.get('sharpe_ratio'),
                    'win_rate': method_metrics.get('win_rate'),
                    'max_drawdown': method_metrics.get('max_drawdown'),
                    'avg_return_per_trade': method_metrics.get('avg_return_per_trade'),
                    'calmar_ratio': method_metrics.get('calmar_ratio'),
                    'annual_return': method_metrics.get('annual_return'),
                    'metrics_available': method_metrics.get('metrics_available'),
                    'metrics_reason': method_metrics.get('metrics_reason'),
                    'trade_status': method_metrics.get('trade_status'),
                    'trades_observed': (
                        trading_sample.get('trades_observed')
                        if isinstance(trading_sample, dict)
                        else method_metrics.get('trades_observed')
                    ),
                    'metrics_reliability': (
                        trading_sample.get('metrics_reliability')
                        if isinstance(trading_sample, dict)
                        else method_metrics.get('metrics_reliability')
                    ),
                    'metrics_reliability_reason': (
                        trading_sample.get('metrics_reliability_reason')
                        if isinstance(trading_sample, dict)
                        else method_metrics.get('metrics_reliability_reason')
                    ),
                    'sample_notice': (
                        trading_sample.get('sample_notice')
                        if isinstance(trading_sample, dict)
                        else method_metrics.get('sample_notice')
                    ),
                    'sample_warning': (
                        trading_sample.get('sample_warning')
                        if isinstance(trading_sample, dict)
                        else method_metrics.get('sample_warning')
                    ),
                }
        except Exception:
            pass

        top_configs.append(hint)

    elapsed = time.time() - start_time
    result: Dict[str, Any] = {
        'success': True,
        'hints': top_configs,
        'partial': bool(timed_out),
        'stop_reason': 'timeout' if timed_out else 'completed',
        'evaluations_completed': evaluations_attempted,
        'search_summary': {
            'symbol': symbol,
            'population': population_size,
            'population_requested': int(population),
            'generations': int(generations),
            'generations_completed': len(history),
            'elapsed_seconds': round(elapsed, 2),
            'fitness_metric': fitness_metric,
            'fitness_score_unit': _optimization_fitness_unit(fitness_metric),
            'fitness_score_direction': (
                'higher_is_better'
                if fitness_metric == 'composite' or metric_mode == 'max'
                else 'lower_is_better'
            ),
            'history_score_direction': 'lower_is_better_internal_objective',
            'timeframes_searched': list(tf_choices),
            'methods_searched': list(method_choices),
            'total_evaluations': evaluations_attempted,
            'unique_configs_returned': len(top_configs),
            'duplicate_results_filtered': int(duplicate_results_filtered),
        },
        'history_tail': history[-10:] if history else [],
    }
    _copy_best_backtest_provenance(result, finite_pop[0][2])

    finite_scores = [
        float(item['fitness_score'])
        for item in top_configs
        if item.get('fitness_comparable') is not False
        if math.isfinite(float(item.get('fitness_score', float('nan'))))
    ]
    if len(finite_scores) > 1 and max(finite_scores) - min(finite_scores) <= 1e-12:
        result['search_summary']['fitness_tie'] = True
        result['warning'] = (
            "Top configurations have identical fitness; treat their ordering as tied."
        )

    return result
