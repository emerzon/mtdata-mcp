"""Per-method regime detection bodies extracted from the MCP entrypoint."""

import math
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...utils.regime_heuristics import infer_market_regime
from ...utils.time import _format_time_minimal
from .. import features as _features_module
from ..features import extract_rolling_features
from .ensemble import (
    _ENSEMBLE_STATE_METHODS,
    _aggregate_precomputed_ensemble,
    _ensemble_state_count_configuration,
    _finite_raw_kurtosis,
)
from .methods.bocpd import (
    _auto_calibrate_bocpd_params,
    _bocpd_reliability_score,
    _default_bocpd_cp_threshold,
    _default_bocpd_hazard_lambda,
    _filter_bocpd_change_points,
    _walkforward_quantile_threshold_calibration,
)
from .methods.hmm import _hmm_reliability_from_gamma
from .methods.ms_ar import _ms_ar_reliability_from_smoothed
from .payload import (
    _consolidate_payload,
    _summary_only_payload,
)
from .smoothing import (
    _canonicalize_regime_labels,
    _confirm_state_changes_causally,
    _hard_state_probability_matrix,
    _normalize_state_probability_matrix,
)
from .summarize import (
    _append_warnings,
    _apply_bocpd_output_mode,
    _apply_state_output_mode,
    _build_all_method_comparison,
    _common_reliability,
    _mark_collapsed_state_confidence,
    _smoothing_warnings,
    _summary_window_size,
)


def _rolling_prefix_std(values: np.ndarray, lookback: int) -> np.ndarray:
    """Return the legacy inclusive rolling standard deviation without Python loops."""
    array = np.asarray(values, dtype=float)
    ends = np.arange(1, array.size + 1, dtype=int)
    starts = np.maximum(0, ends - (int(lookback) + 1))
    counts = ends - starts
    cumulative = np.concatenate(([0.0], np.cumsum(array)))
    cumulative_squares = np.concatenate(([0.0], np.cumsum(array**2)))
    means = (cumulative[ends] - cumulative[starts]) / counts
    variances = (
        (cumulative_squares[ends] - cumulative_squares[starts]) / counts
        - means**2
    )
    return np.sqrt(np.maximum(variances, 0.0))


def _rolling_band_energy(bands: List[np.ndarray], window: int) -> np.ndarray:
    """Compute leading-partial rolling mean-square energy for each band."""
    if not bands:
        return np.empty((0, 0), dtype=float)
    n_bars = len(bands[0])
    energy_matrix = np.zeros((n_bars, len(bands)), dtype=float)
    window_ends = np.arange(1, n_bars + 1, dtype=int)
    window_starts = np.maximum(0, window_ends - int(window))
    window_lengths = window_ends - window_starts
    for band_index, band in enumerate(bands):
        squared = np.asarray(band, dtype=float) ** 2
        cumulative = np.concatenate(([0.0], np.cumsum(squared)))
        energy_matrix[:, band_index] = (
            cumulative[window_ends] - cumulative[window_starts]
        ) / window_lengths
    return energy_matrix


_PELT_DIRECTION_T_STAT_THRESHOLD = 1.96


def _pelt_return_direction(
    segment: np.ndarray,
    mean_value: float,
) -> tuple[str, Optional[float], bool]:
    values = np.asarray(segment, dtype=float)
    if values.size < 2:
        return "neutral", None, False
    sample_std = float(np.std(values, ddof=1))
    if not np.isfinite(sample_std) or sample_std <= 1e-12:
        significant = bool(abs(float(mean_value)) > 1e-12)
        direction = (
            "positive" if mean_value > 0 else "negative" if mean_value < 0 else "neutral"
        )
        return (direction if significant else "neutral"), None, significant
    mean_t_stat = float(mean_value) / (sample_std / np.sqrt(float(values.size)))
    significant = bool(abs(mean_t_stat) >= _PELT_DIRECTION_T_STAT_THRESHOLD)
    if not significant:
        return "neutral", mean_t_stat, False
    return ("positive" if mean_value > 0 else "negative"), mean_t_stat, True


def _coerce_param(
    params: Dict[str, Any],
    key: str,
    *,
    default: Any,
    cast: Any,
    error: Optional[str] = None,
) -> tuple[Any, Optional[str]]:
    raw = params.get(key, default)
    if raw is None:
        return default, None
    try:
        return cast(raw), None
    except Exception:
        if error is not None:
            return None, error
        return default, None


def _garch_tier_thresholds(
    conditional_volatility: np.ndarray,
    n_states: int,
    explicit_threshold: Optional[float],
) -> Tuple[List[float], str]:
    """Return cut points for GARCH conditional-volatility tiers."""
    if explicit_threshold is not None and int(n_states) == 2:
        return [float(explicit_threshold)], "explicit_absolute"
    percentiles = np.linspace(0, 100, int(n_states) + 1)[1:-1]
    thresholds = [
        float(np.percentile(conditional_volatility, percentile))
        for percentile in percentiles
    ]
    return thresholds, "full_window_percentiles"


def _feature_cluster_separation(
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Return the share of feature-space variance explained by cluster labels."""
    feature_array = np.asarray(features, dtype=float)
    label_array = np.asarray(labels, dtype=int).reshape(-1)
    if (
        feature_array.ndim != 2
        or feature_array.shape[0] != label_array.size
        or feature_array.shape[0] < 2
    ):
        return 0.0

    finite_rows = np.isfinite(feature_array).all(axis=1)
    feature_array = feature_array[finite_rows]
    label_array = label_array[finite_rows]
    if feature_array.shape[0] < 2 or np.unique(label_array).size < 2:
        return 0.0

    overall_center = np.mean(feature_array, axis=0)
    total_ss = float(np.sum((feature_array - overall_center) ** 2))
    if total_ss <= np.finfo(float).eps:
        return 0.0

    within_ss = 0.0
    for label in np.unique(label_array):
        cluster = feature_array[label_array == label]
        cluster_center = np.mean(cluster, axis=0)
        within_ss += float(np.sum((cluster - cluster_center) ** 2))
    return float(np.clip(1.0 - (within_ss / total_ss), 0.0, 1.0))


def _wavelet_detail_bands(
    series: np.ndarray,
    wavelet_name: str,
    level: int,
    *,
    boundary_mode: str = "symmetric",
    pywt_module: Any = None,
) -> List[np.ndarray]:
    """Reconstruct DWT detail bands without circular window coupling."""
    if pywt_module is None:
        import pywt as pywt_module

    values = np.asarray(series, dtype=float).reshape(-1)
    coeffs = pywt_module.wavedec(
        values,
        wavelet_name,
        mode=boundary_mode,
        level=level,
    )
    bands: List[np.ndarray] = []
    for index in range(1, len(coeffs)):
        isolated = [np.zeros_like(coefficient) for coefficient in coeffs]
        isolated[index] = coeffs[index]
        band = pywt_module.waverec(
            isolated,
            wavelet_name,
            mode=boundary_mode,
        )
        bands.append(np.asarray(band[: values.size], dtype=float))
    return bands


# Bars required before BOCPD under-segmentation checks kick in. Below this
# window length a single-segment result is unremarkable and not worth warning
# about; above it, the absence of any change point becomes suspicious.
_BOCPD_UNDERSEG_MIN_BARS = 100
# Single-bar absolute return (in std units) above which we flag a possible
# missed change point even when BOCPD posterior never crossed cp_threshold.
_BOCPD_UNDERSEG_PEAK_Z = 3.5


def _peak_abs_return(series: np.ndarray) -> float:
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    std = float(np.std(arr))
    if not np.isfinite(std) or std <= 1e-12:
        return 0.0
    return float(np.max(np.abs(arr)) / std)


def _bocpd_under_segmentation_warnings(
    *,
    total_bars: int,
    change_point_count: int,
    raw_change_point_count: Optional[int] = None,
    reliability: Any,
    peak_abs_return: float,
) -> List[str]:
    """Surface likely BOCPD under-segmentation.

    BOCPD's Gaussian conjugate priors can absorb a violent but short-lived
    move as a fat-tail outlier inside a single regime, so a flash crash that
    PELT, MS-AR, and clustering all flag is silently missed. Surface a
    warning when the model returns zero change points over a long window,
    especially when reliability confidence is low or when the window contains
    a multi-sigma single-bar move that other methods would have caught.
    """
    if total_bars < _BOCPD_UNDERSEG_MIN_BARS:
        return []
    if change_point_count > 0:
        return []

    warnings: List[str] = []
    if int(raw_change_point_count or 0) > 0:
        warnings.append(
            "BOCPD candidates crossed the probability threshold, but robustness "
            "filters rejected all of them; the reported stable segment is uncertain. "
            "Review confirmation, cooldown, and edge-filter settings."
        )
    confidence_value: Optional[float] = None
    if isinstance(reliability, dict):
        raw_conf = reliability.get("confidence")
        try:
            if raw_conf is not None:
                confidence_value = float(raw_conf)
        except (TypeError, ValueError):
            confidence_value = None
    if confidence_value is not None and confidence_value < 0.5:
        warnings.append(
            "BOCPD reported no change points over a long window with low "
            "reliability confidence; possible under-segmentation. Compare "
            "with PELT or ms_ar for cross-validation."
        )
    if peak_abs_return >= _BOCPD_UNDERSEG_PEAK_Z:
        warnings.append(
            f"Window contains a {peak_abs_return:.1f}σ single-bar move but "
            "BOCPD reported no change points; the move may have been absorbed "
            "as a fat-tail outlier. Try a lower cp_threshold or hazard_lambda."
        )
    return warnings


def _resolve_state_count_param(
    params: Dict[str, Any],
    *,
    default: int,
    method: str,
    canonical: str = "n_states",
) -> tuple[Optional[int], Optional[str], str, List[str]]:
    raw_value = params.get(canonical)
    if raw_value is None:
        raw_value = default
    try:
        value = int(raw_value)
    except Exception:
        return None, f"{canonical} must be an integer >= 2 for {method}.", canonical, []
    return value, None, canonical, []


def _observed_state_mean_vol(
    states: np.ndarray,
    values: np.ndarray,
) -> tuple[List[float], List[float]]:
    means: List[float] = []
    vols: List[float] = []
    for state in sorted({int(s) for s in np.unique(states) if int(s) >= 0}):
        mask = states == state
        means.append(float(np.mean(values[mask])))
        vols.append(float(np.std(values[mask])))
    return means, vols


def _method_parameter_warnings(
    method: str,
    params: Dict[str, Any],
    *,
    threshold: Optional[float],
    requested_lookback: int,
    requested_min_regime_bars: int,
    include_series: bool,
    max_regimes: int,
    output: str,
    lookback_mapped_to_window: bool = False,
) -> List[str]:
    warnings_out: List[str] = []
    if method != "bocpd" and (threshold is not None or "threshold" in params):
        warnings_out.append(
            "threshold only applies to BOCPD change-point detection and is ignored "
            f"for method='{method}'."
        )
    if method == "rule_based":
        if (
            requested_lookback >= 0
            and "window_bars" in params
            and not bool(lookback_mapped_to_window)
        ):
            warnings_out.append(
                "lookback was ignored for rule_based because params.window_bars was "
                "also provided; remove params.window_bars to use lookback as the "
                "rule-based analysis window."
            )
        elif "lookback" in params:
            warnings_out.append(
                "params.lookback is ignored for rule_based; use the top-level "
                "lookback argument or params.window_bars."
            )
        if requested_min_regime_bars >= 0 or "min_regime_bars" in params:
            warnings_out.append(
                "min_regime_bars is not used by rule_based because it does not "
                "estimate regime boundaries or persistence."
            )
        if max_regimes != 10:
            warnings_out.append(
                "max_regimes has no effect for rule_based because it does not "
                "produce historical regime segments."
            )
        if include_series:
            warnings_out.append(
                "include_series is not available for rule_based because it classifies "
                "one aggregate window rather than estimating a per-bar state sequence."
            )
    return warnings_out


def _resolve_bocpd_priors(
    params: Dict[str, Any],
    series: np.ndarray,
) -> Dict[str, float]:
    """Extract BOCPD prior hyper-parameters from *params* dict.

    If a prior param (mu0, kappa0, alpha0, beta0) is explicitly provided,
    use it.  Otherwise fall back to data-driven defaults derived from the
    series statistics, which are more appropriate than the hard-coded
    ``bocpd_gaussian`` defaults (mu0=0, kappa0=1, alpha0=1, beta0=1)
    when the data mean / variance is far from those assumptions.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]

    # Data-driven defaults
    if x.size >= 10:
        mu_data = float(np.mean(x))
        var_data = float(np.var(x, ddof=0))
        var_safe = max(var_data, 1e-16)
        dd_mu0 = mu_data
        dd_kappa0 = 1.0
        dd_alpha0 = max(1.0, x.size / 20.0)
        dd_beta0 = max(1e-8, var_safe * dd_alpha0)
    else:
        dd_mu0, dd_kappa0, dd_alpha0, dd_beta0 = 0.0, 1.0, 1.0, 1.0

    mode = str(params.get("prior_mode", "data_driven") or "data_driven").strip().lower()
    if mode == "fixed":
        dd_mu0, dd_kappa0, dd_alpha0, dd_beta0 = 0.0, 1.0, 1.0, 1.0

    mu0, _ = _coerce_param(params, "mu0", default=dd_mu0, cast=float)
    kappa0, _ = _coerce_param(params, "kappa0", default=dd_kappa0, cast=float)
    alpha0, _ = _coerce_param(params, "alpha0", default=dd_alpha0, cast=float)
    beta0, _ = _coerce_param(params, "beta0", default=dd_beta0, cast=float)

    return {
        "mu0": float(mu0),
        "kappa0": max(1e-8, float(kappa0)),
        "alpha0": max(0.5, float(alpha0)),
        "beta0": max(1e-12, float(beta0)),
    }

_RULE_BASED_RECOMMENDED_WINDOW_BARS = 160


_REGIME_METHOD_RUNTIME_GUIDANCE: Dict[str, Dict[str, str]] = {
    "rule_based": {
        "speed_tier": "fast",
        "use_case": "quick trend/ranging/transition snapshot",
        "cost_notes": "deterministic indicator-style calculations",
    },
    "bocpd": {
        "speed_tier": "medium",
        "use_case": "change-point and transition timing",
        "cost_notes": "calibration and run-length filtering cost grows with history",
    },
    "pelt": {
        "speed_tier": "fast",
        "use_case": "offline structural-break segmentation",
        "cost_notes": "ruptures PELT with pruning; cost depends on model and history length",
    },
    "hmm": {
        "speed_tier": "medium",
        "use_case": "probabilistic return/price state segmentation",
        "cost_notes": "Gaussian HMM fit with forward filtering",
    },
    "gmm": {
        "speed_tier": "medium",
        "use_case": "independent Gaussian mixture segmentation",
        "cost_notes": "i.i.d. mixture fit without Markov transitions",
    },
    "clustering": {
        "speed_tier": "medium",
        "use_case": "rolling feature cluster regimes",
        "cost_notes": "feature extraction and clustering can be slow on large windows",
    },
    "garch": {
        "speed_tier": "medium",
        "use_case": "GARCH conditional-volatility tier classification",
        "cost_notes": "optimization fit; convergence varies by symbol/history",
    },
    "wavelet": {
        "speed_tier": "medium",
        "use_case": "multi-resolution energy regimes",
        "cost_notes": "depends on PyWavelets and decomposition level",
    },
    "ms_ar": {
        "speed_tier": "slow",
        "use_case": "Markov-switching autoregressive regimes",
        "cost_notes": "statsmodels maximum-likelihood fit can be long-running",
    },
    "ensemble": {
        "speed_tier": "slow",
        "use_case": "consensus across selected regime methods",
        "cost_notes": "runs multiple sub-methods",
    },
    "all": {
        "speed_tier": "slow",
        "use_case": "cross-method comparison and diagnostics",
        "cost_notes": "runs every method plus ensemble; use faster methods for low-latency checks",
    },
}


def _regime_runtime_guidance(methods: List[str]) -> Dict[str, Dict[str, str]]:
    return {
        method: dict(_REGIME_METHOD_RUNTIME_GUIDANCE[method])
        for method in methods
        if method in _REGIME_METHOD_RUNTIME_GUIDANCE
    }


def _suggest_faster_regime_methods(methods: List[str]) -> List[str]:
    suggestions: List[str] = []
    for candidate in ("rule_based", "bocpd", "hmm"):
        if candidate not in methods and candidate not in suggestions:
            suggestions.append(candidate)
    for method in methods:
        guidance = _REGIME_METHOD_RUNTIME_GUIDANCE.get(method, {})
        if guidance.get("speed_tier") == "slow":
            continue
        if method not in suggestions:
            suggestions.append(method)
    return suggestions[:3]


def _detect_bocpd(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    target: str,
    method: str,
    x: np.ndarray,
    t_fmt: List[Any],
    p: Dict[str, Any],
    lookback: int,
    output: str,
    include_series: bool,
    max_regimes: int,
    min_regime_bars_val: int,
    threshold: Optional[float],
    min_regime_bars: Optional[int],
    calibration_returns: np.ndarray,
) -> Dict[str, Any]:
    from ...utils.bocpd import bocpd_gaussian

    hazard_mode = (
        str(p.get("hazard_mode", "auto_calibrated") or "auto_calibrated")
        .strip()
        .lower()
    )
    if hazard_mode in {"auto", "calibrated"}:
        hazard_mode = "auto_calibrated"
    if hazard_mode not in {"auto_default", "auto_calibrated"}:
        hazard_mode = "auto_calibrated"

    hazard_src = "params"
    threshold_src = "arg"
    calibration_info: Optional[Dict[str, Any]] = None
    threshold_calibration_info: Optional[Dict[str, Any]] = None

    auto_hazard = _default_bocpd_hazard_lambda(symbol, timeframe)
    auto_threshold = _default_bocpd_cp_threshold(symbol, timeframe)
    if hazard_mode == "auto_calibrated":
        auto_hazard, auto_threshold, calibration_info = (
            _auto_calibrate_bocpd_params(
                returns=calibration_returns, symbol=symbol, timeframe=timeframe
            )
        )

    if "hazard_lambda" in p and p.get("hazard_lambda") is not None:
        raw_hazard = p.get("hazard_lambda")
        try:
            hazard_f = float(raw_hazard)
        except (TypeError, ValueError):
            return {
                    "error": (
                        f"params.hazard_lambda must be a positive integer "
                        f"(got {raw_hazard!r})."
                    )
                }
        if not math.isfinite(hazard_f) or hazard_f != int(hazard_f) or int(hazard_f) < 1:
            return {
                    "error": (
                        "params.hazard_lambda must be a positive integer "
                        f"(got {raw_hazard!r})."
                    )
                }
        hazard_lambda = int(hazard_f)
    else:
        hazard_lambda = int(auto_hazard)
        hazard_src = (
            "auto_calibrated"
            if hazard_mode == "auto_calibrated"
            else "auto_default"
        )
    if "cp_threshold" in p and p.get("cp_threshold") is not None:
        threshold_used = float(p.get("cp_threshold"))
        threshold_src = "params.cp_threshold"
    elif "threshold" in p and p.get("threshold") is not None:
        threshold_used = float(p.get("threshold"))
        threshold_src = "params.threshold"
    elif threshold is None:
        threshold_used = float(auto_threshold)
        threshold_src = (
            "auto_calibrated"
            if hazard_mode == "auto_calibrated"
            else "auto_default"
        )
    else:
        threshold_used = float(threshold)
        threshold_src = "arg"
    max_rl, _ = _coerce_param(
        p,
        "max_run_length",
        default=min(1000, x.size),
        cast=int,
    )
    threshold_cal_mode = (
        str(
            p.get("cp_threshold_calibration_mode", "walkforward_quantile")
            or "walkforward_quantile"
        )
        .strip()
        .lower()
    )
    if threshold_cal_mode in {"auto", "walkforward", "quantile"}:
        threshold_cal_mode = "walkforward_quantile"
    if (
        threshold_src in {"auto_calibrated", "auto_default"}
        and threshold_cal_mode == "walkforward_quantile"
    ):
        target_fa, _ = _coerce_param(
            p,
            "threshold_target_false_alarm_rate",
            default=0.02,
            cast=float,
        )
        cal_window, _ = _coerce_param(
            p,
            "threshold_calibration_window",
            default=None,
            cast=int,
        )
        cal_step, _ = _coerce_param(
            p,
            "threshold_calibration_step",
            default=None,
            cast=int,
        )
        cal_max_windows, _ = _coerce_param(
            p,
            "threshold_calibration_max_windows",
            default=6,
            cast=int,
        )
        cal_boot, _ = _coerce_param(
            p,
            "threshold_calibration_bootstraps",
            default=2,
            cast=int,
        )
        threshold_used, threshold_calibration_info = (
            _walkforward_quantile_threshold_calibration(
                series=x,
                hazard_lambda=hazard_lambda,
                base_threshold=threshold_used,
                target_false_alarm_rate=target_fa,
                window=cal_window,
                step=cal_step,
                max_windows=cal_max_windows,
                bootstrap_runs=cal_boot,
                max_run_length=max_rl,
            )
        )
    bocpd_priors = _resolve_bocpd_priors(p, x)
    res = bocpd_gaussian(
        x,
        hazard_lambda=hazard_lambda,
        max_run_length=max_rl,
        mu0=bocpd_priors["mu0"],
        kappa0=bocpd_priors["kappa0"],
        alpha0=bocpd_priors["alpha0"],
        beta0=bocpd_priors["beta0"],
    )
    cp_prob = np.asarray(
        res.get("cp_prob", np.zeros_like(x, dtype=float)), dtype=float
    )
    raw_cp_idx = [
        int(i)
        for i, v in enumerate(cp_prob.tolist())
        if np.isfinite(v) and float(v) >= float(threshold_used)
    ]
    cp_confirm_bars, _ = _coerce_param(
        p,
        "cp_confirm_bars",
        default=1,
        cast=int,
    )
    cp_confirm_relaxed_mult, _ = _coerce_param(
        p,
        "cp_confirm_relaxed_mult",
        default=0.90,
        cast=float,
    )
    if "cp_edge_multiplier" in p and p.get("cp_edge_multiplier") is not None:
        cp_edge_multiplier, _ = _coerce_param(
            p,
            "cp_edge_multiplier",
            default=1.08,
            cast=float,
        )
    else:
        # When threshold is already calibrated via walk-forward null quantiles,
        # avoid double-tightening the edge gate.
        if (
            threshold_src in {"auto_calibrated", "auto_default"}
            and isinstance(threshold_calibration_info, dict)
            and bool(threshold_calibration_info.get("calibrated", False))
        ):
            cp_edge_multiplier = 1.0
        else:
            cp_edge_multiplier = 1.08
    min_cp_distance_bars, _ = _coerce_param(
        p,
        "min_cp_distance_bars",
        default=max(2, min_regime_bars_val),
        cast=int,
    )
    cp_idx, cp_filter_meta = _filter_bocpd_change_points(
        cp_prob=cp_prob,
        threshold=float(threshold_used),
        min_distance_bars=int(max(1, min_cp_distance_bars)),
        min_regime_bars=int(max(1, min_regime_bars_val)),
        confirm_bars=int(max(1, cp_confirm_bars)),
        confirm_relaxed_mult=float(cp_confirm_relaxed_mult),
        edge_multiplier=float(cp_edge_multiplier),
    )
    cps = [
        {"idx": i, "time": t_fmt[i], "prob": float(cp_prob[i])} for i in cp_idx
    ]
    tuning_hint: Optional[str] = None
    if len(cps) == 0:
        if (
            len(raw_cp_idx) > 0
            and int(cp_filter_meta.get("filtered_count", 0)) > 0
        ):
            tuning_hint = (
                "Change-point candidates were filtered by robustness guards "
                "(confirmation/cooldown/edge checks). Tune cp_confirm_bars, "
                "min_cp_distance_bars, or cp_edge_multiplier if needed."
            )
        else:
            tuning_hint = (
                "No change points detected. Try lowering threshold or reducing "
                f"hazard_lambda (currently {hazard_lambda}); active threshold={threshold_used:.2f}."
            )
    if isinstance(threshold_calibration_info, dict):
        expected_fa_rate = float(
            threshold_calibration_info.get("target_false_alarm_rate", 0.02)
        )
        calibration_age_bars = int(
            threshold_calibration_info.get(
                "points",
                calibration_info.get("points", 0)
                if isinstance(calibration_info, dict)
                else 0,
            )
        )
        threshold_calibrated = bool(
            threshold_calibration_info.get("calibrated", False)
        )
    else:
        expected_fa_rate = 0.02
        calibration_age_bars = int(
            calibration_info.get("points", 0)
            if isinstance(calibration_info, dict)
            else 0
        )
        threshold_calibrated = False
    reliability = _bocpd_reliability_score(
        cp_prob=cp_prob,
        cp_indices=cp_idx,
        threshold=float(threshold_used),
        lookback=int(lookback),
        min_regime_bars=int(max(1, min_regime_bars_val)),
        expected_false_alarm_rate=float(expected_fa_rate),
        calibration_age_bars=int(calibration_age_bars),
        threshold_calibrated=bool(threshold_calibrated),
    )
    reliability = _common_reliability(
        reliability,
        source="bocpd_calibration",
    )
    payload = {
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "method": method,
        "target": target,
        "times": t_fmt,
        "cp_prob": [
            float(v) for v in np.asarray(cp_prob, dtype=float).tolist()
        ],
        "change_points": cps,
        "_series_values": [
            float(v) for v in np.asarray(x, dtype=float).tolist()
        ],
        "threshold": float(threshold_used),
        "reliability": reliability,
        "params_used": {
            "hazard_lambda": hazard_lambda,
            "hazard_lambda_source": hazard_src,
            "cp_threshold": float(threshold_used),
            "cp_threshold_source": threshold_src,
            "hazard_mode": hazard_mode,
            "max_run_length": max_rl,
            "cp_filter": cp_filter_meta,
            "priors": bocpd_priors,
        },
    }
    if isinstance(calibration_info, dict):
        payload["params_used"]["auto_calibration"] = calibration_info
    if isinstance(threshold_calibration_info, dict):
        payload["params_used"]["cp_threshold_calibration"] = (
            threshold_calibration_info
        )
    if tuning_hint is not None:
        payload["tuning_hint"] = tuning_hint
    _append_warnings(
        payload,
        _bocpd_under_segmentation_warnings(
            total_bars=int(x.size),
            change_point_count=len(cp_idx),
            raw_change_point_count=len(raw_cp_idx),
            reliability=payload.get("reliability"),
            peak_abs_return=_peak_abs_return(x),
        ),
    )
    if output in ("summary", "compact"):
        payload = _apply_bocpd_output_mode(
            payload,
            output=output,
            lookback=lookback,
            cp_prob=cp_prob,
            change_points=cps,
            raw_cp_idx=raw_cp_idx,
            reliability=reliability,
            expected_fa_rate=expected_fa_rate,
            calibration_age_bars=calibration_age_bars,
            tuning_hint=tuning_hint,
        )
        if output == "summary":
            return payload

    return _consolidate_payload(
            payload,
            method,
            output,
            include_series=include_series,
            max_regimes=max_regimes,
        )


def _detect_pelt(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    target: str,
    x: np.ndarray,
    t_fmt: List[Any],
    p: Dict[str, Any],
    output: str,
    include_series: bool,
    max_regimes: int,
    min_regime_bars_val: int,
) -> Dict[str, Any]:
    try:
        import ruptures as rpt
    except ImportError:
        return {
                "error": "ruptures is required for PELT regime detection.",
                "error_code": "dependency_missing",
                "details": {"method": "pelt", "requires": ["ruptures"]},
            }

    model = str(p.get("model", "l2") or "l2").strip().lower()
    if model not in {"l1", "l2", "rbf", "normal", "ar"}:
        return {"error": "params.model must be one of: l1, l2, rbf, normal, ar."}
    min_size, min_size_error = _coerce_param(
        p,
        "min_size",
        default=max(2, int(min_regime_bars_val)),
        cast=int,
        error="params.min_size must be an integer >= 2.",
    )
    if min_size_error is not None or int(min_size) < 2:
        return {"error": min_size_error or "params.min_size must be >= 2."}
    jump, jump_error = _coerce_param(
        p,
        "jump",
        default=1,
        cast=int,
        error="params.jump must be an integer >= 1.",
    )
    if jump_error is not None or int(jump) < 1:
        return {"error": jump_error or "params.jump must be >= 1."}
    penalty_raw = p.get("penalty")
    if penalty_raw is None or str(penalty_raw).strip().lower() == "auto":
        variance = float(np.var(x, ddof=0))
        penalty = max(1e-12, 3.0 * np.log(max(int(x.size), 2)) * variance)
        penalty_source = "bic_like_auto"
    else:
        try:
            penalty = float(penalty_raw)
        except Exception:
            return {"error": "params.penalty must be a positive number."}
        penalty_source = "params"
    if not np.isfinite(penalty) or penalty <= 0.0:
        return {"error": "params.penalty must be a positive finite number."}

    signal = np.asarray(x, dtype=float).reshape(-1, 1)
    try:
        breakpoints = rpt.Pelt(
            model=model,
            min_size=int(min_size),
            jump=int(jump),
        ).fit(signal).predict(pen=float(penalty))
    except Exception as exc:
        return {"error": f"PELT regime detection failed: {exc}"}

    segment_ends = [int(value) for value in breakpoints if int(value) > 0]
    if not segment_ends or segment_ends[-1] != int(x.size):
        segment_ends.append(int(x.size))
    regimes: List[Dict[str, Any]] = []
    change_points: List[Dict[str, Any]] = []
    start_idx = 0
    global_volatility = max(float(np.std(x, ddof=0)), 1e-12)
    for regime_id, end_idx in enumerate(segment_ends):
        if end_idx <= start_idx:
            continue
        segment = np.asarray(x[start_idx:end_idx], dtype=float)
        mean_value = float(np.mean(segment))
        volatility = float(np.std(segment, ddof=0))
        if target == "return":
            direction, mean_t_stat, direction_significant = (
                _pelt_return_direction(segment, mean_value)
            )
            vol_label = "high_vol" if volatility > global_volatility else "low_vol"
            label = f"{direction}_{vol_label}"
        else:
            direction = "rising" if segment[-1] > segment[0] else "falling" if segment[-1] < segment[0] else "flat"
            label = direction
        row = {
            "regime": int(regime_id),
            "label": label,
            "start": t_fmt[start_idx],
            "end": t_fmt[end_idx - 1],
            "bars": int(end_idx - start_idx),
            "mean": round(mean_value, 8),
            "volatility": round(volatility, 8),
        }
        if target == "return":
            row["mean_t_stat"] = (
                round(float(mean_t_stat), 4)
                if mean_t_stat is not None and np.isfinite(mean_t_stat)
                else None
            )
            row["direction_significant"] = bool(direction_significant)
        regimes.append(row)
        if start_idx > 0:
            change_points.append(
                {"idx": int(start_idx), "time": t_fmt[start_idx]}
            )
        start_idx = end_idx

    if not regimes:
        return {"error": "PELT produced no valid regime segments."}
    latest = regimes[-1]
    mean_contrast = float(np.std([float(row["mean"]) for row in regimes], ddof=0))
    confidence = min(1.0, mean_contrast / global_volatility) if global_volatility > 0 else 0.0
    current_regime = {
        "regime_id": latest["regime"],
        "label": latest["label"],
        "since": latest["start"],
        "bars": latest["bars"],
        "regime_confidence": round(float(confidence), 4),
    }
    payload: Dict[str, Any] = {
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "method": "pelt",
        "target": target,
        "current_regime": current_regime,
        "regimes": regimes[-int(max_regimes) :] if output != "full" else regimes,
        "change_points": change_points,
        "summary": {
            "segments": int(len(regimes)),
            "change_points_count": int(len(change_points)),
            "current_segment_bars": int(latest["bars"]),
        },
        "reliability": _common_reliability(
            None,
            source="pelt_segment_separation",
            confidence=confidence,
        ),
        "params_used": {
            "model": model,
            "penalty": round(float(penalty), 10),
            "penalty_source": penalty_source,
            "min_size": int(min_size),
            "jump": int(jump),
            "direction_t_stat_threshold": _PELT_DIRECTION_T_STAT_THRESHOLD,
        },
    }
    if include_series and output == "full":
        payload["series"] = {
            "times": t_fmt,
            "values": [float(value) for value in x.tolist()],
        }
    return payload


def _detect_ms_ar(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    target: str,
    method: str,
    x: np.ndarray,
    t_fmt: List[Any],
    p: Dict[str, Any],
    lookback: int,
    output: str,
    include_series: bool,
    max_regimes: int,
    min_regime_bars_val: int,
) -> Dict[str, Any]:
    try:
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,  # type: ignore
        )
    except ImportError:
        return {
                "error": "statsmodels MarkovAutoregression not available. Install statsmodels.",
                "error_code": "dependency_missing",
                "details": {"method": "ms_ar", "requires": ["statsmodels"]},
            }
    n_states_msar, n_states_error, n_states_source, state_count_warnings = (
        _resolve_state_count_param(
            p,
            default=2,
            method="ms_ar",
        )
    )
    if n_states_error is not None:
        return {"error": n_states_error}
    if n_states_msar is None or n_states_msar < 2:
        return {"error": "n_states must be >= 2 for ms_ar."}
    order, _ = _coerce_param(p, "order", default=1, cast=int)
    if order < 1:
        return {"error": "params.order must be >= 1 for ms_ar."}
    if x.size <= order:
        return {
                "error": (
                    f"ms_ar order={order} requires more than {order} "
                    "usable observations."
                )
            }
    try:
        mod = MarkovAutoregression(
            endog=x,
            k_regimes=max(2, n_states_msar),
            trend="c",
            order=order,
            switching_ar=True,
            switching_variance=True,
        )
        maxiter, _ = _coerce_param(p, "maxiter", default=100, cast=int)
        res = mod.fit(disp=False, maxiter=maxiter)
        inference = str(p.get("inference", "filtered")).strip().lower()
        if inference not in {"filtered", "smoothed"}:
            return {"error": "params.inference must be 'filtered' or 'smoothed'."}
        marginal = (
            res.filtered_marginal_probabilities
            if inference == "filtered"
            else res.smoothed_marginal_probabilities
        )
        if hasattr(marginal, "values"):
            marginal = marginal.values
        probs = np.asarray(marginal, dtype=float)
        x_model = np.asarray(x[order:], dtype=float)
        t_fmt_model = list(t_fmt[order:])
        if probs.ndim != 2 or probs.shape[0] != x_model.size:
            raise ValueError(
                "MarkovAutoregression probability rows did not match "
                "the AR-aligned analysis window"
            )
        raw_state = np.argmax(probs, axis=1)
        state, smoothing_meta = _confirm_state_changes_causally(
            np.asarray(raw_state, dtype=int), min_regime_bars_val
        )
        state, probs, canon_meta = _canonicalize_regime_labels(
            state,
            probs,
            x_model,
        )
        smoothing_meta["relabeled"] = canon_meta.get("relabeled", False)
        mle_retvals = getattr(res, "mle_retvals", None)
        converged = None
        if isinstance(mle_retvals, dict):
            converged = mle_retvals.get("converged")
        elif mle_retvals is not None and hasattr(mle_retvals, "get"):
            try:
                converged = mle_retvals.get("converged")
            except Exception:
                converged = getattr(mle_retvals, "converged", None)
        param_names = list(getattr(getattr(res, "model", None), "param_names", []) or [])
        try:
            param_values = np.asarray(getattr(res, "params", []), dtype=float).reshape(-1)
        except Exception:
            param_values = np.array([], dtype=float)
        fitted_params = {
            str(name): float(value)
            for name, value in zip(param_names, param_values)
            if np.isfinite(value)
        }
    except Exception as ex:
        return {"error": f"MS-AR fitting error: {ex}"}

    # Build regime parameters (mean/vol per regime) for states that
    # actually have observations after smoothing/canonicalization.
    # Iterating range(n_states_msar) produced phantom entries (0.0
    # mean / 0.0 vol) for states that smoothing had eliminated,
    # which then leaked into payload labels and downstream scoring.
    unique_canonical_states = sorted(int(s) for s in np.unique(state).tolist())
    msar_regime_params: Dict[str, Any] = {
        "mean_return": [],
        "volatility": [],
    }
    label_mapping = {
        int(old): int(new)
        for old, new in canon_meta.get("mapping", {}).items()
    }
    canonical_to_native = {new: old for old, new in label_mapping.items()}
    intercepts: List[float] = []
    innovation_volatility: List[float] = []
    ar_coefficients: List[List[float]] = []
    for s in unique_canonical_states:
        mask = state == s
        if mask.any():
            msar_regime_params["mean_return"].append(float(np.mean(x_model[mask])))
            msar_regime_params["volatility"].append(float(np.std(x_model[mask])))
        else:
            msar_regime_params["mean_return"].append(0.0)
            msar_regime_params["volatility"].append(0.0)
        native_state = canonical_to_native.get(s, s)
        intercept = fitted_params.get(f"const[{native_state}]")
        sigma2 = fitted_params.get(f"sigma2[{native_state}]")
        ar_values = [
            fitted_params.get(f"ar.L{lag}[{native_state}]")
            for lag in range(1, order + 1)
        ]
        if intercept is not None:
            intercepts.append(float(intercept))
        if sigma2 is not None and sigma2 >= 0.0:
            innovation_volatility.append(float(np.sqrt(sigma2)))
        if all(value is not None for value in ar_values):
            ar_coefficients.append([float(value) for value in ar_values])
    if len(intercepts) == len(unique_canonical_states):
        msar_regime_params["intercept"] = intercepts
    if len(innovation_volatility) == len(unique_canonical_states):
        msar_regime_params["innovation_volatility"] = innovation_volatility
    if len(ar_coefficients) == len(unique_canonical_states):
        msar_regime_params["ar_coefficients"] = ar_coefficients

    payload = {
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "method": method,
        "target": target,
        "times": t_fmt_model,
        "state": [int(s) for s in state.tolist()],
        "state_probabilities": [
            [float(v) for v in row] for row in probs.tolist()
        ],
        "regime_params": msar_regime_params,
        "params_used": {
            "n_states": int(n_states_msar),
            "state_count_param": n_states_source,
            "order": order,
            "inference": inference,
            "model_fit_scope": "full_window",
            "state_postprocess": "causal_confirmation",
            "state_probability_alignment": "pre_confirmation_model_probabilities",
            "regime_params_order": "canonical",
            "min_regime_bars": int(min_regime_bars_val),
            "relabeled": bool(canon_meta.get("relabeled", False)),
            "smoothing_applied": bool(
                smoothing_meta.get("smoothing_applied", False)
            ),
            "transitions_before": int(
                smoothing_meta.get("transitions_before", 0)
            ),
            "transitions_after": int(
                smoothing_meta.get("transitions_after", 0)
            ),
        },
    }
    if canon_meta.get("mapping"):
        payload["params_used"]["label_mapping"] = canon_meta["mapping"]
    _append_warnings(payload, state_count_warnings)
    _append_warnings(payload, _smoothing_warnings(method, smoothing_meta))
    if converged is not None:
        payload["params_used"]["converged"] = bool(converged)
        if not bool(converged):
            _append_warnings(
                payload,
                [
                    "MS-AR model did not converge; regime probabilities may be unreliable."
                ],
            )
    # Add reliability info
    reliability = _ms_ar_reliability_from_smoothed(
        smoothed_probs=probs,
        params_used=payload["params_used"],
    )
    payload["reliability"] = _common_reliability(
        reliability,
        source=f"ms_ar_{inference}_probabilities",
    )

    if output in ("summary", "compact"):
        n = _summary_window_size(lookback, len(state))
        st_tail = state[-n:] if n > 0 else state
        last_s = int(state[-1]) if len(state) else None
        unique, counts = np.unique(st_tail, return_counts=True)
        shares = {
            int(k): float(c) / float(len(st_tail) or 1)
            for k, c in zip(unique, counts)
        }
        summary = {
            "lookback": int(n),
            "last_state": last_s,
            "state_shares": shares,
            "transitions_before": int(
                smoothing_meta.get("transitions_before", 0)
            ),
            "transitions_after": int(
                smoothing_meta.get("transitions_after", 0)
            ),
            "smoothing_applied": bool(
                smoothing_meta.get("smoothing_applied", False)
            ),
        }
        payload = _apply_state_output_mode(
            payload,
            output=output,
            lookback=lookback,
            summary=summary,
        )
        if output == "summary":
            return payload

    return _consolidate_payload(
            payload,
            method,
            output,
            include_series=include_series,
            max_regimes=max_regimes,
        )


def _detect_hmm_or_gmm(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    target: str,
    method: str,
    x: np.ndarray,
    t_fmt: List[Any],
    p: Dict[str, Any],
    lookback: int,
    output: str,
    include_series: bool,
    max_regimes: int,
    min_regime_bars_val: int,
) -> Dict[str, Any]:
    n_states, n_states_error, n_states_source, state_count_warnings = (
        _resolve_state_count_param(
            p,
            default=2,
            method=method,
        )
    )
    if n_states_error is not None:
        return {"error": n_states_error}
    if n_states is None or n_states < 2:
        return {"error": f"n_states must be >= 2 for {method}."}
    inference = str(p.get("inference", "filtered")).strip().lower()
    if inference not in {"filtered", "smoothed"}:
        return {"error": "params.inference must be 'filtered' or 'smoothed'."}
    hmm_fit: Dict[str, Any] = {}
    if method == "hmm":
        try:
            from ...forecast.monte_carlo import fit_gaussian_hmm_1d
        except Exception as ex:
            return {"error": f"Gaussian HMM import error: {ex}"}
        fit_gaussian_hmm_1d = globals().get(
            "fit_gaussian_hmm_1d", fit_gaussian_hmm_1d
        )
        try:
            hmm_fit = fit_gaussian_hmm_1d(
                x,
                n_states=n_states,
                max_iter=int(p.get("maxiter", 80)),
                tol=float(p.get("tol", 1e-6)),
                seed=int(p.get("seed", 42)),
            )
        except ImportError:
            return {
                "error": "hmmlearn GaussianHMM is unavailable.",
                "error_code": "dependency_missing",
                "details": {"method": "hmm", "requires": ["hmmlearn"]},
            }
        mu = np.asarray(hmm_fit["mu"], dtype=float)
        sigma = np.asarray(hmm_fit["sigma"], dtype=float)
        w = np.asarray(hmm_fit["state_occupancy"], dtype=float)
        gamma = np.asarray(
            hmm_fit[f"{inference}_probabilities"], dtype=float
        )
    else:
        try:
            from ...forecast.monte_carlo import fit_gaussian_mixture_1d
        except Exception as ex:
            return {"error": f"Gaussian mixture import error: {ex}"}
        fit_gaussian_mixture_1d = globals().get(
            "fit_gaussian_mixture_1d", fit_gaussian_mixture_1d
        )
        w, mu, sigma, gamma, _ = fit_gaussian_mixture_1d(
            x, n_states=n_states
        )
        if "inference" in p:
            state_count_warnings.append(
                "params.inference does not apply to gmm responsibilities."
            )
    gamma_matrix = _normalize_state_probability_matrix(
        gamma,
        rows=x.size,
        requested_states=len(mu),
    )
    raw_state = (
        np.argmax(gamma_matrix, axis=1)
        if gamma_matrix.size
        else np.zeros(x.size, dtype=int)
    )
    state, smoothing_meta = _confirm_state_changes_causally(
        np.asarray(raw_state, dtype=int), min_regime_bars_val
    )
    state, gamma_for_payload, canon_meta = _canonicalize_regime_labels(
        state,
        gamma_matrix,
        x,
    )
    smoothing_meta["relabeled"] = canon_meta.get("relabeled", False)
    if not isinstance(gamma_for_payload, np.ndarray):
        gamma_for_payload = gamma_matrix
    label_mapping = {
        int(old): int(new)
        for old, new in canon_meta.get("mapping", {}).items()
    }
    if label_mapping:
        native_order = sorted(
            range(len(mu)),
            key=lambda old: label_mapping.get(old, len(label_mapping) + old),
        )
        mu = mu[native_order]
        sigma = sigma[native_order]
        w = w[native_order]
    regime_params: Dict[str, Any] = {
        "mu": [float(v) for v in mu.tolist()],
        "sigma": [float(v) for v in sigma.tolist()],
    }
    if method == "hmm":
        transition_matrix = np.asarray(hmm_fit["trans"], dtype=float)
        initial_probabilities = np.asarray(hmm_fit["start_prob"], dtype=float)
        if label_mapping:
            transition_matrix = transition_matrix[np.ix_(native_order, native_order)]
            initial_probabilities = initial_probabilities[native_order]
        regime_params.update({
            "transition_matrix": [
                [float(v) for v in row]
                for row in transition_matrix.tolist()
            ],
            "initial_probabilities": [
                float(v) for v in initial_probabilities.tolist()
            ],
            "state_occupancy": [float(v) for v in w.tolist()],
        })
    else:
        regime_params["weights"] = [float(v) for v in w.tolist()]
    payload = {
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "method": method,
        "target": target,
        "times": t_fmt,
        "state": [int(s) for s in state.tolist()],
        "state_probabilities": [
            [float(v) for v in row] for row in gamma_for_payload.tolist()
        ],
        "regime_params": regime_params,
        "params_used": {
            "n_states": int(n_states),
            "fitted_n_states": int(len(mu)),
            "state_count_param": n_states_source,
            "inference": inference if method == "hmm" else "component_responsibility",
            "model_fit_scope": "full_window",
            "state_postprocess": "causal_confirmation",
            "state_probability_alignment": "pre_confirmation_model_probabilities",
            "min_regime_bars": int(min_regime_bars_val),
            "relabeled": bool(canon_meta.get("relabeled", False)),
            "regime_params_order": "canonical",
            "smoothing_applied": bool(
                smoothing_meta.get("smoothing_applied", False)
            ),
            "transitions_before": int(
                smoothing_meta.get("transitions_before", 0)
            ),
            "transitions_after": int(
                smoothing_meta.get("transitions_after", 0)
            ),
        },
    }
    effective_n_states = int(len(mu))
    payload["requested_n_states"] = int(n_states)
    payload["effective_n_states"] = effective_n_states
    if method == "hmm":
        payload["params_used"].update({
            "converged": bool(hmm_fit.get("converged", False)),
            "log_likelihood": float(hmm_fit.get("log_likelihood", 0.0)),
        })
    if canon_meta.get("mapping"):
        payload["params_used"]["label_mapping"] = canon_meta["mapping"]
    _append_warnings(payload, state_count_warnings)
    _append_warnings(payload, _smoothing_warnings(method, smoothing_meta))
    if effective_n_states < n_states:
        _append_warnings(
            payload,
            [
                f"{method.upper()} state collapse: requested "
                f"{int(n_states)} states but fitted {effective_n_states}; "
                "regime output uses the reduced-state model."
            ],
        )
    # Add reliability info
    reliability = _hmm_reliability_from_gamma(gamma_for_payload)
    payload["reliability"] = _common_reliability(
        reliability,
        source=(
            f"hmm_{inference}_probabilities"
            if method == "hmm"
            else "gmm_component_responsibilities"
        ),
    )
    if effective_n_states < n_states:
        payload["reliability"].update(
            {
                "confidence": 0.0,
                "reliability_label": "low",
                "confidence_note": (
                    "Requested states collapsed during fitting; classification "
                    "confidence is not identifiable from the reduced-state result."
                ),
            }
        )

    if output in ("summary", "compact"):
        n = _summary_window_size(lookback, len(state))
        st_tail = state[-n:] if n > 0 else state
        last_s = int(state[-1]) if len(state) else None
        unique, counts = np.unique(st_tail, return_counts=True)
        shares = {
            int(k): float(c) / float(len(st_tail) or 1)
            for k, c in zip(unique, counts)
        }
        order = np.argsort(sigma)
        ranks = {int(s): int(r) for r, s in enumerate(order)}
        summary = {
            "lookback": int(n),
            "last_state": last_s,
            "state_shares": shares,
            "state_sigma": {int(i): float(sigma[i]) for i in range(len(sigma))},
            "state_order_by_sigma": ranks,
            "transitions_before": int(
                smoothing_meta.get("transitions_before", 0)
            ),
            "transitions_after": int(
                smoothing_meta.get("transitions_after", 0)
            ),
            "smoothing_applied": bool(
                smoothing_meta.get("smoothing_applied", False)
            ),
        }
        payload = _apply_state_output_mode(
            payload,
            output=output,
            lookback=lookback,
            summary=summary,
        )
        if output == "summary":
            return payload

    consolidated = _consolidate_payload(
        payload,
        method,
        output,
        include_series=include_series,
        max_regimes=max_regimes,
    )
    if effective_n_states < n_states:
        consolidated = _mark_collapsed_state_confidence(consolidated)
    return consolidated


def _detect_clustering(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    target: str,
    method: str,
    x: np.ndarray,
    t_fmt: List[Any],
    p: Dict[str, Any],
    lookback: int,
    output: str,
    include_series: bool,
    max_regimes: int,
    min_regime_bars_val: int,
) -> Dict[str, Any]:
    try:
        standard_scaler_cls = globals().get("StandardScaler")
        kmeans_cls = globals().get("KMeans")
        pca_cls = globals().get("PCA")
        if standard_scaler_cls is None:
            from sklearn.preprocessing import (
                StandardScaler as standard_scaler_cls,
            )
        if kmeans_cls is None:
            from sklearn.cluster import KMeans as kmeans_cls
        if pca_cls is None:
            from sklearn.decomposition import PCA as pca_cls
        algorithm = str(p.get("algorithm", "kmeans")).strip().lower()
        spectral_cls = None
        if algorithm == "spectral":
            _sc = globals().get("SpectralClustering")
            if _sc is None:
                from sklearn.cluster import (
                    SpectralClustering as _sc,
                )
            spectral_cls = _sc
    except ImportError as ex:
        return {"error": f"Clustering dependencies missing: {ex}"}
    window_size, _ = _coerce_param(p, "window_size", default=20, cast=int)
    n_states_cluster, n_states_error, n_states_source, state_count_warnings = (
        _resolve_state_count_param(
            p,
            default=3,
            method="clustering",
        )
    )
    if n_states_error is not None:
        return {"error": n_states_error}
    if n_states_cluster is None or n_states_cluster < 2:
        return {"error": "n_states must be >= 2 for clustering."}
    use_pca = bool(p.get("use_pca", True))
    n_components, _ = _coerce_param(p, "n_components", default=3, cast=int)
    clustering_warnings: List[str] = []
    if target == "price":
        clustering_warnings.append(
            "Clustering on price features may produce level-dependent regimes. Consider target='return'."
        )

    # Extract features (use 'return' or 'price'? 'return' is stationary, usually better)
    # x is already computed based on target input
    extract_rolling_features_impl = globals().get(
        "extract_rolling_features", extract_rolling_features
    )
    if extract_rolling_features_impl is extract_rolling_features:
        extract_rolling_features_impl = (
            _features_module.extract_rolling_features
        )
    features_df = extract_rolling_features_impl(x, window_size=window_size)

    # Align features with time
    # valid_indices are where features are not NaN
    valid_mask = ~features_df.isna().any(axis=1)
    X_valid = features_df.loc[valid_mask]

    if X_valid.empty:
        return {
                "error": "Not enough data for feature extraction (check window_size)"
            }

    # Normalize
    scaler = standard_scaler_cls()
    X_scaled = scaler.fit_transform(X_valid)

    # PCA
    if use_pca and X_scaled.shape[1] > n_components:
        pca = pca_cls(n_components=min(n_components, X_scaled.shape[1]))
        X_final = pca.fit_transform(X_scaled)
    else:
        X_final = X_scaled

    # Cluster
    n_samples = X_final.shape[0]
    if n_samples < n_states_cluster:
        return {
                "error": f"Not enough samples ({n_samples}) for {n_states_cluster} clusters"
            }

    if algorithm == "spectral" and spectral_cls is not None:
        affinity = str(p.get("affinity", "nearest_neighbors")).strip().lower()
        sc_kwargs: Dict[str, Any] = {
            "n_clusters": n_states_cluster,
            "affinity": affinity,
            "random_state": 42,
            "assign_labels": "kmeans",
            "n_init": 1,
        }
        if affinity == "nearest_neighbors":
            sc_kwargs["n_neighbors"] = min(
                max(5, n_samples // 10), n_samples - 1
            )
        sc = spectral_cls(**sc_kwargs)
        labels = sc.fit_predict(X_final)
        valid_probs = None
    else:
        # Seed centroids from evenly-spaced rows for deterministic
        # initialization. KMeans still asks joblib for its OpenMP thread
        # count before initialization; MCP startup warms that Windows CPU
        # topology cache before requests enter asyncio worker threads.
        idx = np.round(np.linspace(0, n_samples - 1, n_states_cluster)).astype(int)
        kmeans = kmeans_cls(
            n_clusters=n_states_cluster,
            random_state=42,
            n_init=1,
            init=X_final[idx],
        )
        labels = kmeans.fit_predict(X_final)
        valid_probs = None
        try:
            distances = np.asarray(kmeans.transform(X_final), dtype=float)
            if distances.shape == (len(X_final), n_states_cluster):
                inverse_distance = 1.0 / (distances + 1e-8)
                valid_probs = inverse_distance / inverse_distance.sum(
                    axis=1, keepdims=True
                )
        except (AttributeError, TypeError, ValueError):
            valid_probs = None

    # Smooth short runs and canonicalize on valid slice only
    labels, smoothing_meta = _confirm_state_changes_causally(
        np.asarray(labels, dtype=int), min_regime_bars_val
    )
    labels, valid_probs, canon_meta = _canonicalize_regime_labels(
        labels,
        valid_probs,
        x[valid_mask],
    )
    smoothing_meta["relabeled"] = canon_meta.get("relabeled", False)

    # Map back to full length (-1 for undefined leading window)
    full_states = np.full(len(x), -1, dtype=int)
    full_states[valid_mask] = labels

    full_probs = None
    if valid_probs is not None:
        full_probs = np.zeros((len(x), n_states_cluster))
        full_probs[valid_mask] = valid_probs

    mean_return, volatility = _observed_state_mean_vol(full_states, x)
    clustering_regime_params = {
        "mean_return": mean_return,
        "volatility": volatility,
    }

    # Reconstruct payload
    payload = {
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "method": method,
        "target": target,
        "times": t_fmt,
        "state": [int(s) for s in full_states.tolist()],
        "regime_params": clustering_regime_params,
        "params_used": {
            "n_states": int(n_states_cluster),
            "state_count_param": n_states_source,
            "algorithm": algorithm,
            "window_size": window_size,
            "use_pca": use_pca,
            "n_components": n_components,
            "min_regime_bars": int(min_regime_bars_val),
            "smoothing_applied": smoothing_meta.get("smoothing_applied", False),
            "transitions_before": int(
                smoothing_meta.get("transitions_before", 0)
            ),
            "transitions_after": int(
                smoothing_meta.get("transitions_after", 0)
            ),
            "model_fit_scope": "full_window",
            "label_scope": "retrospective_canonical",
        },
    }
    if full_probs is not None:
        payload["state_probabilities"] = [
            [float(v) for v in row] for row in full_probs.tolist()
        ]
    else:
        payload["confidence_basis"] = "unavailable_for_clustering_method"
    if clustering_warnings:
        payload["warnings"] = clustering_warnings
    _append_warnings(payload, state_count_warnings)
    _append_warnings(payload, _smoothing_warnings(method, smoothing_meta))

    # Summary stats
    if output in ("summary", "compact"):
        n_summary = _summary_window_size(lookback, len(full_states))
        st_tail = full_states[-n_summary:] if n_summary > 0 else full_states
        # Filter out -1
        st_tail_valid = st_tail[st_tail != -1]

        unique, counts = np.unique(st_tail_valid, return_counts=True)
        shares = {
            int(k): float(c) / float(len(st_tail_valid) or 1)
            for k, c in zip(unique, counts)
        }

        summary = {
            "lookback": int(n_summary),
            "last_state": int(full_states[-1]) if len(full_states) else None,
            "state_shares": shares,
            "transitions_before": int(
                smoothing_meta.get("transitions_before", 0)
            ),
            "transitions_after": int(
                smoothing_meta.get("transitions_after", 0)
            ),
            "smoothing_applied": bool(
                smoothing_meta.get("smoothing_applied", False)
            ),
        }
        payload = _apply_state_output_mode(
            payload,
            output=output,
            lookback=lookback,
            summary=summary,
        )
        if output == "summary":
            return payload

    # Score the final labels in the feature space used to fit them.
    feature_variance_ratio = _feature_cluster_separation(X_final, labels)
    reliability_score = min(
        1.0, feature_variance_ratio * 2
    )  # Scale for interpretability

    payload["reliability"] = {
        "confidence": round(reliability_score, 4),
        "feature_variance_ratio": round(feature_variance_ratio, 4),
        "source": "feature_cluster_separation",
    }
    payload["reliability"] = _common_reliability(
        payload["reliability"],
        source="feature_cluster_separation",
    )

    return _consolidate_payload(
            payload,
            method,
            output,
            include_series=include_series,
            max_regimes=max_regimes,
        )


def _detect_garch(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    target: str,
    method: str,
    x: np.ndarray,
    t_fmt: List[Any],
    p: Dict[str, Any],
    lookback: int,
    output: str,
    include_series: bool,
    max_regimes: int,
    min_regime_bars_val: int,
) -> Dict[str, Any]:
    # GARCH-based volatility regime detection
    garch_warnings: List[str] = []
    if target != "return":
        return {
                "error": "GARCH regime detection requires target='return'; price levels are non-stationary for this volatility model.",
                "error_code": "invalid_target",
                "details": {"method": "garch", "target": target},
            }
    try:
        from arch import arch_model
    except ImportError:
        from ..error_envelope import build_error_payload

        return build_error_payload(
            "arch package required for GARCH regime detection. Install: pip install arch",
            code="missing_dependency",
            operation="regime_detect",
            details={"method": "garch", "requires": ["arch"]},
        )

    # Auto-detect optimal n_states if not explicitly provided
    # Based on volatility distribution characteristics
    n_states_input = p.get("n_states")
    state_count_warnings: List[str] = []

    if n_states_input is None:
        # Calculate rolling realized volatility for better characterization
        # Use 20-bar rolling window to capture volatility clustering
        window = min(20, len(x) // 4)
        if window < 5:
            window = 5

        # Rolling standard deviation of returns
        # The historical implementation used ``i - window`` as the
        # inclusive start, so preserve its window + 1 observations.
        rolling_vol = _rolling_prefix_std(x, window)
        rolling_vol = rolling_vol[np.isfinite(rolling_vol) & (rolling_vol > 0)]

        if len(rolling_vol) > 10:
            # Use ratio of 90th to 10th percentile to measure volatility range
            vol_p90 = np.percentile(rolling_vol, 90)
            vol_p10 = np.percentile(rolling_vol, 10)
            vol_ratio = vol_p90 / vol_p10 if vol_p10 > 1e-9 else 1.0

            # Also calculate kurtosis of returns (fat tails indicator)
            returns_kurt = _finite_raw_kurtosis(x)

            # Infer optimal states based on vol_ratio and kurtosis
            # High vol_ratio (10+) or high kurtosis (>6) suggests need for more states
            if vol_ratio > 10.0 or returns_kurt > 6.0:
                n_states_auto = (
                    4  # Very volatile - need very_low/low/high/very_high
                )
            elif vol_ratio > 5.0 or returns_kurt > 4.0:
                n_states_auto = 3  # Moderately volatile - low/moderate/high
            else:
                n_states_auto = 2  # Stable - binary classification sufficient

            auto_detect_metrics = {
                "vol_ratio_90_10": round(vol_ratio, 2),
                "returns_kurtosis": round(returns_kurt, 2),
            }
        else:
            # Insufficient data, default to 3 states
            n_states_auto = 3
            auto_detect_metrics = {}

        n_states_garch = n_states_auto
        garch_auto_n_states = True
    else:
        try:
            n_states_garch = int(n_states_input)
        except Exception:
            return {"error": "n_states must be an integer >= 2 for garch."}
        garch_auto_n_states = False
    garch_p, _ = _coerce_param(p, "p_order", default=1, cast=int)
    garch_q, _ = _coerce_param(p, "q_order", default=1, cast=int)
    vol_threshold, _ = _coerce_param(
        p, "vol_threshold", default=None, cast=float
    )

    if n_states_garch < 2:
        return {"error": "n_states must be >= 2 for garch method."}

    # Fit GARCH model
    try:
        # Scale returns for numerical stability
        scale = 100.0
        x_scaled = x * scale

        am = arch_model(
            x_scaled,
            vol="GARCH",
            p=max(1, garch_p),
            q=max(1, garch_q),
            dist="normal",
            mean="Constant",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", show_warning=False)

        # Extract conditional volatility
        conditional_vol = res.conditional_volatility / scale  # Unscale

        # Create regime states based on volatility levels
        # Strategy: sort volatilities and assign regimes based on percentiles
        # State 0 = lowest vol, State N-1 = highest vol
        valid_vol = conditional_vol[np.isfinite(conditional_vol)]

        if len(valid_vol) < n_states_garch * 10:
            return {
                    "error": f"Insufficient data for GARCH regime detection (need {n_states_garch * 10}+ bars)"
                }

        # Determine volatility thresholds
        thresholds, threshold_scope = _garch_tier_thresholds(
            valid_vol,
            n_states_garch,
            vol_threshold,
        )

        # Assign states based on volatility levels
        state = np.zeros(len(conditional_vol), dtype=int)
        for i, thresh in enumerate(thresholds):
            state[conditional_vol > thresh] = i + 1

        # Handle non-finite values
        state[~np.isfinite(conditional_vol)] = -1

        # Causally confirm changes, then align the hard-assignment
        # probabilities with the emitted state path.
        state, smoothing_meta = _confirm_state_changes_causally(
            np.asarray(state, dtype=int), min_regime_bars_val
        )
        probs = _hard_state_probability_matrix(state, n_states_garch)

        # Build regime parameters
        regime_params = {"volatility": [], "mean_return": []}
        for s in range(n_states_garch):
            mask = state == s
            if mask.any():
                regime_params["volatility"].append(
                    float(np.mean(conditional_vol[mask]))
                )
                regime_params["mean_return"].append(
                    float(np.mean(x[mask])) if mask.sum() > 0 else 0.0
                )
            else:
                regime_params["volatility"].append(0.0)
                regime_params["mean_return"].append(0.0)

        # Build payload
        # Check if n_states seems appropriate for this asset
        garch_warnings = []
        vol_std = float(np.std(valid_vol))
        vol_mean = float(np.mean(valid_vol))
        cv = (
            vol_std / vol_mean if vol_mean > 1e-9 else 0
        )  # Coefficient of variation

        # Heuristic: High CV (>1.0) suggests volatile asset needing more states
        if cv > 1.0 and n_states_garch < 3:
            garch_warnings.append(
                f"High volatility variation detected (CV={cv:.2f}). "
                f"Consider n_states=3 or 4 for better regime separation."
            )

        # Build volatility characteristics for transparency
        vol_characteristics = {
            "cv": round(cv, 4),
            "mean": round(float(np.mean(valid_vol)), 6),
            "std": round(float(np.std(valid_vol)), 6),
            "percentile_33": round(float(np.percentile(valid_vol, 33)), 6),
            "percentile_66": round(float(np.percentile(valid_vol, 66)), 6),
        }

        # Add auto-detection metrics if applicable
        if garch_auto_n_states and auto_detect_metrics:
            vol_characteristics["auto_detection"] = auto_detect_metrics

        payload = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "method": method,
            "target": target,
            "times": t_fmt,
            "state": [int(s) for s in state.tolist()],
            "state_probabilities": [
                [float(v) for v in row] for row in probs.tolist()
            ],
            "conditional_volatility": [
                float(v) for v in conditional_vol.tolist()
            ],
            "regime_params": regime_params,
            "params_used": {
                "n_states": int(n_states_garch),
                "n_states_auto": bool(garch_auto_n_states),
                "p_order": int(garch_p),
                "q_order": int(garch_q),
                "min_regime_bars": int(min_regime_bars_val),
                "smoothing_applied": bool(
                    smoothing_meta.get("smoothing_applied", False)
                ),
                "transitions_before": int(
                    smoothing_meta.get("transitions_before", 0)
                ),
                "transitions_after": int(
                    smoothing_meta.get("transitions_after", 0)
                ),
                "classification": "conditional_volatility_tiers",
                "threshold_scope": threshold_scope,
                "volatility_thresholds": [float(v) for v in thresholds],
                "model_fit_scope": "full_window",
            },
            "volatility_characteristics": vol_characteristics,
        }
        if garch_warnings:
            payload["warnings"] = garch_warnings
        _append_warnings(payload, state_count_warnings)
        _append_warnings(payload, _smoothing_warnings(method, smoothing_meta))

        # Model fit is reported separately; it is not a confidence
        # score for the derived percentile-tier classification.
        if hasattr(res, "aic") and hasattr(res, "bic"):
            payload["model_fit"] = {
                "aic": float(res.aic),
                "bic": float(res.bic),
                "loglikelihood": float(res.loglikelihood)
                if hasattr(res, "loglikelihood")
                else None,
            }

        # Add summary for compact/summary output
        if output in ("summary", "compact"):
            n = _summary_window_size(lookback, len(state))
            st_tail = state[-n:] if n > 0 else state
            vol_tail = conditional_vol[-n:] if n > 0 else conditional_vol
            last_s = int(state[-1]) if len(state) else None

            unique, counts = np.unique(
                st_tail[st_tail >= 0], return_counts=True
            )
            shares = {
                int(k): float(c) / float(len(st_tail[st_tail >= 0]) or 1)
                for k, c in zip(unique, counts)
            }

            summary = {
                "lookback": int(n),
                "last_state": last_s,
                "state_shares": shares,
                "current_conditional_vol": float(conditional_vol[-1])
                if len(conditional_vol)
                else None,
                "avg_conditional_vol": float(np.mean(vol_tail))
                if len(vol_tail)
                else None,
                "transitions_before": int(
                    smoothing_meta.get("transitions_before", 0)
                ),
                "transitions_after": int(
                    smoothing_meta.get("transitions_after", 0)
                ),
                "smoothing_applied": bool(
                    smoothing_meta.get("smoothing_applied", False)
                ),
            }
            payload = _apply_state_output_mode(
                payload,
                output=output,
                lookback=lookback,
                summary=summary,
            )
            if output == "summary":
                return payload

        return _consolidate_payload(
                payload,
                method,
                output,
                include_series=include_series,
                max_regimes=max_regimes,
            )

    except Exception as ex:
        return {"error": f"GARCH regime detection failed: {str(ex)}"}


def _detect_rule_based(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    target: str,
    method: str,
    output: str,
    rule_based_config: Optional[Dict[str, Any]],
    price_series: np.ndarray,
    price_times: np.ndarray,
    global_warnings: List[str],
) -> Dict[str, Any]:
    # Rule-based trend/ranging/transition detection
    if rule_based_config is None:
        return {"error": "Internal error resolving rule_based parameters."}

    efficiency_threshold = float(rule_based_config["efficiency_threshold"])
    trend_strength_threshold = float(rule_based_config["trend_strength_threshold"])
    requested_window_bars = int(rule_based_config["window_bars"])

    # Ensure window isn't too large
    window_bars = min(requested_window_bars, len(price_series))
    if window_bars < requested_window_bars:
        global_warnings.append(
            "params.window_bars exceeded available finite price bars; "
            f"requested {requested_window_bars}, using {int(window_bars)}."
        )

    if window_bars < 20:
        return {
                "error": f"Insufficient data for rule-based regime (need 20+ bars, got {window_bars})"
            }

    regime_metrics = infer_market_regime(
        price_series,
        window_bars=window_bars,
        efficiency_threshold=efficiency_threshold,
        trend_strength_threshold=trend_strength_threshold,
    )
    if regime_metrics is None:
        return {"error": "Insufficient finite price data for rule-based regime."}
    regime_state = str(regime_metrics["state"])
    direction = str(regime_metrics["direction"])
    trend_strength = float(regime_metrics["trend_strength"])
    efficiency_ratio = float(regime_metrics["efficiency_ratio"])
    window_move_pct_raw = float(regime_metrics["window_move_pct"])
    ranging_efficiency_threshold = max(0.1, 0.55 * efficiency_threshold)

    direction_bias = {
        "bullish": "upward",
        "bearish": "downward",
        "neutral": "neutral",
    }.get(direction, "neutral")
    if regime_state == "ranging":
        interpretation = (
            f"Price is ranging with a {direction_bias} net move over "
            f"{int(window_bars)} bars; direction is a window bias, not a trend classification."
        )
    elif regime_state == "transition":
        interpretation = (
            f"Price is in transition with a {direction_bias} net move over "
            f"{int(window_bars)} bars."
        )
    else:
        interpretation = (
            f"Price is trending {direction_bias} over {int(window_bars)} bars."
        )

    state_note = None
    if (
        regime_state == "ranging"
        and trend_strength >= trend_strength_threshold
    ):
        state_note = (
            "trend_strength exceeds threshold, but efficiency_ratio indicates "
            "a choppy path; state uses both metrics."
        )

    # Build an aggregate-window classification. This method does not detect
    # historical boundaries, persistence, or a per-bar state sequence.
    trend_strength_out = round(trend_strength, 4)
    efficiency_ratio_out = round(efficiency_ratio, 4)
    window_move_pct = round(window_move_pct_raw, 4)
    window_quality: Optional[Dict[str, Any]] = None
    if int(window_bars) < _RULE_BASED_RECOMMENDED_WINDOW_BARS:
        window_quality = {
            "status": "limited_history",
            "lookback_too_short": True,
            "recommended_min_bars": _RULE_BASED_RECOMMENDED_WINDOW_BARS,
            "window_bars": int(window_bars),
        }
    regime_info = {
        "state": regime_state,
        "state_label_native": regime_state,
        "state_label_canonical": regime_state,
        "direction_basis": "net_window_move",
        "interpretation": interpretation,
        "trend_strength": trend_strength_out,
        "efficiency_ratio": efficiency_ratio_out,
        "window_bars": int(window_bars),
        "window_move_pct": window_move_pct,
        "signal_source": "price",
    }
    if regime_state == "trending":
        regime_info["direction"] = direction
    else:
        regime_info["window_bias"] = direction
    if state_note:
        regime_info["note"] = state_note
    confidence = 0.0
    try:
        if regime_state == "trending":
            confidence = (
                min(1.0, efficiency_ratio / max(float(efficiency_threshold), 1e-9))
                + min(
                    1.0,
                    trend_strength / max(float(trend_strength_threshold), 1e-9),
                )
            ) / 2.0
        elif regime_state == "ranging":
            confidence = (
                max(0.0, ranging_efficiency_threshold - efficiency_ratio)
                / max(float(ranging_efficiency_threshold), 1e-9)
            )
        else:
            transition_span = max(
                float(efficiency_threshold - ranging_efficiency_threshold),
                1e-9,
            )
            distance_from_boundary = min(
                max(0.0, efficiency_ratio - ranging_efficiency_threshold),
                max(0.0, efficiency_threshold - efficiency_ratio),
            )
            confidence = min(1.0, (distance_from_boundary / transition_span) * 2.0)
        confidence = min(1.0, max(0.0, float(confidence)))
    except Exception:
        confidence = 0.0
    regime_confidence = round(float(confidence), 4)
    regime_id_by_state = {"ranging": 0, "trending": 1, "transition": 2}
    regime_id = regime_id_by_state.get(regime_state, 2)
    rule_t_fmt = [_format_time_minimal(tt) for tt in price_times]
    classification_start = (
        rule_t_fmt[-int(window_bars)]
        if len(rule_t_fmt) >= int(window_bars)
        else rule_t_fmt[0]
    )
    classification_end = rule_t_fmt[-1]
    classification_window = {
        "start": classification_start,
        "end": classification_end,
        "bars": int(window_bars),
        "basis": "aggregate_price_window",
    }
    current_regime = {
        "regime_id": int(regime_id),
        "label": regime_state,
        "regime_confidence": regime_confidence,
        "classification_scope": "aggregate_window",
        "boundary_status": "not_estimated",
        "persistence_status": "not_estimated",
        "state_label_native": regime_state,
        "state_label_canonical": regime_state,
        "headline": f"regime={regime_state}; window_bias={direction}",
    }
    if regime_state == "trending":
        current_regime["direction"] = direction
    else:
        current_regime["window_bias"] = direction
    regime_payload = dict(regime_info)

    reliability = _common_reliability(
        {
            "confidence": regime_confidence,
            "trend_strength": trend_strength_out,
            "efficiency_ratio": efficiency_ratio_out,
        },
        source="rule_based_trend_efficiency",
    )
    payload = {
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "method": method,
        "target": target,
        "regime": regime_payload,
        "current_regime": current_regime,
        "classification_window": classification_window,
        "regime_info": {
            int(regime_id): {
                "label": regime_state,
                **(
                    {"direction": direction}
                    if regime_state == "trending"
                    else {"window_bias": direction}
                ),
                "trend_strength": trend_strength_out,
                "efficiency_ratio": efficiency_ratio_out,
                "window_move_pct": window_move_pct,
            }
        },
        "reliability": reliability,
        "params_used": {
            "efficiency_threshold": float(efficiency_threshold),
            "trend_strength_threshold": float(trend_strength_threshold),
            "window_bars": int(window_bars),
            "signal_source": "price",
        },
    }
    if window_quality:
        payload["data_quality"] = window_quality
        current_regime["window_quality"] = window_quality["status"]
    if output == "summary":
        payload["summary"] = {
            "classification_window_bars": int(window_bars),
            "last_state": int(regime_id),
            "label": regime_state,
            **(
                {"direction": direction}
                if regime_state == "trending"
                else {"window_bias": direction}
            ),
            "headline": f"regime={regime_state}; window_bias={direction}",
            "regime_confidence": regime_confidence,
        }
        payload["summary"] = {
            key: value
            for key, value in payload["summary"].items()
            if value is not None
        }
        if regime_state != "trending" or state_note:
            payload["summary"]["direction_basis"] = "net_window_move"
            payload["summary"]["interpretation"] = interpretation
        if state_note:
            payload["summary"]["note"] = state_note
        return _summary_only_payload(payload)
    if output == "compact":
        compact_current_regime = dict(current_regime)
        for key in (
            "state_label_native",
            "state_label_canonical",
            "headline",
        ):
            compact_current_regime.pop(key, None)
        compact_current_regime.update(
            {
                "trend_strength": trend_strength_out,
                "efficiency_ratio": efficiency_ratio_out,
                "window_move_pct": window_move_pct,
            }
        )
        if regime_state != "trending" or state_note:
            compact_current_regime["direction_basis"] = "net_window_move"
            compact_current_regime["interpretation"] = interpretation
        if regime_state != "trending":
            compact_current_regime["direction_role"] = "window_bias_not_trend"
        if state_note:
            compact_current_regime["note"] = state_note
        payload = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "method": method,
            "target": target,
            "signal_status": (
                "information_only" if regime_state == "trending" else "not_actionable"
            ),
            "current_regime": compact_current_regime,
            "classification_window": classification_window,
        }
        if window_quality:
            payload["data_quality"] = window_quality

    return payload


def _detect_wavelet(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    target: str,
    method: str,
    x: np.ndarray,
    t_fmt: List[Any],
    p: Dict[str, Any],
    lookback: int,
    output: str,
    include_series: bool,
    max_regimes: int,
    min_regime_bars_val: int,
) -> Dict[str, Any]:
    # Multi-resolution wavelet energy regime detection.
    # Decomposes the series via DWT, computes rolling energy at each
    # decomposition level, then clusters the energy feature vectors
    # to identify regimes that differ in frequency content.
    try:
        import pywt as _pywt
    except ImportError:
        return {
                "error": "PyWavelets required for wavelet regime detection. "
                "Install: pip install PyWavelets"
            }

    wavelet_name = str(p.get("wavelet", "db4")).strip()
    n_states_wv, n_states_error, n_states_source, state_count_warnings = (
        _resolve_state_count_param(
            p,
            default=3,
            method="wavelet",
        )
    )
    if n_states_error is not None:
        return {"error": n_states_error}
    if n_states_wv is None:
        n_states_wv = 3
    energy_window, _ = _coerce_param(p, "energy_window", default=30, cast=int)

    if n_states_wv < 2:
        return {"error": "n_states must be >= 2 for wavelet method."}
    if energy_window < 1:
        return {"error": "params.energy_window must be a positive integer."}
    if len(x) < energy_window + 10:
        return {
                "error": f"Insufficient data for wavelet regime detection "
                f"(need {energy_window + 10}+ bars, got {len(x)})"
            }

    # Determine decomposition level
    try:
        w = _pywt.Wavelet(wavelet_name)
    except Exception:
        return {"error": f"Unknown wavelet: {wavelet_name}"}
    max_level = _pywt.dwt_max_level(len(x), w.dec_len)
    user_level = p.get("level")
    if user_level is not None:
        level = max(1, min(int(user_level), max_level))
    else:
        level = max(1, min(4, max_level))

    # Symmetric extension avoids coupling the window head into its tail.
    boundary_mode = "symmetric"
    bands = _wavelet_detail_bands(
        x,
        wavelet_name,
        level,
        boundary_mode=boundary_mode,
        pywt_module=_pywt,
    )

    if not bands:
        return {"error": "Wavelet decomposition produced no detail bands."}

    # Compute rolling energy (variance) for each band
    n_bars = len(x)
    n_bands = len(bands)
    energy_matrix = _rolling_band_energy(bands, energy_window)

    # Normalize energy rows to proportions (energy distribution across scales)
    row_sums = energy_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-16, 1.0, row_sums)
    energy_props = energy_matrix / row_sums

    # Cluster energy profiles into regimes using KMeans
    # (sklearn is already available from clustering branch pattern)
    try:
        from sklearn.cluster import KMeans as _WvKMeans
        from sklearn.preprocessing import StandardScaler as _WvScaler
    except ImportError:
        return {"error": "sklearn required for wavelet regime clustering."}

    # Skip leading bars where energy window isn't fully populated
    valid_start = min(energy_window, n_bars - 1)
    E_valid = energy_props[valid_start:]
    if len(E_valid) < n_states_wv:
        return {
                "error": f"Not enough valid bars ({len(E_valid)}) for "
                f"{n_states_wv} wavelet regimes."
            }

    scaler = _WvScaler()
    E_scaled = scaler.fit_transform(E_valid)

    n_valid = E_scaled.shape[0]
    idx = np.round(np.linspace(0, n_valid - 1, n_states_wv)).astype(int)
    km = _WvKMeans(
        n_clusters=n_states_wv,
        random_state=42,
        n_init=1,
        init=E_scaled[idx],
    )
    labels = km.fit_predict(E_scaled)

    # Build probability matrix from cluster distances
    distances = km.transform(E_scaled)  # (n_valid, n_states_wv)
    inv_dist = 1.0 / (distances + 1e-8)
    probs_valid = inv_dist / inv_dist.sum(axis=1, keepdims=True)

    # Smooth and canonicalize
    labels, smoothing_meta = _confirm_state_changes_causally(
        np.asarray(labels, dtype=int), min_regime_bars_val
    )
    labels, probs_valid, canon_meta = _canonicalize_regime_labels(
        labels,
        probs_valid,
        x[valid_start:],
    )
    smoothing_meta["relabeled"] = canon_meta.get("relabeled", False)

    # Map back to full length
    full_states = np.full(n_bars, -1, dtype=int)
    full_states[valid_start:] = labels
    full_probs = np.zeros((n_bars, n_states_wv))
    full_probs[valid_start:] = probs_valid

    # Compute per-regime energy profiles for interpretability
    regime_energy_profiles: Dict[str, Any] = {}
    wavelet_regime_params: Dict[str, Any] = {
        "mean_return": [],
        "volatility": [],
        "energy_profiles": regime_energy_profiles,
        "n_bands": n_bands,
        "band_labels": [f"D{i}" for i in range(1, n_bands + 1)],
    }
    x_valid = x[valid_start:]
    mean_return, volatility = _observed_state_mean_vol(labels, x_valid)
    wavelet_regime_params["mean_return"] = mean_return
    wavelet_regime_params["volatility"] = volatility
    for s in sorted({int(v) for v in np.unique(labels) if int(v) >= 0}):
        mask = labels == s
        profile = energy_props[valid_start:][mask].mean(axis=0)
        regime_energy_profiles[str(s)] = {
            f"band_{bi}_energy": round(float(v), 6)
            for bi, v in enumerate(profile)
        }

    payload = {
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "method": method,
        "target": target,
        "times": t_fmt,
        "state": [int(s) for s in full_states.tolist()],
        "state_probabilities": [
            [float(v) for v in row] for row in full_probs.tolist()
        ],
        "regime_params": wavelet_regime_params,
        "params_used": {
            "wavelet": wavelet_name,
            "level": level,
            "n_states": n_states_wv,
            "state_count_param": n_states_source,
            "energy_window": energy_window,
            "energy_window_mode": "trailing",
            "boundary_mode": boundary_mode,
            "model_fit_scope": "full_window",
            "min_regime_bars": int(min_regime_bars_val),
            "smoothing_applied": smoothing_meta.get("smoothing_applied", False),
        },
    }
    _append_warnings(payload, state_count_warnings)
    _append_warnings(payload, _smoothing_warnings(method, smoothing_meta))
    max_state_prob = np.max(probs_valid, axis=1) if probs_valid.size else np.array([])
    payload["reliability"] = _common_reliability(
        {
            "confidence": round(float(np.mean(max_state_prob)), 4)
            if max_state_prob.size
            else 0.0,
            "mean_state_probability": round(float(np.mean(max_state_prob)), 4)
            if max_state_prob.size
            else 0.0,
        },
        source="wavelet_cluster_distance",
    )

    if output in ("summary", "compact"):
        n_summary = _summary_window_size(lookback, len(full_states))
        st_tail = full_states[-n_summary:] if n_summary > 0 else full_states
        st_tail_valid = st_tail[st_tail != -1]
        unique, counts = np.unique(st_tail_valid, return_counts=True)
        shares = {
            int(k): float(c) / float(len(st_tail_valid) or 1)
            for k, c in zip(unique, counts)
        }
        summary = {
            "lookback": int(n_summary),
            "last_state": int(full_states[-1]) if len(full_states) else None,
            "state_shares": shares,
        }
        payload = _apply_state_output_mode(
            payload,
            output=output,
            lookback=lookback,
            summary=summary,
        )
        if output == "summary":
            return payload

    return _consolidate_payload(
            payload,
            method,
            output,
            include_series=include_series,
            max_regimes=max_regimes,
        )


def _run_regime_method(
    *,
    method: str,
    symbol: str,
    timeframe: str,
    target: str,
    x: np.ndarray,
    t_fmt: List[Any],
    p: Dict[str, Any],
    lookback: int,
    output: str,
    include_series: bool,
    max_regimes: int,
    min_regime_bars_val: int,
    threshold: Optional[float] = None,
    min_regime_bars: Optional[int] = None,
    calibration_returns: Optional[np.ndarray] = None,
    price_series: Optional[np.ndarray] = None,
    price_times: Optional[np.ndarray] = None,
    rule_based_config: Optional[Dict[str, Any]] = None,
    global_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run one private regime method against already-prepared series."""
    if method == "bocpd":
        returns = (
            np.asarray(calibration_returns, dtype=float)
            if calibration_returns is not None
            else np.asarray(x, dtype=float)
        )
        return _detect_bocpd(
            symbol=symbol,
            timeframe=timeframe,
            target=target,
            method=method,
            x=x,
            t_fmt=t_fmt,
            p=p,
            lookback=lookback,
            output=output,
            include_series=include_series,
            max_regimes=max_regimes,
            min_regime_bars_val=min_regime_bars_val,
            threshold=threshold,
            min_regime_bars=min_regime_bars,
            calibration_returns=returns,
        )
    if method == "pelt":
        return _detect_pelt(
            symbol=symbol,
            timeframe=timeframe,
            target=target,
            x=x,
            t_fmt=t_fmt,
            p=p,
            output=output,
            include_series=include_series,
            max_regimes=max_regimes,
            min_regime_bars_val=min_regime_bars_val,
        )
    if method == "ms_ar":
        return _detect_ms_ar(
            symbol=symbol,
            timeframe=timeframe,
            target=target,
            method=method,
            x=x,
            t_fmt=t_fmt,
            p=p,
            lookback=lookback,
            output=output,
            include_series=include_series,
            max_regimes=max_regimes,
            min_regime_bars_val=min_regime_bars_val,
        )
    if method in {"hmm", "gmm"}:
        return _detect_hmm_or_gmm(
            symbol=symbol,
            timeframe=timeframe,
            target=target,
            method=method,
            x=x,
            t_fmt=t_fmt,
            p=p,
            lookback=lookback,
            output=output,
            include_series=include_series,
            max_regimes=max_regimes,
            min_regime_bars_val=min_regime_bars_val,
        )
    if method == "clustering":
        return _detect_clustering(
            symbol=symbol,
            timeframe=timeframe,
            target=target,
            method=method,
            x=x,
            t_fmt=t_fmt,
            p=p,
            lookback=lookback,
            output=output,
            include_series=include_series,
            max_regimes=max_regimes,
            min_regime_bars_val=min_regime_bars_val,
        )
    if method == "garch":
        return _detect_garch(
            symbol=symbol,
            timeframe=timeframe,
            target=target,
            method=method,
            x=x,
            t_fmt=t_fmt,
            p=p,
            lookback=lookback,
            output=output,
            include_series=include_series,
            max_regimes=max_regimes,
            min_regime_bars_val=min_regime_bars_val,
        )
    if method == "wavelet":
        return _detect_wavelet(
            symbol=symbol,
            timeframe=timeframe,
            target=target,
            method=method,
            x=x,
            t_fmt=t_fmt,
            p=p,
            lookback=lookback,
            output=output,
            include_series=include_series,
            max_regimes=max_regimes,
            min_regime_bars_val=min_regime_bars_val,
        )
    if method == "rule_based":
        config = rule_based_config or {
            "efficiency_threshold": 0.35,
            "trend_strength_threshold": 1.25,
            "window_bars": 160,
        }
        series = price_series if price_series is not None else x
        times = price_times if price_times is not None else np.asarray([], dtype=float)
        return _detect_rule_based(
            symbol=symbol,
            timeframe=timeframe,
            target=target,
            method=method,
            output=output,
            rule_based_config=config,
            price_series=np.asarray(series, dtype=float),
            price_times=np.asarray(times, dtype=float),
            global_warnings=global_warnings if global_warnings is not None else [],
        )
    return {"error": f"Unsupported regime method: {method}"}


def _detect_ensemble(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    target: str,
    method: str,
    x: np.ndarray,
    t_fmt: List[Any],
    p: Dict[str, Any],
    lookback: int,
    output: str,
    include_series: bool,
    max_regimes: int,
    min_regime_bars_val: int,
    threshold: Optional[float],
    fetch_limit: Optional[int],
    denoise: Any,
    min_regime_bars: Optional[int],
) -> Dict[str, Any]:
    from .api import (
        _normalize_regime_method_name,
        _regime_params_for_method,
    )

    # Consensus regime detection: run multiple fast methods and
    # aggregate their state_probabilities via soft or hard voting.
    default_sub = ["hmm", "clustering", "wavelet"]
    sub_methods_raw = p.get("methods", default_sub)
    if isinstance(sub_methods_raw, str):
        sub_methods_raw = [m.strip() for m in sub_methods_raw.split(",")]
    sub_methods: List[str] = []
    unsupported_methods: List[str] = []
    for candidate in sub_methods_raw:
        normalized = _normalize_regime_method_name(candidate)
        if normalized not in _ENSEMBLE_STATE_METHODS:
            if normalized not in unsupported_methods:
                unsupported_methods.append(normalized)
            continue
        if normalized not in sub_methods:
            sub_methods.append(normalized)
    if unsupported_methods:
        return {
                "error": (
                    "Ensemble methods must be return-canonicalized state "
                    "classifiers. Supported methods: clustering, gmm, hmm, "
                    "ms_ar, wavelet. Unsupported: "
                    + ", ".join(unsupported_methods)
                    + "."
                ),
                "error_code": "invalid_ensemble_methods",
            }
    if not sub_methods:
        return {"error": "No valid sub-methods for ensemble."}

    voting_input = p.get("voting", "soft")
    if not isinstance(voting_input, str) or not voting_input.strip():
        return {
                "error": "Ensemble voting must be one of: soft, hard.",
                "error_code": "invalid_ensemble_voting",
            }
    voting = voting_input.strip().lower()
    if voting not in {"soft", "hard"}:
        return {
                "error": (
                    f"Unsupported ensemble voting mode '{voting_input}'. "
                    "Expected one of: soft, hard."
                ),
                "error_code": "invalid_ensemble_voting",
            }

    (
        n_states_ens,
        n_states_source,
        ens_auto_n_states,
        ens_auto_metrics,
        state_count_warnings,
        state_count_error,
    ) = _ensemble_state_count_configuration(p, x)
    if state_count_error:
        return {"error": state_count_error}

    # Run each sub-method with include_series so we get raw state data
    sub_results: List[Dict[str, Any]] = []
    sub_errors: List[str] = []
    for sm in sub_methods:
        sub_params = _regime_params_for_method(p, sm)
        sub_params.pop("methods", None)
        sub_params.pop("voting", None)
        sub_params.setdefault("n_states", n_states_ens)
        try:
            sr = _run_regime_method(
                method=sm,
                symbol=symbol,
                timeframe=timeframe,
                target=target,
                x=x,
                t_fmt=t_fmt,
                p=sub_params,
                lookback=lookback,
                output="full",
                include_series=True,
                max_regimes=max_regimes,
                min_regime_bars_val=min_regime_bars_val,
                threshold=threshold,
                min_regime_bars=min_regime_bars,
            )
        except Exception as exc:
            sub_errors.append(f"{sm}: {exc}")
            continue
        if isinstance(sr, dict) and sr.get("error"):
            sub_errors.append(f"{sm}: {sr['error']}")
            continue
        sub_results.append({"method": sm, "result": sr})

    if not sub_results:
        return {
                "error": f"All ensemble sub-methods failed: {'; '.join(sub_errors)}"
            }

    return _aggregate_precomputed_ensemble(
            symbol=symbol,
            timeframe=timeframe,
            target=target,
            x=x,
            t_fmt=t_fmt,
            sub_results=sub_results,
            sub_errors=sub_errors,
            requested_methods=sub_methods,
            voting=voting,
            n_states_ens=n_states_ens,
            n_states_source=n_states_source,
            ens_auto_n_states=ens_auto_n_states,
            ens_auto_metrics=ens_auto_metrics,
            state_count_warnings=state_count_warnings,
            min_regime_bars_val=min_regime_bars_val,
            output=output,
            lookback=lookback,
            include_series=include_series,
            max_regimes=max_regimes,
            aggregation_source="fitted_submethods",
        )


def _detect_all(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    target: str,
    method: str,
    x: np.ndarray,
    t_fmt: List[Any],
    p: Dict[str, Any],
    lookback: int,
    output: str,
    include_series: bool,
    max_regimes: int,
    min_regime_bars_val: int,
    threshold: Optional[float],
    fetch_limit: Optional[int],
    denoise: Any,
    min_regime_bars: Optional[int],
    verbosity_output: str,
    calibration_returns: Optional[np.ndarray] = None,
    price_series: Optional[np.ndarray] = None,
    price_times: Optional[np.ndarray] = None,
    rule_based_config: Optional[Dict[str, Any]] = None,
    global_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    from .api import _regime_params_for_method

    # Run all methods and return individual results for comparison
    detail_value = output
    sub_detail = "full" if verbosity_output == "full" else "compact"
    include_series_for_subcalls = bool(include_series) and sub_detail == "full"
    all_methods = [
        "bocpd",
        "pelt",
        "hmm",
        "gmm",
        "ms_ar",
        "clustering",
        "garch",
        "wavelet",
        "rule_based",
    ]
    results_by_method: Dict[str, Any] = {}
    all_errors: List[str] = []
    method_durations_ms: Dict[str, float] = {}
    method_errors: Dict[str, str] = {}
    ensemble_sub_results: List[Dict[str, Any]] = []

    for m in all_methods:
        method_started_at = time.perf_counter()
        try:
            sub_params = _regime_params_for_method(p, m)
            # Only set default n_states for methods that don't auto-detect
            # GARCH auto-detects optimal n_states, don't force a default
            if m in ("hmm", "ms_ar", "clustering"):
                sub_params.setdefault("n_states", 2)
            # GARCH: if n_states not explicitly set, leave it out for auto-detection
            ensemble_eligible = m in _ENSEMBLE_STATE_METHODS
            sr = _run_regime_method(
                method=m,
                symbol=symbol,
                timeframe=timeframe,
                target=target,
                x=x,
                t_fmt=t_fmt,
                p=sub_params,
                lookback=lookback,
                output="full" if ensemble_eligible else sub_detail,
                include_series=(
                    True if ensemble_eligible else include_series_for_subcalls
                ),
                max_regimes=max_regimes,
                min_regime_bars_val=min_regime_bars_val,
                threshold=threshold,
                min_regime_bars=min_regime_bars,
                calibration_returns=calibration_returns,
                price_series=price_series,
                price_times=price_times,
                rule_based_config=rule_based_config,
                global_warnings=global_warnings,
            )
            if isinstance(sr, dict) and not sr.get("error"):
                if ensemble_eligible:
                    ensemble_sub_results.append(
                        {"method": m, "result": sr}
                    )
                # Strip redundant fields that are already at top level
                # (symbol, timeframe, method, target, success)
                cleaned_result = {
                    k: v
                    for k, v in sr.items()
                    if k
                    not in (
                        "symbol",
                        "timeframe",
                        "method",
                        "target",
                        "success",
                    )
                }
                if not include_series_for_subcalls:
                    cleaned_result.pop("series", None)
                results_by_method[m] = cleaned_result
            else:
                error_text = (
                    str(sr.get("error", "unknown error"))
                    if isinstance(sr, dict)
                    else "unknown error"
                )
                method_errors[m] = error_text
                all_errors.append(f"{m}: {error_text}")
        except Exception as exc:
            method_errors[m] = str(exc)
            all_errors.append(f"{m}: {exc}")
        finally:
            method_durations_ms[m] = round(
                (time.perf_counter() - method_started_at) * 1000.0,
                3,
            )

    if not results_by_method:
        return {
                "error": f"All methods failed: {'; '.join(all_errors)}",
                "error_code": "regime_methods_failed",
                "runtime": {
                    "completed_methods": [],
                    "failed_methods": list(method_errors.keys()),
                    "method_errors": method_errors,
                    "method_durations_ms": method_durations_ms,
                    "partial_results": False,
                    "suggested_faster_methods": _suggest_faster_regime_methods(
                        all_methods
                    ),
                    "method_guidance": _regime_runtime_guidance(
                        ["all", *all_methods]
                    ),
                },
            }

    # Aggregate the first-pass state series into the consensus view.
    ensemble_health: Dict[str, Any] = {}
    try:
        ensemble_started_at = time.perf_counter()
        requested_voters = [
            method_name
            for method_name in all_methods
            if method_name in _ENSEMBLE_STATE_METHODS
        ]
        voting = str(p.get("voting", "soft") or "").strip().lower()
        if voting not in {"soft", "hard"}:
            raise ValueError(
                "Ensemble voting must be one of: soft, hard."
            )
        (
            n_states_ens,
            n_states_source,
            ens_auto_n_states,
            ens_auto_metrics,
            state_count_warnings,
            state_count_error,
        ) = _ensemble_state_count_configuration(p, x)
        if state_count_error:
            raise ValueError(state_count_error)
        ensemble_errors = [
            f"{method_name}: {method_errors[method_name]}"
            for method_name in requested_voters
            if method_name in method_errors
        ]
        ensemble_result = _aggregate_precomputed_ensemble(
            symbol=symbol,
            timeframe=timeframe,
            target=target,
            x=x,
            t_fmt=t_fmt,
            sub_results=ensemble_sub_results,
            sub_errors=ensemble_errors,
            requested_methods=requested_voters,
            voting=voting,
            n_states_ens=n_states_ens,
            n_states_source=n_states_source,
            ens_auto_n_states=ens_auto_n_states,
            ens_auto_metrics=ens_auto_metrics,
            state_count_warnings=state_count_warnings,
            min_regime_bars_val=min_regime_bars_val,
            output=sub_detail,
            lookback=lookback,
            include_series=include_series_for_subcalls,
            max_regimes=max_regimes,
            aggregation_source="reused_all_first_pass",
        )
        if isinstance(ensemble_result, dict) and not ensemble_result.get(
            "error"
        ):
            ensemble_health = dict(
                ensemble_result.get("ensemble_health") or {}
            )
            # Strip redundant fields
            results_by_method["ensemble"] = {
                k: v
                for k, v in ensemble_result.items()
                if k
                not in ("symbol", "timeframe", "method", "target", "success")
            }
            method_durations_ms["ensemble"] = round(
                (time.perf_counter() - ensemble_started_at) * 1000.0,
                3,
            )
        else:
            error_text = (
                str(ensemble_result.get("error", "unknown error"))
                if isinstance(ensemble_result, dict)
                else "unknown error"
            )
            method_errors["ensemble"] = error_text
            method_durations_ms["ensemble"] = round(
                (time.perf_counter() - ensemble_started_at) * 1000.0,
                3,
            )
    except Exception as exc:
        method_errors["ensemble"] = str(exc)
        method_durations_ms["ensemble"] = round(
            (time.perf_counter() - ensemble_started_at) * 1000.0,
            3,
        )

    attempted_components = [*all_methods, "ensemble"]
    succeeded_components = [
        method_name
        for method_name in attempted_components
        if method_name in results_by_method
    ]
    failed_components = [
        method_name
        for method_name in attempted_components
        if method_name in method_errors
    ]
    comparison = _build_all_method_comparison(results_by_method)
    comparison["methods_failed"] = failed_components
    ensemble_aggregated = "ensemble" in results_by_method
    ensemble_degraded = bool(ensemble_health.get("degraded"))

    summary_payload: Optional[Dict[str, Any]] = None
    if detail_value in {"summary", "compact"}:
        summary_payload = {
            "methods_attempted": int(len(attempted_components)),
            "methods_succeeded": int(len(succeeded_components)),
            "methods_failed": int(len(failed_components)),
        }
        if ensemble_aggregated:
            summary_payload["ensemble_aggregated"] = True
            summary_payload["ensemble_degraded"] = ensemble_degraded
            summary_payload["ensemble_voters_used"] = len(
                ensemble_health.get("used_voters") or []
            )
            summary_payload["ensemble_voters_requested"] = len(
                ensemble_health.get("requested_voters") or []
            )
        agreement_summary = comparison.get("agreement")
        if isinstance(agreement_summary, dict):
            summary_payload["agreement"] = agreement_summary
        # Summary/compact modes drop per-method regimes and diagnostics.
        compact_comparison = {
            "methods_run": comparison.get("methods_run"),
            "methods_failed": comparison.get("methods_failed"),
        }
        if comparison.get("method_windows"):
            compact_comparison["method_windows"] = comparison.get("method_windows")
        if detail_value == "compact":
            compact_comparison["agreement"] = comparison.get("agreement")
        comparison = compact_comparison
    runtime_payload: Dict[str, Any] = {
        "completed_methods": list(succeeded_components),
        "failed_methods": list(failed_components),
        "partial_results": bool(failed_components or ensemble_degraded),
    }
    if ensemble_aggregated:
        runtime_payload["ensemble_aggregated"] = True
        runtime_payload["ensemble_voters"] = ensemble_health
        runtime_payload["ensemble_degraded"] = ensemble_degraded
        runtime_payload["ensemble_aggregation_source"] = (
            "reused_all_first_pass"
        )
    if method_errors:
        runtime_payload["method_errors"] = method_errors
    if detail_value == "full":
        runtime_payload["method_durations_ms"] = method_durations_ms
        runtime_payload["suggested_faster_methods"] = (
            _suggest_faster_regime_methods(all_methods)
        )
        runtime_payload["method_guidance"] = _regime_runtime_guidance(
            ["all", *all_methods, "ensemble"]
        )

    payload = {
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "method": method,
        "target": target,
        "detail": detail_value,
        "comparison": comparison,
        "runtime": runtime_payload,
    }
    if ensemble_aggregated:
        payload["ensemble_health"] = ensemble_health
    if detail_value == "full":
        payload["params_used"] = {
            "methods_attempted": attempted_components,
            "methods_succeeded": list(succeeded_components),
            "methods_failed": list(failed_components),
        }
        if ensemble_aggregated:
            payload["params_used"]["ensemble_aggregated"] = True
    if summary_payload is not None:
        payload["summary"] = summary_payload
    if detail_value == "full":
        payload["results"] = results_by_method
    warnings_out: List[str] = []
    if failed_components:
        error_summary = "; ".join(
            f"{method_name}: {method_errors[method_name]}"
            for method_name in failed_components
        )
        warnings_out.append(f"Method errors: {error_summary}")
    if ensemble_degraded:
        excluded = ", ".join(
            str(name)
            for name in ensemble_health.get("excluded_voters") or []
        )
        warnings_out.append(
            "Ensemble consensus is degraded; excluded voters: "
            + (excluded or "unknown")
            + "."
        )
    if warnings_out:
        payload["warnings"] = warnings_out

    return payload
