"""Regime detection implementation."""

import logging
import math
import time
from typing import Annotated, Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import Field

from ...forecast.common import fetch_history as _fetch_history
from ...forecast.common import log_returns_from_prices as _log_returns_from_prices
from ...shared.schema import DenoiseSpecInput, DetailLiteral, TimeframeLiteral
from ...shared.validators import unknown_mapping_keys_error
from ...utils.denoise import resolve_denoise_base_col
from ...utils.freshness import completed_bar_freshness_fields
from ...utils.mt5 import (
    MT5ConnectionError,
    ensure_mt5_connection_or_raise,
    resolve_public_symbol,
)
from ...utils.time import _format_time_minimal
from ...utils.utils import validate_historical_range
from .._mcp_instance import mcp
from ..error_envelope import build_error_payload
from ..execution_logging import (
    infer_result_success,
    log_operation_finish,
    log_operation_start,
)
from ..mt5_gateway import create_mt5_gateway, mt5_connection_error
from ..output_contract import (
    attach_completed_bar_input_policy,
    normalize_output_detail,
    normalize_output_verbosity_detail,
)
from ..runtime_metadata import attach_mt5_source
from .detect import (
    _BOCPD_UNDERSEG_MIN_BARS,
    _BOCPD_UNDERSEG_PEAK_Z,
    _PELT_DIRECTION_T_STAT_THRESHOLD,
    _REGIME_METHOD_RUNTIME_GUIDANCE,
    _RULE_BASED_RECOMMENDED_WINDOW_BARS,
    _bocpd_under_segmentation_warnings,
    _coerce_param,
    _detect_all,
    _detect_ensemble,
    _feature_cluster_separation,
    _garch_tier_thresholds,
    _method_parameter_warnings,
    _peak_abs_return,
    _pelt_return_direction,
    _regime_runtime_guidance,
    _resolve_bocpd_priors,
    _resolve_state_count_param,
    _rolling_band_energy,
    _rolling_prefix_std,
    _run_regime_method,
    _suggest_faster_regime_methods,
    _wavelet_detail_bands,
)
from .ensemble import (
    _ENSEMBLE_STATE_METHODS,
    _aggregate_precomputed_ensemble,
    _align_states_to_return_centroids,
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
    _count_state_transitions,
    _hard_state_probability_matrix,
    _normalize_state_probability_matrix,
    _state_runs,
)
from .summarize import (
    _DIRECTION_SIGNALS,
    _VOLATILITY_SIGNALS,
    _append_warnings,
    _apply_bocpd_output_mode,
    _apply_state_output_mode,
    _build_all_method_comparison,
    _build_semantic_agreement,
    _coerce_optional_float,
    _common_reliability,
    _lookup_regime_info_entry,
    _mark_collapsed_state_confidence,
    _normalize_direction_signal,
    _normalize_volatility_signal,
    _reliability_label,
    _smoothing_warnings,
    _summarize_bocpd_current_regime,
    _summarize_current_regime_for_comparison,
    _summarize_rule_based_current_regime,
    _summary_window_size,
)

logger = logging.getLogger(__name__)

_REGIME_COMMON_PARAM_KEYS = {"lookback", "min_regime_bars"}
_REGIME_METHOD_PARAM_KEYS = {
    "bocpd": {
        "cp_confirm_bars",
        "cp_confirm_relaxed_mult",
        "cp_edge_multiplier",
        "cp_threshold",
        "cp_threshold_calibration_mode",
        "hazard_lambda",
        "hazard_mode",
        "max_run_length",
        "min_cp_distance_bars",
        "threshold",
        "threshold_calibration_bootstraps",
        "threshold_calibration_max_windows",
        "threshold_calibration_step",
        "threshold_calibration_window",
        "threshold_target_false_alarm_rate",
    },
    "pelt": {"jump", "min_size", "model", "penalty"},
    "ms_ar": {"inference", "maxiter", "n_states", "order"},
    "hmm": {"inference", "maxiter", "n_states", "seed", "tol"},
    "gmm": {"inference", "n_states"},
    "clustering": {
        "affinity",
        "algorithm",
        "n_components",
        "n_states",
        "use_pca",
        "window_size",
    },
    "garch": {"n_states", "p_order", "q_order", "vol_threshold"},
    "rule_based": {
        "efficiency_threshold",
        "trend_strength_threshold",
        "window_bars",
    },
    "wavelet": {"energy_window", "level", "n_states", "wavelet"},
}
_REGIME_ENSEMBLE_METHODS = {"clustering", "gmm", "hmm", "ms_ar", "wavelet"}


def _regime_param_keys(method: str) -> set[str]:
    if method == "ensemble":
        method_keys = set().union(
            *(_REGIME_METHOD_PARAM_KEYS[name] for name in _REGIME_ENSEMBLE_METHODS)
        ) | {"methods", "n_states", "voting"}
    elif method == "all":
        method_keys = set().union(*_REGIME_METHOD_PARAM_KEYS.values()) | {"voting"}
    else:
        method_keys = set(_REGIME_METHOD_PARAM_KEYS.get(method, set()))
    return _REGIME_COMMON_PARAM_KEYS | method_keys


def _regime_params_for_method(params: Dict[str, Any], method: str) -> Dict[str, Any]:
    allowed = _regime_param_keys(method)
    return {key: value for key, value in params.items() if key in allowed}


def _regime_connection_error() -> Optional[Dict[str, Any]]:
    return mt5_connection_error(
        create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
    )


def _method_min_fetch_limit(method: str) -> int:
    if method == "rule_based":
        return 20
    return 10


def _invalid_fetch_limit_error(message: str, *, fetch_limit: Any, method: str) -> Dict[str, Any]:
    return build_error_payload(
        message,
        code="cli_invalid_arguments",
        operation="regime_detect",
        details={"fetch_limit": fetch_limit, "method": method},
        remediation=(
            "Pass a positive fetch_limit at or above the method minimum, or omit "
            "it to use the timeframe lookback plus warmup."
        ),
        documentation="docs/forecast/REGIMES.md",
    )


def _history_fetch_limit(fetch_limit: Optional[int], lookback: int) -> int:
    if fetch_limit is not None:
        return int(fetch_limit)
    return int(max(int(lookback), 50)) + 20


def _normalize_regime_method_name(method: Any) -> str:
    text = str(method or "").strip().lower()
    return text


# Timeframe-based default parameters for regime detection
_TIMEFRAME_DEFAULTS: Dict[str, Dict[str, int]] = {
    # Intraday high-frequency
    "M1": {"lookback": 3000, "min_regime_bars": 30},  # ~2 days, 30 min regimes
    "M5": {"lookback": 2000, "min_regime_bars": 12},  # ~7 days, 1 hour regimes
    "M15": {"lookback": 1000, "min_regime_bars": 8},  # ~10 days, 2 hour regimes
    "M30": {"lookback": 800, "min_regime_bars": 6},  # ~16 days, 3 hour regimes
    # Standard intraday/swing
    "H1": {"lookback": 500, "min_regime_bars": 4},  # ~21 days, 4 hour regimes
    "H2": {"lookback": 400, "min_regime_bars": 3},  # ~33 days, 6 hour regimes
    "H4": {"lookback": 300, "min_regime_bars": 3},  # ~50 days, 12 hour regimes
    "H6": {"lookback": 250, "min_regime_bars": 2},  # ~62 days, 12 hour regimes
    "H8": {"lookback": 200, "min_regime_bars": 2},  # ~66 days, 16 hour regimes
    "H12": {"lookback": 150, "min_regime_bars": 2},  # ~75 days, 24 hour regimes
    # Daily and higher
    "D1": {"lookback": 200, "min_regime_bars": 2},  # ~200 days, 2 day regimes
    "W1": {"lookback": 100, "min_regime_bars": 2},  # ~100 weeks, 2 week regimes
    "MN1": {"lookback": 48, "min_regime_bars": 2},  # ~48 months, 2 month regimes
}


def _attach_regime_usage_notice(result: Dict[str, Any]) -> None:
    if not isinstance(result, dict) or result.get("error"):
        return
    result.setdefault("is_signal", False)
    result.setdefault("usage", "information_only")
    result.setdefault(
        "calibration",
        {
            "confidence": "model or heuristic assignment score, not historical hit rate",
            "note": (
                "Regime labels describe observed state. Validate with backtests "
                "before using direction/confidence as a trading signal."
            ),
        },
    )


def _get_timeframe_defaults(timeframe: str) -> Dict[str, int]:
    """Get sensible defaults for regime detection based on timeframe.

    Higher frequency timeframes need more bars for meaningful analysis
    and higher min_regime_bars to avoid micro-noise.
    """
    tf = str(timeframe).strip().upper()
    return _TIMEFRAME_DEFAULTS.get(tf, {"lookback": 300, "min_regime_bars": 5})


@mcp.tool()
def regime_detect(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    fetch_limit: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    method: Literal[
        "bocpd",
        "pelt",
        "hmm",
        "gmm",
        "ms_ar",
        "clustering",
        "garch",
        "rule_based",
        "wavelet",
        "ensemble",
        "all",
    ] = "rule_based",  # type: ignore
    target: Literal["return", "price"] = "return",  # type: ignore
    params: Optional[Dict[str, Any]] = None,
    denoise: DenoiseSpecInput = None,
    threshold: Optional[float] = None,
    detail: DetailLiteral = "compact",
    lookback: Annotated[Optional[int], Field(ge=1)] = None,
    include_series: bool = False,
    min_regime_bars: Optional[int] = None,
    max_regimes: Annotated[int, Field(ge=1)] = 10,  # Compact/standard segment cap
) -> Dict[str, Any]:
    """Detect regimes and/or change-points over a bounded history window.

    - fetch_limit: Optional bars to fetch/analyze. Negative or method-insufficient
      values are rejected before fetch (`cli_invalid_arguments`). For recent-history
      requests, omission tracks the effective lookback plus warmup bars. An explicit
      start/end range is analyzed in full unless fetch_limit or lookback is supplied.
      For rule_based, an explicit fetch_limit also becomes params.window_bars when
      that parameter and lookback are omitted; at least 20 bars are required.
      Other methods require at least 10 bars.
    - start/end: Optional UTC-compatible analysis window. If provided, an explicit
      `fetch_limit` or `lookback` caps bars analyzed after the window is fetched.
    - method: Default is 'rule_based' (fast trend/ranging/transition classification).
      Other options: 'bocpd' (Bayesian online change-point; Gaussian), 'pelt' (offline penalized change-point segmentation), 'hmm' (Gaussian hidden Markov model), 'gmm' (i.i.d. Gaussian mixture),
      'ms_ar' (Markov-switching AR), 'clustering' (rolling-feature clustering via tsfresh + KMeans/Spectral),
      'garch' (GARCH conditional-volatility tiers),
      'wavelet' (multi-resolution wavelet energy regime detection via PyWavelets),
      'ensemble' (consensus across multiple methods), 'all' (runs all methods for comparison, may be slow).
    - params (clustering): optional `algorithm` = 'kmeans' (default) | 'spectral' (sklearn SpectralClustering).
      Optional `affinity` for spectral (default 'nearest_neighbors').
    - params (wavelet): optional `wavelet` (default 'db4'), `level` (auto), `n_states` (default 3),
      `energy_window` (default 30 bars).
    - params (ensemble): optional `methods` list (default ['hmm', 'clustering', 'wavelet']),
      `voting` = 'soft' (probability averaging, default) | 'hard' (majority vote).
    - params (bocpd): optional `hazard_mode` = auto_default|auto_calibrated (defaults to auto_calibrated).
      Explicit `hazard_lambda` / `cp_threshold` always take precedence over auto selection.
      The top-level `threshold` defaults to None for automatic calibration;
      any numeric value, including 0.5, is a fixed cutoff.
      Optional robustness params:
        `cp_threshold_calibration_mode` (default `walkforward_quantile`),
        `threshold_target_false_alarm_rate`,
        `cp_confirm_bars` (default `1`, live-oriented),
        `min_cp_distance_bars`, `cp_edge_multiplier`.
    - include_series: If True, include raw time series data (probs, states) in output. Default False.
    - lookback: Number of recent observations to analyze when `fetch_limit` is omitted,
      and the summary window when `fetch_limit` is provided. Extra history may be fetched
      for feature warmup but is excluded from model fitting. Omit for timeframe-based defaults:
        M1: 3000, M5: 2000, M15: 1000, M30: 800, H1: 500, H2: 400, H4: 300, H6-H12: 200-150, D1: 200, W1: 100, MN1: 48
    - min_regime_bars: Confirm a new state only after it persists for this many
        consecutive bars. Confirmation is causal and never rewrites earlier labels.
        Omit for timeframe-based defaults: M1: 30, M5: 12, M15-M30: 6-8, H1-H4: 3-4, D1+: 2
    - max_regimes: Maximum recent regime segment rows in compact and standard
        output (default 10). Compact keeps current_regime plus those last N
        rows. Full mode shows all available windows.
    - detail:
        Compact output is the public default. Use `detail="full"` for richer
        consolidated output. Raw `series` is included only if
        include_series=True.

    Output Structure (state-based methods: hmm, ms_ar, clustering, garch, wavelet, ensemble):
        - success: bool - Whether detection succeeded
        - symbol: str - Symbol analyzed
        - timeframe: str - Timeframe used
        - method: str - Method used
        - target: str - 'return' or 'price'
        - regimes: List[Dict] - Regime segments with start, end, bars, regime ID, label, regime_confidence
        - regime_info: Dict - Descriptive info for each regime (label, mean_return, volatility, etc.)
        - summary: Dict - Quick stats including last_state, state_shares, transitions, smoothing status
        - state_probabilities: List[List[float]] - Probability of each regime at each bar (full output only)
        - reliability: Dict - Confidence score and source (method-dependent)
        - params_used: Dict - Parameters actually used
        - warnings: List[str] - Any warnings (optional)

    Method-Specific Notes:
        - 'bocpd': Returns transition-oriented compact/full output:
          `current_regime`, `transition_summary`, `regime_context`, and `regimes`.
          These describe whether a new change point has been confirmed, how long the
          current segment has persisted, and derived bias/volatility context from the
          target series. Raw `cp_prob` and `change_points` remain available in `series`
          when include_series=True. Reliability is based on calibration quality.
          Best for detecting transition timing.
        - 'hmm', 'ms_ar', 'clustering': Return 'state' array and 'state_probabilities'.
          HMM/MS-AR probabilities are model posteriors before causal state-change
          confirmation, so their argmax can temporarily differ from emitted state.
          Labels like 'positive_low_vol' describe regime characteristics (return + volatility).
          Reliability based on model fit or cluster separation.
        - 'garch': Fits GARCH(1,1), then classifies its conditional-volatility
          path into relative full-window percentile tiers (or an explicit
          absolute threshold for two states). This is not a switching-GARCH model.
          n_states is AUTO-DETECTED by default from realized-vol percentile spread
          (vol_ratio_90_10) plus raw return kurtosis:
            vol_ratio > 10 or kurtosis > 6 → 4 states
            vol_ratio > 5 or kurtosis > 4 → 3 states
            otherwise → 2 states (10 or fewer usable volatility observations defaults to 3)
          Explicit n_states parameter overrides auto-detection.
          Uses percentile-based classification with volatility characteristics reported in output.
        - 'rule_based': Returns an aggregate-window `current_regime` classification
          with state (trending/ranging/transition), direction
          (bullish/bearish/neutral), trend_strength, and efficiency_ratio.
          `classification_window` identifies the exact start, end, and bar count.
          This method does not estimate onset, persistence, historical segments, or
          a per-bar state series, so those fields are omitted.
          Trend metrics use the recent price window so direction/window_move_pct stay coherent
          even when target='return'. Best for quick trend classification.
        - 'wavelet': Returns 'regime_params' with 'energy_profiles' showing frequency distribution.
          Best for detecting regimes at different time scales.
        - 'ensemble': Consensus across multiple methods with heuristic n_states selection.
          Default voters are HMM, clustering, and wavelet. Only state methods
          whose IDs are canonicalized by return are accepted; change-point,
          rule-based, and GARCH volatility-tier methods cannot vote.
          For ensemble only, omitted n_states is selected by raw
          return-distribution kurtosis:
            kurtosis > 6.0 → 6 states
            kurtosis > 4.5 → 5 states
            kurtosis > 3.5 → 4 states
            kurtosis ≤ 3.5 → 3 states
          Labels are derived from observed return sign and volatility tiers so they stay aligned
          with regime statistics. 'ensemble_info' shows voting method and mean_agreement.
          Explicit n_states overrides auto-detection.
        - 'all': Returns a cross-method 'comparison' dict with semantic agreement metrics.
          Compact output keeps the comparison view concise; `detail='full'`
          includes richer per-method outputs.
          Best for method comparison.
    """
    requested_method = str(method).strip().lower()
    method = _normalize_regime_method_name(requested_method)
    requested_target = str(target).strip().lower()
    started_at = time.perf_counter()
    global_warnings: List[str] = []
    if method == "rule_based" and requested_target != "price":
        target = "price"
        global_warnings.append(
            "rule_based uses price-path efficiency and trend metrics; "
            "requested target='return' was normalized to target='price'."
        )
    symbol_input: Optional[str] = None
    analysis_window_meta: Dict[str, Any] = {}
    freshness_meta: Dict[str, Any] = {}
    log_operation_start(
        logger,
        operation="regime_detect",
        symbol=symbol,
        timeframe=timeframe,
        method=requested_method,
        target=target,
        detail=detail,
        fetch_limit=fetch_limit,
        start=start,
        end=end,
    )

    def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(result, dict) and "error" not in result:
            result = attach_completed_bar_input_policy(result)
            for key, value in freshness_meta.items():
                result.setdefault(key, value)
            if requested_method != method:
                result.setdefault("requested_method", requested_method)
                result.setdefault("method_effective", method)
                result.setdefault(
                    "method_note",
                    f"Requested method '{requested_method}' is handled by the '{method}' implementation.",
                )
            if requested_target != target:
                result.setdefault("requested_target", requested_target)
                result["target"] = target
                result.setdefault("effective_target", target)
            _append_warnings(result, global_warnings)
            if analysis_window_meta:
                result.setdefault("analysis_window", dict(analysis_window_meta))
            result.setdefault("timezone", "UTC")
            result["symbol"] = symbol
            if symbol_input is not None:
                result["symbol_input"] = symbol_input
            _attach_regime_usage_notice(result)
            result = attach_mt5_source(result)
        log_operation_finish(
            logger,
            operation="regime_detect",
            started_at=started_at,
            success=infer_result_success(result),
            symbol=symbol,
            timeframe=timeframe,
            method=requested_method,
            target=target,
            detail=detail,
            fetch_limit=fetch_limit,
            start=start,
            end=end,
        )
        return result

    output = normalize_output_detail(detail)
    verbosity_output = normalize_output_verbosity_detail(detail)
    try:
        lookback = None if lookback is None else int(lookback)
    except (TypeError, ValueError):
        return _finish({"error": "lookback must be an integer >= 1 when provided."})
    try:
        min_regime_bars = (
            None if min_regime_bars is None else int(min_regime_bars)
        )
    except (TypeError, ValueError):
        return _finish({"error": "min_regime_bars must be an integer >= 1 when provided."})
    if lookback is not None and lookback < 1:
        return _finish({"error": "lookback must be >= 1 when provided."})
    if min_regime_bars is not None and min_regime_bars < 1:
        return _finish({"error": "min_regime_bars must be >= 1 when provided."})
    if method == "bocpd":
        threshold_candidates = (
            ("params.cp_threshold", (params or {}).get("cp_threshold")),
            ("params.threshold", (params or {}).get("threshold")),
            ("threshold", threshold),
        )
        for threshold_name, threshold_value in threshold_candidates:
            if threshold_value is None:
                continue
            try:
                threshold_float = float(threshold_value)
            except (TypeError, ValueError):
                return _finish(
                    {"error": f"{threshold_name} must be a probability between 0 and 1."}
                )
            if not math.isfinite(threshold_float) or not 0.0 <= threshold_float <= 1.0:
                return _finish(
                    {"error": f"{threshold_name} must be a probability between 0 and 1."}
                )
    range_error = validate_historical_range(start, end)
    if range_error is not None:
        return _finish(range_error)
    gateway = create_mt5_gateway(
        ensure_connection_impl=ensure_mt5_connection_or_raise,
    )
    connection_error = mt5_connection_error(gateway)
    if connection_error is not None:
        return _finish(connection_error)
    symbol, symbol_input = resolve_public_symbol(symbol, gateway=gateway)
    try:
        p = dict(params or {})
        parameter_error = unknown_mapping_keys_error(
            p,
            _regime_param_keys(method),
            subject=f"regime params for method '{method}'",
        )
        if parameter_error is not None:
            return _finish(parameter_error)
        requested_lookback = -1 if lookback is None else int(lookback)
        requested_min_regime_bars = -1 if min_regime_bars is None else int(min_regime_bars)

        # Apply timeframe-based defaults if not explicitly provided
        tf_defaults = _get_timeframe_defaults(timeframe)
        effective_lookback = int(lookback) if lookback is not None else tf_defaults["lookback"]
        effective_min_regime_bars = (
            int(min_regime_bars)
            if min_regime_bars is not None
            else tf_defaults["min_regime_bars"]
        )
        if fetch_limit is not None:
            try:
                fetch_limit_value = int(fetch_limit)
            except (TypeError, ValueError):
                return _finish(
                    _invalid_fetch_limit_error(
                        "fetch_limit must be an integer.",
                        fetch_limit=fetch_limit,
                        method=method,
                    )
                )
            min_fetch = _method_min_fetch_limit(method)
            if fetch_limit_value < 0:
                return _finish(
                    _invalid_fetch_limit_error(
                        "fetch_limit must be a positive integer; negative values "
                        "are not coerced to the default window.",
                        fetch_limit=fetch_limit_value,
                        method=method,
                    )
                )
            if fetch_limit_value < min_fetch:
                return _finish(
                    _invalid_fetch_limit_error(
                        f"fetch_limit must be >= {min_fetch} for method='{method}'.",
                        fetch_limit=fetch_limit_value,
                        method=method,
                    )
                )
            fetch_limit = fetch_limit_value

        lookback_mapped_to_window = False
        fetch_limit_mapped_to_window = False
        needs_rule_based_config = method in {"rule_based", "all"}
        if needs_rule_based_config and "window_bars" not in p:
            if lookback is not None:
                p["window_bars"] = int(effective_lookback)
                lookback_mapped_to_window = True
            elif fetch_limit is not None:
                p["window_bars"] = int(fetch_limit)
                fetch_limit_mapped_to_window = True

        min_regime_bars_val, min_regime_bars_error = _coerce_param(
            p,
            "min_regime_bars",
            default=effective_min_regime_bars,
            cast=int,
            error="min_regime_bars must be an integer >= 1.",
        )
        if min_regime_bars_error is not None:
            return _finish({"error": min_regime_bars_error})
        if min_regime_bars_val < 1:
            return _finish({"error": "min_regime_bars must be >= 1."})

        # Override lookback with effective value (will be used throughout function)
        lookback = p.get("lookback", effective_lookback)
        global_warnings.extend(
            _method_parameter_warnings(
                method,
                p,
                threshold=threshold,
                requested_lookback=int(requested_lookback),
                requested_min_regime_bars=int(requested_min_regime_bars),
                include_series=bool(include_series),
                max_regimes=int(max_regimes),
                output=output,
                lookback_mapped_to_window=lookback_mapped_to_window,
            )
        )

        rule_based_config: Optional[Dict[str, Any]] = None
        effective_fetch_limit = _history_fetch_limit(fetch_limit, lookback)
        full_explicit_range = bool(start or end) and fetch_limit is None and requested_lookback < 0
        if needs_rule_based_config:
            efficiency_threshold, efficiency_error = _coerce_param(
                p,
                "efficiency_threshold",
                default=0.35,
                cast=float,
                error="params.efficiency_threshold must be a positive number.",
            )
            if efficiency_error is not None:
                return _finish({"error": efficiency_error})
            if not np.isfinite(float(efficiency_threshold)) or float(efficiency_threshold) <= 0.0:
                return _finish({"error": "params.efficiency_threshold must be > 0."})

            trend_strength_threshold, trend_strength_error = _coerce_param(
                p,
                "trend_strength_threshold",
                default=1.25,
                cast=float,
                error="params.trend_strength_threshold must be a positive number.",
            )
            if trend_strength_error is not None:
                return _finish({"error": trend_strength_error})
            if (
                not np.isfinite(float(trend_strength_threshold))
                or float(trend_strength_threshold) <= 0.0
            ):
                return _finish({"error": "params.trend_strength_threshold must be > 0."})

            requested_window_bars, window_error = _coerce_param(
                p,
                "window_bars",
                default=160,
                cast=int,
                error="params.window_bars must be an integer >= 20.",
            )
            if window_error is not None:
                return _finish({"error": window_error})
            if int(requested_window_bars) < 20:
                if fetch_limit_mapped_to_window:
                    return _finish({
                        "error": (
                            "fetch_limit must be >= 20 for method='rule_based'; "
                            "increase the requested history window or choose another method."
                        )
                    })
                if lookback_mapped_to_window:
                    return _finish({
                        "error": (
                            "--lookback must be >= 20 for method='rule_based'; "
                            "increase the requested history window or choose another method."
                        )
                    })
                return _finish({"error": "params.window_bars must be >= 20."})
            if fetch_limit is not None and int(fetch_limit) < int(requested_window_bars):
                return _finish({
                    "error": (
                        f"fetch_limit ({int(fetch_limit)}) must be greater than or equal to "
                        f"params.window_bars ({int(requested_window_bars)}) for "
                        "method='rule_based'."
                    )
                })

            rule_based_config = {
                "efficiency_threshold": float(efficiency_threshold),
                "trend_strength_threshold": float(trend_strength_threshold),
                "window_bars": int(requested_window_bars),
            }
            effective_fetch_limit = (
                int(fetch_limit)
                if fetch_limit is not None
                else int(max(effective_fetch_limit, int(requested_window_bars)))
            )

        history_kwargs: Dict[str, Any] = {"as_of": None}
        if start or end:
            history_kwargs.update({"start": start, "end": end})
        df = _fetch_history(symbol, timeframe, effective_fetch_limit, **history_kwargs)
        if not start and not end and len(df) and "time" in df:
            freshness_meta = completed_bar_freshness_fields(
                symbol,
                timeframe,
                df["time"].iloc[-1],
                item="bar",
            )
            stale_warning = freshness_meta.get("stale_warning")
            if stale_warning and stale_warning not in global_warnings:
                global_warnings.append(str(stale_warning))
        fetched_range_bars = len(df)
        if (start or end) and not full_explicit_range and len(df) > effective_fetch_limit:
            df = df.iloc[-effective_fetch_limit:].reset_index(drop=True)
        if start or end:
            analysis_window_meta.update(
                {
                    "range_bars_fetched": int(fetched_range_bars),
                    "bars_analyzed": int(len(df)),
                    "truncated": bool(fetched_range_bars > len(df)),
                    "fetch_limit_applied": (
                        None if full_explicit_range else int(effective_fetch_limit)
                    ),
                }
            )
            if len(df) and "time" in df:
                analysis_window_meta["effective_start"] = _format_time_minimal(
                    float(df["time"].iloc[0])
                )
                analysis_window_meta["effective_end"] = _format_time_minimal(
                    float(df["time"].iloc[-1])
                )
        if len(df) < 10:
            return _finish({"error": "Insufficient history"})
        base_col = resolve_denoise_base_col(
            df, denoise, base_col="close"
        )
        y = df[base_col].astype(float).to_numpy()
        times = df["time"].astype(float).to_numpy()
        price_mask = np.isfinite(y)
        price_series = y[price_mask]
        price_times = times[price_mask]
        try:
            return_series = _log_returns_from_prices(y)
        except ValueError as exc:
            return _finish({"error": str(exc)})
        calibration_returns = return_series
        calibration_returns = calibration_returns[np.isfinite(calibration_returns)]
        if target == "return":
            x_raw = return_series
            return_mask = np.isfinite(x_raw)
            x = x_raw[return_mask]
            t = times[1:][return_mask]
        else:
            x = price_series
            t = times[price_mask]

        if x.size < 2:
            return _finish({"error": "Insufficient finite observations after filter"})

        if method == "rule_based":
            analysis_limit = int(rule_based_config["window_bars"])
            bars_analyzed = min(analysis_limit, int(price_series.size))
            warmup_bars = max(0, int(len(df)) - bars_analyzed)
        else:
            analysis_limit = (
                int(x.size)
                if full_explicit_range
                else int(fetch_limit)
                if fetch_limit is not None
                else int(lookback)
            )
            if fetch_limit is not None and requested_lookback >= 0:
                warning = (
                    "fit_window=fetch_limit; lookback used only for summary."
                )
                if warning not in global_warnings:
                    global_warnings.append(warning)
                observations_for_summary = max(0, int(x.size))
                if requested_lookback > observations_for_summary:
                    truncation_warning = (
                        "summary_window_truncated: requested lookback="
                        f"{int(requested_lookback)} but only "
                        f"{observations_for_summary} observations are available "
                        "after converting prices to returns; raise fetch_limit."
                    )
                    if truncation_warning not in global_warnings:
                        global_warnings.append(truncation_warning)
            observations_available = int(x.size)
            if observations_available > analysis_limit:
                x = x[-analysis_limit:]
                t = t[-analysis_limit:]
            calibration_returns = calibration_returns[-analysis_limit:]
            bars_analyzed = int(x.size)
            price_bars_used = bars_analyzed + (1 if target == "return" else 0)
            warmup_bars = max(0, int(len(df)) - price_bars_used)

        analysis_window_meta.update(
            {
                "bars_fetched": int(fetched_range_bars),
                "warmup_bars": int(warmup_bars),
                "bars_analyzed": int(bars_analyzed),
                "analysis_limit": int(analysis_limit),
            }
        )

        # format times
        t_fmt = [_format_time_minimal(tt) for tt in t]

        if method not in {"ensemble", "all"}:
            return _finish(_run_regime_method(
                method=method,
                symbol=symbol,
                timeframe=timeframe,
                target=target,
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
                calibration_returns=calibration_returns,
                price_series=price_series,
                price_times=price_times,
                rule_based_config=rule_based_config,
                global_warnings=global_warnings,
            ))

        if method == "ensemble":
            return _finish(_detect_ensemble(
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
                fetch_limit=fetch_limit,
                denoise=denoise,
                min_regime_bars=min_regime_bars,
            ))

        elif method == "all":
            return _finish(_detect_all(
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
                fetch_limit=fetch_limit,
                denoise=denoise,
                min_regime_bars=min_regime_bars,
                verbosity_output=verbosity_output,
                calibration_returns=calibration_returns,
                price_series=price_series,
                price_times=price_times,
                rule_based_config=rule_based_config,
                global_warnings=global_warnings,
            ))

    except Exception as e:
        return _finish({"error": f"Error detecting regimes: {str(e)}"})
