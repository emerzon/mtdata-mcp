"""Ensemble aggregation and comparison support for regime voting."""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .payload import _consolidate_payload
from .smoothing import (
    _canonicalize_regime_labels,
    _confirm_state_changes_causally,
)
from .summarize import (
    _append_warnings,
    _apply_state_output_mode,
    _common_reliability,
    _smoothing_warnings,
    _summary_window_size,
)

_ENSEMBLE_LEADING_PAD_MAX = 24
_ENSEMBLE_STATE_METHODS = frozenset(
    {"hmm", "gmm", "ms_ar", "clustering", "wavelet"}
)


def _finite_raw_kurtosis(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 3.0
    scale = float(np.max(np.abs(array)))
    if not np.isfinite(scale) or scale <= 1e-12:
        return 3.0
    scaled = array / scale
    std = float(np.std(scaled))
    if not np.isfinite(std) or std <= 1e-9:
        return 3.0
    with np.errstate(over="ignore", invalid="ignore"):
        standardized = (scaled - float(np.mean(scaled))) / std
        kurtosis = float(np.mean(standardized**4))
    return kurtosis if np.isfinite(kurtosis) else 3.0


def _align_states_to_return_centroids(
    states: np.ndarray,
    probabilities: np.ndarray,
    target_series: np.ndarray,
    target_centroids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
    """Map one voter's states into shared return-centroid bins."""
    state_array = np.asarray(states, dtype=int).reshape(-1)
    probability_array = np.asarray(probabilities, dtype=float)
    series_array = np.asarray(target_series, dtype=float).reshape(-1)
    centroids = np.asarray(target_centroids, dtype=float).reshape(-1)
    if (
        probability_array.ndim != 2
        or probability_array.shape[0] != state_array.size
        or series_array.size != state_array.size
        or probability_array.shape[1] < 1
        or centroids.size < 2
    ):
        raise ValueError("State probabilities cannot be aligned to ensemble centroids.")

    probability_array = np.nan_to_num(
        probability_array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    probability_array = np.clip(probability_array, 0.0, None)
    row_sums = np.sum(probability_array, axis=1, keepdims=True)
    positive_rows = row_sums[:, 0] > 0.0
    probability_array[positive_rows] /= row_sums[positive_rows]

    source_count = int(probability_array.shape[1])
    fallback_bins = np.rint(
        np.linspace(0, centroids.size - 1, source_count)
    ).astype(int)
    state_map: Dict[int, int] = {}
    finite_series = np.isfinite(series_array)
    for source_state in range(source_count):
        observations = series_array[
            (state_array == source_state) & finite_series
        ]
        source_centroid = (
            float(np.mean(observations))
            if observations.size
            else float(centroids[fallback_bins[source_state]])
        )
        state_map[source_state] = int(
            np.argmin(np.abs(centroids - source_centroid))
        )

    aligned_probabilities = np.zeros((state_array.size, centroids.size), dtype=float)
    for source_state, target_state in state_map.items():
        aligned_probabilities[:, target_state] += probability_array[:, source_state]

    valid_mask = (
        (state_array >= 0)
        & (state_array < source_count)
        & positive_rows
    )
    aligned_states = np.full(state_array.size, -1, dtype=int)
    for source_state, target_state in state_map.items():
        aligned_states[valid_mask & (state_array == source_state)] = target_state
    aligned_probabilities[~valid_mask] = 0.0
    return aligned_states, aligned_probabilities, valid_mask, state_map


def _ensemble_state_count_configuration(
    params: Dict[str, Any],
    values: np.ndarray,
) -> tuple[int, str, bool, Dict[str, Any], List[str], Optional[str]]:
    n_states_input = params.get("n_states")
    if n_states_input is not None:
        try:
            n_states = int(n_states_input)
        except Exception:
            return 0, "n_states", False, {}, [], "n_states must be an integer >= 2 for ensemble."
        if n_states < 2:
            return 0, "n_states", False, {}, [], "n_states must be >= 2 for ensemble."
        return n_states, "n_states", False, {}, [], None

    returns_kurt = _finite_raw_kurtosis(values)
    if returns_kurt > 6.0:
        n_states = 6
    elif returns_kurt > 4.5:
        n_states = 5
    elif returns_kurt > 3.5:
        n_states = 4
    else:
        n_states = 3
    heuristic = {
        "method": "return_kurtosis_thresholds",
        "returns_kurtosis": round(returns_kurt, 2),
    }
    warning = (
        "Ensemble n_states was selected by a return-kurtosis heuristic, "
        "not statistical model selection. Set params.n_states explicitly "
        "and validate it through backtesting when state count matters."
    )
    return (
        n_states,
        "return_kurtosis_heuristic",
        True,
        heuristic,
        [warning],
        None,
    )


def _aggregate_precomputed_ensemble(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    target: str,
    x: np.ndarray,
    t_fmt: List[Any],
    sub_results: List[Dict[str, Any]],
    sub_errors: List[str],
    requested_methods: List[str],
    voting: str,
    n_states_ens: int,
    n_states_source: str,
    ens_auto_n_states: bool,
    ens_auto_metrics: Dict[str, Any],
    state_count_warnings: List[str],
    min_regime_bars_val: int,
    output: str,
    lookback: int,
    include_series: bool,
    max_regimes: int,
    aggregation_source: str,
) -> Dict[str, Any]:
    """Aggregate already-fitted regime state series without refitting voters."""
    method = "ensemble"
    # Extract return-canonicalized state arrays from sub-results.
    state_arrays: List[np.ndarray] = []
    prob_arrays: List[np.ndarray] = []  # (n_bars, n_states) per method
    prob_valid_masks: List[np.ndarray] = []
    method_names: List[str] = []
    sub_method_state_counts: Dict[str, int] = {}
    sub_method_state_maps: Dict[str, Dict[str, int]] = {}
    ref_len = len(t_fmt)
    finite_target = x[np.isfinite(x)]
    target_quantiles = (
        np.arange(n_states_ens, dtype=float) + 0.5
    ) / float(n_states_ens)
    target_centroids = np.quantile(finite_target, target_quantiles)

    for sr_info in sub_results:
        sm_name = sr_info["method"]
        sr = sr_info["result"]
        series = sr.get("series", {})

        raw_state = series.get("state", sr.get("state", []))
        raw_probs = series.get(
            "state_probabilities", sr.get("state_probabilities", [])
        )
        if raw_state is None:
            sub_errors.append(
                f"{sm_name}: state series did not match the ensemble window"
            )
            continue
        try:
            state_len = len(raw_state)
        except TypeError:
            sub_errors.append(
                f"{sm_name}: state series did not match the ensemble window"
            )
            continue
        if state_len != ref_len:
            pad = ref_len - state_len
            if 0 < state_len < ref_len and pad <= _ENSEMBLE_LEADING_PAD_MAX:
                raw_state = [-1] * pad + list(raw_state)
                if raw_probs is not None:
                    try:
                        probs_len = len(raw_probs)
                    except TypeError:
                        probs_len = -1
                    if probs_len == state_len:
                        try:
                            width = len(raw_probs[0]) if raw_probs else 0
                        except TypeError:
                            width = 0
                        zero = [0.0] * width
                        raw_probs = [zero] * pad + list(raw_probs)
            else:
                sub_errors.append(
                    f"{sm_name}: state series did not match the ensemble window"
                )
                continue

        st = np.asarray(raw_state, dtype=int)
        if raw_probs is not None and len(raw_probs) == ref_len:
            pr = np.asarray(raw_probs, dtype=float)
            if pr.ndim != 2 or pr.shape[1] < 1:
                sub_errors.append(
                    f"{sm_name}: state probabilities were not a usable matrix"
                )
                continue
        else:
            reported_params = sr.get("regime_params", {})
            reported_count = 0
            if isinstance(reported_params, dict):
                for key in ("mean_return", "mu", "volatility", "sigma"):
                    values = reported_params.get(key)
                    if isinstance(values, (list, tuple, np.ndarray)):
                        reported_count = max(reported_count, len(values))
            occupied_count = int(np.max(st)) + 1 if np.any(st >= 0) else 0
            source_count = max(reported_count, occupied_count)
            if source_count < 1:
                sub_errors.append(
                    f"{sm_name}: no usable states or probabilities"
                )
                continue
            pr = np.zeros((ref_len, source_count))
            for i, s in enumerate(st):
                if 0 <= int(s) < source_count:
                    pr[i, s] = 1.0

        source_count = int(pr.shape[1])
        sub_method_state_counts[sm_name] = source_count
        try:
            st, pr, valid_mask, state_map = (
                _align_states_to_return_centroids(
                    st,
                    pr,
                    x,
                    target_centroids,
                )
            )
        except ValueError as exc:
            sub_errors.append(f"{sm_name}: {exc}")
            continue
        if not np.any(valid_mask):
            sub_errors.append(f"{sm_name}: no valid aligned state rows")
            continue
        state_arrays.append(st)
        prob_arrays.append(pr)
        prob_valid_masks.append(valid_mask)
        method_names.append(sm_name)
        sub_method_state_maps[sm_name] = {
            str(source): int(target)
            for source, target in state_map.items()
        }

    if not prob_arrays:
        return {"error": "No sub-methods produced usable state data."}

    # Aggregate
    if voting == "hard":
        # Majority vote over methods that have a valid state for the bar.
        ensemble_state = np.full(ref_len, -1, dtype=int)
        ensemble_probs = np.zeros((ref_len, n_states_ens))
        for t_idx in range(ref_len):
            votes = [
                int(state_arr[t_idx])
                for state_arr, valid_mask in zip(
                    state_arrays, prob_valid_masks
                )
                if bool(valid_mask[t_idx])
                and 0 <= int(state_arr[t_idx]) < n_states_ens
            ]
            if not votes:
                continue
            counts = Counter(votes)
            majority, _count = counts.most_common(1)[0]
            ensemble_state[t_idx] = int(majority)
            for state_id, count in counts.items():
                ensemble_probs[t_idx, int(state_id)] = float(count) / float(
                    len(votes)
                )
    else:
        # Soft voting: average probabilities across valid methods per bar.
        ensemble_probs = np.zeros((ref_len, n_states_ens))
        valid_counts = np.zeros(ref_len, dtype=float)
        for pr, valid_mask in zip(prob_arrays, prob_valid_masks):
            rows = valid_mask & (np.sum(pr, axis=1) > 0)
            if not np.any(rows):
                continue
            ensemble_probs[rows] += pr[rows]
            valid_counts[rows] += 1.0
        valid_rows = valid_counts > 0
        if np.any(valid_rows):
            ensemble_probs[valid_rows] = (
                ensemble_probs[valid_rows] / valid_counts[valid_rows, None]
            )
        ensemble_state = np.full(ref_len, -1, dtype=int)
        ensemble_state[valid_rows] = np.argmax(
            ensemble_probs[valid_rows],
            axis=1,
        ).astype(int)

    # Smooth and canonicalize
    valid_ensemble_mask = (ensemble_state >= 0) & (
        np.sum(ensemble_probs, axis=1) > 0
    )
    if not np.any(valid_ensemble_mask):
        return {"error": "No valid ensemble state rows after voting."}

    valid_state, smoothing_meta = _confirm_state_changes_causally(
        ensemble_state[valid_ensemble_mask], min_regime_bars_val
    )
    valid_probs = ensemble_probs[valid_ensemble_mask]
    valid_state, valid_probs, canon_meta = _canonicalize_regime_labels(
        valid_state,
        valid_probs,
        x[valid_ensemble_mask],
    )
    ensemble_state = np.full(ref_len, -1, dtype=int)
    ensemble_probs = np.zeros((ref_len, n_states_ens))
    ensemble_state[valid_ensemble_mask] = valid_state
    ensemble_probs[valid_ensemble_mask] = valid_probs
    smoothing_meta["relabeled"] = canon_meta.get("relabeled", False)

    # Agreement score: fraction of methods that agree per bar
    agreement = np.zeros(ref_len)
    for t_idx in range(ref_len):
        votes = [
            int(state_arr[t_idx])
            for state_arr, valid_mask in zip(state_arrays, prob_valid_masks)
            if bool(valid_mask[t_idx])
            and 0 <= int(state_arr[t_idx]) < n_states_ens
        ]
        if votes:
            most_common = max(set(votes), key=votes.count)
            agreement[t_idx] = votes.count(most_common) / len(votes)

    # Compute regime parameters (mean, vol) for each ensemble state
    mean_agreement = round(
        float(np.mean(agreement[valid_ensemble_mask])),
        4,
    )
    ensemble_regime_params = {
        "mean_return": [],
        "volatility": [],
    }
    for s in range(n_states_ens):
        mask = ensemble_state == s
        if mask.any():
            ensemble_regime_params["mean_return"].append(
                float(np.mean(x[mask]))
            )
            ensemble_regime_params["volatility"].append(float(np.std(x[mask])))
        else:
            ensemble_regime_params["mean_return"].append(0.0)
            ensemble_regime_params["volatility"].append(0.0)

    voter_errors: Dict[str, str] = {}
    for error_text in sub_errors:
        voter_name, separator, reason = str(error_text).partition(":")
        voter_key = voter_name.strip()
        if separator and voter_key:
            voter_errors[voter_key] = reason.strip()
    excluded_methods = [
        voter for voter in requested_methods if voter not in method_names
    ]
    ensemble_health = {
        "requested_voters": list(requested_methods),
        "used_voters": list(method_names),
        "excluded_voters": excluded_methods,
        "voter_errors": voter_errors,
        "degraded": bool(excluded_methods or sub_errors),
        "aggregation_source": aggregation_source,
    }

    payload = {
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "method": method,
        "target": target,
        "times": t_fmt,
        "state": [int(s) for s in ensemble_state.tolist()],
        "state_probabilities": [
            [float(v) for v in row] for row in ensemble_probs.tolist()
        ],
        "regime_params": ensemble_regime_params,
        "ensemble_health": ensemble_health,
        "ensemble_info": {
            "sub_methods": method_names,
            "requested_methods": list(requested_methods),
            "excluded_methods": excluded_methods,
            "aggregation_source": aggregation_source,
            "voting": voting,
            "mean_agreement": mean_agreement,
            "alignment_mode": "return_quantile_centroids",
            "shared_state_centroids": [
                float(value) for value in target_centroids.tolist()
            ],
            "sub_method_state_counts": sub_method_state_counts,
            "sub_method_state_maps": sub_method_state_maps,
        },
        "params_used": {
            "methods": method_names,
            "voting": voting,
            "n_states": n_states_ens,
            "state_count_param": n_states_source,
            "n_states_auto": bool(ens_auto_n_states),
            "n_methods_succeeded": len(method_names),
            "min_regime_bars": int(min_regime_bars_val),
            "smoothing_applied": smoothing_meta.get("smoothing_applied", False),
        },
    }
    if ens_auto_metrics:
        payload["params_used"]["state_count_heuristic"] = ens_auto_metrics
        payload["auto_detection"] = ens_auto_metrics
    if sub_errors:
        payload["warnings"] = [f"Sub-method errors: {'; '.join(sub_errors)}"]
    _append_warnings(payload, state_count_warnings)
    _append_warnings(payload, _smoothing_warnings(method, smoothing_meta))
    payload["reliability"] = _common_reliability(
        {
            "confidence": mean_agreement,
            "methods_considered": method_names,
        },
        source="ensemble_return_centroid_agreement",
    )

    if output in ("summary", "compact"):
        n_summary = _summary_window_size(lookback, len(ensemble_state))
        st_tail = (
            ensemble_state[-n_summary:] if n_summary > 0 else ensemble_state
        )
        st_tail_valid = st_tail[st_tail >= 0]
        unique, counts = np.unique(st_tail_valid, return_counts=True)
        shares = {
            int(k): float(c) / float(len(st_tail_valid) or 1)
            for k, c in zip(unique, counts)
        }
        summary = {
            "lookback": int(n_summary),
            "last_state": int(ensemble_state[-1])
            if len(ensemble_state)
            else None,
            "state_shares": shares,
            "mean_agreement": mean_agreement,
        }
        payload = _apply_state_output_mode(
            payload,
            output=output,
            lookback=lookback,
            summary=summary,
        )
        if output == "summary":
            payload["ensemble_health"] = ensemble_health
            return payload

    result = _consolidate_payload(
        payload,
        method,
        output,
        include_series=include_series,
        max_regimes=max_regimes,
    )
    result["ensemble_health"] = ensemble_health
    return result
