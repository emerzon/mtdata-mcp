"""Payload consolidation utilities for regime detection output formatting."""

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from ...utils.coercion import coerce_finite_float as _finite_float
from .smoothing import _canonicalize_regime_labels

# Direction label requires mean return to be at least this many standard errors
# away from zero. Aligns BOCPD segment labeling with the PELT gate so that a
# stationary-return window is no longer branded trending_down just because the
# cumulative return happened to land slightly negative over a long window.
_DIRECTION_T_STAT_THRESHOLD = 1.96


def _mean_return_t_stat(
    mu: float,
    sigma: Optional[float],
    samples: Optional[int],
) -> Optional[float]:
    if sigma is None or samples is None or int(samples) < 2:
        return None
    sigma_value = float(sigma)
    if not np.isfinite(sigma_value) or sigma_value <= 1e-12:
        return None
    return float(mu) / (sigma_value / np.sqrt(float(samples)))


def _return_level_from_mean(
    mu: float,
    *,
    sigma: Optional[float] = None,
    samples: Optional[int] = None,
) -> str:
    """Map a state mean to direction using its sampling uncertainty."""
    t_stat = _mean_return_t_stat(mu, sigma, samples)
    if t_stat is None or abs(t_stat) < _DIRECTION_T_STAT_THRESHOLD:
        return "neutral"
    return "positive" if mu > 0 else "negative"


def _relative_volatility_levels(sigmas: Any) -> Dict[int, str]:
    """Return state-indexed volatility tiers relative to one fitted run."""
    finite: List[tuple[int, float]] = []
    if isinstance(sigmas, (list, tuple, np.ndarray)):
        for index, value in enumerate(sigmas):
            parsed = _finite_float(value)
            if parsed is not None:
                finite.append((index, parsed))
    unique = sorted({value for _, value in finite})
    if not unique:
        return {}
    if len(unique) == 1:
        tier_by_value = {unique[0]: "mod_vol"}
    elif len(unique) == 2:
        tier_by_value = {unique[0]: "low_vol", unique[1]: "high_vol"}
    elif len(unique) == 3:
        tier_by_value = {
            unique[0]: "low_vol",
            unique[1]: "mod_vol",
            unique[2]: "high_vol",
        }
    else:
        tier_by_value = {}
        denominator = len(unique) - 1
        for rank, value in enumerate(unique):
            percentile = rank / denominator
            if percentile <= 0.2:
                tier = "very_low_vol"
            elif percentile <= 0.45:
                tier = "low_vol"
            elif percentile <= 0.75:
                tier = "mod_vol"
            else:
                tier = "high_vol"
            tier_by_value[value] = tier
    return {index: tier_by_value[value] for index, value in finite}


def _trader_hmm_label(
    mu: float,
    volatility: str,
    *,
    sigma: Optional[float] = None,
    samples: Optional[int] = None,
) -> str:
    direction = _return_level_from_mean(mu, sigma=sigma, samples=samples)
    if volatility in {"very_low_vol", "low_vol"}:
        tempo = "quiet"
    elif volatility == "mod_vol":
        tempo = "steady"
    else:
        tempo = "volatile"

    if direction == "positive":
        return f"bullish_{tempo}"
    if direction == "negative":
        return f"bearish_{tempo}"
    if tempo == "quiet":
        return "quiet_range"
    if tempo == "volatile":
        return "volatile_range"
    return "range"


def _hmm_trading_interpretation(label: str) -> str:
    direction, _, tempo = label.partition("_")
    if label == "range":
        return "Balanced returns with moderate volatility; mean-reversion setups may fit better than breakout trades."
    if label == "quiet_range":
        return "Balanced returns with low volatility; expect muted movement until a breakout catalyst appears."
    if label == "volatile_range":
        return "Directionless but volatile regime; reduce size or require stronger confirmation."
    if direction == "bullish":
        if tempo == "quiet":
            return "Low-volatility upward drift; favor cautious trend-following or pullback entries."
        if tempo == "steady":
            return "Constructive upward regime with normal volatility; trend-following conditions are supportive."
        return "Upward bias with elevated volatility; trend may persist but risk controls matter."
    if direction == "bearish":
        if tempo == "quiet":
            return "Low-volatility downward drift; avoid assuming strong trend momentum without confirmation."
        if tempo == "steady":
            return "Constructive downward regime with normal volatility; bearish trend strategies are better aligned."
        return "Downward bias with elevated volatility; defensive positioning and tighter risk limits are warranted."
    return "Statistical regime with mixed directional evidence; inspect mean return and volatility before acting."


def _wavelet_volatility_label(level: Optional[str]) -> Optional[str]:
    """Return a comparison-friendly volatility tier for wavelet labels."""
    if level is None:
        return None
    if level == "mod_vol":
        return "moderate_vol"
    return level


def _wavelet_energy_weights(energy_profile: Any) -> List[float]:
    """Extract normalized wavelet band weights in band order."""
    if not isinstance(energy_profile, dict):
        return []

    ordered_pairs: List[tuple[int, float]] = []
    for key, value in energy_profile.items():
        text = str(key).strip().lower()
        if not text.startswith("band_") or not text.endswith("_energy"):
            continue
        idx_text = text[len("band_") : -len("_energy")]
        try:
            band_idx = int(idx_text)
            band_value = float(value)
        except (TypeError, ValueError):
            continue
        if band_idx < 0 or not np.isfinite(band_value) or band_value < 0:
            continue
        ordered_pairs.append((band_idx, band_value))

    if not ordered_pairs:
        return []

    ordered_pairs.sort(key=lambda item: item[0])
    weights = [value for _, value in ordered_pairs]
    total = float(sum(weights))
    if total <= 0:
        return []
    return [value / total for value in weights]


def _wavelet_frequency_character(energy_profile: Any) -> Optional[str]:
    """Describe whether a wavelet state is noise-, mixed-, or trend-dominant."""
    weights = _wavelet_energy_weights(energy_profile)
    if not weights:
        return None

    if len(weights) == 1:
        return "mixed_freq"

    max_index = len(weights) - 1
    centroid = sum(index * weight for index, weight in enumerate(weights)) / max_index
    if centroid <= 0.3:
        return "noise_dominant"
    if centroid >= 0.7:
        return "trend_dominant"
    return "mixed_freq"


def _wavelet_concentration_character(energy_profile: Any) -> Optional[str]:
    """Describe whether the wavelet energy is concentrated or diffuse."""
    weights = _wavelet_energy_weights(energy_profile)
    if not weights:
        return None

    top_share = max(weights)
    if top_share >= 0.55:
        return "concentrated"
    if top_share <= 0.4:
        return "diffuse"
    return None


def _compose_wavelet_label(
    *,
    mean_return: Optional[float],
    return_sigma: Optional[float],
    samples: Optional[int],
    volatility_level: Optional[str],
    energy_profile: Any,
    target: str,
    include_concentration: bool,
) -> Optional[str]:
    """Build a descriptive wavelet label from return, energy shape, and volatility."""
    parts: List[str] = []
    if target != "price" and mean_return is not None:
        parts.append(
            _return_level_from_mean(
                float(mean_return), sigma=return_sigma, samples=samples
            )
        )

    frequency_character = _wavelet_frequency_character(energy_profile)
    if include_concentration:
        concentration = _wavelet_concentration_character(energy_profile)
        if concentration is not None:
            parts.append(concentration)
    if frequency_character is not None:
        parts.append(frequency_character)

    vol_level = _wavelet_volatility_label(volatility_level)
    if vol_level is not None:
        parts.append(vol_level)

    if not parts:
        return None
    return "_".join(parts)


def _build_wavelet_labels(
    regime_params: Dict[str, Any],
    *,
    target: str,
    indexed_states: List[tuple[int, int]],
    state_counts: Optional[Dict[int, int]] = None,
) -> Dict[int, str]:
    """Build wavelet labels with volatility relative to the fitted states."""
    mu_list = regime_params.get("mu") or regime_params.get("mean_return") or []
    sigma_list = regime_params.get("sigma") or regime_params.get("volatility") or []
    energy_profiles = regime_params.get("energy_profiles")
    if not isinstance(energy_profiles, dict):
        energy_profiles = {}
    volatility_levels = _relative_volatility_levels(sigma_list)

    labels: Dict[int, str] = {}
    label_counts: Dict[str, int] = {}
    for source_state, regime_id in indexed_states:
        mean_return = (
            float(mu_list[source_state]) if source_state < len(mu_list) else None
        )
        label = _compose_wavelet_label(
            mean_return=mean_return,
            return_sigma=(
                float(sigma_list[source_state])
                if source_state < len(sigma_list)
                else None
            ),
            samples=(state_counts or {}).get(regime_id),
            volatility_level=volatility_levels.get(source_state),
            energy_profile=energy_profiles.get(str(source_state)),
            target=target,
            include_concentration=False,
        )
        if label is None:
            continue
        labels[regime_id] = label
        label_counts[label] = label_counts.get(label, 0) + 1

    for source_state, regime_id in indexed_states:
        base_label = labels.get(regime_id)
        if base_label is None or label_counts.get(base_label, 0) <= 1:
            continue
        mean_return = (
            float(mu_list[source_state]) if source_state < len(mu_list) else None
        )
        refined_label = _compose_wavelet_label(
            mean_return=mean_return,
            return_sigma=(
                float(sigma_list[source_state])
                if source_state < len(sigma_list)
                else None
            ),
            samples=(state_counts or {}).get(regime_id),
            volatility_level=volatility_levels.get(source_state),
            energy_profile=energy_profiles.get(str(source_state)),
            target=target,
            include_concentration=True,
        )
        if refined_label is not None:
            labels[regime_id] = refined_label

    return labels


def _bocpd_transition_risk(last_probability: Any, threshold: Any) -> str:
    """Bucket the latest BOCPD change-point probability into a risk label."""
    prob = _finite_float(last_probability)
    threshold_value = _finite_float(threshold)

    if prob is None:
        return "unknown"
    if threshold_value is not None and threshold_value > 0:
        ratio = prob / max(threshold_value, 1e-9)
        if ratio >= 1.0:
            return "high"
        if ratio >= 0.6:
            return "moderate"
        return "low"

    if prob >= 0.7:
        return "high"
    if prob >= 0.35:
        return "moderate"
    return "low"


def _bocpd_transition_activity(
    change_points_count: Any,
    recent_cp_density: Any,
) -> str:
    """Summarize recent change-point activity in human terms."""
    try:
        count = max(0, int(change_points_count))
    except (TypeError, ValueError):
        count = 0

    density = _finite_float(recent_cp_density)
    if count == 0:
        return "none"
    if density is not None:
        if density >= 0.05:
            return "elevated"
        if density >= 0.02:
            return "active"
    if count == 1:
        return "isolated"
    if count >= 3:
        return "elevated"
    return "active"


def _bocpd_calibration_status(reliability: Dict[str, Any]) -> str:
    """Map BOCPD calibration diagnostics to a simple status string."""
    if bool(reliability.get("threshold_calibrated", False)):
        return "calibrated"

    age = _finite_float(reliability.get("calibration_age_bars"))
    if age is not None and age > 0:
        return "estimated"
    return "unavailable"


def _bocpd_segment_status(
    *,
    change_points_count: Any,
    bars_since_change: Any,
    transition_risk: str,
) -> str:
    """Describe the current BOCPD segment in transition-oriented language."""
    try:
        count = max(0, int(change_points_count))
    except (TypeError, ValueError):
        count = 0

    try:
        bars = max(0, int(bars_since_change))
    except (TypeError, ValueError):
        bars = 0

    if count == 0:
        return "no_recent_change_detected"
    if transition_risk in {"moderate", "high"}:
        return "transition_watch"
    if bars <= 3:
        return "recent_change_detected"
    return "post_change_segment"


def _canonical_regime_label(label: Any, *, method: str) -> Optional[str]:
    text = str(label or "").strip().lower()
    if not text:
        return None
    if method == "bocpd":
        if text in {"stable", "no_recent_change_detected"}:
            return "ranging"
        if "bullish" in text:
            return "trending_up"
        if "bearish" in text:
            return "trending_down"
        if "change" in text or "transition" in text:
            return "transition"
    if text in {"ranging", "range", "quiet_range", "stable"}:
        return "ranging"
    if text in {"trending_up", "bullish", "bullish_segment"} or text.startswith("bullish_"):
        return "trending_up"
    if text in {"trending_down", "bearish", "bearish_segment"} or text.startswith("bearish_"):
        return "trending_down"
    if text in {"trending", "volatile_trend"}:
        return "trending"
    if "transition" in text or "change" in text:
        return "transition"
    if "high_vol" in text or "very_high_vol" in text:
        return "high_volatility"
    if "low_vol" in text or "very_low_vol" in text or "quiet" in text:
        return "low_volatility"
    return text


def _attach_label_fields(row: Dict[str, Any], *, method: str) -> None:
    label = row.get("label")
    if label in (None, ""):
        return
    row.setdefault("state_label_native", label)
    canonical = _canonical_regime_label(label, method=method)
    if canonical:
        row.setdefault("state_label_canonical", canonical)


def _build_bocpd_segment_context(
    series_values: np.ndarray,
    *,
    target: Optional[str],
) -> Dict[str, Any]:
    """Derive contextual statistics for a BOCPD segment from the target series.

    Direction labels (``bullish`` / ``bearish``) are gated on a t-statistic of
    at least ``_DIRECTION_T_STAT_THRESHOLD`` so a long, stationary window is
    no longer branded trending just because the cumulative return happened to
    drift slightly negative or positive. Statistical insignificant segments
    fall back to ``flat``.
    """
    values = np.asarray(series_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}

    target_name = str(target or "return").strip().lower()
    if target_name == "price":
        base = float(values[0]) if abs(float(values[0])) > 1e-9 else 1e-9
        return_pct = float((values[-1] - values[0]) / base * 100.0)
        if values.size > 1:
            safe_prices = np.maximum(values, 1e-12)
            log_returns = np.diff(np.log(safe_prices))
            log_returns = log_returns[np.isfinite(log_returns)]
            volatility_pct = (
                float(np.std(log_returns) * 100.0) if log_returns.size else 0.0
            )
        else:
            volatility_pct = 0.0
        mean_for_tstat = float(np.mean(log_returns)) if values.size > 1 and log_returns.size else 0.0
        std_for_tstat = float(np.std(log_returns, ddof=1)) if log_returns.size > 1 else 0.0
        n_for_tstat = int(log_returns.size) if values.size > 1 else 0
    else:
        total_log_return = float(np.sum(values))
        return_pct = float(np.expm1(total_log_return) * 100.0)
        volatility_pct = float(np.std(values) * 100.0) if values.size > 1 else 0.0
        mean_for_tstat = float(np.mean(values))
        std_for_tstat = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        n_for_tstat = int(values.size)

    mean_t_stat: Optional[float] = None
    if n_for_tstat >= 2 and np.isfinite(std_for_tstat) and std_for_tstat > 1e-12:
        mean_t_stat = float(mean_for_tstat / (std_for_tstat / np.sqrt(n_for_tstat)))

    direction_significant = bool(
        mean_t_stat is not None and abs(mean_t_stat) >= _DIRECTION_T_STAT_THRESHOLD
    )
    if not direction_significant:
        bias = "flat"
    elif return_pct > 0:
        bias = "bullish"
    else:
        bias = "bearish"

    context = {
        "source": f"derived_from_{target_name}_series",
        "bias": bias,
        "return_pct": round(return_pct, 4),
        "volatility_pct": round(volatility_pct, 4),
    }
    if mean_t_stat is not None and np.isfinite(mean_t_stat):
        context["mean_t_stat"] = round(float(mean_t_stat), 4)
        context["direction_significant"] = bool(direction_significant)
    return context


def _interpret_regime_label(
    regime_id: int,
    mu: Optional[float],
    sigma: Optional[float],
    method: str,
    total_states: int,
    target: Optional[str] = "return",
    volatility_level: Optional[str] = None,
    samples: Optional[int] = None,
) -> str:
    """Generate human-readable regime label from statistical parameters.

    For HMM/MS-AR with mean/volatility: creates descriptive labels like
    'low_return_low_vol' or 'high_return_high_vol'.
    For other methods: returns basic regime identifier.
    """
    target_name = str(target or "return").strip().lower()

    if target_name == "price":
        if total_states == 2:
            return "low_price_regime" if regime_id == 0 else "high_price_regime"
        if total_states == 3:
            labels = ("low_price_regime", "mid_price_regime", "high_price_regime")
            if 0 <= regime_id < len(labels):
                return labels[regime_id]
        if total_states == 4:
            labels = (
                "lowest_price_regime",
                "low_price_regime",
                "high_price_regime",
                "highest_price_regime",
            )
            if 0 <= regime_id < len(labels):
                return labels[regime_id]
        return f"price_regime_{regime_id}"

    if (
        method in ("hmm", "gmm", "ms_ar", "clustering")
        and mu is not None
        and sigma is not None
    ):
        return_level = _return_level_from_mean(
            float(mu), sigma=float(sigma), samples=samples
        )
        vol_level = volatility_level or "mod_vol"
        return f"{return_level}_{vol_level}"

    # GARCH volatility regimes - labels based on volatility level
    if method == "garch" and sigma is not None:
        # Consistent volatility labeling across all state counts
        # Each state gets a descriptive label based on its tier
        if total_states == 2:
            labels = ["low_vol", "high_vol"]
        elif total_states == 3:
            labels = ["low_vol", "moderate_vol", "high_vol"]
        elif total_states == 4:
            labels = ["low_vol", "moderate_vol", "high_vol", "very_high_vol"]
        else:
            # For 5+ states, use consistent naming pattern
            labels = [
                "very_low_vol",
                "low_vol",
                "moderate_vol",
                "high_vol",
                "very_high_vol",
            ]
            if total_states > 5:
                # Extend with numbered tiers for extreme cases
                labels = labels + [
                    f"extreme_vol_{i}" for i in range(6, total_states + 1)
                ]

        return labels[min(regime_id, len(labels) - 1)]

    # Wavelet labels are built in _build_regime_descriptions so they can use
    # the full set of state statistics and energy profiles together.
    if method == "wavelet" and sigma is not None:
        vol_level = _wavelet_volatility_label(volatility_level)
        if vol_level is not None:
            return vol_level

    # Ensemble method - derive labels from observed return sign + volatility.
    # This avoids state-index names that can contradict the regime statistics.
    if method == "ensemble" and mu is not None and sigma is not None:
        return_level = _return_level_from_mean(
            float(mu), sigma=float(sigma), samples=samples
        )
        vol_level = volatility_level or "mod_vol"
        return f"{return_level}_{vol_level}"

    # Fallback for methods without mu/sigma
    if total_states <= 2:
        return "regime_bearish" if regime_id == 0 else "regime_bullish"
    return f"regime_{regime_id}"


def _normalize_label_mapping(label_mapping: Any) -> Dict[int, int]:
    """Normalize serialized old->new state mappings to integer keys/values."""
    if not isinstance(label_mapping, dict):
        return {}

    normalized: Dict[int, int] = {}
    for old_state, new_state in label_mapping.items():
        try:
            normalized[int(old_state)] = int(new_state)
        except (TypeError, ValueError):
            continue
    return normalized


def _build_regime_descriptions(
    regime_params: Optional[Dict[str, Any]],
    method: str,
    *,
    target: Optional[str] = "return",
    label_mapping: Any = None,
    state_counts: Optional[Dict[int, int]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Build a mapping of regime_id -> descriptive info from regime_params.

    regime_params contains: weights, mu (mean returns/levels), sigma (dispersion)
    Returns dict with descriptive labels and statistics for each regime.
    """
    if not regime_params:
        return {}

    # Support both old key names (mu/sigma) and new key names (mean_return/volatility)
    mu_list = regime_params.get("mu") or regime_params.get("mean_return")
    sigma_list = regime_params.get("sigma") or regime_params.get("volatility")
    weights_list = regime_params.get("weights")

    # Only skip if we don't have the expected parameters for this method type
    if method in ("hmm", "gmm", "ms_ar", "garch") and (not mu_list or not sigma_list):
        return {}

    if not isinstance(mu_list, (list, tuple)) or not isinstance(
        sigma_list, (list, tuple)
    ):
        return {}

    descriptions: Dict[int, Dict[str, Any]] = {}
    n_states = len(mu_list)
    normalized_mapping = _normalize_label_mapping(label_mapping)
    indexed_states = (
        sorted(normalized_mapping.items(), key=lambda item: item[1])
        if normalized_mapping
        else [(i, i) for i in range(n_states)]
    )
    described_states = len(indexed_states) if indexed_states else n_states
    target_name = str(target or "return").strip().lower()
    wavelet_labels = (
        _build_wavelet_labels(
            regime_params,
            target=target_name,
            indexed_states=indexed_states,
            state_counts=state_counts,
        )
        if method == "wavelet"
        else {}
    )
    volatility_levels = _relative_volatility_levels(sigma_list)

    for source_state, regime_id in indexed_states:
        if source_state < 0 or source_state >= n_states:
            continue
        mu = float(mu_list[source_state]) if source_state < len(mu_list) else None
        sigma = (
            float(sigma_list[source_state]) if source_state < len(sigma_list) else None
        )
        weight = (
            float(weights_list[source_state])
            if weights_list and source_state < len(weights_list)
            else None
        )
        samples = int((state_counts or {}).get(regime_id, 0) or 0) or None

        label = wavelet_labels.get(regime_id)
        if label is None:
            label = _interpret_regime_label(
                regime_id,
                mu,
                sigma,
                method,
                described_states,
                target=target_name,
                volatility_level=volatility_levels.get(source_state),
                samples=samples,
            )

        desc: Dict[str, Any] = {"label": label}
        if method in {"hmm", "gmm"} and target_name != "price" and mu is not None and sigma is not None:
            stat_label = label
            label = _trader_hmm_label(
                float(mu),
                volatility_levels.get(source_state, "mod_vol"),
                sigma=float(sigma),
                samples=samples,
            )
            desc["label"] = label
            desc["stat_label"] = stat_label
            desc["trading_interpretation"] = _hmm_trading_interpretation(label)
        if mu is not None:
            if target_name == "price":
                desc["mean_price"] = round(mu, 6)
            else:
                desc["mean_return"] = round(mu, 6)
                desc["mean_return_pct"] = round(mu * 100, 4)
        if sigma is not None:
            if target_name == "price":
                desc["std_dev"] = round(sigma, 6)
            else:
                desc["volatility"] = round(sigma, 6)
                desc["volatility_pct"] = round(sigma * 100, 4)
        if target_name != "price" and mu is not None and sigma is not None:
            mean_t_stat = _mean_return_t_stat(float(mu), float(sigma), samples)
            desc["direction_criterion"] = "abs_mean_t_stat_gte_1.96"
            desc["direction_significant"] = bool(
                mean_t_stat is not None
                and abs(mean_t_stat) >= _DIRECTION_T_STAT_THRESHOLD
            )
            if mean_t_stat is not None:
                desc["mean_t_stat"] = round(mean_t_stat, 4)
        if weight is not None:
            desc["weight"] = round(weight, 4)

        descriptions[regime_id] = desc

    return descriptions


def _consolidate_payload(  # noqa: C901
    payload: Dict[str, Any],
    method: str,
    output_mode: str,
    include_series: bool = False,
    max_regimes: int = 10,
) -> Dict[str, Any]:
    """Consolidate time series into regime segments and restructure payload."""
    try:
        times = payload.get("times")
        if not times or not isinstance(times, list):
            return payload

        # Prepare consolidation
        segments: List[Dict[str, Any]] = []

        # Extract states/regimes
        states: List[int] = []
        probs: List[Any] = []
        cps_idx: set = set()  # Initialize for all methods to avoid unbound reference

        if method == "bocpd":
            # For BOCPD, we define regimes by change points
            # We can create a 'regime_id' that increments at each CP
            # We also look at 'change_points' list in payload
            if "change_points" in payload and isinstance(
                payload["change_points"], list
            ):
                for cp in payload["change_points"]:
                    if isinstance(cp, dict) and "idx" in cp:
                        cps_idx.add(cp["idx"])

            curr_regime = 0
            # Reconstruct per-step state
            for i in range(len(times)):
                if i in cps_idx:
                    curr_regime += 1
                states.append(curr_regime)

            # Probs
            raw_probs = payload.get("cp_prob")
            if isinstance(raw_probs, list):
                probs = raw_probs
            else:
                probs = [0.0] * len(times)

            raw_series = payload.get("_series_values")
            if isinstance(raw_series, list) and states:
                canon_state, _, canon_meta = _canonicalize_regime_labels(
                    np.asarray(states, dtype=int),
                    None,
                    np.asarray(raw_series, dtype=float),
                )
                states = [int(value) for value in canon_state.tolist()]
                params_used = payload.get("params_used")
                if isinstance(params_used, dict) and canon_meta.get("relabeled", False):
                    params_used["relabeled"] = True
                    params_used["label_mapping"] = canon_meta.get("mapping", {})

        elif method in ("ms_ar", "hmm", "gmm", "clustering", "garch", "wavelet", "ensemble"):
            raw_state = payload.get("state")
            if isinstance(raw_state, list):
                states = raw_state

            # Probs
            # structure is usually list of lists [ [p0, p1...], ... ]
            raw_probs = payload.get("state_probabilities")
            # We might just store the max prob or the prob of the current state?
            if isinstance(raw_probs, list) and raw_probs:
                if isinstance(raw_probs[0], list):
                    # Pick prob of selected state
                    for s, p_vec in zip(states, raw_probs):
                        if isinstance(p_vec, list) and 0 <= s < len(p_vec):
                            probs.append(p_vec[s])
                        else:
                            probs.append(None)
                else:
                    probs = raw_probs  # Should not happen based on current logic but safe fallback

        if not states or len(states) != len(times):
            # Fallback if creation failed
            return payload

        filtered_entries = []
        for idx, state in enumerate(states):
            try:
                state_value = int(state)
            except Exception:
                state_value = None
            if state_value is None or state_value < 0:
                continue
            prob_value = probs[idx] if idx < len(probs) else None
            filtered_entries.append((idx, times[idx], state_value, prob_value))
        if not filtered_entries:
            times = []
            states = []
            probs = []
            original_indices = []
        elif len(filtered_entries) != len(times):
            times = [entry[1] for entry in filtered_entries]
            states = [entry[2] for entry in filtered_entries]
            probs = [entry[3] for entry in filtered_entries]
            original_indices = [entry[0] for entry in filtered_entries]
        else:
            original_indices = list(range(len(times)))

        # Consolidate
        # Loop through
        if times:
            curr_start = times[0]
            curr_state = states[0]
            curr_start_index = original_indices[0]
            curr_prob_sum = 0.0
            curr_prob_count = 0
            curr_count = 0
            curr_transition_conf = None
            if method == "bocpd" and curr_start_index in cps_idx and probs:
                try:
                    curr_transition_conf = float(probs[0])
                except Exception:
                    curr_transition_conf = None

            i = 0
            while i < len(times):
                t = times[i]
                s = states[i]
                p = probs[i] if i < len(probs) and probs[i] is not None else None

                # Check change (state change)
                # For BOCPD, 's' changes exactly at CP.
                if s != curr_state and curr_count > 0:
                    # close segment
                    segment = {
                        "start": curr_start,
                        "end": times[i - 1] if i > 0 else curr_start,
                        "duration": curr_count,
                        "start_index": curr_start_index,
                        "end_index": original_indices[i - 1] if i > 0 else curr_start_index,
                        "regime": curr_state,  # state ID or regime ID
                        "transition_conf": curr_transition_conf,
                    }
                    if curr_prob_count:
                        segment["regime_confidence"] = (
                            curr_prob_sum / curr_prob_count
                        )
                    segments.append(segment)
                    # New segment
                    curr_start = t
                    curr_state = s
                    curr_start_index = original_indices[i]
                    curr_prob_sum = 0.0
                    curr_prob_count = 0
                    curr_count = 0
                    curr_transition_conf = None
                    if method == "bocpd":
                        try:
                            curr_transition_conf = None if p is None else float(p)
                        except Exception:
                            curr_transition_conf = None

                if p is not None:
                    try:
                        probability = float(p)
                    except (TypeError, ValueError):
                        probability = float("nan")
                    if np.isfinite(probability):
                        curr_prob_sum += probability
                        curr_prob_count += 1
                curr_count += 1
                i += 1

            # Final segment
            if curr_count > 0:
                segment = {
                    "start": curr_start,
                    "end": times[-1],
                    "duration": curr_count,
                    "start_index": curr_start_index,
                    "end_index": original_indices[-1],
                    "regime": curr_state,
                    "transition_conf": curr_transition_conf,
                }
                if curr_prob_count:
                    segment["regime_confidence"] = curr_prob_sum / curr_prob_count
                segments.append(segment)

        # Post-process segments for readability.
        # BOCPD: regime_confidence = 1 - mean(cp_prob over segment) = segment
        # stability (low internal change-point mass), not label-class confidence.
        # Peak boundary evidence is exposed separately as transition_prob_at_start.
        # HMM/GMM/etc.: regime_confidence is average state probability over the segment.

        if method == "garch":
            for segment in segments:
                segment.pop("regime_confidence", None)

        params_used = payload.get("params_used")
        label_mapping = (
            params_used.get("label_mapping") if isinstance(params_used, dict) else None
        )
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        reliability = (
            payload.get("reliability") if isinstance(payload.get("reliability"), dict) else {}
        )
        raw_series_values = payload.get("_series_values")
        raw_series = None
        if method == "bocpd" and isinstance(raw_series_values, list):
            try:
                raw_series = np.asarray(raw_series_values, dtype=float)
            except Exception:
                raw_series = None

        # Build regime descriptions for labeling segments
        regime_params = payload.get("regime_params")
        regime_descriptions = _build_regime_descriptions(
            regime_params,
            method,
            target=payload.get("target"),
            # Older/native-order HMM and GMM payloads need the label mapping;
            # current detector payloads publish parameters in canonical order.
            label_mapping=(
                label_mapping
                if method in {"hmm", "gmm"}
                and not (
                    isinstance(params_used, dict)
                    and params_used.get("regime_params_order") == "canonical"
                )
                else None
            ),
            state_counts=dict(Counter(int(state) for state in states)),
        )

        final_segments = []
        for _i, seg in enumerate(segments):
            regime_id = seg["regime"]
            if method == "bocpd":
                row = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "bars": seg["duration"],
                    "regime": regime_id,
                }
                if seg.get("transition_conf") is not None:
                    row["transition_prob_at_start"] = round(
                        float(seg["transition_conf"]),
                        4,
                    )
                if raw_series is not None and raw_series.size:
                    start_index = max(0, int(seg.get("start_index", 0)))
                    end_index = max(start_index, int(seg.get("end_index", start_index)))
                    if start_index < raw_series.size:
                        segment_slice = raw_series[start_index : min(end_index + 1, raw_series.size)]
                        row.update(
                            _build_bocpd_segment_context(
                                segment_slice,
                                target=payload.get("target"),
                            )
                        )
                bias = row.get("bias")
                if isinstance(bias, str) and bias:
                    row["label"] = f"{bias}_segment"
                else:
                    row["label"] = f"segment_{regime_id}"
                avg_transition_prob = round(float(seg["regime_confidence"]), 4)
                row["avg_transition_prob"] = avg_transition_prob
                row["regime_confidence"] = round(
                    max(0.0, min(1.0, 1.0 - avg_transition_prob)),
                    4,
                )
            else:
                row = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "bars": seg["duration"],
                    "regime": regime_id,
                }
                # Add descriptive label if available
                if regime_id in regime_descriptions:
                    row["label"] = regime_descriptions[regime_id].get(
                        "label", f"regime_{regime_id}"
                    )
                if seg.get("regime_confidence") is not None:
                    row["regime_confidence"] = round(
                        float(seg["regime_confidence"]), 4
                    )
            _attach_label_fields(row, method=method)
            final_segments.append(row)

        # Build current segment/regime info for trading (last segment only)
        current_regime = None
        transition_summary = None
        regime_context = None
        if final_segments:
            last_seg = final_segments[-1]
            if method == "bocpd":
                recent_change_points_count = summary.get(
                    "change_points_count",
                    len(payload.get("change_points", []))
                    if isinstance(payload.get("change_points"), list)
                    else 0,
                )
                latest_transition_probability = summary.get("last_cp_prob")
                if latest_transition_probability is None and probs:
                    latest_transition_probability = probs[-1]
                transition_risk = _bocpd_transition_risk(
                    latest_transition_probability,
                    payload.get("threshold"),
                )
                total_change_points_count = (
                    len(payload.get("change_points", []))
                    if isinstance(payload.get("change_points"), list)
                    else 0
                )
                current_regime = {
                    "status": _bocpd_segment_status(
                        change_points_count=total_change_points_count,
                        bars_since_change=last_seg["bars"],
                        transition_risk=transition_risk,
                    ),
                    "regime_id": last_seg.get("regime"),
                    "label": last_seg.get("label"),
                    "since": last_seg["start"],
                    "bars_since_change": last_seg["bars"],
                    "bars": last_seg["bars"],
                    "transition_risk": transition_risk,
                }
                _attach_label_fields(current_regime, method=method)
                latest_transition_probability_value = _finite_float(
                    latest_transition_probability,
                )
                if latest_transition_probability_value is not None:
                    current_regime["latest_transition_probability"] = round(
                        latest_transition_probability_value,
                        4,
                    )
                regime_context = {
                    key: last_seg[key]
                    for key in ("source", "bias", "return_pct", "volatility_pct")
                    if key in last_seg
                } or None
                transition_summary = {
                    "window_bars": int(
                        summary.get("lookback", len(times) if isinstance(times, list) else 0)
                    ),
                    "recent_change_points_count": int(recent_change_points_count),
                    "recent_transition_activity": _bocpd_transition_activity(
                        recent_change_points_count,
                        reliability.get("recent_cp_density"),
                    ),
                    "calibration_status": _bocpd_calibration_status(reliability),
                }
                for source_key, output_key in (
                    ("max_cp_prob", "max_transition_probability"),
                    ("mean_cp_prob", "mean_transition_probability"),
                ):
                    probability_value = _finite_float(summary.get(source_key))
                    if probability_value is not None:
                        transition_summary[output_key] = round(
                            probability_value,
                            4,
                        )
                for source_key in (
                    "raw_change_points_count",
                    "accepted_change_points_count",
                    "rejected_change_points_count",
                ):
                    count_value = summary.get(source_key)
                    if count_value is not None:
                        transition_summary[source_key] = int(count_value)
                expected_false_alarm_rate = _finite_float(
                    reliability.get("expected_false_alarm_rate"),
                )
                if expected_false_alarm_rate is not None:
                    transition_summary["expected_false_alarm_rate"] = round(
                        expected_false_alarm_rate,
                        4,
                    )
            else:
                current_regime = {
                    "regime_id": last_seg["regime"],
                    "label": last_seg.get("label", f"regime_{last_seg['regime']}"),
                    "since": last_seg["start"],
                    "bars": last_seg["bars"],
                }
                _attach_label_fields(current_regime, method=method)
                regime_confidence = last_seg.get("regime_confidence") or last_seg.get(
                    "transition_conf"
                )
                if regime_confidence is not None:
                    current_regime["regime_confidence"] = regime_confidence

        if output_mode == "compact" and isinstance(current_regime, dict):
            current_regime.pop("regime_id", None)

        # Restructure payload:
        new_payload: Dict[str, Any] = {
            "symbol": payload.get("symbol"),
            "timeframe": payload.get("timeframe"),
            "method": payload.get("method"),
            "success": True,
        }
        if method == "garch":
            new_payload["regime_confidence_status"] = (
                "not_applicable_hard_volatility_tiers"
            )
            new_payload["regime_confidence_note"] = (
                "GARCH volatility tiers are deterministic assignments; no "
                "probabilistic regime confidence is reported."
            )
        if (
            isinstance(params_used, dict)
            and params_used.get("model_fit_scope") == "full_window"
        ):
            threshold_scope = params_used.get("threshold_scope")
            new_payload["historical_labels_are_retrospective"] = True
            new_payload["historical_label_scope"] = (
                "retrospective_full_window_fit_and_percentile_thresholds"
                if threshold_scope == "full_window_percentiles"
                else "retrospective_full_window_model_fit"
            )
            new_payload["point_in_time_guidance"] = (
                "Use rolling --end cutoffs when evaluating historical labels as "
                "live strategy inputs. Example: regime_detect EURUSD --timeframe H1 "
                "--method hmm --end 2026-08-20T12:00:00Z."
            )
        for state_count_key in ("requested_n_states", "effective_n_states"):
            if state_count_key in payload:
                new_payload[state_count_key] = payload[state_count_key]

        if method == "bocpd":
            if current_regime:
                new_payload["current_regime"] = current_regime
            if transition_summary:
                new_payload["transition_summary"] = transition_summary
            if regime_context:
                new_payload["regime_context"] = regime_context
                new_payload.setdefault(
                    "units",
                    {
                        "regime_context.return_pct": "percent",
                        "regime_context.volatility_pct": "percent",
                        "scale_note": "Percent values use 1.0 = 1%.",
                    },
                )

            total_regimes = len(final_segments)
            if output_mode == "compact" and max_regimes > 0 and total_regimes > max_regimes:
                new_payload["regimes"] = final_segments[-max_regimes:]
                new_payload["regimes_truncated"] = True
                new_payload["total_regimes"] = total_regimes
                new_payload["showing_regimes"] = max_regimes
                new_payload["has_more"] = True
                omitted = max(0, total_regimes - max_regimes)
                new_payload["history_hint"] = (
                    f"{omitted} older regime segment(s) omitted; use detail='full' "
                    "or increase max_regimes."
                )
            else:
                new_payload["regimes"] = final_segments
                new_payload["total_regimes"] = total_regimes
                if output_mode == "compact":
                    new_payload["has_more"] = False
        else:
            # Core trading info (compact)
            if current_regime:
                new_payload["current_regime"] = current_regime

            # Limit regimes for compact mode (keep most recent N)
            total_regimes = len(final_segments)
            if output_mode == "compact" and max_regimes > 0 and total_regimes > max_regimes:
                # Keep only the most recent max_regimes
                truncated_segments = final_segments[-max_regimes:]
                new_payload["regimes"] = truncated_segments
                new_payload["regimes_truncated"] = True
                new_payload["total_regimes"] = total_regimes
                new_payload["showing_regimes"] = max_regimes
                new_payload["has_more"] = True
                omitted = max(0, total_regimes - max_regimes)
                new_payload["history_hint"] = (
                    f"{omitted} older regime segment(s) omitted; use detail='full' "
                    "or increase max_regimes."
                )
            else:
                new_payload["regimes"] = final_segments
                if output_mode == "compact":
                    new_payload["total_regimes"] = total_regimes
                    new_payload["has_more"] = False

        if regime_descriptions:
            observed_regimes = {
                int(row["regime"])
                for row in final_segments
                if isinstance(row, dict) and row.get("regime") is not None
            }
            for regime_id, description in regime_descriptions.items():
                observed = int(regime_id) in observed_regimes
                description.setdefault("observed_in_window", observed)
                if not observed:
                    description.setdefault(
                        "note",
                        "Learned model state not observed in the current decoded window.",
                    )
                if "weight" in description:
                    description.setdefault(
                        "weight_note",
                        "Model state weight, not necessarily the fraction of displayed bars.",
                    )
            if method == "ensemble" and output_mode == "compact":
                regime_descriptions = {
                    regime_id: description
                    for regime_id, description in regime_descriptions.items()
                    if int(regime_id) in observed_regimes
                }
            new_payload["regime_info"] = regime_descriptions
        if "reliability" in payload:
            new_payload["reliability"] = payload["reliability"]
        if "warnings" in payload:
            new_payload["warnings"] = payload["warnings"]

        # FULL mode: add technical details
        if output_mode == "full":
            if "threshold" in payload:
                new_payload["threshold"] = payload["threshold"]
            if "tuning_hint" in payload:
                new_payload["tuning_hint"] = payload["tuning_hint"]
            if "params_used" in payload:
                new_payload["params_used"] = payload["params_used"]
            if "regime_params" in payload:
                new_payload["regime_params"] = payload["regime_params"]
            if "model_fit" in payload:
                new_payload["model_fit"] = payload["model_fit"]
            if "ensemble_info" in payload:
                new_payload["ensemble_info"] = payload["ensemble_info"]

            # Raw series only in full mode
            if include_series:
                series_data = {}
                for k in [
                    "times",
                    "cp_prob",
                    "state",
                    "state_probabilities",
                    "change_points",
                    "conditional_volatility",
                ]:
                    if k in payload:
                        series_data[k] = payload[k]
                if series_data:
                    new_payload["series"] = series_data

        return new_payload

    except Exception as e:
        # Fallback to original payload on error
        payload["consolidation_error"] = str(e)
        payload["error"] = f"Regime output consolidation failed: {e}"
        payload["error_code"] = "regime_consolidation_failed"
        payload["partial_failure"] = True
        payload["success"] = False
        return payload


def _summary_only_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a minimal summary payload with no regimes or series."""
    out: Dict[str, Any] = {
        "symbol": payload.get("symbol"),
        "timeframe": payload.get("timeframe"),
        "method": payload.get("method"),
        "target": payload.get("target"),
        "success": bool(payload.get("success", True)),
    }
    if "summary" in payload:
        out["summary"] = payload["summary"]
    if "classification_window" in payload:
        out["classification_window"] = payload["classification_window"]
    if "reliability" in payload:
        out["reliability"] = payload["reliability"]
    if "params_used" in payload:
        out["params_used"] = payload["params_used"]
    if "threshold" in payload:
        out["threshold"] = payload["threshold"]
    if "tuning_hint" in payload:
        out["tuning_hint"] = payload["tuning_hint"]
    if "warnings" in payload:
        out["warnings"] = payload["warnings"]
    return out


__all__ = [
    "_consolidate_payload",
    "_summary_only_payload",
    "_interpret_regime_label",
    "_build_regime_descriptions",
]
