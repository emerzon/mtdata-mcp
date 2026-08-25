"""Current-regime summaries, reliability, and output-mode helpers."""

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from .payload import _summary_only_payload


def _summary_window_size(lookback: int, size: int) -> int:
    try:
        lookback_i = int(lookback)
    except Exception:
        lookback_i = int(size)
    return min(max(lookback_i, 0), int(size))


_DIRECTION_SIGNALS = frozenset({"bullish", "bearish", "neutral"})
_VOLATILITY_SIGNALS = frozenset(
    {"very_low_vol", "low_vol", "moderate_vol", "high_vol", "very_high_vol"}
)


def _coerce_optional_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _lookup_regime_info_entry(regime_info: Any, regime_id: Any) -> Dict[str, Any]:
    if not isinstance(regime_info, dict):
        return {}

    candidates: List[Any] = []
    if regime_id is not None:
        candidates.append(regime_id)
        try:
            candidates.append(int(regime_id))
        except (TypeError, ValueError):
            pass
        candidates.append(str(regime_id))

    for candidate in candidates:
        details = regime_info.get(candidate)
        if isinstance(details, dict):
            return details
    return {}


def _normalize_direction_signal(
    label: Any,
    *,
    mean_return: Any = None,
) -> Optional[str]:
    text = str(label or "").strip().lower()
    if "bullish" in text or "positive" in text:
        return "bullish"
    if "bearish" in text or "negative" in text:
        return "bearish"
    if "neutral" in text:
        return "neutral"

    # A mean without state dispersion and occupancy cannot support a
    # timeframe-independent direction claim. State labels encode the shared
    # t-statistic criterion when sufficient evidence exists.
    return None


def _normalize_volatility_signal(
    label: Any,
    *,
    volatility: Any = None,
) -> Optional[str]:
    text = str(label or "").strip().lower()
    if "very_high_vol" in text or "extreme_vol" in text:
        return "very_high_vol"
    if "moderate_vol" in text or "mod_vol" in text:
        return "moderate_vol"
    if "very_low_vol" in text:
        return "very_low_vol"
    if "quiet" in text:
        return "low_vol"
    if "high_vol" in text or "volatile" in text:
        return "high_vol"
    if "low_vol" in text or "stable" in text:
        return "low_vol"

    # Raw per-bar volatility has no symbol/timeframe-independent semantic
    # threshold. Callers should supply a run-relative label when available.
    return None


def _reliability_label(confidence: Any) -> str:
    value = _coerce_optional_float(confidence)
    if value is None:
        return "unknown"
    if value >= 0.8:
        return "high"
    if value >= 0.6:
        return "medium"
    if value >= 0.4:
        return "low"
    return "very_low"


def _common_reliability(
    reliability: Optional[Dict[str, Any]],
    *,
    source: str,
    confidence: Any = None,
) -> Dict[str, Any]:
    out = dict(reliability or {})
    if "confidence" not in out:
        out["confidence"] = round(float(_coerce_optional_float(confidence) or 0.0), 4)
    out.setdefault("reliability_label", _reliability_label(out.get("confidence")))
    confidence_value = _coerce_optional_float(out.get("confidence"))
    if confidence_value is not None and confidence_value < 0.55:
        out.setdefault(
            "confidence_note",
            "Low-confidence regime classification; use a wider lookback, compare methods, or treat the current regime as tentative.",
        )
    out.setdefault("source", source)
    return out


def _append_warnings(payload: Dict[str, Any], warnings_to_add: List[str]) -> None:
    if not warnings_to_add:
        return
    existing = payload.get("warnings")
    warnings_list = list(existing) if isinstance(existing, list) else []
    for warning_text in warnings_to_add:
        if warning_text not in warnings_list:
            warnings_list.append(warning_text)
    payload["warnings"] = warnings_list


def _smoothing_warnings(method: str, smoothing_meta: Dict[str, Any]) -> List[str]:
    pending_state = smoothing_meta.get("pending_state")
    pending_bars = int(smoothing_meta.get("pending_bars", 0) or 0)
    if pending_state is None or pending_bars <= 0:
        return []
    return [
        f"method='{method}' has candidate state {pending_state} pending confirmation "
        f"({pending_bars}/{int(smoothing_meta.get('pending_bars_required', 1))} "
        "required consecutive bars); the current emitted regime is retained."
    ]


def _summarize_rule_based_current_regime(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    current_regime = result.get("current_regime")
    if not isinstance(current_regime, dict):
        return None
    regime = result.get("regime")
    if not isinstance(regime, dict):
        regime = current_regime
    entry = {
        key: regime.get(key)
        for key in (
            "state",
            "direction",
            "trend_strength",
            "efficiency_ratio",
            "window_bars",
            "window_move_pct",
            "signal_source",
        )
        if regime.get(key) is not None
    }
    for key in (
        "regime_id",
        "label",
        "classification_scope",
        "boundary_status",
        "persistence_status",
    ):
        value = current_regime.get(key)
        if value is not None:
            entry[key] = value
    regime_confidence = current_regime.get("regime_confidence")
    if regime_confidence is not None:
        entry["regime_confidence"] = regime_confidence
    return entry or None


def _summarize_bocpd_current_regime(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    current_regime = result.get("current_regime")
    if not isinstance(current_regime, dict) or not current_regime:
        regimes = result.get("regimes")
        if isinstance(regimes, list) and regimes and isinstance(regimes[-1], dict):
            last = regimes[-1]
            current_regime = {
                "since": last.get("start"),
                "bars_since_change": last.get("bars"),
            }
        else:
            return None

    entry = {
        key: current_regime.get(key)
        for key in (
            "status",
            "since",
            "bars_since_change",
            "transition_risk",
            "latest_transition_probability",
        )
        if current_regime.get(key) is not None
    }

    transition_summary = result.get("transition_summary")
    if not isinstance(transition_summary, dict):
        transition_summary = result.get("summary")
    if isinstance(transition_summary, dict):
        recent_change_points_count = transition_summary.get(
            "recent_change_points_count",
            transition_summary.get("change_points_count"),
        )
        if recent_change_points_count is not None:
            entry["recent_change_points_count"] = recent_change_points_count
        for key in ("recent_transition_activity", "calibration_status"):
            value = transition_summary.get(key)
            if value is not None:
                entry[key] = value

    regime_context = result.get("regime_context")
    if isinstance(regime_context, dict):
        for key in ("bias", "return_pct", "volatility_pct"):
            value = regime_context.get(key)
            if value is not None:
                entry[key] = value

    return entry or None


def _summarize_current_regime_for_comparison(
    method: str,
    result: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict):
        return None

    if method == "bocpd":
        return _summarize_bocpd_current_regime(result)

    if method == "rule_based":
        return _summarize_rule_based_current_regime(result)

    current = result.get("current_regime")
    if not isinstance(current, dict) or not current:
        regimes = result.get("regimes")
        if isinstance(regimes, list) and regimes and isinstance(regimes[-1], dict):
            last = regimes[-1]
            current = {
                "regime_id": last.get("regime"),
                "label": last.get("label"),
                "regime_confidence": last.get("regime_confidence"),
                "since": last.get("start"),
                "bars": last.get("bars"),
            }
        else:
            return None

    regime_id = current.get("regime_id")
    regime_stats = _lookup_regime_info_entry(result.get("regime_info"), regime_id)
    label = current.get("label")
    if label is None and regime_stats:
        label = regime_stats.get("label")

    entry: Dict[str, Any] = {
        key: current.get(key)
        for key in ("regime_id", "since", "bars")
        if current.get(key) is not None
    }
    if label is not None:
        entry["label"] = label

    regime_confidence = current.get("regime_confidence")
    if regime_confidence is not None:
        entry["regime_confidence"] = regime_confidence

    direction = None
    if method in {"hmm", "ms_ar", "ensemble"}:
        direction = _normalize_direction_signal(
            label,
            mean_return=regime_stats.get("mean_return"),
        )
    elif method in {"clustering", "pelt"}:
        direction = _normalize_direction_signal(label)
    if direction is not None:
        entry["direction"] = direction

    volatility = None
    if method in {"hmm", "ms_ar", "garch", "ensemble", "wavelet", "pelt"}:
        volatility = _normalize_volatility_signal(
            label,
            volatility=regime_stats.get("volatility"),
        )
    if volatility is not None:
        entry["volatility"] = volatility

    for key in ("mean_return_pct", "volatility_pct"):
        value = regime_stats.get(key)
        if value is not None:
            entry[key] = value

    if method == "bocpd":
        summary = result.get("summary")
        if isinstance(summary, dict):
            for key in ("last_cp_prob", "change_points_count"):
                value = summary.get(key)
                if value is not None:
                    entry[key] = value

    return entry or None


def _build_semantic_agreement(current_regimes: Dict[str, Any]) -> Dict[str, Any]:
    agreement: Dict[str, Any] = {"basis": "semantic_signals"}

    direction_votes = {
        method: entry["direction"]
        for method, entry in current_regimes.items()
        if isinstance(entry, dict) and entry.get("direction") in _DIRECTION_SIGNALS
    }
    volatility_votes = {
        method: entry["volatility"]
        for method, entry in current_regimes.items()
        if isinstance(entry, dict) and entry.get("volatility") in _VOLATILITY_SIGNALS
    }

    def _consensus(votes: Dict[str, str]) -> Optional[Dict[str, Any]]:
        if len(votes) < 2:
            return None
        counts = Counter(votes.values())
        majority, count = counts.most_common(1)[0]
        return {
            "majority": majority,
            "agreement_pct": round(count / len(votes) * 100.0, 2),
            "methods_considered": list(votes.keys()),
        }

    direction_consensus = _consensus(direction_votes)
    if direction_consensus is not None:
        agreement["direction"] = direction_consensus

    volatility_consensus = _consensus(volatility_votes)
    if volatility_consensus is not None:
        agreement["volatility"] = volatility_consensus

    return agreement


def _method_window_bars(method: str, result: Any) -> Optional[int]:
    if not isinstance(result, dict):
        return None
    candidates: List[Any] = []
    params_used = result.get("params_used")
    if isinstance(params_used, dict):
        candidates.append(params_used.get("window_bars"))
    classification_window = result.get("classification_window")
    if isinstance(classification_window, dict):
        candidates.append(classification_window.get("bars"))
    if method == "rule_based":
        current_regime = result.get("current_regime")
        if isinstance(current_regime, dict):
            candidates.append(current_regime.get("window_bars"))
        regime = result.get("regime")
        if isinstance(regime, dict):
            candidates.append(regime.get("window_bars"))
    for value in candidates:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _build_all_method_comparison(results_by_method: Dict[str, Any]) -> Dict[str, Any]:
    current_regimes: Dict[str, Any] = {}
    method_windows: Dict[str, int] = {}
    for method, result in results_by_method.items():
        current_regimes[method] = _summarize_current_regime_for_comparison(
            method,
            result,
        )
        window_bars = _method_window_bars(method, result)
        if window_bars is not None:
            method_windows[str(method)] = window_bars

    comparison: Dict[str, Any] = {
        "methods_run": list(results_by_method.keys()),
        "current_regimes": current_regimes,
        "agreement": _build_semantic_agreement(current_regimes),
    }
    if method_windows:
        comparison["method_windows"] = method_windows
    return comparison


def _apply_bocpd_output_mode(
    payload: Dict[str, Any],
    *,
    output: str,
    lookback: int,
    cp_prob: np.ndarray,
    change_points: List[Dict[str, Any]],
    raw_cp_idx: List[int],
    reliability: Dict[str, Any],
    expected_fa_rate: float,
    calibration_age_bars: int,
    tuning_hint: Optional[str],
) -> Dict[str, Any]:
    n = _summary_window_size(lookback, len(cp_prob))
    tail = (
        np.asarray(cp_prob[-n:], dtype=float)
        if n > 0
        else np.asarray(cp_prob, dtype=float)
    )
    recent_floor = len(cp_prob) - n
    recent_cps = [cp for cp in change_points if cp.get("idx", 0) >= recent_floor]
    summary = {
        "lookback": int(n),
        "last_cp_prob": float(cp_prob[-1]) if len(cp_prob) else float("nan"),
        "max_cp_prob": float(np.nanmax(tail)) if tail.size else float("nan"),
        "mean_cp_prob": float(np.nanmean(tail)) if tail.size else float("nan"),
        "change_points_count": int(len(recent_cps)),
        "accepted_change_points_count": int(len(recent_cps)),
        "raw_change_points_count": int(
            sum(1 for idx in raw_cp_idx if int(idx) >= recent_floor)
        ),
        "rejected_change_points_count": int(
            max(
                0,
                sum(1 for idx in raw_cp_idx if int(idx) >= recent_floor)
                - int(len(recent_cps)),
            )
        ),
        "recent_change_points": recent_cps[-5:],
        "confidence": float(reliability.get("confidence", 0.0)),
        "expected_false_alarm_rate": float(
            reliability.get("expected_false_alarm_rate", expected_fa_rate)
        ),
        "calibration_age_bars": int(
            reliability.get("calibration_age_bars", calibration_age_bars)
        ),
    }
    if tuning_hint is not None:
        summary["tuning_hint"] = tuning_hint
    payload["summary"] = summary
    if output == "summary":
        return _summary_only_payload(payload)
    return payload


def _apply_state_output_mode(
    payload: Dict[str, Any],
    *,
    output: str,
    lookback: int,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply output mode filtering.

    - 'summary': Return stats only, no regimes
    - 'compact': Trading-focused (regimes, current_regime, regime_info, reliability)
    - 'full': Research-focused (adds raw series, params, technical details)
    """
    payload["summary"] = summary
    if output == "summary":
        return _summary_only_payload(payload)
    # Note: Raw series (times, state, state_probabilities) are now handled
    # in _consolidate_payload based on output_mode
    return payload


def _mark_collapsed_state_confidence(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Prevent a one-state posterior from masquerading as model certainty."""

    def _mark_segment(row: Dict[str, Any]) -> None:
        if row.get("regime_confidence") is not None:
            row["raw_posterior_mass"] = row["regime_confidence"]
        row["regime_confidence"] = 0.0
        row["label"] = "unidentifiable"
        for key in (
            "direction",
            "state_label_native",
            "state_label_canonical",
        ):
            row.pop(key, None)
        row["label_quality"] = "unidentifiable_state_collapse"

    current = payload.get("current_regime")
    if isinstance(current, dict):
        _mark_segment(current)
    regimes = payload.get("regimes")
    if isinstance(regimes, list):
        for segment in regimes:
            if not isinstance(segment, dict):
                continue
            _mark_segment(segment)
    regime_info = payload.get("regime_info")
    if isinstance(regime_info, dict):
        for description in regime_info.values():
            if not isinstance(description, dict):
                continue
            description["label"] = "unidentifiable"
            for key in ("direction", "stat_label", "trading_interpretation"):
                description.pop(key, None)
            description["label_quality"] = "unidentifiable_state_collapse"
    payload["status"] = "unidentifiable"
    payload["signal_status"] = "not_actionable"
    return payload
