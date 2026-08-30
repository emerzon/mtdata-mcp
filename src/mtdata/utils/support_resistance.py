"""Weighted support/resistance detection from historical retests."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..shared.constants import (
    TIMEFRAME_SECONDS as _TIMEFRAME_SECONDS,
)
from .coercion import coerce_finite_float as _as_finite_float
from .time import _format_time_minimal, bar_close_epoch, format_epoch_utc, parse_iso_utc

_METHOD_NAME = "weighted_retests"
_DEFAULT_REACTION_BARS = 6
_DEFAULT_ADX_PERIOD = 14
_DEFAULT_BOUNCE_WEIGHT = 0.8
_DEFAULT_ADX_WEIGHT = 0.35
_DEFAULT_VOLUME_WEIGHT = 0.25
_DEFAULT_VOLUME_RATIO_CAP = 5.0
_DEFAULT_SWING_REVERSAL_ATR = 1.3
_DEFAULT_EPISODE_TOUCH_DECAY = 0.25
_DEFAULT_ADAPTIVE_TOLERANCE_ATR_MULT = 0.25
_DEFAULT_ADAPTIVE_TOLERANCE_MIN_SCALE = 0.75
_DEFAULT_ADAPTIVE_TOLERANCE_MAX_SCALE = 1.85
_DEFAULT_ADAPTIVE_REACTION_MIN_SCALE = 0.67
_DEFAULT_ADAPTIVE_REACTION_MAX_SCALE = 1.6
_DEFAULT_ADAPTIVE_RECENT_WINDOW = 8
_DEFAULT_ZONE_STD_MULT = 1.5
_DEFAULT_ZONE_ATR_MULT = 0.2
_MAX_ZONE_WIDTH_ATR = 2.0
_DEFAULT_BREAKOUT_BUFFER_ATR_MULT = 0.15
_DEFAULT_BREAKOUT_PENALTY_BASE = 0.35
_DEFAULT_BREAKOUT_PENALTY_ATR_MULT = 0.25
_DEFAULT_ROLE_REVERSAL_BONUS = 0.65
_DEFAULT_MTF_DEDUPE_FACTOR = 0.35
_DEFAULT_MTF_CONFIRMATION_BONUS = 0.2
_EMPTY_SIDE_REMEDIATION = (
    "Increase lookback, widen the start/end window, or lower min_touches"
)
_DEFAULT_STRUCTURE_GAP_WARNING_PCT = 12.0
_FIBONACCI_RETRACEMENTS = (0.236, 0.382, 0.5, 0.618, 0.786)
_FIBONACCI_EXTENSIONS = (1.272, 1.618)
_FIBONACCI_LEVEL_DECIMALS = 6
_FIBONACCI_SWING_SELECTION_RULE = "most_recent_completed_swing_bracketing_current_price_else_latest_completed_swing"
_FIBONACCI_TIMEFRAME_SELECTION_RULE = (
    "highest_timeframe_grid_bracketing_current_price_else_most_recent_completed_grid"
)
_AUTO_TIMEFRAMES = ("M15", "H1", "H4", "D1")
_TIMEFRAME_WEIGHTS = {
    "M15": 0.9,
    "H1": 1.0,
    "H4": 1.15,
    "D1": 1.3,
}


def get_auto_support_resistance_timeframes() -> tuple[str, ...]:
    return _AUTO_TIMEFRAMES


def _to_epoch(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            out = float(value)
            return out if math.isfinite(out) else None
        if hasattr(value, "timestamp"):
            out = float(value.timestamp())
            return out if math.isfinite(out) else None
    except Exception:
        return None
    return None


def _format_time(timestamp: Optional[float]) -> Optional[str]:
    if isinstance(timestamp, str):
        cleaned = timestamp.strip()
        return cleaned or None
    if timestamp is None:
        return None
    try:
        epoch = float(timestamp)
        if not math.isfinite(epoch):
            return None
        return _format_time_minimal(epoch)
    except Exception:
        return None


def _parse_output_time(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return parse_iso_utc(cleaned).timestamp()
        except Exception:
            return None
    return None


def _bar_close_as_of_epoch(
    open_epoch: Optional[float],
    timeframe: Any,
) -> Optional[float]:
    if open_epoch is None:
        return None
    tf = str(timeframe or "").strip().upper()
    if not tf or tf == "AUTO":
        return open_epoch
    try:
        return bar_close_epoch(open_epoch, tf)
    except (OverflowError, TypeError, ValueError):
        return open_epoch


def _payload_close_as_of_epoch(payload: Dict[str, Any]) -> Optional[float]:
    parsed = _parse_output_time(payload.get("structure_as_of"))
    if parsed is not None:
        return parsed
    window = payload.get("window") if isinstance(payload.get("window"), dict) else None
    open_epoch = _parse_output_time((window or {}).get("end"))
    return _bar_close_as_of_epoch(open_epoch, payload.get("timeframe"))


def _to_numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.array([], dtype=float)
    series = pd.to_numeric(frame[column], errors="coerce")
    return series.to_numpy(dtype=float, copy=False)


def _normalize_volume_weighting(volume_weighting: Optional[str]) -> str:
    raw = str(volume_weighting or "off").strip().lower()
    if raw in {"", "off", "none", "false", "0"}:
        return "off"
    if raw == "auto":
        return raw
    raise ValueError("volume_weighting must be 'off' or 'auto'")


def _resolve_volume_series(
    frame: pd.DataFrame,
    *,
    volume_weighting: str,
) -> tuple[Optional[np.ndarray], Optional[str], Optional[float]]:
    mode = _normalize_volume_weighting(volume_weighting)
    if mode == "off":
        return None, None, None

    for column in ("real_volume", "volume", "tick_volume"):
        if column not in frame.columns:
            continue
        values = _to_numeric_array(frame, column)
        finite = values[np.isfinite(values) & (values > 0.0)]
        if finite.size == 0:
            continue
        baseline = float(np.nanmedian(finite))
        if math.isfinite(baseline) and baseline > 0.0:
            return values, column, baseline
    return None, None, None


def _last_finite(values: np.ndarray) -> Optional[float]:
    for value in values[::-1]:
        if math.isfinite(float(value)):
            return float(value)
    return None


def _match_nearest_cluster(
    clusters: List[Dict[str, Any]],
    value: float,
    tolerance: float,
) -> Optional[Dict[str, Any]]:
    best_cluster: Optional[Dict[str, Any]] = None
    best_delta: Optional[float] = None
    for cluster in clusters:
        ref = float(cluster["value"])
        threshold = max(abs(ref), abs(value), 1e-9) * float(tolerance)
        delta = abs(ref - value)
        if delta <= threshold and (best_delta is None or delta < best_delta):
            best_cluster = cluster
            best_delta = delta
    return best_cluster


def _weighted_average(items: List[tuple[float, float]]) -> Optional[float]:
    total_weight = 0.0
    total_value = 0.0
    for weight, value in items:
        weight_value = float(weight)
        item_value = float(value)
        if not math.isfinite(weight_value) or not math.isfinite(item_value) or weight_value <= 0.0:
            continue
        total_weight += weight_value
        total_value += weight_value * item_value
    if total_weight <= 0.0:
        return None
    return total_value / total_weight


def _round_output_price(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(round(out, _FIBONACCI_LEVEL_DECIMALS))


def _format_ratio_label(ratio: float) -> str:
    pct = float(ratio) * 100.0
    return f"{pct:.1f}".rstrip("0").rstrip(".") + "%"


def _compute_atr_and_adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    *,
    period: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    prev_close = np.concatenate(([closes[0]], closes[:-1]))
    tr = np.maximum.reduce(
        [
            highs - lows,
            np.abs(highs - prev_close),
            np.abs(lows - prev_close),
        ]
    )

    up_move = np.diff(highs, prepend=highs[0])
    down_move = -np.diff(lows, prepend=lows[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

    alpha = 1.0 / float(max(1, int(period)))
    tr_s = pd.Series(tr, dtype=float)
    plus_s = pd.Series(plus_dm, dtype=float)
    minus_s = pd.Series(minus_dm, dtype=float)

    atr = tr_s.ewm(alpha=alpha, adjust=False, min_periods=max(1, int(period))).mean()
    plus_smoothed = plus_s.ewm(alpha=alpha, adjust=False, min_periods=max(1, int(period))).mean()
    minus_smoothed = minus_s.ewm(alpha=alpha, adjust=False, min_periods=max(1, int(period))).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * plus_smoothed / atr
        minus_di = 100.0 * minus_smoothed / atr
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)

    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=max(1, int(period))).mean()
    return (
        atr.to_numpy(dtype=float, copy=False),
        adx.to_numpy(dtype=float, copy=False),
    )


def _resolve_adaptive_settings(
    closes: np.ndarray,
    atr: np.ndarray,
    *,
    base_tolerance_fraction: float,
    base_reaction_bars: int,
) -> Dict[str, Any]:
    atr_fraction_values: List[float] = []
    for close_value, atr_value in zip(closes, atr):
        close_float = abs(float(close_value))
        atr_float = float(atr_value)
        if close_float <= 1e-9 or not math.isfinite(close_float) or not math.isfinite(atr_float) or atr_float <= 0.0:
            continue
        atr_fraction_values.append(atr_float / close_float)

    if not atr_fraction_values:
        return {
            "adaptive_mode": "atr_regime",
            "effective_tolerance_fraction": float(base_tolerance_fraction),
            "effective_reaction_bars": int(base_reaction_bars),
            "current_atr_fraction": None,
            "baseline_atr_fraction": None,
            "volatility_ratio": 1.0,
        }

    atr_fraction_array = np.asarray(atr_fraction_values, dtype=float)
    recent_window = max(3, min(_DEFAULT_ADAPTIVE_RECENT_WINDOW, len(atr_fraction_array)))
    current_slice = atr_fraction_array[-recent_window:]
    baseline_slice = atr_fraction_array[:-recent_window]
    if baseline_slice.size == 0:
        baseline_slice = atr_fraction_array

    baseline_atr_fraction = float(np.nanmedian(baseline_slice))
    current_atr_fraction = float(np.nanmedian(current_slice))
    if not math.isfinite(baseline_atr_fraction) or baseline_atr_fraction <= 0.0:
        baseline_atr_fraction = (
            current_atr_fraction
            if math.isfinite(current_atr_fraction) and current_atr_fraction > 0.0
            else 0.0
        )

    volatility_ratio = 1.0
    if baseline_atr_fraction > 0.0 and math.isfinite(current_atr_fraction):
        volatility_ratio = current_atr_fraction / baseline_atr_fraction
    if not math.isfinite(volatility_ratio) or volatility_ratio <= 0.0:
        volatility_ratio = 1.0

    clipped_ratio = min(max(volatility_ratio, 0.5), 2.5)
    tolerance_scale = min(
        max(math.sqrt(clipped_ratio), _DEFAULT_ADAPTIVE_TOLERANCE_MIN_SCALE),
        _DEFAULT_ADAPTIVE_TOLERANCE_MAX_SCALE,
    )
    reaction_scale = min(
        max(1.0 / math.sqrt(clipped_ratio), _DEFAULT_ADAPTIVE_REACTION_MIN_SCALE),
        _DEFAULT_ADAPTIVE_REACTION_MAX_SCALE,
    )
    atr_anchor = max(
        0.0,
        current_atr_fraction * _DEFAULT_ADAPTIVE_TOLERANCE_ATR_MULT,
    )
    base_tolerance_value = max(0.0, float(base_tolerance_fraction))
    scaled_tolerance = base_tolerance_value * tolerance_scale
    if volatility_ratio >= 1.0:
        effective_tolerance_fraction = max(scaled_tolerance, atr_anchor)
    else:
        # In compressed regimes keep the ATR floor informative, but never let it widen
        # the zone beyond the caller's base tolerance.
        compressed_anchor = atr_anchor * clipped_ratio
        effective_tolerance_fraction = min(base_tolerance_value, max(scaled_tolerance, compressed_anchor))
    effective_tolerance_fraction = min(0.05, max(0.0, effective_tolerance_fraction))
    effective_reaction_bars = max(1, int(round(float(base_reaction_bars) * reaction_scale)))

    return {
        "adaptive_mode": "atr_regime",
        "effective_tolerance_fraction": float(effective_tolerance_fraction),
        "effective_reaction_bars": int(effective_reaction_bars),
        "current_atr_fraction": (
            float(current_atr_fraction)
            if math.isfinite(current_atr_fraction)
            else None
        ),
        "baseline_atr_fraction": (
            float(baseline_atr_fraction)
            if math.isfinite(baseline_atr_fraction) and baseline_atr_fraction > 0.0
            else None
        ),
        "volatility_ratio": float(volatility_ratio),
    }


def _build_test(
    *,
    test_type: str,
    index: int,
    level_value: float,
    highs: np.ndarray,
    lows: np.ndarray,
    atr: np.ndarray,
    adx: np.ndarray,
    volume: Optional[np.ndarray],
    volume_baseline: Optional[float],
    epochs: List[Optional[float]],
    reaction_bars: int,
    decay_half_life_bars: int,
) -> Optional[Dict[str, Any]]:
    future_start = int(index) + 1
    future_stop = min(len(highs), future_start + int(reaction_bars))
    if future_start >= future_stop:
        return None
    reaction_complete = future_stop - future_start >= int(reaction_bars)

    if test_type == "support":
        favorable = float(np.nanmax(highs[future_start:future_stop]) - level_value)
    else:
        favorable = float(level_value - np.nanmin(lows[future_start:future_stop]))
    if not math.isfinite(favorable):
        favorable = 0.0
    favorable = max(0.0, favorable)

    atr_value = float(atr[index]) if index < len(atr) and math.isfinite(float(atr[index])) else float("nan")
    if not math.isfinite(atr_value) or atr_value <= 0.0:
        local_range = float(highs[index] - lows[index]) if math.isfinite(float(highs[index] - lows[index])) else 0.0
        atr_value = max(local_range, abs(level_value) * 0.001, 1e-9)

    bounce_atr = max(0.0, favorable / atr_value)
    pretest_adx = 0.0
    if index > 0 and index - 1 < len(adx):
        adx_value = float(adx[index - 1])
        if math.isfinite(adx_value):
            pretest_adx = max(0.0, adx_value)
    elif index < len(adx):
        adx_value = float(adx[index])
        if math.isfinite(adx_value):
            pretest_adx = max(0.0, adx_value)

    age_bars = max(0, len(highs) - 1 - int(index))
    half_life = max(1, int(decay_half_life_bars))
    decay_weight = math.exp(-math.log(2.0) * float(age_bars) / float(half_life))

    retest_component = decay_weight
    bounce_component = decay_weight * _DEFAULT_BOUNCE_WEIGHT * math.log1p(bounce_atr)
    adx_component = decay_weight * _DEFAULT_ADX_WEIGHT * (min(pretest_adx, 60.0) / 25.0)
    volume_ratio = None
    volume_component = 0.0
    if (
        volume is not None
        and volume_baseline is not None
        and volume_baseline > 0.0
        and 0 <= index < len(volume)
    ):
        volume_value = float(volume[index])
        if math.isfinite(volume_value) and volume_value > 0.0:
            volume_ratio = max(0.0, volume_value / float(volume_baseline))
            volume_excess = max(0.0, min(volume_ratio, _DEFAULT_VOLUME_RATIO_CAP) - 1.0)
            volume_component = decay_weight * _DEFAULT_VOLUME_WEIGHT * math.log1p(volume_excess)
    score = retest_component + bounce_component + adx_component + volume_component

    return {
        "type": test_type,
        "index": int(index),
        "value": float(level_value),
        "atr_value": float(atr_value),
        "timestamp": epochs[index] if index < len(epochs) else None,
        "age_bars": int(age_bars),
        "decay_weight": float(decay_weight),
        "bounce_atr": float(bounce_atr),
        "reaction_complete": bool(reaction_complete),
        "pretest_adx": float(pretest_adx),
        "retest_component": float(retest_component),
        "bounce_component": float(bounce_component),
        "adx_component": float(adx_component),
        "volume_ratio": None if volume_ratio is None else float(volume_ratio),
        "volume_component": float(volume_component),
        "score": float(score),
    }


def _collect_local_extrema_candidates(highs: np.ndarray, lows: np.ndarray) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for index in range(1, len(highs) - 1):
        low_center = float(lows[index])
        low_prev = float(lows[index - 1])
        low_next = float(lows[index + 1])
        if all(math.isfinite(v) for v in (low_center, low_prev, low_next)) and low_center <= low_prev and low_center <= low_next:
            candidates.append({"type": "support", "index": int(index), "value": float(low_center)})

        high_center = float(highs[index])
        high_prev = float(highs[index - 1])
        high_next = float(highs[index + 1])
        if all(math.isfinite(v) for v in (high_center, high_prev, high_next)) and high_center >= high_prev and high_center >= high_next:
            candidates.append({"type": "resistance", "index": int(index), "value": float(high_center)})
    candidates.sort(key=lambda item: (int(item["index"]), 0 if str(item["type"]) == "support" else 1))
    return candidates


def _is_more_extreme_candidate(candidate: Dict[str, Any], current: Dict[str, Any]) -> bool:
    candidate_value = float(candidate["value"])
    current_value = float(current["value"])
    candidate_index = int(candidate["index"])
    current_index = int(current["index"])
    candidate_type = str(candidate["type"])
    if candidate_type == "support":
        return candidate_value < current_value or (
            math.isclose(candidate_value, current_value, rel_tol=0.0, abs_tol=1e-12) and candidate_index > current_index
        )
    return candidate_value > current_value or (
        math.isclose(candidate_value, current_value, rel_tol=0.0, abs_tol=1e-12) and candidate_index > current_index
    )


def _filter_swing_candidates(
    candidates: List[Dict[str, Any]],
    *,
    atr: np.ndarray,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    accepted: List[Dict[str, Any]] = []
    fallback_atr = 1e-9
    finite_atr = [float(value) for value in atr if math.isfinite(float(value)) and float(value) > 0.0]
    if finite_atr:
        fallback_atr = float(np.nanmedian(np.asarray(finite_atr, dtype=float)))
        if not math.isfinite(fallback_atr) or fallback_atr <= 0.0:
            fallback_atr = 1e-9

    def _atr_for_index(index: int) -> float:
        if 0 <= index < len(atr):
            value = float(atr[index])
            if math.isfinite(value) and value > 0.0:
                return value
        return fallback_atr

    for candidate in candidates:
        if not accepted:
            accepted.append(dict(candidate))
            continue

        last = accepted[-1]
        if str(candidate["type"]) == str(last["type"]):
            if _is_more_extreme_candidate(candidate, last):
                candidate_index = int(candidate["index"])
                last_index = int(last["index"])
                extension_threshold = max(_atr_for_index(candidate_index), _atr_for_index(last_index), 1e-9) * _DEFAULT_SWING_REVERSAL_ATR
                extension = abs(float(candidate["value"]) - float(last["value"]))
                if math.isfinite(extension) and extension >= extension_threshold:
                    accepted.append(dict(candidate))
                else:
                    accepted[-1] = dict(candidate)
            continue

        candidate_index = int(candidate["index"])
        last_index = int(last["index"])
        candidate_atr = _atr_for_index(candidate_index)
        last_atr = _atr_for_index(last_index)
        reversal_threshold = max(candidate_atr, last_atr, fallback_atr, 1e-9) * _DEFAULT_SWING_REVERSAL_ATR
        swing_move = abs(float(candidate["value"]) - float(last["value"]))
        if math.isfinite(swing_move) and swing_move >= reversal_threshold:
            accepted.append(dict(candidate))

    return accepted


def _collect_tests(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    *,
    epochs: List[Optional[float]],
    reaction_bars: int,
    adx_period: int,
    decay_half_life_bars: int,
    atr: Optional[np.ndarray] = None,
    adx: Optional[np.ndarray] = None,
    volume: Optional[np.ndarray] = None,
    volume_baseline: Optional[float] = None,
) -> List[Dict[str, Any]]:
    atr_values = atr
    adx_values = adx
    if atr_values is None or adx_values is None:
        atr_values, adx_values = _compute_atr_and_adx(highs, lows, closes, period=adx_period)
    tests: List[Dict[str, Any]] = []
    candidates = _collect_local_extrema_candidates(highs, lows)
    swing_candidates = _filter_swing_candidates(candidates, atr=atr_values)
    for candidate in swing_candidates:
        test = _build_test(
            test_type=str(candidate["type"]),
            index=int(candidate["index"]),
            level_value=float(candidate["value"]),
            highs=highs,
            lows=lows,
            atr=atr_values,
            adx=adx_values,
            volume=volume,
            volume_baseline=volume_baseline,
            epochs=epochs,
            reaction_bars=reaction_bars,
            decay_half_life_bars=decay_half_life_bars,
        )
        if test is not None:
            tests.append(test)

    return tests


def _cluster_tests(tests: List[Dict[str, Any]], *, tolerance_fraction: float) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    for test in sorted(tests, key=lambda item: float(item["value"])):
        value = float(test["value"])
        best_cluster = _match_nearest_cluster(clusters, value, tolerance_fraction)

        score = max(float(test["score"]), 1e-9)
        timestamp = test.get("timestamp")
        if best_cluster is None:
            clusters.append(
                {
                    "value": float(value),
                    "weight_sum": float(score),
                    "touches": 1,
                    "score_base": float(test["score"]),
                    "score": float(test["score"]),
                    "retest_score": float(test["retest_component"]),
                    "bounce_score": float(test["bounce_component"]),
                    "adx_score": float(test["adx_component"]),
                    "volume_score": float(test.get("volume_component", 0.0)),
                    "first_time": timestamp,
                    "last_time": timestamp,
                    "first_index": int(test["index"]),
                    "last_index": int(test["index"]),
                    "support_tests": 1 if test["type"] == "support" else 0,
                    "resistance_tests": 1 if test["type"] == "resistance" else 0,
                    "value_sq_sum": float(value * value * score),
                    "touch_min": float(value),
                    "touch_max": float(value),
                    "atr_metric_sum": float(test["atr_value"]) * score,
                    "atr_weight_sum": float(score),
                    "bounce_metric_sum": float(test["bounce_atr"]) * score,
                    "adx_metric_sum": float(test["pretest_adx"]) * score,
                    "metric_weight_sum": float(score),
                    "volume_metric_sum": 0.0,
                    "volume_metric_weight_sum": 0.0,
                    "tests": [dict(test)],
                }
            )
            volume_ratio = _as_finite_float(test.get("volume_ratio"))
            if volume_ratio is not None:
                clusters[-1]["volume_metric_sum"] = float(volume_ratio) * score
                clusters[-1]["volume_metric_weight_sum"] = float(score)
            continue

        cluster = best_cluster
        new_weight = float(cluster["weight_sum"]) + score
        cluster["value"] = (float(cluster["value"]) * float(cluster["weight_sum"]) + value * score) / new_weight
        cluster["weight_sum"] = new_weight
        cluster["touches"] = int(cluster["touches"]) + 1
        cluster["score_base"] = float(cluster["score_base"]) + float(test["score"])
        cluster["score"] = float(cluster["score"]) + float(test["score"])
        cluster["retest_score"] = float(cluster["retest_score"]) + float(test["retest_component"])
        cluster["bounce_score"] = float(cluster["bounce_score"]) + float(test["bounce_component"])
        cluster["adx_score"] = float(cluster["adx_score"]) + float(test["adx_component"])
        cluster["volume_score"] = float(cluster.get("volume_score", 0.0)) + float(test.get("volume_component", 0.0))
        cluster["first_index"] = min(int(cluster["first_index"]), int(test["index"]))
        cluster["last_index"] = max(int(cluster["last_index"]), int(test["index"]))
        cluster["support_tests"] = int(cluster["support_tests"]) + (1 if test["type"] == "support" else 0)
        cluster["resistance_tests"] = int(cluster["resistance_tests"]) + (1 if test["type"] == "resistance" else 0)
        cluster["value_sq_sum"] = float(cluster["value_sq_sum"]) + float(value * value * score)
        cluster["touch_min"] = min(float(cluster["touch_min"]), float(value))
        cluster["touch_max"] = max(float(cluster["touch_max"]), float(value))
        cluster["atr_metric_sum"] = float(cluster["atr_metric_sum"]) + float(test["atr_value"]) * score
        cluster["atr_weight_sum"] = float(cluster["atr_weight_sum"]) + float(score)
        cluster["bounce_metric_sum"] = float(cluster["bounce_metric_sum"]) + float(test["bounce_atr"]) * score
        cluster["adx_metric_sum"] = float(cluster["adx_metric_sum"]) + float(test["pretest_adx"]) * score
        cluster["metric_weight_sum"] = float(cluster["metric_weight_sum"]) + score
        volume_ratio = _as_finite_float(test.get("volume_ratio"))
        if volume_ratio is not None:
            cluster["volume_metric_sum"] = float(cluster.get("volume_metric_sum", 0.0)) + float(volume_ratio) * score
            cluster["volume_metric_weight_sum"] = float(cluster.get("volume_metric_weight_sum", 0.0)) + score
        cluster.setdefault("tests", []).append(dict(test))

        if timestamp is not None:
            if cluster["first_time"] is None or float(timestamp) < float(cluster["first_time"]):
                cluster["first_time"] = timestamp
            if cluster["last_time"] is None or float(timestamp) > float(cluster["last_time"]):
                cluster["last_time"] = timestamp

    return clusters


def _apply_episode_metrics(cluster: Dict[str, Any], *, episode_gap_bars: int) -> None:
    tests = sorted(cluster.get("tests", []), key=lambda item: (int(item.get("index", -1)), str(item.get("type", ""))))
    if not tests:
        cluster["episodes"] = 0
        cluster["support_episodes"] = 0
        cluster["resistance_episodes"] = 0
        cluster["episode_details"] = []
        return

    gap_value = max(1, int(episode_gap_bars))
    episodes: List[Dict[str, Any]] = []
    current_episode: Optional[Dict[str, Any]] = None

    for test in tests:
        test_index = int(test.get("index", -1))
        test_type = str(test.get("type", ""))
        if (
            current_episode is None
            or test_type != str(current_episode.get("type", ""))
            or test_index - int(current_episode.get("last_index", test_index)) > gap_value
        ):
            current_episode = {
                "type": test_type,
                "first_index": test_index,
                "last_index": test_index,
                "first_time": test.get("timestamp"),
                "last_time": test.get("timestamp"),
                "tests": [dict(test)],
            }
            episodes.append(current_episode)
            continue

        current_episode["last_index"] = test_index
        current_episode["last_time"] = test.get("timestamp")
        current_episode.setdefault("tests", []).append(dict(test))

    adjusted_retest = 0.0
    adjusted_bounce = 0.0
    adjusted_adx = 0.0
    adjusted_volume = 0.0
    adjusted_base = 0.0
    episode_details: List[Dict[str, Any]] = []
    support_episodes = 0
    resistance_episodes = 0

    for episode in episodes:
        episode_tests = sorted(
            episode.get("tests", []),
            key=lambda item: (float(item.get("score", 0.0)), int(item.get("index", -1))),
            reverse=True,
        )
        episode_score = 0.0
        episode_retest = 0.0
        episode_bounce = 0.0
        episode_adx = 0.0
        episode_volume = 0.0
        for rank, test in enumerate(episode_tests):
            weight = 1.0 if rank == 0 else _DEFAULT_EPISODE_TOUCH_DECAY
            episode_score += float(test.get("score", 0.0)) * weight
            episode_retest += float(test.get("retest_component", 0.0)) * weight
            episode_bounce += float(test.get("bounce_component", 0.0)) * weight
            episode_adx += float(test.get("adx_component", 0.0)) * weight
            episode_volume += float(test.get("volume_component", 0.0)) * weight

        adjusted_base += episode_score
        adjusted_retest += episode_retest
        adjusted_bounce += episode_bounce
        adjusted_adx += episode_adx
        adjusted_volume += episode_volume
        episode_type = str(episode.get("type", ""))
        if episode_type == "support":
            support_episodes += 1
        elif episode_type == "resistance":
            resistance_episodes += 1
        episode_details.append(
            {
                "type": episode_type,
                "touches": len(episode_tests),
                "first_touch": _format_time(episode.get("first_time")),
                "last_touch": _format_time(episode.get("last_time")),
            }
        )

    cluster["episodes"] = len(episodes)
    cluster["support_episodes"] = int(support_episodes)
    cluster["resistance_episodes"] = int(resistance_episodes)
    cluster["episode_details"] = episode_details
    cluster["score_base"] = float(adjusted_base)
    cluster["retest_score"] = float(adjusted_retest)
    cluster["bounce_score"] = float(adjusted_bounce)
    cluster["adx_score"] = float(adjusted_adx)
    cluster["volume_score"] = float(adjusted_volume)
    cluster["score"] = float(adjusted_base)


def _dominant_source(cluster: Dict[str, Any]) -> str:
    support_episodes = int(cluster.get("support_episodes", 0))
    resistance_episodes = int(cluster.get("resistance_episodes", 0))
    if support_episodes > resistance_episodes:
        return "support"
    if resistance_episodes > support_episodes:
        return "resistance"

    support_tests = int(cluster.get("support_tests", 0))
    resistance_tests = int(cluster.get("resistance_tests", 0))
    if support_tests > resistance_tests:
        return "support"
    if resistance_tests > support_tests:
        return "resistance"
    return "mixed"


def _cluster_atr_avg(cluster: Dict[str, Any]) -> Optional[float]:
    weight = float(cluster.get("atr_weight_sum", 0.0))
    if weight > 0.0:
        value = float(cluster.get("atr_metric_sum", 0.0)) / weight
        if math.isfinite(value) and value > 0.0:
            return value
    zone_low = cluster.get("zone_low")
    zone_high = cluster.get("zone_high")
    zone_width_atr = cluster.get("zone_width_atr")
    try:
        if zone_low is not None and zone_high is not None and zone_width_atr not in (None, 0):
            zone_width = float(zone_high) - float(zone_low)
            out = zone_width / float(zone_width_atr)
            if math.isfinite(out) and out > 0.0:
                return out
    except Exception:
        pass
    return None


def _recompute_touches_after_zone_cap(
    cluster: Dict[str, Any],
    *,
    zone_low: Optional[float],
    zone_high: Optional[float],
) -> None:
    if zone_low is None or zone_high is None:
        return
    tests = cluster.get("tests")
    if not isinstance(tests, list) or not tests:
        return
    in_zone: List[Dict[str, Any]] = []
    support_tests = 0
    resistance_tests = 0
    for test in tests:
        if not isinstance(test, dict):
            continue
        price = _as_finite_float(test.get("value"))
        if price is None:
            continue
        if float(zone_low) - 1e-12 <= price <= float(zone_high) + 1e-12:
            in_zone.append(test)
            if str(test.get("type") or "") == "support":
                support_tests += 1
            elif str(test.get("type") or "") == "resistance":
                resistance_tests += 1
    cluster["touches"] = len(in_zone)
    cluster["support_tests"] = support_tests
    cluster["resistance_tests"] = resistance_tests
    if in_zone:
        prices = [float(test["value"]) for test in in_zone]
        cluster["touch_min"] = min(prices)
        cluster["touch_max"] = max(prices)


def _resolve_zone(
    cluster: Dict[str, Any],
    *,
    tolerance_fraction: float,
    window_low: Optional[float] = None,
    window_high: Optional[float] = None,
) -> Dict[str, Any]:
    raw_value = cluster.get("value")
    if raw_value is None:
        raw_low = cluster.get("zone_low")
        raw_high = cluster.get("zone_high")
        if raw_low is None or raw_high is None:
            raise ValueError("Cluster requires value or zone bounds")
        raw_value = (float(raw_low) + float(raw_high)) * 0.5
    value = float(raw_value)

    def _bounded_zone(
        low: float,
        high: float,
        atr_avg: Optional[float],
        *,
        window_low: Optional[float] = None,
        window_high: Optional[float] = None,
    ) -> Dict[str, Any]:
        capped = False
        if atr_avg is not None and atr_avg > 0.0:
            max_width = float(atr_avg) * _MAX_ZONE_WIDTH_ATR
            if high - low > max_width:
                orig_low, orig_high = low, high
                half_width = max_width * 0.5
                low = value - half_width
                high = value + half_width
                capped = True
                covers_window_low = (
                    window_low is not None and orig_low <= float(window_low) + 1e-12
                )
                covers_window_high = (
                    window_high is not None and orig_high >= float(window_high) - 1e-12
                )
                nearer_low = False
                if covers_window_low and window_low is not None:
                    if not covers_window_high or window_high is None:
                        nearer_low = True
                    else:
                        nearer_low = abs(value - float(window_low)) <= abs(
                            value - float(window_high)
                        )
                if nearer_low and window_low is not None and low > float(window_low):
                    low = float(window_low)
                    high = low + max_width
                elif (
                    covers_window_high
                    and window_high is not None
                    and high < float(window_high)
                ):
                    high = float(window_high)
                    low = high - max_width
        width = max(0.0, high - low)
        return {
            "zone_low": low,
            "zone_high": high,
            "zone_width": width,
            "zone_width_atr": (
                None
                if atr_avg is None or atr_avg <= 0.0
                else width / float(atr_avg)
            ),
            "zone_width_capped": capped,
            "atr_avg": atr_avg,
        }

    try:
        zone_low = cluster.get("zone_low")
        zone_high = cluster.get("zone_high")
        if zone_low is not None and zone_high is not None:
            low = float(zone_low)
            high = float(zone_high)
            if math.isfinite(low) and math.isfinite(high) and high >= low:
                return _bounded_zone(
                    low,
                    high,
                    _cluster_atr_avg(cluster),
                    window_low=window_low,
                    window_high=window_high,
                )
    except Exception:
        pass

    weight_sum = max(float(cluster.get("weight_sum", 0.0)), 1e-9)
    value_sq_sum = float(cluster.get("value_sq_sum", value * value * weight_sum))
    variance = max(0.0, (value_sq_sum / weight_sum) - (value * value))
    weighted_std = math.sqrt(variance)
    atr_avg = _cluster_atr_avg(cluster)
    tol_pad = max(abs(value), 1e-9) * max(float(tolerance_fraction), 0.0) * 0.5
    atr_pad = (
        atr_avg or max(abs(value), 1e-9) * max(float(tolerance_fraction), 0.0)
    ) * _DEFAULT_ZONE_ATR_MULT
    dispersion_pad = weighted_std * _DEFAULT_ZONE_STD_MULT
    half_width = max(tol_pad, atr_pad, dispersion_pad, 1e-9)
    touch_min = float(cluster.get("touch_min", value))
    touch_max = float(cluster.get("touch_max", value))
    zone_low = min(touch_min, value - half_width)
    zone_high = max(touch_max, value + half_width)
    return _bounded_zone(
        zone_low,
        zone_high,
        atr_avg,
        window_low=window_low,
        window_high=window_high,
    )


def _analyze_cluster_state(
    cluster: Dict[str, Any],
    *,
    closes: np.ndarray,
    epochs: List[Optional[float]],
    tolerance_fraction: float,
    window_low: Optional[float] = None,
    window_high: Optional[float] = None,
) -> None:
    dominant_source = _dominant_source(cluster)
    zone = _resolve_zone(
        cluster,
        tolerance_fraction=tolerance_fraction,
        window_low=window_low,
        window_high=window_high,
    )
    cluster["zone_low"] = zone["zone_low"]
    cluster["zone_high"] = zone["zone_high"]
    cluster["zone_width"] = zone["zone_width"]
    cluster["zone_width_atr"] = zone["zone_width_atr"]
    cluster["zone_width_capped"] = bool(zone["zone_width_capped"])
    if zone["zone_width_capped"]:
        _recompute_touches_after_zone_cap(
            cluster,
            zone_low=zone["zone_low"],
            zone_high=zone["zone_high"],
        )

    cluster["decisive_break_count"] = 0
    cluster["avg_breach_atr"] = None
    cluster["first_break_time"] = None
    cluster["first_break_index"] = None
    cluster["last_break_time"] = None
    cluster["last_break_index"] = None
    cluster["role_reversal_count"] = 0
    cluster["breakout_penalty"] = 0.0
    cluster["role_reversal_bonus"] = 0.0
    cluster["status"] = "intact"

    if dominant_source not in {"support", "resistance"}:
        cluster["score"] = max(0.0, float(cluster.get("score_base", cluster.get("score", 0.0))))
        return

    zone_low = zone["zone_low"]
    zone_high = zone["zone_high"]
    atr_avg = zone["atr_avg"]
    if zone_low is None or zone_high is None:
        cluster["score"] = max(0.0, float(cluster.get("score_base", cluster.get("score", 0.0))))
        return

    zone_mid = 0.5 * (float(zone_low) + float(zone_high))
    buffer = max(
        (atr_avg or 0.0) * _DEFAULT_BREAKOUT_BUFFER_ATR_MULT,
        max(abs(zone_mid), 1e-9) * max(float(tolerance_fraction), 0.0) * 0.25,
        1e-9,
    )
    analysis_start = min(int(test["index"]) for test in cluster.get("tests", []) or [{"index": 0}])
    start = max(0, analysis_start)
    observed = np.asarray(closes[start:], dtype=float)
    finite = np.isfinite(observed)
    if dominant_source == "support":
        breach = np.where(finite, np.maximum(0.0, float(zone_low) - observed), 0.0)
        broken = finite & (observed < (float(zone_low) - buffer))
    else:
        breach = np.where(finite, np.maximum(0.0, observed - float(zone_high)), 0.0)
        broken = finite & (observed > (float(zone_high) + buffer))

    if broken.size:
        episode_starts = np.flatnonzero(
            broken & np.concatenate((np.asarray([True]), ~broken[:-1]))
        )
    else:
        episode_starts = np.asarray([], dtype=int)
    break_indices = (episode_starts + start).tolist()
    break_times = [epochs[index] if index < len(epochs) else None for index in break_indices]
    if episode_starts.size:
        breach_atr = breach / atr_avg if atr_avg is not None and atr_avg > 0.0 else np.zeros_like(breach)
        breach_episode_max = np.maximum.reduceat(
            np.where(broken, breach_atr, 0.0), episode_starts
        ).tolist()
    else:
        breach_episode_max = []

    breakout_count = len(break_indices)
    avg_breach_atr = float(np.mean(breach_episode_max)) if breach_episode_max else None
    first_break_index = break_indices[0] if break_indices else None
    first_break_time = break_times[0] if break_times else None
    last_break_index = break_indices[-1] if break_indices else None
    last_break_time = break_times[-1] if break_times else None

    role_reversal_tests = 0
    if breakout_count and first_break_index is not None:
        expected_new_role = "resistance" if dominant_source == "support" else "support"
        role_reversal_tests = sum(
            1
            for test in cluster.get("tests", [])
            if str(test.get("type")) == expected_new_role and int(test.get("index", -1)) > int(first_break_index)
        )

    breakout_penalty = 0.0
    if breakout_count:
        breakout_penalty = breakout_count * (
            _DEFAULT_BREAKOUT_PENALTY_BASE
            + _DEFAULT_BREAKOUT_PENALTY_ATR_MULT * min(float(avg_breach_atr or 0.0), 4.0)
        )
    role_reversal_bonus = 0.0
    if role_reversal_tests:
        role_reversal_bonus = min(1.5, _DEFAULT_ROLE_REVERSAL_BONUS * float(role_reversal_tests))

    cluster["decisive_break_count"] = int(breakout_count)
    cluster["avg_breach_atr"] = None if avg_breach_atr is None else float(avg_breach_atr)
    cluster["first_break_time"] = first_break_time
    cluster["first_break_index"] = first_break_index
    cluster["last_break_time"] = last_break_time
    cluster["last_break_index"] = last_break_index
    cluster["role_reversal_count"] = int(role_reversal_tests)
    cluster["breakout_penalty"] = float(breakout_penalty)
    cluster["role_reversal_bonus"] = float(role_reversal_bonus)

    if role_reversal_tests:
        cluster["status"] = "role_reversal"
    elif breakout_count:
        cluster["status"] = "broken"
    else:
        cluster["status"] = "intact"

    base_score = float(cluster.get("score_base", cluster.get("score", 0.0)))
    cluster["score"] = max(0.0, base_score - breakout_penalty + role_reversal_bonus)


def _format_level(
    cluster: Dict[str, Any],
    *,
    current_price: Optional[float],
    tolerance_fraction: float,
    window_low: Optional[float] = None,
    window_high: Optional[float] = None,
) -> Dict[str, Any]:
    value = float(cluster["value"])
    zone = _resolve_zone(
        cluster,
        tolerance_fraction=tolerance_fraction,
        window_low=window_low,
        window_high=window_high,
    )
    if zone["zone_width_capped"]:
        _recompute_touches_after_zone_cap(
            cluster,
            zone_low=zone["zone_low"],
            zone_high=zone["zone_high"],
        )
    dominant_source = _dominant_source(cluster)
    level_type = dominant_source if dominant_source in {"support", "resistance"} else "support"
    distance = None
    distance_pct = None
    if current_price is not None and math.isfinite(current_price):
        contains_current_price = bool(
            zone["zone_low"] is not None
            and zone["zone_high"] is not None
            and float(zone["zone_low"]) <= current_price <= float(zone["zone_high"])
        )
        level_type = (
            "range"
            if contains_current_price
            else "support"
            if value <= current_price
            else "resistance"
        )
        distance = value - current_price
        distance_pct = (distance / max(abs(current_price), 1e-9)) * 100.0
    else:
        contains_current_price = False

    metric_weight_sum = float(cluster.get("metric_weight_sum", 0.0))
    avg_bounce_atr = None
    avg_pretest_adx = None
    if metric_weight_sum > 0.0:
        avg_bounce_atr = float(cluster["bounce_metric_sum"]) / metric_weight_sum
        avg_pretest_adx = float(cluster["adx_metric_sum"]) / metric_weight_sum
    volume_metric_weight_sum = float(cluster.get("volume_metric_weight_sum", 0.0))
    avg_test_volume_ratio = None
    if volume_metric_weight_sum > 0.0:
        avg_test_volume_ratio = float(cluster.get("volume_metric_sum", 0.0)) / volume_metric_weight_sum

    base_score = float(cluster.get("score_base", cluster.get("score", 0.0)))
    breakout_penalty = float(cluster.get("breakout_penalty", 0.0))
    role_reversal_bonus = float(cluster.get("role_reversal_bonus", 0.0))
    mtf_confirmation_bonus = float(cluster.get("mtf_confirmation_bonus", 0.0))
    volume_score = float(cluster.get("volume_score", 0.0))
    total_score = max(0.0, base_score - breakout_penalty + role_reversal_bonus + mtf_confirmation_bonus)
    last_break_time = cluster.get("last_break_time")

    level = {
        "type": level_type,
        "value": float(round(value, 6)),
        "touches": int(cluster["touches"]),
        "episodes": int(cluster.get("episodes", cluster.get("touches", 0))),
        "score": float(round(total_score, 4)),
        "distance": None if distance is None else float(round(distance, 6)),
        "distance_pct": None if distance_pct is None else float(round(distance_pct, 6)),
        "zone_low": None if zone["zone_low"] is None else float(round(float(zone["zone_low"]), 6)),
        "zone_high": None if zone["zone_high"] is None else float(round(float(zone["zone_high"]), 6)),
        "zone_width": None if zone["zone_width"] is None else float(round(float(zone["zone_width"]), 6)),
        "zone_width_atr": None if zone["zone_width_atr"] is None else float(round(float(zone["zone_width_atr"]), 4)),
        "zone_width_capped": bool(zone["zone_width_capped"]),
        "contains_current_price": contains_current_price,
        "first_touch": _format_time(cluster.get("first_time")),
        "last_touch": _format_time(cluster.get("last_time")),
        "dominant_source": dominant_source,
        "status": str(cluster.get("status", "intact")),
        "source_tests": {
            "support": int(cluster.get("support_tests", 0)),
            "resistance": int(cluster.get("resistance_tests", 0)),
        },
        "source_episodes": {
            "support": int(cluster.get("support_episodes", cluster.get("support_tests", 0))),
            "resistance": int(cluster.get("resistance_episodes", cluster.get("resistance_tests", 0))),
        },
        "avg_bounce_atr": None if avg_bounce_atr is None else float(round(avg_bounce_atr, 4)),
        "avg_pretest_adx": None if avg_pretest_adx is None else float(round(avg_pretest_adx, 4)),
        "avg_test_volume_ratio": None if avg_test_volume_ratio is None else float(round(avg_test_volume_ratio, 4)),
        "volume_source": cluster.get("volume_source"),
        "breakout_analysis": {
            "decisive_break_count": int(cluster.get("decisive_break_count", 0)),
            "avg_breach_atr": None
            if cluster.get("avg_breach_atr") is None
            else float(round(float(cluster["avg_breach_atr"]), 4)),
            "last_break_time": _format_time(last_break_time) if last_break_time is not None else None,
            "role_reversal_count": int(cluster.get("role_reversal_count", 0)),
        },
        "episode_details": list(cluster.get("episode_details", [])),
        "score_breakdown": {
            "base": float(round(base_score, 4)),
            "retests": float(round(float(cluster["retest_score"]), 4)),
            "bounce": float(round(float(cluster["bounce_score"]), 4)),
            "adx": float(round(float(cluster["adx_score"]), 4)),
            "volume": float(round(volume_score, 4)),
            "breakout_penalty": float(round(breakout_penalty, 4)),
            "role_reversal_bonus": float(round(role_reversal_bonus, 4)),
            "mtf_confirmation_bonus": float(round(mtf_confirmation_bonus, 4)),
            "total": float(round(total_score, 4)),
        },
    }
    if (
        level_type in {"support", "resistance"}
        and dominant_source in {"support", "resistance"}
        and level_type != dominant_source
    ):
        level["role_transition"] = True
    return level


def _timeframe_weight(timeframe: Any) -> float:
    return float(_TIMEFRAME_WEIGHTS.get(str(timeframe or "").upper(), 1.0))


def _timeframe_seconds(timeframe: Any) -> Optional[int]:
    return _TIMEFRAME_SECONDS.get(str(timeframe or "").upper())


def _timeframe_sort_key(timeframe: Any) -> tuple[int, str]:
    key = str(timeframe or "").upper()
    try:
        return (_AUTO_TIMEFRAMES.index(key), key)
    except ValueError:
        return (len(_AUTO_TIMEFRAMES), key)


def _pick_first_current_price(results: List[Dict[str, Any]]) -> Optional[float]:
    for result in results:
        try:
            value = result.get("current_price")
            if value is None:
                continue
            out = float(value)
            if math.isfinite(out):
                return out
        except Exception:
            continue
    return None


def _current_price_position(
    *,
    current_price: Optional[float],
    low_value: float,
    high_value: float,
) -> Optional[str]:
    if current_price is None:
        return None
    price_value = float(current_price)
    if not math.isfinite(price_value):
        return None
    if price_value < low_value:
        return "below_swing_low"
    if price_value > high_value:
        return "above_swing_high"
    return "within_swing"


def _build_fibonacci_level(
    *,
    ratio: float,
    value: float,
    current_price: Optional[float],
    kind: str,
    projection: Optional[str] = None,
) -> Dict[str, Any]:
    raw_level_value = float(value)
    rounded_level_value = _round_output_price(raw_level_value)
    level_value = rounded_level_value if rounded_level_value is not None else raw_level_value
    out: Dict[str, Any] = {
        "label": _format_ratio_label(float(ratio)),
        "ratio": float(ratio),
        "kind": str(kind),
        "value": level_value,
    }
    if projection:
        out["projection"] = str(projection)
    if current_price is None or not math.isfinite(float(current_price)):
        return out

    price_value = float(current_price)
    distance = level_value - price_value
    out["type"] = "support" if level_value <= price_value else "resistance"
    rounded_distance = _round_output_price(distance)
    out["distance"] = rounded_distance if rounded_distance is not None else distance
    if abs(price_value) > 1e-9:
        distance_pct = (distance / price_value) * 100.0
        rounded_distance_pct = _round_output_price(distance_pct)
        out["distance_pct"] = rounded_distance_pct if rounded_distance_pct is not None else distance_pct
    return out


def _nearest_fibonacci_levels(
    levels: List[Dict[str, Any]],
    *,
    current_price: Optional[float],
) -> Dict[str, Dict[str, Any]]:
    if current_price is None or not math.isfinite(float(current_price)):
        return {}
    price_value = float(current_price)
    supports = [
        level for level in levels
        if isinstance(level, dict)
        and math.isfinite(float(level.get("value", float("nan"))))
        and float(level["value"]) <= price_value
    ]
    resistances = [
        level for level in levels
        if isinstance(level, dict)
        and math.isfinite(float(level.get("value", float("nan"))))
        and float(level["value"]) > price_value
    ]
    nearest: Dict[str, Dict[str, Any]] = {}
    if supports:
        nearest["support"] = max(supports, key=lambda level: float(level["value"]))
    if resistances:
        nearest["resistance"] = min(resistances, key=lambda level: float(level["value"]))
    return nearest


def _summarize_fibonacci_grid(levels: List[Dict[str, Any]]) -> tuple[str, Dict[str, int]]:
    counts = {"support": 0, "resistance": 0}
    for level in levels:
        if not isinstance(level, dict):
            continue
        level_type = str(level.get("type") or "").strip().lower()
        if level_type in counts:
            counts[level_type] += 1

    counts["total"] = counts["support"] + counts["resistance"]
    if counts["support"] and counts["resistance"]:
        return "both_sides", counts
    if counts["support"]:
        return "support_only", counts
    if counts["resistance"]:
        return "resistance_only", counts
    return "unclassified", counts


def _extract_fibonacci_swing_bounds(swing: Dict[str, Any]) -> Optional[tuple[float, float]]:
    anchor_low = swing.get("anchor_low") if isinstance(swing.get("anchor_low"), dict) else {}
    anchor_high = swing.get("anchor_high") if isinstance(swing.get("anchor_high"), dict) else {}
    low_value = _as_finite_float(anchor_low.get("value"))
    high_value = _as_finite_float(anchor_high.get("value"))
    if low_value is None or high_value is None:
        start = swing.get("start") if isinstance(swing.get("start"), dict) else {}
        end = swing.get("end") if isinstance(swing.get("end"), dict) else {}
        start_value = _as_finite_float(start.get("value"))
        end_value = _as_finite_float(end.get("value"))
        if start_value is None or end_value is None:
            return None
        low_value = min(start_value, end_value)
        high_value = max(start_value, end_value)
    if high_value < low_value:
        low_value, high_value = high_value, low_value
    return (float(low_value), float(high_value))


def _fibonacci_selection_reason(*, contains_current: bool) -> str:
    return (
        "most_recent_completed_swing_bracketing_current_price"
        if contains_current
        else "latest_completed_swing"
    )


def _copy_fibonacci_point(point: Dict[str, Any], *, allowed_keys: tuple[str, ...]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in allowed_keys:
        value = point.get(key)
        if value is not None:
            out[key] = value
    return out


def _build_fibonacci_payload_from_swing(
    *,
    start: Dict[str, Any],
    end: Dict[str, Any],
    current_price: Optional[float],
    timeframe: Optional[str],
    selection_reason: str,
) -> Optional[Dict[str, Any]]:
    start_type = str(start.get("type") or "")
    end_type = str(end.get("type") or "")
    start_value = _as_finite_float(start.get("value"))
    end_value = _as_finite_float(end.get("value"))
    if start_value is None or end_value is None:
        return None

    low_value = min(start_value, end_value)
    high_value = max(start_value, end_value)
    range_value = high_value - low_value
    if not math.isfinite(range_value) or range_value <= 0.0:
        return None

    direction = "up" if start_type == "support" and end_type == "resistance" else "down"
    current_position = _current_price_position(
        current_price=current_price,
        low_value=low_value,
        high_value=high_value,
    )
    contains_current = current_position == "within_swing"

    retracements: List[Dict[str, Any]] = []
    for ratio in _FIBONACCI_RETRACEMENTS:
        if direction == "up":
            fib_value = high_value - (float(ratio) * range_value)
        else:
            fib_value = low_value + (float(ratio) * range_value)
        retracements.append(
            _build_fibonacci_level(
                ratio=float(ratio),
                value=fib_value,
                current_price=current_price,
                kind="retracement",
            )
        )

    extension_projections = ["upside" if direction == "up" else "downside"]
    if current_position == "above_swing_high" and "upside" not in extension_projections:
        extension_projections.append("upside")
    elif current_position == "below_swing_low" and "downside" not in extension_projections:
        extension_projections.append("downside")

    extensions: List[Dict[str, Any]] = []
    for projection in extension_projections:
        for ratio in _FIBONACCI_EXTENSIONS:
            extension_offset = (float(ratio) - 1.0) * range_value
            fib_value = high_value + extension_offset if projection == "upside" else low_value - extension_offset
            extensions.append(
                _build_fibonacci_level(
                    ratio=float(ratio),
                    value=fib_value,
                    current_price=current_price,
                    kind="extension",
                    projection=projection,
                )
            )

    levels = sorted(retracements + extensions, key=lambda level: float(level["value"]))
    nearest = _nearest_fibonacci_levels(levels, current_price=current_price)
    fib_grid_coverage, fib_grid_counts = _summarize_fibonacci_grid(levels)
    low_anchor = start if start_value <= end_value else end
    high_anchor = start if start_value >= end_value else end

    return {
        "mode": "single",
        "timeframe": timeframe,
        "selection_rule": _FIBONACCI_SWING_SELECTION_RULE,
        "selection_reason": selection_reason,
        "current_price_used": None
        if current_price is None or not math.isfinite(float(current_price))
        else float(round(float(current_price), _FIBONACCI_LEVEL_DECIMALS)),
        "swing": {
            "direction": direction,
            "range": _round_output_price(range_value) if _round_output_price(range_value) is not None else range_value,
            "contains_current_price": contains_current,
            "current_price_position": current_position,
            "start": _copy_fibonacci_point(start, allowed_keys=("type", "index", "time", "value")),
            "end": _copy_fibonacci_point(end, allowed_keys=("type", "index", "time", "value")),
            "anchor_low": _copy_fibonacci_point(low_anchor, allowed_keys=("time", "value")),
            "anchor_high": _copy_fibonacci_point(high_anchor, allowed_keys=("time", "value")),
        },
        "retracements": retracements,
        "extensions": extensions,
        "levels": levels,
        "nearest": nearest,
        "fib_grid_coverage": fib_grid_coverage,
        "fib_grid_counts": fib_grid_counts,
    }


def _compute_fibonacci_payload(
    *,
    highs: np.ndarray,
    lows: np.ndarray,
    atr: np.ndarray,
    epochs: List[Optional[float]],
    current_price: Optional[float],
    timeframe: Optional[str],
) -> Optional[Dict[str, Any]]:
    candidates = _collect_local_extrema_candidates(highs, lows)
    swings = _filter_swing_candidates(candidates, atr=atr)
    if len(swings) < 2:
        return None

    best_pair: Optional[tuple[Dict[str, Any], Dict[str, Any]]] = None
    best_key: Optional[tuple[int, float, float]] = None
    best_contains_current = False
    selection_candidates: List[Dict[str, Any]] = []
    for index in range(1, len(swings)):
        start = swings[index - 1]
        end = swings[index]
        start_type = str(start.get("type") or "")
        end_type = str(end.get("type") or "")
        if start_type == end_type:
            continue
        try:
            start_value = float(start["value"])
            end_value = float(end["value"])
            start_index = int(start["index"])
            end_index = int(end["index"])
        except Exception:
            continue
        if end_index <= start_index:
            continue
        if not all(math.isfinite(value) for value in (start_value, end_value)):
            continue
        low_value = min(start_value, end_value)
        high_value = max(start_value, end_value)
        range_value = high_value - low_value
        if not math.isfinite(range_value) or range_value <= 0.0:
            continue
        start_epoch = epochs[start_index] if 0 <= start_index < len(epochs) else None
        contains_current = bool(
            current_price is not None
            and math.isfinite(float(current_price))
            and low_value <= float(current_price) <= high_value
        )
        end_epoch = epochs[end_index] if 0 <= end_index < len(epochs) else None
        recency_value = float(end_epoch) if end_epoch is not None and math.isfinite(float(end_epoch)) else float(end_index)
        key = (1 if contains_current else 0, recency_value, range_value)
        direction = "up" if start_type == "support" and end_type == "resistance" else "down"
        selection_candidates.append(
            {
                "candidate_id": f"{start_index}:{end_index}:{direction}",
                "contains_current_price": contains_current,
                "direction": direction,
                "range": _round_output_price(range_value) if _round_output_price(range_value) is not None else range_value,
                "start": {
                    "type": start_type,
                    "time": _format_time(start_epoch),
                    "value": _round_output_price(start_value) if _round_output_price(start_value) is not None else start_value,
                },
                "end": {
                    "type": end_type,
                    "time": _format_time(end_epoch),
                    "value": _round_output_price(end_value) if _round_output_price(end_value) is not None else end_value,
                },
                "_selection_key": key,
                "_recency": recency_value,
                "_range_raw": range_value,
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_pair = (dict(start), dict(end))
            best_contains_current = contains_current

    if best_pair is None:
        return None

    start, end = best_pair
    def _point_payload(point: Dict[str, Any]) -> Dict[str, Any]:
        point_index = int(point["index"])
        point_time = epochs[point_index] if 0 <= point_index < len(epochs) else None
        return {
            "type": str(point.get("type") or ""),
            "index": point_index,
            "time": _format_time(point_time),
            "value": _round_output_price(point["value"]) if _round_output_price(point["value"]) is not None else float(point["value"]),
        }

    payload = _build_fibonacci_payload_from_swing(
        start=_point_payload(start),
        end=_point_payload(end),
        current_price=current_price,
        timeframe=timeframe,
        selection_reason=_fibonacci_selection_reason(contains_current=best_contains_current),
    )
    if not isinstance(payload, dict):
        return payload

    best_start_index = int(start["index"])
    best_end_index = int(end["index"])
    best_direction = str(payload.get("swing", {}).get("direction") or "")
    best_recency = float(best_key[1]) if best_key is not None else float("-inf")
    best_range = float(best_key[2]) if best_key is not None else float("-inf")
    ranked_candidates = sorted(
        selection_candidates,
        key=lambda candidate: tuple(candidate.get("_selection_key", (0, -1.0, -1.0))),
        reverse=True,
    )
    payload_candidates: List[Dict[str, Any]] = []
    for rank, candidate in enumerate(ranked_candidates, start=1):
        candidate_out = {
            "rank": rank,
            "candidate_id": candidate.get("candidate_id"),
            "selected": bool(candidate.get("candidate_id") == f"{best_start_index}:{best_end_index}:{best_direction}"),
            "contains_current_price": bool(candidate.get("contains_current_price")),
            "direction": candidate.get("direction"),
            "range": candidate.get("range"),
            "start": dict(candidate.get("start") or {}),
            "end": dict(candidate.get("end") or {}),
        }
        if not candidate_out["selected"]:
            if bool(candidate.get("contains_current_price")) != best_contains_current:
                rejected_reason = "does_not_bracket_current_price"
            elif float(candidate.get("_recency", float("-inf"))) < best_recency:
                rejected_reason = "older_than_selected_candidate"
            elif float(candidate.get("_range_raw", float("-inf"))) < best_range:
                rejected_reason = "smaller_than_selected_candidate"
            else:
                rejected_reason = "lower_selection_priority"
            candidate_out["rejected_reason"] = rejected_reason
        payload_candidates.append(candidate_out)

    payload["selection_candidates"] = payload_candidates
    payload["selection_summary"] = {
        "candidate_count": len(payload_candidates),
        "selected_candidate_id": f"{best_start_index}:{best_end_index}:{best_direction}",
        "selection_basis": "contains_current_price_then_recency_then_range",
    }
    return payload

def _select_auto_fibonacci_payload(
    results: List[Dict[str, Any]],
    *,
    current_price: Optional[float],
) -> Optional[Dict[str, Any]]:
    candidates: List[tuple[tuple[int, float, float], str, Dict[str, Any], bool]] = []
    available_timeframes: List[str] = []
    timeframe_candidates: List[Dict[str, Any]] = []
    for payload in results:
        timeframe = str(payload.get("timeframe") or "").upper()
        fibonacci = payload.get("fibonacci")
        if not timeframe:
            continue
        available_timeframes.append(timeframe)
        if not isinstance(fibonacci, dict):
            continue
        swing = fibonacci.get("swing") if isinstance(fibonacci.get("swing"), dict) else {}
        bounds = _extract_fibonacci_swing_bounds(swing)
        if bounds is not None and current_price is not None and math.isfinite(float(current_price)):
            low_value, high_value = bounds
            contains_current = low_value <= float(current_price) <= high_value
        else:
            contains_current = bool(swing.get("contains_current_price"))
        end_time = None
        end_payload = swing.get("end")
        if isinstance(end_payload, dict):
            end_time = _parse_output_time(end_payload.get("time"))
        weight = _timeframe_weight(timeframe)
        recency = float(end_time) if end_time is not None and math.isfinite(float(end_time)) else -1.0
        timeframe_candidates.append(
            {
                "timeframe": timeframe,
                "contains_current_price": contains_current,
                "timeframe_weight": float(weight),
                "end_time": end_payload.get("time") if isinstance(end_payload, dict) else None,
                "_selection_key": (1 if contains_current else 0, float(weight), recency),
                "_recency": recency,
            }
        )
        candidates.append(((1 if contains_current else 0, float(weight), recency), timeframe, dict(fibonacci), contains_current))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected_key, selected_timeframe, selected_payload, contains_current = candidates[0]
    swing = selected_payload.get("swing") if isinstance(selected_payload.get("swing"), dict) else {}
    start = swing.get("start") if isinstance(swing.get("start"), dict) else None
    end = swing.get("end") if isinstance(swing.get("end"), dict) else None
    rebuilt = None
    if isinstance(start, dict) and isinstance(end, dict):
        rebuilt = _build_fibonacci_payload_from_swing(
            start=dict(start),
            end=dict(end),
            current_price=current_price,
            timeframe=selected_timeframe,
            selection_reason=_fibonacci_selection_reason(contains_current=contains_current),
        )
    selected_out = rebuilt if isinstance(rebuilt, dict) else dict(selected_payload)
    selected_out["mode"] = "auto"
    selected_out["selected_timeframe"] = selected_timeframe
    selected_out["available_timeframes"] = sorted(set(available_timeframes), key=_timeframe_sort_key)
    selected_out["timeframe_selection_rule"] = _FIBONACCI_TIMEFRAME_SELECTION_RULE
    ranked_timeframes = sorted(
        timeframe_candidates,
        key=lambda candidate: tuple(candidate.get("_selection_key", (0, 0.0, -1.0))),
        reverse=True,
    )
    timeframe_selection_candidates: List[Dict[str, Any]] = []
    for rank, candidate in enumerate(ranked_timeframes, start=1):
        candidate_out = {
            "rank": rank,
            "timeframe": candidate.get("timeframe"),
            "selected": str(candidate.get("timeframe")) == selected_timeframe,
            "contains_current_price": bool(candidate.get("contains_current_price")),
            "timeframe_weight": candidate.get("timeframe_weight"),
            "end_time": candidate.get("end_time"),
        }
        if not candidate_out["selected"]:
            if bool(candidate.get("contains_current_price")) != contains_current:
                rejected_reason = "timeframe_grid_does_not_bracket_current_price"
            elif float(candidate.get("timeframe_weight", 0.0)) < float(selected_key[1]):
                rejected_reason = "lower_timeframe_weight"
            elif float(candidate.get("_recency", float("-inf"))) < float(selected_key[2]):
                rejected_reason = "older_than_selected_timeframe"
            else:
                rejected_reason = "lower_selection_priority"
            candidate_out["rejected_reason"] = rejected_reason
        timeframe_selection_candidates.append(candidate_out)

    selected_out["timeframe_selection_candidates"] = timeframe_selection_candidates
    summary = selected_out.get("selection_summary") if isinstance(selected_out.get("selection_summary"), dict) else {}
    summary.update(
        {
            "timeframe_candidate_count": len(timeframe_selection_candidates),
            "selected_timeframe": selected_timeframe,
            "timeframe_selection_basis": "contains_current_price_then_timeframe_weight_then_recency",
        }
    )
    selected_out["selection_summary"] = summary
    return selected_out


def _collect_support_resistance_warnings(
    *,
    fibonacci: Optional[Dict[str, Any]],
    coverage_gaps: Optional[Dict[str, Dict[str, Any]]] = None,
    zone_overlap: Optional[Dict[str, Any]] = None,
    support_count: Optional[int] = None,
    resistance_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []

    swing = fibonacci.get("swing") if isinstance(fibonacci, dict) and isinstance(fibonacci.get("swing"), dict) else {}
    current_position = str(swing.get("current_price_position") or "").strip()
    nearest = fibonacci.get("nearest") if isinstance(fibonacci, dict) and isinstance(fibonacci.get("nearest"), dict) else {}
    if current_position == "above_swing_high":
        warnings.append(
            {
                "code": "fibonacci_price_above_swing_high",
                "message": (
                    "Current price is above the selected Fibonacci swing high; "
                    "upside extension targets were added to avoid an exhausted grid."
                ),
                "selected_timeframe": None if not isinstance(fibonacci, dict) else fibonacci.get("selected_timeframe", fibonacci.get("timeframe")),
                "nearest_resistance": nearest.get("resistance"),
            }
        )
    elif current_position == "below_swing_low":
        warnings.append(
            {
                "code": "fibonacci_price_below_swing_low",
                "message": (
                    "Current price is below the selected Fibonacci swing low; "
                    "downside extension targets were added to avoid an exhausted grid."
                ),
                "selected_timeframe": None if not isinstance(fibonacci, dict) else fibonacci.get("selected_timeframe", fibonacci.get("timeframe")),
                "nearest_support": nearest.get("support"),
            }
        )

    fib_grid_coverage = str(fibonacci.get("fib_grid_coverage") or "").strip() if isinstance(fibonacci, dict) else ""
    if fib_grid_coverage == "support_only":
        warnings.append(
            {
                "code": "fibonacci_grid_support_only",
                "grid_coverage": fib_grid_coverage,
                "message": (
                    "All classified Fibonacci levels are below the current price; "
                    "the selected grid currently offers support only."
                ),
                "selected_timeframe": None if not isinstance(fibonacci, dict) else fibonacci.get("selected_timeframe", fibonacci.get("timeframe")),
            }
        )
    elif fib_grid_coverage == "resistance_only":
        warnings.append(
            {
                "code": "fibonacci_grid_resistance_only",
                "grid_coverage": fib_grid_coverage,
                "message": (
                    "All classified Fibonacci levels are above the current price; "
                    "the selected grid currently offers resistance only."
                ),
                "selected_timeframe": None if not isinstance(fibonacci, dict) else fibonacci.get("selected_timeframe", fibonacci.get("timeframe")),
            }
        )

    overlap_width = _as_finite_float((zone_overlap or {}).get("overlap_width"))
    if isinstance(zone_overlap, dict) and overlap_width is not None and overlap_width > 0.0:
        current_price_in_overlap = bool(zone_overlap.get("current_price_in_overlap"))
        message = "Nearest support and resistance zones overlap."
        if current_price_in_overlap:
            message += " Current price sits inside both zones, so the range signal is diluted."
        else:
            message += " The market remains structurally compressed until price exits the overlap."
        warnings.append(
            {
                "code": "overlapping_nearest_zones",
                "overlap_low": zone_overlap.get("overlap_low"),
                "overlap_high": zone_overlap.get("overlap_high"),
                "overlap_width": float(round(overlap_width, 6)),
                "current_price_in_overlap": current_price_in_overlap,
                "message": message,
            }
        )

    for side, gap in (coverage_gaps or {}).items():
        if not isinstance(gap, dict) or not bool(gap.get("is_structural_vacuum")):
            continue
        gap_pct = _as_finite_float(gap.get("distance_pct"))
        if gap_pct is None:
            continue
        beyond_filter = bool(gap.get("beyond_max_distance_filter"))
        suffix = " It is also beyond the active max_distance_pct filter." if beyond_filter else ""
        warnings.append(
            {
                "code": f"structural_gap_{side}",
                "side": side,
                "distance_pct": float(round(gap_pct, 6)),
                "level_value": gap.get("level_value"),
                "message": (
                    f"Nearest {side} level is {gap_pct:.1f}% away; "
                    f"historical structure is sparse on the {side} side of the market.{suffix}"
                ),
            }
        )

    if support_count == 0 and (resistance_count or 0) > 0:
        warnings.append(
            {
                "code": "no_support_levels",
                "message": (
                    "No support levels qualified in the lookback window; price is "
                    "likely near a recent low so most structure sits above it. "
                    f"{_EMPTY_SIDE_REMEDIATION} to surface deeper supports."
                ),
            }
        )
    elif resistance_count == 0 and (support_count or 0) > 0:
        warnings.append(
            {
                "code": "no_resistance_levels",
                "message": (
                    "No resistance levels qualified in the lookback window; price is "
                    "likely near a recent high so most structure sits below it. "
                    f"{_EMPTY_SIDE_REMEDIATION} to surface higher resistances."
                ),
            }
        )

    return warnings


def _annotate_strength_metrics(levels: List[Dict[str, Any]]) -> None:
    if not levels:
        return

    score_values = [_as_finite_float(level.get("score")) for level in levels]
    finite_scores = [score if score is not None else 0.0 for score in score_values]
    min_score = min(finite_scores)
    max_score = max(finite_scores)
    spread = max_score - min_score
    count = len(levels)
    ranked_indexes = sorted(
        range(count),
        key=lambda index: (-finite_scores[index], index),
    )
    rank_by_index = {
        index: rank for rank, index in enumerate(ranked_indexes, start=1)
    }

    for index, (level, score) in enumerate(zip(levels, score_values)):
        rank = rank_by_index[index]
        level["strength_rank"] = rank
        percentile = 1.0 if count == 1 else 1.0 - (float(rank - 1) / float(count - 1))
        if score is None or spread <= 1e-12:
            normalized = 1.0
        else:
            normalized = (score - min_score) / spread
        level["strength_percentile"] = float(round(max(0.0, min(1.0, percentile)), 4))
        level["strength_score_normalized"] = float(
            round(max(0.0, min(1.0, normalized)), 4)
        )


def _annotate_proximity_ranks(levels: List[Dict[str, Any]]) -> None:
    if not levels:
        return
    distances: List[tuple[float, float]] = []
    for level in levels:
        distance_pct = _as_finite_float(level.get("distance_pct"))
        distance = _as_finite_float(level.get("distance"))
        distances.append(
            (
                abs(distance_pct) if distance_pct is not None else float("inf"),
                abs(distance) if distance is not None else 0.0,
            )
        )
    ranked_indexes = sorted(
        range(len(levels)),
        key=lambda index: (distances[index][0], distances[index][1], index),
    )
    for rank, index in enumerate(ranked_indexes, start=1):
        levels[index]["proximity_rank"] = rank


def _drop_zero_score_when_stronger_exist(levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter zero-score levels out of the ranked output when stronger levels exist.

    A clamped-to-zero score means the level was broken through (breakout penalty
    overwhelmed any base strength and role-reversal bonus). Returning such a
    level alongside stronger ones misleads users who reasonably expect every
    ranked level to be actionable. When every candidate is zero-score we keep
    the list as-is so the tool still surfaces *some* structure for the user to
    inspect, rather than returning an empty result.
    """
    if not levels:
        return levels

    def _score_or_zero(level: Dict[str, Any]) -> float:
        value = _as_finite_float(level.get("score"))
        return 0.0 if value is None else float(value)

    has_positive = any(_score_or_zero(level) > 0.0 for level in levels)
    if not has_positive:
        return levels
    return [level for level in levels if _score_or_zero(level) > 0.0]


def _normalize_max_distance_pct(max_distance_pct: Optional[float]) -> Optional[float]:
    if max_distance_pct is None:
        return None
    value = _as_finite_float(max_distance_pct)
    if value is None or value < 0.0:
        raise ValueError("max_distance_pct must be a finite non-negative number")
    return float(value)


def _nearest_level_by_distance(levels: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not levels:
        return None
    ranked: List[tuple[float, Dict[str, Any]]] = []
    for level in levels:
        distance_pct = _as_finite_float(level.get("distance_pct"))
        if distance_pct is None:
            continue
        ranked.append((abs(distance_pct), level))
    if ranked:
        ranked.sort(key=lambda item: (item[0], abs(_as_finite_float(item[1].get("distance")) or 0.0)))
        return ranked[0][1]
    return levels[0]


def _build_coverage_gaps(
    *,
    support_levels: List[Dict[str, Any]],
    resistance_levels: List[Dict[str, Any]],
    max_distance_pct: Optional[float],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    threshold = float(_DEFAULT_STRUCTURE_GAP_WARNING_PCT)
    max_filter = _normalize_max_distance_pct(max_distance_pct)

    for side, levels in (("support", support_levels), ("resistance", resistance_levels)):
        nearest = _nearest_level_by_distance(levels)
        if not isinstance(nearest, dict):
            continue
        distance_pct = _as_finite_float(nearest.get("distance_pct"))
        if distance_pct is None:
            continue
        gap_pct = abs(distance_pct)
        entry: Dict[str, Any] = {
            "side": side,
            "level_value": nearest.get("value"),
            "distance_pct": float(round(gap_pct, 6)),
            "threshold_pct": float(round(threshold, 6)),
            "is_structural_vacuum": bool(gap_pct >= threshold),
        }
        if max_filter is not None:
            entry["beyond_max_distance_filter"] = bool(gap_pct > max_filter)
        out[side] = entry
    return out


def _build_zone_overlap(
    *,
    support_level: Optional[Dict[str, Any]],
    resistance_level: Optional[Dict[str, Any]],
    current_price: Optional[float],
) -> Optional[Dict[str, Any]]:
    if not isinstance(support_level, dict) or not isinstance(resistance_level, dict):
        return None

    support_low = _as_finite_float(support_level.get("zone_low"))
    support_high = _as_finite_float(support_level.get("zone_high"))
    resistance_low = _as_finite_float(resistance_level.get("zone_low"))
    resistance_high = _as_finite_float(resistance_level.get("zone_high"))
    if None in (support_low, support_high, resistance_low, resistance_high):
        return None

    overlap_low = max(float(support_low), float(resistance_low))
    overlap_high = min(float(support_high), float(resistance_high))
    if overlap_high - overlap_low <= 1e-9:
        return None

    price_value = _as_finite_float(current_price)
    current_price_in_overlap = bool(
        price_value is not None and overlap_low <= float(price_value) <= overlap_high
    )
    return {
        "support_value": support_level.get("value"),
        "resistance_value": resistance_level.get("value"),
        "support_zone_low": float(round(float(support_low), 6)),
        "support_zone_high": float(round(float(support_high), 6)),
        "resistance_zone_low": float(round(float(resistance_low), 6)),
        "resistance_zone_high": float(round(float(resistance_high), 6)),
        "overlap_low": float(round(overlap_low, 6)),
        "overlap_high": float(round(overlap_high, 6)),
        "overlap_width": float(round(overlap_high - overlap_low, 6)),
        "current_price_in_overlap": current_price_in_overlap,
    }


def _filter_levels_by_distance(
    levels: List[Dict[str, Any]],
    *,
    max_distance_pct: Optional[float],
) -> List[Dict[str, Any]]:
    threshold = _normalize_max_distance_pct(max_distance_pct)
    if threshold is None:
        return list(levels)

    filtered: List[Dict[str, Any]] = []
    for level in levels:
        distance_pct = _as_finite_float(level.get("distance_pct"))
        if distance_pct is None or abs(distance_pct) <= threshold:
            filtered.append(level)
    return filtered


def _build_merge_signature(level: Dict[str, Any], timeframe: str) -> Optional[Dict[str, Any]]:
    first_touch = _parse_output_time(level.get("first_touch"))
    last_touch = _parse_output_time(level.get("last_touch"))
    if first_touch is None and last_touch is None:
        return None
    if first_touch is None:
        first_touch = last_touch
    if last_touch is None:
        last_touch = first_touch
    if first_touch is None or last_touch is None:
        return None
    start = min(float(first_touch), float(last_touch))
    end = max(float(first_touch), float(last_touch))
    timeframe_seconds = float(_timeframe_seconds(timeframe) or 3600)
    return {
        "timeframe": str(timeframe).upper(),
        "dominant_source": str(level.get("dominant_source") or level.get("type") or "mixed"),
        "window_start": start,
        "window_end": end,
        "pad_seconds": timeframe_seconds,
    }


def _signatures_match_same_event(existing: Dict[str, Any], incoming: Dict[str, Any]) -> bool:
    existing_source = str(existing.get("dominant_source") or "mixed")
    incoming_source = str(incoming.get("dominant_source") or "mixed")
    if existing_source != incoming_source and "mixed" not in {existing_source, incoming_source}:
        return False

    try:
        existing_start = float(existing["window_start"])
        existing_end = float(existing["window_end"])
        incoming_start = float(incoming["window_start"])
        incoming_end = float(incoming["window_end"])
    except Exception:
        return False

    if not all(math.isfinite(value) for value in (existing_start, existing_end, incoming_start, incoming_end)):
        return False

    gap = 0.0
    if existing_end < incoming_start:
        gap = incoming_start - existing_end
    elif incoming_end < existing_start:
        gap = existing_start - incoming_end
    pad = max(
        float(existing.get("pad_seconds", 0.0)),
        float(incoming.get("pad_seconds", 0.0)),
        60.0,
    )
    return gap <= (pad * 2.0)


def _cluster_has_same_event_signature(
    cluster: Dict[str, Any],
    *,
    signature: Optional[Dict[str, Any]],
    timeframe: str,
) -> bool:
    if signature is None:
        return False
    for existing in cluster.get("merge_signatures", []):
        if str(existing.get("timeframe") or "").upper() == str(timeframe).upper():
            continue
        if _signatures_match_same_event(existing, signature):
            return True
    return False


def merge_support_resistance_results(  # noqa: C901
    results: List[Dict[str, Any]],
    *,
    symbol: Optional[str] = None,
    timeframe: str = "auto",
    limit: Optional[int] = None,
    tolerance_fraction: float = 0.0015,
    min_touches: int = 2,
    max_levels: int = 4,
    reaction_bars: int = _DEFAULT_REACTION_BARS,
    adx_period: int = _DEFAULT_ADX_PERIOD,
    decay_half_life_bars: Optional[int] = None,
    max_distance_pct: Optional[float] = None,
    volume_weighting: str = "off",
) -> Dict[str, Any]:
    if not results:
        raise ValueError("No history available")
    max_distance_value = _normalize_max_distance_pct(max_distance_pct)

    adaptive_tolerance_pairs: List[tuple[float, float]] = []
    adaptive_reaction_pairs: List[tuple[float, float]] = []
    volatility_ratio_pairs: List[tuple[float, float]] = []
    current_atr_pairs: List[tuple[float, float]] = []
    baseline_atr_pairs: List[tuple[float, float]] = []
    requested_volume_weighting = _normalize_volume_weighting(volume_weighting)
    derived_volume_weighting = _normalize_volume_weighting(results[0].get("volume_weighting"))
    volume_weighting_mode = derived_volume_weighting if derived_volume_weighting != "off" else requested_volume_weighting
    volume_sources_seen: set[str] = set()
    for payload in results:
        tf = str(payload.get("timeframe") or "").upper()
        tf_weight = _timeframe_weight(tf)
        volume_source = str(payload.get("volume_source") or "").strip()
        if volume_source:
            volume_sources_seen.add(volume_source)
        for key, bucket, scale in (
            ("effective_tolerance_pct", adaptive_tolerance_pairs, 0.01),
            ("effective_reaction_bars", adaptive_reaction_pairs, 1.0),
            ("volatility_ratio", volatility_ratio_pairs, 1.0),
            ("current_atr_pct", current_atr_pairs, 0.01),
            ("baseline_atr_pct", baseline_atr_pairs, 0.01),
        ):
            try:
                value = payload.get(key)
                if value is None:
                    continue
                numeric = float(value)
                if math.isfinite(numeric):
                    bucket.append((tf_weight, numeric * scale))
            except Exception:
                continue

    merge_tolerance_value = _weighted_average(adaptive_tolerance_pairs)
    if merge_tolerance_value is None:
        merge_tolerance_value = float(tolerance_fraction)
    merge_reaction_value = _weighted_average(adaptive_reaction_pairs)
    if merge_reaction_value is None:
        merge_reaction_value = float(reaction_bars)
    merge_volatility_ratio = _weighted_average(volatility_ratio_pairs)
    if merge_volatility_ratio is None:
        merge_volatility_ratio = 1.0
    merge_current_atr_fraction = _weighted_average(current_atr_pairs)
    merge_baseline_atr_fraction = _weighted_average(baseline_atr_pairs)

    clusters: List[Dict[str, Any]] = []
    for payload in results:
        tf = str(payload.get("timeframe") or "").upper()
        tf_weight = _timeframe_weight(tf)
        for level in payload.get("levels") or []:
            try:
                value = float(level["value"])
                raw_score = float(level.get("score", 0.0))
            except Exception:
                continue
            weighted_score = raw_score * tf_weight
            if not math.isfinite(value) or not math.isfinite(weighted_score) or weighted_score <= 0.0:
                continue

            best_cluster = _match_nearest_cluster(clusters, value, merge_tolerance_value)

            breakdown = level.get("score_breakdown") if isinstance(level.get("score_breakdown"), dict) else {}
            source_tests = level.get("source_tests") if isinstance(level.get("source_tests"), dict) else {}
            source_episodes = level.get("source_episodes") if isinstance(level.get("source_episodes"), dict) else {}
            breakout_analysis = level.get("breakout_analysis") if isinstance(level.get("breakout_analysis"), dict) else {}
            first_touch = str(level.get("first_touch") or "").strip() or None
            last_touch = str(level.get("last_touch") or "").strip() or None
            touches = max(1, int(level.get("touches", 1)))
            episodes = max(1, int(level.get("episodes", touches)))
            avg_bounce = level.get("avg_bounce_atr")
            avg_adx = level.get("avg_pretest_adx")
            avg_volume_ratio = level.get("avg_test_volume_ratio")
            volume_source = str(level.get("volume_source") or "").strip() or None
            base_score = float(
                breakdown.get(
                    "base",
                    float(breakdown.get("retests", 0.0))
                    + float(breakdown.get("bounce", 0.0))
                    + float(breakdown.get("adx", 0.0))
                    + float(breakdown.get("volume", 0.0)),
                )
            )
            breakout_penalty = float(breakdown.get("breakout_penalty", 0.0))
            role_reversal_bonus = float(breakdown.get("role_reversal_bonus", 0.0))
            decisive_break_count = int(breakout_analysis.get("decisive_break_count", 0))
            role_reversal_count = int(breakout_analysis.get("role_reversal_count", 0))
            avg_breach_atr = breakout_analysis.get("avg_breach_atr")
            zone_low = level.get("zone_low")
            zone_high = level.get("zone_high")
            zone_width_atr = level.get("zone_width_atr")
            status = str(level.get("status") or "").strip() or None
            merge_signature = _build_merge_signature(level, tf)
            mtf_confirmation_bonus = float(breakdown.get("mtf_confirmation_bonus", 0.0))

            if best_cluster is None:
                cluster = {
                    "value": float(value),
                    "weight_sum": float(weighted_score),
                    "touches": int(touches),
                    "episodes": int(episodes),
                    "score_base": float(base_score) * tf_weight,
                    "score": float(weighted_score),
                    "retest_score": float(breakdown.get("retests", raw_score)) * tf_weight,
                    "bounce_score": float(breakdown.get("bounce", 0.0)) * tf_weight,
                    "adx_score": float(breakdown.get("adx", 0.0)) * tf_weight,
                    "volume_score": float(breakdown.get("volume", 0.0)) * tf_weight,
                    "breakout_penalty": float(breakout_penalty) * tf_weight,
                    "role_reversal_bonus": float(role_reversal_bonus) * tf_weight,
                    "mtf_confirmation_bonus": float(mtf_confirmation_bonus),
                    "support_tests": int(source_tests.get("support", 0)),
                    "resistance_tests": int(source_tests.get("resistance", 0)),
                    "support_episodes": int(source_episodes.get("support", source_tests.get("support", 0))),
                    "resistance_episodes": int(source_episodes.get("resistance", source_tests.get("resistance", 0))),
                    "first_touch": first_touch,
                    "last_touch": last_touch,
                    "zone_low": None if zone_low is None else float(zone_low),
                    "zone_high": None if zone_high is None else float(zone_high),
                    "zone_width_atr_sum": 0.0,
                    "zone_weight_sum": 0.0,
                    "bounce_metric_sum": 0.0,
                    "adx_metric_sum": 0.0,
                    "metric_weight_sum": 0.0,
                    "volume_metric_sum": 0.0,
                    "volume_metric_weight_sum": 0.0,
                    "volume_sources": {volume_source} if volume_source else set(),
                    "decisive_break_count": int(decisive_break_count),
                    "role_reversal_count": int(role_reversal_count),
                    "avg_breach_atr_sum": 0.0,
                    "breakout_metric_weight_sum": 0.0,
                    "last_break_time": breakout_analysis.get("last_break_time"),
                    "status": status or "intact",
                    "cross_timeframe_dedupe_count": 0,
                    "deduped_timeframes": set(),
                    "merge_signatures": [merge_signature] if merge_signature is not None else [],
                    "source_timeframes": {tf} if tf else set(),
                    "timeframe_scores": {tf: float(weighted_score)} if tf else {},
                    "timeframe_raw_scores": {tf: float(raw_score)} if tf else {},
                    "timeframe_touches": {tf: int(touches)} if tf else {},
                    "timeframe_episodes": {tf: int(episodes)} if tf else {},
                    "timeframe_weights": {tf: float(tf_weight)} if tf else {},
                    "timeframe_merge_modes": {tf: "full"} if tf else {},
                }
                if isinstance(avg_bounce, (int, float)) and math.isfinite(float(avg_bounce)):
                    cluster["bounce_metric_sum"] = float(avg_bounce) * float(weighted_score)
                    cluster["metric_weight_sum"] += float(weighted_score)
                if isinstance(avg_adx, (int, float)) and math.isfinite(float(avg_adx)):
                    cluster["adx_metric_sum"] = float(avg_adx) * float(weighted_score)
                    if cluster["metric_weight_sum"] <= 0.0:
                        cluster["metric_weight_sum"] += float(weighted_score)
                if isinstance(avg_volume_ratio, (int, float)) and math.isfinite(float(avg_volume_ratio)):
                    cluster["volume_metric_sum"] = float(avg_volume_ratio) * float(weighted_score)
                    cluster["volume_metric_weight_sum"] = float(weighted_score)
                if zone_width_atr is not None:
                    try:
                        z_width_atr = float(zone_width_atr)
                        if math.isfinite(z_width_atr):
                            cluster["zone_width_atr_sum"] = z_width_atr * float(weighted_score)
                            cluster["zone_weight_sum"] = float(weighted_score)
                    except Exception:
                        pass
                if isinstance(avg_breach_atr, (int, float)) and math.isfinite(float(avg_breach_atr)):
                    cluster["avg_breach_atr_sum"] = float(avg_breach_atr) * max(float(decisive_break_count), 1.0)
                    cluster["breakout_metric_weight_sum"] = max(float(decisive_break_count), 1.0)
                clusters.append(cluster)
                continue

            cluster = best_cluster
            is_same_event = bool(
                tf
                and tf not in cluster.get("source_timeframes", set())
                and _cluster_has_same_event_signature(cluster, signature=merge_signature, timeframe=tf)
            )
            contribution_scale = _DEFAULT_MTF_DEDUPE_FACTOR if is_same_event else 1.0
            scaled_weighted_score = float(weighted_score) * contribution_scale
            confirmation_bonus = 0.0
            if is_same_event:
                prior_scores = [
                    float(cluster["timeframe_scores"].get(existing_tf, 0.0))
                    for existing_tf in cluster.get("source_timeframes", set())
                    if float(cluster["timeframe_scores"].get(existing_tf, 0.0)) > 0.0
                ]
                if prior_scores:
                    confirmation_bonus = min(float(weighted_score), max(prior_scores)) * _DEFAULT_MTF_CONFIRMATION_BONUS
            effective_weighted_score = scaled_weighted_score + confirmation_bonus
            new_weight = float(cluster["weight_sum"]) + effective_weighted_score
            cluster["value"] = (float(cluster["value"]) * float(cluster["weight_sum"]) + value * effective_weighted_score) / new_weight
            cluster["weight_sum"] = new_weight
            cluster["touches"] = int(cluster["touches"]) + (1 if is_same_event else int(touches))
            cluster["episodes"] = int(cluster.get("episodes", 0)) + (0 if is_same_event else int(episodes))
            cluster["score_base"] = float(cluster["score_base"]) + float(base_score) * tf_weight * contribution_scale
            cluster["score"] = float(cluster["score"]) + effective_weighted_score
            cluster["retest_score"] = float(cluster["retest_score"]) + float(breakdown.get("retests", raw_score)) * tf_weight * contribution_scale
            cluster["bounce_score"] = float(cluster["bounce_score"]) + float(breakdown.get("bounce", 0.0)) * tf_weight * contribution_scale
            cluster["adx_score"] = float(cluster["adx_score"]) + float(breakdown.get("adx", 0.0)) * tf_weight * contribution_scale
            cluster["volume_score"] = float(cluster.get("volume_score", 0.0)) + float(breakdown.get("volume", 0.0)) * tf_weight * contribution_scale
            cluster["breakout_penalty"] = float(cluster["breakout_penalty"]) + float(breakout_penalty) * tf_weight * contribution_scale
            cluster["role_reversal_bonus"] = float(cluster["role_reversal_bonus"]) + float(role_reversal_bonus) * tf_weight * contribution_scale
            cluster["mtf_confirmation_bonus"] = float(cluster.get("mtf_confirmation_bonus", 0.0)) + confirmation_bonus
            cluster["support_tests"] = int(cluster["support_tests"]) + (
                1 if is_same_event and int(source_tests.get("support", 0)) > 0 else int(source_tests.get("support", 0))
            )
            cluster["resistance_tests"] = int(cluster["resistance_tests"]) + (
                1 if is_same_event and int(source_tests.get("resistance", 0)) > 0 else int(source_tests.get("resistance", 0))
            )
            cluster["support_episodes"] = int(cluster.get("support_episodes", 0)) + (
                0 if is_same_event else int(source_episodes.get("support", source_tests.get("support", 0)))
            )
            cluster["resistance_episodes"] = int(cluster.get("resistance_episodes", 0)) + (
                0 if is_same_event else int(source_episodes.get("resistance", source_tests.get("resistance", 0)))
            )
            if first_touch is not None:
                current_first = cluster.get("first_touch")
                cluster["first_touch"] = first_touch if current_first is None else min(str(current_first), first_touch)
            if last_touch is not None:
                current_last = cluster.get("last_touch")
                cluster["last_touch"] = last_touch if current_last is None else max(str(current_last), last_touch)
            if zone_low is not None:
                low = float(zone_low)
                cluster["zone_low"] = low if cluster.get("zone_low") is None else min(float(cluster["zone_low"]), low)
            if zone_high is not None:
                high = float(zone_high)
                cluster["zone_high"] = high if cluster.get("zone_high") is None else max(float(cluster["zone_high"]), high)
            if tf:
                cluster["source_timeframes"].add(tf)
                cluster["timeframe_scores"][tf] = float(cluster["timeframe_scores"].get(tf, 0.0)) + effective_weighted_score
                cluster["timeframe_raw_scores"][tf] = float(cluster["timeframe_raw_scores"].get(tf, 0.0)) + float(raw_score)
                cluster["timeframe_touches"][tf] = int(cluster["timeframe_touches"].get(tf, 0)) + (1 if is_same_event else int(touches))
                cluster["timeframe_episodes"][tf] = int(cluster["timeframe_episodes"].get(tf, 0)) + int(episodes)
                cluster["timeframe_weights"][tf] = float(tf_weight)
                cluster["timeframe_merge_modes"][tf] = "deduped" if is_same_event else "full"
            if volume_source:
                cluster.setdefault("volume_sources", set()).add(volume_source)
            if isinstance(avg_bounce, (int, float)) and math.isfinite(float(avg_bounce)):
                cluster["bounce_metric_sum"] = float(cluster["bounce_metric_sum"]) + float(avg_bounce) * scaled_weighted_score
                cluster["metric_weight_sum"] = float(cluster["metric_weight_sum"]) + scaled_weighted_score
            if isinstance(avg_adx, (int, float)) and math.isfinite(float(avg_adx)):
                cluster["adx_metric_sum"] = float(cluster["adx_metric_sum"]) + float(avg_adx) * scaled_weighted_score
                if not (isinstance(avg_bounce, (int, float)) and math.isfinite(float(avg_bounce))):
                    cluster["metric_weight_sum"] = float(cluster["metric_weight_sum"]) + scaled_weighted_score
            if isinstance(avg_volume_ratio, (int, float)) and math.isfinite(float(avg_volume_ratio)):
                cluster["volume_metric_sum"] = float(cluster.get("volume_metric_sum", 0.0)) + float(avg_volume_ratio) * scaled_weighted_score
                cluster["volume_metric_weight_sum"] = float(cluster.get("volume_metric_weight_sum", 0.0)) + scaled_weighted_score
            if zone_width_atr is not None:
                try:
                    z_width_atr = float(zone_width_atr)
                    if math.isfinite(z_width_atr):
                        cluster["zone_width_atr_sum"] = float(cluster["zone_width_atr_sum"]) + z_width_atr * scaled_weighted_score
                        cluster["zone_weight_sum"] = float(cluster["zone_weight_sum"]) + scaled_weighted_score
                except Exception:
                    pass
            if is_same_event:
                cluster["decisive_break_count"] = max(int(cluster["decisive_break_count"]), int(decisive_break_count))
                cluster["role_reversal_count"] = max(int(cluster["role_reversal_count"]), int(role_reversal_count))
                cluster["cross_timeframe_dedupe_count"] = int(cluster.get("cross_timeframe_dedupe_count", 0)) + 1
                if tf:
                    cluster["deduped_timeframes"].add(tf)
            else:
                cluster["decisive_break_count"] = int(cluster["decisive_break_count"]) + int(decisive_break_count)
                cluster["role_reversal_count"] = int(cluster["role_reversal_count"]) + int(role_reversal_count)
            if isinstance(avg_breach_atr, (int, float)) and math.isfinite(float(avg_breach_atr)):
                weight = max(float(decisive_break_count), 1.0) * contribution_scale
                cluster["avg_breach_atr_sum"] = float(cluster["avg_breach_atr_sum"]) + float(avg_breach_atr) * weight
                cluster["breakout_metric_weight_sum"] = float(cluster["breakout_metric_weight_sum"]) + weight
            last_break_time = breakout_analysis.get("last_break_time")
            if isinstance(last_break_time, str) and last_break_time.strip():
                current_break_time = cluster.get("last_break_time")
                cluster["last_break_time"] = last_break_time if current_break_time is None else max(str(current_break_time), last_break_time)
            if status == "role_reversal":
                cluster["status"] = status
            elif status and cluster.get("status") != "role_reversal":
                cluster["status"] = status
            if merge_signature is not None:
                cluster.setdefault("merge_signatures", []).append(merge_signature)

    usable_clusters = [cluster for cluster in clusters if int(cluster.get("episodes", cluster.get("touches", 0))) >= max(1, int(min_touches))]

    current_price = _pick_first_current_price(results)
    formatted_levels: List[Dict[str, Any]] = []
    for cluster in usable_clusters:
        zone_weight_sum = float(cluster.get("zone_weight_sum", 0.0))
        if zone_weight_sum > 0.0:
            cluster["zone_width_atr"] = float(cluster["zone_width_atr_sum"]) / zone_weight_sum
        breakout_weight = float(cluster.get("breakout_metric_weight_sum", 0.0))
        if breakout_weight > 0.0:
            cluster["avg_breach_atr"] = float(cluster["avg_breach_atr_sum"]) / breakout_weight
        volume_weight = float(cluster.get("volume_metric_weight_sum", 0.0))
        if volume_weight > 0.0:
            cluster["avg_test_volume_ratio"] = float(cluster.get("volume_metric_sum", 0.0)) / volume_weight
        volume_sources = sorted(cluster.get("volume_sources", set()))
        if volume_sources:
            cluster["volume_source"] = volume_sources[0] if len(volume_sources) == 1 else "multiple"
            cluster["volume_sources"] = volume_sources
        cluster["score"] = max(
            0.0,
            float(cluster.get("score_base", 0.0))
            - float(cluster.get("breakout_penalty", 0.0))
            + float(cluster.get("role_reversal_bonus", 0.0))
            + float(cluster.get("mtf_confirmation_bonus", 0.0)),
        )
        level = _format_level(
            cluster,
            current_price=current_price,
            tolerance_fraction=float(merge_tolerance_value),
        )
        level["touches"] = int(cluster.get("touches", level.get("touches", 0)))
        level["episodes"] = int(cluster.get("episodes", level.get("episodes", level.get("touches", 0))))
        level["first_touch"] = cluster.get("first_touch")
        level["last_touch"] = cluster.get("last_touch")
        timeframes = sorted(cluster.get("source_timeframes", set()), key=_timeframe_sort_key)
        level["source_timeframes"] = timeframes
        level["merge_details"] = {
            "cross_timeframe_dedupe_count": int(cluster.get("cross_timeframe_dedupe_count", 0)),
            "deduped_timeframes": sorted(cluster.get("deduped_timeframes", set()), key=_timeframe_sort_key),
        }
        if cluster.get("volume_sources"):
            level["volume_sources"] = list(cluster["volume_sources"])
        level["timeframe_contributions"] = [
            {
                "timeframe": tf,
                "weight": float(cluster["timeframe_weights"].get(tf, _timeframe_weight(tf))),
                "raw_score": float(round(float(cluster["timeframe_raw_scores"].get(tf, 0.0)), 4)),
                "weighted_score": float(round(float(cluster["timeframe_scores"].get(tf, 0.0)), 4)),
                "touches": int(cluster["timeframe_touches"].get(tf, 0)),
                "episodes": int(cluster["timeframe_episodes"].get(tf, cluster["timeframe_touches"].get(tf, 0))),
                "merge_mode": str(cluster["timeframe_merge_modes"].get(tf, "full")),
            }
            for tf in timeframes
        ]
        formatted_levels.append(level)

    support_candidates = [dict(level) for level in formatted_levels if level.get("type") == "support"]
    resistance_candidates = [dict(level) for level in formatted_levels if level.get("type") == "resistance"]
    range_candidates = [dict(level) for level in formatted_levels if level.get("type") == "range"]

    support_candidates.sort(key=lambda level: (-float(level.get("score", 0.0)), -float(level.get("value", 0.0))))
    resistance_candidates.sort(key=lambda level: (-float(level.get("score", 0.0)), float(level.get("value", 0.0))))
    _annotate_strength_metrics(support_candidates)
    _annotate_strength_metrics(resistance_candidates)
    range_candidates.sort(
        key=lambda level: (
            abs(float(level.get("distance_pct", 0.0))),
            -float(level.get("score", 0.0)),
        )
    )
    _annotate_strength_metrics(range_candidates)
    coverage_gaps = _build_coverage_gaps(
        support_levels=support_candidates,
        resistance_levels=resistance_candidates,
        max_distance_pct=max_distance_value,
    )
    support_candidates = _filter_levels_by_distance(support_candidates, max_distance_pct=max_distance_value)
    resistance_candidates = _filter_levels_by_distance(resistance_candidates, max_distance_pct=max_distance_value)
    # Drop breakout-broken (clamped-to-zero) levels when stronger ones remain.
    support_candidates = _drop_zero_score_when_stronger_exist(support_candidates)
    resistance_candidates = _drop_zero_score_when_stronger_exist(resistance_candidates)
    range_candidates = _drop_zero_score_when_stronger_exist(range_candidates)

    max_levels_value = max(1, int(max_levels))
    supports = support_candidates[:max_levels_value]
    resistances = resistance_candidates[:max_levels_value]
    ranges = range_candidates[:max_levels_value]
    supports.sort(key=lambda level: (-float(level.get("value", 0.0)), -float(level.get("score", 0.0))))
    resistances.sort(key=lambda level: (float(level.get("value", 0.0)), -float(level.get("score", 0.0))))
    _annotate_strength_metrics(supports)
    _annotate_strength_metrics(resistances)
    _annotate_strength_metrics(ranges)
    _annotate_proximity_ranks(supports)
    _annotate_proximity_ranks(resistances)
    _annotate_proximity_ranks(ranges)
    zone_overlap = _build_zone_overlap(
        support_level=supports[0] if supports else None,
        resistance_level=resistances[0] if resistances else None,
        current_price=current_price,
    )

    start_values = [
        str((payload.get("window") or {}).get("start") or "").strip()
        for payload in results
        if isinstance(payload.get("window"), dict) and str((payload.get("window") or {}).get("start") or "").strip()
    ]
    end_values = [
        str((payload.get("window") or {}).get("end") or "").strip()
        for payload in results
        if isinstance(payload.get("window"), dict) and str((payload.get("window") or {}).get("end") or "").strip()
    ]
    timeframes_analyzed = [
        str(payload.get("timeframe") or "").upper()
        for payload in sorted(results, key=lambda item: _timeframe_sort_key(item.get("timeframe")))
        if str(payload.get("timeframe") or "").strip()
    ]
    structure_close_epoch = None
    source_bar_open_epoch = None
    for payload in results:
        close_epoch = _payload_close_as_of_epoch(payload)
        if close_epoch is None:
            continue
        if structure_close_epoch is not None and close_epoch < structure_close_epoch:
            continue
        window = payload.get("window") if isinstance(payload.get("window"), dict) else None
        open_epoch = _parse_output_time(payload.get("source_bar_open"))
        if open_epoch is None:
            open_epoch = _parse_output_time((window or {}).get("end"))
        structure_close_epoch = close_epoch
        source_bar_open_epoch = open_epoch
    fibonacci = _select_auto_fibonacci_payload(results, current_price=current_price)
    warnings = _collect_support_resistance_warnings(
        fibonacci=fibonacci,
        coverage_gaps=coverage_gaps,
        zone_overlap=zone_overlap,
    )

    return {
        "success": True,
        "symbol": symbol if symbol is not None else results[0].get("symbol"),
        "timeframe": str(timeframe),
        "mode": "auto",
        "source": "mt5_history",
        "structure_as_of": format_epoch_utc(structure_close_epoch),
        "source_bar_open": _format_time(source_bar_open_epoch),
        "timezone": "UTC",
        "timeframes_analyzed": timeframes_analyzed,
        "timeframe_weights": {tf: _timeframe_weight(tf) for tf in timeframes_analyzed},
        "per_timeframe": [
            {
                "timeframe": str(payload.get("timeframe")),
                "supports": len(payload.get("supports") or []),
                "resistances": len(payload.get("resistances") or []),
                "current_price": payload.get("current_price"),
                "window": payload.get("window"),
                "effective_tolerance_pct": payload.get("effective_tolerance_pct"),
                "effective_reaction_bars": payload.get("effective_reaction_bars"),
                "volatility_ratio": payload.get("volatility_ratio"),
                "current_atr_pct": payload.get("current_atr_pct"),
                "baseline_atr_pct": payload.get("baseline_atr_pct"),
                "volume_source": payload.get("volume_source"),
            }
            for payload in sorted(results, key=lambda item: _timeframe_sort_key(item.get("timeframe")))
        ],
        "limit": int(limit) if limit is not None else int(results[0].get("limit", 0) or 0),
        "method": _METHOD_NAME,
        "tolerance_pct": float(tolerance_fraction) * 100.0,
        "effective_tolerance_pct": float(merge_tolerance_value) * 100.0,
        "min_touches": int(max(1, int(min_touches))),
        "qualification_basis": "episodes",
        "score_basis": {
            "scale": "unbounded_nonnegative",
            "higher_is_stronger": True,
            "definition": (
                "base reaction, recency, and volume strength minus breakout "
                "penalties plus role-reversal and multi-timeframe bonuses"
            ),
            "comparison": (
                "rank within this response; use strength_score_normalized for "
                "a relative 0-to-1 scale"
            ),
            "strength_rank": (
                "1 is strongest among levels of the same type in this response"
            ),
        },
        "units": {
            "distance_pct": (
                "signed_percent (negative=below_reference, "
                "positive=above_reference, 1.0=1%)"
            ),
            "tolerance_pct": "percent (1.0=1%)",
            "effective_tolerance_pct": "percent (1.0=1%)",
            "current_atr_pct": "percent (1.0=1%)",
            "baseline_atr_pct": "percent (1.0=1%)",
        },
        "max_levels": int(max_levels_value),
        "max_distance_pct": None if max_distance_value is None else float(max_distance_value),
        "volume_weighting": volume_weighting_mode,
        "volume_source": next(iter(volume_sources_seen)) if len(volume_sources_seen) == 1 else ("multiple" if volume_sources_seen else None),
        "volume_sources": sorted(volume_sources_seen),
        "reaction_bars": int(max(1, int(reaction_bars))),
        "effective_reaction_bars": int(max(1, int(round(merge_reaction_value)))),
        "adx_period": int(max(2, int(adx_period))),
        "adaptive_mode": "atr_regime",
        "volatility_ratio": float(merge_volatility_ratio),
        "current_atr_pct": (
            None
            if merge_current_atr_fraction is None
            else float(merge_current_atr_fraction) * 100.0
        ),
        "baseline_atr_pct": (
            None
            if merge_baseline_atr_fraction is None
            else float(merge_baseline_atr_fraction) * 100.0
        ),
        "decay_half_life_bars": None if decay_half_life_bars is None else int(decay_half_life_bars),
        "current_price": None if current_price is None else float(round(current_price, 6)),
        "current_price_source": str(
            results[0].get("current_price_source") or "last_completed_bar_close"
        ),
        "reference_price_structure": results[0].get(
            "reference_price_structure"
        ),
        "window": {
            "start": min(start_values) if start_values else None,
            "end": max(end_values) if end_values else None,
        },
        "fibonacci": fibonacci,
        "coverage_gaps": coverage_gaps,
        "zone_overlap": zone_overlap,
        "warnings": warnings,
        "supports": supports,
        "resistances": resistances,
        "ranges": ranges,
        "levels": supports + resistances + ranges,
    }


def compact_support_resistance_level(level: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(level, dict):
        return None
    out: Dict[str, Any] = {}
    for key in (
        "type",
        "value",
        "distance_pct",
        "touches",
        "episodes",
        "score",
        "status",
        "strength_rank",
        "proximity_rank",
        "last_touch",
        "zone_width",
        "zone_width_atr",
        "zone_width_capped",
        "contains_current_price",
        "avg_test_volume_ratio",
        "volume_source",
        "role_transition",
    ):
        value = level.get(key)
        if key in {"zone_width_capped", "contains_current_price"} and not value:
            continue
        if value is not None:
            out[str(key)] = _round_support_resistance_display_metric(str(key), value)
    source_timeframes = level.get("source_timeframes")
    if isinstance(source_timeframes, list) and source_timeframes:
        out["source_timeframes"] = list(source_timeframes)
    dominant_source = level.get("dominant_source")
    if isinstance(dominant_source, str) and dominant_source.strip():
        out["dominant_source"] = dominant_source
    return out or None


def _standard_support_resistance_level(level: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(level, dict):
        return None
    out: Dict[str, Any] = {}
    for key in (
        "type",
        "value",
        "distance",
        "distance_pct",
        "touches",
        "episodes",
        "status",
        "score",
        "strength_rank",
        "proximity_rank",
        "role_transition",
        "strength_percentile",
        "strength_score_normalized",
        "last_touch",
        "zone_low",
        "zone_high",
        "zone_width",
        "zone_width_atr",
        "zone_width_capped",
        "contains_current_price",
        "avg_test_volume_ratio",
        "volume_source",
    ):
        value = level.get(key)
        if value is not None:
            out[str(key)] = _round_support_resistance_display_metric(str(key), value)
    volume_sources = level.get("volume_sources")
    if isinstance(volume_sources, list) and volume_sources:
        out["volume_sources"] = list(volume_sources)
    source_timeframes = level.get("source_timeframes")
    if isinstance(source_timeframes, list) and source_timeframes:
        out["source_timeframes"] = list(source_timeframes)
    dominant_source = level.get("dominant_source")
    if isinstance(dominant_source, str) and dominant_source.strip():
        out["dominant_source"] = dominant_source
    return out or None


def _round_support_resistance_display_metric(key: str, value: Any) -> Any:
    decimals_by_key = {
        "score": 2,
        "distance_pct": 4,
        "zone_width_atr": 2,
        "avg_test_volume_ratio": 3,
        "strength_percentile": 2,
        "strength_score_normalized": 3,
    }
    decimals = decimals_by_key.get(str(key))
    if decimals is None:
        return value
    numeric = _as_finite_float(value)
    if numeric is None:
        return value
    rounded = round(float(numeric), decimals)
    return 0.0 if rounded == -0.0 else float(rounded)


def _compact_support_resistance_levels(
    levels: Any,
    *,
    standard: bool = False,
) -> List[Dict[str, Any]]:
    if not isinstance(levels, list):
        return []
    formatter = _standard_support_resistance_level if standard else compact_support_resistance_level
    return [
        compacted
        for compacted in (formatter(level) for level in levels)
        if compacted
    ]


def _has_historical_role_source(levels: List[Dict[str, Any]]) -> bool:
    return any(
        str(level.get("dominant_source") or "").strip().lower()
        in {"support", "resistance"}
        for level in levels
    )


def compact_support_resistance_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    out: Dict[str, Any] = {}
    for key in (
        "success",
        "symbol",
        "timeframe",
        "mode",
        "current_price",
        "current_price_source",
        "current_price_as_of",
        "current_price_time_basis",
        "reference_price_warning_code",
        "reference_quote_as_of",
        "reference_quote_usable_for_live_trading",
        "reference_quote_freshness_state",
        "reference_quote_freshness_reason",
        "source",
        "structure_as_of",
        "source_bar_open",
        "timezone",
        "timeframes_analyzed",
        "score_basis",
        "units",
        "input_bar_policy",
    ):
        value = payload.get(key)
        if value is not None:
            out[str(key)] = value

    supports = payload.get("supports")
    resistances = payload.get("resistances")
    ranges = payload.get("ranges")
    if any(isinstance(rows, list) for rows in (supports, resistances, ranges)):
        counts: Dict[str, int] = {}
        if isinstance(supports, list):
            counts["support"] = len(supports)
        if isinstance(resistances, list):
            counts["resistance"] = len(resistances)
        if isinstance(ranges, list) and ranges:
            counts["range"] = len(ranges)
        if counts:
            counts["total"] = sum(counts.values())
            out["level_counts"] = counts
            missing_sides: List[str] = [
                side for side in ("support", "resistance") if counts.get(side) == 0
            ]
            if missing_sides:
                out["level_scan_note"] = (
                    "No "
                    + "/".join(missing_sides)
                    + " levels qualified inside the scan filters. "
                    "Try a wider start/end window, lower min_touches, or higher lookback."
                )

    window = payload.get("window")
    if isinstance(window, dict) and any(window.get(key) for key in ("start", "end")):
        out["scan_window"] = {
            key: window.get(key)
            for key in ("start", "end")
            if window.get(key) is not None
        }
    for key in (
        "lookback",
        "limit",
        "max_distance_pct",
        "min_touches",
        "effective_reaction_bars",
    ):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            out[key] = value
    lookback_bars = payload.get("lookback_bars")
    if lookback_bars in (None, "", [], {}):
        lookback_bars = payload.get("lookback")
    if lookback_bars in (None, "", [], {}):
        lookback_bars = payload.get("limit")
    if lookback_bars not in (None, "", [], {}):
        try:
            out["lookback_bars"] = int(lookback_bars)
        except (TypeError, ValueError):
            out["lookback_bars"] = lookback_bars

    support_levels = _compact_support_resistance_levels(supports)
    if isinstance(supports, list):
        out["supports"] = support_levels
    resistance_levels = _compact_support_resistance_levels(resistances)
    if isinstance(resistances, list):
        out["resistances"] = resistance_levels
    range_levels = _compact_support_resistance_levels(ranges)
    if isinstance(ranges, list):
        out["ranges"] = range_levels
    if _has_historical_role_source(support_levels + resistance_levels + range_levels):
        out["role_note"] = (
            "type=current side vs price; dominant_source=historical test role; "
            "role_transition=true marks a support/resistance flip."
        )

    warnings = payload.get("warnings")
    if isinstance(warnings, list) and warnings:
        out["warnings"] = list(warnings)
    return out or dict(payload)


def standard_support_resistance_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    out = compact_support_resistance_payload(payload)

    for key in (
        "max_distance_pct",
        "tolerance_pct",
        "effective_tolerance_pct",
        "current_atr_pct",
        "baseline_atr_pct",
        "volume_weighting",
        "volume_source",
        "window",
    ):
        value = payload.get(key)
        if value not in (None, {}, []):
            out[key] = value

    volume_sources = payload.get("volume_sources")
    if isinstance(volume_sources, list) and volume_sources:
        out["volume_sources"] = list(volume_sources)

    supports = _compact_support_resistance_levels(payload.get("supports"), standard=True)
    if supports:
        out["supports"] = supports

    resistances = _compact_support_resistance_levels(payload.get("resistances"), standard=True)
    if resistances:
        out["resistances"] = resistances
    ranges = _compact_support_resistance_levels(payload.get("ranges"), standard=True)
    if ranges:
        out["ranges"] = ranges

    return out or dict(payload)


def _support_resistance_level_key(level: Any, fallback: str) -> str:
    if not isinstance(level, dict):
        return fallback
    level_type = str(level.get("type") or "level").strip().lower() or "level"
    value = level.get("value")
    try:
        value_text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    except Exception:
        value_text = str(value or fallback)
    return f"{level_type}_{value_text}".replace(".", "_").replace("-", "m")


def _support_resistance_level_diagnostics(levels: Any) -> Dict[str, Any]:
    if not isinstance(levels, list):
        return {}
    diagnostics: Dict[str, Any] = {}
    diagnostic_keys = (
        "source_tests",
        "source_episodes",
        "avg_bounce_atr",
        "avg_pretest_adx",
        "avg_test_volume_ratio",
        "volume_source",
        "breakout_analysis",
        "episode_details",
        "score_breakdown",
    )
    for index, level in enumerate(levels, start=1):
        if not isinstance(level, dict):
            continue
        detail = {
            key: level.get(key)
            for key in diagnostic_keys
            if level.get(key) not in (None, "", [], {})
        }
        if not detail:
            continue
        detail["type"] = level.get("type")
        detail["value"] = level.get("value")
        diagnostics[_support_resistance_level_key(level, f"level_{index}")] = detail
    return diagnostics


def full_support_resistance_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    out = standard_support_resistance_payload(payload)
    for key in ("method", "coverage_gaps", "zone_overlap", "meta"):
        value = payload.get(key)
        if value not in (None, {}, []):
            out[key] = value
    if isinstance(payload.get("fibonacci"), dict):
        out["fibonacci"] = dict(payload["fibonacci"])
    diagnostics: Dict[str, Any] = {}
    for key in ("supports", "resistances", "ranges"):
        section = _support_resistance_level_diagnostics(payload.get(key))
        if section:
            diagnostics[key] = section
    if diagnostics:
        out["diagnostics"] = diagnostics
    return out or dict(payload)


def compute_support_resistance_levels(
    frame: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    limit: Optional[int] = None,
    tolerance_fraction: float = 0.0015,
    min_touches: int = 2,
    max_levels: int = 4,
    reaction_bars: int = _DEFAULT_REACTION_BARS,
    adx_period: int = _DEFAULT_ADX_PERIOD,
    decay_half_life_bars: Optional[int] = None,
    max_distance_pct: Optional[float] = None,
    volume_weighting: str = "off",
    reference_price: Optional[float] = None,
    reference_price_source: Optional[str] = None,
) -> Dict[str, Any]:
    if frame is None or getattr(frame, "empty", True):
        raise ValueError("No history available")
    required_cols = ("high", "low", "close")
    missing = [column for column in required_cols if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if len(frame) < 3:
        raise ValueError("Need at least 3 bars to compute support/resistance levels")

    tolerance_value = float(tolerance_fraction)
    if tolerance_value < 0.0:
        raise ValueError("tolerance_fraction must be non-negative")
    volume_weighting_mode = _normalize_volume_weighting(volume_weighting)
    min_touches_value = max(1, int(min_touches))
    max_levels_value = max(1, int(max_levels))
    max_distance_value = _normalize_max_distance_pct(max_distance_pct)
    reaction_bars_value = max(1, int(reaction_bars))
    adx_period_value = max(2, int(adx_period))
    half_life_value = (
        max(5, int(decay_half_life_bars))
        if decay_half_life_bars is not None
        else max(20, int(round(len(frame) / 4.0)))
    )

    highs = _to_numeric_array(frame, "high")
    lows = _to_numeric_array(frame, "low")
    closes = _to_numeric_array(frame, "close")
    finite_lows = lows[np.isfinite(lows)]
    finite_highs = highs[np.isfinite(highs)]
    window_low = float(np.min(finite_lows)) if finite_lows.size else None
    window_high = float(np.max(finite_highs)) if finite_highs.size else None
    volume_values, volume_source, volume_baseline = _resolve_volume_series(
        frame,
        volume_weighting=volume_weighting_mode,
    )
    structure_price = _last_finite(closes)
    current_price = _as_finite_float(reference_price)
    if current_price is None or current_price <= 0:
        current_price = structure_price
        resolved_price_source = "last_completed_bar_close"
    else:
        resolved_price_source = str(reference_price_source or "external_reference")
    atr, adx = _compute_atr_and_adx(highs, lows, closes, period=adx_period_value)
    adaptive_settings = _resolve_adaptive_settings(
        closes,
        atr,
        base_tolerance_fraction=tolerance_value,
        base_reaction_bars=reaction_bars_value,
    )
    effective_tolerance_value = float(adaptive_settings["effective_tolerance_fraction"])
    effective_reaction_bars = int(adaptive_settings["effective_reaction_bars"])

    epochs = [_to_epoch(value) for value in frame["time"].tolist()] if "time" in frame.columns else []
    window_start = next((value for value in epochs if value is not None), None)
    window_end = next((value for value in reversed(epochs) if value is not None), None)
    structure_close_epoch = _bar_close_as_of_epoch(window_end, timeframe)

    tests = _collect_tests(
        highs,
        lows,
        closes,
        epochs=epochs,
        reaction_bars=effective_reaction_bars,
        adx_period=adx_period_value,
        decay_half_life_bars=half_life_value,
        atr=atr,
        adx=adx,
        volume=volume_values,
        volume_baseline=volume_baseline,
    )
    clusters = _cluster_tests(tests, tolerance_fraction=effective_tolerance_value)
    for cluster in clusters:
        _apply_episode_metrics(cluster, episode_gap_bars=max(1, effective_reaction_bars))
        _analyze_cluster_state(
            cluster,
            closes=closes,
            epochs=epochs,
            tolerance_fraction=effective_tolerance_value,
            window_low=window_low,
            window_high=window_high,
        )
    usable_clusters = [cluster for cluster in clusters if int(cluster.get("episodes", cluster["touches"])) >= min_touches_value]

    formatted_levels = [
        _format_level(
            {**cluster, "volume_source": volume_source},
            current_price=current_price,
            tolerance_fraction=effective_tolerance_value,
            window_low=window_low,
            window_high=window_high,
        )
        for cluster in usable_clusters
    ]
    support_candidates = [dict(level) for level in formatted_levels if level.get("type") == "support"]
    resistance_candidates = [dict(level) for level in formatted_levels if level.get("type") == "resistance"]
    range_candidates = [dict(level) for level in formatted_levels if level.get("type") == "range"]

    support_candidates.sort(key=lambda level: (-float(level.get("score", 0.0)), -float(level.get("value", 0.0))))
    resistance_candidates.sort(key=lambda level: (-float(level.get("score", 0.0)), float(level.get("value", 0.0))))
    _annotate_strength_metrics(support_candidates)
    _annotate_strength_metrics(resistance_candidates)
    range_candidates.sort(
        key=lambda level: (
            abs(float(level.get("distance_pct", 0.0))),
            -float(level.get("score", 0.0)),
        )
    )
    _annotate_strength_metrics(range_candidates)
    coverage_gaps = _build_coverage_gaps(
        support_levels=support_candidates,
        resistance_levels=resistance_candidates,
        max_distance_pct=max_distance_value,
    )
    support_candidates = _filter_levels_by_distance(support_candidates, max_distance_pct=max_distance_value)
    resistance_candidates = _filter_levels_by_distance(resistance_candidates, max_distance_pct=max_distance_value)
    support_candidates = _drop_zero_score_when_stronger_exist(support_candidates)
    resistance_candidates = _drop_zero_score_when_stronger_exist(resistance_candidates)
    range_candidates = _drop_zero_score_when_stronger_exist(range_candidates)

    supports = support_candidates[:max_levels_value]
    resistances = resistance_candidates[:max_levels_value]
    ranges = range_candidates[:max_levels_value]
    supports.sort(key=lambda level: (-float(level.get("value", 0.0)), -float(level.get("score", 0.0))))
    resistances.sort(key=lambda level: (float(level.get("value", 0.0)), -float(level.get("score", 0.0))))
    _annotate_strength_metrics(supports)
    _annotate_strength_metrics(resistances)
    _annotate_strength_metrics(ranges)
    _annotate_proximity_ranks(supports)
    _annotate_proximity_ranks(resistances)
    _annotate_proximity_ranks(ranges)
    zone_overlap = _build_zone_overlap(
        support_level=supports[0] if supports else None,
        resistance_level=resistances[0] if resistances else None,
        current_price=current_price,
    )
    fibonacci = _compute_fibonacci_payload(
        highs=highs,
        lows=lows,
        atr=atr,
        epochs=epochs,
        current_price=current_price,
        timeframe=timeframe,
    )
    warnings = _collect_support_resistance_warnings(
        fibonacci=fibonacci,
        coverage_gaps=coverage_gaps,
        zone_overlap=zone_overlap,
        support_count=len(supports),
        resistance_count=len(resistances),
    )

    return {
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "mode": "single",
        "source": "mt5_history",
        "structure_as_of": format_epoch_utc(structure_close_epoch),
        "source_bar_open": _format_time(window_end),
        "timezone": "UTC",
        "timeframes_analyzed": [str(timeframe)] if timeframe is not None else [],
        "limit": int(limit) if limit is not None else len(frame),
        "method": _METHOD_NAME,
        "tolerance_pct": float(tolerance_value) * 100.0,
        "effective_tolerance_pct": float(effective_tolerance_value) * 100.0,
        "min_touches": int(min_touches_value),
        "qualification_basis": "episodes",
        "score_basis": {
            "scale": "unbounded_nonnegative",
            "higher_is_stronger": True,
            "definition": (
                "base reaction, recency, and volume strength minus breakout "
                "penalties plus role-reversal and multi-timeframe bonuses"
            ),
            "comparison": (
                "rank within this response; use strength_score_normalized for "
                "a relative 0-to-1 scale"
            ),
            "strength_rank": (
                "1 is strongest among levels of the same type in this response"
            ),
        },
        "units": {
            "distance_pct": (
                "signed_percent (negative=below_reference, "
                "positive=above_reference, 1.0=1%)"
            ),
            "tolerance_pct": "percent (1.0=1%)",
            "effective_tolerance_pct": "percent (1.0=1%)",
            "current_atr_pct": "percent (1.0=1%)",
            "baseline_atr_pct": "percent (1.0=1%)",
        },
        "max_levels": int(max_levels_value),
        "max_distance_pct": None if max_distance_value is None else float(max_distance_value),
        "volume_weighting": volume_weighting_mode,
        "volume_source": volume_source,
        "reaction_bars": int(reaction_bars_value),
        "effective_reaction_bars": int(effective_reaction_bars),
        "adx_period": int(adx_period_value),
        "adaptive_mode": str(adaptive_settings["adaptive_mode"]),
        "volatility_ratio": float(adaptive_settings["volatility_ratio"]),
        "current_atr_pct": (
            None
            if adaptive_settings["current_atr_fraction"] is None
            else float(adaptive_settings["current_atr_fraction"]) * 100.0
        ),
        "baseline_atr_pct": (
            None
            if adaptive_settings["baseline_atr_fraction"] is None
            else float(adaptive_settings["baseline_atr_fraction"]) * 100.0
        ),
        "decay_half_life_bars": int(half_life_value),
        "current_price": None if current_price is None else float(round(current_price, 6)),
        "current_price_source": resolved_price_source,
        "reference_price_structure": (
            None if structure_price is None else float(round(structure_price, 6))
        ),
        "window": {
            "start": _format_time(window_start),
            "end": _format_time(window_end),
        },
        "fibonacci": fibonacci,
        "coverage_gaps": coverage_gaps,
        "zone_overlap": zone_overlap,
        "warnings": warnings,
        "supports": supports,
        "resistances": resistances,
        "ranges": ranges,
        "levels": supports + resistances + ranges,
    }
