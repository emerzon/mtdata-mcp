from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

from ..utils.utils import to_float_np
from .common import PatternResultBase, _coerce_pattern_time_epoch
from .common import interval_overlap_ratio as _interval_overlap_ratio
from .elliott_adaptation import resolve_elliott_adaptation


@dataclass
class ElliottWaveConfig:
    """Configuration for Elliott Wave detection.

    Scope notes:
    - Impulses are *strict motive* 5-wave structures (no diagonals).
    - Corrections are outer ABC candidates; internal 5-3-5 subdivisions are not validated.
    - Detection is single ZigZag degree; multi-degree nesting is not modeled.

    Structural scoring is based on hard-rule validity and Fibonacci-template
    fit. The optional GMM classifier is diagnostic-only. When scale_mode is
    "auto", request-scoped trailing-window calibration selects market-relative
    ZigZag scales and may conservatively smooth the close-based pivot signal.
    Its filters are causal, but its terminal selection is not a walk-forward
    historical backtest.
    """

    # Pivot detection controls (used by analysis and threshold scans)
    min_prominence_pct: float = 0.5
    min_distance: int = 5
    scale_mode: str = "auto"  # "auto" derives request-scoped market scales
    adaptive_denoise: str = "auto"  # "auto", "off", or "diagnostic"
    adaptive_window_bars: int = 240
    adaptive_min_improvement: float = 0.05

    # New options
    schema_version: int = 2
    pivot_price_source: str = "close"  # "close" or causal OHLC-extrema geometry
    swing_threshold_pct: Optional[float] = None  # ZigZag swing threshold in percent
    scan_thresholds_pct: Optional[List[float]] = None
    scan_min_distances: Optional[List[int]] = None
    gmm_components: int = 2  # Impulsive vs corrective clusters
    min_gmm_waves: int = 12
    enable_gmm_classifier: bool = False
    wave_min_len: int = 3
    use_volume_confirmation: bool = True
    volume_confirm_min_ratio: float = 1.05
    volume_confirm_bonus: float = 0.06
    volume_confirm_penalty: float = 0.04

    # Empirical impulse-wave calibration weights. These are not Fibonacci
    # ratios themselves; they weight the relative importance of each alignment
    # component when scoring a 5-wave impulse candidate:
    #   [0] wave-2 retracement quality
    #   [1] wave-3 extension quality
    #   [2] wave-4 retracement quality
    #   [3] wave-5 completion / proportionality
    impulse_fib_weights: List[float] = field(
        default_factory=lambda: [0.30, 0.35, 0.20, 0.15]
    )

    # Hybrid confidence blending. Each rule/classifier pair is renormalized
    # before use so callers can express relative preference without manually
    # forcing the weights to sum to 1.0.
    impulse_rule_weight: float = 0.65
    impulse_cls_weight: float = 0.35
    correction_rule_weight: float = 0.60
    correction_cls_weight: float = 0.40
    use_regime_context: bool = True
    regime_window_bars: int = 160
    regime_trend_strength_threshold: float = 1.25
    regime_efficiency_trending_threshold: float = 0.35
    regime_alignment_bonus: float = 0.05
    regime_countertrend_penalty: float = 0.05
    min_impulse_bars: int = 30
    min_correction_bars: int = 20

    # Multi-scale scan controls
    scan_skip_repeated_pivots: bool = True
    scan_early_stop_repeats: int = 3
    scan_scenario_overlap_ratio: float = 0.9
    scan_timeframes: List[str] = field(default_factory=lambda: ["H1", "H4", "D1"])
    max_scan_timeframes: int = 3

    # Analyzer controls
    min_confidence: float = 0.0
    min_structural_score: float = 0.25
    unconfirmed_pattern_penalty: float = 0.12
    unconfirmed_terminal_pivot_penalty: float = 0.10
    unconfirmed_terminal_pivot_confidence_cap: float = 0.72
    top_k: int = 10
    include_fallback_candidate: bool = True
    recent_bars: Optional[int] = None
    correction_min_c_vs_a: float = 0.25
    correction_exclusion_bar_tolerance: int = 1
    correction_exclusion_overlap_ratio: float = 0.9
    max_pattern_span_bars: Optional[int] = None
    max_pattern_age_bars: Optional[int] = None
    pattern_types: List[str] = field(default_factory=lambda: ["impulse", "correction"])

    def validate(self) -> None:
        _validate_config(self)


@dataclass
class ElliottWaveResult(PatternResultBase):
    """Result of an Elliott Wave pattern detection."""

    wave_type: str  # "Impulse", "Correction", or "Candidate"
    wave_sequence: List[int]
    details: Dict[str, Any] = field(default_factory=dict)
    available_at_index: Optional[int] = None
    available_at_time: Optional[float] = None
    structure_state: str = "developing"


@dataclass(frozen=True)
class ElliottPivot:
    index: int
    kind: str
    confirmed: bool
    confirmation_index: Optional[int] = None
    price: Optional[float] = None
    price_source: str = "close"


@dataclass
class ElliottRuleEvaluation:
    valid: bool
    fib_score: float
    metrics: Dict[str, float]
    violations: List[str] = field(default_factory=list)


@dataclass
class ElliottScenario:
    pivots: List[int]
    bullish: bool
    confidence: float
    cls_score: float
    rule_eval: ElliottRuleEvaluation
    threshold_used: float
    min_distance_used: int
    fallback_candidate: bool = False
    synthetic_terminal_pivot: bool = False
    wave_type: str = "Impulse"
    validated_wave_type: Optional[str] = None
    classification_available: bool = False
    base_confidence: Optional[float] = None
    pivot_confirmations: Optional[List[bool]] = None
    confidence_adjustments: Dict[str, float] = field(default_factory=dict)
    pivot_records: Optional[List[ElliottPivot]] = None


def _normalize_pattern_types(config: ElliottWaveConfig, *, strict: bool = False) -> set[str]:
    raw = getattr(config, "pattern_types", None)
    if not raw:
        if strict:
            raise ValueError("pattern_types must contain impulse and/or correction")
        return {"impulse", "correction"}
    out: set[str] = set()
    for item in raw:
        s = str(item).strip().lower()
        if s in ("impulse", "impulses"):
            out.add("impulse")
        elif s in ("correction", "corrective", "abc"):
            out.add("correction")
    if strict and not out:
        raise ValueError("pattern_types must contain impulse and/or correction")
    return out or {"impulse", "correction"}


def _validate_config(config: ElliottWaveConfig) -> None:
    if int(config.schema_version) != 2:
        raise ValueError("schema_version must be 2")
    source = str(config.pivot_price_source or "").strip().lower()
    if source not in {"close", "ohlc"}:
        raise ValueError("pivot_price_source must be 'close' or 'ohlc'")
    if str(config.scale_mode or "").strip().lower() not in {"auto", "fixed"}:
        raise ValueError("scale_mode must be 'auto' or 'fixed'")
    if str(config.adaptive_denoise or "").strip().lower() not in {
        "auto",
        "off",
        "diagnostic",
    }:
        raise ValueError("adaptive_denoise must be 'auto', 'off', or 'diagnostic'")
    _normalize_pattern_types(config, strict=True)

    for name in (
        "min_distance",
        "wave_min_len",
        "min_gmm_waves",
        "min_impulse_bars",
        "min_correction_bars",
        "adaptive_window_bars",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or int(value) < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    if config.recent_bars is not None and (
        isinstance(config.recent_bars, bool) or int(config.recent_bars) < 1
    ):
        raise ValueError("recent_bars must be None or an integer >= 1")
    if isinstance(config.top_k, bool) or int(config.top_k) < 0:
        raise ValueError("top_k must be an integer >= 0")
    if int(config.gmm_components) != 2:
        raise ValueError("gmm_components must be 2")

    thresholds: List[Any] = [config.min_prominence_pct]
    if config.swing_threshold_pct is not None:
        thresholds.append(config.swing_threshold_pct)
    for values in (config.scan_thresholds_pct,):
        if values is not None:
            if not isinstance(values, (list, tuple)):
                raise ValueError("Elliott threshold scans must be lists")
            thresholds.extend(values)
    if not all(np.isfinite(float(v)) and float(v) > 0.0 for v in thresholds):
        raise ValueError("all Elliott swing thresholds must be finite and > 0")
    for values in (config.scan_min_distances,):
        if values is not None:
            if not isinstance(values, (list, tuple)):
                raise ValueError("Elliott distance scans must be lists")
            if any(isinstance(v, bool) or int(v) < 1 for v in values):
                raise ValueError("all Elliott scan distances must be integers >= 1")
    if not np.isfinite(float(config.min_confidence)) or not 0.0 <= float(config.min_confidence) <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    if not np.isfinite(float(config.min_structural_score)) or not 0.0 <= float(config.min_structural_score) <= 1.0:
        raise ValueError("min_structural_score must be between 0 and 1")
    if not np.isfinite(float(config.adaptive_min_improvement)) or not 0.0 <= float(config.adaptive_min_improvement) <= 1.0:
        raise ValueError("adaptive_min_improvement must be between 0 and 1")
    for name in ("max_pattern_span_bars", "max_pattern_age_bars"):
        value = getattr(config, name)
        if value is not None and (isinstance(value, bool) or int(value) < 1):
            raise ValueError(f"{name} must be None or an integer >= 1")
    if isinstance(config.max_scan_timeframes, bool) or int(config.max_scan_timeframes) < 1:
        raise ValueError("max_scan_timeframes must be an integer >= 1")


def _zigzag_pivots_indices(
    close: np.ndarray, threshold_pct: float
) -> Tuple[List[int], List[str]]:
    """Lightweight ZigZag pivot detection returning indices and directions."""

    n = int(close.size)
    if n <= 1:
        return [], []
    finite_idx = np.flatnonzero(np.isfinite(close))
    if finite_idx.size <= 1:
        return [], []
    start_i = int(finite_idx[0])
    start_p = float(close[start_i])

    piv_idx: List[int] = []
    piv_dir: List[str] = []  # "up" for a peak and "down" for a trough

    pivot_i = start_i
    trend: Optional[str] = None
    last_ext_i = start_i
    last_ext_p = start_p
    pre_high_i = start_i
    pre_high_p = start_p
    pre_low_i = start_i
    pre_low_p = start_p

    def _pivot_direction_for_trend(next_trend: Optional[str]) -> str:
        if next_trend == "up":
            return "down"
        if next_trend == "down":
            return "up"
        return "flat"

    for i in range(start_i + 1, n):
        p = float(close[i])
        if not np.isfinite(p):
            continue

        if trend is None:
            if p >= pre_high_p:
                pre_high_p = p
                pre_high_i = i
            if p <= pre_low_p:
                pre_low_p = p
                pre_low_i = i

            up_change = (
                (p - pre_low_p) / abs(pre_low_p) * 100.0 if pre_low_p != 0 else 0.0
            )
            down_change = (
                (pre_high_p - p) / abs(pre_high_p) * 100.0 if pre_high_p != 0 else 0.0
            )
            can_start_up = up_change >= threshold_pct and pre_low_i < i
            can_start_down = down_change >= threshold_pct and pre_high_i < i

            if can_start_up and can_start_down:
                if pre_low_i > pre_high_i:
                    trend = "up"
                    pivot_i = pre_low_i
                elif pre_high_i > pre_low_i:
                    trend = "down"
                    pivot_i = pre_high_i
                elif up_change >= down_change:
                    trend = "up"
                    pivot_i = pre_low_i
                else:
                    trend = "down"
                    pivot_i = pre_high_i
            elif can_start_up:
                trend = "up"
                pivot_i = pre_low_i
            elif can_start_down:
                trend = "down"
                pivot_i = pre_high_i
            else:
                continue
            last_ext_i = i
            last_ext_p = p
            piv_idx.append(pivot_i)
            piv_dir.append(_pivot_direction_for_trend(trend))

        if trend == "up":
            if p > last_ext_p:
                last_ext_p = p
                last_ext_i = i
            retr = (
                (last_ext_p - p) / abs(last_ext_p) * 100.0 if last_ext_p != 0 else 0.0
            )
            if retr >= threshold_pct:
                piv_idx.append(last_ext_i)
                piv_dir.append("up")
                trend = "down"
                pivot_i = i
                last_ext_i = i
                last_ext_p = p
        else:
            if p < last_ext_p:
                last_ext_p = p
                last_ext_i = i
            retr = (
                (p - last_ext_p) / abs(last_ext_p) * 100.0 if last_ext_p != 0 else 0.0
            )
            if retr >= threshold_pct:
                piv_idx.append(last_ext_i)
                piv_dir.append("down")
                trend = "up"
                pivot_i = i
                last_ext_i = i
                last_ext_p = p

    if len(piv_idx) == 0 or piv_idx[-1] != last_ext_i:
        piv_idx.append(last_ext_i)
        piv_dir.append("up" if trend == "up" else "down" if trend == "down" else "flat")

    return piv_idx, piv_dir


def _ohlc_zigzag_pivots_indices(
    high: np.ndarray, low: np.ndarray, threshold_pct: float
) -> Tuple[List[int], List[str]]:
    """Causal ZigZag that tracks peaks on highs and troughs on lows."""
    if high.size != low.size or high.size <= 1:
        return [], []
    finite = np.flatnonzero(np.isfinite(high) & np.isfinite(low) & (high >= low))
    if finite.size <= 1:
        return [], []
    start = int(finite[0])
    pre_high_i = pre_low_i = start
    pre_high = float(high[start])
    pre_low = float(low[start])
    trend: Optional[str] = None
    ext_i = start
    ext_price = pre_low
    pivots: List[int] = []
    directions: List[str] = []

    for i in range(start + 1, int(high.size)):
        hi, lo = float(high[i]), float(low[i])
        if not (np.isfinite(hi) and np.isfinite(lo) and hi >= lo):
            continue
        if trend is None:
            if hi >= pre_high:
                pre_high, pre_high_i = hi, i
            if lo <= pre_low:
                pre_low, pre_low_i = lo, i
            up = (hi - pre_low) / abs(pre_low) * 100.0 if pre_low else 0.0
            down = (pre_high - lo) / abs(pre_high) * 100.0 if pre_high else 0.0
            if up < threshold_pct and down < threshold_pct:
                continue
            if up >= down:
                trend, ext_i, ext_price = "up", i, hi
                pivots.append(pre_low_i)
                directions.append("down")
            else:
                trend, ext_i, ext_price = "down", i, lo
                pivots.append(pre_high_i)
                directions.append("up")
            continue
        if trend == "up":
            if hi > ext_price:
                ext_i, ext_price = i, hi
            # A wide bar that just set the high must not confirm against its own low.
            if i == ext_i:
                continue
            reversal = (ext_price - lo) / abs(ext_price) * 100.0 if ext_price else 0.0
            if reversal >= threshold_pct:
                pivots.append(ext_i)
                directions.append("up")
                trend, ext_i, ext_price = "down", i, lo
        else:
            if lo < ext_price:
                ext_i, ext_price = i, lo
            if i == ext_i:
                continue
            reversal = (hi - ext_price) / abs(ext_price) * 100.0 if ext_price else 0.0
            if reversal >= threshold_pct:
                pivots.append(ext_i)
                directions.append("down")
                trend, ext_i, ext_price = "up", i, hi
    if not pivots or pivots[-1] != ext_i:
        pivots.append(ext_i)
        directions.append("up" if trend == "up" else "down")
    return pivots, directions


def _enforce_min_distance_on_pivots(
    pivots: List[int], close: np.ndarray, min_distance: int
) -> List[int]:
    """Ensure pivots are monotonic and separated by at least min_distance bars."""

    if not pivots:
        return []

    md = int(max(1, min_distance))
    sorted_unique: List[int] = []
    for idx in pivots:
        i = int(idx)
        if i < 0 or i >= int(close.size):
            continue
        if not sorted_unique or i > sorted_unique[-1]:
            sorted_unique.append(i)
        elif i == sorted_unique[-1]:
            continue

    if not sorted_unique:
        return []
    if len(sorted_unique) <= 1:
        return sorted_unique

    out = [sorted_unique[0]]
    for idx in sorted_unique[1:]:
        prev = out[-1]
        if idx - prev >= md:
            out.append(idx)
            continue
        # Keep the earlier origin. Replacing it shifts a 1-5 count off wave 0.
        if len(out) == 1:
            continue
        anchor = out[-2]
        old_move = abs(float(close[prev]) - float(close[anchor]))
        new_move = abs(float(close[idx]) - float(close[anchor]))
        if new_move >= old_move:
            out[-1] = idx
    return out


def _enforce_pivot_alternation(pivots: List[int], close: np.ndarray) -> List[int]:
    """Keep a monotonic alternating high/low pivot sequence after distance filtering.

    Min-distance filtering can drop intermediate pivots and leave consecutive
    same-direction legs. Merge those legs by retaining the more extreme end.
    """
    if not pivots:
        return []
    n = int(close.size)
    cleaned: List[int] = []
    for raw in pivots:
        idx = int(raw)
        if idx < 0 or idx >= n or not np.isfinite(float(close[idx])):
            continue
        if cleaned and idx == cleaned[-1]:
            continue
        if cleaned and idx < cleaned[-1]:
            continue
        cleaned.append(idx)
    if len(cleaned) <= 2:
        return cleaned

    out: List[int] = [cleaned[0]]
    for idx in cleaned[1:]:
        if len(out) == 1:
            if float(close[idx]) != float(close[out[0]]):
                out.append(idx)
            continue
        prev = out[-1]
        anchor = out[-2]
        prev_move = float(close[prev]) - float(close[anchor])
        new_move = float(close[idx]) - float(close[prev])
        if new_move == 0.0:
            continue
        if prev_move == 0.0:
            out[-1] = idx
            continue
        # Same sign => non-alternating; keep the farther extreme from the anchor.
        if (prev_move > 0 and new_move > 0) or (prev_move < 0 and new_move < 0):
            if abs(float(close[idx]) - float(close[anchor])) >= abs(
                float(close[prev]) - float(close[anchor])
            ):
                out[-1] = idx
            continue
        out.append(idx)
    return out


def _build_pivot_records(
    pivots: List[int],
    close: np.ndarray,
    threshold_pct: float,
    *,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    kinds: Optional[Dict[int, str]] = None,
    prices: Optional[Dict[int, float]] = None,
) -> List[ElliottPivot]:
    """Attach the first bar that made each geometric pivot causally knowable."""
    records: List[ElliottPivot] = []
    threshold = float(threshold_pct)
    use_ohlc = (
        isinstance(high, np.ndarray)
        and isinstance(low, np.ndarray)
        and high.size == close.size
        and low.size == close.size
    )
    for pos, idx in enumerate(pivots):
        mapped_kind = kinds.get(int(idx)) if isinstance(kinds, dict) else None
        if mapped_kind in {"peak", "trough"}:
            kind = str(mapped_kind)
        elif len(pivots) == 1:
            kind = "unknown"
        elif pos == 0:
            kind = "trough" if close[pivots[1]] > close[idx] else "peak"
        elif pos == len(pivots) - 1:
            kind = "peak" if close[idx] > close[pivots[pos - 1]] else "trough"
        else:
            kind = "peak" if close[idx] > max(close[pivots[pos - 1]], close[pivots[pos + 1]]) else "trough"

        confirmation_index: Optional[int] = None
        pivot_price = float(
            prices[int(idx)]
            if isinstance(prices, dict) and int(idx) in prices
            else high[idx]
            if use_ohlc and kind == "peak"
            else low[idx]
            if use_ohlc
            else close[idx]
        )
        for bar in range(int(idx) + 1, int(close.size)):
            price = float(
                low[bar]
                if use_ohlc and kind == "peak"
                else high[bar]
                if use_ohlc
                else close[bar]
            )
            if not np.isfinite(price) or pivot_price == 0.0:
                continue
            reversal = (
                (pivot_price - price) / abs(pivot_price) * 100.0
                if kind == "peak"
                else (price - pivot_price) / abs(pivot_price) * 100.0
            )
            if reversal >= threshold:
                confirmation_index = int(bar)
                break
        records.append(
            ElliottPivot(
                index=int(idx),
                kind=kind,
                confirmed=confirmation_index is not None,
                confirmation_index=confirmation_index,
                price=float(
                    prices[int(idx)]
                    if isinstance(prices, dict) and int(idx) in prices
                    else high[idx]
                    if use_ohlc and kind == "peak"
                    else low[idx]
                    if use_ohlc and kind == "trough"
                    else close[idx]
                ),
                price_source="ohlc" if use_ohlc else "close",
            )
        )
    return records


def _enforce_kind_alternation(
    pivots: List[int], kinds: Dict[int, str], prices: Dict[int, float]
) -> List[int]:
    """Retain alternating peak/trough provenance after pivot filtering."""
    out: List[int] = []
    for raw in pivots:
        idx = int(raw)
        kind = kinds.get(idx)
        if kind not in {"peak", "trough"}:
            continue
        if not out or kinds.get(out[-1]) != kind:
            out.append(idx)
            continue
        previous = out[-1]
        better = (
            float(prices[idx]) >= float(prices[previous])
            if kind == "peak"
            else float(prices[idx]) <= float(prices[previous])
        )
        if better:
            out[-1] = idx
    return out


def _segment_waves_from_pivots(pivots: List[int]) -> List[Tuple[int, int]]:
    waves: List[Tuple[int, int]] = []
    if len(pivots) < 2:
        return waves
    for i in range(len(pivots) - 1):
        s, e = pivots[i], pivots[i + 1]
        if e > s:
            waves.append((s, e))
    return waves


def _extract_wave_features_with_index(
    waves: List[Tuple[int, int]],
    close: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Extract per-wave features and preserve the source wave index mapping."""
    features: List[List[float]] = []
    wave_index_map: Dict[int, int] = {}
    for wave_idx, (start, end) in enumerate(waves):
        if end <= start:
            continue
        s = float(close[start])
        e = float(close[end])
        if not np.isfinite(s) or not np.isfinite(e):
            continue

        duration = float(end - start)
        price_change = e - s
        slope = price_change / duration if duration > 0 else 0.0
        normalized_price_change = price_change / s if s != 0 else 0.0
        wave_index_map[wave_idx] = len(features)
        features.append([duration, normalized_price_change, slope])

    if not features:
        return np.empty((0, 3), dtype=float), {}
    return np.asarray(features, dtype=float), wave_index_map


def _cluster_impulsive_score(cluster_mean: np.ndarray) -> float:
    price_change = abs(float(cluster_mean[1])) if cluster_mean.size > 1 else 0.0
    slope = abs(float(cluster_mean[2])) if cluster_mean.size > 2 else 0.0
    # A negative standardized duration means shorter-than-average, which must
    # not become "impulsive" merely because it is far from the mean.
    duration = max(0.0, float(cluster_mean[0])) if cluster_mean.size > 0 else 0.0
    return float(price_change + 0.75 * slope + 0.1 * duration)


def _select_impulsive_cluster(cluster_means: np.ndarray) -> Optional[int]:
    if cluster_means.ndim != 2 or cluster_means.shape[0] < 1:
        return None
    scores = np.asarray(
        [
            _cluster_impulsive_score(cluster_means[i])
            for i in range(cluster_means.shape[0])
        ],
        dtype=float,
    )
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        return None
    return int(np.argmax(scores))


def _cluster_direction_score(cluster_mean: np.ndarray) -> float:
    price_change = float(cluster_mean[1]) if cluster_mean.size > 1 else 0.0
    slope = float(cluster_mean[2]) if cluster_mean.size > 2 else 0.0
    return float(price_change + 0.75 * slope)


def _select_directional_cluster(
    cluster_means: np.ndarray, bullish: bool
) -> Optional[int]:
    if cluster_means.ndim != 2 or cluster_means.shape[0] < 1:
        return None
    scores = np.asarray(
        [
            _cluster_direction_score(cluster_means[i])
            for i in range(cluster_means.shape[0])
        ],
        dtype=float,
    )
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        return None
    return int(np.argmax(scores) if bullish else np.argmin(scores))


def _normalized_impulse_fib_weights(
    config: Optional[ElliottWaveConfig],
) -> Tuple[float, float, float, float]:
    defaults = (0.30, 0.35, 0.20, 0.15)
    if config is None:
        return defaults
    raw = getattr(config, "impulse_fib_weights", None)
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return defaults
    try:
        weights = [float(v) for v in raw]
    except Exception:
        return defaults
    if not all(np.isfinite(v) and v >= 0.0 for v in weights):
        return defaults
    total = float(sum(weights))
    if total <= 1e-12:
        return defaults
    return tuple(float(v / total) for v in weights)


def _classify_waves(
    features: np.ndarray,
    config: ElliottWaveConfig,
) -> Tuple[
    np.ndarray,
    Optional[GaussianMixture],
    Optional[StandardScaler],
    Optional[np.ndarray],
    Optional[int],
]:
    """Classify waves with GMM; return cluster labels, model, scaler, probabilities and impulsive cluster id."""

    feature_count = (
        int(features.shape[1]) if features.ndim == 2 and features.size else 0
    )
    statistical_floor = max(
        int(config.gmm_components),
        int(max(1, feature_count)) * 4,
        int(config.gmm_components) * int(max(1, feature_count)) * 2,
    )
    min_waves = max(
        int(config.gmm_components),
        int(max(1, getattr(config, "min_gmm_waves", 12))),
        statistical_floor,
    )
    if features.shape[0] < min_waves:
        return np.array([]), None, None, None, None

    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        feature_std = np.nanstd(features, axis=0)
        if (
            not np.all(np.isfinite(feature_std))
            or float(np.nanmax(np.abs(feature_std))) <= 1e-8
        ):
            return np.array([]), None, None, None, None

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        if not np.all(np.isfinite(scaled)):
            return np.array([]), None, None, None, None

        gmm = GaussianMixture(
            n_components=config.gmm_components,
            random_state=42,
            # Skip default KMeans initialization — it blocks indefinitely in
            # asyncio.to_thread worker threads on Windows (joblib CPU probe).
            init_params="random_from_data",
        )
        gmm.fit(scaled)
        labels = gmm.predict(scaled)
        if np.unique(labels).size < int(config.gmm_components):
            return np.array([]), None, None, None, None
        probs = gmm.predict_proba(scaled)
        impulsive_cluster = _select_impulsive_cluster(gmm.means_)
        if impulsive_cluster is None:
            return np.array([]), None, None, None, None
        return labels.astype(int), gmm, scaler, probs, impulsive_cluster
    except Exception:
        return np.array([]), None, None, None, None


def _window_hit(v: float, lo: float, hi: float, taper: float = 0.2) -> float:
    rng = hi - lo
    if rng <= 0:
        return 0.0
    if lo <= v <= hi:
        return 1.0
    if v < lo:
        d = (lo - v) / (taper * rng)
        return float(max(0.0, 1.0 - d))
    d = (v - hi) / (taper * rng)
    return float(max(0.0, 1.0 - d))


def _evaluate_impulse_rules(
    c: np.ndarray,
    piv: List[int],
    bullish: bool,
    config: Optional[ElliottWaveConfig] = None,
) -> ElliottRuleEvaluation:
    """Validate core 5-wave impulse rules and compute Fibonacci alignment score."""

    if len(piv) != 6:
        return ElliottRuleEvaluation(
            valid=False, fib_score=0.0, metrics={}, violations=["pivot_count_not_6"]
        )

    w = [float(c[i]) for i in piv]
    if not all(np.isfinite(v) for v in w):
        return ElliottRuleEvaluation(
            valid=False, fib_score=0.0, metrics={}, violations=["non_finite_prices"]
        )

    L = [w[i + 1] - w[i] for i in range(5)]
    absL = [abs(x) for x in L]

    sgn = 1.0 if bullish else -1.0
    expected = [sgn, -sgn, sgn, -sgn, sgn]
    violations: List[str] = []

    if not all((L[i] > 0) if expected[i] > 0 else (L[i] < 0) for i in range(5)):
        violations.append("direction_sequence_invalid")

    # Rule 2: Wave 2 does not retrace beyond start of wave 1.
    if bullish and w[2] <= w[0]:
        violations.append("wave2_over_retrace")
    if (not bullish) and w[2] >= w[0]:
        violations.append("wave2_over_retrace")

    # Rule 3 compares the three motive-wave lengths on one price scale. Dividing
    # each leg by its own start price can reverse their ordering after a large
    # advance and admit a genuinely shortest wave 3.
    pct1 = absL[0] / abs(w[0]) if w[0] != 0 else 0.0
    pct3 = absL[2] / abs(w[2]) if w[2] != 0 else 0.0
    pct5 = absL[4] / abs(w[4]) if w[4] != 0 else 0.0
    if absL[2] < min(absL[0], absL[4]):
        violations.append("wave3_shortest")

    if bullish and w[3] <= w[1]:
        violations.append("wave3_does_not_exceed_wave1")
    if (not bullish) and w[3] >= w[1]:
        violations.append("wave3_does_not_exceed_wave1")

    # Rule 4: Wave 4 does not overlap Wave 1 territory.
    if bullish and w[4] <= w[1]:
        violations.append("wave4_overlap")
    if (not bullish) and w[4] >= w[1]:
        violations.append("wave4_overlap")

    r2 = absL[1] / absL[0] if absL[0] > 0 else 0.0
    r4 = absL[3] / absL[2] if absL[2] > 0 else 0.0
    e3 = absL[2] / absL[0] if absL[0] > 0 else 0.0
    rel53 = absL[4] / absL[0] if absL[0] > 0 else 0.0
    rel5_alt = absL[4] / abs(w[3] - w[0]) if (w[3] != w[0]) else 0.0

    # Wider common-practice windows (with taper outside). Narrow "ideal"
    # windows used to mid-score many hard-valid non-extended impulses.
    s2 = _window_hit(r2, 0.236, 0.786, taper=0.25)
    s4 = _window_hit(r4, 0.146, 0.500, taper=0.30)
    s3 = max(
        _window_hit(e3, 1.000, 1.618, taper=0.25),
        _window_hit(e3, 1.618, 2.618, taper=0.20),
    )
    s5 = max(
        _window_hit(rel53, 0.618, 1.618, taper=0.25),
        _window_hit(rel5_alt, 0.382, 1.000, taper=0.25),
    )
    w2, w3, w4, w5 = _normalized_impulse_fib_weights(config)
    fib_score = float(w2 * s2 + w3 * s3 + w4 * s4 + w5 * s5)
    direction = 1.0 if bullish else -1.0
    wave5_target_equal_wave1 = float(w[4] + direction * absL[0])
    wave5_target_0618_wave3 = float(w[4] + direction * 0.618 * absL[2])
    wave5_target_zone_low = float(
        min(wave5_target_equal_wave1, wave5_target_0618_wave3)
    )
    wave5_target_zone_high = float(
        max(wave5_target_equal_wave1, wave5_target_0618_wave3)
    )
    valid = len(violations) == 0
    rule_confidence = _rule_confidence_from_scores(valid=valid, fib_score=fib_score)

    metrics = {
        "r2": float(r2),
        "r4": float(r4),
        "e3": float(e3),
        "rel53": float(rel53),
        "rel5_alt": float(rel5_alt),
        "wave1_pct": float(pct1),
        "wave3_pct": float(pct3),
        "wave5_pct": float(pct5),
        "fib_score": fib_score,
        "fib_template_fit": float(fib_score),
        "rule_confidence": float(rule_confidence),
        "s2": float(s2),
        "s3": float(s3),
        "s4": float(s4),
        "s5": float(s5),
        "wave5_target_equal_wave1": wave5_target_equal_wave1,
        "wave5_target_0618_wave3": wave5_target_0618_wave3,
        "wave5_target_zone_low": wave5_target_zone_low,
        "wave5_target_zone_high": wave5_target_zone_high,
        "wave5_targets_are_retrospective": True,
        "fib_weight_wave2": float(w2),
        "fib_weight_wave3": float(w3),
        "fib_weight_wave4": float(w4),
        "fib_weight_wave5": float(w5),
    }
    return ElliottRuleEvaluation(
        valid=valid,
        fib_score=fib_score,
        metrics=metrics,
        violations=violations,
    )


def _rule_confidence_from_scores(*, valid: bool, fib_score: float) -> float:
    """Return template fit for valid geometry; invalid structures score zero."""
    if not valid:
        return 0.0
    fib = float(min(1.0, max(0.0, fib_score)))
    return fib


def _rule_confidence_from_eval(rule_eval: ElliottRuleEvaluation) -> float:
    metrics = rule_eval.metrics if isinstance(rule_eval.metrics, dict) else {}
    raw = metrics.get("rule_confidence")
    if raw is not None:
        try:
            return float(min(1.0, max(0.0, float(raw))))
        except Exception:
            pass
    return _rule_confidence_from_scores(
        valid=bool(rule_eval.valid), fib_score=float(rule_eval.fib_score)
    )


def _apply_confirmation_confidence_adjustments(
    confidence: float,
    pivot_confirmations: Optional[List[bool]],
    config: Optional[ElliottWaveConfig] = None,
) -> Tuple[float, Dict[str, float]]:
    adjusted = float(min(1.0, max(0.0, confidence)))
    adjustments: Dict[str, float] = {}
    if not isinstance(pivot_confirmations, list) or len(pivot_confirmations) == 0:
        return adjusted, adjustments

    confirmations = [bool(v) for v in pivot_confirmations]
    pattern_confirmed = bool(all(confirmations))
    if not pattern_confirmed:
        factor = (
            float(getattr(config, "unconfirmed_terminal_pivot_confidence_cap", 0.72))
            if config is not None
            else 0.72
        )
        factor = float(min(1.0, max(0.0, factor)))
        adjusted *= factor
        adjustments["confirmation_factor"] = factor

    adjusted = float(min(1.0, max(0.0, adjusted)))
    return adjusted, adjustments


def _evaluate_correction_rules(
    c: np.ndarray,
    piv: List[int],
    bullish: bool,
    config: Optional[ElliottWaveConfig] = None,
) -> ElliottRuleEvaluation:
    """Validate outer 3-leg ABC geometry and compute a template-fit score."""

    if len(piv) != 4:
        return ElliottRuleEvaluation(
            valid=False, fib_score=0.0, metrics={}, violations=["pivot_count_not_4"]
        )

    w = [float(c[i]) for i in piv]
    if not all(np.isfinite(v) for v in w):
        return ElliottRuleEvaluation(
            valid=False, fib_score=0.0, metrics={}, violations=["non_finite_prices"]
        )

    L = [w[i + 1] - w[i] for i in range(3)]
    absL = [abs(x) for x in L]
    sgn = 1.0 if bullish else -1.0
    expected = [sgn, -sgn, sgn]
    violations: List[str] = []

    if not all((L[i] > 0) if expected[i] > 0 else (L[i] < 0) for i in range(3)):
        violations.append("direction_sequence_invalid")

    # B should not fully retrace wave A start in a standard zigzag interpretation.
    if bullish and w[2] <= w[0]:
        violations.append("waveB_over_retrace")
    if (not bullish) and w[2] >= w[0]:
        violations.append("waveB_over_retrace")

    # C should have meaningful size versus A to avoid noise classifications.
    min_c_vs_a = (
        float(getattr(config, "correction_min_c_vs_a", 0.25))
        if config is not None
        else 0.25
    )
    if absL[2] < min_c_vs_a * absL[0]:
        violations.append("waveC_too_short")

    b_retrace = absL[1] / absL[0] if absL[0] > 0 else 0.0
    c_vs_a = absL[2] / absL[0] if absL[0] > 0 else 0.0
    # Zigzag-ABC template only (flats/triangles/combinations are out of scope).
    score_b = _window_hit(b_retrace, 0.382, 0.786, taper=0.25)
    score_c = _window_hit(c_vs_a, 0.618, 1.618, taper=0.25)
    fib_score = float(0.5 * (score_b + score_c))
    valid = len(violations) == 0
    rule_confidence = _rule_confidence_from_scores(valid=valid, fib_score=fib_score)

    metrics = {
        "b_retrace": float(b_retrace),
        "c_vs_a": float(c_vs_a),
        "fib_score": fib_score,
        "fib_template_fit": float(fib_score),
        "rule_confidence": float(rule_confidence),
        "structure_type": "outer_abc_candidate",
        "taxonomy_note": (
            "Outer ABC geometry only; internal 5-3-5 subdivisions are not validated."
        ),
    }
    return ElliottRuleEvaluation(
        valid=valid,
        fib_score=fib_score,
        metrics=metrics,
        violations=violations,
    )


def _classification_score_window(
    probs: Optional[np.ndarray],
    cluster_means: Optional[np.ndarray],
    k: int,
    bullish: bool,
    window_len: int,
    trend_slots: List[int],
    counter_slots: List[int],
    wave_index_map: Optional[Dict[int, int]] = None,
    impulsive_cluster: Optional[int] = None,
) -> float:
    if probs is None or cluster_means is None:
        return 0.5
    if cluster_means.ndim != 2 or cluster_means.shape[0] < 1:
        return 0.5

    # Prefer the magnitude-based impulsive cluster when available; fall back to
    # directional cluster so scoring still works if selection fails.
    cluster_for_trend: Optional[int]
    if impulsive_cluster is not None and 0 <= int(impulsive_cluster) < int(
        cluster_means.shape[0]
    ):
        cluster_for_trend = int(impulsive_cluster)
    else:
        cluster_for_trend = _select_directional_cluster(cluster_means, bullish)
    if cluster_for_trend is None:
        return 0.5
    if wave_index_map is not None:
        mapped_rows: List[int] = []
        for offset in range(window_len):
            mapped = wave_index_map.get(int(k + offset))
            if mapped is None:
                return 0.5
            mapped_rows.append(int(mapped))
        window_probs = probs[np.asarray(mapped_rows, dtype=int), cluster_for_trend]
    else:
        if probs.shape[0] < (k + window_len):
            return 0.5
        window_probs = probs[k : k + window_len, cluster_for_trend]
    if window_probs.shape[0] != window_len:
        return 0.5

    trend_vals = [
        float(window_probs[i]) for i in trend_slots if 0 <= int(i) < window_len
    ]
    counter_vals = [
        float(window_probs[i]) for i in counter_slots if 0 <= int(i) < window_len
    ]
    if not trend_vals and not counter_vals:
        return 0.5

    trend_score = float(sum(trend_vals) / len(trend_vals)) if trend_vals else 0.5
    counter_score = (
        float(sum(counter_vals) / len(counter_vals)) if counter_vals else 0.5
    )
    cls_score = 0.5 * trend_score + 0.5 * (1.0 - counter_score)
    return float(min(1.0, max(0.0, cls_score)))


def _classify_waves_through_index(
    features: np.ndarray,
    config: ElliottWaveConfig,
    wave_index_map: Dict[int, int],
    through_wave_idx: int,
    *,
    cache: Optional[
        Dict[
            int,
            Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, Optional[int]],
        ]
    ] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, Optional[int]]:
    through_key = int(through_wave_idx)
    if through_key in wave_index_map:
        max_row = int(wave_index_map[through_key])
    else:
        eligible_rows = [
            int(row)
            for wave_idx, row in wave_index_map.items()
            if int(wave_idx) <= through_key
        ]
        if not eligible_rows:
            return None, None, False, None
        max_row = max(eligible_rows)

    if max_row < 0:
        return None, None, False, None
    if cache is not None and max_row in cache:
        return cache[max_row]

    prefix_features = features[: max_row + 1]
    _, gmm, _, probs, impulsive_cluster = _classify_waves(prefix_features, config)
    cluster_means = gmm.means_ if gmm is not None else None
    classification_available = probs is not None and cluster_means is not None
    result = (
        probs,
        cluster_means,
        classification_available,
        int(impulsive_cluster) if impulsive_cluster is not None else None,
    )
    if cache is not None:
        cache[max_row] = result
    return result


def _result_priority_score(result: ElliottWaveResult) -> float:
    span = max(0, int(result.end_index) - int(result.start_index))
    span_bonus = min(0.35, float(span) / 180.0)
    return float(result.confidence) + span_bonus


def _result_sort_key(
    result: ElliottWaveResult,
) -> Tuple[int, float, int, int, str, Tuple[int, ...]]:
    details = result.details if isinstance(result.details, dict) else {}
    pattern_confirmed = bool(details.get("pattern_confirmed", True))
    span = max(0, int(result.end_index) - int(result.start_index))
    return (
        0 if pattern_confirmed else 1,
        -_result_priority_score(result),
        -span,
        -int(result.end_index),
        str(result.wave_type).lower(),
        tuple(int(i) for i in result.wave_sequence),
    )


def _contiguous_pivot_slices(pivots: List[int], size: int) -> set[Tuple[int, ...]]:
    if size <= 0 or len(pivots) < size:
        return set()
    return {
        tuple(int(v) for v in pivots[i : i + size])
        for i in range(len(pivots) - size + 1)
    }


def _pivot_sequence_is_near_match(
    candidate: List[int],
    reference: Tuple[int, ...],
    *,
    bar_tolerance: int,
    overlap_ratio: float,
) -> bool:
    if len(candidate) != len(reference) or not candidate:
        return False
    if all(
        abs(int(a) - int(b)) <= int(bar_tolerance) for a, b in zip(candidate, reference)
    ):
        return True
    overlap = max(
        _interval_overlap_ratio(
            int(candidate[0]), int(candidate[-1]), int(reference[0]), int(reference[-1])
        ),
        _interval_coverage_ratio(
            int(candidate[0]), int(candidate[-1]), int(reference[0]), int(reference[-1])
        ),
    )
    return bool(overlap >= float(overlap_ratio))


def _scenario_signature(
    scenarios: List["ElliottScenario"],
) -> Tuple[Tuple[str, bool, int, int, int], ...]:
    entries = {
        (
            str(scenario.wave_type).lower(),
            bool(scenario.bullish),
            int(len(scenario.pivots)),
            int(scenario.pivots[0]),
            int(scenario.pivots[-1]),
        )
        for scenario in scenarios
        if scenario.pivots
    }
    return tuple(sorted(entries))


def _scenario_signatures_overlap(
    current: Tuple[Tuple[str, bool, int, int, int], ...],
    prior: Tuple[Tuple[str, bool, int, int, int], ...],
    *,
    overlap_ratio: float,
) -> bool:
    if not current or not prior or len(current) != len(prior):
        return False
    unmatched = list(prior)
    for item in current:
        match_index: Optional[int] = None
        for idx, prior_item in enumerate(unmatched):
            if item[:3] != prior_item[:3]:
                continue
            overlap = max(
                _interval_overlap_ratio(
                    int(item[3]), int(item[4]), int(prior_item[3]), int(prior_item[4])
                ),
                _interval_coverage_ratio(
                    int(item[3]), int(item[4]), int(prior_item[3]), int(prior_item[4])
                ),
            )
            if overlap >= float(overlap_ratio):
                match_index = idx
                break
        if match_index is None:
            return False
        unmatched.pop(match_index)
    return not unmatched


def _elliott_result_key(result: ElliottWaveResult) -> Tuple[str, Tuple[int, ...]]:
    return str(result.wave_type), tuple(int(i) for i in result.wave_sequence)


def _upsert_elliott_result(
    results_by_key: Dict[Tuple[str, Tuple[int, ...]], ElliottWaveResult],
    result: ElliottWaveResult,
) -> None:
    key = _elliott_result_key(result)
    prior = results_by_key.get(key)
    if prior is None or float(result.confidence) > float(prior.confidence):
        results_by_key[key] = result


def _interval_coverage_ratio(
    a_start: int, a_end: int, b_start: int, b_end: int
) -> float:
    lo = max(int(a_start), int(b_start))
    hi = min(int(a_end), int(b_end))
    inter = max(0, hi - lo + 1)
    span = max(0, int(a_end) - int(a_start) + 1)
    if span <= 0:
        return 0.0
    return float(inter) / float(span)


def _filter_overlapping_corrections(
    results: List[ElliottWaveResult],
    *,
    overlap_threshold: float = 0.8,
) -> List[ElliottWaveResult]:
    impulses = [
        result for result in results if str(result.wave_type).lower() == "impulse"
    ]
    if not impulses:
        return results

    filtered: List[ElliottWaveResult] = []
    for result in results:
        if str(result.wave_type).lower() != "correction":
            filtered.append(result)
            continue
        overlaps_impulse = any(
            max(
                _interval_overlap_ratio(
                    int(result.start_index),
                    int(result.end_index),
                    int(impulse.start_index),
                    int(impulse.end_index),
                ),
                _interval_coverage_ratio(
                    int(result.start_index),
                    int(result.end_index),
                    int(impulse.start_index),
                    int(impulse.end_index),
                ),
            )
            >= float(overlap_threshold)
            for impulse in impulses
        )
        if not overlaps_impulse:
            filtered.append(result)
    return filtered


def _result_direction(result: ElliottWaveResult) -> str:
    details = result.details if isinstance(result.details, dict) else {}
    direction = details.get("sequence_direction")
    if direction in (None, ""):
        direction = details.get("trend")
    return str(direction or "").strip().lower()


def _is_nested_same_direction_result(
    candidate: ElliottWaveResult,
    reference: ElliottWaveResult,
    *,
    containment_threshold: float,
) -> bool:
    if str(candidate.wave_type).lower() != str(reference.wave_type).lower():
        return False
    candidate_direction = _result_direction(candidate)
    reference_direction = _result_direction(reference)
    if (
        candidate_direction
        and reference_direction
        and candidate_direction != reference_direction
    ):
        return False
    candidate_start = int(candidate.start_index)
    candidate_end = int(candidate.end_index)
    reference_start = int(reference.start_index)
    reference_end = int(reference.end_index)
    if candidate_start < reference_start or candidate_end > reference_end:
        return False
    overlap = max(
        _interval_overlap_ratio(
            candidate_start, candidate_end, reference_start, reference_end
        ),
        _interval_coverage_ratio(
            candidate_start, candidate_end, reference_start, reference_end
        ),
    )
    return bool(overlap >= float(containment_threshold))


def _filter_nested_results(
    results: List[ElliottWaveResult],
    *,
    containment_threshold: float = 0.9,
) -> List[ElliottWaveResult]:
    if len(results) <= 1:
        return results
    kept: List[ElliottWaveResult] = []

    def _dominance_key(
        result: ElliottWaveResult,
    ) -> Tuple[int, int, float, int, str, Tuple[int, ...]]:
        details = result.details if isinstance(result.details, dict) else {}
        pattern_confirmed = bool(details.get("pattern_confirmed", True))
        span = max(0, int(result.end_index) - int(result.start_index))
        return (
            0 if pattern_confirmed else 1,
            -span,
            -_result_priority_score(result),
            -int(result.end_index),
            str(result.wave_type).lower(),
            tuple(int(i) for i in result.wave_sequence),
        )

    for result in sorted(results, key=_dominance_key):
        if any(
            _is_nested_same_direction_result(
                result, prior, containment_threshold=containment_threshold
            )
            for prior in kept
        ):
            continue
        kept.append(result)
    return kept


def _dedupe_similar_waves(
    results: List[ElliottWaveResult],
    *,
    proximity_bars: int = 24,
) -> List[ElliottWaveResult]:
    """Deduplicate near-duplicate patterns of the same type and direction.

    Keep the highest-confidence pattern when multiple patterns share wave type
    and sequence direction and end within ``proximity_bars`` of each other.
    Opposite-direction structures are retained even if they end nearby.
    """
    if len(results) <= 1:
        return results

    def _group_key(result: ElliottWaveResult) -> Tuple[str, str]:
        return (str(result.wave_type).lower(), _result_direction(result) or "")

    by_key: Dict[Tuple[str, str], List[ElliottWaveResult]] = {}
    for r in results:
        by_key.setdefault(_group_key(r), []).append(r)

    kept: List[ElliottWaveResult] = []
    for key, group in by_key.items():
        sorted_group = sorted(
            group,
            key=lambda r: (
                0 if bool((r.details or {}).get("pattern_confirmed", True)) else 1,
                -float(r.confidence),
                -int(r.end_index),
            ),
        )
        for candidate in sorted_group:
            is_duplicate = False
            for kept_pattern in [k for k in kept if _group_key(k) == key]:
                end_diff = abs(int(candidate.end_index) - int(kept_pattern.end_index))
                overlap = _interval_overlap_ratio(
                    int(candidate.start_index),
                    int(candidate.end_index),
                    int(kept_pattern.start_index),
                    int(kept_pattern.end_index),
                )
                if end_diff <= proximity_bars and overlap >= 0.5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(candidate)

    kept.sort(key=lambda r: (-int(r.end_index), -float(r.confidence)))
    return kept


def _group_scale_alternatives(
    results: List[ElliottWaveResult],
) -> List[ElliottWaveResult]:
    """Collapse near-identical cross-scale counts without removing nested degrees."""
    if len(results) <= 1:
        return results

    groups: List[List[ElliottWaveResult]] = []
    for result in results:
        matched: Optional[List[ElliottWaveResult]] = None
        for group in groups:
            reference = group[0]
            if str(result.wave_type).lower() != str(reference.wave_type).lower():
                continue
            if _result_direction(result) != _result_direction(reference):
                continue
            if len(result.wave_sequence) != len(reference.wave_sequence):
                continue
            overlap = _interval_overlap_ratio(
                int(result.start_index),
                int(result.end_index),
                int(reference.start_index),
                int(reference.end_index),
            )
            tolerance = max(
                3,
                int(
                    0.10
                    * min(
                        int(result.end_index) - int(result.start_index) + 1,
                        int(reference.end_index) - int(reference.start_index) + 1,
                    )
                ),
            )
            aligned = all(
                abs(int(a) - int(b)) <= tolerance
                for a, b in zip(result.wave_sequence, reference.wave_sequence)
            )
            if overlap >= 0.6 and aligned:
                matched = group
                break
        if matched is None:
            groups.append([result])
        else:
            matched.append(result)

    selected: List[ElliottWaveResult] = []
    for group_index, group in enumerate(groups, start=1):
        winner = min(group, key=_result_sort_key)
        details = winner.details if isinstance(winner.details, dict) else {}
        details["alternate_group_id"] = f"alt-{group_index:03d}"
        details["alternate_count"] = int(len(group))
        details["alternate_scan_scales"] = sorted(
            {
                str((item.details or {}).get("scan_scale_id") or "")
                for item in group
                if str((item.details or {}).get("scan_scale_id") or "")
            }
        )
        winner.details = details
        selected.append(winner)
    return selected


def _causal_result_sort_key(
    result: ElliottWaveResult, *, n_bars: int, recent_bars: int
) -> Tuple[int, float, float, int, str, Tuple[int, ...]]:
    details = result.details if isinstance(result.details, dict) else {}
    state = str(details.get("structure_state") or result.structure_state).lower()
    available = int(
        details.get("available_at_index")
        if details.get("available_at_index") is not None
        else result.available_at_index
        if result.available_at_index is not None
        else result.end_index
    )
    recent = available >= int(n_bars - max(1, recent_bars))
    category = (
        0
        if state == "developing"
        else 1
        if state == "fallback"
        else 2
        if recent
        else 3
    )
    structural = float(details.get("structural_score") or 0.0)
    candidate = float(details.get("candidate_score") or 0.0)
    return (
        category,
        -structural,
        -candidate,
        -available,
        str(result.wave_type).lower(),
        tuple(int(i) for i in result.wave_sequence),
    )


class ElliottWaveAnalyzer:
    """Facade-style analyzer inspired by ta4j's scenario pipeline."""

    def __init__(
        self,
        close: np.ndarray,
        times: np.ndarray,
        config: ElliottWaveConfig,
        *,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        pivot_signal: Optional[np.ndarray] = None,
    ):
        self.close = close
        self.pivot_signal = (
            np.asarray(pivot_signal, dtype=float)
            if isinstance(pivot_signal, np.ndarray)
            and int(pivot_signal.size) == int(close.size)
            else close
        )
        self.times = times
        self.config = config
        self.high = (
            high if isinstance(high, np.ndarray) else np.asarray([], dtype=float)
        )
        self.low = low if isinstance(low, np.ndarray) else np.asarray([], dtype=float)
        # Request-scoped caches
        self._pivot_cache: Dict[Tuple[float, int], List[int]] = {}
        self._pivot_record_cache: Dict[Tuple[float, int], List[ElliottPivot]] = {}
        self._wave_feature_cache: Dict[
            Tuple[float, int], Tuple[np.ndarray, Dict[int, int], List[Tuple[int, int]]]
        ] = {}

    def _normalized_pivot_price_source(self) -> str:
        raw = (
            str(getattr(self.config, "pivot_price_source", "close") or "close")
            .strip()
            .lower()
        )
        return raw if raw in {"close", "ohlc"} else "close"

    def _pivot_display_price(
        self,
        idx: int,
        *,
        pivot_pos: int,
        bullish: bool,
        pivot_record: Optional[ElliottPivot] = None,
    ) -> float:
        close_price = float(self.close[idx])
        source = self._normalized_pivot_price_source()
        if source == "close":
            return close_price
        if pivot_record is not None and pivot_record.price is not None:
            return float(pivot_record.price)
        is_peak = bool(
            pivot_record.kind == "peak"
            if pivot_record is not None
            else (pivot_pos % 2 == 1)
            if bullish
            else (pivot_pos % 2 == 0)
        )
        price_source = self.high if is_peak else self.low
        if price_source.size > int(idx):
            try:
                price = float(price_source[int(idx)])
                if np.isfinite(price):
                    return price
            except Exception:
                pass
        return close_price

    def _display_wave5_targets(self, pivot_prices: List[float]) -> Dict[str, float]:
        if len(pivot_prices) < 6:
            return {}
        wave1_move = float(pivot_prices[1]) - float(pivot_prices[0])
        wave3_move = float(pivot_prices[3]) - float(pivot_prices[2])
        wave4_price = float(pivot_prices[4])
        equal_wave1 = float(wave4_price + wave1_move)
        wave3_0618 = float(wave4_price + (0.618 * wave3_move))
        return {
            "equal_wave1": equal_wave1,
            "wave3_0_618": wave3_0618,
            "zone_low": float(min(equal_wave1, wave3_0618)),
            "zone_high": float(max(equal_wave1, wave3_0618)),
            "retrospective": True,
        }

    def _price_series_for_pivots(
        self,
        piv_seq: List[int],
        *,
        bullish: bool,
        pivot_records: Optional[List[ElliottPivot]] = None,
    ) -> np.ndarray:
        """Return a price series where pivot indices use the configured price source.

        ZigZag still locates pivots on close; rule evaluation and display then
        share this series so invalidation/targets match the validated geometry.
        """
        source = self._normalized_pivot_price_source()
        if source == "close":
            return self.close
        arr = np.array(self.close, copy=True, dtype=float)
        for j, idx in enumerate(piv_seq):
            record = (
                pivot_records[j]
                if isinstance(pivot_records, list) and j < len(pivot_records)
                else None
            )
            arr[int(idx)] = float(
                self._pivot_display_price(
                    int(idx),
                    pivot_pos=j,
                    bullish=bullish,
                    pivot_record=record,
                )
            )
        return arr

    def _sequence_bullish(
        self,
        piv_seq: List[int],
        pivot_records: Optional[List[ElliottPivot]] = None,
    ) -> Optional[bool]:
        if len(piv_seq) < 2:
            return None
        start = float(self.close[int(piv_seq[0])])
        end = float(self.close[int(piv_seq[-1])])
        if not (np.isfinite(start) and np.isfinite(end)) or start == end:
            return None
        bullish = bool(end > start)
        if self._normalized_pivot_price_source() == "close":
            return bullish
        if isinstance(pivot_records, list) and len(pivot_records) == len(piv_seq):
            prices = [
                float(record.price)
                for record in pivot_records
                if record.price is not None
            ]
            if len(prices) == len(piv_seq):
                if prices[-1] > prices[0]:
                    return True
                if prices[-1] < prices[0]:
                    return False
        prices = [
            float(self._pivot_display_price(int(idx), pivot_pos=j, bullish=bullish))
            for j, idx in enumerate(piv_seq)
        ]
        if prices[-1] > prices[0]:
            return True
        if prices[-1] < prices[0]:
            return False
        return bullish

    def _get_pivots(self, threshold_pct: float, min_distance: int) -> List[int]:
        """Return pivot indices, caching by (threshold, min_distance)."""
        key = (float(threshold_pct), int(min_distance))
        cached = self._pivot_cache.get(key)
        if cached is not None:
            return cached
        if self._normalized_pivot_price_source() == "ohlc":
            piv_idx, piv_dir = _ohlc_zigzag_pivots_indices(
                self.high, self.low, float(threshold_pct)
            )
            geometry = np.array(self.close, copy=True, dtype=float)
            for idx, direction in zip(piv_idx, piv_dir):
                geometry[int(idx)] = float(
                    self.high[int(idx)] if direction == "up" else self.low[int(idx)]
                )
        else:
            piv_idx, piv_dir = _zigzag_pivots_indices(
                self.pivot_signal, float(threshold_pct)
            )
            geometry = self.pivot_signal
        kind_by_index = {
            int(idx): "peak" if direction == "up" else "trough"
            for idx, direction in zip(piv_idx, piv_dir)
            if direction in {"up", "down"}
        }
        price_by_index = {
            int(idx): float(geometry[int(idx)])
            for idx in piv_idx
        }
        piv_idx = _enforce_min_distance_on_pivots(piv_idx, geometry, min_distance)
        piv_idx = _enforce_pivot_alternation(piv_idx, geometry)
        piv_idx = _enforce_kind_alternation(
            piv_idx, kind_by_index, price_by_index
        )
        self._pivot_cache[key] = piv_idx
        records = _build_pivot_records(
            piv_idx,
            geometry,
            float(threshold_pct),
            high=self.high if self._normalized_pivot_price_source() == "ohlc" else None,
            low=self.low if self._normalized_pivot_price_source() == "ohlc" else None,
            kinds=kind_by_index,
            prices=price_by_index,
        )
        if self._normalized_pivot_price_source() == "close":
            from dataclasses import replace as _replace_pivot

            rewritten: List[ElliottPivot] = []
            for record in records:
                idx = int(record.index)
                if 0 <= idx < int(self.close.size) and np.isfinite(float(self.close[idx])):
                    rewritten.append(
                        _replace_pivot(
                            record,
                            price=float(self.close[idx]),
                            price_source="close",
                        )
                    )
                else:
                    rewritten.append(record)
            records = rewritten
        self._pivot_record_cache[key] = records
        return piv_idx

    def pivot_signature(
        self, threshold_pct: float, min_distance: int
    ) -> Tuple[int, ...]:
        """Return pivot signature tuple, leveraging the pivot cache."""
        return tuple(int(i) for i in self._get_pivots(threshold_pct, min_distance))

    def analyze_once(  # noqa: C901 - explicit rule pipeline is audit-friendly
        self, threshold_pct: float, min_distance: int
    ) -> List[ElliottScenario]:
        piv_idx = self._get_pivots(float(threshold_pct), int(min_distance))
        if len(piv_idx) < 4:
            return []

        # Cache wave features by the same key
        feat_key = (float(threshold_pct), int(min_distance))
        cached_feat = self._wave_feature_cache.get(feat_key)
        if cached_feat is not None:
            features, wave_index_map, waves = cached_feat
        else:
            waves = _segment_waves_from_pivots(piv_idx)
            if not waves:
                return []
            features, wave_index_map = _extract_wave_features_with_index(
                waves, self.close
            )
            if features.shape[0] == 0:
                return []
            self._wave_feature_cache[feat_key] = (features, wave_index_map, waves)

        out: List[ElliottScenario] = []
        pattern_types = _normalize_pattern_types(self.config)
        total_waves = len(waves)
        min_wave_span = int(max(1, max(min_distance, self.config.wave_min_len)))
        correction_exclusions: List[Tuple[int, ...]] = []
        correction_bar_tolerance = max(
            0, int(getattr(self.config, "correction_exclusion_bar_tolerance", 1))
        )
        correction_overlap_ratio = float(
            max(
                0.0,
                min(
                    1.0, getattr(self.config, "correction_exclusion_overlap_ratio", 0.9)
                ),
            )
        )
        classification_cache: Dict[
            int,
            Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, Optional[int]],
        ] = {}
        record_by_index = {
            int(record.index): record
            for record in self._pivot_record_cache.get(
                (float(threshold_pct), int(min_distance)), []
            )
        }

        if "impulse" in pattern_types:
            for k in range(0, total_waves - 4):
                piv_seq = [int(x) for x in piv_idx[k : k + 6]]
                if len(piv_seq) < 6:
                    continue

                if any((piv_seq[j + 1] - piv_seq[j]) < min_wave_span for j in range(5)):
                    continue
                if (piv_seq[-1] - piv_seq[0] + 1) < int(self.config.min_impulse_bars):
                    continue
                if self.config.max_pattern_span_bars is not None and (
                    piv_seq[-1] - piv_seq[0] + 1
                ) > int(self.config.max_pattern_span_bars):
                    continue
                if self.config.max_pattern_age_bars is not None and (
                    int(self.close.size) - 1 - piv_seq[-1]
                ) > int(self.config.max_pattern_age_bars):
                    continue

                scenario_records = [record_by_index[idx] for idx in piv_seq]
                bullish_opt = self._sequence_bullish(piv_seq, scenario_records)
                if bullish_opt is None:
                    continue
                bullish = bool(bullish_opt)

                price_series = self._price_series_for_pivots(
                    piv_seq, bullish=bullish, pivot_records=scenario_records
                )
                rule_eval = _evaluate_impulse_rules(
                    price_series, piv_seq, bullish=bullish, config=self.config
                )
                if not rule_eval.valid:
                    continue
                structural_score = float(rule_eval.fib_score)
                if structural_score < float(self.config.min_structural_score):
                    continue

                if bool(getattr(self.config, "enable_gmm_classifier", False)):
                    probs, cluster_means, classification_available, impulsive_cluster = _classify_waves_through_index(
                        features,
                        self.config,
                        wave_index_map,
                        through_wave_idx=int(k + 4),
                        cache=classification_cache,
                    )
                else:
                    probs, cluster_means, classification_available, impulsive_cluster = (None, None, False, None)
                cls_score = _classification_score_window(
                    probs,
                    cluster_means,
                    k,
                    bullish,
                    window_len=5,
                    trend_slots=[0, 2, 4],
                    counter_slots=[1, 3],
                    wave_index_map=wave_index_map,
                    impulsive_cluster=impulsive_cluster,
                )
                pivot_confirmations = [record.confirmed for record in scenario_records]
                base_confidence = structural_score
                confidence, confidence_adjustments = (
                    _apply_confirmation_confidence_adjustments(
                        base_confidence,
                        pivot_confirmations,
                        self.config,
                    )
                )
                if confidence < float(self.config.min_confidence):
                    continue

                out.append(
                    ElliottScenario(
                        pivots=piv_seq,
                        bullish=bullish,
                        confidence=confidence,
                        base_confidence=base_confidence,
                        cls_score=cls_score,
                        rule_eval=rule_eval,
                        threshold_used=float(threshold_pct),
                        min_distance_used=int(min_distance),
                        wave_type="Impulse",
                        classification_available=classification_available,
                        pivot_confirmations=pivot_confirmations,
                        confidence_adjustments=confidence_adjustments,
                        pivot_records=scenario_records,
                    )
                )
                correction_exclusions.extend(_contiguous_pivot_slices(piv_seq, 4))

        if "correction" in pattern_types:
            for k in range(0, total_waves - 2):
                piv_seq = [int(x) for x in piv_idx[k : k + 4]]
                if len(piv_seq) < 4:
                    continue
                if any(
                    _pivot_sequence_is_near_match(
                        piv_seq,
                        excluded,
                        bar_tolerance=correction_bar_tolerance,
                        overlap_ratio=correction_overlap_ratio,
                    )
                    for excluded in correction_exclusions
                ):
                    continue

                if any((piv_seq[j + 1] - piv_seq[j]) < min_wave_span for j in range(3)):
                    continue
                if (piv_seq[-1] - piv_seq[0] + 1) < int(self.config.min_correction_bars):
                    continue
                if self.config.max_pattern_span_bars is not None and (
                    piv_seq[-1] - piv_seq[0] + 1
                ) > int(self.config.max_pattern_span_bars):
                    continue
                if self.config.max_pattern_age_bars is not None and (
                    int(self.close.size) - 1 - piv_seq[-1]
                ) > int(self.config.max_pattern_age_bars):
                    continue

                scenario_records = [record_by_index[idx] for idx in piv_seq]
                bullish_opt = self._sequence_bullish(piv_seq, scenario_records)
                if bullish_opt is None:
                    continue
                bullish = bool(bullish_opt)

                price_series = self._price_series_for_pivots(
                    piv_seq, bullish=bullish, pivot_records=scenario_records
                )
                rule_eval = _evaluate_correction_rules(
                    price_series,
                    piv_seq,
                    bullish=bullish,
                    config=self.config,
                )
                if not rule_eval.valid:
                    continue
                structural_score = float(rule_eval.fib_score)
                if structural_score < float(self.config.min_structural_score):
                    continue

                if bool(getattr(self.config, "enable_gmm_classifier", False)):
                    probs, cluster_means, classification_available, impulsive_cluster = _classify_waves_through_index(
                        features,
                        self.config,
                        wave_index_map,
                        through_wave_idx=int(k + 2),
                        cache=classification_cache,
                    )
                else:
                    probs, cluster_means, classification_available, impulsive_cluster = (None, None, False, None)
                cls_score = _classification_score_window(
                    probs,
                    cluster_means,
                    k,
                    bullish,
                    window_len=3,
                    trend_slots=[0, 2],
                    counter_slots=[1],
                    wave_index_map=wave_index_map,
                    impulsive_cluster=impulsive_cluster,
                )
                pivot_confirmations = [record.confirmed for record in scenario_records]
                base_confidence = structural_score
                confidence, confidence_adjustments = (
                    _apply_confirmation_confidence_adjustments(
                        base_confidence,
                        pivot_confirmations,
                        self.config,
                    )
                )
                if confidence < float(self.config.min_confidence):
                    continue

                out.append(
                    ElliottScenario(
                        pivots=piv_seq,
                        bullish=bullish,
                        confidence=confidence,
                        base_confidence=base_confidence,
                        cls_score=cls_score,
                        rule_eval=rule_eval,
                        threshold_used=float(threshold_pct),
                        min_distance_used=int(min_distance),
                        wave_type="Correction",
                        classification_available=classification_available,
                        pivot_confirmations=pivot_confirmations,
                        confidence_adjustments=confidence_adjustments,
                        pivot_records=scenario_records,
                    )
                )

        return out

    def build_result(self, scenario: ElliottScenario) -> ElliottWaveResult:
        piv_seq = [int(x) for x in scenario.pivots]
        start_index = int(piv_seq[0])
        end_index = int(piv_seq[-1])
        pivot_price_source = self._normalized_pivot_price_source()
        wave_type_key = str(scenario.wave_type).lower()
        if wave_type_key == "correction" and len(piv_seq) == 4:
        # S/A/B/C are outer pivot labels; internal subdivisions are not modeled.
            labels = ["S", "A", "B", "C"]
            structure_type = "outer_abc_candidate"
            pattern_family = "correction"
        elif wave_type_key == "impulse" and len(piv_seq) == 6:
            # Standard Elliott pivot numbers 0..5 (not wave-leg names).
            labels = [str(j) for j in range(len(piv_seq))]
            structure_type = "impulse_strict"
            pattern_family = "impulse"
        elif wave_type_key == "candidate":
            labels = [str(j) for j in range(len(piv_seq))]
            structure_type = "candidate"
            pattern_family = "candidate"
        else:
            labels = [str(j) for j in range(len(piv_seq))]
            structure_type = wave_type_key or "unknown"
            pattern_family = wave_type_key or "unknown"

        wave_points_labeled: List[Dict[str, Any]] = []
        pivot_prices: List[float] = []
        records = scenario.pivot_records or []
        for j, idx in enumerate(piv_seq):
            ti = PatternResultBase.resolve_time(self.times, idx)
            is_confirmed = True
            if isinstance(scenario.pivot_confirmations, list) and j < len(
                scenario.pivot_confirmations
            ):
                is_confirmed = bool(scenario.pivot_confirmations[j])
            pivot_price = self._pivot_display_price(
                int(idx),
                pivot_pos=j,
                bullish=bool(scenario.bullish),
                pivot_record=records[j] if j < len(records) else None,
            )
            pivot_prices.append(float(pivot_price))
            wave_points_labeled.append(
                {
                    "label": labels[j],
                    "index": int(idx),
                    "time": ti,
                    "price": float(pivot_price),
                    "is_confirmed": bool(is_confirmed),
                }
            )

        confirmations = (
            [bool(v) for v in scenario.pivot_confirmations]
            if isinstance(scenario.pivot_confirmations, list)
            else []
        )
        pattern_confirmed = bool(all(confirmations)) if confirmations else True
        terminal_confirmed = bool(confirmations[-1]) if confirmations else True
        available_at_index = end_index
        if records and all(record.confirmed for record in records):
            confirmed_at = [
                int(record.confirmation_index)
                for record in records
                if record.confirmation_index is not None
            ]
            if confirmed_at:
                available_at_index = max(confirmed_at)
        elif self.close.size:
            available_at_index = int(self.close.size - 1)
        invalidation_level = float(pivot_prices[0]) if pivot_prices else None
        sequence_direction = "bull" if scenario.bullish else "bear"
        rule_confidence = _rule_confidence_from_eval(scenario.rule_eval)
        template_fit = float(min(1.0, max(0.0, scenario.rule_eval.fib_score)))
        structural_score = template_fit if scenario.rule_eval.valid else 0.0
        span_bars = int(end_index - start_index + 1)
        scale_id = (
            f"t{float(scenario.threshold_used):g}-d{int(scenario.min_distance_used)}"
        )

        details: Dict[str, Any] = {
            "wave_points": [float(price) for price in pivot_prices],
            "wave_points_labeled": wave_points_labeled,
            "bullish": bool(scenario.bullish),
            "trend": sequence_direction,
            "sequence_direction": sequence_direction,
            "pattern_family": pattern_family,
            "structure_type": structure_type,
            "schema_version": 2,
            "validation_scope": "outer_leg_heuristic",
            "pivot_price_source": pivot_price_source,
            # Unified price contract: rules and display share pivot_price_source.
            "rule_price_source": pivot_price_source,
            "price_contract": "unified",
            "fib_metrics": dict(scenario.rule_eval.metrics),
            "rule_valid": bool(scenario.rule_eval.valid),
            "template_fit": template_fit,
            "fib_template_fit": template_fit,
            "structural_score": structural_score,
            "confidence_basis": "heuristic_not_probability",
            "rule_confidence": float(rule_confidence),
            "rule_violations": list(scenario.rule_eval.violations),
            "cls_score": float(scenario.cls_score),
            "base_confidence": float(
                scenario.base_confidence
                if scenario.base_confidence is not None
                else scenario.confidence
            ),
            "scan_threshold_pct": float(scenario.threshold_used),
            "scan_scale_id": scale_id,
            "min_distance_used": int(scenario.min_distance_used),
            "fallback_candidate": bool(scenario.fallback_candidate),
            "synthetic_terminal_pivot": bool(scenario.synthetic_terminal_pivot),
            "classification_available": bool(scenario.classification_available),
            "pattern_confirmed": pattern_confirmed,
            "structure_complete": pattern_confirmed,
            "terminal_confirmed": terminal_confirmed,
            "has_unconfirmed_terminal_pivot": not terminal_confirmed,
            "invalidation_level": invalidation_level,
            "structure_state": (
                "fallback"
                if scenario.fallback_candidate
                else "confirmed"
                if pattern_confirmed
                else "developing"
            ),
            "geometry_end_index": end_index,
            "span_bars": span_bars,
            "span_share_of_lookback": float(span_bars / max(1, self.close.size)),
            "available_at_index": int(available_at_index),
            "available_at_time": PatternResultBase.resolve_time(
                self.times, int(available_at_index)
            ),
            "status_basis": (
                "causal_confirmation_given_terminal_scale"
                if str(getattr(self.config, "scale_mode", "")).strip().lower() == "auto"
                else "causal_confirmation"
            ),
        }
        if scenario.fallback_candidate:
            details["candidate_score"] = float(scenario.confidence)
        if records:
            details["pivots"] = [
                {
                    "index": int(record.index),
                    "time": PatternResultBase.resolve_time(
                        self.times, int(record.index)
                    ),
                    "price": float(pivot_prices[pos]),
                    "kind": str(record.kind),
                    "confirmed": bool(record.confirmed),
                    "confirmation_index": (
                        int(record.confirmation_index)
                        if record.confirmation_index is not None
                        else None
                    ),
                    "confirmation_time": PatternResultBase.resolve_time(
                        self.times, int(record.confirmation_index)
                    )
                    if record.confirmation_index is not None
                    else None,
                }
                for pos, record in enumerate(records)
            ]
        if scenario.confidence_adjustments:
            details["confidence_adjustments"] = {
                str(key): float(value)
                for key, value in scenario.confidence_adjustments.items()
            }
        if wave_type_key == "correction":
            details["correction_metrics"] = dict(scenario.rule_eval.metrics)
            details["taxonomy_note"] = (
                "Outer ABC geometry only; 5-3-5 subdivisions and larger-degree trend are not validated."
            )
        if wave_type_key == "impulse":
            details["taxonomy_note"] = (
                "Strict motive impulse (no diagonals); single ZigZag degree only."
            )
            wave5_targets = self._display_wave5_targets(pivot_prices)
            if not wave5_targets:
                wave5_targets = {}
                for src_key, out_key in (
                    ("wave5_target_equal_wave1", "equal_wave1"),
                    ("wave5_target_0618_wave3", "wave3_0_618"),
                    ("wave5_target_zone_low", "zone_low"),
                    ("wave5_target_zone_high", "zone_high"),
                ):
                    value = scenario.rule_eval.metrics.get(src_key)
                    if value is None:
                        continue
                    try:
                        wave5_targets[out_key] = float(value)
                    except Exception:
                        continue
                if wave5_targets:
                    wave5_targets["retrospective"] = True
            if wave5_targets:
                details["wave5_targets"] = wave5_targets
        if scenario.fallback_candidate and scenario.validated_wave_type:
            details["candidate_validates_as"] = str(
                scenario.validated_wave_type
            ).lower()

        return ElliottWaveResult(
            wave_type=scenario.wave_type,
            wave_sequence=piv_seq,
            confidence=float(scenario.confidence),
            start_index=start_index,
            end_index=end_index,
            start_time=PatternResultBase.resolve_time(self.times, start_index),
            end_time=PatternResultBase.resolve_time(self.times, end_index),
            details=details,
            available_at_index=int(available_at_index),
            available_at_time=PatternResultBase.resolve_time(
                self.times, int(available_at_index)
            ),
            structure_state=str(details["structure_state"]),
        )

    def build_fallback(
        self, threshold_base: float, min_distance: int
    ) -> Optional[ElliottWaveResult]:
        if not bool(self.config.include_fallback_candidate):
            return None

        n = int(self.close.size)
        if n < 2:
            return None
        if not np.isfinite(float(self.close[-1])):
            return None

        thr_cand = float(min(0.2, max(0.01, threshold_base)))
        piv_idx = list(self._get_pivots(thr_cand, max(1, min_distance)))
        if len(piv_idx) < 1:
            return None

        synthetic_terminal_pivot = False
        if int(piv_idx[-1]) != int(n - 1):
            piv_idx = list(piv_idx) + [int(n - 1)]
            synthetic_terminal_pivot = True
            piv_idx = _enforce_min_distance_on_pivots(
                piv_idx, self.pivot_signal, max(1, min_distance)
            )
            piv_idx = _enforce_pivot_alternation(piv_idx, self.pivot_signal)
            if not piv_idx or int(piv_idx[-1]) != int(n - 1):
                return None
        if len(piv_idx) < 2:
            return None

        pattern_types = _normalize_pattern_types(self.config)
        preferred_len = 6 if "impulse" in pattern_types else 4
        seq_len = min(preferred_len, len(piv_idx))
        piv_seq = [int(i) for i in piv_idx[-seq_len:]]
        min_wave_span = int(max(1, max(min_distance, self.config.wave_min_len)))
        if any(
            (piv_seq[j + 1] - piv_seq[j]) < min_wave_span
            for j in range(len(piv_seq) - 1)
        ):
            return None
        bullish_opt = self._sequence_bullish(piv_seq)
        bullish = bool(bullish_opt) if bullish_opt is not None else bool(
            float(self.close[piv_seq[-1]]) > float(self.close[piv_seq[0]])
        )

        rule_eval = ElliottRuleEvaluation(
            valid=False, fib_score=0.0, metrics={}, violations=["fallback_candidate"]
        )
        validated_wave_type: Optional[str] = None
        price_series = self._price_series_for_pivots(piv_seq, bullish=bullish)
        if len(piv_seq) == 6 and "impulse" in pattern_types:
            rule_eval = _evaluate_impulse_rules(
                price_series, piv_seq, bullish=bullish, config=self.config
            )
            if rule_eval.valid:
                validated_wave_type = "Impulse"
        elif len(piv_seq) == 4 and "correction" in pattern_types:
            rule_eval = _evaluate_correction_rules(
                price_series,
                piv_seq,
                bullish=bullish,
                config=self.config,
            )
            if rule_eval.valid:
                validated_wave_type = "Correction"

        if validated_wave_type is not None:
            conf_raw = 0.35 * float(_rule_confidence_from_eval(rule_eval))
        elif len(piv_seq) in (4, 6):
            conf_raw = 0.30 * float(rule_eval.fib_score)
        else:
            conf_raw = 0.2
        conf = float(min(0.45, max(0.1, conf_raw)))
        required_span = (
            int(self.config.min_impulse_bars)
            if len(piv_seq) == 6
            else int(self.config.min_correction_bars)
            if len(piv_seq) == 4
            else 1
        )
        if (piv_seq[-1] - piv_seq[0] + 1) < required_span:
            return None
        if self.config.max_pattern_span_bars is not None and (
            piv_seq[-1] - piv_seq[0] + 1
        ) > int(self.config.max_pattern_span_bars):
            return None
        if conf < float(self.config.min_confidence):
            return None

        fallback_kinds = {
            int(idx): (
                "peak"
                if ((pos % 2 == 1) if bullish else (pos % 2 == 0))
                else "trough"
            )
            for pos, idx in enumerate(piv_seq)
        }
        fallback_prices = {
            int(idx): float(
                self.high[int(idx)]
                if self._normalized_pivot_price_source() == "ohlc"
                and fallback_kinds[int(idx)] == "peak"
                else self.low[int(idx)]
                if self._normalized_pivot_price_source() == "ohlc"
                else self.close[int(idx)]
            )
            for idx in piv_seq
        }
        pivot_records = _build_pivot_records(
            piv_seq,
            self.pivot_signal,
            thr_cand,
            high=self.high if self._normalized_pivot_price_source() == "ohlc" else None,
            low=self.low if self._normalized_pivot_price_source() == "ohlc" else None,
            kinds=fallback_kinds,
            prices=fallback_prices,
        )
        if pivot_records:
            pivot_records[-1] = ElliottPivot(
                index=pivot_records[-1].index,
                kind=pivot_records[-1].kind,
                confirmed=False,
                confirmation_index=None,
                price=pivot_records[-1].price,
                price_source=pivot_records[-1].price_source,
            )

        scenario = ElliottScenario(
            pivots=piv_seq,
            bullish=bullish,
            confidence=conf,
            cls_score=0.5,
            rule_eval=rule_eval,
            threshold_used=thr_cand,
            min_distance_used=int(min_distance),
            fallback_candidate=True,
            synthetic_terminal_pivot=bool(synthetic_terminal_pivot),
            wave_type="Candidate",
            validated_wave_type=validated_wave_type,
            classification_available=False,
            pivot_confirmations=[record.confirmed for record in pivot_records],
            pivot_records=pivot_records,
        )
        return self.build_result(scenario)


def _adaptive_close_pivots(
    signal: np.ndarray, threshold_pct: float, min_distance: int
) -> List[int]:
    """Build close-signal pivots with the same cleanup used by the analyzer."""
    pivots, directions = _zigzag_pivots_indices(signal, float(threshold_pct))
    kinds = {
        int(index): "peak" if direction == "up" else "trough"
        for index, direction in zip(pivots, directions)
        if direction in {"up", "down"}
    }
    prices = {int(index): float(signal[int(index)]) for index in pivots}
    pivots = _enforce_min_distance_on_pivots(
        pivots, signal, max(1, int(min_distance))
    )
    pivots = _enforce_pivot_alternation(pivots, signal)
    return _enforce_kind_alternation(pivots, kinds, prices)


def _adaptive_ohlc_pivots(
    signal: np.ndarray,
    threshold_pct: float,
    min_distance: int,
    *,
    high: np.ndarray,
    low: np.ndarray,
) -> List[int]:
    """Build OHLC pivots for adaptive density selection."""
    n = int(signal.size)
    h = np.asarray(high[:n], dtype=float)
    l = np.asarray(low[:n], dtype=float)
    pivots, directions = _ohlc_zigzag_pivots_indices(
        h, l, float(threshold_pct)
    )
    geometry = np.asarray(signal, dtype=float).copy()
    kinds: Dict[int, str] = {}
    prices: Dict[int, float] = {}
    for index, direction in zip(pivots, directions):
        idx = int(index)
        kind = "peak" if direction == "up" else "trough"
        price = float(h[idx] if kind == "peak" else l[idx])
        kinds[idx] = kind
        prices[idx] = price
        geometry[idx] = price
    pivots = _enforce_min_distance_on_pivots(
        pivots, geometry, max(1, int(min_distance))
    )
    pivots = _enforce_pivot_alternation(pivots, geometry)
    return _enforce_kind_alternation(pivots, kinds, prices)


def detect_elliott_waves(  # noqa: C901 - orchestration kept explicit for scan auditability
    df: pd.DataFrame, config: Optional[ElliottWaveConfig] = None
) -> List[ElliottWaveResult]:
    if config is None:
        config = ElliottWaveConfig()
    _validate_config(config)

    if not isinstance(df, pd.DataFrame) or "close" not in df.columns:
        return []

    if "time" in df.columns:
        t = _coerce_pattern_time_epoch(df["time"], int(df["close"].shape[0]))
        if t.size == 0 or not np.any(np.isfinite(t)):
            t = np.asarray([], dtype=float)
    else:
        t = np.asarray([], dtype=float)
    c = to_float_np(df["close"])
    h = to_float_np(df["high"]) if "high" in df.columns else np.asarray([], dtype=float)
    l = to_float_np(df["low"]) if "low" in df.columns else np.asarray([], dtype=float)
    n = int(c.size)
    if n == 0 or not np.any(np.isfinite(c)):
        return []
    pattern_types = _normalize_pattern_types(config, strict=True)
    gap = max(int(config.min_distance), int(config.wave_min_len))
    family_minimums: List[int] = []
    if "impulse" in pattern_types:
        family_minimums.append(max(5 * gap + 1, int(config.min_impulse_bars)))
    if "correction" in pattern_types:
        family_minimums.append(max(3 * gap + 1, int(config.min_correction_bars)))
    if not family_minimums or n < min(family_minimums):
        return []
    if str(config.pivot_price_source).lower() == "ohlc":
        if h.size != n or l.size != n:
            raise ValueError("pivot_price_source='ohlc' requires matching high and low columns")
        valid_ohlc = np.isfinite(h) & np.isfinite(l) & (h >= l)
        if not np.all(valid_ohlc):
            raise ValueError("OHLC pivot geometry requires finite high >= low on every bar")

    explicit_scan = isinstance(config.scan_thresholds_pct, list) and bool(config.scan_thresholds_pct)
    explicit_scale = bool(explicit_scan or config.swing_threshold_pct is not None)
    adaptive_scan_pairs: Optional[List[Tuple[float, int]]] = None
    pivot_signal = c
    if str(config.scale_mode).strip().lower() == "auto" and not explicit_scale:
        pivot_builder = _adaptive_close_pivots
        if str(config.pivot_price_source).strip().lower() == "ohlc":
            pivot_builder = partial(
                _adaptive_ohlc_pivots,
                high=h,
                low=l,
            )
        adaptation = resolve_elliott_adaptation(
            c,
            h,
            l,
            pivot_builder=pivot_builder,
            scale_mode=config.scale_mode,
            adaptive_denoise=config.adaptive_denoise,
            adaptive_window_bars=int(config.adaptive_window_bars),
            adaptive_min_improvement=float(config.adaptive_min_improvement),
            min_distance=int(config.min_distance),
            pivot_price_source=str(config.pivot_price_source),
            external_denoise_applied=bool(
                getattr(config, "_external_denoise_applied", False)
                or bool(df.attrs.get("pattern_denoise_applied"))
            ),
            fallback_threshold_pct=float(config.min_prominence_pct),
        )
        pivot_signal = adaptation.pivot_signal
        adaptive_scan_pairs = list(adaptation.scan_pairs)
        adaptation_diagnostics = dict(adaptation.diagnostics)
    else:
        threshold = float(
            config.swing_threshold_pct
            if config.swing_threshold_pct is not None
            else config.min_prominence_pct
        )
        diagnostic_thresholds = [threshold]
        diagnostic_distances = [int(config.min_distance)]
        if explicit_scan:
            diagnostic_thresholds = [
                float(value) for value in config.scan_thresholds_pct or []
            ]
            diagnostic_distances = (
                [int(value) for value in config.scan_min_distances]
                if isinstance(config.scan_min_distances, list)
                and config.scan_min_distances
                else sorted(
                    {
                        max(1, int(config.min_distance) - 2),
                        int(config.min_distance),
                        int(config.min_distance) + 2,
                        int(config.min_distance) + 4,
                    }
                )
            )
        adaptation_diagnostics = {
            "adaptive": False,
            "mode": str(config.scale_mode),
            "causality": "causal",
            "fallback_reason": (
                "explicit_scale_precedence" if explicit_scale else "fixed_mode"
            ),
            "selected_filter": {
                "method": (
                    "external_explicit"
                    if bool(
                        getattr(config, "_external_denoise_applied", False)
                        or bool(df.attrs.get("pattern_denoise_applied"))
                    )
                    else "none"
                ),
                "params": {},
            },
            "scan_pairs": [
                {
                    "threshold_pct": float(scan_threshold),
                    "min_distance": int(scan_distance),
                }
                for scan_threshold in diagnostic_thresholds
                for scan_distance in diagnostic_distances
            ],
        }
    df.attrs["elliott_adaptation"] = adaptation_diagnostics

    analyzer = ElliottWaveAnalyzer(
        c,
        t,
        config,
        high=h,
        low=l,
        pivot_signal=pivot_signal,
    )

    results_by_key: Dict[Tuple[str, Tuple[int, ...]], ElliottWaveResult] = {}
    fallback_result: Optional[ElliottWaveResult] = None

    use_multi_scan = explicit_scan
    if adaptive_scan_pairs is not None:
        for threshold, distance in adaptive_scan_pairs:
            scenarios = analyzer.analyze_once(float(threshold), int(distance))
            for scenario in scenarios:
                _upsert_elliott_result(
                    results_by_key, analyzer.build_result(scenario)
                )
    elif use_multi_scan:
        thr_list = config.scan_thresholds_pct or []
        md_base = int(max(1, config.min_distance))
        md_list = (
            config.scan_min_distances
            if isinstance(config.scan_min_distances, list)
            and len(config.scan_min_distances) > 0
            else sorted({max(1, md_base - 2), md_base, md_base + 2, md_base + 4})
        )

        seen_pivot_signatures: set[Tuple[int, ...]] = set()
        prior_scenario_signature: Tuple[Tuple[str, bool, int, int, int], ...] = tuple()
        repeated_scenario_runs = 0
        stop_after_repeats = max(0, int(config.scan_early_stop_repeats))
        scenario_overlap_ratio = float(
            max(0.0, min(1.0, config.scan_scenario_overlap_ratio))
        )
        stop_scan = False
        for thr_val in thr_list:
            try:
                thr_f = float(thr_val)
            except Exception:
                continue
            for md in md_list:
                try:
                    md_i = int(md)
                except Exception:
                    continue
                if bool(config.scan_skip_repeated_pivots):
                    signature = analyzer.pivot_signature(thr_f, md_i)
                    if len(signature) > 1 and signature in seen_pivot_signatures:
                        continue
                    if len(signature) > 1:
                        seen_pivot_signatures.add(signature)

                scenarios = analyzer.analyze_once(thr_f, md_i)
                if stop_after_repeats > 0:
                    current_signature = _scenario_signature(scenarios)
                    if current_signature and _scenario_signatures_overlap(
                        current_signature,
                        prior_scenario_signature,
                        overlap_ratio=scenario_overlap_ratio,
                    ):
                        repeated_scenario_runs += 1
                    else:
                        repeated_scenario_runs = 0
                    prior_scenario_signature = current_signature
                for scenario in scenarios:
                    result = analyzer.build_result(scenario)
                    _upsert_elliott_result(results_by_key, result)
                if (
                    stop_after_repeats > 0
                    and repeated_scenario_runs >= stop_after_repeats
                ):
                    stop_scan = True
                    break
            if stop_scan:
                break
    else:
        thr = float(
            config.swing_threshold_pct
            if config.swing_threshold_pct is not None
            else config.min_prominence_pct
        )
        scenarios = analyzer.analyze_once(thr, int(max(1, config.min_distance)))
        for scenario in scenarios:
            _upsert_elliott_result(results_by_key, analyzer.build_result(scenario))

    configured_recent = getattr(config, "recent_bars", None)
    recent_bars = int(
        max(
            1,
            int(configured_recent)
            if configured_recent is not None
            else max(3, min(20, round(n * 0.05))),
        )
    )
    results = _filter_overlapping_corrections(list(results_by_key.values()))
    # Apply proximity-based deduplication (Option C) to remove near-duplicate patterns
    results = _dedupe_similar_waves(results, proximity_bars=24)
    results = _filter_nested_results(results)
    results = _group_scale_alternatives(results)
    results.sort(
        key=lambda result: _causal_result_sort_key(
            result, n_bars=n, recent_bars=recent_bars
        )
    )
    has_recent = any(
        int(
            (r.details or {}).get("available_at_index")
            if (r.details or {}).get("available_at_index") is not None
            else r.available_at_index
            if r.available_at_index is not None
            else r.end_index
        )
        >= int(n - recent_bars)
        for r in results
    )
    if not has_recent:
        thr_base = float(
            adaptation_diagnostics.get("base_threshold_pct")
            if adaptation_diagnostics.get("base_threshold_pct") is not None
            else config.swing_threshold_pct
            if config.swing_threshold_pct is not None
            else config.min_prominence_pct
        )
        fallback = analyzer.build_fallback(thr_base, int(max(1, config.min_distance)))
        if fallback is not None:
            fallback_result = fallback
            _upsert_elliott_result(results_by_key, fallback)
            results = _filter_overlapping_corrections(
                list(results_by_key.values())
            )
            # Re-apply deduplication after adding fallback
            results = _dedupe_similar_waves(results, proximity_bars=24)
            results = _group_scale_alternatives(results)

    results.sort(
        key=lambda result: _causal_result_sort_key(
            result, n_bars=n, recent_bars=recent_bars
        )
    )
    k = int(getattr(config, "top_k", 10))
    if k > 0:
        results = results[:k]
        if fallback_result is not None and all(
            _elliott_result_key(result) != _elliott_result_key(fallback_result)
            for result in results
        ):
            if results:
                results[-1] = fallback_result
            else:
                results = [fallback_result]
            results.sort(
                key=lambda result: _causal_result_sort_key(
                    result, n_bars=n, recent_bars=recent_bars
                )
            )
    adaptation_summary = {
        key: value
        for key, value in adaptation_diagnostics.items()
        if key != "candidate_metrics"
    }
    selected_filter = adaptation_summary.get("selected_filter")
    for result in results:
        details = result.details if isinstance(result.details, dict) else {}
        details["adaptation"] = adaptation_summary
        if isinstance(selected_filter, dict):
            details["pivot_filter"] = dict(selected_filter)
        result.details = details
    return results
