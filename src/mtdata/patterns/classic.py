from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.time import bar_close_epoch
from .classic_impl.config import (
    ClassicDetectorConfig,
    ClassicPatternResult,
    validate_classic_detector_config,
)
from .classic_impl.continuation import detect_cup_handle, detect_flags_pennants
from .classic_impl.reversal import (
    detect_head_shoulders,
    detect_rounding,
    detect_tops_bottoms,
)
from .classic_impl.shapes import (
    detect_broadening,
    detect_diamonds,
    detect_rectangles,
    detect_triangles,
    detect_wedges,
)
from .classic_impl.trend import detect_channels, detect_trend_lines
from .classic_impl.utils import (
    _calibrate_confidence,
    _detect_pivots_close,
)
from .common import interval_overlap_ratio, prepare_ohlc_pattern_inputs

__all__ = [
    "ClassicDetectorConfig",
    "ClassicPatternResult",
    "detect_classic_patterns",
]


def _prepare_classic_inputs(
    df: pd.DataFrame,
    cfg: ClassicDetectorConfig,
) -> Optional[tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    return prepare_ohlc_pattern_inputs(
        df,
        max_bars=int(cfg.max_bars),
        min_input_bars=max(20, int(getattr(cfg, "min_input_bars", 100))),
        log_label="Classic pattern detection",
        log_extra="; pivot geometry will be less reliable than true OHLC.",
        time_mode="empty",
    )


def _detect_classic_patterns_once(
    t: np.ndarray,
    c: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    n: int,
    cfg: ClassicDetectorConfig,
    *,
    peaks: Optional[np.ndarray] = None,
    troughs: Optional[np.ndarray] = None,
) -> List[ClassicPatternResult]:
    if peaks is None or troughs is None:
        peaks, troughs = _detect_pivots_close(c, cfg, h, l)
    else:
        peaks = np.asarray(peaks, dtype=int)
        troughs = np.asarray(troughs, dtype=int)

    # scipy peak detection can identify a recent local extremum before enough
    # right-hand bars exist to treat it as a stable trading pivot. Apply the
    # same conservative confirmation gap used by the historical prefix scan.
    pivot_confirm_gap = max(2, int(getattr(cfg, "min_distance", 5)))
    pivot_cutoff = max(0, int(n) - pivot_confirm_gap)
    peaks = peaks[peaks < pivot_cutoff]
    troughs = troughs[troughs < pivot_cutoff]

    results: List[ClassicPatternResult] = []
    results.extend(detect_trend_lines(c, peaks, troughs, t, cfg, high=h, low=l))
    results.extend(detect_channels(c, peaks, troughs, t, cfg, high=h, low=l))
    results.extend(detect_rectangles(c, peaks, troughs, t, cfg, high=h, low=l))
    results.extend(detect_triangles(c, peaks, troughs, t, cfg, high=h, low=l))
    results.extend(detect_wedges(c, peaks, troughs, t, cfg, high=h, low=l))
    results.extend(detect_broadening(c, peaks, troughs, t, cfg, high=h, low=l))
    results.extend(detect_diamonds(c, t, cfg, h, l, peaks=peaks, troughs=troughs))
    results.extend(
        detect_tops_bottoms(c, peaks, troughs, t, cfg, high=h, low=l)
    )
    results.extend(
        detect_head_shoulders(c, peaks, troughs, t, cfg, high=h, low=l)
    )
    results.extend(detect_rounding(c, t, cfg))
    results.extend(
        detect_flags_pennants(c, h, l, t, n, cfg, peaks=peaks, troughs=troughs)
    )
    results.extend(detect_cup_handle(c, t, cfg))
    return results


def _attach_classic_availability(
    results: List[ClassicPatternResult],
    t: np.ndarray,
    *,
    available_at_index: int,
    pivot_confirmation_bars: int,
    detection_scope: str,
    timeframe: Optional[str] = None,
) -> None:
    for result in results:
        if not isinstance(result.details, dict):
            result.details = {}
        result.details.setdefault("available_at_index", int(available_at_index))
        bar_open = result.resolve_time(t, int(available_at_index))
        if bar_open is not None:
            result.details["detection_bar_open"] = bar_open
        if bar_open is not None and timeframe:
            result.details["available_at_time"] = bar_close_epoch(bar_open, timeframe)
            result.details["available_at_time_basis"] = "completed_bar_close"
        else:
            result.details.pop("available_at_time", None)
            result.details["available_at_time_basis"] = "unavailable_without_timeframe_or_timestamp"
        result.details.setdefault("pivot_confirmation_bars", int(pivot_confirmation_bars))
        result.details.setdefault("status_basis", "causal_as_of_detection_with_confirmed_pivots")
        result.details.setdefault("detection_scope", detection_scope)


def _pattern_overlap_ratio(a: ClassicPatternResult, b: ClassicPatternResult) -> float:
    return interval_overlap_ratio(
        int(a.start_index),
        int(a.end_index),
        int(b.start_index),
        int(b.end_index),
    )


def _prefer_pattern_candidate(
    current: ClassicPatternResult,
    candidate: ClassicPatternResult,
) -> ClassicPatternResult:
    current_rank = (
        1 if str(current.status).lower() == "completed" else 0,
        int(current.end_index),
        float(current.confidence),
    )
    candidate_rank = (
        1 if str(candidate.status).lower() == "completed" else 0,
        int(candidate.end_index),
        float(candidate.confidence),
    )
    return candidate if candidate_rank > current_rank else current


def _merge_scanned_patterns(
    existing: List[ClassicPatternResult],
    new_results: List[ClassicPatternResult],
    cfg: ClassicDetectorConfig,
) -> List[ClassicPatternResult]:
    overlap_min = float(max(0.0, min(1.0, getattr(cfg, "scan_dedupe_overlap", 0.8))))
    merged = list(existing)
    for candidate in new_results:
        match_i: Optional[int] = None
        for i, prior in enumerate(merged):
            if prior.name != candidate.name:
                continue
            if _pattern_overlap_ratio(prior, candidate) < overlap_min:
                continue
            match_i = i
            break
        if match_i is None:
            merged.append(candidate)
            continue
        merged[match_i] = _prefer_pattern_candidate(merged[match_i], candidate)
    return merged


def _scan_classic_patterns(
    t: np.ndarray,
    c: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    cfg: ClassicDetectorConfig,
    *,
    timeframe: Optional[str] = None,
) -> List[ClassicPatternResult]:
    n_total = int(c.size)
    step = max(1, int(getattr(cfg, "scan_step_bars", 10)))
    min_prefix = max(
        int(getattr(cfg, "min_input_bars", 100)),
        int(getattr(cfg, "scan_min_prefix_bars", 120)),
    )
    prefix_ends = list(range(min_prefix, n_total + 1, step))
    if not prefix_ends or prefix_ends[-1] != n_total:
        prefix_ends.append(n_total)

    scan_cfg = replace(cfg, scan_historical=False)
    pivot_confirm_gap = max(2, int(getattr(scan_cfg, "min_distance", 5)))

    merged: List[ClassicPatternResult] = []
    for end in prefix_ends:
        prefix_peaks, prefix_troughs = _detect_pivots_close(
            c[:end],
            scan_cfg,
            h[:end],
            l[:end],
        )
        prefix_peaks = np.asarray(prefix_peaks, dtype=int)
        prefix_troughs = np.asarray(prefix_troughs, dtype=int)
        pivot_cutoff = max(0, int(end) - pivot_confirm_gap)
        batch = _detect_classic_patterns_once(
            t[:end],
            c[:end],
            h[:end],
            l[:end],
            int(end),
            scan_cfg,
            peaks=prefix_peaks[prefix_peaks < pivot_cutoff],
            troughs=prefix_troughs[prefix_troughs < pivot_cutoff],
        )
        _attach_classic_availability(
            batch,
            t[:end],
            available_at_index=int(end) - 1,
            pivot_confirmation_bars=pivot_confirm_gap,
            detection_scope="causal_prefix_scan",
            timeframe=timeframe,
        )
        merged = _merge_scanned_patterns(merged, batch, cfg)
    return merged


def _postprocess_classic_results(
    results: List[ClassicPatternResult],
    cfg: ClassicDetectorConfig,
    n: int,
) -> List[ClassicPatternResult]:
    # Apply the configured result recency bound to every lifecycle state.
    max_age = int(getattr(cfg, "max_pattern_age_bars", 500))
    if max_age > 0:
        cutoff_index = n - max_age
        results = [r for r in results if r.end_index >= cutoff_index]

    # Apply the configured geometry-span bound to every lifecycle state.
    max_span = int(getattr(cfg, "max_pattern_span_bars", 0))
    if max_span > 0:
        results = [r for r in results if (r.end_index - r.start_index) <= max_span]

    for r in results:
        raw_conf = float(r.confidence)
        cal_conf = _calibrate_confidence(raw_conf, r.name, cfg)
        if not isinstance(r.details, dict):
            r.details = {}
        if bool(getattr(cfg, "calibrate_confidence", False)):
            r.details["raw_confidence"] = float(raw_conf)
            r.details["calibrated_confidence"] = float(cal_conf)
        r.confidence = float(cal_conf)

    # Cap forming pattern confidence — a pattern still forming cannot be 100% certain
    _FORMING_CONFIDENCE_CAP = 0.95
    for i, r in enumerate(results):
        if r.status == "forming" and r.confidence > _FORMING_CONFIDENCE_CAP:
            results[i] = replace(r, confidence=_FORMING_CONFIDENCE_CAP)

    min_conf = float(max(0.0, min(1.0, getattr(cfg, "min_confidence", 0.0))))
    if min_conf > 0.0:
        results = [r for r in results if float(r.confidence) >= min_conf]

    if bool(getattr(cfg, "include_lifecycle_metadata", True)):
        for r in results:
            if not isinstance(r.details, dict):
                r.details = {}
            if r.status == "completed":
                r.details.setdefault("lifecycle_state", "confirmed")
            else:
                r.details.setdefault("lifecycle_state", "forming")

    results.sort(key=lambda r: (r.end_index, r.confidence), reverse=True)
    return results


def detect_classic_patterns(
    df: pd.DataFrame, cfg: Optional[ClassicDetectorConfig] = None,
    *, timeframe: Optional[str] = None,
) -> List[ClassicPatternResult]:
    """Detect patterns on completed OHLCV bars timestamped at their open.

    Supply ``timeframe`` (or ``df.attrs['timeframe']``) to report availability
    at the detection bar's close. Without it only the bar index is available.
    """
    if cfg is None:
        cfg = ClassicDetectorConfig()
    config_warnings = validate_classic_detector_config(cfg)
    if config_warnings:
        import logging

        _log = logging.getLogger(__name__)
        for w in config_warnings:
            _log.warning("ClassicDetectorConfig: %s", w)
    prepared = _prepare_classic_inputs(df, cfg)
    if prepared is None:
        return []

    _, t, c, h, l, n = prepared
    timeframe = timeframe or df.attrs.get("timeframe")
    if bool(getattr(cfg, "scan_historical", False)):
        results = _scan_classic_patterns(t, c, h, l, cfg, timeframe=timeframe)
    else:
        results = _detect_classic_patterns_once(t, c, h, l, n, cfg)
        _attach_classic_availability(
            results,
            t,
            available_at_index=int(n) - 1,
            pivot_confirmation_bars=max(2, int(getattr(cfg, "min_distance", 5))),
            detection_scope="right_edge_as_of_input_window",
            timeframe=timeframe,
        )

    return _postprocess_classic_results(results, cfg, n)
