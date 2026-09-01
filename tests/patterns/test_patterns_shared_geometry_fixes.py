"""Regression tests for shared pivot geometry and ensemble merge fixes.

Covers per-bar high/low repair (previously a single bad bar discarded the whole
array and mis-sized the ATR-adaptive thresholds for both peaks and troughs),
disclosure of that degradation, and order-independent ensemble merging.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtdata.core.patterns_support import _merge_classic_ensemble
from mtdata.patterns.classic_impl.config import ClassicDetectorConfig
from mtdata.patterns.common import (
    compute_pivot_thresholds,
    data_quality_warnings,
    detect_pivots,
    interval_containment_ratio,
    interval_overlap_ratio,
    repair_ohlc_extremes,
)


def _wicky_series(n: int = 200):
    """Close series whose wicks carry structure beyond close-to-close noise."""
    rng = np.random.RandomState(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + np.abs(rng.normal(0, 1.5, n))
    low = close - np.abs(rng.normal(0, 1.5, n))
    return close, high, low


class TestRepairOhlcExtremes:
    def test_clean_input_is_untouched(self):
        close, high, low = _wicky_series()
        hi, lo, repaired_high, repaired_low = repair_ohlc_extremes(close, high, low)
        assert repaired_high == 0
        assert repaired_low == 0
        assert np.array_equal(hi, high)
        assert np.array_equal(lo, low)

    def test_only_the_bad_bar_is_replaced(self):
        close, high, low = _wicky_series(50)
        bad = high.copy()
        bad[7] = np.nan
        hi, _, repaired_high, repaired_low = repair_ohlc_extremes(close, bad, low)
        assert repaired_high == 1
        assert repaired_low == 0
        assert hi[7] == pytest.approx(close[7])
        untouched = [i for i in range(50) if i != 7]
        assert np.array_equal(hi[untouched], high[untouched])

    def test_geometry_violation_is_repaired(self):
        close, high, low = _wicky_series(30)
        bad_high = high.copy()
        bad_high[3] = close[3] - 1.0  # high below the close is impossible
        bad_low = low.copy()
        bad_low[4] = close[4] + 1.0  # low above the close is impossible
        hi, lo, repaired_high, repaired_low = repair_ohlc_extremes(
            close, bad_high, bad_low
        )
        assert repaired_high == 1
        assert repaired_low == 1
        assert hi[3] == pytest.approx(close[3])
        assert lo[4] == pytest.approx(close[4])

    def test_missing_arrays_fall_back_to_close(self):
        close, _, _ = _wicky_series(30)
        hi, lo, repaired_high, repaired_low = repair_ohlc_extremes(close, None, None)
        assert np.array_equal(hi, close)
        assert np.array_equal(lo, close)
        assert repaired_high == 0
        assert repaired_low == 0

    def test_length_mismatch_falls_back_to_close(self):
        close, high, low = _wicky_series(30)
        hi, _, repaired_high, _ = repair_ohlc_extremes(close, high[:10], low)
        assert np.array_equal(hi, close)
        assert repaired_high == 30

    def test_non_finite_close_is_never_substituted_in(self):
        close, high, low = _wicky_series(20)
        close = close.copy()
        close[5] = np.nan
        hi, lo, _, _ = repair_ohlc_extremes(close, high, low)
        # The close cannot repair anything at index 5, so the real high survives.
        assert hi[5] == pytest.approx(high[5])
        assert lo[5] == pytest.approx(low[5])


class TestPivotDegradationIsLocalised:
    def test_single_nan_no_longer_collapses_peak_detection(self):
        close, high, low = _wicky_series()
        cfg = ClassicDetectorConfig()
        peaks, troughs = detect_pivots(close, cfg, high=high, low=low)

        nan_high = high.copy()
        nan_high[0] = np.nan
        peaks_nan, troughs_nan = detect_pivots(close, cfg, high=nan_high, low=low)

        # One repaired bar at the very start must not restructure the series.
        assert abs(len(peaks_nan) - len(peaks)) <= 1
        # The low side had valid data throughout, so it must be unaffected.
        assert np.array_equal(troughs_nan, troughs)

    def test_atr_threshold_is_not_computed_from_a_hybrid_bar(self):
        close, high, low = _wicky_series()
        cfg = ClassicDetectorConfig()
        assert cfg.pivot_use_atr_adaptive_prominence is True

        clean_prom, clean_dist = compute_pivot_thresholds(close, high, low, cfg)
        nan_high = high.copy()
        nan_high[0] = np.nan
        hi, lo, _, _ = repair_ohlc_extremes(close, nan_high, low)
        repaired_prom, repaired_dist = compute_pivot_thresholds(close, hi, lo, cfg)

        # Previously `high` collapsed to `close` while `low` stayed real, giving a
        # true range of close-low and an ATR ~0.62x the real one.
        assert repaired_prom == pytest.approx(clean_prom, rel=0.05)
        assert repaired_dist == clean_dist

    def test_total_loss_still_degrades_to_close_only(self):
        close, _, _ = _wicky_series()
        cfg = ClassicDetectorConfig()
        expected = detect_pivots(close, cfg, high=None, low=None)
        actual = detect_pivots(
            close,
            cfg,
            high=np.full(close.size, np.nan),
            low=np.full(close.size, np.nan),
        )
        assert np.array_equal(actual[0], expected[0])
        assert np.array_equal(actual[1], expected[1])


class TestOhlcDegradationIsDisclosed:
    def test_missing_high_low_warns(self):
        close, _, _ = _wicky_series(60)
        df = pd.DataFrame({"time": np.arange(60, dtype=float), "close": close})
        warnings = data_quality_warnings(df)
        assert any("close-only" in warning for warning in warnings)

    def test_unusable_bars_are_counted(self):
        close, high, low = _wicky_series(60)
        bad_high = high.copy()
        bad_high[5] = np.nan
        bad_high[9] = close[9] - 1.0
        df = pd.DataFrame({
            "time": np.arange(60, dtype=float),
            "close": close,
            "high": bad_high,
            "low": low,
        })
        warnings = data_quality_warnings(df)
        geometry = [w for w in warnings if "candle geometry" in w]
        assert len(geometry) == 1
        assert "2 high/low value(s)" in geometry[0]

    def test_clean_ohlc_is_silent(self):
        close, high, low = _wicky_series(60)
        df = pd.DataFrame({
            "time": np.arange(60, dtype=float),
            "close": close,
            "high": high,
            "low": low,
        })
        warnings = data_quality_warnings(df)
        assert not any(
            "candle geometry" in w or "close-only" in w for w in warnings
        )


class TestIntervalContainment:
    def test_nested_interval_scores_full_containment(self):
        assert interval_containment_ratio(100, 140, 60, 200) == pytest.approx(1.0)
        # Intersection-over-union rates the same pair very low.
        assert interval_overlap_ratio(100, 140, 60, 200) < 0.3

    def test_disjoint_intervals_score_zero(self):
        assert interval_containment_ratio(0, 10, 50, 60) == 0.0

    def test_identical_intervals_score_one(self):
        assert interval_containment_ratio(10, 20, 10, 20) == pytest.approx(1.0)

    def test_partial_overlap_is_relative_to_shorter_interval(self):
        # Intersection is 10..20 (11 bars); shorter interval is 0..20 (21 bars).
        assert interval_containment_ratio(0, 20, 10, 40) == pytest.approx(11 / 21)


class TestEnsembleMergeStability:
    WEIGHTS = {"native": 1.0, "stock_pattern": 1.0, "third": 1.0}

    @staticmethod
    def _pattern(name, start, end, confidence=0.8):
        return {
            "name": name,
            "start_index": start,
            "end_index": end,
            "confidence": confidence,
        }

    def test_merge_is_independent_of_engine_order(self):
        a = self._pattern("Double Top", 100, 140, 0.8)
        b = self._pattern("Double Top", 120, 175, 0.7)
        c = self._pattern("Double Top", 150, 190, 0.6)
        forward = _merge_classic_ensemble(
            {"native": [a], "stock_pattern": [b], "third": [c]}, self.WEIGHTS
        )
        reverse = _merge_classic_ensemble(
            {"third": [c], "stock_pattern": [b], "native": [a]}, self.WEIGHTS
        )
        assert len(forward) == len(reverse)
        assert sorted(p["support_count"] for p in forward) == sorted(
            p["support_count"] for p in reverse
        )

    def test_nested_same_name_detections_collapse(self):
        tight = self._pattern("Head and Shoulders", 100, 140, 0.8)
        wide = self._pattern("Head and Shoulders", 60, 200, 0.7)
        merged = _merge_classic_ensemble(
            {"native": [tight], "stock_pattern": [wide]}, self.WEIGHTS
        )
        assert len(merged) == 1
        assert merged[0]["support_count"] == 2
        assert sorted(merged[0]["source_engines"]) == ["native", "stock_pattern"]

    def test_sequential_same_name_detections_stay_separate(self):
        first = self._pattern("Double Top", 0, 40)
        second = self._pattern("Double Top", 120, 160)
        merged = _merge_classic_ensemble({"native": [first, second]}, self.WEIGHTS)
        assert len(merged) == 2

    def test_different_names_never_merge(self):
        merged = _merge_classic_ensemble(
            {
                "native": [self._pattern("Double Top", 100, 140)],
                "stock_pattern": [self._pattern("Double Bottom", 100, 140)],
            },
            self.WEIGHTS,
        )
        assert len(merged) == 2
