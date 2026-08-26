"""Tests for patterns/classic_impl/utils.py — pure math helpers (no MT5)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import mtdata.patterns.classic_impl.utils as utils_mod
import mtdata.patterns.common as patterns_common
from mtdata.patterns.classic_impl.config import ClassicDetectorConfig
from mtdata.patterns.classic_impl.utils import (
    _calibrate_confidence,
    _collect_calibration_points,
    _conf,
    _count_touches,
    _detect_pivots_close,
    _find_recent_breakout,
    _fit_line,
    _fit_line_robust,
    _is_converging,
    _level_close,
    _paa,
    _template_hs_variants,
    _tol_abs_from_close,
    _znorm,
)
from mtdata.patterns.common import (
    compute_atr_sma,
    compute_pivot_thresholds,
    fallback_local_extrema,
)
from mtdata.utils.dtw import dtw_distance


class TestLevelClose:
    def test_same_values(self):
        assert _level_close(100.0, 100.0, 0.5) is True

    def test_within_tol(self):
        assert _level_close(100.0, 100.3, 0.5) is True

    def test_outside_tol(self):
        assert _level_close(100.0, 101.0, 0.5) is False

    def test_zero_value(self):
        assert _level_close(0.0, 0.0, 0.5) is True
        assert _level_close(0.0, 1.0, 0.5) is False


class TestFitLine:
    def test_perfect_line(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = 2.0 * x + 1.0
        slope, intercept, r2 = _fit_line(x, y)
        assert abs(slope - 2.0) < 1e-6
        assert abs(intercept - 1.0) < 1e-6
        assert abs(r2 - 1.0) < 1e-6

    def test_single_point(self):
        x = np.array([5.0])
        y = np.array([10.0])
        slope, intercept, r2 = _fit_line(x, y)
        assert slope == 0.0
        assert intercept == 10.0

    def test_noisy(self):
        rng = np.random.RandomState(42)
        x = np.arange(50, dtype=float)
        y = 0.5 * x + 10 + rng.normal(0, 0.5, 50)
        slope, _, r2 = _fit_line(x, y)
        assert abs(slope - 0.5) < 0.1
        assert r2 > 0.8

    def test_perfect_horizontal_line_has_unit_r2(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([5.0, 5.0, 5.0, 5.0])
        slope, intercept, r2 = _fit_line(x, y)
        assert abs(slope) < 1e-12
        assert abs(intercept - 5.0) < 1e-6
        assert abs(r2 - 1.0) < 1e-6


class TestFitLineRobust:
    def test_ordinary_fallback(self):
        cfg = ClassicDetectorConfig(use_robust_fit=False)
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = 3.0 * x + 5.0
        slope, intercept, r2 = _fit_line_robust(x, y, cfg)
        assert abs(slope - 3.0) < 1e-6
        assert abs(r2 - 1.0) < 1e-6

    def test_ransac(self):
        cfg = ClassicDetectorConfig(use_robust_fit=True)
        x = np.arange(20, dtype=float)
        y = 2.0 * x + 1.0
        y[5] = 999.0  # outlier
        slope, intercept, r2 = _fit_line_robust(x, y, cfg)
        assert abs(slope - 2.0) < 0.5
        assert r2 > 0.99

    def test_ransac_uses_consensus_for_moderately_noisy_small_sample(self):
        cfg = ClassicDetectorConfig(use_robust_fit=True)
        x = np.arange(4, dtype=float)
        y = np.array([10.0, 11.0, 10.5, 12.0], dtype=float)

        robust = _fit_line_robust(x, y, cfg)
        ordinary = _fit_line(x, y)

        assert robust == pytest.approx(ordinary)

    def test_ransac_preserves_unit_r2_for_perfect_horizontal_line(self):
        cfg = ClassicDetectorConfig(use_robust_fit=True)
        x = np.arange(6, dtype=float)
        y = np.full(6, 7.5, dtype=float)
        slope, intercept, r2 = _fit_line_robust(x, y, cfg)
        assert abs(slope) < 1e-12
        assert abs(intercept - 7.5) < 1e-6
        assert abs(r2 - 1.0) < 1e-6

    def test_residual_threshold_is_shift_invariant(self, monkeypatch):
        thresholds = []

        class _FakeRansac:
            def __init__(self, *, residual_threshold, **kwargs):
                _ = kwargs
                thresholds.append(float(residual_threshold))
                self.estimator_ = SimpleNamespace(coef_=np.array([1.0]), intercept_=0.0)

            def fit(self, X, y):
                _ = X
                _ = y
                return self

        monkeypatch.setattr(utils_mod, "_get_ransac_regressor_cls", lambda: _FakeRansac)

        cfg = ClassicDetectorConfig(use_robust_fit=True, ransac_residual_scale=2.5)
        x = np.arange(6, dtype=float)
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 50.0], dtype=float)

        _fit_line_robust(x, y, cfg)
        _fit_line_robust(x, y + 1000.0, cfg)

        assert len(thresholds) == 2
        assert thresholds[0] == pytest.approx(thresholds[1])


class TestZnorm:
    def test_standard(self):
        a = np.array([10.0, 20.0, 30.0])
        z = _znorm(a)
        assert abs(z.mean()) < 1e-6
        assert abs(z.std() - 1.0) < 1e-6

    def test_empty(self):
        z = _znorm(np.array([]))
        assert z.size == 0

    def test_constant(self):
        z = _znorm(np.array([5.0, 5.0, 5.0]))
        assert np.allclose(z, 0.0)


class TestPaa:
    def test_identity(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        result = _paa(a, 4)
        assert np.allclose(result, a)

    def test_downsample(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        result = _paa(a, 2)
        assert len(result) == 2
        assert abs(result[0] - 1.5) < 0.5
        assert abs(result[1] - 3.5) < 0.5

    def test_empty(self):
        result = _paa(np.array([]), 5)
        assert result.size == 0


class TestDtwDistance:
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        d = dtw_distance(a, a)
        assert d == 0.0

    def test_different(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 1.0, 1.0])
        d = dtw_distance(a, b)
        assert d > 0.0

class TestTemplateHsVariants:
    def test_length(self):
        variants = _template_hs_variants(80)
        assert len(variants) >= 1
        for t in variants:
            assert len(t) == 80

    def test_min_length(self):
        variants = _template_hs_variants(5)
        for t in variants:
            assert len(t) == 20  # min is 20

    def test_inverse(self):
        t = _template_hs_variants(80, inverse=False)[0]
        ti = _template_hs_variants(80, inverse=True)[0]
        # Inverted should be negated (after z-norm, they should be opposite)
        assert np.corrcoef(t, ti)[0, 1] < -0.9

    def test_znormed(self):
        for t in _template_hs_variants(80):
            assert abs(t.mean()) < 1e-6


class TestComputeAtr:
    def test_basic(self):
        n = 50
        h = np.full(n, 102.0)
        l = np.full(n, 98.0)
        c = np.full(n, 100.0)
        atr = compute_atr_sma(h, l, c, 14)
        assert len(atr) == n
        # ATR should be ~4.0 for constant range
        finite = atr[np.isfinite(atr)]
        assert abs(finite[-1] - 4.0) < 0.5

    def test_empty(self):
        atr = compute_atr_sma(np.array([]), np.array([]), np.array([]), 14)
        assert atr.size == 0


class TestPivotThresholds:
    def test_returns_positive(self):
        c = np.linspace(100, 110, 200)
        h = c + 0.5
        l = c - 0.5
        cfg = ClassicDetectorConfig()
        prom, dist = compute_pivot_thresholds(c, h, l, cfg)
        assert prom > 0
        assert dist >= 2


class TestDetectPivotsClose:
    def test_sinewave_finds_pivots(self):
        x = np.sin(np.linspace(0, 8 * np.pi, 400)) * 10 + 100
        cfg = ClassicDetectorConfig()
        peaks, troughs = _detect_pivots_close(x, cfg)
        assert len(peaks) > 0
        assert len(troughs) > 0

    def test_short_window_can_still_find_pivots(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 24)) * 3 + 100
        cfg = ClassicDetectorConfig(min_distance=3)

        peaks, troughs = _detect_pivots_close(x, cfg)

        assert peaks.size > 0
        assert troughs.size > 0

    def test_too_short(self):
        x = np.array([100.0, 101.0, 99.0])
        cfg = ClassicDetectorConfig()
        peaks, troughs = _detect_pivots_close(x, cfg)
        assert peaks.size == 0
        assert troughs.size == 0

    def test_unexpected_find_peaks_error_is_not_flattened(self, monkeypatch):
        x = np.linspace(100.0, 110.0, 200)
        cfg = ClassicDetectorConfig(
            pivot_use_atr_adaptive_prominence=False,
            pivot_use_atr_adaptive_distance=False,
        )

        monkeypatch.setattr(
            patterns_common,
            "find_peaks",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("boom")),
        )

        with pytest.raises(TypeError, match="boom"):
            _detect_pivots_close(x, cfg)

    def test_fallback_local_extrema_recovers_when_find_peaks_undershoots(self, monkeypatch):
        x = np.array([100.0, 102.0, 99.0, 103.0, 98.0, 104.0, 97.0] * 40, dtype=float)
        cfg = ClassicDetectorConfig(
            pivot_use_atr_adaptive_prominence=False,
            pivot_use_atr_adaptive_distance=False,
            min_prominence_pct=50.0,
            min_distance=4,
            pivot_enable_fallback=True,
            pivot_fallback_order=2,
        )

        monkeypatch.setattr(
            patterns_common,
            "find_peaks",
            lambda *_args, **_kwargs: (np.array([], dtype=int), {}),
        )

        peaks, troughs = _detect_pivots_close(x, cfg)

        assert peaks.size >= 2
        assert troughs.size >= 2

    def test_fallback_local_extrema_keeps_plateau_midpoint(self):
        peaks = fallback_local_extrema(
            np.array([1.0, 2.0, 2.0, 2.0, 1.0], dtype=float),
            min_dist=1,
            order=1,
            prefer_high=True,
        )

        assert peaks.tolist() == [2]

    def test_fallback_local_extrema_allows_exact_window_boundary(self):
        peaks = fallback_local_extrema(
            np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float),
            min_dist=1,
            order=2,
            prefer_high=True,
        )

        assert peaks.tolist() == [2]


class TestConfidenceHelpers:
    def test_conf_normalizes_non_unit_weights(self):
        cfg = ClassicDetectorConfig(
            touch_weight=2.0,
            r2_weight=1.0,
            geometry_weight=1.0,
            min_touches=2,
        )

        # min_touches=2 → touch_score saturates at 2*min_touches; 2 touches → 0.5
        conf = _conf(2, 1.0, 1.0, cfg)
        # weights 2/1/1 → [0.5, 0.25, 0.25]; scores [0.5, 1, 1] → 0.5*0.5+0.25+0.25=0.75
        assert conf == pytest.approx(0.75)

        conf_full = _conf(4, 1.0, 1.0, cfg)
        assert conf_full == pytest.approx(1.0)


class TestIsConverging:
    def test_converging(self):
        n = 100
        upper = np.linspace(110, 105, n)  # narrowing
        lower = np.linspace(90, 95, n)
        cfg = ClassicDetectorConfig()
        assert _is_converging(upper, lower, 10, n, cfg) is True

    def test_diverging(self):
        n = 100
        upper = np.linspace(105, 110, n)  # widening
        lower = np.linspace(95, 90, n)
        cfg = ClassicDetectorConfig()
        assert _is_converging(upper, lower, 10, n, cfg) is False


class TestCountTouches:
    def test_close_touches(self):
        n = 50
        x = np.arange(n, dtype=float)
        c = np.full(n, 100.0)
        upper = np.full(n, 102.0)
        lower = np.full(n, 98.0)
        # Peaks where close touches upper
        c[10] = 102.0
        c[20] = 102.0
        c[30] = 98.0  # trough
        peaks = np.array([10, 20])
        troughs = np.array([30])
        count = _count_touches(upper, lower, peaks, troughs, c, 0.5)
        assert count >= 2


class TestFindRecentBreakout:
    def test_returns_most_recent_breakout(self):
        close = np.array([100.0, 101.0, 99.0, 102.0])
        upper = np.array([100.5, 100.5, 100.5, 100.5])
        direction, idx = _find_recent_breakout(close, upper=upper, lookback_bars=4)
        assert direction == "up"
        assert idx == 3

    def test_ignores_ambiguous_bar_after_boundaries_cross(self):
        close = np.array([100.0, 102.0, 101.0])
        upper = np.array([101.5, 101.5, 100.0])
        lower = np.array([98.5, 98.5, 102.0])

        direction, idx = _find_recent_breakout(
            close,
            upper=upper,
            lower=lower,
            lookback_bars=3,
        )

        assert direction == "up"
        assert idx == 1


class TestCalibrateConfidence:
    def test_no_calibration(self):
        cfg = ClassicDetectorConfig(calibrate_confidence=False)
        assert _calibrate_confidence(0.7, "triangle", cfg) == 0.7

    def test_with_calibration_map(self):
        cfg = ClassicDetectorConfig(
            calibrate_confidence=True,
            confidence_calibration_map={
                "default": {"0.40": 0.35, "0.70": 0.62, "0.90": 0.82}
            },
            confidence_calibration_blend=1.0,
        )
        calibrated = _calibrate_confidence(0.7, "triangle", cfg)
        assert isinstance(calibrated, float)
        assert 0.0 <= calibrated <= 1.0


class TestCollectCalibrationPoints:
    def test_empty_map(self):
        result = _collect_calibration_points({}, "triangle")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_default_only(self):
        m = {"default": {"0.5": 0.4, "0.8": 0.7}}
        result = _collect_calibration_points(m, "nonexistent")
        assert len(result) == 2
