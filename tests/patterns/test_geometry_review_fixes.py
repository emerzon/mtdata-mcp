"""Regression tests for the harmonic/classic/fractal correctness review.

Each class pins a specific defect the review found, so a future relaxation of the
gate that fixed it fails loudly rather than silently restoring the old output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtdata.patterns.classic_impl.config import ClassicDetectorConfig
from mtdata.patterns.classic_impl.continuation import detect_cup_handle
from mtdata.patterns.classic_impl.reversal import (
    detect_head_shoulders,
    detect_rounding,
    detect_tops_bottoms,
)
from mtdata.patterns.classic_impl.utils import (
    _boundary_apex_index,
    _find_recent_breakout,
    _mask_boundaries_after_apex,
    _neckline_break_lookahead,
)
from mtdata.patterns.fractal import FractalDetectorConfig, detect_fractal_patterns
from mtdata.patterns.harmonic import (
    _XABCD_SPECS,
    HarmonicDetectorConfig,
    detect_harmonic_patterns,
)


def _random_walk_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.6, n))
    return pd.DataFrame(
        {
            "time": np.arange(n, dtype=float),
            "open": close,
            "high": close + np.abs(rng.normal(0, 0.3, n)),
            "low": close - np.abs(rng.normal(0, 0.3, n)),
            "close": close,
        }
    )


class TestHarmonicLegSpacing:
    """find_peaks spaces peaks and troughs separately, never against each other.

    Interleaving the two arrays could therefore build an XABCD from pivots one bar
    apart, where a Fibonacci ratio between single candles is noise.
    """

    def test_no_leg_is_shorter_than_min_distance(self):
        cfg = HarmonicDetectorConfig(
            pivot_use_atr_adaptive_prominence=False,
            pivot_use_atr_adaptive_distance=False,
        )
        legs: list[int] = []
        for seed in range(30):
            for result in detect_harmonic_patterns(
                _random_walk_frame(420, seed), cfg
            ):
                indexes = sorted(result.details["pivot_indexes"].values())
                legs.extend(b - a for a, b in zip(indexes, indexes[1:]))

        assert legs, "probe produced no harmonic patterns to inspect"
        assert min(legs) >= cfg.min_distance

    def test_one_bar_leg_window_is_rejected(self):
        from mtdata.patterns.harmonic import _SwingPoint, _window_legs_are_swings

        window = [
            _SwingPoint(index=index, kind=kind, price=price)
            for index, kind, price in (
                (100, "low", 90.0),
                (120, "high", 110.0),
                (140, "low", 95.0),
                (141, "high", 108.0),
                (160, "low", 92.0),
            )
        ]
        assert _window_legs_are_swings(window, 1) is True
        assert _window_legs_are_swings(window, 5) is False


class TestHarmonicAvailabilityClaim:
    """available_at_index asserted a point-in-time the pattern was not knowable.

    Widening the confirmation gap from 1x to 6x min_distance left a residual of
    irreproducible claims instead of converging, because prominence is a
    whole-window statistic, so the authoritative value is the right edge.
    """

    def test_every_availability_claim_is_reproducible(self):
        cfg = HarmonicDetectorConfig()
        for seed in range(12):
            n = 420
            df = _random_walk_frame(n, seed)
            for result in detect_harmonic_patterns(df, cfg):
                available = int(result.details["available_at_index"])
                assert available == n - 1
                target = sorted(result.details["pivot_indexes"].values())
                reproduced = detect_harmonic_patterns(
                    df.iloc[: available + 1], cfg
                )
                assert any(
                    sorted(item.details["pivot_indexes"].values()) == target
                    for item in reproduced
                )

    def test_estimate_is_reported_separately_and_flagged(self):
        cfg = HarmonicDetectorConfig()
        results = detect_harmonic_patterns(_random_walk_frame(420, 1), cfg)
        assert results
        details = results[0].details
        assert details["available_at_index_basis"] == "input_window_right_edge"
        assert "earliest_possible_index_estimate" in details
        assert "not safe for backtest" in details["earliest_possible_index_caveat"]


class TestHarmonicFiveZeroLabels:
    """The 5-0 reuses the XABCD ratio machinery but its points are named 0,X,A,B,C."""

    def test_five_o_declares_canonical_point_names(self):
        assert _XABCD_SPECS["five_o"].pivot_labels == ("0", "X", "A", "B", "C")

    def test_other_xabcd_patterns_keep_family_labels(self):
        assert _XABCD_SPECS["gartley"].pivot_labels is None


class TestHarmonicDedupeIgnoresLabels:
    """Dedupe keyed on sorted label/value pairs, so it depended on point names."""

    def test_same_pivots_collapse_across_differently_labelled_families(self):
        from mtdata.patterns.harmonic import HarmonicPatternResult, _dedupe_results

        indexes = {10: None, 20: None, 30: None, 40: None, 50: None}

        def _result(name: str, labels: list[str], confidence: float):
            return HarmonicPatternResult(
                name=name,
                status="completed",
                confidence=confidence,
                start_index=10,
                end_index=50,
                start_time=10.0,
                end_time=50.0,
                bias="bullish",
                entry_price=100.0,
                target_prices=[110.0],
                invalidation_price=90.0,
                details={
                    "pivot_indexes": dict(zip(labels, indexes)),
                },
            )

        deduped = _dedupe_results(
            [
                _result("Bullish Gartley", ["X", "A", "B", "C", "D"], 0.7),
                _result("Bullish Shark", ["O", "X", "A", "B", "C"], 0.9),
            ]
        )
        assert len(deduped) == 1
        assert deduped[0].name == "Bullish Shark"


class TestRoundingFitQuality:
    """Nothing required the price path to be rounded; confidence was amplitude only."""

    @staticmethod
    def _cfg() -> ClassicDetectorConfig:
        return ClassicDetectorConfig(rounding_window_sizes=[220])

    @staticmethod
    def _times(n: int) -> np.ndarray:
        return np.arange(n, dtype=float)

    def test_single_spike_on_a_flat_series_is_rejected(self):
        n = 221
        close = np.full(n, 120.0)
        close[n // 2] = 80.0
        close[-1] = 130.0
        assert detect_rounding(close, self._times(n), self._cfg()) == []

    def test_sharp_v_is_rejected(self):
        n = 221
        half = n // 2
        close = np.concatenate(
            [np.linspace(120.0, 80.0, half), np.linspace(80.0, 120.0, n - half)]
        )
        close[-1] = 130.0
        assert detect_rounding(close, self._times(n), self._cfg()) == []

    def test_true_saucer_is_accepted_and_reports_its_fit(self):
        n = 221
        x = np.linspace(-1.0, 1.0, n)
        close = 100.0 + 12.0 * x**2
        close[-1] = close[0] + 4.0
        found = detect_rounding(close, self._times(n), self._cfg())

        assert len(found) == 1
        details = found[0].details
        assert found[0].name == "Rounding Bottom"
        assert details["quad_r2"] > 0.99
        assert details["turn_width_bars"] > 0.35 * 220

    def test_amplitude_alone_no_longer_drives_confidence(self):
        n = 221
        x = np.linspace(-1.0, 1.0, n)
        rng = np.random.RandomState(7)

        clean = 100.0 + 12.0 * x**2
        clean[-1] = clean[0] + 4.0
        noisy = 100.0 + 12.0 * x**2 + rng.normal(0, 0.8, n)
        noisy[-1] = clean[0] + 4.0

        clean_found = detect_rounding(clean, self._times(n), self._cfg())
        noisy_found = detect_rounding(noisy, self._times(n), self._cfg())
        assert clean_found and noisy_found
        assert noisy_found[0].details["quad_r2"] < clean_found[0].details["quad_r2"]
        assert noisy_found[0].confidence < clean_found[0].confidence

    def test_random_walks_are_not_roundings(self):
        n = 221
        flagged = 0
        for seed in range(25):
            rng = np.random.RandomState(seed)
            walk = 100.0 + np.cumsum(rng.normal(0, 0.6, n))
            if detect_rounding(walk, self._times(n), self._cfg()):
                flagged += 1
        assert flagged == 0

    def test_end_index_matches_the_confirmation_bar(self):
        n = 221
        x = np.linspace(-1.0, 1.0, n)
        close = 100.0 + 12.0 * x**2
        close[-1] = close[0] + 4.0
        found = detect_rounding(close, self._times(n), self._cfg())

        assert found[0].end_index == n - 1
        assert found[0].details["breakout_index"] == n - 1


class TestLateNecklineBreak:
    """Bounding the forward neckline scan by breakout_lookahead left already
    triggered patterns reported as forming forever."""

    @staticmethod
    def _double_top(break_at: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        close = np.full(n, 95.0)
        close[38:43] = [95.0, 98.0, 100.0, 98.0, 95.0]
        close[58:63] = [95.0, 92.0, 91.0, 92.0, 95.0]
        close[78:83] = [95.0, 98.0, 100.0, 98.0, 95.0]
        close[break_at:] = 85.0
        return close, np.array([40, 80], dtype=int), np.array([60], dtype=int)

    def test_break_far_beyond_the_trailing_window_completes(self):
        n = 200
        close, peaks, troughs = self._double_top(150, n)
        cfg = ClassicDetectorConfig(same_level_tol_pct=0.4)
        found = detect_tops_bottoms(
            close, peaks, troughs, np.arange(n, dtype=float), cfg
        )

        tops = [item for item in found if item.name == "Double Top"]
        assert tops, "double top geometry was not detected at all"
        assert tops[0].status == "completed"
        assert tops[0].details["breakout_index"] == 150

    def test_configured_horizon_still_bounds_the_scan(self):
        n = 200
        close, peaks, troughs = self._double_top(150, n)
        cfg = ClassicDetectorConfig(
            same_level_tol_pct=0.4, neckline_break_lookahead_bars=8
        )
        found = detect_tops_bottoms(
            close, peaks, troughs, np.arange(n, dtype=float), cfg
        )
        tops = [item for item in found if item.name == "Double Top"]
        assert not tops or tops[0].details["breakout_index"] is None

    def test_zero_means_scan_to_the_end_of_the_series(self):
        cfg = ClassicDetectorConfig(neckline_break_lookahead_bars=0)
        assert _neckline_break_lookahead(cfg, n=200, start_idx=80) == 119

    def test_configured_value_is_capped_by_remaining_bars(self):
        cfg = ClassicDetectorConfig(neckline_break_lookahead_bars=500)
        assert _neckline_break_lookahead(cfg, n=200, start_idx=190) == 9

    @staticmethod
    def _cup_with_late_rim_break() -> np.ndarray:
        x = np.linspace(-1.0, 1.0, 150)
        cup = 100.0 + 12.0 * x**2
        rim = float(cup[0])
        handle = np.array([rim - 2.5, rim - 3.5, rim - 4.0, rim - 3.0, rim - 2.0])
        return np.concatenate(
            [
                np.full(40, rim + 1.0),
                cup,
                handle,
                np.full(60, rim - 2.0),
                np.full(20, rim + 5.0),
            ]
        )

    def test_cup_handle_rim_break_long_after_the_handle_completes(self):
        close = self._cup_with_late_rim_break()
        n = close.size
        found = detect_cup_handle(
            close, np.arange(n, dtype=float), ClassicDetectorConfig()
        )

        assert found, "cup-and-handle geometry was not detected at all"
        assert found[0].status == "completed"
        assert found[0].details["breakout_index"] == 255

    def test_cup_handle_respects_a_configured_break_horizon(self):
        close = self._cup_with_late_rim_break()
        n = close.size
        found = detect_cup_handle(
            close,
            np.arange(n, dtype=float),
            ClassicDetectorConfig(neckline_break_lookahead_bars=8),
        )

        assert found
        assert found[0].status == "forming"
        assert found[0].details["breakout_index"] is None


class TestPostApexBoundaries:
    """Ordering was verified only to the last pivot, but the breakout scan reads
    the trailing bars, where converging lines have crossed and inverted."""

    @staticmethod
    def _crossed_boundaries(n: int = 300):
        x = np.arange(n, dtype=float)
        upper = -0.5 * x + 225.0
        lower = 0.5 * x - 25.0
        return upper, lower

    def test_apex_index_is_found(self):
        upper, lower = self._crossed_boundaries()
        assert _boundary_apex_index(upper, lower, start_idx=0) == 250

    def test_parallel_boundaries_have_no_apex(self):
        x = np.arange(300, dtype=float)
        assert _boundary_apex_index(x + 10.0, x - 10.0, start_idx=0) is None

    @pytest.mark.parametrize("price", [60.0, 140.0])
    def test_no_breakout_is_reported_past_the_apex(self, price):
        n = 300
        upper, lower = self._crossed_boundaries(n)
        close = np.full(n, 100.0)
        close[-1] = price

        raw_dir, raw_idx = _find_recent_breakout(
            close, upper=upper, lower=lower, tol_abs=0.0, lookback_bars=8
        )
        assert raw_dir is not None and raw_idx is not None

        masked_upper, masked_lower, apex = _mask_boundaries_after_apex(
            upper, lower, start_idx=0
        )
        assert apex == 250
        direction, index = _find_recent_breakout(
            close,
            upper=masked_upper,
            lower=masked_lower,
            tol_abs=0.0,
            lookback_bars=8,
        )
        assert direction is None
        assert index is None

    def test_pre_apex_breakout_is_still_detected(self):
        n = 300
        upper, lower = self._crossed_boundaries(n)
        masked_upper, masked_lower, _ = _mask_boundaries_after_apex(
            upper, lower, start_idx=0
        )
        close = np.full(n, 100.0)
        close[100] = float(upper[100]) + 5.0

        direction, index = _find_recent_breakout(
            close,
            upper=masked_upper,
            lower=masked_lower,
            tol_abs=0.0,
            lookback_bars=n,
        )
        assert direction == "up"
        assert index == 100


class TestHeadShouldersReview:
    """A 0.6% shoulder gate silently rejected realistic formations, and no scalar
    neckline was emitted for the measured move."""

    @staticmethod
    def _hs_series(right_shoulder: float):
        close = np.array(
            [80.0, 95.0, 88.0, 110.0, 88.0, right_shoulder, 70.0], dtype=float
        )
        return close, np.array([1, 3, 5], dtype=int), np.array([2, 4], dtype=int)

    @pytest.mark.parametrize("right_shoulder", [95.0, 96.0, 97.5])
    def test_shoulders_a_few_percent_apart_are_accepted(self, right_shoulder):
        close, peaks, troughs = self._hs_series(right_shoulder)
        cfg = ClassicDetectorConfig(use_dtw_check=False, use_robust_fit=False)
        found = detect_head_shoulders(
            close, peaks, troughs, np.arange(close.size, dtype=float), cfg
        )
        assert any(item.name == "Head and Shoulders" for item in found)

    def test_grossly_mismatched_shoulders_are_still_rejected(self):
        close, peaks, troughs = self._hs_series(78.0)
        cfg = ClassicDetectorConfig(use_dtw_check=False, use_robust_fit=False)
        found = detect_head_shoulders(
            close, peaks, troughs, np.arange(close.size, dtype=float), cfg
        )
        assert not [item for item in found if item.name == "Head and Shoulders"]

    def test_a_scalar_neckline_is_emitted_for_the_measured_move(self):
        close, peaks, troughs = self._hs_series(95.0)
        cfg = ClassicDetectorConfig(use_dtw_check=False, use_robust_fit=False)
        found = detect_head_shoulders(
            close, peaks, troughs, np.arange(close.size, dtype=float), cfg
        )
        pattern = next(item for item in found if item.name == "Head and Shoulders")
        details = pattern.details

        assert details["neckline"] is not None
        assert np.isfinite(float(details["neckline"]))
        expected = details["neck_slope"] * details["neckline_index"] + (
            details["neck_intercept"]
        )
        assert float(details["neckline"]) == pytest.approx(float(expected))
        assert float(details["neckline"]) < float(details["head"])


class TestMeasuredTouchCounts:
    """Flags, pennants and broadening formations passed a literal 4 into _conf,
    which saturated the touch term and made it a constant."""

    def test_conf_touch_term_saturates_at_twice_min_touches(self):
        from mtdata.patterns.classic_impl.utils import _conf

        cfg = ClassicDetectorConfig(min_touches=2)
        assert _conf(4, 0.0, 1.0, cfg) == pytest.approx(_conf(9, 0.0, 1.0, cfg))
        assert _conf(2, 0.5, 1.0, cfg) < _conf(4, 0.5, 1.0, cfg)

    def test_detectors_no_longer_hardcode_the_touch_count(self):
        import inspect

        from mtdata.patterns.classic_impl import continuation, shapes

        assert "_conf(4," not in inspect.getsource(continuation.detect_flags_pennants)
        assert "_conf(4," not in inspect.getsource(shapes.detect_broadening)


class TestDiamondSlopeScaling:
    """The diamond gate used max_flat_slope directly, while every other shape
    detector scales it by median price."""

    def test_threshold_scales_with_price_level(self):
        from mtdata.patterns.classic_impl.utils import _effective_flat_slope

        cfg = ClassicDetectorConfig()
        cheap = _effective_flat_slope(np.full(200, 1.2), cfg)
        rich = _effective_flat_slope(np.full(200, 5000.0), cfg)
        assert rich > cheap

    def test_threshold_rises_with_the_analysed_segment(self):
        from mtdata.patterns.classic_impl.utils import _effective_flat_slope

        cfg = ClassicDetectorConfig()
        # A slope that clears the gate on a cheap instrument must not clear it on
        # an index priced three orders of magnitude higher.
        candidate_slope = 0.02
        assert candidate_slope > _effective_flat_slope(np.full(200, 100.0), cfg)
        assert candidate_slope < _effective_flat_slope(np.full(200, 5000.0), cfg)


class TestFractalGeometryDisclosure:
    """A close-only frame produced a payload indistinguishable from the OHLC case."""

    @staticmethod
    def _frame(*, with_extremes: bool) -> pd.DataFrame:
        close = np.array(
            [10.0, 11.0, 12.5, 11.0, 10.0, 9.0, 8.5, 9.5, 10.5, 11.5, 10.5],
            dtype=float,
        )
        data = {"time": np.arange(close.size, dtype=float), "close": close}
        if with_extremes:
            data["high"] = close + 0.5
            data["low"] = close - 0.5
        return pd.DataFrame(data)

    def test_close_only_frame_is_labelled(self):
        cfg = FractalDetectorConfig(left_bars=2, right_bars=2)
        found = detect_fractal_patterns(self._frame(with_extremes=False), cfg)
        assert found
        assert all(
            item.details["geometry_price_source"] == "close" for item in found
        )

    def test_ohlc_frame_is_labelled(self):
        cfg = FractalDetectorConfig(left_bars=2, right_bars=2)
        found = detect_fractal_patterns(self._frame(with_extremes=True), cfg)
        assert found
        assert all(
            item.details["geometry_price_source"] == "high_low" for item in found
        )
