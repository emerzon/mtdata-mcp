"""Extended coverage tests for patterns/elliott.py targeting uncovered lines."""
import numpy as np
import pandas as pd
import pytest

import mtdata.patterns.elliott as elliott_mod
from mtdata.patterns.elliott import (
    ElliottRuleEvaluation,
    ElliottScenario,
    ElliottWaveAnalyzer,
    ElliottWaveConfig,
    ElliottWaveResult,
    _apply_confirmation_confidence_adjustments,
    _classification_score_window,
    _classify_waves,
    _enforce_min_distance_on_pivots,
    _enforce_pivot_alternation,
    _evaluate_correction_rules,
    _evaluate_impulse_rules,
    _filter_nested_results,
    _normalize_pattern_types,
    _result_sort_key,
    _rule_confidence_from_eval,
    _segment_waves_from_pivots,
    _window_hit,
    _zigzag_pivots_indices,
    detect_elliott_waves,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _trending_close(n=200, seed=42):
    """Generate a zig-zag close array suitable for wave detection."""
    rng = np.random.RandomState(seed)
    base = np.cumsum(rng.randn(n) * 0.5) + 100.0
    return np.maximum(base, 10.0)


def _impulse_close():
    """Hand-crafted 5-wave bullish impulse: 0→1→2→3→4→5."""
    return np.array([100, 120, 108, 150, 135, 160], dtype=float)


def _correction_close():
    """Hand-crafted ABC bullish correction: S→A→B→C."""
    return np.array([100, 115, 107, 125], dtype=float)


def _make_df(close, n=None):
    if n is None:
        n = len(close)
    t = np.arange(n, dtype=float) * 3600 + 1_700_000_000
    return pd.DataFrame({"time": t[:len(close)], "close": close})


# ===== _zigzag_pivots_indices (lines 124-139) ==============================

class TestZigzagPivotsExtended:
    def test_empty_array(self):
        piv, dirs = _zigzag_pivots_indices(np.array([]), 1.0)
        assert piv == [] and dirs == []

    def test_single_element(self):
        piv, dirs = _zigzag_pivots_indices(np.array([100.0]), 1.0)
        assert piv == [] and dirs == []

    def test_both_up_and_down_start_low_first(self):
        """Lines 124-128: can_start_up and can_start_down, pre_low_i > pre_high_i → up."""
        close = np.array([100.0, 95.0, 110.0], dtype=float)
        piv, dirs = _zigzag_pivots_indices(close, 3.0)
        assert len(piv) >= 1

    def test_both_start_high_first(self):
        """Lines 128-131: pre_high_i > pre_low_i → down."""
        close = np.array([100.0, 108.0, 90.0], dtype=float)
        piv, dirs = _zigzag_pivots_indices(close, 3.0)
        assert len(piv) >= 1

    def test_both_start_equal_indices_up_wins(self):
        """Lines 132-135: same index, up_change >= down_change → up."""
        close = np.array([100.0, 100.0, 115.0], dtype=float)
        piv, dirs = _zigzag_pivots_indices(close, 5.0)
        assert len(piv) >= 1

    def test_both_start_equal_indices_down_wins(self):
        """Lines 136-139: same index, down_change > up_change → down."""
        close = np.array([100.0, 100.0, 80.0], dtype=float)
        piv, dirs = _zigzag_pivots_indices(close, 5.0)
        assert len(piv) >= 1

    def test_only_up_start(self):
        """Lines 140-142."""
        close = np.array([100.0, 99.0, 99.5, 115.0], dtype=float)
        piv, dirs = _zigzag_pivots_indices(close, 10.0)
        assert len(piv) >= 1

    def test_only_down_start(self):
        """Lines 144-147."""
        close = np.array([100.0, 101.0, 100.5, 85.0], dtype=float)
        piv, dirs = _zigzag_pivots_indices(close, 10.0)
        assert len(piv) >= 1

    def test_nan_skipped(self):
        """Line 108: non-finite values skipped."""
        close = np.array([100.0, np.nan, 95.0, 110.0, 90.0], dtype=float)
        piv, dirs = _zigzag_pivots_indices(close, 3.0)
        assert all(np.isfinite(close[i]) or i == 0 for i in piv)

    def test_leading_nan_anchors_on_first_finite_price(self):
        close = np.array([np.nan, np.nan, 100.0, 95.0, 110.0, 90.0], dtype=float)
        piv, dirs = _zigzag_pivots_indices(close, 3.0)
        assert piv
        assert all(i >= 2 for i in piv)
        assert len(piv) == len(dirs)

    def test_long_zigzag(self):
        close = _trending_close(100, seed=0)
        piv, dirs = _zigzag_pivots_indices(close, 2.0)
        assert len(piv) >= 3


# ===== _enforce_min_distance_on_pivots (lines 218-225) =====================

class TestEnforceMinDistanceExtended:
    def test_empty(self):
        assert _enforce_min_distance_on_pivots([], np.array([1.0]), 3) == []

    def test_single_pivot(self):
        assert _enforce_min_distance_on_pivots([0], np.array([100.0]), 3) == [0]

    def test_close_pivots_merge_anchor(self):
        """Lines 218-225: pivots too close, anchor-based merge keeps better move."""
        close = np.array([100, 110, 108, 95, 130, 125], dtype=float)
        pivots = [0, 1, 2, 3, 4, 5]
        out = _enforce_min_distance_on_pivots(pivots, close, 3)
        assert len(out) < len(pivots)

    def test_close_pivots_first_pair(self):
        """Lines 218-220: len(out)==1, replace."""
        close = np.array([100, 110, 120], dtype=float)
        out = _enforce_min_distance_on_pivots([0, 1, 2], close, 5)
        assert len(out) >= 1

    def test_out_of_bounds_skipped(self):
        """Lines 199: idx out of bounds."""
        close = np.array([100.0, 110.0], dtype=float)
        out = _enforce_min_distance_on_pivots([-1, 0, 5, 1], close, 1)
        assert -1 not in out and 5 not in out

    def test_duplicate_pivot(self):
        """Line 203-204: duplicate index skipped."""
        close = np.array([100, 110, 120], dtype=float)
        out = _enforce_min_distance_on_pivots([0, 0, 1, 2], close, 1)
        assert out.count(0) <= 1


# ===== _classify_waves (lines 280-284) =====================================

class TestClassifyWaves:
    def test_too_few_features(self):
        features = np.array([[1.0, 0.1, 0.01]])
        labels, gmm, scaler, probs, imp = _classify_waves(features, ElliottWaveConfig())
        assert labels.size == 0

    def test_respects_min_gmm_wave_floor(self):
        rng = np.random.RandomState(7)
        features = rng.randn(6, 3)
        labels, gmm, scaler, probs, imp = _classify_waves(
            features,
            ElliottWaveConfig(min_gmm_waves=8),
        )
        assert labels.size == 0
        assert gmm is None
        assert scaler is None
        assert probs is None
        assert imp is None

    def test_default_floor_requires_statistically_stable_sample(self):
        rng = np.random.RandomState(11)
        features = rng.randn(10, 3)
        labels, gmm, scaler, probs, imp = _classify_waves(features, ElliottWaveConfig())
        assert labels.size == 0
        assert gmm is None
        assert scaler is None
        assert probs is None
        assert imp is None

    def test_enough_features(self):
        """Lines 280-284: successful GMM classification."""
        rng = np.random.RandomState(42)
        features = np.vstack([
            rng.randn(10, 3) * [5, 0.02, 0.01],
            rng.randn(10, 3) * [3, 0.05, 0.02] + [0, 0.1, 0.05],
        ])
        labels, gmm, scaler, probs, imp_cluster = _classify_waves(features, ElliottWaveConfig())
        assert labels.shape[0] == 20
        assert gmm is not None
        assert probs is not None
        assert imp_cluster in (0, 1)

    def test_non_finite_scaled(self):
        """Lines 275-276: non-finite after scaling."""
        features = np.array([[np.inf, 0, 0], [1, 2, 3]], dtype=float)
        labels, _, _, _, _ = _classify_waves(features, ElliottWaveConfig())
        assert labels.size == 0


# ===== _classification_score_window (lines 422-440) ========================

class TestClassificationScoreWindow:
    def test_none_probs(self):
        """Line 420-421."""
        assert _classification_score_window(None, None, 0, True, 5, [0, 2], [1]) == 0.5

    def test_insufficient_probs(self):
        """Line 422-423."""
        probs = np.random.rand(3, 2)
        means = np.random.rand(2, 3)
        assert _classification_score_window(probs, means, 0, True, 5, [0], [1]) == 0.5

    def test_bad_cluster_means_shape(self):
        """Line 424-425."""
        probs = np.random.rand(20, 2)
        means = np.array([1.0])  # 1D
        assert _classification_score_window(probs, means, 0, True, 5, [0], [1]) == 0.5

    def test_valid_bullish(self):
        """Lines 427-440."""
        rng = np.random.RandomState(0)
        probs = rng.rand(20, 2)
        means = np.array([[0.1, 0.5, 0.1], [-0.1, -0.3, 0.0]])
        score = _classification_score_window(probs, means, 0, True, 5, [0, 2, 4], [1, 3])
        assert 0.0 <= score <= 1.0

    def test_valid_bearish(self):
        rng = np.random.RandomState(0)
        probs = rng.rand(20, 2)
        means = np.array([[0.1, 0.5, 0.1], [-0.1, -0.3, 0.0]])
        score = _classification_score_window(probs, means, 0, False, 5, [0, 2, 4], [1, 3])
        assert 0.0 <= score <= 1.0

    def test_empty_slots(self):
        """Lines 434-435: no trend_vals and no counter_vals."""
        probs = np.random.rand(20, 2)
        means = np.array([[0.1, 0.5], [-0.1, -0.3]])
        score = _classification_score_window(probs, means, 0, True, 5, [], [])
        assert score == 0.5

    def test_window_mismatch(self):
        """Line 429: window_probs.shape[0] != window_len."""
        probs = np.random.rand(7, 2)
        means = np.array([[0.1, 0.5], [-0.1, -0.3]])
        score = _classification_score_window(probs, means, 5, True, 5, [0], [1])
        assert score == 0.5

    def test_missing_wave_mapping_returns_neutral(self):
        probs = np.random.rand(3, 2)
        means = np.array([[0.1, 0.5], [-0.1, -0.3]])
        score = _classification_score_window(
            probs,
            means,
            0,
            True,
            3,
            [0, 2],
            [1],
            wave_index_map={0: 0, 2: 1, 3: 2},
        )
        assert score == 0.5


# ===== _evaluate_impulse_rules and _evaluate_correction_rules ==============

class TestPivotAlternation:
    def test_merges_same_direction_legs(self):
        close = np.array([100.0, 110.0, 115.0, 105.0, 120.0], dtype=float)
        # 0→1→2 are same-direction up legs if all kept as pivots.
        pivots = [0, 1, 2, 3, 4]
        cleaned = _enforce_pivot_alternation(pivots, close)
        prices = [float(close[i]) for i in cleaned]
        for i in range(1, len(prices) - 1):
            left = prices[i] - prices[i - 1]
            right = prices[i + 1] - prices[i]
            assert left * right < 0.0


class TestEvaluateImpulseRules:
    def test_valid_bullish(self):
        c = _impulse_close()
        ev = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=True)
        assert ev.valid
        assert ev.fib_score >= 0

    def test_direction_label_is_not_named_alternation(self):
        c = np.array([100, 120, 130, 110, 140, 160], dtype=float)
        ev = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=True)
        assert "direction_sequence_invalid" in ev.violations
        assert "alternation_failed" not in ev.violations

    def test_wrong_pivot_count(self):
        c = np.array([100, 120, 110], dtype=float)
        ev = _evaluate_impulse_rules(c, [0, 1, 2], bullish=True)
        assert not ev.valid
        assert "pivot_count_not_6" in ev.violations

    def test_wave2_over_retrace_bullish(self):
        """Lines 323-324."""
        c = np.array([100, 120, 90, 150, 135, 160], dtype=float)
        ev = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=True)
        assert "wave2_over_retrace" in ev.violations

    def test_wave2_over_retrace_bearish(self):
        """Lines 325-326."""
        c = np.array([160, 135, 170, 100, 108, 90], dtype=float)
        ev = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=False)
        assert "wave2_over_retrace" in ev.violations

    def test_wave3_shortest(self):
        """Lines 329-330."""
        c = np.array([100, 120, 115, 118, 110, 150], dtype=float)
        ev = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=True)
        assert "wave3_shortest" in ev.violations

    def test_wave4_overlap_bullish(self):
        """Lines 333-334."""
        c = np.array([100, 120, 108, 150, 115, 160], dtype=float)
        ev = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=True)
        assert "wave4_overlap" in ev.violations

    def test_wave4_overlap_bearish(self):
        """Lines 335-336."""
        c = np.array([160, 140, 148, 110, 145, 90], dtype=float)
        ev = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=False)
        assert "wave4_overlap" in ev.violations

    def test_non_finite(self):
        c = np.array([100, np.nan, 110, 130, 120, 140], dtype=float)
        ev = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=True)
        assert "non_finite_prices" in ev.violations

    def test_impulse_evaluation_exposes_score_metrics(self):
        c = _impulse_close()
        evaluation = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], True)
        assert isinstance(evaluation.valid, bool)
        assert evaluation.fib_score == evaluation.metrics["fib_score"]
        assert evaluation.metrics["fib_template_fit"] == evaluation.fib_score
        assert evaluation.metrics["rule_confidence"] == pytest.approx(
            _rule_confidence_from_eval(evaluation)
        )
        if evaluation.valid:
            assert evaluation.metrics["rule_confidence"] >= 0.55

    def test_valid_modest_extension_is_not_mid_scored(self):
        # W3 ≈ 1.1×W1 — hard-valid but previously Fib-window mid-scored.
        c = np.array([100.0, 120.0, 112.0, 134.0, 126.0, 146.0], dtype=float)
        evaluation = _evaluate_impulse_rules(c, list(range(6)), bullish=True)
        assert evaluation.valid is True
        assert evaluation.metrics["rule_confidence"] >= 0.55


class TestEvaluateCorrectionRules:
    def test_valid_bullish(self):
        c = _correction_close()
        ev = _evaluate_correction_rules(c, [0, 1, 2, 3], bullish=True)
        assert isinstance(ev.fib_score, float)

    def test_direction_label_is_not_named_alternation(self):
        c = np.array([100, 115, 120, 105], dtype=float)
        ev = _evaluate_correction_rules(c, [0, 1, 2, 3], bullish=True)
        assert "direction_sequence_invalid" in ev.violations
        assert "alternation_failed" not in ev.violations

    def test_wrong_count(self):
        ev = _evaluate_correction_rules(np.array([1, 2, 3], dtype=float), [0, 1, 2], True)
        assert "pivot_count_not_4" in ev.violations

    def test_waveB_over_retrace_bullish(self):
        c = np.array([100, 115, 95, 125], dtype=float)
        ev = _evaluate_correction_rules(c, [0, 1, 2, 3], bullish=True)
        assert "waveB_over_retrace" in ev.violations

    def test_waveB_over_retrace_bearish(self):
        c = np.array([125, 110, 130, 100], dtype=float)
        ev = _evaluate_correction_rules(c, [0, 1, 2, 3], bullish=False)
        assert "waveB_over_retrace" in ev.violations

    def test_waveC_too_short(self):
        c = np.array([100, 120, 115, 116], dtype=float)
        ev = _evaluate_correction_rules(c, [0, 1, 2, 3], bullish=True)
        assert "waveC_too_short" in ev.violations

    def test_waveC_quarter_of_a_is_allowed_by_default(self):
        c = np.array([100, 120, 115, 120], dtype=float)
        ev = _evaluate_correction_rules(c, [0, 1, 2, 3], bullish=True)
        assert "waveC_too_short" not in ev.violations


class TestResultSortKey:
    def test_breaks_ties_deterministically(self):
        shared = dict(
            confidence=0.8,
            end_index=10,
            start_index=0,
            start_time=None,
            end_time=None,
            details={},
        )
        results = [
            ElliottWaveResult(wave_type="Impulse", wave_sequence=[2, 4, 6, 8, 10, 12], **shared),
            ElliottWaveResult(wave_type="Correction", wave_sequence=[1, 3, 5, 10], **shared),
            ElliottWaveResult(wave_type="Impulse", wave_sequence=[1, 3, 5, 7, 9, 10], **shared),
        ]

        ordered = sorted(results, key=_result_sort_key)

        assert [r.wave_type for r in ordered] == ["Correction", "Impulse", "Impulse"]
        assert ordered[1].wave_sequence == [1, 3, 5, 7, 9, 10]

    def test_prefers_confirmed_longer_structures_before_shorter_counts(self):
        results = [
            ElliottWaveResult(
                wave_type="Correction",
                wave_sequence=[0, 10, 20, 30],
                confidence=0.95,
                start_index=0,
                end_index=30,
                start_time=None,
                end_time=None,
                details={"pattern_confirmed": True, "trend": "bull"},
            ),
            ElliottWaveResult(
                wave_type="Impulse",
                wave_sequence=[0, 25, 50, 75, 100, 125],
                confidence=0.8,
                start_index=0,
                end_index=125,
                start_time=None,
                end_time=None,
                details={"pattern_confirmed": True, "trend": "bull"},
            ),
        ]

        ordered = sorted(results, key=_result_sort_key)

        assert ordered[0].wave_type == "Impulse"

    def test_prefers_confirmed_patterns_over_unconfirmed_when_span_matches(self):
        results = [
            ElliottWaveResult(
                wave_type="Impulse",
                wave_sequence=[0, 1, 2, 3, 4, 5],
                confidence=0.9,
                start_index=0,
                end_index=50,
                start_time=None,
                end_time=None,
                details={"pattern_confirmed": False, "trend": "bull"},
            ),
            ElliottWaveResult(
                wave_type="Impulse",
                wave_sequence=[0, 2, 4, 6, 8, 10],
                confidence=0.8,
                start_index=0,
                end_index=50,
                start_time=None,
                end_time=None,
                details={"pattern_confirmed": True, "trend": "bull"},
            ),
        ]

        ordered = sorted(results, key=_result_sort_key)

        assert ordered[0].details["pattern_confirmed"] is True

    def test_filter_nested_results_discards_shorter_same_type_same_direction(self):
        results = [
            ElliottWaveResult(
                wave_type="Impulse",
                wave_sequence=[0, 25, 50, 75, 100, 125],
                confidence=0.8,
                start_index=0,
                end_index=125,
                start_time=None,
                end_time=None,
                details={"pattern_confirmed": True, "trend": "bull"},
            ),
            ElliottWaveResult(
                wave_type="Impulse",
                wave_sequence=[10, 20, 30, 40, 50, 60],
                confidence=0.91,
                start_index=10,
                end_index=60,
                start_time=None,
                end_time=None,
                details={"pattern_confirmed": True, "trend": "bull"},
            ),
            ElliottWaveResult(
                wave_type="Correction",
                wave_sequence=[10, 20, 30, 40],
                confidence=0.97,
                start_index=10,
                end_index=40,
                start_time=None,
                end_time=None,
                details={"pattern_confirmed": True, "trend": "bear"},
            ),
        ]

        filtered = _filter_nested_results(results)

        assert [r.wave_type for r in filtered] == ["Impulse", "Correction"]


# ===== ElliottWaveAnalyzer.build_result (lines 576-577) ====================

class TestBuildResult:
    def test_correction_labels(self):
        """Lines 567-568: correction → S, A, B, C labels."""
        c = _correction_close()
        t = np.arange(4, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(c, t, ElliottWaveConfig())
        rule_eval = _evaluate_correction_rules(c, [0, 1, 2, 3], bullish=True)
        scenario = ElliottScenario(
            pivots=[0, 1, 2, 3], bullish=True, confidence=0.5, cls_score=0.5,
            rule_eval=rule_eval, threshold_used=1.0, min_distance_used=1,
            wave_type="Correction",
        )
        result = analyzer.build_result(scenario)
        labels = [wp["label"] for wp in result.details["wave_points_labeled"]]
        assert labels == ["S", "A", "B", "C"]
        assert result.details["pattern_family"] == "correction"
        assert result.details["structure_type"] == "outer_abc_candidate"
        assert result.details["sequence_direction"] == "bull"
        assert "prior_impulse_direction" not in result.details
        assert "correction_metrics" in result.details
        assert "outer abc" in str(result.details.get("taxonomy_note", "")).lower()

    def test_impulse_labels(self):
        c = _impulse_close()
        t = np.arange(6, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(c, t, ElliottWaveConfig())
        rule_eval = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=True)
        scenario = ElliottScenario(
            pivots=[0, 1, 2, 3, 4, 5], bullish=True, confidence=0.7, cls_score=0.6,
            rule_eval=rule_eval, threshold_used=1.0, min_distance_used=1,
            wave_type="Impulse",
        )
        result = analyzer.build_result(scenario)
        labels = [wp["label"] for wp in result.details["wave_points_labeled"]]
        assert labels == ["0", "1", "2", "3", "4", "5"]
        assert result.details["structure_type"] == "impulse_strict"
        assert result.details["rule_price_source"] == result.details["pivot_price_source"]
        assert result.details["price_contract"] == "unified"

    def test_build_result_marks_unconfirmed_terminal_pivot_and_classifier_state(self):
        c = _impulse_close()
        t = np.arange(6, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(c, t, ElliottWaveConfig())
        rule_eval = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=True)
        scenario = ElliottScenario(
            pivots=[0, 1, 2, 3, 4, 5],
            bullish=True,
            confidence=0.7,
            base_confidence=0.92,
            cls_score=0.5,
            rule_eval=rule_eval,
            threshold_used=1.0,
            min_distance_used=1,
            wave_type="Impulse",
            classification_available=False,
            pivot_confirmations=[True, True, True, True, True, False],
            confidence_adjustments={
                "unconfirmed_pattern_penalty": 0.12,
                "unconfirmed_terminal_pivot_penalty": 0.10,
            },
        )

        result = analyzer.build_result(scenario)

        assert result.details["classification_available"] is False
        assert result.details["pattern_confirmed"] is False
        assert result.details["has_unconfirmed_terminal_pivot"] is True
        assert result.details["base_confidence"] == pytest.approx(0.92)
        assert result.details["confidence_adjustments"]["unconfirmed_pattern_penalty"] == pytest.approx(0.12)
        assert result.details["wave_points_labeled"][-1]["is_confirmed"] is False

    def test_fallback_result_exposes_validated_wave_type(self):
        c = _impulse_close()
        t = np.arange(6, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(c, t, ElliottWaveConfig())
        rule_eval = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=True)
        scenario = ElliottScenario(
            pivots=[0, 1, 2, 3, 4, 5],
            bullish=True,
            confidence=0.4,
            cls_score=0.5,
            rule_eval=rule_eval,
            threshold_used=0.5,
            min_distance_used=1,
            fallback_candidate=True,
            wave_type="Candidate",
            validated_wave_type="Impulse",
        )

        result = analyzer.build_result(scenario)

        assert result.wave_type == "Candidate"
        assert result.details["candidate_validates_as"] == "impulse"

    def test_impulse_result_includes_wave5_targets(self):
        c = _impulse_close()
        t = np.arange(6, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(c, t, ElliottWaveConfig())
        rule_eval = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=True)
        scenario = ElliottScenario(
            pivots=[0, 1, 2, 3, 4, 5],
            bullish=True,
            confidence=0.7,
            cls_score=0.6,
            rule_eval=rule_eval,
            threshold_used=1.0,
            min_distance_used=1,
            wave_type="Impulse",
        )

        result = analyzer.build_result(scenario)

        assert "wave5_targets" in result.details
        assert result.details["wave5_targets"]["zone_high"] >= result.details["wave5_targets"]["zone_low"]
        assert result.details["wave5_targets"]["equal_wave1"] == pytest.approx(rule_eval.metrics["wave5_target_equal_wave1"])

    def test_impulse_result_can_use_wick_aware_pivot_prices(self):
        c = _impulse_close()
        h = np.array([101.0, 123.0, 110.0, 155.0, 136.0, 165.0], dtype=float)
        l = np.array([97.0, 118.0, 105.0, 149.0, 130.0, 159.0], dtype=float)
        t = np.arange(6, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(c, t, ElliottWaveConfig(pivot_price_source="ohlc"), high=h, low=l)
        rule_eval = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=True)
        scenario = ElliottScenario(
            pivots=[0, 1, 2, 3, 4, 5],
            bullish=True,
            confidence=0.7,
            cls_score=0.6,
            rule_eval=rule_eval,
            threshold_used=1.0,
            min_distance_used=1,
            wave_type="Impulse",
        )

        result = analyzer.build_result(scenario)

        expected_prices = [97.0, 123.0, 105.0, 155.0, 130.0, 165.0]
        assert result.details["wave_points"] == pytest.approx(expected_prices)
        assert [point["price"] for point in result.details["wave_points_labeled"]] == pytest.approx(expected_prices)
        assert result.details["invalidation_level"] == pytest.approx(97.0)
        assert result.details["pivot_price_source"] == "ohlc"
        # Unified contract: rules and display share the configured pivot source.
        assert result.details["rule_price_source"] == "ohlc"
        assert result.details["price_contract"] == "unified"
        assert result.details["wave5_targets"]["equal_wave1"] == pytest.approx(156.0)
        assert result.details["wave5_targets"]["wave3_0_618"] == pytest.approx(160.9)
        assert result.details["wave5_targets"]["retrospective"] is True


# ===== ElliottWaveAnalyzer.build_fallback (lines 623-651) ==================

class TestBuildFallback:
    def test_disabled_fallback(self):
        """Line 618-619."""
        cfg = ElliottWaveConfig(include_fallback_candidate=False)
        c = _trending_close(50)
        t = np.arange(50, dtype=float) * 3600
        analyzer = ElliottWaveAnalyzer(c, t, cfg)
        assert analyzer.build_fallback(1.0, 3) is None

    def test_too_short(self):
        """Lines 621-623."""
        cfg = ElliottWaveConfig()
        c = np.array([100.0])
        t = np.array([0.0])
        analyzer = ElliottWaveAnalyzer(c, t, cfg)
        assert analyzer.build_fallback(1.0, 3) is None

    def test_fallback_produces_result(self):
        """Lines 625-667: full fallback path."""
        cfg = ElliottWaveConfig(include_fallback_candidate=True, min_distance=1)
        close = _trending_close(60, seed=10)
        t = np.arange(60, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(close, t, cfg)
        result = analyzer.build_fallback(0.5, 1)
        # May or may not produce depending on data, but shouldn't crash
        assert result is None or isinstance(result, ElliottWaveResult)

    def test_fallback_correction_only(self):
        """Line 637: correction-only pattern_types → preferred_len=4."""
        cfg = ElliottWaveConfig(
            include_fallback_candidate=True, min_distance=1,
            pattern_types=["correction"],
        )
        close = _trending_close(60, seed=20)
        t = np.arange(60, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(close, t, cfg)
        result = analyzer.build_fallback(0.5, 1)
        assert result is None or isinstance(result, ElliottWaveResult)

    def test_fallback_marks_synthetic_terminal_pivot(self, monkeypatch):
        cfg = ElliottWaveConfig(include_fallback_candidate=True, min_distance=1, pattern_types=["correction"])
        close = _trending_close(20, seed=21)
        t = np.arange(20, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(close, t, cfg)

        monkeypatch.setattr(elliott_mod, "_zigzag_pivots_indices", lambda *_args, **_kwargs: ([0, 4, 8], ["up", "down", "up"]))
        monkeypatch.setattr(elliott_mod, "_enforce_min_distance_on_pivots", lambda pivots, *_args, **_kwargs: list(pivots))

        result = analyzer.build_fallback(0.5, 1)

        assert result is not None
        assert result.wave_sequence[-1] == len(close) - 1
        assert result.details["synthetic_terminal_pivot"] is True


# ===== detect_elliott_waves (lines 672-728, 747-751) =======================

class TestDetectElliottWaves:
    def test_none_config(self):
        """Line 671-672."""
        df = _make_df(_trending_close(100))
        results = detect_elliott_waves(df, None)
        assert isinstance(results, list)

    def test_not_dataframe(self):
        """Line 674-675."""
        assert detect_elliott_waves("not_a_df") == []

    def test_no_close_column(self):
        df = pd.DataFrame({"open": [1, 2, 3]})
        assert detect_elliott_waves(df) == []

    def test_too_short(self):
        df = _make_df(np.array([100.0, 110.0]))
        assert detect_elliott_waves(df) == []

    def test_threshold_scan_default_distances(self):
        close = _trending_close(200, seed=5)
        df = _make_df(close)
        cfg = ElliottWaveConfig(
            scan_thresholds_pct=[0.2, 0.3, 0.5, 0.75, 1.0],
            min_distance=3,
            min_confidence=0.0,
        )
        results = detect_elliott_waves(df, cfg)
        assert isinstance(results, list)

    def test_threshold_scan_custom_thresholds(self):
        close = _trending_close(200, seed=6)
        df = _make_df(close)
        cfg = ElliottWaveConfig(
            scan_thresholds_pct=[0.5, 1.0],
            scan_min_distances=[2, 4], min_confidence=0.0,
        )
        results = detect_elliott_waves(df, cfg)
        assert isinstance(results, list)

    def test_single_threshold_path(self):
        close = _trending_close(200, seed=7)
        df = _make_df(close)
        cfg = ElliottWaveConfig(min_prominence_pct=1.0, min_distance=3)
        results = detect_elliott_waves(df, cfg)
        assert isinstance(results, list)

    def test_single_explicit_swing_threshold(self):
        close = _trending_close(200, seed=8)
        df = _make_df(close)
        cfg = ElliottWaveConfig(swing_threshold_pct=0.5, min_distance=3)
        results = detect_elliott_waves(df, cfg)
        assert isinstance(results, list)

    def test_top_k_limits(self):
        """Line 754-756."""
        close = _trending_close(200, seed=9)
        df = _make_df(close)
        cfg = ElliottWaveConfig(
            scan_thresholds_pct=[0.2, 0.3, 0.5, 0.75, 1.0],
            min_distance=2,
            min_confidence=0.0,
            top_k=2,
        )
        results = detect_elliott_waves(df, cfg)
        assert len(results) <= 2

    def test_correction_only(self):
        close = _trending_close(200, seed=11)
        df = _make_df(close)
        cfg = ElliottWaveConfig(
            scan_thresholds_pct=[0.2, 0.3, 0.5, 0.75, 1.0],
            min_distance=2,
            pattern_types=["correction"],
            min_confidence=0.0,
        )
        results = detect_elliott_waves(df, cfg)
        assert all(r.wave_type in ("Correction", "Candidate") for r in results)

    def test_dedup_keeps_higher_confidence(self):
        """Lines 720-726: duplicate key keeps higher confidence."""
        close = _trending_close(200, seed=12)
        df = _make_df(close)
        cfg = ElliottWaveConfig(
            scan_thresholds_pct=[0.3, 0.5, 0.7],
            scan_min_distances=[2, 3], min_confidence=0.0,
        )
        results = detect_elliott_waves(df, cfg)
        keys = [(r.wave_type, tuple(r.wave_sequence)) for r in results]
        assert len(keys) == len(set(keys)), "Duplicate wave sequences should be deduped"

    def test_fallback_appended_when_no_recent(self):
        """Lines 738-745: fallback candidate appended."""
        rng = np.random.RandomState(55)
        # Build data where waves end early, leaving tail without recent detection
        close = np.concatenate([
            _trending_close(50, seed=55),
            np.full(50, 100.0) + rng.randn(50) * 0.01,
        ])
        df = _make_df(close)
        cfg = ElliottWaveConfig(
            scan_thresholds_pct=[0.2, 0.3, 0.5, 0.75, 1.0],
            min_distance=2, min_confidence=0.0,
            include_fallback_candidate=True, recent_bars=3,
        )
        results = detect_elliott_waves(df, cfg)
        assert isinstance(results, list)

    def test_threshold_scan_duplicate_sequence_keeps_highest_confidence(self, monkeypatch):
        df = _make_df(np.linspace(100.0, 120.0, 80))
        cfg = ElliottWaveConfig(
            scan_thresholds_pct=[0.2, 0.5],
            scan_min_distances=[1],
            include_fallback_candidate=False,
            min_confidence=0.0,
            top_k=10,
        )

        monkeypatch.setattr(
            elliott_mod.ElliottWaveAnalyzer,
            "pivot_signature",
            lambda _self, threshold_pct, _min_distance: (0, 10, 20, 30, 40, int(float(threshold_pct) * 100)),
        )

        def _fake_analyze(self, threshold_pct, min_distance):
            _ = self
            return [
                ElliottScenario(
                    pivots=[0, 10, 20, 30],
                    bullish=True,
                    confidence=float(threshold_pct),
                    cls_score=0.5,
                    rule_eval=ElliottRuleEvaluation(valid=True, fib_score=0.5, metrics={}),
                    threshold_used=float(threshold_pct),
                    min_distance_used=int(min_distance),
                    wave_type="Correction",
                )
            ]

        monkeypatch.setattr(elliott_mod.ElliottWaveAnalyzer, "analyze_once", _fake_analyze)

        results = detect_elliott_waves(df, cfg)

        assert len(results) == 1
        assert results[0].wave_type == "Correction"
        assert results[0].confidence == pytest.approx(0.5)

    def test_threshold_scan_skips_repeated_pivot_signatures(self, monkeypatch):
        df = _make_df(np.linspace(100.0, 120.0, 80))
        cfg = ElliottWaveConfig(
            scan_thresholds_pct=[0.2, 0.5],
            scan_min_distances=[1],
            include_fallback_candidate=False,
            min_confidence=0.0,
            top_k=10,
        )
        call_log = []

        monkeypatch.setattr(
            elliott_mod.ElliottWaveAnalyzer,
            "pivot_signature",
            lambda *_args, **_kwargs: (0, 10, 20, 30),
        )

        def _fake_analyze(self, threshold_pct, min_distance):
            _ = self
            call_log.append((float(threshold_pct), int(min_distance)))
            return [
                ElliottScenario(
                    pivots=[0, 10, 20, 30],
                    bullish=True,
                    confidence=0.6,
                    cls_score=0.5,
                    rule_eval=ElliottRuleEvaluation(valid=True, fib_score=0.5, metrics={}),
                    threshold_used=float(threshold_pct),
                    min_distance_used=int(min_distance),
                    wave_type="Correction",
                )
            ]

        monkeypatch.setattr(elliott_mod.ElliottWaveAnalyzer, "analyze_once", _fake_analyze)

        results = detect_elliott_waves(df, cfg)

        assert len(call_log) == 1
        assert len(results) == 1

    def test_threshold_scan_stops_after_repeated_similar_scenarios(self, monkeypatch):
        df = _make_df(np.linspace(100.0, 120.0, 120))
        cfg = ElliottWaveConfig(
            scan_thresholds_pct=[0.2, 0.3, 0.4, 0.5],
            scan_min_distances=[1],
            include_fallback_candidate=False,
            min_confidence=0.0,
            scan_skip_repeated_pivots=False,
            scan_early_stop_repeats=2,
            scan_scenario_overlap_ratio=0.95,
        )
        call_log = []

        monkeypatch.setattr(
            elliott_mod.ElliottWaveAnalyzer,
            "pivot_signature",
            lambda _self, threshold_pct, _min_distance: (0, 10, 20, 30, int(float(threshold_pct) * 100)),
        )

        def _fake_analyze(self, threshold_pct, min_distance):
            _ = self
            call_log.append((float(threshold_pct), int(min_distance)))
            shift = int(round(float(threshold_pct) * 10.0))
            return [
                ElliottScenario(
                    pivots=[0, 20 + shift, 40, 60],
                    bullish=True,
                    confidence=0.6,
                    cls_score=0.5,
                    rule_eval=ElliottRuleEvaluation(valid=True, fib_score=0.5, metrics={}),
                    threshold_used=float(threshold_pct),
                    min_distance_used=int(min_distance),
                    wave_type="Correction",
                )
            ]

        monkeypatch.setattr(elliott_mod.ElliottWaveAnalyzer, "analyze_once", _fake_analyze)

        results = detect_elliott_waves(df, cfg)

        assert len(call_log) < 4
        assert results
        assert all(result.wave_type == "Correction" for result in results)

    def test_threshold_scan_filters_correction_overlap_with_impulse(self, monkeypatch):
        df = _make_df(np.linspace(100.0, 120.0, 80))
        cfg = ElliottWaveConfig(
            scan_thresholds_pct=[0.2, 0.5],
            scan_min_distances=[1],
            include_fallback_candidate=False,
            min_confidence=0.0,
            top_k=10,
        )

        monkeypatch.setattr(
            elliott_mod.ElliottWaveAnalyzer,
            "pivot_signature",
            lambda _self, threshold_pct, _min_distance: (0, 10, 20, 30, 40, int(float(threshold_pct) * 100)),
        )

        def _fake_analyze(self, threshold_pct, min_distance):
            _ = self
            if float(threshold_pct) < 0.3:
                return [
                    ElliottScenario(
                        pivots=[0, 10, 20, 30],
                        bullish=True,
                        confidence=0.6,
                        cls_score=0.5,
                        rule_eval=ElliottRuleEvaluation(valid=True, fib_score=0.5, metrics={}),
                        threshold_used=float(threshold_pct),
                        min_distance_used=int(min_distance),
                        wave_type="Correction",
                    )
                ]
            return [
                ElliottScenario(
                    pivots=[0, 10, 20, 30, 40, 50],
                    bullish=True,
                    confidence=0.7,
                    cls_score=0.5,
                    rule_eval=ElliottRuleEvaluation(valid=True, fib_score=0.5, metrics={}),
                    threshold_used=float(threshold_pct),
                    min_distance_used=int(min_distance),
                    wave_type="Impulse",
                )
            ]

        monkeypatch.setattr(elliott_mod.ElliottWaveAnalyzer, "analyze_once", _fake_analyze)

        results = detect_elliott_waves(df, cfg)

        assert any(result.wave_type == "Impulse" for result in results)
        assert not any(result.wave_type == "Correction" for result in results)

    def test_analyze_once_skips_correction_subsequence_of_impulse(self, monkeypatch):
        close = _impulse_close()
        t = np.arange(close.size, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(
            close,
            t,
            ElliottWaveConfig(
                min_distance=1,
                wave_min_len=1,
                min_impulse_bars=1,
                min_correction_bars=1,
                min_confidence=0.0,
                pattern_types=["impulse", "correction"],
                include_fallback_candidate=False,
            ),
        )

        monkeypatch.setattr(elliott_mod, "_zigzag_pivots_indices", lambda *_args, **_kwargs: ([0, 1, 2, 3, 4, 5], ["down", "up", "down", "up", "down", "up"]))

        fake_gmm = type("FakeGMM", (), {"means_": np.array([[0.0, -0.1, 0.0], [0.0, 0.1, 0.0]])})()
        monkeypatch.setattr(
            elliott_mod,
            "_classify_waves",
            lambda features, config: (
                np.array([1, 0, 1, 0, 1], dtype=int),
                fake_gmm,
                None,
                np.full((features.shape[0], 2), 0.5, dtype=float),
                1,
            ),
        )

        scenarios = analyzer.analyze_once(1.0, 1)

        assert any(s.wave_type == "Impulse" for s in scenarios)
        assert not any(s.wave_type == "Correction" for s in scenarios)

    def test_analyze_once_skips_near_matching_correction_subsequence_of_impulse(self, monkeypatch):
        close = np.array([100.0, 120.0, 108.0, 150.0, 135.0, 158.0, 160.0], dtype=float)
        t = np.arange(close.size, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(
            close,
            t,
            ElliottWaveConfig(
                min_distance=1,
                wave_min_len=1,
                min_impulse_bars=1,
                min_correction_bars=1,
                min_confidence=0.0,
                pattern_types=["impulse", "correction"],
                include_fallback_candidate=False,
                correction_exclusion_bar_tolerance=1,
                correction_exclusion_overlap_ratio=0.9,
            ),
        )

        monkeypatch.setattr(
            elliott_mod,
            "_zigzag_pivots_indices",
            lambda *_args, **_kwargs: ([0, 1, 2, 3, 4, 5, 6], ["down", "up", "down", "up", "down", "up", "up"]),
        )
        monkeypatch.setattr(
            elliott_mod,
            "_contiguous_pivot_slices",
            lambda pivots, size: {
                tuple(
                    int(v + (1 if idx == len(pivots[:size]) - 1 else 0))
                    for idx, v in enumerate(pivots[offset : offset + size])
                )
                for offset in range(len(pivots) - size + 1)
            },
        )

        fake_gmm = type("FakeGMM", (), {"means_": np.array([[0.0, -0.1, 0.0], [0.0, 0.1, 0.0]])})()
        monkeypatch.setattr(
            elliott_mod,
            "_classify_waves",
            lambda features, config: (
                np.arange(features.shape[0], dtype=int) % 2,
                fake_gmm,
                None,
                np.full((features.shape[0], 2), 0.5, dtype=float),
                1,
            ),
        )

        scenarios = analyzer.analyze_once(1.0, 1)

        assert any(s.wave_type == "Impulse" for s in scenarios)
        assert not any(s.wave_type == "Correction" for s in scenarios)

    def test_analyze_once_uses_rule_score_when_classification_unavailable(self, monkeypatch):
        close = _impulse_close()
        t = np.arange(close.size, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(
            close,
            t,
            ElliottWaveConfig(
                min_distance=1,
                wave_min_len=1,
                min_impulse_bars=1,
                min_confidence=0.0,
                pattern_types=["impulse"],
                include_fallback_candidate=False,
            ),
        )

        monkeypatch.setattr(elliott_mod, "_zigzag_pivots_indices", lambda *_args, **_kwargs: ([0, 1, 2, 3, 4, 5], ["down", "up", "down", "up", "down", "up"]))
        monkeypatch.setattr(
            elliott_mod,
            "_classify_waves",
            lambda features, config: (np.array([]), None, None, None, None),
        )

        scenarios = analyzer.analyze_once(1.0, 1)

        assert scenarios
        assert scenarios[0].classification_available is False
        rule_conf = _rule_confidence_from_eval(scenarios[0].rule_eval)
        expected, adjustments = _apply_confirmation_confidence_adjustments(
            rule_conf,
            scenarios[0].pivot_confirmations,
            analyzer.config,
        )
        assert scenarios[0].confidence == pytest.approx(expected)
        assert scenarios[0].base_confidence == pytest.approx(rule_conf)
        assert scenarios[0].confidence_adjustments == adjustments
        assert scenarios[0].pivot_confirmations[:-1] == [True] * (len(scenarios[0].pivot_confirmations) - 1)
        assert scenarios[0].pivot_confirmations[-1] is False

    def test_analyze_once_keeps_gmm_diagnostic_only(self, monkeypatch):
        close = _impulse_close()
        t = np.arange(close.size, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(
            close,
            t,
            ElliottWaveConfig(
                min_distance=1,
                wave_min_len=1,
                min_impulse_bars=1,
                enable_gmm_classifier=True,
                min_confidence=0.0,
                pattern_types=["impulse"],
                include_fallback_candidate=False,
            ),
        )

        monkeypatch.setattr(elliott_mod, "_zigzag_pivots_indices", lambda *_args, **_kwargs: ([0, 1, 2, 3, 4, 5], ["down", "up", "down", "up", "down", "up"]))
        monkeypatch.setattr(
            elliott_mod,
            "_classify_waves",
            lambda features, config: (
                np.arange(features.shape[0], dtype=int),
                type("FakeGMM", (), {"means_": np.array([[0.0, -0.1, 0.0], [0.0, 0.1, 0.0]])})(),
                None,
                np.full((features.shape[0], 2), 0.75, dtype=float),
                1,
            ),
        )

        scenarios = analyzer.analyze_once(1.0, 1)

        assert scenarios
        rule_conf = _rule_confidence_from_eval(scenarios[0].rule_eval)
        expected = rule_conf
        adjusted, adjustments = _apply_confirmation_confidence_adjustments(
            expected,
            scenarios[0].pivot_confirmations,
            analyzer.config,
        )
        assert scenarios[0].base_confidence == pytest.approx(expected)
        assert scenarios[0].confidence == pytest.approx(adjusted)
        assert scenarios[0].confidence_adjustments == adjustments

    def test_apply_confirmation_confidence_adjustments_caps_unconfirmed_terminal(self):
        cfg = ElliottWaveConfig(
            unconfirmed_terminal_pivot_confidence_cap=0.72,
        )

        adjusted, adjustments = _apply_confirmation_confidence_adjustments(
            0.98,
            [True, True, True, False],
            cfg,
        )

        assert adjusted == pytest.approx(0.98 * 0.72)
        assert adjustments["confirmation_factor"] == pytest.approx(0.72)

    def test_analyze_once_classifies_with_prefix_only(self, monkeypatch):
        close = np.array([100.0, 120.0, 108.0, 150.0, 135.0, 160.0, 148.0], dtype=float)
        t = np.arange(close.size, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(
            close,
            t,
            ElliottWaveConfig(
                min_distance=1,
                wave_min_len=1,
                min_impulse_bars=1,
                enable_gmm_classifier=True,
                min_confidence=0.0,
                pattern_types=["impulse"],
                include_fallback_candidate=False,
            ),
        )

        monkeypatch.setattr(
            elliott_mod,
            "_zigzag_pivots_indices",
            lambda *_args, **_kwargs: ([0, 1, 2, 3, 4, 5, 6], ["down", "up", "down", "up", "down", "up", "down"]),
        )
        monkeypatch.setattr(
            elliott_mod,
            "_evaluate_impulse_rules",
            lambda *args, **kwargs: ElliottRuleEvaluation(valid=True, fib_score=0.5, metrics={}, violations=[]),
        )

        seen_feature_lengths = []

        def _fake_classify(features, config):
            _ = config
            seen_feature_lengths.append(int(features.shape[0]))
            probs = np.full((features.shape[0], 2), 0.75, dtype=float)
            fake_gmm = type("FakeGMM", (), {"means_": np.array([[0.0, -0.1, 0.0], [0.0, 0.1, 0.0]])})()
            return np.zeros(features.shape[0], dtype=int), fake_gmm, None, probs, 1

        monkeypatch.setattr(elliott_mod, "_classify_waves", _fake_classify)

        scenarios = analyzer.analyze_once(1.0, 1)

        assert scenarios
        assert seen_feature_lengths == [5, 6]


# ===== _window_hit =========================================================

class TestWindowHit:
    def test_inside(self):
        assert _window_hit(0.5, 0.3, 0.7) == 1.0

    def test_below_tapered(self):
        score = _window_hit(0.28, 0.3, 0.7)
        assert 0.0 < score < 1.0

    def test_above_tapered(self):
        score = _window_hit(0.72, 0.3, 0.7)
        assert 0.0 < score < 1.0

    def test_zero_range(self):
        assert _window_hit(0.5, 0.5, 0.5) == 0.0

    def test_far_below(self):
        assert _window_hit(-10.0, 0.3, 0.7) == 0.0

    def test_far_above(self):
        assert _window_hit(10.0, 0.3, 0.7) == 0.0


# ===== _normalize_pattern_types ============================================

class TestNormalizePatternTypes:
    def test_default(self):
        cfg = ElliottWaveConfig()
        assert _normalize_pattern_types(cfg) == {"impulse", "correction"}

    def test_impulses_alias(self):
        cfg = ElliottWaveConfig(pattern_types=["impulses"])
        assert "impulse" in _normalize_pattern_types(cfg)

    def test_abc_alias(self):
        cfg = ElliottWaveConfig(pattern_types=["abc"])
        assert "correction" in _normalize_pattern_types(cfg)

    def test_empty_falls_back(self):
        cfg = ElliottWaveConfig(pattern_types=[])
        assert _normalize_pattern_types(cfg) == {"impulse", "correction"}

    def test_unknown_ignored(self):
        cfg = ElliottWaveConfig(pattern_types=["unknown"])
        # Falls back to default since no valid entries
        assert _normalize_pattern_types(cfg) == {"impulse", "correction"}
