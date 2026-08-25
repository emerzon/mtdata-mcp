"""Tests for core/regime.py — pure consolidation helpers (no MT5)."""
import numpy as np
import pytest

from mtdata.core.regime.api import (
    _apply_bocpd_output_mode,
    _consolidate_payload,
    _smoothing_warnings,
    _summary_only_payload,
)
from mtdata.core.regime.smoothing import (
    _confirm_state_changes_causally,
    _hard_state_probability_matrix,
)


def test_causal_state_confirmation_never_rewrites_prefix() -> None:
    raw = np.array([0, 0, 1, 1, 0, 1, 1, 1], dtype=int)
    full, meta = _confirm_state_changes_causally(raw, 3)
    prefix, _ = _confirm_state_changes_causally(raw[:6], 3)
    assert full[:6].tolist() == prefix.tolist()
    assert full.tolist() == [0, 0, 0, 0, 0, 0, 0, 1]
    assert meta["postprocess"] == "causal_confirmation"
    assert meta["confirmation_semantics"] == (
        "consecutive_raw_state_evidence_before_transition"
    )
    assert meta["pending_state"] is None
    assert meta["pending_bars"] == 0


def test_causal_state_confirmation_reports_pending_terminal_candidate() -> None:
    emitted, meta = _confirm_state_changes_causally(
        np.array([0, 0, 1, 1], dtype=int),
        3,
    )

    assert emitted.tolist() == [0, 0, 0, 0]
    assert meta["pending_state"] == 1
    assert meta["pending_bars"] == 2
    assert meta["pending_bars_required"] == 3
    assert _smoothing_warnings("hmm", meta) == [
        "method='hmm' has candidate state 1 pending confirmation (2/3 required "
        "consecutive bars); the current emitted regime is retained."
    ]


def test_bocpd_compact_lookback_preserves_true_segment_age() -> None:
    times = [f"T{i}" for i in range(200)]
    cp_prob = np.zeros(200, dtype=float)
    cp_prob[80] = 0.9
    change_points = [{"idx": 80, "time": "T80", "prob": 0.9}]
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "timeframe": "H1",
        "method": "bocpd",
        "target": "return",
        "times": times,
        "cp_prob": cp_prob.tolist(),
        "change_points": change_points,
        "_series_values": np.zeros(200, dtype=float).tolist(),
        "threshold": 0.5,
    }

    compact_payload = _apply_bocpd_output_mode(
        payload,
        output="compact",
        lookback=50,
        cp_prob=cp_prob,
        change_points=change_points,
        raw_cp_idx=[80],
        reliability={},
        expected_fa_rate=0.02,
        calibration_age_bars=200,
        tuning_hint=None,
    )
    result = _consolidate_payload(compact_payload, "bocpd", "compact")

    assert len(compact_payload["times"]) == 200
    assert result["current_regime"]["since"] == "T80"
    assert result["current_regime"]["bars_since_change"] == 120
    assert result["current_regime"]["status"] == "post_change_segment"
    assert result["transition_summary"]["recent_change_points_count"] == 0
    assert result["total_regimes"] == 2


def test_hard_probabilities_follow_causally_confirmed_state() -> None:
    raw = np.array([0, 0, 1, 1, 1, 0, 0], dtype=int)
    emitted, _ = _confirm_state_changes_causally(raw, 3)
    probs = _hard_state_probability_matrix(emitted, 2)

    assert emitted.tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert np.all(probs[np.arange(emitted.size), emitted] == 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


class TestSummaryOnlyPayload:
    def test_basic(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "bocpd",
            "target": "return",
            "success": True,
            "summary": {"regimes_found": 3},
            "params_used": {"hazard_lambda": 250},
            "threshold": 0.5,
            "times": [1, 2, 3],
            "cp_prob": [0.1, 0.8, 0.1],
        }
        result = _summary_only_payload(payload)
        assert result["symbol"] == "EURUSD"
        assert result["method"] == "bocpd"
        assert result["success"] is True
        assert "summary" in result
        assert "params_used" in result
        assert "threshold" in result
        # Should NOT include raw data
        assert "times" not in result
        assert "cp_prob" not in result

    def test_minimal(self):
        payload = {"symbol": "X", "method": "hmm"}
        result = _summary_only_payload(payload)
        assert result["symbol"] == "X"
        assert result["success"] is True
        assert "summary" not in result


class TestConsolidatePayload:
    def test_with_regimes(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "bocpd",
            "success": True,
            "summary": {"total": 3},
            "times": [1.0, 2.0, 3.0, 4.0, 5.0],
            "state": [0, 0, 1, 1, 0],
            "cp_prob": [0.1, 0.1, 0.9, 0.1, 0.8],
        }
        result = _consolidate_payload(payload, "bocpd", "full", include_series=True)
        assert result["symbol"] == "EURUSD"
        assert result["success"] is True
        assert "regimes" in result
        assert "current_regime" in result
        assert "transition_summary" in result

    def test_compact_mode(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "bocpd",
            "success": True,
            "times": [1.0, 2.0, 3.0],
            "state": [0, 0, 1],
            "cp_prob": [0.1, 0.1, 0.9],
        }
        result = _consolidate_payload(payload, "bocpd", "compact", include_series=True)
        assert "regimes" in result
        assert "current_regime" in result
        assert result["has_more"] is False

    def test_compact_keeps_lookback_summary(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "hmm",
            "success": True,
            "times": [1.0, 2.0, 3.0, 4.0],
            "state": [0, 0, 1, 1],
            "summary": {
                "lookback": 20,
                "last_state": 1,
                "state_shares": {0: 0.25, 1: 0.75},
                "transitions_after": 1,
                "smoothing_applied": True,
            },
        }
        result = _consolidate_payload(payload, "hmm", "compact")
        assert result["summary"] == {
            "lookback": 20,
            "last_state": 1,
            "state_shares": {0: 0.25, 1: 0.75},
            "transitions_after": 1,
        }

    def test_no_series_when_not_requested(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "bocpd",
            "success": True,
            "times": [1.0, 2.0],
            "state": [0, 1],
            "cp_prob": [0.1, 0.9],
        }
        result = _consolidate_payload(payload, "bocpd", "full", include_series=False)
        assert "series" not in result

    def test_params_preserved(self):
        payload = {
            "symbol": "X",
            "method": "hmm",
            "success": True,
            "params_used": {"n_states": 2},
            "times": [1.0],
            "state": [0],
        }
        result = _consolidate_payload(payload, "hmm", "full")
        assert result.get("params_used") == {"n_states": 2}

    def test_garch_hard_tiers_do_not_claim_probabilistic_confidence(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "garch",
            "success": True,
            "times": [1.0, 2.0, 3.0],
            "state": [0, 0, 1],
            "state_probabilities": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            "params_used": {"n_states": 2},
        }

        result = _consolidate_payload(payload, "garch", "full")

        assert result["regime_confidence_status"] == (
            "not_applicable_hard_volatility_tiers"
        )
        assert "regime_confidence" not in result["current_regime"]
        assert all("regime_confidence" not in row for row in result["regimes"])

    def test_bocpd_regimes_are_canonicalized_by_series_mean(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "bocpd",
            "success": True,
            "times": [1.0, 2.0, 3.0, 4.0],
            "cp_prob": [0.1, 0.1, 0.9, 0.1],
            "change_points": [{"idx": 2, "time": 3.0, "prob": 0.9}],
            "_series_values": [0.4, 0.5, -0.2, -0.1],
            "params_used": {},
        }

        result = _consolidate_payload(payload, "bocpd", "full")

        assert [regime["bias"] for regime in result["regimes"]] == [
            "bullish",
            "bearish",
        ]
        assert result["regimes"][0]["state_label_native"] == "bullish_segment"
        assert result["regimes"][0]["state_label_canonical"] == "trending_up"
        assert result["units"]["regime_context.return_pct"] == "percent"
        assert result["units"]["scale_note"] == "Percent values use 1.0 = 1%."
        assert result["params_used"]["relabeled"] is True
        assert result["params_used"]["label_mapping"] == {"0": 1, "1": 0}

    def test_all_invalid_states_are_dropped_from_consolidated_regimes(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "clustering",
            "success": True,
            "times": [1.0, 2.0, 3.0],
            "state": [-1, -1, -1],
            "state_probabilities": [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        }

        result = _consolidate_payload(payload, "clustering", "full")

        assert result["success"] is True
        assert result["regimes"] == []

    def test_hmm_regime_info_uses_canonical_label_mapping(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "hmm",
            "target": "return",
            "success": True,
            "times": [f"T{i}" for i in range(100)],
            "state": [0] * 50 + [1] * 50,
            "regime_params": {
                "weights": [0.6, 0.4],
                "mu": [0.002, -0.001],
                "sigma": [0.0035, 0.0004],
            },
            "params_used": {
                "relabeled": True,
                "label_mapping": {"0": 1, "1": 0},
            },
        }

        result = _consolidate_payload(payload, "hmm", "full")

        assert result["regimes"][0]["label"] == "bearish_quiet"
        assert result["regimes"][1]["label"] == "bullish_volatile"
        assert result["regimes"][0]["state_label_native"] == "bearish_quiet"
        assert result["regimes"][0]["state_label_canonical"] == "trending_down"
        assert result["current_regime"]["state_label_canonical"] == "trending_up"
        assert result["regime_info"][0]["label"] == "bearish_quiet"
        assert result["regime_info"][0]["stat_label"] == "negative_low_vol"
        assert "downward drift" in result["regime_info"][0]["trading_interpretation"]
        assert result["regime_info"][1]["stat_label"] == "positive_high_vol"
        assert result["regime_info"][0]["mean_return"] == pytest.approx(-0.001)
        assert result["regime_info"][0]["volatility"] == pytest.approx(0.0004)
        assert result["regime_info"][1]["mean_return"] == pytest.approx(0.002)
        assert result["regime_info"][1]["volatility"] == pytest.approx(0.0035)
        assert result["regime_info"][0]["observed_in_window"] is True
        assert result["regime_info"][1]["observed_in_window"] is True

    def test_msar_regime_info_does_not_remap_canonical_parameters_twice(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "ms_ar",
            "target": "return",
            "success": True,
            "times": [f"T{i}" for i in range(100)],
            "state": [0] * 50 + [1] * 50,
            "regime_params": {
                "mean_return": [-0.002, 0.001],
                "volatility": [0.003, 0.0005],
            },
            "params_used": {
                "relabeled": True,
                "label_mapping": {"0": 1, "1": 0},
            },
        }

        result = _consolidate_payload(payload, "ms_ar", "full")

        assert result["regime_info"][0]["mean_return"] == pytest.approx(-0.002)
        assert result["regime_info"][1]["mean_return"] == pytest.approx(0.001)
        assert result["regimes"][0]["label"].startswith("negative_")
        assert result["regimes"][1]["label"].startswith("positive_")

    def test_compact_regime_history_reports_has_more_and_hint(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "hmm",
            "target": "return",
            "success": True,
            "times": ["T1", "T2", "T3", "T4", "T5"],
            "state": [0, 1, 0, 1, 0],
            "state_probabilities": [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.9, 0.1],
            ],
            "regime_params": {
                "weights": [0.6, 0.4],
                "mu": [0.002, -0.001],
                "sigma": [0.0035, 0.0004],
            },
        }

        result = _consolidate_payload(payload, "hmm", "compact", max_regimes=2)

        assert result["has_more"] is True
        assert result["total_regimes"] == 5
        assert result["showing_regimes"] == 2
        assert len(result["regimes"]) == 2
        assert result["regimes"][0]["start"] == "T4"
        assert result["last_transition"]["at"] == "T5"
        assert "older regime segment(s) omitted" in result["history_hint"]
        assert "label" in result["current_regime"]

    def test_standard_hmm_output_caps_regime_rows(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "hmm",
            "target": "return",
            "success": True,
            "times": ["T1", "T2", "T3", "T4", "T5"],
            "state": [0, 1, 0, 1, 0],
            "state_probabilities": [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.9, 0.1],
            ],
            "regime_params": {
                "weights": [0.6, 0.4],
                "mu": [0.002, -0.001],
                "sigma": [0.0035, 0.0004],
            },
        }

        result = _consolidate_payload(payload, "hmm", "standard", max_regimes=2)

        assert result["total_regimes"] == 5
        assert len(result["regimes"]) == 2
        assert result["regimes_truncated"] is True

    def test_hmm_regime_info_marks_unobserved_model_states(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "hmm",
            "target": "return",
            "success": True,
            "times": ["T1", "T2"],
            "state": [0, 0],
            "state_probabilities": [[0.95, 0.05], [0.90, 0.10]],
            "regime_params": {
                "weights": [0.7, 0.3],
                "mu": [-0.001, 0.002],
                "sigma": [0.0004, 0.0035],
            },
        }

        result = _consolidate_payload(payload, "hmm", "compact")

        assert result["regime_info"][0]["observed_in_window"] is True
        assert result["regime_info"][1]["observed_in_window"] is False
        assert "not observed" in result["regime_info"][1]["note"]
        assert "not necessarily the fraction" in result["regime_info"][1]["weight_note"]

    @pytest.mark.parametrize(
        ("method", "threshold_scope", "expected_scope"),
        [
            ("hmm", None, "retrospective_full_window_model_fit"),
            ("ms_ar", None, "retrospective_full_window_model_fit"),
            (
                "garch",
                "full_window_percentiles",
                "retrospective_full_window_fit_and_percentile_thresholds",
            ),
        ],
    )
    def test_compact_discloses_retrospective_historical_labels(
        self,
        method,
        threshold_scope,
        expected_scope,
    ):
        params_used = {"model_fit_scope": "full_window"}
        if threshold_scope is not None:
            params_used["threshold_scope"] = threshold_scope
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": method,
            "target": "return",
            "success": True,
            "times": ["T1", "T2"],
            "state": [0, 1],
            "state_probabilities": [[1.0, 0.0], [0.0, 1.0]],
            "params_used": params_used,
        }

        result = _consolidate_payload(payload, method, "compact")

        assert result["historical_labels_are_retrospective"] is True
        assert result["historical_label_scope"] == expected_scope
        assert "rolling --end" in result["point_in_time_guidance"]
        assert f"--method {method}" in result["point_in_time_guidance"]

    def test_compact_omits_retrospective_disclosure_for_online_labels(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "bocpd",
            "success": True,
            "times": ["T1", "T2"],
            "cp_prob": [0.1, 0.2],
            "params_used": {"model_fit_scope": "online"},
        }

        result = _consolidate_payload(payload, "bocpd", "compact")

        assert "historical_labels_are_retrospective" not in result
        assert "historical_label_scope" not in result

    def test_hmm_price_target_uses_price_level_metadata(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "hmm",
            "target": "price",
            "success": True,
            "times": ["T1", "T2"],
            "state": [0, 1],
            "state_probabilities": [[0.95, 0.05], [0.05, 0.95]],
            "regime_params": {
                "weights": [0.55, 0.45],
                "mu": [1.105, 1.255],
                "sigma": [0.01, 0.015],
            },
        }

        result = _consolidate_payload(payload, "hmm", "full")

        assert result["regimes"][0]["label"] == "low_price_regime"
        assert result["regimes"][1]["label"] == "high_price_regime"
        assert result["regime_info"][0]["mean_price"] == pytest.approx(1.105)
        assert result["regime_info"][0]["std_dev"] == pytest.approx(0.01)
        assert "mean_return" not in result["regime_info"][0]
        assert "volatility_pct" not in result["regime_info"][0]

    def test_clustering_regime_info_uses_descriptive_labels(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "clustering",
            "target": "return",
            "success": True,
            "times": [f"T{i}" for i in range(150)],
            "state": [0] * 50 + [1] * 50 + [2] * 50,
            "regime_params": {
                "mean_return": [-0.001, 0.0, 0.002],
                "volatility": [0.0004, 0.0015, 0.004],
            },
        }

        result = _consolidate_payload(payload, "clustering", "compact")

        labels = {entry["label"] for entry in result["regime_info"].values()}
        assert "regime_0" not in labels
        assert any(label.startswith("negative_") for label in labels)
        assert any(label.startswith("positive_") for label in labels)

    def test_ensemble_compact_regime_info_hides_unobserved_states(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "ensemble",
            "target": "return",
            "success": True,
            "times": ["T1", "T2"],
            "state": [0, 0],
            "state_probabilities": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "regime_params": {
                "mean_return": [-0.001, 0.0, 0.0],
                "volatility": [0.0004, 0.0, 0.0],
            },
        }

        result = _consolidate_payload(payload, "ensemble", "compact")

        assert set(result["regime_info"].keys()) == {0}


def test_bocpd_segment_context_labels_stationary_window_flat():
    """A long stationary-return window with small cumulative drift must be `flat`.

    Regression test: with the old 0.05% cumulative-return threshold, a 199-bar
    EURUSD H1 window with stationary log returns could be branded trending_down
    just because cumulative return happened to land at -0.67%. The new t-stat
    gate requires the mean return to be statistically distinguishable from zero.
    """
    from mtdata.core.regime.payload import _build_bocpd_segment_context

    rng = np.random.default_rng(42)
    # 199 stationary returns with small negative drift (not statistically
    # distinguishable from zero). Cumulative return ~ -0.67%.
    returns = rng.normal(loc=-0.0000338, scale=0.0006, size=199)
    ctx = _build_bocpd_segment_context(returns, target="return")
    assert ctx["bias"] == "flat"
    assert ctx["direction_significant"] is False


def test_bocpd_segment_context_labels_significant_trend():
    """A segment with statistically significant positive drift must be `bullish`."""
    from mtdata.core.regime.payload import _build_bocpd_segment_context

    rng = np.random.default_rng(7)
    returns = rng.normal(loc=0.002, scale=0.0005, size=50)
    ctx = _build_bocpd_segment_context(returns, target="return")
    assert ctx["bias"] == "bullish"
    assert ctx["direction_significant"] is True
    assert ctx["mean_t_stat"] >= 1.96


def test_bocpd_under_segmentation_warning_on_long_quiet_window_with_outlier():
    """A long single-segment window with a multi-sigma move must warn.

    Regression test for the Jun 5 EURUSD flash crash: BOCPD fit a single
    regime across 199 bars despite a 4-bar crash that other methods (PELT,
    MS-AR, clustering) all detected. The audit showed the crash was silently
    absorbed as a fat-tail outlier.
    """
    from mtdata.core.regime.api import _bocpd_under_segmentation_warnings

    rng = np.random.default_rng(42)
    calm = rng.normal(0.0, 0.0005, size=195)
    crash = np.array([-0.0047, -0.0013, -0.0012, -0.0023])
    series = np.concatenate([calm[:100], crash, calm[100:]])

    peak_z = float(np.max(np.abs(series)) / float(np.std(series)))
    warnings = _bocpd_under_segmentation_warnings(
        total_bars=int(series.size),
        change_point_count=0,
        reliability={"confidence": 0.4},
        peak_abs_return=peak_z,
    )
    assert any("under-segmentation" in w for w in warnings)
    assert any("σ single-bar move" in w for w in warnings)


def test_bocpd_under_segmentation_silent_on_short_windows():
    """Short windows with no CPs must NOT warn (single regime is unremarkable)."""
    from mtdata.core.regime.api import _bocpd_under_segmentation_warnings

    warnings = _bocpd_under_segmentation_warnings(
        total_bars=50,
        change_point_count=0,
        reliability={"confidence": 0.4},
        peak_abs_return=5.0,
    )
    assert warnings == []
