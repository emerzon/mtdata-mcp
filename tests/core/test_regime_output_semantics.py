from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.core.regime import api as regime_api
from mtdata.core.regime import api as regime_mod
from mtdata.core.regime import detect as regime_detect_mod
from mtdata.core.regime.api import (
    _build_all_method_comparison,
    _consolidate_payload,
    regime_detect,
)
from mtdata.core.regime.payload import _build_regime_descriptions


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


@pytest.fixture(autouse=True)
def _skip_mt5_connection(monkeypatch):
    monkeypatch.setattr(regime_mod, "ensure_mt5_connection_or_raise", lambda: None)


def _downtrend_df(n: int = 120) -> pd.DataFrame:
    t = np.arange(float(n))
    close = np.linspace(100.0, 70.0, n)
    return pd.DataFrame({"time": t, "close": close})


def _choppy_bearish_df(n: int = 120) -> pd.DataFrame:
    t = np.arange(float(n))
    alternating = np.where(np.arange(n) % 2 == 0, 2.0, -2.0)
    close = 100.0 + alternating + np.linspace(0.0, -10.0, n)
    return pd.DataFrame({"time": t, "close": close})


def _flat_df(n: int = 120) -> pd.DataFrame:
    t = np.arange(float(n))
    close = np.full(n, 100.0)
    return pd.DataFrame({"time": t, "close": close})


def test_build_all_method_comparison_uses_semantic_signals() -> None:
    comparison = _build_all_method_comparison(
        {
            "bocpd": {
                "current_regime": {
                    "status": "no_recent_change_detected",
                    "since": "T1",
                    "bars_since_change": 500,
                    "transition_risk": "low",
                },
                "transition_summary": {
                    "recent_change_points_count": 0,
                    "recent_transition_activity": "none",
                    "calibration_status": "calibrated",
                },
                "regime_context": {
                    "bias": "bullish",
                    "return_pct": 4.2,
                    "volatility_pct": 1.1,
                },
            },
            "hmm": {
                "current_regime": {
                    "regime_id": 1,
                    "label": "positive_mod_vol",
                    "regime_confidence": 0.857,
                },
                "regime_info": {
                    1: {
                        "label": "positive_mod_vol",
                        "mean_return": 0.000122,
                        "mean_return_pct": 0.0122,
                        "volatility": 0.00279,
                        "volatility_pct": 0.279,
                    }
                },
            },
            "clustering": {
                "current_regime": {
                    "regime_id": 1,
                    "label": "regime_bullish",
                    "regime_confidence": 1.0,
                }
            },
            "garch": {
                "current_regime": {
                    "regime_id": 0,
                    "label": "low_vol",
                    "regime_confidence": 1.0,
                },
                "regime_info": {
                    0: {
                        "label": "low_vol",
                        "mean_return": 0.000105,
                        "mean_return_pct": 0.0105,
                        "volatility": 0.00367,
                        "volatility_pct": 0.367,
                    }
                },
            },
            "rule_based": {
                "current_regime": {
                    "state": "ranging",
                    "direction": "bullish",
                    "trend_strength": 0.1497,
                    "efficiency_ratio": 0.0009,
                    "window_bars": 160,
                    "window_move_pct": 1.25,
                    "signal_source": "price",
                }
            },
            "ensemble": {
                "current_regime": {
                    "regime_id": 1,
                    "label": "negative_mod_vol",
                    "regime_confidence": 0.3524,
                },
                "regime_info": {
                    1: {
                        "label": "negative_mod_vol",
                        "mean_return": -0.000415,
                        "mean_return_pct": -0.0415,
                        "volatility": 0.00386,
                        "volatility_pct": 0.386,
                    }
                },
            },
        }
    )

    assert comparison["current_regimes"]["bocpd"]["status"] == "no_recent_change_detected"
    assert comparison["current_regimes"]["bocpd"]["recent_transition_activity"] == "none"
    assert comparison["current_regimes"]["bocpd"]["bias"] == "bullish"
    assert comparison["current_regimes"]["hmm"]["regime_confidence"] == 0.857
    assert comparison["current_regimes"]["garch"]["volatility"] == "low_vol"
    assert comparison["current_regimes"]["rule_based"]["signal_source"] == "price"
    assert "majority_state" not in comparison["agreement"]
    assert comparison["agreement"]["direction"] == {
        "majority": "bullish",
        "agreement_pct": 75.0,
        "methods_considered": ["hmm", "clustering", "rule_based", "ensemble"],
    }
    assert comparison["agreement"]["volatility"] == {
        "majority": "moderate_vol",
        "agreement_pct": 66.67,
        "methods_considered": ["hmm", "garch", "ensemble"],
    }


def test_regime_detect_all_respects_full_and_summary_detail(monkeypatch) -> None:
    real = _unwrap(regime_api.regime_detect)
    monkeypatch.setattr(regime_api, "_fetch_history", lambda *args, **kwargs: _downtrend_df(120))
    monkeypatch.setattr(regime_api, "mt5_connection_error", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        regime_detect_mod,
        "_build_all_method_comparison",
        lambda results: {
            "methods_run": sorted(results.keys()),
            "agreement": {"basis": "test"},
        },
    )

    subcall_details: list[tuple[str, str, bool]] = []

    def fake_recursive(*args, **kwargs):
        method_name = str(kwargs.get("method") or "")
        detail_name = str(kwargs.get("output") or kwargs.get("detail") or "")
        include_series = bool(kwargs.get("include_series"))
        subcall_details.append((method_name, detail_name, include_series))
        result = {
            "success": True,
            "symbol": kwargs.get("symbol"),
            "timeframe": kwargs.get("timeframe"),
            "method": method_name,
            "target": kwargs.get("target"),
        }
        if detail_name == "full":
            result["detail_marker"] = f"{method_name}-full"
            if include_series:
                result["series"] = {
                    "state": [0] * 119,
                    "state_probabilities": [[1.0, 0.0]] * 119,
                }
        return result

    monkeypatch.setattr(regime_detect_mod, "_run_regime_method", fake_recursive)

    full = real("EURUSD", method="all", detail="full", include_series=True)
    assert full["detail"] == "full"
    assert "results" in full
    assert full["results"]["bocpd"]["detail_marker"] == "bocpd-full"
    assert full["results"]["hmm"]["series"]["state"] == [0] * 119
    assert subcall_details
    assert all(detail == "full" for _, detail, _ in subcall_details)
    assert all(include_series is True for _, _, include_series in subcall_details)
    assert not any(method == "ensemble" for method, _, _ in subcall_details)
    assert full["ensemble_health"]["used_voters"] == [
        "hmm",
        "gmm",
        "ms_ar",
        "clustering",
        "wavelet",
    ]
    assert full["ensemble_health"]["aggregation_source"] == (
        "reused_all_first_pass"
    )

    subcall_details.clear()
    summary = real("EURUSD", method="all", detail="summary", include_series=True)
    assert summary["detail"] == "summary"
    assert "results" not in summary
    # Summary mode must not embed per-method regimes or params_used.
    assert "params_used" not in summary
    assert summary["summary"]["methods_succeeded"] == summary["summary"]["methods_attempted"]
    assert summary["summary"]["ensemble_aggregated"] is True
    assert "ensemble" in summary["runtime"]["completed_methods"]
    assert summary["runtime"]["ensemble_aggregated"] is True
    assert summary["summary"]["agreement"] == {"basis": "test"}
    comparison = summary.get("comparison", {})
    assert "current_regimes" not in comparison
    assert "agreement" not in comparison
    assert subcall_details
    assert not any(method == "ensemble" for method, _, _ in subcall_details)
    for method_name, detail_name, include_series in subcall_details:
        if method_name in regime_api._ENSEMBLE_STATE_METHODS:
            assert detail_name == "full"
            assert include_series is True
        else:
            assert detail_name == "compact"
            assert include_series is False


def test_regime_detect_all_reports_runtime_diagnostics_for_partial_results(monkeypatch) -> None:
    real = _unwrap(regime_api.regime_detect)
    monkeypatch.setattr(regime_api, "_fetch_history", lambda *args, **kwargs: _downtrend_df(120))
    monkeypatch.setattr(regime_api, "mt5_connection_error", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        regime_detect_mod,
        "_build_all_method_comparison",
        lambda results: {"methods_run": sorted(results.keys())},
    )

    def fake_recursive(*args, **kwargs):
        method_name = str(kwargs.get("method") or "")
        if method_name == "ms_ar":
            return {"error": "simulated slow fit timeout"}
        result = {
            "success": True,
            "symbol": kwargs.get("symbol"),
            "timeframe": kwargs.get("timeframe"),
            "method": method_name,
            "target": kwargs.get("target"),
            "current_regime": {"label": "neutral"},
        }
        if method_name in regime_api._ENSEMBLE_STATE_METHODS:
            result["series"] = {
                "state": [0] * 119,
                "state_probabilities": [[1.0, 0.0]] * 119,
            }
        return result

    monkeypatch.setattr(regime_detect_mod, "_run_regime_method", fake_recursive)

    result = real("EURUSD", method="all", detail="compact")

    assert result["success"] is True
    assert "results" not in result
    assert "params_used" not in result
    assert result["runtime"]["method_execution_status"] == "partial"
    assert result["runtime"]["ensemble_status"] == "degraded"
    assert result["runtime"]["request_completion_status"] == "partial_method_results"
    assert "partial_results" not in result["runtime"]
    assert "ms_ar" in result["runtime"]["failed_methods"]
    assert result["runtime"]["method_errors"]["ms_ar"] == "simulated slow fit timeout"
    assert "method_durations_ms" not in result["runtime"]
    assert "method_guidance" not in result["runtime"]
    assert "suggested_faster_methods" not in result["runtime"]
    assert "current_regimes" not in result["comparison"]
    assert result["summary"]["methods_succeeded"] < result["summary"]["methods_attempted"]
    assert result["summary"]["methods_failed"] == 1
    assert result["runtime"]["ensemble_degraded"] is True
    assert result["runtime"]["ensemble_voters"]["excluded_voters"] == ["ms_ar"]


def test_regime_detect_all_excludes_unidentifiable_voter(monkeypatch) -> None:
    real = _unwrap(regime_api.regime_detect)
    monkeypatch.setattr(
        regime_api,
        "_fetch_history",
        lambda *args, **kwargs: _downtrend_df(120),
    )
    monkeypatch.setattr(regime_api, "mt5_connection_error", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        regime_detect_mod,
        "_build_all_method_comparison",
        lambda results: {"methods_run": sorted(results.keys())},
    )

    def fake_recursive(*args, **kwargs):
        method_name = str(kwargs.get("method") or "")
        result = {
            "success": True,
            "symbol": kwargs.get("symbol"),
            "timeframe": kwargs.get("timeframe"),
            "method": method_name,
            "target": kwargs.get("target"),
            "current_regime": {"label": "neutral"},
        }
        if method_name in regime_api._ENSEMBLE_STATE_METHODS:
            result["series"] = {
                "state": [0] * 119,
                "state_probabilities": [[1.0, 0.0]] * 119,
            }
        if method_name == "hmm":
            result.update(
                {
                    "status": "unidentifiable",
                    "effective_n_states": 1,
                    "current_regime": {
                        "label": "unidentifiable",
                        "regime_confidence": 0.0,
                        "label_quality": "unidentifiable_state_collapse",
                    },
                }
            )
        return result

    monkeypatch.setattr(regime_detect_mod, "_run_regime_method", fake_recursive)

    result = real("EURUSD", method="all", detail="full")

    assert "hmm" not in result["ensemble_health"]["used_voters"]
    assert "hmm" in result["ensemble_health"]["excluded_voters"]
    assert "unidentifiable state collapse" in result["ensemble_health"][
        "voter_errors"
    ]["hmm"]
    assert result["ensemble_health"]["degraded"] is True


def test_regime_detect_all_pads_short_ms_ar_voter(monkeypatch) -> None:
    real = _unwrap(regime_api.regime_detect)
    monkeypatch.setattr(
        regime_api,
        "_fetch_history",
        lambda *args, **kwargs: _downtrend_df(120),
    )
    monkeypatch.setattr(regime_api, "mt5_connection_error", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        regime_detect_mod,
        "_build_all_method_comparison",
        lambda results: {"methods_run": sorted(results.keys())},
    )
    calls: list[str] = []

    def fake_recursive(*args, **kwargs):
        method_name = str(kwargs.get("method") or "")
        calls.append(method_name)
        result = {
            "success": True,
            "symbol": kwargs.get("symbol"),
            "timeframe": kwargs.get("timeframe"),
            "method": method_name,
            "target": kwargs.get("target"),
            "current_regime": {"label": "neutral"},
        }
        if method_name in regime_api._ENSEMBLE_STATE_METHODS:
            length = 118 if method_name == "ms_ar" else 119
            result["series"] = {
                "state": [0] * length,
                "state_probabilities": [[1.0, 0.0]] * length,
            }
        return result

    monkeypatch.setattr(regime_detect_mod, "_run_regime_method", fake_recursive)

    result = real("EURUSD", method="all", detail="compact")

    assert len(calls) == 9
    assert "ensemble" not in calls
    assert result["runtime"]["failed_methods"] == []
    assert "ms_ar" in result["runtime"]["completed_methods"]
    assert result["runtime"]["method_execution_status"] == "complete"
    assert result["runtime"]["ensemble_status"] == "complete"
    assert result["runtime"]["request_completion_status"] == "complete"
    assert "partial_results" not in result["runtime"]
    assert result["runtime"]["ensemble_degraded"] is False
    health = result["ensemble_health"]
    assert "ms_ar" in health["used_voters"]
    assert "ms_ar" not in health["excluded_voters"]
    assert result["summary"]["methods_failed"] == 0


def test_regime_detect_all_excludes_wildly_short_ms_ar_voter(monkeypatch) -> None:
    real = _unwrap(regime_api.regime_detect)
    monkeypatch.setattr(
        regime_api,
        "_fetch_history",
        lambda *args, **kwargs: _downtrend_df(120),
    )
    monkeypatch.setattr(regime_api, "mt5_connection_error", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        regime_detect_mod,
        "_build_all_method_comparison",
        lambda results: {"methods_run": sorted(results.keys())},
    )

    def fake_recursive(*args, **kwargs):
        method_name = str(kwargs.get("method") or "")
        result = {
            "success": True,
            "symbol": kwargs.get("symbol"),
            "timeframe": kwargs.get("timeframe"),
            "method": method_name,
            "target": kwargs.get("target"),
            "current_regime": {"label": "neutral"},
        }
        if method_name in regime_api._ENSEMBLE_STATE_METHODS:
            length = 50 if method_name == "ms_ar" else 119
            result["series"] = {
                "state": [0] * length,
                "state_probabilities": [[1.0, 0.0]] * length,
            }
        return result

    monkeypatch.setattr(regime_detect_mod, "_run_regime_method", fake_recursive)

    result = real("EURUSD", method="all", detail="compact")

    health = result["ensemble_health"]
    assert health["excluded_voters"] == ["ms_ar"]
    assert "state series did not match" in health["voter_errors"]["ms_ar"]
    assert result["runtime"]["ensemble_status"] == "degraded"


def test_ensemble_labels_require_significant_mean_return() -> None:
    descriptions = _build_regime_descriptions(
        {
            "mean_return": [
                -0.000725,
                -0.000415,
                0.000079,
                0.000179,
                0.000607,
                0.00119,
            ],
            "volatility": [0.00435, 0.00386, 0.00466, 0.00614, 0.00469, 0.00593],
        },
        "ensemble",
        state_counts={index: 1000 for index in range(6)},
    )

    assert descriptions[0]["label"].startswith("negative_")
    assert descriptions[1]["label"].startswith("negative_")
    assert descriptions[2]["label"].startswith("neutral_")
    assert descriptions[3]["label"].startswith("neutral_")
    assert descriptions[4]["label"].startswith("positive_")
    assert "bearish" not in descriptions[2]["label"]


def test_regime_volatility_labels_are_scale_invariant() -> None:
    low_scale = _build_regime_descriptions(
        {
            "mean_return": [-0.0002, 0.0002],
            "volatility": [0.0001, 0.0004],
        },
        "hmm",
        state_counts={0: 100, 1: 100},
    )
    high_scale = _build_regime_descriptions(
        {
            "mean_return": [-0.0002, 0.0002],
            "volatility": [0.01, 0.04],
        },
        "hmm",
        state_counts={0: 100, 1: 100},
    )

    assert [low_scale[i]["stat_label"] for i in range(2)] == [
        "negative_low_vol",
        "positive_high_vol",
    ]
    assert [high_scale[i]["stat_label"] for i in range(2)] == [
        "neutral_low_vol",
        "neutral_high_vol",
    ]
    assert low_scale[0]["label"] == "bearish_quiet"
    assert high_scale[1]["label"] == "volatile_range"
    assert low_scale[0]["direction_significant"] is True
    assert high_scale[0]["direction_significant"] is False


def test_single_regime_uses_neutral_relative_volatility_tier() -> None:
    descriptions = _build_regime_descriptions(
        {"mean_return": [0.0], "volatility": [0.25]},
        "ensemble",
    )

    assert descriptions[0]["label"] == "neutral_mod_vol"


def test_wavelet_labels_include_frequency_character() -> None:
    descriptions = _build_regime_descriptions(
        {
            "mean_return": [-0.00025, -0.00016, 0.00055],
            "volatility": [0.0043, 0.00398, 0.00503],
            "energy_profiles": {
                "0": {
                    "band_0_energy": 0.084377,
                    "band_1_energy": 0.212702,
                    "band_2_energy": 0.271006,
                    "band_3_energy": 0.431915,
                },
                "1": {
                    "band_0_energy": 0.081442,
                    "band_1_energy": 0.114511,
                    "band_2_energy": 0.196177,
                    "band_3_energy": 0.60787,
                },
                "2": {
                    "band_0_energy": 0.037996,
                    "band_1_energy": 0.081256,
                    "band_2_energy": 0.419504,
                    "band_3_energy": 0.461244,
                },
            },
        },
        "wavelet",
        state_counts={0: 500, 1: 500, 2: 500},
    )

    assert descriptions[0]["label"] == "neutral_mixed_freq_moderate_vol"
    assert descriptions[1]["label"] == "neutral_trend_dominant_low_vol"
    assert descriptions[2]["label"] == "positive_trend_dominant_high_vol"


def test_bocpd_consolidation_uses_segment_language() -> None:
    out = _consolidate_payload(
        {
            "symbol": "X",
            "timeframe": "H1",
            "method": "bocpd",
            "target": "return",
            "times": ["T1", "T2", "T3", "T4"],
            "cp_prob": [0.05, 0.9, 0.1, 0.08],
            "change_points": [{"idx": 1, "time": "T2", "prob": 0.9}],
            "_series_values": [0.01, 0.02, -0.01, -0.02],
            "threshold": 0.5,
            "reliability": {
                "expected_false_alarm_rate": 0.02,
                "recent_cp_density": 0.0,
                "threshold_calibrated": True,
            },
            "summary": {
                "lookback": 4,
                "last_cp_prob": 0.08,
                "max_cp_prob": 0.9,
                "mean_cp_prob": 0.2825,
                "change_points_count": 1,
                "raw_change_points_count": 2,
                "accepted_change_points_count": 1,
                "rejected_change_points_count": 1,
            },
        },
        "bocpd",
        "compact",
    )

    assert "current_regime" in out
    assert out["current_regime"]["status"] == "recent_change_detected"
    assert out["transition_summary"]["recent_change_points_count"] == 1
    assert out["transition_summary"]["max_transition_probability"] == 0.9
    assert out["transition_summary"]["mean_transition_probability"] == 0.2825
    assert out["transition_summary"]["raw_change_points_count"] == 2
    assert out["transition_summary"]["accepted_change_points_count"] == 1
    assert out["transition_summary"]["rejected_change_points_count"] == 1
    assert (
        out["transition_summary"]["raw_change_points_count"]
        == out["transition_summary"]["accepted_change_points_count"]
        + out["transition_summary"]["rejected_change_points_count"]
    )
    assert "filtered_change_points_count" not in out["transition_summary"]
    assert out["transition_summary"]["calibration_status"] == "calibrated"
    assert out["regime_context"]["source"] == "derived_from_return_series"
    assert out["regimes"][1]["transition_prob_at_start"] == 0.9
    assert "current_segment" not in out
    assert "segments" not in out


def test_consolidate_payload_uses_regime_confidence_name() -> None:
    out = _consolidate_payload(
        {
            "symbol": "X",
            "timeframe": "H1",
            "method": "hmm",
            "times": ["T1", "T2", "T3", "T4"],
            "state": [0, 0, 1, 1],
            "state_probabilities": [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.15, 0.85],
                [0.2, 0.8],
            ],
        },
        "hmm",
        "compact",
    )

    current_regime = out["current_regime"]
    assert current_regime["regime_confidence"] == 0.825
    assert "confidence" not in current_regime
    assert "regime_id" not in current_regime


def test_rule_based_uses_price_window_metrics_for_return_target() -> None:
    raw = _unwrap(regime_detect)
    history = _downtrend_df()
    expected_segment = history["close"].to_numpy()[-60:]
    expected_move_pct = round(
        (expected_segment[-1] - expected_segment[0]) / expected_segment[0] * 100.0,
        4,
    )

    with (
        patch("mtdata.core.regime.api._fetch_history", return_value=history),
        patch("mtdata.core.regime.api.resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime.api._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.core.regime.detect._format_time_minimal", side_effect=lambda x: f"T{x}"),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=len(history),
            method="rule_based",
            target="return",
            params={"window_bars": 60},
            detail="full",
        )

    regime = out["regime"]
    assert regime["direction"] == "bearish"
    assert regime["signal_source"] == "price"
    assert regime["window_move_pct"] == expected_move_pct
    assert out["current_regime"]["label"] == "trending"
    assert "state" not in out["current_regime"]
    assert out["params_used"]["signal_source"] == "price"
    assert out["requested_target"] == "return"
    assert out["effective_target"] == "price"
    assert out["target"] == "price"
    assert any("normalized" in warning for warning in out["warnings"])


def test_rule_based_exposes_classification_window_without_segment_claims() -> None:
    raw = _unwrap(regime_detect)

    with (
        patch("mtdata.core.regime.api._fetch_history", return_value=_downtrend_df(100)),
        patch("mtdata.core.regime.api.resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime.api._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.core.regime.detect._format_time_minimal", side_effect=lambda x: f"T{x}"),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=100,
            method="rule_based",
            target="return",
            params={"window_bars": 100},
            detail="full",
            include_series=True,
        )

    assert out["classification_window"] == {
        "start": "T0.0",
        "end": "T99.0",
        "bars": 100,
        "basis": "aggregate_price_window",
    }
    assert out["current_regime"]["classification_scope"] == "aggregate_window"
    assert out["current_regime"]["boundary_status"] == "not_estimated"
    assert out["current_regime"]["persistence_status"] == "not_estimated"
    assert "since" not in out["current_regime"]
    assert "bars" not in out["current_regime"]
    assert "regimes" not in out
    assert "total_regimes" not in out
    assert "series" not in out
    assert "include_series is not available for rule_based" in "\n".join(
        out["warnings"]
    )


def test_rule_based_rejects_limit_below_explicit_window() -> None:
    raw = _unwrap(regime_detect)
    out = raw(
        symbol="TEST",
        timeframe="H1",
        fetch_limit=100,
        method="rule_based",
        params={"window_bars": 160},
        detail="full",
    )

    assert out["error"] == (
        "fetch_limit (100) must be greater than or equal to params.window_bars (160) "
        "for method='rule_based'."
    )


@pytest.mark.parametrize("fetch_limit", [20, 160, 200])
def test_rule_based_explicit_fetch_limit_controls_default_window(
    fetch_limit: int,
) -> None:
    raw = _unwrap(regime_detect)
    captured: dict[str, int] = {}

    def fake_fetch_history(_symbol: str, _timeframe: str, fetch_limit: int, *, as_of=None):
        captured["limit"] = fetch_limit
        return _downtrend_df(fetch_limit)

    with (
        patch("mtdata.core.regime.api._fetch_history", side_effect=fake_fetch_history),
        patch("mtdata.core.regime.api.resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime.api._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.core.regime.detect._format_time_minimal", side_effect=lambda x: f"T{x}"),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
        fetch_limit=fetch_limit,
            method="rule_based",
            detail="full",
        )

    assert captured["limit"] == fetch_limit
    assert out["regime"]["window_bars"] == fetch_limit
    assert out["params_used"]["window_bars"] == fetch_limit


def test_rule_based_rejects_limit_below_minimum_window() -> None:
    raw = _unwrap(regime_detect)
    out = raw(symbol="TEST", timeframe="H1", fetch_limit=1, method="rule_based")

    assert out["error"].startswith("fetch_limit must be >= 20")
    assert out["error_code"] == "cli_invalid_arguments"


def test_regime_detect_rejects_negative_fetch_limit_before_fetch() -> None:
    raw = _unwrap(regime_detect)
    called = {"count": 0}

    def fake_fetch_history(*_args, **_kwargs):
        called["count"] += 1
        raise AssertionError("fetch should not run for invalid fetch_limit")

    with patch("mtdata.core.regime.api._fetch_history", side_effect=fake_fetch_history):
        out = raw(symbol="TEST", timeframe="H1", fetch_limit=-1, method="hmm")

    assert called["count"] == 0
    assert out["success"] is False
    assert out["error_code"] == "cli_invalid_arguments"
    assert "negative" in out["error"].lower()


def test_regime_detect_rejects_insufficient_hmm_fetch_limit() -> None:
    raw = _unwrap(regime_detect)
    with patch("mtdata.core.regime.api._fetch_history") as mocked_fetch:
        out = raw(symbol="TEST", timeframe="H1", fetch_limit=5, method="hmm")

    mocked_fetch.assert_not_called()
    assert out["error_code"] == "cli_invalid_arguments"
    assert "fetch_limit must be >= 10" in out["error"]


def test_rule_based_rejects_lookback_below_minimum_window() -> None:
    raw = _unwrap(regime_detect)
    out = raw(symbol="TEST", timeframe="H1", lookback=1, method="rule_based")

    assert out["error"].startswith("--lookback must be >= 20")
    assert "params.window_bars" not in out["error"]


@pytest.mark.parametrize(
    ("params", "expected_error"),
    [
        ({"window_bars": "bad"}, "params.window_bars must be an integer >= 20."),
        ({"window_bars": 19}, "params.window_bars must be >= 20."),
        (
            {"efficiency_threshold": "bad"},
            "params.efficiency_threshold must be a positive number.",
        ),
        ({"efficiency_threshold": 0}, "params.efficiency_threshold must be > 0."),
        ({"efficiency_threshold": 2}, "params.efficiency_threshold must be <= 1."),
        (
            {"trend_strength_threshold": "bad"},
            "params.trend_strength_threshold must be a positive number.",
        ),
        (
            {"trend_strength_threshold": -1},
            "params.trend_strength_threshold must be > 0.",
        ),
    ],
)
def test_rule_based_rejects_invalid_params(params: dict, expected_error: str) -> None:
    raw = _unwrap(regime_detect)

    out = raw(
        symbol="TEST",
        timeframe="H1",
        method="rule_based",
        params=params,
    )

    assert out["error"] == expected_error


def test_rule_based_ranging_confidence_uses_ranging_evidence() -> None:
    raw = _unwrap(regime_detect)

    with (
        patch("mtdata.core.regime.api._fetch_history", return_value=_flat_df()),
        patch("mtdata.core.regime.api.resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime.api._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.core.regime.detect._format_time_minimal", side_effect=lambda x: f"T{x}"),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            method="rule_based",
            params={"window_bars": 60},
            detail="full",
        )

    assert out["regime"]["state"] == "ranging"
    assert out["regime"]["efficiency_ratio"] == 0.0
    assert out["current_regime"]["regime_confidence"] == 1.0
    assert out["reliability"]["confidence"] == 1.0


def test_rule_based_trending_confidence_tracks_threshold_margin() -> None:
    raw = _unwrap(regime_detect)

    def detect_with(params: dict) -> dict:
        with (
            patch("mtdata.core.regime.api._fetch_history", return_value=_downtrend_df(160)),
            patch("mtdata.core.regime.api.resolve_denoise_base_col", return_value="close"),
            patch("mtdata.core.regime.api._format_time_minimal", side_effect=lambda x: f"T{x}"),
            patch("mtdata.core.regime.detect._format_time_minimal", side_effect=lambda x: f"T{x}"),
        ):
            return raw(
                symbol="TEST",
                timeframe="H1",
                method="rule_based",
                params={"window_bars": 160, **params},
                detail="full",
            )

    marginal = detect_with(
        {"efficiency_threshold": 0.95, "trend_strength_threshold": 3.3}
    )
    strong = detect_with(
        {"efficiency_threshold": 0.35, "trend_strength_threshold": 1.25}
    )

    assert marginal["regime"]["state"] == "trending"
    assert strong["regime"]["state"] == "trending"
    assert 0.0 <= marginal["current_regime"]["regime_confidence"] < 1.0
    assert (
        marginal["current_regime"]["regime_confidence"]
        < strong["current_regime"]["regime_confidence"]
        < 1.0
    )


def test_rule_based_explains_ranging_direction_bias() -> None:
    raw = _unwrap(regime_detect)

    with (
        patch("mtdata.core.regime.api._fetch_history", return_value=_choppy_bearish_df()),
        patch("mtdata.core.regime.api.resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime.api._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.core.regime.detect._format_time_minimal", side_effect=lambda x: f"T{x}"),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=120,
            method="rule_based",
            params={"window_bars": 60},
            detail="full",
        )

    regime = out["regime"]
    assert regime["state"] == "ranging"
    assert "direction" not in regime
    assert regime["window_bias"] == "bearish"
    assert regime["direction_basis"] == "net_window_move"
    assert "window bias, not a trend classification" in regime["interpretation"]
    assert regime["trend_strength"] >= out["params_used"]["trend_strength_threshold"]
    assert "efficiency_ratio indicates a choppy path" in regime["note"]
    assert out["current_regime"]["label"] == "ranging"
    assert "state" not in out["current_regime"]


def test_rule_based_compact_explains_direction_bias() -> None:
    raw = _unwrap(regime_detect)

    with (
        patch("mtdata.core.regime.api._fetch_history", return_value=_choppy_bearish_df()),
        patch("mtdata.core.regime.api.resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime.api._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.core.regime.detect._format_time_minimal", side_effect=lambda x: f"T{x}"),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=120,
            method="rule_based",
            params={"window_bars": 60},
        )

    current_regime = out["current_regime"]
    assert out["signal_status"] == "not_actionable"
    assert out["classification_window"]["bars"] == 60
    assert "bars" not in current_regime
    assert "since" not in current_regime
    assert current_regime["label"] == "ranging"
    assert "direction" not in current_regime
    assert current_regime["window_bias"] == "bearish"
    assert current_regime["direction_role"] == "window_bias_not_trend"
    assert "headline" not in current_regime
    assert "state_label_native" not in current_regime
    assert "state_label_canonical" not in current_regime
    assert current_regime["regime_confidence"] == pytest.approx(0.8793)
    assert "state" not in current_regime
    assert "efficiency_ratio" in current_regime
    assert "trend_strength" in current_regime
    assert "window_move_pct" in current_regime
    assert current_regime["direction_basis"] == "net_window_move"
    assert "window bias, not a trend classification" in current_regime["interpretation"]
    assert "efficiency_ratio indicates a choppy path" in current_regime["note"]
    assert "regime" not in out
    assert "regimes" not in out
    assert "regime_info" not in out
    assert "reliability" not in out
    assert "total_regimes" not in out
    assert "params_used" not in out


def test_rule_based_summary_explains_direction_bias() -> None:
    raw = _unwrap(regime_detect)

    with (
        patch("mtdata.core.regime.api._fetch_history", return_value=_choppy_bearish_df()),
        patch("mtdata.core.regime.api.resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime.api._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.core.regime.detect._format_time_minimal", side_effect=lambda x: f"T{x}"),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=120,
            method="rule_based",
            params={"window_bars": 60},
            detail="summary",
        )

    summary = out["summary"]
    assert summary["label"] == "ranging"
    assert "direction" not in summary
    assert summary["window_bias"] == "bearish"
    assert summary["direction_basis"] == "net_window_move"
    assert "window bias, not a trend classification" in summary["interpretation"]
    assert out["classification_window"]["bars"] == 60
    assert out["classification_window"]["basis"] == "aggregate_price_window"


def test_rule_based_warns_for_inapplicable_parameters() -> None:
    raw = _unwrap(regime_detect)

    with (
        patch("mtdata.core.regime.api._fetch_history", return_value=_choppy_bearish_df()),
        patch("mtdata.core.regime.api.resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime.api._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.core.regime.detect._format_time_minimal", side_effect=lambda x: f"T{x}"),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=120,
            method="rule_based",
            lookback=50,
            min_regime_bars=5,
            max_regimes=2,
            detail="full",
        )

    warnings = "\n".join(out.get("warnings", []))
    assert "lookback was ignored for rule_based because params.window_bars was also provided" not in warnings
    assert "min_regime_bars is not used by rule_based" in warnings
    assert "max_regimes has no effect for rule_based" in warnings

    rejected = raw(
        symbol="TEST",
        timeframe="H1",
        fetch_limit=120,
        method="rule_based",
        threshold=0.9,
        lookback=50,
        detail="full",
    )
    assert rejected.get("success") is False
    assert rejected.get("error_code") == "cli_invalid_arguments"
    assert "bocpd" in str(rejected.get("error") or "").lower()


def test_regime_detect_all_uses_same_rule_based_window_as_standalone() -> None:
    real = _unwrap(regime_detect)
    history = _choppy_bearish_df(120)

    with (
        patch("mtdata.core.regime.api._fetch_history", return_value=history),
        patch("mtdata.core.regime.api.resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime.api._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.core.regime.detect._format_time_minimal", side_effect=lambda x: f"T{x}"),
    ):
        standalone = real(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=120,
            method="rule_based",
            lookback=50,
        )

    captured: dict[str, object] = {}

    def fake_recursive(*args, **kwargs):
        method_name = str(kwargs.get("method") or "")
        result = {
            "success": True,
            "symbol": kwargs.get("symbol"),
            "timeframe": kwargs.get("timeframe"),
            "method": method_name,
            "target": kwargs.get("target"),
            "current_regime": {"label": "neutral"},
        }
        if method_name == "rule_based":
            config = kwargs.get("rule_based_config") or {}
            captured["rule_based_config"] = dict(config)
            window_bars = int(config.get("window_bars") or 160)
            result["params_used"] = {"window_bars": window_bars}
            result["classification_window"] = {"bars": window_bars}
            result["current_regime"] = {
                "label": "ranging",
                "window_bars": window_bars,
            }
        if method_name in regime_api._ENSEMBLE_STATE_METHODS:
            result["series"] = {
                "state": [0] * 119,
                "state_probabilities": [[1.0, 0.0]] * 119,
            }
        return result

    with (
        patch("mtdata.core.regime.api._fetch_history", return_value=history),
        patch("mtdata.core.regime.api.mt5_connection_error", return_value=None),
        patch("mtdata.core.regime.detect._run_regime_method", side_effect=fake_recursive),
    ):
        all_out = real(
            symbol="TEST",
            timeframe="H1",
            method="all",
            lookback=50,
            fetch_limit=120,
            detail="compact",
        )

    assert standalone["classification_window"]["bars"] == 50
    assert captured["rule_based_config"]["window_bars"] == 50
    assert all_out["comparison"]["method_windows"]["rule_based"] == 50
    assert captured["rule_based_config"]["window_bars"] == standalone["classification_window"]["bars"]


def test_rule_based_lookback_controls_window_when_window_bars_omitted() -> None:
    raw = _unwrap(regime_detect)

    with (
        patch("mtdata.core.regime.api._fetch_history", return_value=_choppy_bearish_df()),
        patch("mtdata.core.regime.api.resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime.api._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.core.regime.detect._format_time_minimal", side_effect=lambda x: f"T{x}"),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=120,
            method="rule_based",
            lookback=50,
    )

    current_regime = out["current_regime"]
    assert out["classification_window"]["bars"] == 50
    assert "bars" not in current_regime
    assert "since" not in current_regime
    assert out["data_quality"]["status"] == "limited_history"
    assert out["data_quality"]["lookback_too_short"] is True
    assert out["data_quality"]["recommended_min_bars"] == 160
    assert "lookback is not used by rule_based" not in "\n".join(out.get("warnings", []))


def test_gmm_reports_distinct_method_and_common_reliability() -> None:
    raw = _unwrap(regime_detect)
    history = _downtrend_df(40)
    gamma = np.tile(np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float), (20, 1))[:39]
    weights = np.array([0.5, 0.5])
    mu = np.array([-0.001, 0.001])
    sigma = np.array([0.0004, 0.002])

    with (
        patch("mtdata.core.regime.api._fetch_history", return_value=history),
        patch("mtdata.core.regime.api.resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime.api._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.core.regime.detect._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch(
            "mtdata.core.regime.detect.fit_gaussian_mixture_1d",
            return_value=(weights, mu, sigma, gamma, None),
            create=True,
        ),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=len(history),
            method="gmm",
            params={"n_states": 2},
            detail="full",
        )

    assert out["method"] == "gmm"
    assert "requested_method" not in out
    assert "method_effective" not in out
    assert out["reliability"]["reliability_label"] in {
        "high",
        "medium",
        "low",
        "very_low",
    }
    assert out["reliability"]["source"] == "gmm_component_responsibilities"


def test_ensemble_rejects_bocpd_change_point_votes() -> None:
    raw = _unwrap(regime_detect)
    n = 40
    history = pd.DataFrame(
        {
            "time": np.arange(float(n)),
            "close": np.linspace(100.0, 120.0, n),
        }
    )
    with (
        patch("mtdata.core.regime.api._fetch_history", return_value=history),
        patch("mtdata.core.regime.api.resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime.api._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.core.regime.detect._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.core.regime.detect._run_regime_method") as call_tool,
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=n,
            method="ensemble",
            target="price",
            params={"methods": ["bocpd"], "n_states": 2},
            threshold=0.5,
            detail="full",
            include_series=True,
            min_regime_bars=1,
        )

    assert out["error_code"] == "invalid_ensemble_methods"
    assert "Unsupported: bocpd" in out["error"]
    call_tool.assert_not_called()


def test_ensemble_discloses_kurtosis_state_count_heuristic() -> None:
    raw = _unwrap(regime_detect)
    n = 40
    history = pd.DataFrame(
        {
            "time": np.arange(float(n)),
            "close": np.linspace(100.0, 120.0, n),
        }
    )
    ref_len = n - 1

    def fake_call_tool(**kwargs):
        n_states = int(kwargs["p"]["n_states"])
        return {
            "success": True,
            "method": "hmm",
            "series": {
                "state": [0] * ref_len,
                "state_probabilities": [
                    [1.0] + [0.0] * (n_states - 1) for _ in range(ref_len)
                ],
            },
        }

    with (
        patch("mtdata.core.regime.api._fetch_history", return_value=history),
        patch("mtdata.core.regime.api.resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime.api._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.core.regime.detect._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.core.regime.ensemble._finite_raw_kurtosis", return_value=4.0),
        patch(
            "mtdata.core.regime.detect._run_regime_method",
            side_effect=fake_call_tool,
        ),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=n,
            method="ensemble",
            params={"methods": ["hmm"]},
            detail="full",
            include_series=True,
            min_regime_bars=1,
        )

    assert out["params_used"]["n_states"] == 4
    assert out["params_used"]["state_count_param"] == "return_kurtosis_heuristic"
    assert out["params_used"]["state_count_heuristic"] == {
        "method": "return_kurtosis_thresholds",
        "returns_kurtosis": 4.0,
    }
    assert any("not statistical model selection" in warning for warning in out["warnings"])


def test_ensemble_keeps_invalid_leading_submethod_rows_undefined() -> None:
    raw = _unwrap(regime_detect)
    n = 40
    history = pd.DataFrame(
        {
            "time": np.arange(float(n)),
            "close": np.linspace(100.0, 120.0, n),
        }
    )
    ref_len = n - 1
    states = [-1, -1] + [1] * (ref_len - 2)
    probs = [[0.0, 0.0], [0.0, 0.0]] + [[0.1, 0.9]] * (ref_len - 2)

    def fake_call_tool(**_kwargs):
        return {
            "success": True,
            "method": "hmm",
            "series": {
                "state": states,
                "state_probabilities": probs,
            },
        }

    with (
        patch("mtdata.core.regime.api._fetch_history", return_value=history),
        patch("mtdata.core.regime.api.resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime.api._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.core.regime.detect._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.core.regime.detect._run_regime_method", side_effect=fake_call_tool),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=n,
            method="ensemble",
            params={"methods": ["hmm"], "n_states": 2},
            detail="full",
            include_series=True,
            min_regime_bars=1,
        )

    assert out["series"]["state"][:2] == [-1, -1]
    assert out["series"]["state"][2] in {0, 1}
    assert out["regimes"][0]["start"] == "T3.0"


def test_garch_rejects_price_target() -> None:
    raw = _unwrap(regime_detect)

    with (
        patch("mtdata.core.regime.api._fetch_history", return_value=_downtrend_df(80)),
        patch("mtdata.core.regime.api.resolve_denoise_base_col", return_value="close"),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=80,
            method="garch",
            target="price",
        )

    assert out["error_code"] == "invalid_target"
    assert "target='return'" in out["error"]


def test_garch_tier_thresholds_report_relative_and_absolute_scope() -> None:
    from mtdata.core.regime.api import _garch_tier_thresholds

    values = np.arange(1.0, 10.0)
    relative, relative_scope = _garch_tier_thresholds(values, 3, None)
    absolute, absolute_scope = _garch_tier_thresholds(values, 2, 4.25)

    assert relative == pytest.approx([3.6666666667, 6.3333333333])
    assert relative_scope == "full_window_percentiles"
    assert absolute == [4.25]
    assert absolute_scope == "explicit_absolute"


def test_wavelet_rejects_non_positive_energy_window() -> None:
    raw = _unwrap(regime_detect)

    with (
        patch("mtdata.core.regime.api._fetch_history", return_value=_downtrend_df(80)),
        patch("mtdata.core.regime.api.resolve_denoise_base_col", return_value="close"),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=80,
            method="wavelet",
            params={"energy_window": 0},
        )

    assert out["error"] == "params.energy_window must be a positive integer."
