from __future__ import annotations

import numpy as np
import pandas as pd

import mtdata.core.causal as causal
from mtdata.core.causal import cointegration, cross


def _raw(tool):
    return getattr(tool, "__wrapped__", tool)


def test_cross_correlation_identifies_first_symbol_lead(monkeypatch):
    rng = np.random.default_rng(4)
    left = rng.normal(size=300)
    right = np.concatenate([np.zeros(3), left[:-3]])
    index = pd.date_range("2025-01-01", periods=300, freq="h")
    series = {
        "LEFT": pd.Series(left, index=index),
        "RIGHT": pd.Series(right, index=index),
    }
    monkeypatch.setattr(cross, "_causal_connection_error", lambda: None)
    monkeypatch.setattr(
        cross,
        "_fetch_series_for_window",
        lambda symbol, *args, **kwargs: (series[symbol], None),
    )

    result = _raw(causal.cross_correlation)(
        symbols="LEFT,RIGHT",
        transform="level",
        max_lag=8,
        min_overlap=50,
        bootstrap_samples=50,
    )

    assert result["success"] is True
    assert result["best"]["lag"] == 3
    assert result["best"]["leader"] == "LEFT"
    assert result["best_nonzero"]["lag"] == 3
    assert result["best_nonzero"]["leader"] is None
    assert result["best_nonzero"]["follower"] is None
    assert result["best_nonzero"]["inference_valid"] is False
    assert result["zero_lag"]["lag"] == 0
    assert result["best"]["inference_valid"] is False
    assert "significant" not in result["best"]
    assert "ci95_low" not in result["best"]
    assert result["context"]["lag_tests"] == 17
    assert result["context"]["significance_correction"] == "bonferroni_across_lags"
    assert result["context"]["ci_per_lag_confidence"] > 0.95
    assert result["context"]["alignment_ok"] is True
    assert result["context"]["aligned_fraction"] == 1.0
    assert any("price-level" in warning for warning in result["warnings"])


def test_cross_correlation_warns_when_symbol_sessions_have_low_overlap(monkeypatch):
    left_index = pd.date_range("2025-01-01", periods=100, freq="h")
    right_index = pd.date_range("2025-01-02 01:00", periods=100, freq="h")
    series = {
        "LEFT": pd.Series(np.arange(100, dtype=float), index=left_index),
        "RIGHT": pd.Series(np.arange(100, dtype=float), index=right_index),
    }
    monkeypatch.setattr(cross, "_causal_connection_error", lambda: None)
    monkeypatch.setattr(
        cross,
        "_fetch_series_for_window",
        lambda symbol, *args, **kwargs: (series[symbol], None),
    )

    result = _raw(causal.cross_correlation)(
        symbols="LEFT,RIGHT",
        transform="level",
        max_lag=2,
        min_overlap=50,
        bootstrap_samples=50,
    )

    assert result["success"] is True
    assert result["context"]["samples_available_by_symbol"] == {
        "LEFT": 100,
        "RIGHT": 100,
    }
    assert result["context"]["samples_raw_aligned"] == 75
    assert result["context"]["aligned_fraction"] == 0.75
    assert result["context"]["alignment_loss_pct"] == 25.0
    assert result["context"]["alignment_ok"] is False
    assert "Session-calendar gaps" in result["warnings"][0]


def test_cross_correlation_measures_alignment_against_longer_series(monkeypatch):
    left_index = pd.date_range("2025-01-01", periods=100, freq="h")
    right_index = left_index[:80]
    series = {
        "LEFT": pd.Series(np.arange(100, dtype=float), index=left_index),
        "RIGHT": pd.Series(np.arange(80, dtype=float), index=right_index),
    }
    monkeypatch.setattr(cross, "_causal_connection_error", lambda: None)
    monkeypatch.setattr(
        cross,
        "_fetch_series_for_window",
        lambda symbol, *args, **kwargs: (series[symbol], None),
    )

    result = _raw(causal.cross_correlation)(
        symbols="LEFT,RIGHT",
        transform="level",
        max_lag=2,
        min_overlap=50,
        bootstrap_samples=50,
    )

    assert result["success"] is True
    assert result["context"]["samples_available_by_symbol"] == {
        "LEFT": 100,
        "RIGHT": 80,
    }
    assert result["context"]["samples_raw_aligned"] == 80
    assert result["context"]["aligned_fraction"] == 0.8
    assert result["context"]["alignment_loss_pct"] == 20.0
    assert result["context"]["alignment_ok"] is False
    assert "larger input series" in result["warnings"][0]


def test_cross_correlation_adjusts_selected_lag_interval(monkeypatch):
    index = pd.date_range("2025-01-01", periods=80, freq="h")
    series = {
        "LEFT": pd.Series(np.arange(80, dtype=float) + 100.0, index=index),
        "RIGHT": pd.Series(np.arange(80, dtype=float) + 100.0, index=index),
    }
    observed: dict[str, float] = {}

    monkeypatch.setattr(cross, "_causal_connection_error", lambda: None)
    monkeypatch.setattr(
        cross,
        "_fetch_series_for_window",
        lambda symbol, *args, **kwargs: (series[symbol], None),
    )

    def _ci(*args, confidence, **kwargs):
        observed["confidence"] = confidence
        return -0.01, 0.01

    monkeypatch.setattr(cross, "_block_bootstrap_correlation_ci", _ci)
    result = _raw(causal.cross_correlation)(
        symbols="LEFT,RIGHT",
        transform="log_return",
        max_lag=2,
        min_overlap=20,
    )

    assert result["success"] is True
    assert observed["confidence"] == 0.99
    assert result["best"]["ci95_low"] == -0.01
    assert result["best"]["ci95_high"] == 0.01
    assert "ci_low" not in result["best"]
    assert "ci_high" not in result["best"]
    assert result["best"]["significant"] is False


def test_cross_correlation_reports_resolved_aligned_window(monkeypatch):
    index = pd.date_range("2025-03-01", periods=10, freq="h")
    series = {
        "LEFT": pd.Series(np.arange(10, dtype=float) + 100.0, index=index),
        "RIGHT": pd.Series(np.arange(10, dtype=float) + 50.0, index=index),
    }
    monkeypatch.setattr(cross, "_causal_connection_error", lambda: None)
    monkeypatch.setattr(
        cross,
        "_fetch_series_for_window",
        lambda symbol, *args, **kwargs: (series[symbol], None),
    )

    result = _raw(causal.cross_correlation)(
        symbols="LEFT,RIGHT",
        transform="level",
        max_lag=2,
        min_overlap=5,
        window_bars=10,
        bootstrap_samples=50,
    )

    assert result["success"] is True
    assert result["context"]["period_start"] == "2025-03-01T00:00Z"
    assert result["context"]["period_end"] == "2025-03-01T09:00Z"
    assert result["context"]["samples_aligned"] == 10


def test_cross_correlation_exposes_zero_lag_and_best_nonzero(monkeypatch):
    rng = np.random.default_rng(7)
    left = rng.normal(size=300)
    right = 0.85 * left + 0.15 * np.concatenate([np.zeros(2), left[:-2]])
    index = pd.date_range("2025-01-01", periods=300, freq="h")
    series = {
        "LEFT": pd.Series(left, index=index),
        "RIGHT": pd.Series(right, index=index),
    }
    monkeypatch.setattr(cross, "_causal_connection_error", lambda: None)
    monkeypatch.setattr(
        cross,
        "_fetch_series_for_window",
        lambda symbol, *args, **kwargs: (series[symbol], None),
    )

    result = _raw(causal.cross_correlation)(
        symbols="LEFT,RIGHT",
        transform="level",
        max_lag=8,
        min_overlap=50,
        bootstrap_samples=50,
    )

    assert result["success"] is True
    assert result["best"]["lag"] == 0
    assert result["best"]["leader"] is None
    assert result["zero_lag"]["lag"] == 0
    assert result["zero_lag"]["correlation"] == result["best"]["correlation"]
    assert result["best_nonzero"]["lag"] != 0
    assert result["best_nonzero"]["leader"] is None
    assert result["best_nonzero"]["follower"] is None
    assert result["best_nonzero"]["inference_valid"] is False


def test_cointegration_johansen_reports_positive_rank(monkeypatch):
    rng = np.random.default_rng(8)
    base = np.cumsum(rng.normal(size=500)) + 100.0
    linked = 1.5 * base + rng.normal(scale=0.2, size=500)
    index = pd.date_range("2024-01-01", periods=500, freq="h")
    series = {
        "AAA": pd.Series(base, index=index),
        "BBB": pd.Series(linked, index=index),
    }
    monkeypatch.setattr(cointegration, "_causal_connection_error", lambda: None)
    monkeypatch.setattr(
        cointegration,
        "_fetch_series_for_window",
        lambda symbol, *args, **kwargs: (series[symbol], None),
    )

    result = _raw(causal.cointegration_test)(
        symbols="AAA,BBB",
        method="johansen",
        transform="level",
        min_overlap=100,
        window_bars=400,
    )

    assert result["success"] is True
    assert result["method"] == "johansen"
    assert result["cointegration_rank"] >= 1
    assert result["cointegrating_vectors"]


def test_cointegration_corrects_significance_across_tested_pairs(monkeypatch):
    index = pd.date_range("2025-01-01", periods=120, freq="h")
    series = {
        symbol: pd.Series(np.linspace(100.0 + offset, 120.0 + offset, 120), index=index)
        for offset, symbol in enumerate(("AAA", "BBB", "CCC"))
    }
    p_values = iter((0.02, 0.03, 0.9))

    monkeypatch.setattr(cointegration, "_causal_connection_error", lambda: None)
    monkeypatch.setattr(
        cointegration,
        "_fetch_series_for_window",
        lambda symbol, *args, **kwargs: (series[symbol], None),
    )
    monkeypatch.setattr(
        "statsmodels.tsa.stattools.coint",
        lambda *args, **kwargs: (-4.0, next(p_values), [-3.9, -3.3, -3.0]),
    )

    result = _raw(causal.cointegration_test)(
        symbols="AAA,BBB,CCC",
        transform="level",
        min_overlap=80,
        significance=0.05,
    )

    assert result["success"] is True
    assert result["summary"]["counts"]["cointegrated"] == 0
    assert [item["p_value_raw"] for item in result["items"]] == [0.02, 0.03, 0.9]
    assert [item["p_value"] for item in result["items"]] == [0.06, 0.06, 0.9]
    assert all(item["p_value_correction"] == "holm_across_pairs" for item in result["items"])
