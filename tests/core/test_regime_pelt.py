from __future__ import annotations

import numpy as np
import pandas as pd

from mtdata.core.regime import api as regime
from mtdata.core.regime.api import _pelt_return_direction
from mtdata.core.regime.detect import _pelt_adjusted_separation_confidence


def _raw_regime_detect():
    fn = regime.regime_detect
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_pelt_direction_requires_statistically_significant_mean() -> None:
    noisy = np.array([0.001, -0.001, 0.0011, -0.0009, 0.0001] * 20)

    direction, mean_t_stat, significant = _pelt_return_direction(
        noisy,
        float(np.mean(noisy)),
    )

    assert direction == "neutral"
    assert mean_t_stat is not None
    assert significant is False


def test_pelt_direction_keeps_significant_drift() -> None:
    trending = np.array([0.0010, 0.0012, 0.0008, 0.0011, 0.0009] * 20)

    direction, mean_t_stat, significant = _pelt_return_direction(
        trending,
        float(np.mean(trending)),
    )

    assert direction == "positive"
    assert mean_t_stat is not None and mean_t_stat > 1.96
    assert significant is True


def test_pelt_detects_structural_break(monkeypatch):
    rng = np.random.default_rng(123)
    returns = np.concatenate(
        [rng.normal(-0.004, 0.001, 120), rng.normal(0.005, 0.001, 120)]
    )
    close = 100.0 * np.exp(np.cumsum(returns))
    frame = pd.DataFrame(
        {
            "time": np.arange(1_700_000_000, 1_700_000_000 + len(close) * 3600, 3600),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "tick_volume": 100,
            "real_volume": 0,
        }
    )
    monkeypatch.setattr(regime, "mt5_connection_error", lambda *args, **kwargs: None)
    monkeypatch.setattr(regime, "_fetch_history", lambda *args, **kwargs: frame)

    result = _raw_regime_detect()(
        symbol="TEST",
        timeframe="H1",
        fetch_limit=len(frame),
        method="pelt",
        target="return",
        params={"penalty": "auto", "min_size": 20},
        detail="full",
    )

    assert result["success"] is True
    assert result["method"] == "pelt"
    assert result["summary"]["change_points_count"] >= 1
    assert len(result["regimes"]) >= 2
    assert result["params_used"]["penalty_source"] == "bic_like_auto_l2"
    assert all(row["direction_significant"] for row in result["regimes"])
    assert all("regime_confidence" in row for row in result["regimes"])


def test_pelt_adjusted_confidence_penalizes_oversegmentation() -> None:
    rng = np.random.default_rng(4)
    values = np.concatenate(
        [rng.normal(0.0, 0.001, 300), rng.normal(0.002, 0.001, 300)]
    )

    correct = _pelt_adjusted_separation_confidence(values, [300, 600])
    oversegmented = _pelt_adjusted_separation_confidence(
        values,
        list(range(5, 601, 5)),
    )

    assert correct > oversegmented


def test_pelt_rbf_auto_penalty_does_not_saturate_minimum_segments(monkeypatch):
    rng = np.random.default_rng(3)
    returns = np.concatenate(
        [
            rng.normal(0.0, 0.001, 300),
            rng.normal(0.0, 0.004, 300),
            rng.normal(0.0, 0.001, 300),
        ]
    )
    close = 100.0 * np.exp(np.r_[0.0, np.cumsum(returns)])
    frame = pd.DataFrame(
        {
            "time": np.arange(
                1_700_000_000,
                1_700_000_000 + len(close) * 3600,
                3600,
            ),
            "close": close,
        }
    )
    monkeypatch.setattr(regime, "mt5_connection_error", lambda *args, **kwargs: None)
    monkeypatch.setattr(regime, "_fetch_history", lambda *args, **kwargs: frame)

    result = _raw_regime_detect()(
        symbol="TEST",
        timeframe="H1",
        fetch_limit=len(frame),
        method="pelt",
        target="return",
        params={"model": "rbf", "penalty": "auto", "min_size": 5},
        detail="full",
    )

    assert result["success"] is True
    assert result["params_used"]["penalty_source"] == "bic_like_auto_rbf"
    assert result["summary"]["segments"] < 20


def test_pelt_lookback_bounds_analyzed_window(monkeypatch):
    returns = np.linspace(-0.002, 0.002, 69)
    close = 100.0 * np.exp(np.r_[0.0, np.cumsum(returns)])
    frame = pd.DataFrame(
        {
            "time": np.arange(1_700_000_000, 1_700_000_000 + len(close) * 3600, 3600),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "tick_volume": 100,
            "real_volume": 0,
        }
    )
    monkeypatch.setattr(regime, "mt5_connection_error", lambda *args, **kwargs: None)
    monkeypatch.setattr(regime, "_fetch_history", lambda *args, **kwargs: frame)

    result = _raw_regime_detect()(
        symbol="TEST",
        timeframe="H1",
        method="pelt",
        target="return",
        lookback=50,
        params={"penalty": "auto", "min_size": 5},
        detail="full",
    )

    assert result["success"] is True
    assert sum(segment["bars"] for segment in result["regimes"]) == 50
    assert result["analysis_window"] == {
        "bars_fetched": 70,
        "warmup_bars": 19,
        "bars_analyzed": 50,
        "analysis_limit": 50,
    }


def test_pelt_explicit_range_without_cap_analyzes_full_window(monkeypatch):
    returns = np.linspace(-0.002, 0.002, 239)
    close = 100.0 * np.exp(np.r_[0.0, np.cumsum(returns)])
    frame = pd.DataFrame(
        {
            "time": np.arange(
                1_700_000_000,
                1_700_000_000 + len(close) * 3600,
                3600,
            ),
            "close": close,
        }
    )
    monkeypatch.setattr(regime, "mt5_connection_error", lambda *args, **kwargs: None)
    monkeypatch.setattr(regime, "_fetch_history", lambda *args, **kwargs: frame)

    result = _raw_regime_detect()(
        symbol="TEST",
        timeframe="H1",
        start="2025-01-01",
        end="2025-02-01",
        method="pelt",
        target="return",
        params={"penalty": "auto", "min_size": 5},
        detail="full",
    )

    assert result["success"] is True
    assert result["analysis_window"]["bars_analyzed"] == 239
    assert result["analysis_window"]["analysis_limit"] == 239
    assert result["analysis_window"]["truncated"] is False
    assert result["analysis_window"]["fetch_limit_applied"] is None
