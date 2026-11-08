from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast import forecast_registry as registry
from mtdata.forecast.interface import ForecastResult
from mtdata.forecast.methods.ensemble import EnsembleMethod


@pytest.fixture
def components(monkeypatch):
    def install(predictors):
        calls = []

        def get(name):
            def forecast(series, horizon, seasonality, params):
                calls.append((name, len(series), dict(params)))
                return ForecastResult(
                    forecast=predictors[name](series, horizon, params),
                )

            return SimpleNamespace(forecast=forecast)

        monkeypatch.setattr(registry.ForecastRegistry, "get", get)
        monkeypatch.setattr(registry.ForecastRegistry, "get_all_method_names", lambda: tuple(predictors))
        monkeypatch.setattr(
            registry,
            "get_forecast_method_availability_snapshot",
            lambda: dict.fromkeys(predictors, True),
        )
        return calls

    return install


def _future_index(series, horizon, params):
    return np.arange(len(series), len(series) + horizon, dtype=float)


def test_ensemble_rmse_weights_remain_non_degenerate(components):
    components({
        "naive": _future_index,
        "theta": lambda s, h, p: _future_index(s, h, p) * 1.000001,
    })
    out = EnsembleMethod().forecast(
        pd.Series(np.arange(40, dtype=float)),
        horizon=2,
        seasonality=1,
        params={"methods": ["naive", "theta"], "mode": "rmse_weighted"},
    )

    weights = out.metadata["weights"]
    assert weights[0] > 0.5
    assert weights[1] > 0.05
    assert sum(weights) == pytest.approx(1.0)
    assert out.metadata["cv_total_rows"] == 12


def test_ensemble_rmse_weighting_requires_cv_for_every_member(components):
    components({
        "naive": _future_index,
        "theta": lambda s, h, p: np.full(h, np.nan),
    })
    with pytest.raises(ValueError, match="theta"):
        EnsembleMethod().forecast(
            pd.Series(np.arange(40, dtype=float)),
            horizon=2,
            seasonality=1,
            params={"methods": ["naive", "theta"], "mode": "rmse_weighted"},
        )


def test_ensemble_rejects_component_dispatch_exceptions(components):
    def broken(series, horizon, params):
        raise RuntimeError("boom")

    components({"naive": _future_index, "bad": broken})
    with pytest.raises(ValueError, match="bad: boom"):
        EnsembleMethod().forecast(
            pd.Series(np.arange(20, dtype=float)),
            horizon=2,
            seasonality=1,
            params={"methods": ["naive", "bad"], "mode": "average"},
        )


@pytest.mark.parametrize("invalid", (None, [1.0], [1.0, np.nan]))
def test_ensemble_rejects_invalid_component_forecasts(components, invalid):
    components({"naive": _future_index, "bad": lambda s, h, p: invalid})
    with pytest.raises(ValueError, match="bad:"):
        EnsembleMethod().forecast(
            pd.Series(np.arange(20, dtype=float)),
            horizon=2,
            seasonality=1,
            params={"methods": ["naive", "bad"]},
        )


@pytest.mark.parametrize(
    "coefficients,semantics,weights",
    [([2.0, 1.0], "normalized_coefficients", [2.0 / 3.0, 1.0 / 3.0]),
     ([1.0, -2.0], "raw_coefficients", [1.0, -2.0])],
)
def test_ensemble_stacking_preserves_fitted_coefficients(components, coefficients, semantics, weights):
    components({
        "linear": _future_index,
        "quadratic": lambda s, h, p: _future_index(s, h, p) ** 2,
    })
    index = np.arange(50, dtype=float)
    series = pd.Series(0.5 + coefficients[0] * index + coefficients[1] * index ** 2)
    out = EnsembleMethod().forecast(
        series,
        horizon=2,
        seasonality=1,
        params={"methods": ["linear", "quadratic"], "mode": "stacking"},
    )

    future = np.array([50.0, 51.0])
    np.testing.assert_allclose(
        out.forecast, 0.5 + coefficients[0] * future + coefficients[1] * future ** 2,
    )
    assert out.metadata["coefficients"] == pytest.approx(coefficients)
    assert out.metadata["weights"] == pytest.approx(weights)
    assert out.metadata["weight_semantics"] == semantics
    assert out.metadata["intercept"] == pytest.approx(0.5)


def test_ensemble_averages_registered_components_and_deduplicates_names(components):
    calls = components({
        "naive": lambda s, h, p: np.array([1.0, 2.0]),
        "theta": lambda s, h, p: np.array([3.0, 4.0]),
    })
    out = EnsembleMethod().forecast(
        pd.Series(np.arange(20, dtype=float)),
        horizon=2,
        seasonality=1,
        params={"methods": ["naive", "theta", "naive"], "weights": [1.0, 3.0]},
    )

    assert [call[0] for call in calls] == ["naive", "theta"]
    assert out.forecast.tolist() == [2.5, 3.5]
    assert out.metadata["components"] == {"naive": [1.0, 2.0], "theta": [3.0, 4.0]}


@pytest.mark.parametrize("weights", ([0.25], [0.2, 0.3, 0.5]))
def test_ensemble_rejects_weight_count_mismatch(components, weights):
    components({"naive": _future_index, "theta": _future_index})
    with pytest.raises(ValueError, match="exactly 2"):
        EnsembleMethod().forecast(
            pd.Series(np.arange(20, dtype=float)),
            horizon=2,
            seasonality=1,
            params={"methods": ["naive", "theta"], "weights": weights},
        )


def test_ensemble_rejects_unavailable_requested_components(components):
    components({"naive": _future_index})
    with pytest.raises(ValueError, match="sf_autoarima"):
        EnsembleMethod().forecast(
            pd.Series(np.arange(20, dtype=float)),
            horizon=2,
            seasonality=1,
            params={"methods": ["naive", "sf_autoarima"]},
        )


def test_ensemble_rejects_insufficient_cv_instead_of_downgrading(components):
    components({"naive": _future_index, "theta": _future_index})
    with pytest.raises(ValueError, match="requires at least 3"):
        EnsembleMethod().forecast(
            pd.Series(np.arange(20, dtype=float)),
            horizon=2,
            seasonality=1,
            params={"methods": ["naive", "theta"], "mode": "rmse_weighted"},
        )


def test_ensemble_rejects_removed_bma_mode(components):
    components({"naive": _future_index, "theta": _future_index})
    with pytest.raises(ValueError, match="rmse_weighted"):
        EnsembleMethod().forecast(
            pd.Series(np.arange(20, dtype=float)),
            horizon=2,
            seasonality=1,
            params={"methods": ["naive", "theta"], "mode": "bma"},
        )


def test_ensemble_params_applied_metadata(components):
    calls = components({"naive": _future_index, "theta": _future_index})
    out = EnsembleMethod().forecast(
        pd.Series(np.arange(20, dtype=float)),
        horizon=2,
        seasonality=1,
        params={
            "methods": ["naive", "theta"],
            "method_params": {"naive": {"alpha": 0.5}, "theta": {}},
        },
    )

    assert out.metadata["params_applied"] == {"naive": {"alpha": 0.5}}
    assert calls == [("naive", 20, {"alpha": 0.5}), ("theta", 20, {})]
