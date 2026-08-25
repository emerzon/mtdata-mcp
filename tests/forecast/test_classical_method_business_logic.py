from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.interface import ForecastResult
from mtdata.forecast.methods import classical as cl


def test_classical_base_metadata_via_naive_method():
    method = cl.NaiveMethod()
    assert method.name == "naive"
    assert method.category == "classical"
    assert method.supports_features == {
        "price": True,
        "return": True,
        "volatility": True,
        "ci": False,
    }
    assert cl.ThetaMethod().supports_features["ci"] is True


def test_naive_and_drift_forecasts_return_expected_values():
    series = pd.Series([10.0, 12.0, 15.0])

    naive = cl.NaiveMethod().forecast(series, horizon=3, seasonality=0, params={})
    assert np.allclose(naive.forecast, [15.0, 15.0, 15.0])
    assert naive.params_used == {}

    with pytest.raises(ValueError, match="requires at least 1 data point"):
        cl.NaiveMethod().forecast(pd.Series(dtype=float), horizon=2, seasonality=0, params={})

    drift = cl.DriftMethod().forecast(series, horizon=3, seasonality=0, params={})
    assert np.allclose(drift.forecast, [17.5, 20.0, 22.5])
    assert drift.params_used == {"slope": 2.5}

    with pytest.raises(ValueError, match="requires at least 2 data points"):
        cl.DriftMethod().forecast(pd.Series([5.0]), horizon=2, seasonality=0, params={})

    with pytest.raises(ValueError, match="requires at least 2 data points"):
        cl.DriftMethod().forecast(pd.Series(dtype=float), horizon=2, seasonality=0, params={})

    with pytest.raises(ValueError, match="finite"):
        cl.DriftMethod().forecast(pd.Series([5.0, np.nan, 9.0]), horizon=2, seasonality=0, params={})


def test_seasonal_naive_validation_and_repeating_pattern():
    method = cl.SeasonalNaiveMethod()
    with pytest.raises(ValueError, match="Insufficient data"):
        method.forecast(pd.Series([1.0, 2.0]), horizon=1, seasonality=3, params={})
    with pytest.raises(ValueError, match="Insufficient data"):
        method.forecast(pd.Series([1.0, 2.0]), horizon=1, seasonality=0, params={})
    with pytest.raises(ValueError, match="finite"):
        method.forecast(pd.Series([1.0, 2.0, np.nan, 4.0]), horizon=2, seasonality=2, params={})

    out = method.forecast(pd.Series([1.0, 2.0, 3.0, 4.0]), horizon=5, seasonality=2, params={})
    assert np.allclose(out.forecast, [3.0, 4.0, 3.0, 4.0, 3.0])
    assert out.params_used == {"m": 2}


def test_theta_forecast_tracks_trend_and_reports_alpha_and_slope(monkeypatch):
    monkeypatch.setattr(
        "mtdata.forecast.methods.statsforecast.GenericStatsForecastMethod.forecast",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    method = cl.ThetaMethod()
    series = pd.Series([2.0, 4.0, 6.0, 8.0])
    out = method.forecast(series, horizon=2, seasonality=0, params={})

    # On linear data y=2t, OLS slope should be near 2.
    assert out.params_used is not None
    assert 0.0 < out.params_used["alpha"] <= 1.0
    assert out.params_used["trend_slope"] == pytest.approx(2.0, abs=1e-10)
    assert out.params_used["m"] == 1
    assert out.params_used["seasonality_applied"] is False
    assert out.forecast.shape == (2,)
    assert np.all(out.forecast > 0.0)


def test_theta_forecast_rejects_non_finite_series_values():
    method = cl.ThetaMethod()

    with pytest.raises(ValueError, match="finite"):
        method.forecast(pd.Series([2.0, np.nan, 6.0]), horizon=2, seasonality=0, params={})


def test_theta_forecast_deseasonalizes_and_restores_requested_cycle(monkeypatch):
    monkeypatch.setattr(
        "mtdata.forecast.methods.statsforecast.GenericStatsForecastMethod.forecast",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    method = cl.ThetaMethod()
    trend = 10.0 + 0.5 * np.arange(24, dtype=float)
    seasonal = np.tile(np.array([-2.0, 1.0, 3.0, -2.0]), 6)
    series = pd.Series(trend + seasonal)

    out = method.forecast(series, horizon=8, seasonality=4, params={})
    nonseasonal = method.forecast(series, horizon=8, seasonality=1, params={})

    assert out.params_used["m"] == 4
    assert out.params_used["seasonality_applied"] is True
    assert not np.allclose(out.forecast, nonseasonal.forecast)
    assert np.allclose(np.diff(out.forecast[:4]), np.diff(out.forecast[4:]))


def test_theta_forecast_rejects_noncanonical_parameters():
    with pytest.raises(ValueError, match="does not accept"):
        cl.ThetaMethod().forecast(
            pd.Series([1.0, 2.0, 3.0]),
            horizon=2,
            seasonality=1,
            params={"alpha": 0.2},
        )


def test_theta_forecast_prefers_statsforecast_default_path(monkeypatch):
    def _fake_forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
        assert params["model_name"] == "OptimizedTheta"
        return ForecastResult(
            forecast=np.array([10.0, 11.0]),
            params_used={"seasonality": seasonality},
        )

    monkeypatch.setattr(
        "mtdata.forecast.methods.statsforecast.GenericStatsForecastMethod.forecast",
        _fake_forecast,
    )

    out = cl.ThetaMethod().forecast(
        pd.Series([2.0, 4.0, 6.0, 8.0]),
        horizon=2,
        seasonality=0,
        params={},
    )

    assert np.allclose(out.forecast, [10.0, 11.0])
    assert out.params_used["model_name"] == "OptimizedTheta"
    assert out.params_used["backend"] == "statsforecast"


def test_theta_forecast_falls_back_to_legacy_when_statsforecast_path_fails(monkeypatch):
    def _fail_forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
        raise RuntimeError("statsforecast unavailable")

    monkeypatch.setattr(
        "mtdata.forecast.methods.statsforecast.GenericStatsForecastMethod.forecast",
        _fail_forecast,
    )

    out = cl.ThetaMethod().forecast(
        pd.Series([2.0, 4.0, 6.0, 8.0]),
        horizon=2,
        seasonality=0,
        params={},
    )

    assert out.params_used["theta"] == 2.0
    assert out.params_used["m"] == 1
    assert out.params_used["seasonality_applied"] is False
    assert out.params_used["trend_slope"] == pytest.approx(2.0, abs=1e-10)

def test_fourier_ols_default_and_custom_params():
    method = cl.FourierOLSMethod()
    series = pd.Series(np.linspace(10.0, 20.0, 24))

    default = method.forecast(series, horizon=3, seasonality=12, params={})
    assert default.params_used == {"seasonality": 12, "terms": 3, "trend": True}
    assert default.forecast.shape == (3,)
    assert np.issubdtype(default.forecast.dtype, np.floating)

    no_seasonality = method.forecast(series, horizon=2, seasonality=0, params={"terms": None, "trend": False})
    assert no_seasonality.params_used == {"seasonality": 0, "terms": 2, "trend": False}
    assert no_seasonality.forecast.shape == (2,)

    custom = method.forecast(series, horizon=2, seasonality=24, params={"terms": 1, "trend": True})
    assert custom.params_used == {"seasonality": 24, "terms": 1, "trend": True}
    assert custom.forecast.shape == (2,)

    seasonality_one = method.forecast(series, horizon=2, seasonality=1, params={"terms": 3, "trend": True})
    assert seasonality_one.params_used == {"seasonality": 1, "terms": 0, "trend": True}
    assert seasonality_one.forecast.shape == (2,)

    with pytest.raises(ValueError, match="finite"):
        method.forecast(pd.Series([10.0, 11.0, np.nan, 13.0]), horizon=2, seasonality=2, params={})


@pytest.mark.parametrize(
    "name",
    [
        "forecast_naive",
        "forecast_drift",
        "forecast_seasonal_naive",
        "forecast_theta",
        "forecast_fourier_ols",
    ],
)
def test_classical_legacy_wrappers_removed(name):
    with pytest.raises(AttributeError):
        getattr(cl, name)
