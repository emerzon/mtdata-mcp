from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.interface import ForecastResult
from mtdata.forecast.methods import ets_arima as ea


def test_ets_arima_method_metadata_on_concrete_method():
    method = ea.SESMethod()
    assert method.name == "ses"
    assert method.category == "ets_arima"
    assert method.required_packages == ["statsmodels"]
    assert method.supports_features == {
        "price": True,
        "return": True,
        "volatility": False,
        "ci": False,
    }


def test_ses_requires_statsmodels(monkeypatch):
    monkeypatch.setattr(ea, "_SM_ETS_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="requires statsmodels"):
        ea.SESMethod().forecast(pd.Series([1.0, 2.0, 3.0]), horizon=2, seasonality=0, params={})


def test_ses_optimized_path_and_alpha_recovery(monkeypatch):
    calls = {}

    class FakeSES:
        def __init__(self, vals, initialization_method):
            calls["init"] = {"vals": np.asarray(vals), "initialization_method": initialization_method}

        def fit(self, **kwargs):
            calls["fit"] = kwargs
            return SimpleNamespace(
                forecast=lambda h: np.arange(10.0, 10.0 + int(h), dtype=float),
                params={"smoothing_level": 0.77},
            )

    monkeypatch.setattr(ea, "_SM_ETS_AVAILABLE", True)
    monkeypatch.setattr(ea, "_SES", FakeSES)

    out = ea.SESMethod().forecast(pd.Series([1.0, 2.0, 3.0]), horizon=3, seasonality=0, params={})
    assert np.allclose(out.forecast, [10.0, 11.0, 12.0])
    assert out.params_used == {"alpha": 0.77}
    assert calls["init"]["initialization_method"] == "heuristic"
    assert calls["fit"] == {"optimized": True}


def test_ses_manual_alpha_path(monkeypatch):
    calls = {}

    class FakeSES:
        def __init__(self, vals, initialization_method):
            calls["init"] = True

        def fit(self, **kwargs):
            calls["fit"] = kwargs
            return SimpleNamespace(forecast=lambda h: np.full(int(h), 5.0, dtype=float), params=None)

    monkeypatch.setattr(ea, "_SM_ETS_AVAILABLE", True)
    monkeypatch.setattr(ea, "_SES", FakeSES)

    out = ea.SESMethod().forecast(pd.Series([1.0, 2.0]), horizon=2, seasonality=0, params={"alpha": "0.2"})
    assert np.allclose(out.forecast, [5.0, 5.0])
    assert out.params_used == {"alpha": "0.2"}
    assert calls["fit"] == {"smoothing_level": 0.2, "optimized": False}


def test_holt_forecast_manual_and_optimized_paths(monkeypatch):
    calls = []

    class FakeETS:
        def __init__(self, vals, **kwargs):
            calls.append(("init", kwargs))

        def fit(self, **kwargs):
            calls.append(("fit", kwargs))
            return SimpleNamespace(forecast=lambda h: np.full(int(h), 9.0, dtype=float))

    monkeypatch.setattr(ea, "_SM_ETS_AVAILABLE", True)
    monkeypatch.setattr(ea, "_ETS", FakeETS)

    method = ea.HoltMethod()
    manual = method.forecast(
        pd.Series([1.0, 2.0, 3.0]), horizon=2, seasonality=0, params={"alpha": 0.3, "damped": True}
    )
    optimized = method.forecast(pd.Series([1.0, 2.0, 3.0]), horizon=1, seasonality=0, params={})

    assert np.allclose(manual.forecast, [9.0, 9.0])
    assert manual.params_used == {"damped": True, "alpha": 0.3}
    assert np.allclose(optimized.forecast, [9.0])
    assert optimized.params_used == {"damped": False}

    assert calls[0][0] == "init"
    assert calls[0][1]["trend"] == "add"
    assert calls[0][1]["damped_trend"] is True
    assert calls[1] == (
        "fit",
        {"optimized": False, "smoothing_level": 0.3, "smoothing_trend": None},
    )
    assert calls[3] == ("fit", {"optimized": True})


def test_holt_winters_validates_seasonality_and_manual_params(monkeypatch):
    calls = {}

    class FakeETS:
        def __init__(self, vals, **kwargs):
            calls["init"] = kwargs

        def fit(self, **kwargs):
            calls["fit"] = kwargs
            return SimpleNamespace(forecast=lambda h: np.arange(1.0, 1.0 + int(h), dtype=float))

    monkeypatch.setattr(ea, "_SM_ETS_AVAILABLE", True)
    monkeypatch.setattr(ea, "_ETS", FakeETS)

    method = ea.HoltWintersAddMethod()
    with pytest.raises(ValueError, match="positive seasonality"):
        method.forecast(pd.Series([1.0, 2.0]), horizon=1, seasonality=0, params={})

    out = method.forecast(
        pd.Series([1.0, 2.0, 3.0, 4.0]),
        horizon=2,
        seasonality=4,
        params={"alpha": 0.1, "beta": 0.2, "gamma": 0.3, "damped": True},
    )
    assert np.allclose(out.forecast, [1.0, 2.0])
    assert out.params_used == {
        "seasonal": "add",
        "m": 4,
        "damped": True,
        "alpha": 0.1,
        "beta": 0.2,
        "gamma": 0.3,
    }
    assert calls["init"]["seasonal"] == "add"
    assert calls["fit"] == {
        "optimized": False,
        "smoothing_level": 0.1,
        "smoothing_trend": 0.2,
        "smoothing_seasonal": 0.3,
    }


def test_holt_winters_mul_routes_to_mul_variant(monkeypatch):
    captured = {}

    def fake_forecast_hw(self, series, horizon, seasonality, params, seasonal_type):
        captured["seasonal_type"] = seasonal_type
        return ForecastResult(forecast=np.array([7.0], dtype=float), params_used={"ok": True})

    monkeypatch.setattr(ea.HoltWintersMulMethod, "_forecast_hw", fake_forecast_hw)
    out = ea.HoltWintersMulMethod().forecast(pd.Series([1.0, 2.0]), horizon=1, seasonality=2, params={})
    assert np.allclose(out.forecast, [7.0])
    assert captured["seasonal_type"] == "mul"


def test_ets_norm_component_mapping_and_errors():
    fn = ea.ETSMethod._norm_component
    assert fn(None) is None
    assert fn("null") is None
    assert fn("a") == "add"
    assert fn("multiplicative") == "mul"
    assert fn("auto", allow_auto=True) == "auto"
    with pytest.raises(ValueError, match="Invalid ETS component"):
        fn("bad")


def test_ets_requires_statsmodels(monkeypatch):
    monkeypatch.setattr(ea, "_SM_ETS_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="ETS requires statsmodels"):
        ea.ETSMethod().forecast(pd.Series([1.0, 2.0]), horizon=1, seasonality=0, params={})


def test_ets_forecast_auto_and_trend_none_forces_not_damped(monkeypatch):
    calls = {}

    class FakeETS:
        def __init__(self, vals, **kwargs):
            calls["init"] = kwargs

        def fit(self, **kwargs):
            calls["fit"] = kwargs
            return SimpleNamespace(
                forecast=lambda h: np.full(int(h), 4.0, dtype=float),
                params={"smoothing_level": 0.11},
            )

    monkeypatch.setattr(ea, "_SM_ETS_AVAILABLE", True)
    monkeypatch.setattr(ea, "_ETS", FakeETS)

    out = ea.ETSMethod().forecast(
        pd.Series([1.0, 2.0, 3.0]),
        horizon=2,
        seasonality=1,
        params={"trend": "none", "seasonal": "auto", "damped": True},
    )
    assert np.allclose(out.forecast, [4.0, 4.0])
    assert out.params_used == {
        "trend": None,
        "seasonal": None,
        "m": 0,
        "damped": False,
        "alpha": 0.11,
    }
    assert calls["init"]["trend"] is None
    assert calls["init"]["seasonal"] is None
    assert calls["init"]["seasonal_periods"] is None
    assert calls["init"]["damped_trend"] is False
    assert calls["fit"] == {"optimized": True}


def test_ets_forecast_rejects_invalid_seasonality(monkeypatch):
    monkeypatch.setattr(ea, "_SM_ETS_AVAILABLE", True)
    with pytest.raises(ValueError, match="requires seasonality >= 2"):
        ea.ETSMethod().forecast(
            pd.Series([1.0, 2.0, 3.0]),
            horizon=1,
            seasonality=1,
            params={"seasonal": "add"},
        )


def test_ets_forecast_manual_path_and_param_extraction(monkeypatch):
    calls = {}

    class FakeETS:
        def __init__(self, vals, **kwargs):
            calls["init"] = kwargs

        def fit(self, **kwargs):
            calls["fit"] = kwargs
            return SimpleNamespace(
                forecast=lambda h: np.arange(2.0, 2.0 + int(h), dtype=float),
                params={"smoothing_level": 0.8, "smoothing_trend": 0.1, "smoothing_seasonal": 0.2},
            )

    monkeypatch.setattr(ea, "_SM_ETS_AVAILABLE", True)
    monkeypatch.setattr(ea, "_ETS", FakeETS)

    out = ea.ETSMethod().forecast(
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        horizon=2,
        seasonality=4,
        params={"alpha": "0.3", "beta": 0.4, "gamma": 0.5, "seasonal": "auto"},
    )
    assert np.allclose(out.forecast, [2.0, 3.0])
    assert out.params_used == {
        "trend": "add",
        "seasonal": "add",
        "m": 4,
        "damped": False,
        "alpha": 0.8,
        "beta": 0.1,
        "gamma": 0.2,
    }
    assert calls["fit"] == {
        "optimized": False,
        "smoothing_level": 0.3,
        "smoothing_trend": 0.4,
        "smoothing_seasonal": 0.5,
    }


def test_ets_forecast_rejects_insufficient_history_for_seasonal_fit(monkeypatch):
    monkeypatch.setattr(ea, "_SM_ETS_AVAILABLE", True)
    with pytest.raises(ValueError, match="requires at least 192 observations"):
        ea.ETSMethod().forecast(
            pd.Series(np.arange(150.0)),
            horizon=12,
            seasonality=96,
            params={"seasonal": "auto"},
            timeframe="M15",
        )


def test_arima_requires_statsmodels(monkeypatch):
    monkeypatch.setattr(ea, "_SM_SARIMAX_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="requires statsmodels"):
        ea.ARIMAMethod().forecast(
            pd.Series([1.0, 2.0]),
            horizon=1,
            seasonality=0,
            params={"p": 1, "d": 1, "q": 1},
        )


def test_arima_prefers_auto_arima_when_no_order_supplied(monkeypatch):
    def _fake_forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
        assert params["model_name"] == "AutoARIMA"
        return ForecastResult(
            forecast=np.array([21.0, 22.0], dtype=float),
            params_used={"seasonality": seasonality},
        )

    monkeypatch.setattr(
        "mtdata.forecast.methods.statsforecast.GenericStatsForecastMethod.forecast",
        _fake_forecast,
    )

    out = ea.ARIMAMethod().forecast(
        pd.Series([10.0, 11.0, 12.0]),
        horizon=2,
        seasonality=0,
        params={},
    )

    assert np.allclose(out.forecast, [21.0, 22.0])
    assert out.params_used["model_name"] == "AutoARIMA"
    assert out.params_used["backend"] == "statsforecast"
    assert out.params_used["auto_selected"] is True


def test_arima_builds_orders_ci_and_exog_metadata(monkeypatch):
    calls = {}

    class FakePred:
        predicted_mean = np.array([30.0, 31.0], dtype=float)

        def conf_int(self, alpha):
            calls["ci_alpha"] = alpha
            return pd.DataFrame({"lower": [29.0, 30.0], "upper": [31.0, 32.0]})

    class FakeRes:
        def get_forecast(self, steps, exog=None):
            calls["forecast"] = {"steps": steps, "exog": exog}
            return FakePred()

    class FakeSarimax:
        def __init__(self, vals, **kwargs):
            calls["init"] = {"vals": vals, **kwargs}

        def fit(self, **kwargs):
            calls["fit"] = kwargs
            return FakeRes()

    monkeypatch.setattr(ea, "_SM_SARIMAX_AVAILABLE", True)
    monkeypatch.setattr(ea, "_SARIMAX", FakeSarimax)

    exog_used = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0]})
    exog_future = pd.DataFrame({"x1": [7.0, 8.0], "x2": [9.0, 10.0]})
    out = ea.ARIMAMethod().forecast(
        pd.Series([10.0, 11.0, 12.0]),
        horizon=2,
        seasonality=12,
        params={"p": 2, "d": 0, "q": 1, "P": 1, "D": 0, "Q": 1, "trend": "n", "alpha": 0.1},
        exog_future=exog_future,
        exog_used=exog_used,
    )

    assert np.allclose(out.forecast, [30.0, 31.0])
    assert out.ci_values is not None
    assert np.allclose(out.ci_values[0], [29.0, 30.0])
    assert np.allclose(out.ci_values[1], [31.0, 32.0])
    assert out.params_used == {
        "order": (2, 0, 1),
        "seasonal_order": (1, 0, 1, 12),
        "trend": "n",
        "exog": {"n_features": 2},
    }
    assert tuple(calls["init"]["order"]) == (2, 0, 1)
    assert tuple(calls["init"]["seasonal_order"]) == (1, 0, 1, 12)
    assert isinstance(calls["init"]["exog"], np.ndarray)
    assert calls["fit"] == {"method": "lbfgs", "disp": False, "maxiter": 100}
    assert calls["forecast"]["steps"] == 2
    assert isinstance(calls["forecast"]["exog"], np.ndarray)
    assert calls["ci_alpha"] == 0.1


def test_sarima_auto_seasonal_order_when_missing(monkeypatch):
    calls = []

    class FakePred:
        predicted_mean = np.array([1.0], dtype=float)

        def conf_int(self, alpha):
            return pd.DataFrame({"l": [0.9], "u": [1.1]})

    class FakeRes:
        def get_forecast(self, steps, exog=None):
            return FakePred()

    class FakeSarimax:
        def __init__(self, vals, **kwargs):
            calls.append(kwargs["seasonal_order"])

        def fit(self, **kwargs):
            return FakeRes()

    monkeypatch.setattr(ea, "_SM_SARIMAX_AVAILABLE", True)
    monkeypatch.setattr(ea, "_SARIMAX", FakeSarimax)

    out_default = ea.SARIMAMethod().forecast(
        pd.Series([1.0, 2.0, 3.0]),
        horizon=1,
        seasonality=6,
        params={"order": (1, 1, 1), "P": 0, "D": 0, "Q": 0},
    )
    out_auto = ea.SARIMAMethod().forecast(
        pd.Series([1.0, 2.0, 3.0]),
        horizon=1,
        seasonality=6,
        params={"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)},
    )
    assert np.allclose(out_default.forecast, [1.0])
    assert np.allclose(out_auto.forecast, [1.0])
    assert tuple(calls[0]) == (0, 1, 1, 6)
    assert tuple(calls[1]) == (0, 1, 1, 6)


def test_arima_conf_int_errors_are_reported_in_metadata(monkeypatch):
    class FakePred:
        predicted_mean = np.array([3.0], dtype=float)

        def conf_int(self, alpha):
            raise RuntimeError("no ci")

    class FakeRes:
        def get_forecast(self, steps, exog=None):
            return FakePred()

    class FakeSarimax:
        def __init__(self, vals, **kwargs):
            pass

        def fit(self, **kwargs):
            return FakeRes()

    monkeypatch.setattr(ea, "_SM_SARIMAX_AVAILABLE", True)
    monkeypatch.setattr(ea, "_SARIMAX", FakeSarimax)
    monkeypatch.setattr(
        "mtdata.forecast.methods.statsforecast.GenericStatsForecastMethod.forecast",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("auto unavailable")),
    )

    out = ea.ARIMAMethod().forecast(pd.Series([1.0, 2.0, 3.0]), horizon=1, seasonality=0, params={})
    assert np.allclose(out.forecast, [3.0])
    assert out.ci_values is None
    assert out.metadata is not None
    assert out.metadata["ci_warning"] == "Failed to compute confidence intervals: no ci"
    assert out.metadata["diagnostics"]["ci"] == {
        "provider": "arima",
        "requested": True,
        "available": False,
        "status": "failed",
        "alpha": 0.05,
        "warning": "Failed to compute confidence intervals: no ci",
        "error": "no ci",
        "error_type": "RuntimeError",
    }


@pytest.mark.parametrize(
    "name",
    ["forecast_ses", "forecast_holt", "forecast_holt_winters", "forecast_sarimax"],
)
def test_legacy_wrappers_removed(name):
    with pytest.raises(AttributeError):
        getattr(ea, name)
