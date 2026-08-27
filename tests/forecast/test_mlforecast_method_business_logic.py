from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast import common as common_mod
from mtdata.forecast.methods import mlforecast as mlm


class _DummyMLMethod(mlm.MLForecastMethod):
    @property
    def name(self) -> str:
        return "dummy_ml"

    def _get_model(self, params):
        return {"model": "dummy"}


def test_mlforecast_base_metadata_on_concrete_method():
    method = _DummyMLMethod()
    assert method.name == "dummy_ml"
    assert method.category == "machine_learning"
    assert method.required_packages == ["mlforecast"]
    assert method.supports_features == {
        "price": True,
        "return": True,
        "volatility": True,
        "ci": False,
    }


def test_mlforecast_forecast_raises_runtime_on_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlforecast", ModuleType("mlforecast"))
    with pytest.raises(RuntimeError, match="Failed to import mlforecast"):
        _DummyMLMethod().forecast(pd.Series([1.0, 2.0, 3.0]), horizon=1, seasonality=0, params={})


def test_mlforecast_forecast_with_default_lags_exog_and_internal_param_filter(monkeypatch):
    calls = {}

    class FakeMLForecast:
        def __init__(self, models, freq, lags):
            calls["init"] = {"models": models, "freq": freq, "lags": lags}

        def fit(self, df, static_features=None, **kwargs):
            calls["fit"] = {"df": df, "static_features": static_features, "kwargs": kwargs}

        def predict(self, h, X_df=None, **kwargs):
            calls["predict"] = {"h": h, "X_df": X_df, **kwargs}
            return pd.DataFrame(
                {
                    "unique_id": ["ts", "ts", "ts"],
                    "ds": [6, 7, 8],
                    "dummy_ml": [11.0, 12.0, 13.0],
                }
            )

    fake_ml_mod = ModuleType("mlforecast")
    fake_ml_mod.MLForecast = FakeMLForecast
    monkeypatch.setitem(sys.modules, "mlforecast", fake_ml_mod)

    exog_used = np.arange(6, dtype=float).reshape(-1, 1)
    exog_future = np.array([[10.0], [11.0], [12.0]])
    out = _DummyMLMethod().forecast(
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        horizon=3,
        seasonality=5,
        params={
            "symbol": "EURUSD",
            "timeframe": "H1",
            "as_of": "2024-01-01",
            "custom": 7,
        },
        exog_future=exog_future,
        exog_used=exog_used,
    )

    assert np.allclose(out.forecast, [11.0, 12.0, 13.0])
    assert out.ci_values is None
    assert out.params_used == {"custom": 7, "lags": [1, 2, 3, 4, 5]}
    assert calls["init"]["freq"] == 1
    assert calls["init"]["lags"] == [1, 2, 3, 4, 5]
    # MLForecast 1.x: historical exog are merged into the train frame
    assert list(calls["fit"]["df"].columns) == ["unique_id", "ds", "y", "x0"]
    assert calls["fit"]["df"]["x0"].tolist() == exog_used[:, 0].tolist()
    assert calls["fit"]["static_features"] == []
    assert calls["predict"]["h"] == 3
    assert calls["predict"]["X_df"]["x0"].tolist() == exog_future[:, 0].tolist()


def test_mlforecast_train_accepts_historical_exog_without_future_exog(monkeypatch):
    calls = {}

    class FakeMLForecast:
        def __init__(self, models, freq, lags):
            calls["init"] = {"models": models, "freq": freq, "lags": lags}

        def fit(self, df, static_features=None, **kwargs):
            calls["fit"] = {"df": df, "static_features": static_features, "kwargs": kwargs}

    fake_ml_mod = ModuleType("mlforecast")
    fake_ml_mod.MLForecast = FakeMLForecast
    monkeypatch.setitem(sys.modules, "mlforecast", fake_ml_mod)

    method = _DummyMLMethod()
    monkeypatch.setattr(method, "serialize_artifact", lambda _artifact: b"artifact")
    exog_used = np.arange(8, dtype=float).reshape(-1, 1)

    result = method.train(
        pd.Series(np.linspace(1.0, 2.0, 8)),
        horizon=2,
        seasonality=3,
        params={},
        exog=exog_used,
    )

    assert result.artifact_bytes == b"artifact"
    assert result.params_used == {"lags": [1, 2, 3]}
    assert calls["fit"]["df"]["x0"].tolist() == exog_used[:, 0].tolist()
    assert calls["fit"]["static_features"] == []


@pytest.mark.parametrize(
    ("exog_used", "exog_future", "expected_message"),
    [
        (
            np.ones((4, 1)),
            None,
            "historical exog present but future exog absent",
        ),
        (
            None,
            np.ones((2, 1)),
            "historical exog absent but future exog present",
        ),
    ],
)
def test_mlforecast_prediction_rejects_exog_presence_mismatch(
    exog_used,
    exog_future,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        _DummyMLMethod().predict_with_model(
            object(),
            pd.Series([1.0, 2.0, 3.0, 4.0]),
            horizon=2,
            seasonality=1,
            params={},
            exog_used=exog_used,
            exog_future=exog_future,
        )


def test_mlforecast_forecast_without_exog_uses_simple_fit_and_predict(monkeypatch):
    calls = {}

    class FakeMLForecast:
        def __init__(self, models, freq, lags):
            calls["lags"] = lags

        def fit(self, df, static_features=None, **kwargs):
            calls["fit_df"] = df
            calls["static_features"] = static_features

        def predict(self, h, X_df=None, **kwargs):
            calls["predict_x"] = X_df
            return pd.DataFrame({"unique_id": ["ts"], "dummy_ml": [9.0]})

    fake_ml_mod = ModuleType("mlforecast")
    fake_ml_mod.MLForecast = FakeMLForecast
    monkeypatch.setitem(sys.modules, "mlforecast", fake_ml_mod)

    monkeypatch.setattr(
        common_mod,
        "_create_training_dataframes",
        lambda *args, **kwargs: (pd.DataFrame({"y": [1.0]}), None, None),
    )
    monkeypatch.setattr(common_mod, "_extract_forecast_values", lambda Yf, horizon, method_name: np.array([9.0], dtype=float))

    out = _DummyMLMethod().forecast(
        pd.Series([1.0, 2.0]),
        horizon=1,
        seasonality=0,
        params={"lags": [2, 4]},
    )

    assert np.allclose(out.forecast, [9.0])
    assert out.params_used == {"lags": [2, 4]}
    assert calls["lags"] == [2, 4]
    assert list(calls["fit_df"].columns) == ["y"]
    assert calls["static_features"] is None
    assert calls["predict_x"] is None


def test_mlforecast_forecast_wraps_runtime_errors(monkeypatch):
    class FakeMLForecast:
        def __init__(self, models, freq, lags):
            pass

        def fit(self, df, static_features=None, **kwargs):
            raise ValueError("fit exploded")

    fake_ml_mod = ModuleType("mlforecast")
    fake_ml_mod.MLForecast = FakeMLForecast
    monkeypatch.setitem(sys.modules, "mlforecast", fake_ml_mod)
    monkeypatch.setattr(
        common_mod,
        "_create_training_dataframes",
        lambda *args, **kwargs: (pd.DataFrame({"y": [1.0]}), None, None),
    )

    with pytest.raises(ValueError, match="fit exploded"):
        _DummyMLMethod().forecast(pd.Series([1.0, 2.0]), horizon=1, seasonality=0, params={})


def test_mlforecast_forecast_requires_unique_id_rows(monkeypatch):
    class FakeMLForecast:
        def __init__(self, models, freq, lags):
            pass

        def fit(self, df, static_features=None, **kwargs):
            return None

        def predict(self, h, X_df=None, **kwargs):
            return pd.DataFrame({"dummy_ml": [9.0]})

    fake_ml_mod = ModuleType("mlforecast")
    fake_ml_mod.MLForecast = FakeMLForecast
    monkeypatch.setitem(sys.modules, "mlforecast", fake_ml_mod)
    monkeypatch.setattr(
        common_mod,
        "_create_training_dataframes",
        lambda *args, **kwargs: (pd.DataFrame({"y": [1.0]}), None, None),
    )

    monkeypatch.setattr(
        common_mod,
        "_extract_forecast_values",
        lambda Yf, horizon, method_name: np.array([9.0], dtype=float),
    )
    result = _DummyMLMethod().forecast(
        pd.Series([1.0, 2.0]), horizon=1, seasonality=0, params={}
    )
    assert list(result.forecast) == [9.0]


def test_mlforecast_random_forest_get_model(monkeypatch):
    captured = {}
    sklearn_mod = ModuleType("sklearn")
    ensemble_mod = ModuleType("sklearn.ensemble")

    class FakeRF:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    ensemble_mod.RandomForestRegressor = FakeRF
    monkeypatch.setitem(sys.modules, "sklearn", sklearn_mod)
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", ensemble_mod)

    model = mlm.MLFRandomForest()._get_model({"n_estimators": "55", "max_depth": 4})
    assert isinstance(model, FakeRF)
    assert captured["kwargs"] == {"n_estimators": 55, "max_depth": 4, "random_state": 42}
    assert mlm.MLFRandomForest().required_packages == ["mlforecast", "scikit-learn"]


def test_mlforecast_lightgbm_get_model(monkeypatch):
    captured = {}
    lgbm_mod = ModuleType("lightgbm")

    class FakeLGBM:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    lgbm_mod.LGBMRegressor = FakeLGBM
    monkeypatch.setitem(sys.modules, "lightgbm", lgbm_mod)

    model = mlm.MLFLightGBM()._get_model(
        {"n_estimators": "10", "learning_rate": "0.1", "num_leaves": "16", "max_depth": "6"}
    )
    assert isinstance(model, FakeLGBM)
    assert captured["kwargs"] == {
        "n_estimators": 10,
        "learning_rate": 0.1,
        "num_leaves": 16,
        "max_depth": 6,
        "random_state": 42,
        "verbosity": -1,
    }
    assert mlm.MLFLightGBM().required_packages == ["mlforecast", "lightgbm"]


def test_generic_mlforecast_model_import_validation_and_param_filtering(monkeypatch):
    method = mlm.GenericMLForecastMethod()
    with pytest.raises(ValueError, match="requires 'model'"):
        method._get_model({})

    with pytest.raises(ValueError, match="not allowed"):
        method._get_model({"model": "missing.module.Class"})

    with pytest.raises(ValueError, match="not allowed"):
        method._get_model({"model": "fake_models.FakeModel"})

    original_import_module = mlm.importlib.import_module

    def fake_import_module(name, package=None):
        if name == "sklearn.ensemble":
            raise ImportError("forced import failure")
        return original_import_module(name, package)

    monkeypatch.setattr(mlm.importlib, "import_module", fake_import_module)
    with pytest.raises(ValueError, match="Could not import ML model"):
        method._get_model({"model": "sklearn.ensemble.RandomForestRegressor"})
    sklearn_mod = ModuleType("sklearn")
    ensemble_mod = ModuleType("sklearn.ensemble")

    class FakeModel:
        def __init__(self, a, b=2):
            self.a = a
            self.b = b

    ensemble_mod.RandomForestRegressor = FakeModel
    monkeypatch.setitem(sys.modules, "sklearn", sklearn_mod)
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", ensemble_mod)

    model = method._get_model({"model": "sklearn.ensemble.RandomForestRegressor", "a": 5, "b": 6, "c": 7})
    assert isinstance(model, FakeModel)
    assert model.a == 5
    assert model.b == 6


@pytest.mark.parametrize("name", ["forecast_mlf_rf", "forecast_mlf_lightgbm"])
def test_mlforecast_legacy_wrappers_removed(name):
    with pytest.raises(AttributeError):
        getattr(mlm, name)


def test_mlforecast_public_params_do_not_advertise_rolling_agg():
    for method_cls in (mlm.MLFRandomForest, mlm.MLFLightGBM, mlm.GenericMLForecastMethod):
        names = {row["name"] for row in method_cls.PARAMS}
        assert "rolling_agg" not in names
