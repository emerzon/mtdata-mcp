from __future__ import annotations

import sys
from types import ModuleType

import pandas as pd
import pytest

from mtdata.forecast import common as common_mod
from mtdata.forecast.methods import statsforecast as sfm


class _DummyStatsMethod(sfm.StatsForecastMethod):
    @property
    def name(self) -> str:
        return "dummy_stats"

    def _get_model(self, seasonality: int, params):
        return {"model": "dummy", "seasonality": seasonality}


def test_statsforecast_param_coercion_uses_shared_cli_rules():
    assert sfm._coerce_params(
        {
            "enabled": "true",
            "missing": "null",
            "code": '"001"',
            "rate": "1e-3",
            "invalid": "nan",
            "nested": '{"period": "12", "flags": ["false", "2"]}',
            "tuple": ("3", "none"),
        }
    ) == {
        "enabled": True,
        "missing": None,
        "code": "001",
        "rate": 0.001,
        "invalid": "nan",
        "nested": {"period": 12, "flags": [False, 2]},
        "tuple": (3, None),
    }


def test_statsforecast_forecast_rejects_invalid_ci_alpha(monkeypatch):
    fake_stats_mod = ModuleType("statsforecast")
    fake_stats_mod.StatsForecast = object
    monkeypatch.setitem(sys.modules, "statsforecast", fake_stats_mod)
    monkeypatch.setattr(
        common_mod,
        "_create_training_dataframes",
        lambda *args, **kwargs: (pd.DataFrame({"y": [1.0]}), None, None),
    )

    with pytest.raises(ValueError, match="ci_alpha must be between 0 and 1"):
        _DummyStatsMethod().forecast(pd.Series([1.0, 2.0, 3.0]), horizon=1, seasonality=1, params={}, ci_alpha=1.0)


def test_statsforecast_forecast_requires_unique_id_rows(monkeypatch):
    class FakeStatsForecast:
        def __init__(self, models, freq):
            pass

        def fit(self, Y_df, X_df=None):
            return None

        def predict(self, h, X_df=None, level=None):
            return pd.DataFrame({"dummy_stats": [1.0]})

    fake_stats_mod = ModuleType("statsforecast")
    fake_stats_mod.StatsForecast = FakeStatsForecast
    monkeypatch.setitem(sys.modules, "statsforecast", fake_stats_mod)
    monkeypatch.setattr(
        common_mod,
        "_create_training_dataframes",
        lambda *args, **kwargs: (pd.DataFrame({"y": [1.0]}), None, None),
    )

    monkeypatch.setattr(
        common_mod,
        "_extract_forecast_values",
        lambda *args, **kwargs: pd.Series([1.0]).to_numpy(),
    )

    result = _DummyStatsMethod().forecast(
        pd.Series([1.0, 2.0, 3.0]), horizon=1, seasonality=1, params={}
    )
    assert list(result.forecast) == [1.0]


def test_statsforecast_forecast_records_ci_diagnostics_when_columns_missing(monkeypatch):
    class FakeStatsForecast:
        def __init__(self, models, freq):
            pass

        def fit(self, Y_df, X_df=None):
            return None

        def predict(self, h, X_df=None, level=None):
            return pd.DataFrame(
                {
                    "unique_id": ["ts"],
                    "dummy_stats": [1.0],
                }
            )

    fake_stats_mod = ModuleType("statsforecast")
    fake_stats_mod.StatsForecast = FakeStatsForecast
    monkeypatch.setitem(sys.modules, "statsforecast", fake_stats_mod)
    monkeypatch.setattr(
        common_mod,
        "_create_training_dataframes",
        lambda *args, **kwargs: (pd.DataFrame({"y": [1.0]}), None, None),
    )
    monkeypatch.setattr(
        common_mod,
        "_extract_forecast_values",
        lambda *args, **kwargs: pd.Series([1.0]).to_numpy(),
    )

    res = _DummyStatsMethod().forecast(pd.Series([1.0, 2.0, 3.0]), horizon=1, seasonality=1, params={}, ci_alpha=0.05)

    assert res.ci_values is None
    assert res.metadata["diagnostics"]["ci"]["available"] is False
    assert res.metadata["diagnostics"]["ci"]["status"] == "unavailable"
    assert res.metadata["diagnostics"]["ci"]["level"] == 95


def test_generic_statsforecast_get_model_wraps_constructor_type_errors(monkeypatch):
    fake_stats_mod = ModuleType("statsforecast")
    models_mod = ModuleType("statsforecast.models")

    class NeedsArg:
        def __init__(self, required):
            self.required = required

    models_mod.NeedsArg = NeedsArg
    fake_stats_mod.models = models_mod

    monkeypatch.setitem(sys.modules, "statsforecast", fake_stats_mod)
    monkeypatch.setitem(sys.modules, "statsforecast.models", models_mod)

    method = sfm.GenericStatsForecastMethod()

    with pytest.raises(ValueError, match="Invalid parameters for StatsForecast model NeedsArg"):
        method._get_model(1, {"model_name": "NeedsArg"})


def test_statsforecast_legacy_wrapper_removed():
    assert not hasattr(sfm, "forecast_statsforecast")
