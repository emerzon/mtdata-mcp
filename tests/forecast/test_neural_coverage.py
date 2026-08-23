"""Coverage tests for mtdata.forecast.methods.neural – targeting uncovered lines."""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Provide fake neuralforecast before any project imports ────────────────────
_nf_models = types.ModuleType("neuralforecast.models")
_nf_models.NHITS = MagicMock(name="NHITS")
_nf_models.NBEATSx = MagicMock(name="NBEATSx")
_nf_models.TFT = MagicMock(name="TFT")
_nf_models.PatchTST = MagicMock(name="PatchTST")

_nf_pkg = types.ModuleType("neuralforecast")
_nf_pkg.models = _nf_models

_orig_mt5 = sys.modules.get("MetaTrader5")
_orig_nf = sys.modules.get("neuralforecast")
_orig_nf_models = sys.modules.get("neuralforecast.models")

sys.modules.setdefault("neuralforecast", _nf_pkg)
sys.modules.setdefault("neuralforecast.models", _nf_models)

# Ensure MetaTrader5 mock exists for transitive imports
_mt5_mock = MagicMock()
_mt5_mock.TIMEFRAME_M1 = 1; _mt5_mock.TIMEFRAME_M2 = 2; _mt5_mock.TIMEFRAME_M3 = 3
_mt5_mock.TIMEFRAME_M4 = 4; _mt5_mock.TIMEFRAME_M5 = 5; _mt5_mock.TIMEFRAME_M6 = 6
_mt5_mock.TIMEFRAME_M10 = 10; _mt5_mock.TIMEFRAME_M12 = 12; _mt5_mock.TIMEFRAME_M15 = 15
_mt5_mock.TIMEFRAME_M20 = 20; _mt5_mock.TIMEFRAME_M30 = 30
_mt5_mock.TIMEFRAME_H1 = 16385; _mt5_mock.TIMEFRAME_H2 = 16386; _mt5_mock.TIMEFRAME_H3 = 16387
_mt5_mock.TIMEFRAME_H4 = 16388; _mt5_mock.TIMEFRAME_H6 = 16390; _mt5_mock.TIMEFRAME_H8 = 16392
_mt5_mock.TIMEFRAME_H12 = 16396; _mt5_mock.TIMEFRAME_D1 = 16408
_mt5_mock.TIMEFRAME_W1 = 32769; _mt5_mock.TIMEFRAME_MN1 = 49153
sys.modules["MetaTrader5"] = _mt5_mock

from mtdata.forecast.interface import ForecastResult
from mtdata.forecast.methods.neural import (
    NBEATSXMethod,
    NeuralForecastMethod,
    NHITSMethod,
    PatchTSTMethod,
    TFTMethod,
    _neural_resolve_hyperparams,
    _neural_resolve_validation_settings,
    _resolve_nf_model_class,
)


@pytest.fixture(autouse=True, scope="module")
def _restore_sys_modules():
    yield
    for name, orig in [("MetaTrader5", _orig_mt5), ("neuralforecast", _orig_nf), ("neuralforecast.models", _orig_nf_models)]:
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_Yf(fh: int):
    """Create a fake NeuralForecast output DataFrame."""
    return pd.DataFrame({
        "unique_id": ["ts"] * fh,
        "ds": list(range(fh)),
        "y_hat": np.linspace(100, 110, fh).tolist(),
    })


def _make_series(n: int = 100):
    return pd.Series(np.linspace(100, 110, n), dtype=float)


class TestNeuralHyperparams:
    def test_unknown_method(self):
        with pytest.raises(RuntimeError, match="Unknown neural method"):
            _resolve_nf_model_class("unknown_model")

    def test_custom_input_size(self):
        input_size, steps, batch_size, lr = _neural_resolve_hyperparams(
            {"input_size": 32}, 100, 12, 24
        )
        assert input_size == 32
        assert steps == 50
        assert batch_size == 32
        assert lr is None

    def test_max_steps_param(self):
        _input_size, steps, _batch_size, _lr = _neural_resolve_hyperparams(
            {"max_steps": 100}, 100, 12, 24
        )
        assert steps == 100

    def test_auto_input_size_reserves_horizon(self):
        input_size, _steps, _batch_size, _lr = _neural_resolve_hyperparams(
            {}, 100, 12, 0
        )
        assert input_size == 88

    def test_auto_validation_uses_horizon_or_fifth(self):
        val_size, patience = _neural_resolve_validation_settings({}, n=100, fh=12, steps=50)
        assert val_size == 20
        assert patience == 10

    def test_short_series_disables_early_stopping(self):
        val_size, patience = _neural_resolve_validation_settings({}, n=5, fh=4, steps=50)
        assert val_size == 0
        assert patience is None


# ── NeuralForecastMethod and subclasses  (lines 90-177) ─────────────────────

class TestNeuralForecastMethodProperties:
    @pytest.mark.parametrize(
        ("cls", "name"),
        [
            (NHITSMethod, "nhits"),
            (NBEATSXMethod, "nbeatsx"),
            (TFTMethod, "tft"),
            (PatchTSTMethod, "patchtst"),
        ],
    )
    def test_method_names(self, cls, name):
        m = cls()
        assert m.name == name
        assert m.category == "neural"
        assert "neuralforecast" in m.required_packages

    def test_supports_features(self):
        m = NHITSMethod()
        sf = m.supports_features
        assert sf["price"] is True
        assert sf["return"] is True
        assert sf["volatility"] is False
        assert sf["ci"] is False


class TestNeuralForecastMethodForecast:
    def test_forecast_uses_ephemeral_train_predict(self):
        m = NHITSMethod()
        series = _make_series(50)
        trained = type("T", (), {"artifact_bytes": b"nf"})()
        m.train = lambda *args, **kwargs: trained
        m.deserialize_artifact = lambda data: object()
        m.predict_with_model = lambda *args, **kwargs: ForecastResult(
            forecast=np.linspace(100, 110, 12),
            params_used={"max_epochs": 50},
        )
        result = m.forecast(series, horizon=12, seasonality=24, params={})
        assert isinstance(result, ForecastResult)
        assert len(result.forecast) == 12

    def test_forecast_too_few_observations(self):
        m = PatchTSTMethod()
        series = _make_series(3)
        with pytest.raises(ValueError, match="at least 5"):
            m.forecast(series, horizon=12, seasonality=24, params={})
