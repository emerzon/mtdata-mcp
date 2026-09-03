"""Tests for mtdata.forecast.methods.pretrained – coverage for lines 96-268, 296-492, 525-702."""

from __future__ import annotations

import importlib
import sys
import threading
import types
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub heavy third-party packages before importing the module under test
# ---------------------------------------------------------------------------

def _make_module(name: str, attrs: dict | None = None) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    return mod


# --- torch stub -----------------------------------------------------------
_torch = _make_module("torch")
_torch.float32 = "float32"
_torch.device = MagicMock(side_effect=lambda x: x)
_torch_cuda = _make_module("torch.cuda")
_torch_cuda.is_available = MagicMock(return_value=False)
_torch.cuda = _torch_cuda
_torch.load = MagicMock(return_value={"hyper_parameters": {"model_kwargs": {}}})

class _FakeTensor:
    def __init__(self, data=None, dtype=None):
        self._data = np.array(data) if data is not None else np.zeros(1)
    def unsqueeze(self, dim):
        return self
    def to(self, device):
        return self
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self._data
    @property
    def shape(self):
        return self._data.shape

_torch.tensor = lambda data, dtype=None: _FakeTensor(data, dtype)
_torch_serial = _make_module("torch.serialization")
_torch_serial.add_safe_globals = None
_torch.serialization = _torch_serial

# --- chronos stub ----------------------------------------------------------
_chronos = _make_module("chronos")

class _FakePipeline:
    @classmethod
    def from_pretrained(cls, model_name, device_map=None):
        return cls()

    def predict_df(
        self,
        df,
        future_df=None,
        prediction_length=10,
        quantile_levels=None,
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
        validate_inputs=True,
        **kwargs,
    ):
        h = int(prediction_length)
        targets = [target] if isinstance(target, str) else list(target)
        q_levels = [float(q) for q in (quantile_levels or [0.5])]
        if future_df is not None and timestamp_column in future_df.columns:
            ts_vals = list(future_df[timestamp_column])
        else:
            ts_vals = list(range(h))
        rows = []
        for tgt in targets:
            for idx in range(h):
                row = {
                    id_column: "series_0",
                    timestamp_column: ts_vals[idx] if idx < len(ts_vals) else idx,
                    "target_name": str(tgt),
                    "predictions": float(idx + 1),
                }
                for q in q_levels:
                    row[str(q)] = float(idx + 1 + q)
                    row[f"{q:g}"] = float(idx + 1 + q)
                rows.append(row)
        return pd.DataFrame(rows)

    def predict_quantiles(self, ctx, prediction_length=10, quantile_levels=None, **kw):
        h = prediction_length
        n_q = len(quantile_levels) if quantile_levels else 1
        q_tensor = _FakeTensor(np.random.rand(1, h, n_q))
        m_tensor = _FakeTensor(np.random.rand(1, h))
        return q_tensor, m_tensor

    def predict(self, ctx, prediction_length=10, **kw):
        return _FakeTensor(np.random.rand(1, prediction_length))

_chronos.ChronosBoltPipeline = _FakePipeline
_chronos.Chronos2Pipeline = _FakePipeline
_chronos.ChronosPipeline = _FakePipeline

# --- timesfm stub ----------------------------------------------------------
_timesfm = _make_module("timesfm")

class _FakeForecastConfig:
    def __init__(self, **kw):
        pass

_timesfm.ForecastConfig = _FakeForecastConfig

class _FakeTimesFMTorch:
    def __init__(self, **kw):
        pass
    def load_checkpoint(self):
        pass
    def compile(self, cfg):
        pass
    def forecast(self, inputs, horizon):
        arr = np.ones((1, horizon))
        return arr, None

_timesfm_torch = _make_module("timesfm.torch")
_timesfm_torch.TimesFM_2p5_200M_torch = _FakeTimesFMTorch
_timesfm_2p5 = _make_module("timesfm.timesfm_2p5")
_timesfm_2p5_torch = _make_module("timesfm.timesfm_2p5.timesfm_2p5_torch")
_timesfm_configs = _make_module("timesfm.configs")
_timesfm_configs.ForecastConfig = _FakeForecastConfig

class _FakeTimesFM3Torch:
    def __init__(self, **kw):
        raise AssertionError("TimesFM 2.5 adapter must not instantiate TimesFM3Torch")

_timesfm.TimesFM3Torch = _FakeTimesFM3Torch

# --- timesfm3 stub ---------------------------------------------------------
_timesfm3 = _make_module("timesfm3")
_timesfm3_forecaster = _make_module("timesfm3.timesfm3_forecaster")


class _FakeTimesFM3Config:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.checkpoint_path = kw.get("checkpoint_path", "google/timesfm-3.0-pytorch")
        self.device = kw.get("device")
        self.per_core_batch_size = kw.get("per_core_batch_size", 4)


class _FakeTimesFM3Output:
    def __init__(self, forecast, quantiles=None):
        self.forecast = forecast
        self.quantiles = quantiles


class _FakeTimesFM3Forecaster:
    instances = 0
    last_predict = None

    def __init__(self, config=None, **kw):
        type(self).instances += 1
        self.config = config
        self.kw = kw

    def predict(self, context, horizon=12, **kwargs):
        type(self).last_predict = {
            "context": np.asarray(context),
            "horizon": int(horizon),
            **kwargs,
        }
        h = int(horizon)
        return _FakeTimesFM3Output(np.ones(h), np.ones((h, 9)))


_timesfm3.TimesFM3Forecaster = _FakeTimesFM3Forecaster
_timesfm3.ModelConfig = _FakeTimesFM3Config
_timesfm3_forecaster.TimesFM3Forecaster = _FakeTimesFM3Forecaster
_timesfm3_forecaster.ModelConfig = _FakeTimesFM3Config
_timesfm3_forecaster.ForecastOutput = _FakeTimesFM3Output

# --- Register all stubs in sys.modules ------------------------------------
# Stubs are installed at module level for the initial import, then
# re-installed before each test (via autouse fixture) to survive cleanup
# by other test modules that share stub keys.
_TORCH_STUBS = {
    "torch": _torch,
    "torch.cuda": _torch_cuda,
    "torch.serialization": _torch_serial,
}
_NON_TORCH_STUBS = {
    "chronos": _chronos,
    "timesfm": _timesfm,
    "timesfm.torch": _timesfm_torch,
    "timesfm.timesfm_2p5": _timesfm_2p5,
    "timesfm.timesfm_2p5.timesfm_2p5_torch": _timesfm_2p5_torch,
    "timesfm.configs": _timesfm_configs,
    "timesfm3": _timesfm3,
    "timesfm3.timesfm3_forecaster": _timesfm3_forecaster,
}

# Install stubs for the module-level import of pretrained
_originals = {}
for name, mod in _NON_TORCH_STUBS.items():
    _originals[name] = sys.modules.get(name)
    sys.modules[name] = mod

# Now import the module under test
import mtdata.forecast.methods.pretrained as pretrained_module
from mtdata.forecast.interface import ForecastResult
from mtdata.forecast.methods.pretrained import (
    ChronosBoltMethod,
    PretrainedMethod,
    TimesFM3Method,
    TimesFMMethod,
    _build_chronos_inputs,
    _ensure_chronos2_history_df,
    _extract_chronos2_predict_df_output,
    _is_chronos2_pipeline,
    _load_chronos_pipeline,
    _resolve_chronos2_multivariate_columns,
    _resolve_chronos_device_map,
    _resolve_chronos_model_defaults,
    _unwrap_chronos_predict,
    _unwrap_chronos_quantiles,
)

# ---------------------------------------------------------------------------
# Per-test fixtures to ensure stubs are present even if other modules
# cleaned them up between tests.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _ensure_stubs():
    """Re-install non-torch stubs before every test & clean up after."""
    saved = {}
    for name, mod in _NON_TORCH_STUBS.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod
    yield
    for name, orig in saved.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig


@pytest.fixture()
def _with_torch_stubs():
    """Temporarily inject torch stubs for a single test, then clean up."""
    saved = {}
    for name, mod in _TORCH_STUBS.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod
    yield
    for name, orig in saved.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig
    try:
        from scipy._lib.array_api_compat.common._helpers import _issubclass_fast
        _issubclass_fast.cache_clear()
    except Exception:
        pass


# ===========================================================================
# Helpers
# ===========================================================================

def _series(n: int = 100) -> pd.Series:
    return pd.Series(np.random.rand(n) * 100 + 50, name="price")


# ===========================================================================
# _resolve_chronos_device_map
# ===========================================================================

class TestResolveChronosDeviceMap:
    def test_empty_no_cuda(self):
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.return_value = False
        assert _resolve_chronos_device_map(None, torch_mod) == "cpu"

    def test_empty_with_cuda(self):
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.return_value = True
        assert _resolve_chronos_device_map("", torch_mod) == "cuda:0"

    def test_explicit_cpu(self):
        assert _resolve_chronos_device_map("cpu", MagicMock()) == "cpu"

    def test_explicit_cuda1(self):
        assert _resolve_chronos_device_map("cuda:1", MagicMock()) == "cuda:1"

    def test_auto_single_gpu(self):
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.return_value = True
        torch_mod.cuda.device_count.return_value = 1
        assert _resolve_chronos_device_map("auto", torch_mod) == "cuda:0"

    def test_auto_multi_gpu(self):
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.return_value = True
        torch_mod.cuda.device_count.return_value = 4
        assert _resolve_chronos_device_map("auto", torch_mod) == "cuda:0"

    def test_auto_no_cuda(self):
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.return_value = False
        assert _resolve_chronos_device_map("auto", torch_mod) == "cpu"

    def test_none_cuda_exception(self):
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.side_effect = RuntimeError("no cuda")
        assert _resolve_chronos_device_map(None, torch_mod) == "cpu"

    def test_whitespace_treated_as_empty(self):
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.return_value = False
        assert _resolve_chronos_device_map("   ", torch_mod) == "cpu"


# ===========================================================================
# PretrainedMethod base class
# ===========================================================================

class TestPretrainedMethodBase:
    def test_metadata(self):
        m = ChronosBoltMethod()
        assert m.category == "pretrained"
        feats = m.supports_features
        assert feats["price"] is True
        assert feats["ci"] is True


# ===========================================================================
# ChronosBoltMethod (lines 96-268)
# ===========================================================================

@pytest.mark.usefixtures("_with_torch_stubs")
class TestChronosBoltMethod:
    def setup_method(self):
        self.method = ChronosBoltMethod()

    def test_name_and_packages(self):
        assert self.method.name == "chronos_bolt"
        assert "chronos" in self.method.required_packages
        assert "torch" in self.method.required_packages

    def test_forecast_basic(self):
        res = self.method.forecast(_series(), horizon=10, seasonality=1, params={})
        assert isinstance(res, ForecastResult)
        assert res.forecast is not None

    def test_forecast_with_model_name(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={"model_name": "amazon/chronos-bolt-small"})
        assert res.params_used["model_name"] == "amazon/chronos-bolt-small"

    def test_forecast_with_context_length(self):
        res = self.method.forecast(_series(200), horizon=10, seasonality=1, params={"context_length": 50})
        assert res is not None

    def test_forecast_zero_context_length(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={"context_length": 0})
        assert res.forecast is not None

    def test_forecast_with_quantiles(self):
        res = self.method.forecast(_series(), horizon=10, seasonality=1, params={"quantiles": [0.1, 0.5, 0.9]})
        assert "quantiles" in res.metadata

    def test_forecast_device_map_cpu(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={"device_map": "cpu"})
        assert res.params_used["device_map"] == "cpu"

    def test_forecast_short_series(self):
        res = self.method.forecast(_series(5), horizon=3, seasonality=1, params={})
        assert res.forecast is not None

    def test_forecast_long_horizon(self):
        res = self.method.forecast(_series(), horizon=50, seasonality=1, params={})
        assert res.forecast is not None

    def test_forecast_with_exog(self):
        s = _series(50)
        exog_hist = np.random.rand(50, 2)
        exog_fut = np.random.rand(10, 2)
        res = self.method.forecast(s, horizon=10, seasonality=1, params={}, exog_used=exog_hist, exog_future=exog_fut)
        assert res.forecast is not None

    def test_forecast_exog_in_params(self):
        s = _series(50)
        res = self.method.forecast(s, horizon=10, seasonality=1, params={
            "exog_used": np.random.rand(50, 2),
            "exog_future": np.random.rand(10, 2),
        })
        assert res.forecast is not None

    def test_forecast_none_params(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params=None)
        assert res.forecast is not None

    def test_default_model_name(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={"method_name": "chronos_bolt"})
        assert "chronos-bolt-base" in res.params_used["model_name"]
        assert res.params_used.get("pipeline") == "ChronosBoltPipeline"

    def test_chronos2_default_model_name_uses_chronos2(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={"method_name": "chronos2"})
        assert res.params_used["model_name"] == "amazon/chronos-2"
        assert res.params_used.get("pipeline") == "Chronos2Pipeline"

    def test_prepare_forecast_call_injects_method_name(self):
        context = types.SimpleNamespace(method="chronos2")
        params, kwargs = self.method.prepare_forecast_call({}, {}, context)
        assert params["method_name"] == "chronos2"
        assert kwargs == {}

    def test_falls_back_when_chronos2_pipeline_init_fails(self):
        class _BrokenChronos2Pipeline:
            @classmethod
            def from_pretrained(cls, model_name, device_map=None):
                raise AttributeError("module 'chronos.chronos2' has no attribute 'ChronosBoltModelForForecasting'")

        saved = getattr(_chronos, "Chronos2Pipeline", None)
        _chronos.Chronos2Pipeline = _BrokenChronos2Pipeline
        pretrained_module.model_cache.clear()
        try:
            with pytest.raises(RuntimeError, match="failed to initialize any compatible Chronos pipeline"):
                self.method.forecast(_series(), horizon=5, seasonality=1, params={"model_name": "amazon/chronos-2", "method_name": "chronos2"})
        finally:
            if saved is None:
                delattr(_chronos, "Chronos2Pipeline")
            else:
                _chronos.Chronos2Pipeline = saved


def test_resolve_chronos_model_defaults_uses_t5_for_chronos2():
    model_name, order = _resolve_chronos_model_defaults("chronos2", {})
    assert model_name == "amazon/chronos-2"
    assert order == ("Chronos2Pipeline",)


def test_resolve_chronos_model_defaults_uses_bolt_for_chronos_bolt():
    model_name, order = _resolve_chronos_model_defaults("chronos_bolt", {})
    assert model_name == "amazon/chronos-bolt-base"
    assert order == ("ChronosBoltPipeline",)


def test_is_chronos2_pipeline_matches_name_or_model() -> None:
    assert _is_chronos2_pipeline("Chronos2Pipeline", "anything") is True
    assert _is_chronos2_pipeline(None, "amazon/chronos-2") is True
    assert _is_chronos2_pipeline("ChronosPipeline", "amazon/chronos-t5-small") is False


@pytest.mark.usefixtures("_with_torch_stubs")
def test_build_chronos_inputs_uses_list_for_chronos2() -> None:
    built = _build_chronos_inputs(np.array([1.0, 2.0, 3.0]), None, "Chronos2Pipeline", "amazon/chronos-2", _torch)
    assert isinstance(built, list)
    assert len(built) == 1


@pytest.mark.usefixtures("_with_torch_stubs")
def test_build_chronos_inputs_keeps_tensor_for_legacy_chronos() -> None:
    built = _build_chronos_inputs(np.array([1.0, 2.0, 3.0]), None, "ChronosPipeline", "amazon/chronos-t5-small", _torch)
    assert isinstance(built, _FakeTensor)


@pytest.mark.usefixtures("_with_torch_stubs")
def test_unwrap_chronos2_quantiles_accepts_list_outputs() -> None:
    q_np, mean_np = _unwrap_chronos_quantiles(
        [_FakeTensor(np.ones((1, 5, 2)))],
        [_FakeTensor(np.arange(5, dtype=float).reshape(1, 5))],
        model_name="amazon/chronos-2",
        pipe_name="Chronos2Pipeline",
    )
    assert q_np.shape == (5, 2)
    assert mean_np.shape == (5,)


@pytest.mark.usefixtures("_with_torch_stubs")
def test_unwrap_chronos2_predict_accepts_list_outputs() -> None:
    f_np = _unwrap_chronos_predict(
        [_FakeTensor(np.arange(5, dtype=float).reshape(1, 5))],
        model_name="amazon/chronos-2",
        pipe_name="Chronos2Pipeline",
    )
    assert f_np.shape == (5,)


def test_resolve_chronos2_multivariate_columns_uses_include_and_tis_when_enabled() -> None:
    history = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "close": [10.0, 11.0, 12.0, 13.0],
            "open": [9.0, 10.0, 11.0, 12.0],
            "rsi_14": [50.0, 51.0, 52.0, 53.0],
        }
    )
    cols = _resolve_chronos2_multivariate_columns(
        history,
        "close",
        {"chronos2_multivariate": True},
        {"include_columns": ["open"], "indicator_columns": ["rsi_14"]},
    )
    assert cols == ["close", "open", "rsi_14"]


def test_resolve_chronos2_multivariate_columns_defaults_to_base_when_disabled() -> None:
    history = pd.DataFrame({"time": [1.0, 2.0, 3.0], "close": [1.0, 2.0, 3.0], "open": [1.0, 2.0, 3.0]})
    cols = _resolve_chronos2_multivariate_columns(
        history,
        "close",
        {},
        {"include_columns": ["open"], "indicator_columns": []},
    )
    assert cols == ["close"]


def test_ensure_chronos2_history_df_injects_missing_base_column() -> None:
    history = pd.DataFrame({"time": [1.0, 2.0, 3.0], "close": [10.0, 11.0, 12.0]})
    series = pd.Series([0.1, 0.2, 0.3], index=history.index, name="__log_return")
    frame = _ensure_chronos2_history_df(history, series=series, base_col="__log_return", timeframe="H1")
    assert "__log_return" in frame.columns
    assert frame["__log_return"].tolist() == [0.1, 0.2, 0.3]


def test_extract_chronos2_predict_df_output_returns_primary_and_multivariate_metadata() -> None:
    pred_df = pd.DataFrame(
        {
            "item_id": ["series_0"] * 4,
            "timestamp": [1, 2, 1, 2],
            "target_name": ["close", "close", "rsi_14", "rsi_14"],
            "predictions": [10.0, 11.0, 50.0, 51.0],
            "0.5": [10.5, 11.5, 50.5, 51.5],
        }
    )
    f_vals, fq, meta = _extract_chronos2_predict_df_output(pred_df, primary_target="close", quantile_levels=[0.5])
    assert f_vals.tolist() == [10.0, 11.0]
    assert fq == {"0.5": [10.5, 11.5]}
    assert meta["multivariate_forecasts"]["rsi_14"]["forecast"] == [50.0, 51.0]


@pytest.mark.usefixtures("_with_torch_stubs")
def test_load_chronos_pipeline_serializes_concurrent_init(monkeypatch):
    pretrained_module.model_cache.clear()

    call_count = [0]
    started = threading.Event()
    release = threading.Event()

    class _BlockingPipeline(_FakePipeline):
        @classmethod
        def from_pretrained(cls, model_name, device_map=None):
            call_count[0] += 1
            started.set()
            release.wait(timeout=2)
            return cls()

    results = []
    errors = []

    def _worker():
        try:
            results.append(
                _load_chronos_pipeline(
                    [("ChronosPipeline", _BlockingPipeline)],
                    "amazon/chronos-2",
                    "cuda:0",
                )
            )
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=_worker)
    t2 = threading.Thread(target=_worker)
    t1.start()
    assert started.wait(timeout=2)
    t2.start()
    release.set()
    t1.join(timeout=2)
    t2.join(timeout=2)

    assert not errors
    assert len(results) == 2
    assert call_count[0] == 1
    assert results[0][1] == "ChronosPipeline"
    assert results[1][1] == "ChronosPipeline"
    assert results[0][0] is results[1][0]

    def test_predict_quantiles_fallback_to_predict(self):
        """When predict_quantiles returns None, falls back to predict."""
        orig = _FakePipeline.predict_quantiles
        _FakePipeline.predict_quantiles = lambda self, *a, **k: (None, None)
        try:
            res = self.method.forecast(_series(), horizon=5, seasonality=1, params={})
            assert res.forecast is not None
        finally:
            _FakePipeline.predict_quantiles = orig

    def test_from_pretrained_type_error_fallback(self):
        """When from_pretrained raises TypeError on device_map kwarg, retries without."""
        call_count = [0]
        orig = _FakePipeline.from_pretrained

        @classmethod
        def _new_from_pretrained(cls, model_name, device_map=None):
            call_count[0] += 1
            if call_count[0] == 1 and device_map is not None:
                raise TypeError("unexpected kwarg device_map")
            return orig.__func__(cls, model_name)

        _FakePipeline.from_pretrained = _new_from_pretrained
        try:
            res = self.method.forecast(_series(), horizon=5, seasonality=1, params={})
            assert res.forecast is not None
        finally:
            _FakePipeline.from_pretrained = orig

    def test_predict_quantiles_type_error_retry(self):
        """predict_quantiles TypeError triggers retry without extra kwargs."""
        call_count = [0]
        orig_pq = _FakePipeline.predict_quantiles

        def _new_pq(self_pipe, ctx, prediction_length=10, quantile_levels=None, **kw):
            call_count[0] += 1
            if call_count[0] == 1 and kw:
                raise TypeError("unexpected kwargs")
            return orig_pq(self_pipe, ctx, prediction_length=prediction_length, quantile_levels=quantile_levels)

        _FakePipeline.predict_quantiles = _new_pq
        try:
            res = self.method.forecast(_series(), horizon=5, seasonality=1, params={"quantiles": [0.5]})
            assert res.forecast is not None
        finally:
            _FakePipeline.predict_quantiles = orig_pq

    def test_no_pipeline_found(self):
        """Error when chronos module has no pipeline classes."""
        saved = {}
        for attr in ("Chronos2Pipeline", "ChronosBoltPipeline", "ChronosPipeline"):
            saved[attr] = getattr(_chronos, attr, None)
            if hasattr(_chronos, attr):
                delattr(_chronos, attr)
        try:
            with pytest.raises(RuntimeError, match="no supported pipeline found"):
                self.method.forecast(_series(), horizon=5, seasonality=1, params={})
        finally:
            for attr, val in saved.items():
                if val is not None:
                    setattr(_chronos, attr, val)

    def test_quantile_shape_q_axis_0(self):
        """Quantile tensor shape (n_q, h) -> q_axis=0."""
        def _pq(self_pipe, ctx, prediction_length=10, quantile_levels=None, **kw):
            n_q = len(quantile_levels) if quantile_levels else 1
            return _FakeTensor(np.random.rand(n_q, prediction_length)), _FakeTensor(np.random.rand(1, prediction_length))
        orig = _FakePipeline.predict_quantiles
        _FakePipeline.predict_quantiles = _pq
        try:
            res = self.method.forecast(_series(), horizon=10, seasonality=1, params={"quantiles": [0.1, 0.9]})
            assert res.metadata["quantiles"]
        finally:
            _FakePipeline.predict_quantiles = orig

    def test_forecast_pads_short_chronos_outputs_to_requested_horizon(self):
        orig = _FakePipeline.predict_quantiles

        def _pq(self_pipe, ctx, prediction_length=10, quantile_levels=None, **kw):
            short_h = max(1, int(prediction_length) - 2)
            n_q = len(quantile_levels) if quantile_levels else 1
            return _FakeTensor(np.ones((1, short_h, n_q))), _FakeTensor(np.arange(short_h, dtype=float).reshape(1, short_h))

        _FakePipeline.predict_quantiles = _pq
        try:
            res = self.method.forecast(_series(), horizon=5, seasonality=1, params={"quantiles": [0.5]})
            assert len(res.forecast) == 5
            assert res.forecast.tolist() == [0.0, 1.0, 2.0, 2.0, 2.0]
        finally:
            _FakePipeline.predict_quantiles = orig

    def test_import_error_raises(self):
        """When chronos not importable, RuntimeError is raised."""
        saved = sys.modules.get("chronos")
        sys.modules["chronos"] = None  # makes __import__ fail
        try:
            m = ChronosBoltMethod()
            with pytest.raises(RuntimeError):
                m.forecast(_series(), horizon=5, seasonality=1, params={})
        finally:
            sys.modules["chronos"] = saved


# ===========================================================================
# TimesFMMethod (lines 296-492)
# ===========================================================================

@pytest.mark.usefixtures("_with_torch_stubs")
class TestTimesFMMethod:
    def setup_method(self):
        self.method = TimesFMMethod()

    def test_name_and_packages(self):
        assert self.method.name == "timesfm"
        assert "timesfm" in self.method.required_packages

    def test_forecast_basic(self):
        res = self.method.forecast(_series(), horizon=10, seasonality=1, params={})
        assert isinstance(res, ForecastResult)
        assert res.forecast is not None
        assert len(res.forecast) == 10

    def test_forecast_with_context_length(self):
        res = self.method.forecast(_series(200), horizon=10, seasonality=1, params={"context_length": 64})
        assert res.forecast is not None

    def test_forecast_with_quantiles_dict(self):
        """Quantiles returned as dict from forecast call."""
        def _forecast_dict(self_mdl, inputs, horizon):
            return np.ones((1, horizon)), {"0.5": np.ones(horizon)}
        orig = _FakeTimesFMTorch.forecast
        _FakeTimesFMTorch.forecast = _forecast_dict
        try:
            res = self.method.forecast(_series(), horizon=10, seasonality=1, params={"quantiles": [0.5]})
            assert res.forecast is not None
        finally:
            _FakeTimesFMTorch.forecast = orig

    def test_forecast_with_quantiles_array(self):
        """Quantiles returned as 3-D array."""
        def _forecast_arr(self_mdl, inputs, horizon):
            return np.ones((1, horizon)), np.random.rand(1, horizon, 9)
        orig = _FakeTimesFMTorch.forecast
        _FakeTimesFMTorch.forecast = _forecast_arr
        try:
            res = self.method.forecast(_series(), horizon=10, seasonality=1, params={"quantiles": [0.1, 0.5, 0.9]})
            assert res.forecast is not None
        finally:
            _FakeTimesFMTorch.forecast = orig

    def test_forecast_model_class_override(self):
        res = self.method.forecast(_series(), horizon=5, seasonality=1, params={"model_class": "TimesFM_2p5_200M_torch"})
        assert res.forecast is not None

    def test_forecast_config_kwargs(self):
        res = self.method.forecast(_series(), horizon=10, seasonality=1, params={"context_length": 128})
        assert res.forecast is not None

    def test_forecast_none_point_raises(self):
        """When forecast returns None point forecast, RuntimeError raised."""
        def _forecast_none(self_mdl, inputs, horizon):
            return None, None
        orig = _FakeTimesFMTorch.forecast
        _FakeTimesFMTorch.forecast = _forecast_none
        try:
            with pytest.raises(RuntimeError, match="no point forecast"):
                self.method.forecast(_series(), horizon=5, seasonality=1, params={})
        finally:
            _FakeTimesFMTorch.forecast = orig

    def test_forecast_2d_point(self):
        """2-D point forecast array handled correctly."""
        def _forecast_2d(self_mdl, inputs, horizon):
            return np.ones((1, horizon)) * 42, None
        orig = _FakeTimesFMTorch.forecast
        _FakeTimesFMTorch.forecast = _forecast_2d
        try:
            res = self.method.forecast(_series(), horizon=5, seasonality=1, params={})
            np.testing.assert_allclose(res.forecast, 42.0)
        finally:
            _FakeTimesFMTorch.forecast = orig

    def test_forecast_constructor_type_error(self):
        """Constructor TypeError falls back with device kwarg."""
        call_count = [0]
        orig_init = _FakeTimesFMTorch.__init__

        def _new_init(self_mdl, **kw):
            call_count[0] += 1
            if call_count[0] == 1 and not kw:
                raise TypeError("needs device")
            orig_init(self_mdl, **kw)

        _FakeTimesFMTorch.__init__ = _new_init
        try:
            res = self.method.forecast(_series(), horizon=5, seasonality=1, params={})
            assert res.forecast is not None
        finally:
            _FakeTimesFMTorch.__init__ = orig_init

    def test_resolve_forecast_config_missing(self):
        """When ForecastConfig missing, raises RuntimeError."""
        saved = _timesfm.ForecastConfig
        del _timesfm.ForecastConfig
        _timesfm_configs.ForecastConfig = None
        try:
            with pytest.raises(RuntimeError, match="ForecastConfig"):
                self.method.forecast(_series(), horizon=5, seasonality=1, params={})
        finally:
            _timesfm.ForecastConfig = saved
            _timesfm_configs.ForecastConfig = _FakeForecastConfig

    def test_import_error_raises(self):
        saved = sys.modules.get("timesfm")
        sys.modules["timesfm"] = None
        try:
            m = TimesFMMethod()
            with pytest.raises(RuntimeError):
                m.forecast(_series(), horizon=5, seasonality=1, params={})
        finally:
            sys.modules["timesfm"] = saved

    def test_short_series(self):
        res = self.method.forecast(_series(3), horizon=2, seasonality=1, params={})
        assert len(res.forecast) == 2

    def test_forecast_call_signature_fallbacks(self):
        """_call_forecast tries multiple signatures."""
        call_count = [0]
        def _tricky_forecast(self_mdl, inputs, horizon):
            call_count[0] += 1
            return np.ones(horizon), None
        orig = _FakeTimesFMTorch.forecast
        _FakeTimesFMTorch.forecast = _tricky_forecast
        try:
            res = self.method.forecast(_series(), horizon=5, seasonality=1, params={})
            assert res.forecast is not None
        finally:
            _FakeTimesFMTorch.forecast = orig

    def test_forecast_returns_dict(self):
        """Model forecast returns dict instead of tuple."""
        def _forecast_dict(self_mdl, inputs, horizon):
            return {"point_forecast": np.ones(horizon), "quantiles": None}
        # The internal _call_forecast tries tuple, then dict, then raw
        orig = _FakeTimesFMTorch.forecast
        _FakeTimesFMTorch.forecast = lambda self, **kw: {"point_forecast": np.ones(kw.get("horizon", 5))}
        try:
            res = self.method.forecast(_series(), horizon=5, seasonality=1, params={})
            assert res.forecast is not None
        finally:
            _FakeTimesFMTorch.forecast = orig


@pytest.mark.parametrize(
    "name",
    ["forecast_chronos_bolt", "forecast_timesfm"],
)
def test_pretrained_wrapper_functions_removed(name):
    with pytest.raises(AttributeError):
        getattr(pretrained_module, name)


@pytest.mark.usefixtures("_with_torch_stubs")
class TestPretrainedErrorContext:
    def test_timesfm_forecast_preserves_root_cause(self, monkeypatch):
        def _boom(self, inputs, horizon):
            raise ValueError("predict failed")

        monkeypatch.setattr(_FakeTimesFMTorch, "forecast", _boom)

        method = TimesFMMethod()
        with pytest.raises(RuntimeError, match="timesfm error: predict failed") as exc_info:
            method.forecast(_series(50), 5, 0, {})

        assert isinstance(exc_info.value.__cause__, ValueError)
        assert str(exc_info.value.__cause__) == "predict failed"


@pytest.mark.usefixtures("_with_torch_stubs")
class TestTimesFM3Method:
    def setup_method(self):
        self.method = TimesFM3Method()
        _FakeTimesFM3Forecaster.instances = 0
        _FakeTimesFM3Forecaster.last_predict = None
        pretrained_module.model_cache.clear()
        pretrained_module._TIMESFM3_LICENSE_WARNED = False

    def test_name_and_packages(self):
        assert self.method.name == "timesfm3"
        assert "timesfm" in self.method.required_packages
        assert self.method.supports_historical_exog is True
        assert self.method.supports_future_exog is True

    def test_forecast_basic(self):
        res = self.method.forecast(_series(), horizon=10, seasonality=1, params={})
        assert isinstance(res, ForecastResult)
        assert res.forecast is not None
        assert len(res.forecast) == 10
        assert res.params_used["timesfm_model"] == "TimesFM3Forecaster"
        assert res.params_used["weights_license"] == "timesfm-non-commercial-license-v1.0"
        assert res.params_used["checkpoint_path"] == "google/timesfm-3.0-pytorch"

    def test_forecast_with_context_length(self):
        res = self.method.forecast(
            _series(200), horizon=8, seasonality=1, params={"context_length": 64}
        )
        assert res.forecast is not None
        assert len(_FakeTimesFM3Forecaster.last_predict["context"]) == 64

    def test_forecast_with_quantiles(self):
        res = self.method.forecast(
            _series(), horizon=6, seasonality=1, params={"quantiles": [0.1, 0.5, 0.9]}
        )
        assert res.forecast is not None
        quantiles = (res.metadata or {}).get("quantiles") or {}
        assert set(quantiles) >= {"0.1", "0.5", "0.9"}
        assert len(quantiles["0.5"]) == 6
        assert _FakeTimesFM3Forecaster.last_predict["return_quantiles"] is True

    def test_default_flags_disable_evaluator_benchmark_defaults(self):
        self.method.forecast(_series(), horizon=4, seasonality=1, params={})
        call = _FakeTimesFM3Forecaster.last_predict
        assert call["use_symmetric_averaging"] is False
        assert call["make_positive"] is False

    def test_past_only_covariates_transpose(self):
        exog = np.arange(20, dtype=float).reshape(10, 2)
        self.method.forecast(
            _series(10),
            horizon=3,
            seasonality=1,
            params={},
            exog_used=exog,
        )
        past_only = _FakeTimesFM3Forecaster.last_predict["past_only_covariates"]
        assert past_only.shape == (2, 10)
        np.testing.assert_allclose(past_only, exog.T)

    def test_feature_consumption_attestation(self):
        hist = np.ones((10, 2), dtype=float)
        future = np.full((4, 2), 2.0, dtype=float)
        res = self.method.forecast(
            _series(10),
            horizon=4,
            seasonality=1,
            params={},
            exog_used=hist,
            exog_future=future,
        )
        consumed = ((res.metadata or {}).get("diagnostics") or {}).get("feature_consumption")
        assert consumed == {
            "status": "consumed",
            "historical_consumed": True,
            "future_consumed": True,
            "historical_rows": 10,
            "future_rows": 4,
            "n_features": 2,
            "adapter_columns": ["x0", "x1"],
        }

    def test_past_future_covariates_concat(self):
        hist = np.ones((10, 2), dtype=float)
        future = np.full((4, 2), 2.0, dtype=float)
        self.method.forecast(
            _series(10),
            horizon=4,
            seasonality=1,
            params={},
            exog_used=hist,
            exog_future=future,
        )
        call = _FakeTimesFM3Forecaster.last_predict
        assert "past_only_covariates" not in call
        past_future = call["past_future_covariates"]
        assert past_future.shape == (2, 14)
        np.testing.assert_allclose(past_future[:, :10], 1.0)
        np.testing.assert_allclose(past_future[:, 10:], 2.0)

    def test_covariate_feature_mismatch_raises(self):
        with pytest.raises(RuntimeError, match="do not match"):
            self.method.forecast(
                _series(8),
                horizon=3,
                seasonality=1,
                params={},
                exog_used=np.ones((8, 2)),
                exog_future=np.ones((3, 3)),
            )

    def test_future_exog_without_history_raises(self):
        with pytest.raises(RuntimeError, match="requires matching historical"):
            self.method.forecast(
                _series(8),
                horizon=3,
                seasonality=1,
                params={},
                exog_future=np.ones((3, 2)),
            )

    def test_cache_reuses_forecaster(self):
        self.method.forecast(_series(), horizon=5, seasonality=1, params={})
        self.method.forecast(_series(), horizon=5, seasonality=1, params={})
        assert _FakeTimesFM3Forecaster.instances == 1

    def test_none_point_forecast_raises(self):
        orig = _FakeTimesFM3Forecaster.predict

        def _none(self_mdl, context, horizon=12, **kwargs):
            return _FakeTimesFM3Output(None, None)

        _FakeTimesFM3Forecaster.predict = _none
        try:
            with pytest.raises(RuntimeError, match="no point forecast"):
                self.method.forecast(_series(), horizon=5, seasonality=1, params={})
        finally:
            _FakeTimesFM3Forecaster.predict = orig

    def test_import_error_raises(self):
        saved = sys.modules.get("timesfm3")
        sys.modules["timesfm3"] = None
        try:
            m = TimesFM3Method()
            with pytest.raises(RuntimeError, match="timesfm3 requires"):
                m.forecast(_series(), horizon=5, seasonality=1, params={})
        finally:
            sys.modules["timesfm3"] = saved

    def test_root_cause_preserved(self):
        orig = _FakeTimesFM3Forecaster.predict

        def _boom(self_mdl, context, horizon=12, **kwargs):
            raise ValueError("decode failed")

        _FakeTimesFM3Forecaster.predict = _boom
        try:
            with pytest.raises(RuntimeError, match="timesfm3 error: decode failed") as exc_info:
                self.method.forecast(_series(), horizon=5, seasonality=1, params={})
            assert isinstance(exc_info.value.__cause__, ValueError)
        finally:
            _FakeTimesFM3Forecaster.predict = orig


@pytest.mark.usefixtures("_with_torch_stubs")
def test_timesfm_2p5_does_not_select_timesfm3_class():
    saved_torch = getattr(_timesfm_torch, "TimesFM_2p5_200M_torch", None)
    saved_2p5 = getattr(_timesfm_2p5_torch, "TimesFM_2p5_200M_torch", None)
    try:
        if saved_torch is not None:
            delattr(_timesfm_torch, "TimesFM_2p5_200M_torch")
        if saved_2p5 is not None:
            delattr(_timesfm_2p5_torch, "TimesFM_2p5_200M_torch")
        method = TimesFMMethod()
        with pytest.raises(RuntimeError, match="no TimesFM 2.5"):
            method.forecast(_series(), horizon=5, seasonality=1, params={})
    finally:
        if saved_torch is not None:
            _timesfm_torch.TimesFM_2p5_200M_torch = saved_torch
        if saved_2p5 is not None:
            _timesfm_2p5_torch.TimesFM_2p5_200M_torch = saved_2p5
