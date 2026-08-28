from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mtdata.forecast.forecast as ff
import mtdata.forecast.forecast_engine as fe
import mtdata.forecast.volatility as fv
from mtdata.forecast.exceptions import ForecastError


def _sample_df(n: int = 40) -> pd.DataFrame:
    t0 = 1_700_000_000
    times = np.arange(t0, t0 + n * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 110.0, n, dtype=float)
    return pd.DataFrame(
        {
            "time": times,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.linspace(1000.0, 1500.0, n),
        }
    )


def test_create_dimred_reducer_selectkbest_and_unknown():
    from mtdata.forecast.forecast_preprocessing import _create_dimred_reducer

    reducer, meta = _create_dimred_reducer("selectkbest", {"k": "bad"})
    out = reducer.fit_transform(np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=float))
    assert out.shape == (3, 2)
    assert meta["k"] == 5

    with pytest.raises(ValueError, match="Unknown dimensionality reduction method"):
        _create_dimred_reducer("unknown", {})


def test_forecast_validates_timeframe_and_seconds(monkeypatch):
    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    assert "Invalid timeframe" in ff.forecast(symbol="EURUSD", timeframe="BAD")["error"]

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {})
    assert ff.forecast(symbol="EURUSD", timeframe="H1")["error"] == "Unsupported timeframe seconds for H1"


def test_forecast_routes_to_volatility_endpoint(monkeypatch):
    captured = []
    monkeypatch.setattr(ff, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(ff, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(
        fv,
        "forecast_volatility",
        lambda **kwargs: captured.append(kwargs)
        or {"volatility": True, "method": kwargs["method"]},
    )

    denoise = {"method": "wavelet", "causality": "zero_phase"}
    out = ff.forecast(
        symbol="EURUSD",
        timeframe="H1",
        quantity="volatility",
        method="theta",
        denoise=denoise,
    )
    assert out == {"volatility": True, "method": "theta"}
    assert captured[-1]["denoise"] == denoise
    assert captured[-1]["method"] == "theta"

    out = ff.forecast(symbol="EURUSD", timeframe="H1", quantity="price", method="vol_garch")
    assert out == {"volatility": True, "method": "vol_garch"}

    requested = ff.forecast(
        symbol="EURUSD",
        timeframe="H1",
        quantity="volatility",
        method="theta",
        ci_alpha=0.05,
    )
    assert requested["ci_status"] == "requested_but_unavailable"
    assert requested["requested_ci_alpha"] == 0.05


def test_forecast_volatility_uses_and_discloses_top_level_lookback(monkeypatch):
    captured = {}

    def fake_volatility(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "params_used": {"lookback": kwargs["params"]["lookback"]},
            "data_window": {"bars_used": kwargs["params"]["lookback"]},
        }

    monkeypatch.setattr(fv, "forecast_volatility", fake_volatility)

    out = ff.forecast(
        symbol="EURUSD",
        timeframe="H1",
        quantity="volatility",
        method="ewma",
        lookback=50,
    )

    assert captured["params"]["lookback"] == 50
    assert out["requested_lookback"] == 50
    assert out["effective_lookback"] == 50
    assert out["lookback_source"] == "forecast_generate.lookback"
    assert out["data_window"]["lookback_source"] == "forecast_generate.lookback"


def test_forecast_volatility_rejects_conflicting_lookbacks() -> None:
    with pytest.raises(ForecastError, match="Conflicting volatility lookbacks"):
        ff.forecast(
            symbol="EURUSD",
            timeframe="H1",
            quantity="volatility",
            method="ewma",
            lookback=50,
            params={"lookback": 500},
        )


def test_forecast_volatility_quantity_rejects_known_non_volatility_method(monkeypatch):
    monkeypatch.setattr(ff, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(ff, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fv, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fv, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(
        fv,
        "_forecast_method_supports",
        lambda method: {
            "price": True,
            "return": False,
            "volatility": False,
            "ci": True,
        } if method == "analog" else {},
    )

    out = ff.forecast(
        symbol="EURUSD",
        timeframe="H1",
        quantity="volatility",
        method="analog",
    )

    assert "does not support quantity='volatility'" in out["error"]
    assert "forecast_volatility_estimate" in out["error"]


def test_forecast_handles_fetch_errors_and_short_history(monkeypatch):
    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("theta",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))

    monkeypatch.setattr(fe, "_fetch_history", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fetch failed")))
    out = ff.forecast(symbol="EURUSD", timeframe="H1", method="theta")
    assert out["error"] == "fetch failed"

    monkeypatch.setattr(fe, "_fetch_history", lambda *args, **kwargs: _sample_df(2))
    out = ff.forecast(symbol="EURUSD", timeframe="H1", method="theta")
    assert out["error"] == "Not enough closed bars to compute forecast"


def test_forecast_rejects_seasonal_naive_without_positive_seasonality(monkeypatch):
    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("seasonal_naive",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: {"seasonality": 0})

    out = ff.forecast(symbol="EURUSD", timeframe="H1", method="seasonal_naive")
    assert "seasonal_naive requires a positive 'seasonality'" in out["error"]


def test_forecast_delegates_to_engine_without_mutating_params(monkeypatch):
    captured = {}
    params = {"alpha": 1}

    def fake_engine(**kwargs):
        captured.update(kwargs)
        return {"success": True, "method": kwargs["method"], "forecast_price": [1.0] * int(kwargs["horizon"])}

    monkeypatch.setattr(fe, "forecast_engine", fake_engine)

    out = ff.forecast(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        horizon=5,
        quantity="return",
        params=params,
        features={"include": "open,high low", "future_covariates": "hour,dow,fourier:24,is_weekend"},
        model_cache="ephemeral",
    )

    assert out["success"] is True
    assert captured["symbol"] == "EURUSD"
    assert captured["timeframe"] == "H1"
    assert captured["params"] == {"alpha": 1}
    assert captured["features"] == {"include": "open,high low", "future_covariates": "hour,dow,fourier:24,is_weekend"}
    assert captured["model_cache"] == "ephemeral"
    assert params == {"alpha": 1}


def test_forecast_defaults_to_reusable_model_cache(monkeypatch):
    captured = {}

    def fake_engine(**kwargs):
        captured.update(kwargs)
        return {"success": True, "forecast_price": [1.0]}

    monkeypatch.setattr(fe, "forecast_engine", fake_engine)

    out = ff.forecast(symbol="EURUSD", timeframe="H1", method="theta", horizon=1)

    assert out["success"] is True
    assert captured["model_cache"] == "reuse"


def test_forecast_engine_error_passthrough(monkeypatch):
    captured = {}

    def fake_engine(**kwargs):
        captured.update(kwargs)
        return {"error": "engine failure"}

    monkeypatch.setattr(fe, "forecast_engine", fake_engine)

    out = ff.forecast(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        horizon=4,
        features={"include": "open,high"},
        dimred_method="pca",
    )

    assert out["error"] == "engine failure"
    assert captured["features"] == {"include": "open,high"}
    assert captured["dimred_method"] == "pca"


def test_execute_forecast_raises_typed_exception_on_error_payload(monkeypatch):
    monkeypatch.setattr(fe, "forecast_engine", lambda **kwargs: {"error": "engine failure"})

    with pytest.raises(ForecastError, match="engine failure"):
        ff.execute_forecast(symbol="EURUSD", timeframe="H1", method="theta")


def test_forecast_target_spec_is_delegated_without_local_validation(monkeypatch):
    captured = {}

    def fake_engine(**kwargs):
        captured.update(kwargs)
        return {"success": True, "forecast_price": [1.0] * int(kwargs["horizon"])}

    monkeypatch.setattr(fe, "forecast_engine", fake_engine)

    out = ff.forecast(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        horizon=2,
        target_spec={"base": "typical", "transform": "return", "k": 3, "indicators": "sma:close:5"},
    )
    assert out["success"] is True
    assert captured["target_spec"] == {"base": "typical", "transform": "return", "k": 3, "indicators": "sma:close:5"}


def test_forecast_raises_typed_exception_on_unexpected_failure(monkeypatch):
    monkeypatch.setattr(fe, "forecast_engine", lambda **kwargs: (_ for _ in ()).throw(ValueError("engine exploded")))

    with pytest.raises(ForecastError, match="engine exploded"):
        ff.forecast(symbol="EURUSD", timeframe="H1", method="theta")
