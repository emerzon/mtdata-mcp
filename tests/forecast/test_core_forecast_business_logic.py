from __future__ import annotations

import importlib
import logging
import sys
from inspect import signature
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError

from mtdata.core import forecast as cf
from mtdata.core import options as opt
from mtdata.core._mcp_tools import _select_output_fields
from mtdata.forecast import barriers_shared
from mtdata.forecast import use_cases as forecast_use_cases
from mtdata.forecast.exceptions import ForecastError, ModelCompatibilityError
from mtdata.forecast.requests import (
    ForecastBacktestRequest,
    ForecastBarrierOptimizeRequest,
    ForecastBarrierProbRequest,
    ForecastConformalIntervalsRequest,
    ForecastGenerateRequest,
    ForecastOptimizeHintsRequest,
    ForecastTuneGeneticRequest,
    ForecastTuneOptunaRequest,
    ForecastVolatilityEstimateRequest,
)
from mtdata.forecast.use_cases import compact as forecast_compact
from mtdata.forecast.use_cases import sktime_index as forecast_sktime_index
from mtdata.utils.mt5 import MT5ConnectionError


def _unwrap(fn):
    current = fn
    while hasattr(current, "__wrapped__"):
        current = current.__wrapped__
    return current


def test_stored_alias_rejects_conflicting_selector():
    with pytest.raises(ForecastError, match="conflicting selector"):
        forecast_use_cases._resolve_stored_model_execution_alias(
            library="statsforecast",
            requested_method="sf_naive",
            resolved_method="statsforecast",
            params={"model_name": "Theta"},
            original_params={"model_name": "Theta"},
            model_id="sf_naive/EURUSD_H1/hash",
        )


@pytest.fixture(autouse=True)
def _skip_mt5_connection(monkeypatch):
    monkeypatch.setattr(cf, "ensure_mt5_connection_or_raise", lambda: None)


def test_attach_timezone_removes_legacy_timestamp_timezone() -> None:
    result = cf._attach_timezone(
        {"success": True, "timestamp_timezone": "America/New_York"},
        operation="forecast_generate",
    )

    assert result["timezone"] == "UTC"
    assert "timestamp_timezone" not in result


def test_forecast_future_as_of_error_has_date_specific_guidance() -> None:
    result = cf._forecast_error_payload(
        "as_of must not be in the future.",
        operation="forecast_generate",
    )

    assert result["error_code"] == "forecast_as_of_in_future"
    assert "forecast_list_methods" not in result["remediation"]
    assert "ISO 8601" in result["remediation"]


def test_forecast_model_mismatch_has_structured_identity_details() -> None:
    error = ModelCompatibilityError(
        "stored model is incompatible",
        model_id="nhits/EURUSD_H1/abc",
        stored_fingerprint={"horizon": 12},
        requested_fingerprint={"horizon": 24},
        mismatches={"horizon": {"stored": 12, "requested": 24}},
    )

    result = cf._forecast_error_payload(error, operation="forecast_generate")

    assert result["error_code"] == "forecast_model_incompatible"
    assert result["details"]["model_id"] == "nhits/EURUSD_H1/abc"
    assert result["details"]["mismatches"]["horizon"] == {
        "stored": 12,
        "requested": 24,
    }


def test_normalize_forecaster_name_and_resolve_variants(monkeypatch):
    monkeypatch.setattr(
        forecast_sktime_index,
        "_registered_sktime_forecasters",
        lambda: {},
    )
    monkeypatch.setattr(
        forecast_sktime_index,
        "_load_sktime_forecaster_index",
        lambda: {},
    )
    monkeypatch.setattr(
        forecast_sktime_index,
        "_discover_sktime_forecasters",
        lambda: {
            "thetaforecaster": ("ThetaForecaster", "sktime.forecasting.theta.ThetaForecaster"),
            "naiveforecaster": ("NaiveForecaster", "sktime.forecasting.naive.NaiveForecaster"),
        },
    )

    assert forecast_use_cases._normalize_forecaster_name("Theta-Forecaster v2") == "thetaforecasterv2"
    assert cf._resolve_sktime_forecaster("theta") == (
        "ThetaForecaster",
        "sktime.forecasting.theta.ThetaForecaster",
    )
    assert cf._resolve_sktime_forecaster("naive_fore") == (
        "NaiveForecaster",
        "sktime.forecasting.naive.NaiveForecaster",
    )
    assert cf._resolve_sktime_forecaster("") is None


def test_resolve_registered_sktime_class_skips_recursive_discovery(monkeypatch):
    monkeypatch.setattr(
        forecast_sktime_index,
        "_registered_sktime_forecasters",
        lambda: {
            "naiveforecaster": (
                "NaiveForecaster",
                "sktime.forecasting.naive.NaiveForecaster",
            ),
        },
    )
    monkeypatch.setattr(
        forecast_sktime_index,
        "_load_sktime_forecaster_index",
        lambda: {},
    )
    monkeypatch.setattr(
        forecast_sktime_index,
        "_discover_sktime_forecasters",
        lambda: pytest.fail("recursive discovery must not run for an exact class"),
    )

    assert cf._resolve_sktime_forecaster("NaiveForecaster") == (
        "NaiveForecaster",
        "sktime.forecasting.naive.NaiveForecaster",
    )


def test_resolve_sktime_class_reuses_persistent_index(monkeypatch):
    monkeypatch.setattr(
        forecast_sktime_index,
        "_registered_sktime_forecasters",
        lambda: {},
    )
    monkeypatch.setattr(
        forecast_sktime_index,
        "_load_sktime_forecaster_index",
        lambda: {
            "ararforecaster": (
                "ARARForecaster",
                "sktime.forecasting.trend.ARARForecaster",
            ),
        },
    )
    monkeypatch.setattr(
        forecast_sktime_index,
        "_discover_sktime_forecasters",
        lambda: pytest.fail("persistent exact lookup must not rediscover sktime"),
    )

    assert cf._resolve_sktime_forecaster("ARARForecaster") == (
        "ARARForecaster",
        "sktime.forecasting.trend.ARARForecaster",
    )


def test_sktime_forecaster_index_round_trip(tmp_path, monkeypatch):
    index_path = tmp_path / "sktime-index.json"
    monkeypatch.setattr(
        forecast_sktime_index,
        "_sktime_forecaster_index_path",
        lambda: index_path,
    )
    mapping = {
        "naiveforecaster": (
            "NaiveForecaster",
            "sktime.forecasting.naive.NaiveForecaster",
        ),
    }

    forecast_use_cases._store_sktime_forecaster_index(mapping)

    assert forecast_use_cases._load_sktime_forecaster_index() == mapping


def test_discover_sktime_forecasters_uses_registry_and_skips_required_ctors(monkeypatch):
    cf._clear_discover_sktime_forecasters_cache()
    monkeypatch.setattr(
        forecast_sktime_index,
        "_store_sktime_forecaster_index",
        lambda _mapping: None,
    )

    class ThetaForecaster:
        def __init__(self, sp=1):
            self.sp = sp

    class NeedsArgForecaster:
        def __init__(self, required):
            self.required = required

    class BaseDeepNetworkPyTorch:
        def __init__(self):
            pass

    fake_registry = ModuleType("sktime.registry")

    def fake_all_estimators(*, estimator_types, return_names):
        assert estimator_types == "forecaster"
        assert return_names is True
        return [
            ("ThetaForecaster", ThetaForecaster),
            ("NeedsArgForecaster", NeedsArgForecaster),
            ("_PrivateForecaster", ThetaForecaster),
        ]

    fake_registry.all_estimators = fake_all_estimators
    monkeypatch.setitem(sys.modules, "sktime.registry", fake_registry)

    mapping = cf._discover_sktime_forecasters()

    assert mapping == {
        "thetaforecaster": ("ThetaForecaster", f"{ThetaForecaster.__module__}.ThetaForecaster"),
    }
    assert "basedeepnetworkpytorch" not in mapping
    cf._clear_discover_sktime_forecasters_cache()


def test_forecast_generate_routes_by_library_and_validates_inputs(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    captured = {}

    def fake_forecast_impl(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "method": kwargs["method"], "params": kwargs["params"]}

    monkeypatch.setattr(cf, "_forecast_impl", fake_forecast_impl)
    monkeypatch.setattr(
        cf,
        "_resolve_sktime_forecaster",
        lambda q: ("ThetaForecaster", "sktime.forecasting.theta.ThetaForecaster") if q == "theta" else None,
    )

    with pytest.raises(Exception):
        ForecastGenerateRequest(symbol="EURUSD", horizon=0)

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="statsforecast", method=""))
    assert out["error"] == "method is required for library=statsforecast"

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="sktime", method="unknown"))
    assert "Unknown sktime forecaster" in out["error"]

    captured.clear()
    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            library="native",
            method="naive",
            params={"bogus": 1},
        )
    )
    assert out["success"] is False
    assert out["error_code"] == "unknown_parameter"
    assert out["unknown_keys"] == ["bogus"]
    assert out["valid_keys"] == []
    assert captured == {}

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method=""))
    assert out["ok"] is True
    assert captured["method"] == "theta"
    assert captured["params"] == {}

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="statsforecast", method="AutoARIMA", params={}))
    assert out["ok"] is True
    assert captured["method"] == "statsforecast"
    assert captured["params"]["model_name"] == "AutoARIMA"
    assert out["method"] == "AutoARIMA"
    assert out["library"] == "statsforecast"

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="statsforecast", method="sf_autoarima", params={}))
    assert out["ok"] is True
    assert captured["method"] == "statsforecast"
    assert captured["params"]["model_name"] == "AutoARIMA"
    assert out["method"] == "sf_autoarima"
    assert out["library"] == "statsforecast"

    stored_model_id = "sf_autoarima/EURUSD_H1/abc123"
    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            library="statsforecast",
            method="sf_autoarima",
            model_id=stored_model_id,
            model_cache="require_existing",
        )
    )
    assert out["ok"] is True
    assert captured["method"] == "sf_autoarima"
    assert "model_name" not in captured["params"]
    assert captured["model_id"] == stored_model_id

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            library="statsforecast",
            method="AutoARIMA",
            model_id=stored_model_id,
            model_cache="require_existing",
        )
    )
    assert out["ok"] is True
    assert captured["method"] == "sf_autoarima"
    assert "model_name" not in captured["params"]

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="statsforecast:autoarima", params={}))
    assert out["success"] is False
    assert "belongs to library 'statsforecast'" in out["error"]

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            library="native",
            method="sf_theta",
        )
    )
    assert out["success"] is False
    assert "belongs to library 'statsforecast'" in out["error"]

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            model_id=stored_model_id,
            model_cache="require_existing",
        )
    )
    assert out["ok"] is True
    assert captured["method"] == "sf_autoarima"
    assert captured["model_id"] == stored_model_id

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="sktime", method="theta", params={}))
    assert out["ok"] is True
    assert captured["method"] == "sktime"
    assert captured["params"]["estimator"] == "sktime.forecasting.theta.ThetaForecaster"

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="sktime", method="skt_theta", params={}))
    assert out["ok"] is True
    assert captured["method"] == "sktime"
    assert captured["params"]["estimator"] == "sktime.forecasting.theta.ThetaForecaster"
    assert out["method"] == "skt_theta"
    assert out["library"] == "sktime"

    stored_sktime_model_id = "skt_naive/EURUSD_H1/abc123"
    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            model_id=stored_sktime_model_id,
            model_cache="require_existing",
        )
    )
    assert out["ok"] is True
    assert captured["method"] == "skt_naive"
    assert "estimator" not in captured["params"]
    assert captured["model_id"] == stored_sktime_model_id

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="sktime:theta", params={}))
    assert out["success"] is False
    assert "belongs to library 'sktime'" in out["error"]

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            library="sktime",
            method="sktime.forecasting.naive.NaiveForecaster",
            params={},
        )
    )
    assert out["ok"] is True
    assert captured["params"]["estimator"] == "sktime.forecasting.naive.NaiveForecaster"

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            library="mlforecast",
            method="sklearn.linear_model.LinearRegression",
            params={},
        )
    )
    assert out["ok"] is True
    assert captured["method"] == "mlforecast"
    assert captured["params"]["model"] == "sklearn.linear_model.LinearRegression"
    assert out["method"] == "sklearn.linear_model.LinearRegression"
    assert out["library"] == "mlforecast"

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="mlforecast:rf", params={"lags": [1, 2, 3]}))
    assert out["success"] is False
    assert "belongs to library 'mlforecast'" in out["error"]

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="native:theta"))
    assert out["ok"] is True
    assert captured["method"] == "theta"
    assert captured["params"] == {}

    with pytest.raises(Exception):
        ForecastGenerateRequest(symbol="EURUSD", library="unsupported", method="x")


def test_forecast_generate_native_theta_uses_provenance_without_nominal_warning(monkeypatch):
    raw = _unwrap(cf.forecast_generate)

    def fake_forecast_impl(**kwargs):
        return {"ok": True, "method": kwargs["method"]}

    monkeypatch.setattr(cf, "_forecast_impl", fake_forecast_impl)

    out = raw(request=ForecastGenerateRequest(symbol="BTCUSD", timeframe="H1", library="native", method="theta", horizon=12))

    assert out["ok"] is True
    assert out["success"] is True
    assert out["method"] == "theta"
    assert out["library"] == "native"
    assert out.get("warnings", []) == []


def test_forecast_generate_native_theta_preserves_actual_interval_guidance(monkeypatch):
    raw = _unwrap(cf.forecast_generate)

    def fake_forecast_impl(**kwargs):
        return {
            "ok": True,
            "method": kwargs["method"],
            "warnings": [
                "Point forecast only for method 'theta'; confidence intervals are unavailable. "
                "Use forecast_conformal_intervals for uncertainty bands."
            ],
        }

    monkeypatch.setattr(cf, "_forecast_impl", fake_forecast_impl)

    out = raw(request=ForecastGenerateRequest(symbol="BTCUSD", timeframe="H1", library="native", method="theta", horizon=12))

    assert out["ok"] is True
    assert out["success"] is True
    assert len(out["warnings"]) == 1
    assert "forecast_conformal_intervals" in out["warnings"][0]
    assert all("StatsForecast theta is available" not in str(w) for w in out["warnings"])


def test_forecast_generate_defaults_to_compact_payload(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(forecast_compact, "_symbol_price_currency", lambda _symbol: "USD")
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "timezone": "UTC",
            "forecast_from": {"time": "t0", "anchor": "last_observation"},
            "forecast_anchor": "next_timeframe_bar_after_last_observation",
            "forecast_step_seconds": 3600,
            "forecast_time": ["t1", "t2", "t3"],
            "forecast_price": [1.0, 1.1, 1.2],
            "forecast_epoch": [1.0, 2.0, 3.0],
            "last_price": 1.05,
            "last_price_source": "candle_close",
            "price_basis": "bid",
            "last_price_age_seconds": 3600,
            "last_price_age": "1h 0m",
            "last_price_stale": False,
            "freshness_basis": "bar_policy",
            "stale_after_seconds": 10800,
            "digits": 5,
        },
    )

    out = raw(request=ForecastGenerateRequest(symbol="BTCUSD", timeframe="H1", method="theta", horizon=3))

    assert "detail" not in out
    assert out["symbol"] == "BTCUSD"
    assert out["timeframe"] == "H1"
    assert out["price_currency"] == "USD"
    assert out["timezone"] == "UTC"
    assert "forecast_from" not in out
    assert "forecast_anchor" not in out
    assert "forecast_step_seconds" not in out
    assert out["last_price"] == 1.05
    assert out["last_price_source"] == "candle_close"
    assert out["price_basis"] == "bid"
    assert out["last_price_stale"] is False
    assert out["freshness"] == "fresh, anchor 1h 0m ago"
    assert "last_price_age_seconds" not in out
    assert "last_price_age" not in out
    assert "freshness_basis" not in out
    assert "stale_after_seconds" not in out
    assert out["forecast_vs_last_price"] == {
        "direction_basis": "horizon_end",
        "direction_threshold_pct": 0.05,
        "direction_threshold_basis": "minimum_effect_size_0.05_pct",
        "first_step_delta": -0.05,
        "horizon_delta": 0.15,
        "first_step_delta_pct": -4.7619,
        "horizon_delta_pct": 14.2857,
        "direction_interval_excludes_last_price": None,
        "direction_interval_basis": "not_available",
        "direction_interpretation": "point_estimate_only",
        "direction_status": "unconfirmed",
        "direction_actionable": False,
        "direction_suppressed_reason": "forecast_uncertainty_not_available",
    }
    assert "point_estimate_direction" not in out["forecast_vs_last_price"]
    assert out["signal_status"] == "not_actionable"
    assert out["uncertainty"] == {
        "status": "not_requested",
        "mode": "point_only",
        "reason": "ci_alpha was not requested; direction is based on the point estimate only.",
        "recommended_tool": "forecast_conformal_intervals",
    }
    assert out["trust_level"] == "adequate"
    assert "trust_blockers" not in out
    assert out["units"]["forecast_vs_last_price.*_delta_pct"] == (
        "percent (1.0 = 1%)"
    )
    assert "forecast_time" not in out
    assert "forecast_price" not in out
    assert out["forecast"] == [
        {"time": "t1", "value": 1.0},
        {"time": "t2", "value": 1.1},
        {"time": "t3", "value": 1.2},
    ]
    assert "series" not in out
    assert "collection_kind" not in out
    assert "collection_contract_version" not in out
    assert "forecast_epoch" not in out


def test_forecast_generate_compact_summarizes_stale_anchor(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1"],
            "forecast_price": [1.0],
            "last_price": 1.05,
            "last_price_age_seconds": 101884,
            "last_price_age": "1d 4h",
            "last_price_stale": True,
            "freshness_basis": "bar_policy",
            "stale_after_seconds": 10800,
            "stale_warning": (
                "Last forecast anchor is older than the bar freshness policy; "
                "market may be closed or broker data may be stale."
            ),
        },
    )

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            timeframe="H1",
            method="theta",
            horizon=1,
        )
    )

    assert out["last_price_stale"] is True
    assert out["freshness"] == "stale, anchor 1d 4h ago (policy: 3h 0m)"
    assert "last_price_age_seconds" not in out
    assert "last_price_age" not in out
    assert "freshness_basis" not in out
    assert "stale_after_seconds" not in out
    assert "stale_warning" not in out


def test_forecast_generate_compact_volatility_uses_summary_row(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "volatility_per_bar": 0.012345,
            "volatility_annualized": 0.194444,
            "volatility_horizon": 0.021234,
            "forecast_time": ["t1", "t2", "t3"],
            "input_evidence": {"source": {"row_sha256": "a" * 64}},
            "fit_diagnostics": {"shape": [99, 2]},
            "params_explained": {"lambda_": "debug explanation"},
            "volatility_interpretation": {"volatility_per_bar": "per bar"},
        },
    )

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            timeframe="H1",
            method="theta",
            quantity="volatility",
            horizon=3,
        )
    )

    assert out["volatility_per_bar"] == pytest.approx(0.012345)
    assert out["quantity"] == "volatility"
    assert out["volatility_horizon"] == pytest.approx(0.021234)
    assert out["forecast_summary_mode"] == "scalar_volatility_estimate"
    assert "no distinct per-step path is implied" in out["quantity_note"]
    assert "forecast_time" not in out
    assert out["forecast"] == [
        {
            "horizon_steps": 3,
            "start_time": "t1",
            "end_time": "t3",
            "volatility_per_bar": 0.012345,
            "volatility_annualized": 0.194444,
            "volatility_horizon": 0.021234,
        }
    ]
    assert "input_evidence" not in out
    assert "fit_diagnostics" not in out
    assert "params_explained" not in out
    assert "volatility_interpretation" not in out


def test_forecast_generate_compact_normalizes_utc_times_and_neutral_delta(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "data_as_of": "2026-06-02 19:00",
            "last_observation_time": "2026-06-02 19:00",
            "forecast_start_time": "2026-06-02 20:00",
            "forecast_start_gap_bars": 1.0,
            "forecast_start_gap_note": "Static gap explanation.",
            "last_price_age_seconds": 3600,
            "last_price_stale": False,
            "denoise_applied": False,
            "forecast_time": ["2026-06-02 20:00", "2026-06-02 21:00"],
            "forecast_bar_states": ["forming", "future"],
            "forecast_time_semantics": "target_bar_open_time",
            "forecast_value_semantics": "target_bar_close",
            "forecast_price": [1.00004, 1.00006],
            "last_price": 1.0,
            "digits": 5,
        },
    )

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            timeframe="H1",
            method="theta",
            horizon=2,
        )
    )

    assert out["data_as_of"] == "2026-06-02T19:00Z"
    assert out["last_observation_time"] == "2026-06-02T19:00Z"
    assert out["timezone"] == "UTC"
    assert out["data_window"] == {
        "last_observation": "2026-06-02T19:00Z",
        "last_bar_complete": True,
        "input_bar_policy": "closed_bars_only",
        "forecast_start": "2026-06-02T20:00Z",
        "forecast_start_gap_bars": 1.0,
        "forecast_time_semantics": "target_bar_open_time",
            "forecast_value_semantics": "target_bar_close",
            "first_forecast_bar_state": "forming",
            "horizon_includes_forming_bar": True,
            "last_observation_age_seconds": 3600,
        "last_observation_stale": False,
    }
    assert "forecast_start_gap_bars" not in out
    assert "forecast_start_gap_note" not in out
    assert "last_price_stale" not in out
    assert "denoise_applied" not in out
    assert out["forecast"] == [
        {
            "time": "2026-06-02T20:00Z",
            "bar_state": "forming",
            "value": 1.00004,
        },
        {
            "time": "2026-06-02T21:00Z",
            "bar_state": "future",
            "value": 1.00006,
        },
    ]
    assert out["forecast_vs_last_price"]["direction"] == "neutral"
    assert out["forecast_vs_last_price"]["direction_threshold_pct"] == 0.05
    assert out["forecast_vs_last_price"]["direction_actionable"] is False


def test_forecast_generate_compact_preserves_history_reliability(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "last_observation_time": "2026-06-02 19:00",
            "forecast_time": ["2026-06-02 20:00"],
            "forecast_price": [1.0],
            "last_price": 1.0,
            "history_sample_ok": False,
            "forecast_reliability": "low",
            "forecast_reliability_reason": "below_recommended_history",
            "recommended_history_bars": 36,
            "history_shortfall_bars": 33,
            "warnings": ["Low-history forecast."],
        },
    )

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            timeframe="H1",
            method="theta",
            horizon=12,
        )
    )

    assert out["history_sample_ok"] is False
    assert out["forecast_reliability"] == "low"
    assert out["forecast_reliability_basis"] == "history_sample_size"
    assert out["forecast_reliability_reason"] == "below_recommended_history"
    assert out["trust_level"] == "low"
    assert "insufficient_history_sample" in out["trust_blockers"]
    assert out["recommended_history_bars"] == 36
    assert out["history_shortfall_bars"] == 33
    assert "Low-history forecast." in out["warnings"]


def test_forecast_generate_combines_sample_and_history_policy_trust(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["2026-06-02 20:00"],
            "forecast_price": [1.0],
            "forecast_reliability": "adequate",
            "history_policy_ok": False,
            "ci_status": "available",
        },
    )

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", method="theta"))

    assert out["forecast_reliability"] == "adequate"
    assert out["forecast_reliability_basis"] == "history_sample_size"
    assert out["trust_level"] == "degraded"
    assert out["trust_blockers"] == ["history_freshness_policy_not_met"]


def test_run_forecast_generate_adds_available_methods_to_invalid_error(monkeypatch):
    monkeypatch.setattr(
        forecast_compact,
        "get_forecast_methods_snapshot",
        lambda: {
            "methods": [
                {"method": "theta", "available": True},
                {"method": "drift", "available": True},
                {"method": "missing_dependency", "available": False},
            ]
        },
    )

    result = forecast_use_cases.run_forecast_generate(
        ForecastGenerateRequest(symbol="EURUSD", method="nope"),
        forecast_impl=lambda **kwargs: {
            "error": "Invalid method: nope. Run forecast_list_methods for the full catalog."
        },
    )

    assert result["valid_values"] == {"method": ["drift", "theta"]}
    assert result["related_tools"] == ["forecast_list_methods"]


def test_forecast_backtest_request_rejects_singular_method_alias():
    with pytest.raises(ValueError, match="method was removed; use methods"):
        ForecastBacktestRequest(symbol="EURUSD", method="theta")


def test_forecast_backtest_request_accepts_methods():
    request = ForecastBacktestRequest(
        symbol="EURUSD",
        methods=["theta"],
        lookback=50,
    )

    assert request.methods == ["theta"]
    assert request.lookback == 50


def test_forecast_backtest_request_validates_anchor_spacing_up_front():
    equal_spacing = ForecastBacktestRequest(
        symbol="EURUSD", horizon=6, steps=3, spacing=6
    )
    assert equal_spacing.spacing == equal_spacing.horizon

    with pytest.raises(
        ValidationError,
        match=r"got spacing=5, horizon=6.*try spacing=6 or steps=1",
    ):
        ForecastBacktestRequest(symbol="EURUSD", horizon=6, steps=3, spacing=5)

    descriptions = {
        name: str(ForecastBacktestRequest.model_fields[name].description)
        for name in ("horizon", "steps", "spacing")
    }
    assert all("spacing" in value.lower() for value in descriptions.values())


def test_forecast_generate_compact_omits_training_period(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "last_observation_time": "2026-01-10 00:00",
            "forecast_time": ["t1"],
            "forecast_price": [1.0],
            "diagnostics": {
                "history_start_time": "2026-01-01 00:00",
                "history_end_time": "2026-01-10 00:00",
                "history_bars_used": 200,
                "target_points_used": 199,
                "lookback_bars_requested": 250,
                "minimum_history_bars_requested": 300,
                "history_bars_received": 240,
            },
        },
    )

    expected_training_period = {
        "start": "2026-01-01T00:00Z",
        "end": "2026-01-10T00:00Z",
        "history_bars_used": 200,
        "target_points_used": 199,
        "lookback_bars_requested": 250,
        "minimum_history_bars_requested": 300,
        "history_bars_received": 240,
        "note": "Forecast was fit on the historical window summarized here.",
    }

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            timeframe="H1",
            method="theta",
        )
    )
    assert "training_period" not in out
    assert "diagnostics" not in out
    assert out["data_window"]["history_start"] == "2026-01-01T00:00Z"
    assert out["data_window"]["history_end"] == "2026-01-10T00:00Z"
    assert out["data_window"]["history_bars_used"] == 200

    standard = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            timeframe="H1",
            method="theta",
            detail="standard",
        )
    )
    assert standard["training_period"] == expected_training_period


def test_forecast_generate_rounds_price_outputs_to_symbol_digits(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1", "t2"],
            # Span more than one tick so path-flatness does not suppress direction.
            "forecast_price": [1.1731445723463942, 1.1741467944693543],
            "last_price": 1.17266,
            "last_price_source": "candle_close",
            "digits": 5,
        },
    )

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", timeframe="H1", method="theta", horizon=2))

    assert out["forecast_vs_last_price"] == {
        "direction_basis": "horizon_end",
        "direction_threshold_pct": 0.05,
        "direction_threshold_basis": "minimum_effect_size_0.05_pct",
        "first_step_delta": 0.00048,
        "horizon_delta": 0.00149,
        "first_step_delta_pct": 0.0409,
        "horizon_delta_pct": 0.1271,
        "direction_interval_excludes_last_price": None,
        "direction_interval_basis": "not_available",
        "direction_interpretation": "point_estimate_only",
        "direction_status": "unconfirmed",
        "direction_actionable": False,
        "direction_suppressed_reason": "forecast_uncertainty_not_available",
    }
    assert "point_estimate_direction" not in out["forecast_vs_last_price"]
    assert out["signal_status"] == "not_actionable"
    assert "forecast_price" not in out
    assert out["forecast"] == [
        {"time": "t1", "value": 1.17314},
        {"time": "t2", "value": 1.17415},
    ]
    assert "series" not in out


def test_forecast_generate_compact_flags_flat_theta_display(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1", "t2", "t3"],
            "forecast_price": [1.168361, 1.168362, 1.168363],
            "last_price": 1.16317,
            "last_price_source": "candle_close",
            "digits": 5,
            "params_used": {"alpha": 0.2, "trend_slope": 0.000002},
        },
    )

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", timeframe="H1", method="theta", horizon=3))

    assert "forecast_price" not in out
    assert out["forecast"] == [
        {"time": "t1", "value": 1.16836},
        {"time": "t2", "value": 1.16836},
        {"time": "t3", "value": 1.16836},
    ]
    assert "forecast_summary" not in out
    assert "theta_signal" not in out
    assert "params_used" not in out
    assert out["path_flat"] is True
    assert out["path_range"] == 0.0
    assert out["point_forecast_mode"] == "flat_model_path"
    assert out["forecast_status"] == "non_informative"
    assert out["signal_status"] == "not_actionable"
    assert out["suggested_methods"] == ["drift", "analog", "fourier_ols"]
    assert out["suggested_uncertainty_tool"] == "forecast_conformal_intervals"
    assert "usable_for_live_trading" not in out
    assert out["forecast_vs_last_price"]["direction"] == "neutral"
    assert out["forecast_vs_last_price"]["direction_basis"] == "flat_path"
    assert out["forecast_vs_last_price"]["direction_suppressed_reason"] == "flat_path"
    assert any("near-flat at displayed price precision" in item for item in out["warnings"])


def test_forecast_generate_compact_flags_one_tick_flat_path(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1", "t2", "t3", "t4", "t5"],
            "forecast_price": [1.16426, 1.16426, 1.16427, 1.16427, 1.16427],
            "last_price": 1.16505,
            "digits": 5,
            "ci_status": "unavailable",
        },
    )

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", timeframe="H1", method="theta", horizon=5))

    assert out["path_flat"] is True
    assert out["path_range"] == 0.00001
    assert out["forecast_status"] == "non_informative"
    assert out["signal_status"] == "not_actionable"
    assert "usable_for_live_trading" not in out
    assert out["forecast_vs_last_price"]["direction"] == "neutral"
    assert out["forecast_vs_last_price"]["direction_basis"] == "flat_path"
    assert out["forecast_vs_last_price"]["direction_suppressed_reason"] == "flat_path"
    assert any("forecast_conformal_intervals" in item for item in out["warnings"])


def test_forecast_generate_compact_marks_unavailable_ci(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1"],
            "forecast_price": [1.0],
            "ci_status": "unavailable",
            "ci_available": False,
            "ci_alpha": 0.05,
            "ci": {
                "status": "unavailable",
                "hint": "Use forecast_conformal_intervals for uncertainty bands.",
            },
            "warnings": [
                "Point forecast only for method 'theta'; confidence intervals are unavailable. "
                "Use forecast_conformal_intervals for uncertainty bands.",
                "Native theta fallback used.",
            ],
        },
    )

    out = raw(
        request=ForecastGenerateRequest(
            symbol="BTCUSD",
            timeframe="H1",
            method="theta",
            horizon=1,
        )
    )

    assert "detail" not in out
    assert "ci" not in out
    assert out["ci_status"] == "unavailable"
    assert out["forecast_mode"] == "point_only"
    assert "ci_available" not in out
    assert "ci_alpha" not in out
    assert out["uncertainty"] == {
        "status": "unavailable",
        "mode": "point_only",
        "reason": (
            "requested intervals are unavailable for this method; "
            "point forecast only."
        ),
        "recommended_tool": "forecast_conformal_intervals",
        "requested_alpha": 0.05,
    }
    assert out["warnings"] == ["Native theta fallback used."]


def test_forecast_generate_full_flags_flat_theta_display(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1", "t2", "t3"],
            "forecast_price": [1.168361, 1.168362, 1.168363],
            "last_price": 1.16317,
            "digits": 5,
        },
    )

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            timeframe="H1",
            method="theta",
            horizon=3,
            detail="full",
        )
    )

    assert out["path_flat"] is True
    assert out["path_range"] == 0.0
    assert out["point_forecast_mode"] == "flat_model_path"
    assert out["forecast_vs_last_price"]["direction"] == "neutral"
    assert out["forecast_vs_last_price"]["direction_basis"] == "flat_path"
    assert any("near-flat at displayed price precision" in item for item in out["warnings"])


def test_forecast_generate_compact_nests_available_ci(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1", "t2"],
            "forecast_bar_states": ["forming", "future"],
            "forecast_price": [100.0, 101.0],
            "lower_price": [99.0, 99.5],
            "upper_price": [101.0, 102.5],
            "ci_status": "available",
            "ci_alpha": 0.05,
            "last_price": 100.0,
        },
    )

    out = raw(
        request=ForecastGenerateRequest(
            symbol="BTCUSD",
            timeframe="H1",
            method="arima",
            horizon=2,
        )
    )

    assert out["uncertainty"] == {
        "status": "available",
        "mode": "interval",
        "alpha": 0.05,
        "summary": {
            "first_low": 99.0,
            "first_high": 101.0,
            "last_low": 99.5,
            "last_high": 102.5,
            "median_width": 2.5,
        },
    }
    assert "intervals" not in out["uncertainty"]
    assert out["ci_status"] == "available"
    assert out["forecast_mode"] == "interval"
    selected = _select_output_fields(
        out,
        "ci_status,forecast_mode,uncertainty",
    )
    assert selected["ci_status"] == "available"
    assert selected["forecast_mode"] == "interval"
    assert "unresolved_output_fields" not in selected
    assert "ci_alpha" not in out
    assert "interval_summary" not in out
    assert "lower_price" not in out
    assert "upper_price" not in out
    assert "forecast_time" not in out
    assert "forecast_price" not in out
    assert out["forecast"] == [
        {
            "time": "t1",
            "bar_state": "forming",
            "value": 100.0,
            "lower_price": 99.0,
            "upper_price": 101.0,
        },
        {
            "time": "t2",
            "bar_state": "future",
            "value": 101.0,
            "lower_price": 99.5,
            "upper_price": 102.5,
        },
    ]
    assert "direction" not in out["forecast_vs_last_price"]
    assert out["forecast_vs_last_price"]["point_estimate_direction"] == "bullish"
    assert out["forecast_vs_last_price"]["direction_actionable"] is False
    assert out["forecast_vs_last_price"]["direction_status"] == "unconfirmed"
    assert out["forecast_vs_last_price"]["direction_suppressed_reason"] == (
        "horizon_interval_contains_last_price"
    )
    assert out["forecast_vs_last_price"]["direction_interval_excludes_last_price"] is False
    assert out["forecast_vs_last_price"]["direction_interval_basis"] == (
        "horizon_interval_vs_last_price"
    )
    assert out["forecast_vs_last_price"]["direction_interpretation"] == (
        "interval_contains_last_price_or_direction_is_neutral"
    )
    assert out["signal_status"] == "not_actionable"


def test_forecast_generate_keeps_direction_when_horizon_interval_confirms_it():
    out = forecast_use_cases._apply_forecast_generate_detail(
        {
            "success": True,
            "method": "arima",
            "horizon": 2,
            "quantity": "price",
            "forecast_time": ["t1", "t2"],
            "forecast_price": [100.5, 101.0],
            "lower_price": [99.5, 100.4],
            "upper_price": [101.0, 102.0],
            "ci_status": "available",
            "ci_alpha": 0.05,
            "last_price": 100.0,
        },
        ForecastGenerateRequest(
            symbol="BTCUSD",
            timeframe="H1",
            method="arima",
            horizon=2,
        ),
    )

    context = out["forecast_vs_last_price"]
    assert context["direction"] == "bullish"
    assert context["direction_actionable"] is True
    assert context["direction_status"] == "interval_confirmed"
    assert context["direction_interval_excludes_last_price"] is True
    assert "point_estimate_direction" not in context
    assert "signal_status" not in out


def test_forecast_generate_compact_return_keeps_price_path_with_labeled_ci():
    out = forecast_use_cases._apply_forecast_generate_detail(
        {
            "success": True,
            "method": "arima",
            "horizon": 2,
            "quantity": "return",
            "forecast_time": ["t1", "t2"],
            "forecast_return": [0.01, -0.005],
            "forecast_price": [101.0, 100.4963],
            "lower_return": [0.002, -0.012],
            "upper_return": [0.018, 0.002],
            "ci_status": "available",
            "ci_alpha": 0.05,
            "last_price": 100.0,
        },
        ForecastGenerateRequest(
            symbol="BTCUSD",
            timeframe="H1",
            method="arima",
            horizon=2,
            quantity="return",
        ),
    )

    assert out["forecast"] == [
        {
            "time": "t1",
            "return": 0.01,
            "price": 101.0,
            "lower_return": 0.002,
            "upper_return": 0.018,
        },
        {
            "time": "t2",
            "return": -0.005,
            "price": 100.4963,
            "lower_return": -0.012,
            "upper_return": 0.002,
        },
    ]
    assert "intervals" not in out["uncertainty"]
    assert out["uncertainty"]["summary"] == {
        "first_low": 0.002,
        "first_high": 0.018,
        "last_low": -0.012,
        "last_high": 0.002,
        "median_width": 0.015,
    }
    assert out["return_unit"] == "return_fraction"
    assert "forecast_price" not in out
    assert "forecast_return" not in out


def test_forecast_generate_compact_projects_analog_diagnostics():
    metadata = {
        "component_status": [
            {
                "timeframe": "H1",
                "role": "primary",
                "status": "contributed",
                "n_paths": 12,
                "component_weight": 1.0,
                "diagnostic": {
                    "window_size": 64,
                    "search_depth": 5000,
                    "ensemble_metrics": {"effective_paths": 8.6},
                },
            }
        ],
        "analogs": [{"values": [1.01, 1.02], "meta": {"score": 0.2}}],
        "ensemble_metrics": {
            "n_paths": 12,
            "effective_paths": 8.6,
            "spread": 0.003,
            "weighted": True,
            "score_summary": {"best": 0.1, "median": 0.2, "worst": 0.9},
            "quality_gate": {
                "status": "passed",
                "ensemble": {"total_paths": 12, "effective_paths": 8.6},
            },
        },
        "timeframe_diagnostics": {"H1": {"window_size": 64}},
    }
    payload = {
        "success": True,
        "method": "analog",
        "horizon": 2,
        "quantity": "price",
        "forecast_time": ["t1", "t2"],
        "forecast_price": [1.01, 1.02],
        **metadata,
    }
    compact = forecast_use_cases._apply_forecast_generate_detail(
        payload,
        ForecastGenerateRequest(symbol="EURUSD", timeframe="H1", method="analog", horizon=2),
    )

    assert "analogs" not in compact
    assert "timeframe_diagnostics" not in compact
    assert compact["component_status"] == [
        {
            "timeframe": "H1",
            "role": "primary",
            "status": "contributed",
            "n_paths": 12,
            "component_weight": 1.0,
        }
    ]
    assert compact["ensemble_metrics"] == {
        "n_paths": 12,
        "effective_paths": 8.6,
        "spread": 0.003,
        "weighted": True,
        "score_summary": {"best": 0.1, "median": 0.2},
        "quality_gate": {"status": "passed"},
    }

    standard = forecast_use_cases._apply_forecast_generate_detail(
        payload,
        ForecastGenerateRequest(
            symbol="EURUSD",
            timeframe="H1",
            method="analog",
            horizon=2,
            detail="standard",
        ),
    )
    assert standard["analogs"] == metadata["analogs"]
    assert standard["timeframe_diagnostics"] == metadata["timeframe_diagnostics"]


def test_forecast_generate_compact_projects_nested_ensemble_analog_metadata():
    out = forecast_use_cases._apply_forecast_generate_detail(
        {
            "success": True,
            "method": "ensemble",
            "horizon": 1,
            "quantity": "price",
            "forecast_time": ["t1"],
            "forecast_price": [1.01],
            "ensemble": {
                "methods": ["analog", "theta"],
                "analogs": [{"values": [1.01]}],
                "component_status": [
                    {
                        "timeframe": "H1",
                        "role": "primary",
                        "status": "contributed",
                        "n_paths": 3,
                        "diagnostic": {"window_size": 64},
                    }
                ],
                "ensemble_metrics": {"n_paths": 3, "effective_paths": 2.5},
                "timeframe_diagnostics": {"H1": {"window_size": 64}},
            },
        },
        ForecastGenerateRequest(symbol="EURUSD", timeframe="H1", method="ensemble", horizon=1),
    )

    assert out["ensemble"] == {
        "methods": ["analog", "theta"],
        "component_status": [
            {"timeframe": "H1", "role": "primary", "status": "contributed", "n_paths": 3}
        ],
        "ensemble_metrics": {"n_paths": 3, "effective_paths": 2.5},
    }


def test_forecast_generate_standard_uses_forecast_rows_not_parallel_arrays(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1", "t2", "t3"],
            "forecast_price": [1.0, 1.1, 1.2],
            "forecast_epoch": [1.0, 2.0, 3.0],
        },
    )

    out = raw(
        request=ForecastGenerateRequest(
            symbol="BTCUSD",
            timeframe="H1",
            method="theta",
            horizon=3,
            detail="standard",
        )
    )

    assert out["detail"] == "standard"
    assert out["canonical_source"] == "forecast"
    assert "forecast_epoch" not in out
    assert "forecast_time" not in out
    assert "forecast_price" not in out
    assert "series" not in out
    assert out["forecast"][0]["time"] == "t1"
    assert "collection_kind" not in out
    assert "collection_contract_version" not in out


def test_forecast_generate_compact_uses_canonical_payload_symbol():
    out = forecast_use_cases._apply_forecast_generate_detail(
        {
            "success": True,
            "symbol": "EURUSD",
            "symbol_requested": "EUR/USD",
            "method": "theta",
            "horizon": 1,
            "quantity": "price",
            "forecast_time": ["t1"],
            "forecast_price": [1.1],
        },
        ForecastGenerateRequest(symbol="EUR/USD", timeframe="H1", method="theta", horizon=1),
    )

    assert out["symbol"] == "EURUSD"
    assert out["symbol_requested"] == "EUR/USD"
    assert out["forecast"][0]["time"] == "t1"


def test_forecast_generate_summary_is_endpoints_not_compact_clone(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1", "t2", "t3"],
            "forecast_price": [1.0, 1.1, 1.2],
            "last_price": 1.05,
            "last_price_source": "candle_close",
            "last_price_age_seconds": 3600,
            "last_price_age": "1h 0m",
            "last_price_stale": False,
        },
    )

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            timeframe="H1",
            method="theta",
            horizon=3,
            detail="summary",
        )
    )

    assert out["detail"] == "summary"
    assert "forecast" not in out
    assert "forecast_time" not in out
    assert "forecast_price" not in out
    assert out["forecast_endpoints"]["start_time"] == "t1"
    assert out["forecast_endpoints"]["end_time"] == "t3"
    assert out["forecast_endpoints"]["start_value"] == 1.0
    assert out["forecast_endpoints"]["end_value"] == 1.2
    assert "forecast_vs_last_price" in out
    assert "freshness" in out
    assert "uncertainty" in out


def test_forecast_generate_full_keeps_engine_arrays_with_canonical_source(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1", "t2"],
            "forecast_price": [1.0, 1.1],
            "forecast_epoch": [1.0, 2.0],
        },
    )

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            timeframe="H1",
            method="theta",
            horizon=2,
            detail="full",
        )
    )

    assert out["detail"] == "full"
    assert out["canonical_source"] == "forecast"
    assert out["forecast_epoch"] == [1.0, 2.0]
    assert out["forecast_price"] == [1.0, 1.1]
    assert out["forecast"][0]["time"] == "t1"


def test_run_forecast_generate_logs_finish_event(caplog):
    with caplog.at_level("DEBUG", logger="mtdata.forecast.use_cases"):
        result = forecast_use_cases.run_forecast_generate(
            ForecastGenerateRequest(symbol="EURUSD", timeframe="H1", library="native", method="theta"),
            forecast_impl=lambda **kwargs: {"ok": True, "method": kwargs["method"]},
            resolve_sktime_forecaster=lambda query: None,
        )
    assert result["ok"] is True
    assert any(
        "event=finish operation=forecast_generate success=True" in record.message
        for record in caplog.records
    )


def test_run_forecast_backtest_derives_target_from_quantity():
    captured = {}

    def fake_backtest_impl(**kwargs):
        captured.update(kwargs)
        return {"success": True}

    result = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", quantity="return"),
        backtest_impl=fake_backtest_impl,
    )

    assert result["success"] is True
    assert captured["quantity"] == "return"
    assert "target" not in captured


def test_run_forecast_backtest_preserves_requested_noncompact_detail():
    def fake_backtest_impl(**kwargs):
        return {
            "success": True,
            "detail": "compact",
            "backtest_plan": {"fits_planned": 15},
            "results": {},
        }

    standard = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", detail="standard"),
        backtest_impl=fake_backtest_impl,
    )
    summary = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", detail="summary"),
        backtest_impl=fake_backtest_impl,
    )

    assert standard["detail"] == "standard"
    assert summary["detail"] == "summary"
    assert standard["backtest_plan"]["fits_planned"] == 15
    assert standard["backtest_plan"]["actual_runtime_seconds"] >= 0.0


def test_run_forecast_backtest_retains_requested_and_effective_window_in_compact():
    def fake_backtest_impl(**kwargs):
        return {
            "success": True,
            "detail": "compact",
            "backtest_plan": {"fits_planned": 1},
            "analysis_time_window": {
                "history_start": "2026-08-01T00:00:00Z",
                "history_end": "2026-08-18T11:00:00Z",
                "evaluation_start": "2026-08-18T08:00:00Z",
                "evaluation_end": "2026-08-18T11:00:00Z",
                "timezone": "UTC",
                "input_bar_policy": "closed_bars_only",
            },
            "results": {},
        }

    result = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(
            symbol="EURUSD",
            start="2026-08-01T00:00:00Z",
            end="2026-08-18T12:00:00Z",
        ),
        backtest_impl=fake_backtest_impl,
    )

    assert result["analysis_time_window"] == {
        "history_start": "2026-08-01T00:00:00Z",
        "history_end": "2026-08-18T11:00:00Z",
        "evaluation_start": "2026-08-18T08:00:00Z",
        "evaluation_end": "2026-08-18T11:00:00Z",
        "timezone": "UTC",
        "input_bar_policy": "closed_bars_only",
        "start": "2026-08-01T00:00:00Z",
        "end": "2026-08-18T12:00:00Z",
        "reference_policy": "historical_candle_close",
    }


def test_run_forecast_backtest_strips_per_anchor_details_in_compact_mode():
    def fake_backtest_impl(**kwargs):
        return {
            "success": True,
            "request": {"detail": "compact"},
            "resolved_request": {"detail": "compact", "methods": ["theta"]},
            "units": {
                "returns": "return_fraction",
                "forecast_error": "price",
                "avg_mae": "price",
                "avg_rmse": "price",
            },
            "results": {
                "theta": {
                    "avg_mae": 1.0,
                    "metrics": {
                        "sample_warning": "Only 1 trades. Annualized risk metrics are suppressed.",
                        "sample_notice": {
                            "code": "annualization_suppressed_low_sample",
                            "trades_observed": 1,
                            "minimum_trades": 30,
                        },
                    },
                    "details": [{"anchor": "2026-01-01 00:00", "success": True}],
                }
            },
        }

    result = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", detail="compact"),
        backtest_impl=fake_backtest_impl,
    )

    assert result["success"] is True
    assert "request" not in result
    assert "resolved_request" not in result
    assert result["results"] == {
        "theta": {
            "avg_mae": 1.0,
            "status": "complete",
            "complete_success": True,
            "details_count": 1,
            "metrics_reliability": "low",
            "metrics_reliability_reason": "low_sample",
            "sample_notice": {
                "code": "annualization_suppressed_low_sample",
                "trades_observed": 1,
                "minimum_trades": 30,
            },
        }
    }
    assert result["slippage_bps"] == 0.0
    assert result["cost_assumptions"]["score_basis"] == "gross_before_execution_costs"
    assert result["cost_assumptions"]["spread_and_commission"] == "not_modeled"
    assert result["units"] == {
        "forecast_error": "price",
        "avg_mae": "price",
    }
    assert "units_profile" not in result
    assert result["ranked_methods"][0]["method"] == "theta"
    assert result["ranked_methods"][0]["ranking_status"] == "unranked"
    assert result["ranked_methods"][0]["unranked_reason"] == "avg_rmse_unavailable"
    assert "metrics" not in result["ranked_methods"][0]


def test_run_forecast_backtest_keeps_collection_contract_and_failure_diagnostics():
    payload = {
        "success": False,
        "error": "No requested forecast method produced a successful backtest observation.",
        "error_code": "forecast_backtest_no_successful_methods",
        "slippage_bps": 0.0,
        "results": {
            "not_a_method": {
                "success": False,
                "successful_tests": 0,
                "num_tests": 1,
                "details": [
                    {
                        "success": False,
                        "error": "Invalid method: not_a_method",
                    }
                ],
                "metrics_available": False,
                "metrics_reason": "no_successful_tests",
            }
        },
    }

    compact = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", methods=["not_a_method"]),
        backtest_impl=lambda **kwargs: payload,
    )
    full = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(
            symbol="EURUSD", methods=["not_a_method"], detail="full"
        ),
        backtest_impl=lambda **kwargs: payload,
    )

    for result in (compact, full):
        assert result["error_code"] == "forecast_backtest_no_successful_methods"
        assert result["methods_total"] == 1
        assert result["methods_failed"] == 1
        assert result["ranked_methods"][0]["error"] == "Invalid method: not_a_method"
        assert isinstance(result["results"], dict)


def test_run_forecast_backtest_compact_keeps_kelly_metrics():
    def fake_backtest_impl(**kwargs):
        return {
            "success": True,
            "spread_bps": 1.5,
            "commission_bps_per_side": 0.0,
            "results": {
                "theta": {
                    "success": True,
                    "avg_mae": 1.0,
                    "metrics": {
                        "win_rate": 0.5,
                        "avg_win_return": 0.03000001,
                        "avg_loss_return": -0.01500001,
                        "avg_loss_magnitude": 0.01500001,
                        "avg_win_loss_ratio": 1.999998,
                        "kelly_fraction": 0.249999,
                        "half_kelly_fraction": 0.124999,
                        "trades_observed": 4,
                    },
                    "details": [{"anchor": "2026-01-01 00:00", "success": True}],
                }
            },
        }

    result = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", detail="compact"),
        backtest_impl=fake_backtest_impl,
    )

    row = result["results"]["theta"]
    assert row["avg_win_return"] == 0.03
    assert row["avg_loss_return"] == -0.015
    assert row["avg_loss_magnitude"] == 0.015
    assert row["avg_win_loss_ratio"] == 2.0
    assert row["kelly_fraction"] == 0.25
    assert row["half_kelly_fraction"] == 0.125
    assert result["cost_assumptions"]["complete"] is True


def test_run_forecast_backtest_compact_omits_trading_metrics_when_costs_incomplete():
    def fake_backtest_impl(**kwargs):
        return {
            "success": True,
            "results": {
                "theta": {
                    "success": True,
                    "avg_rmse": 0.0012,
                    "avg_mae": 1.0,
                    "avg_directional_accuracy": 0.6,
                    "metrics": {
                        "win_rate": 0.8,
                        "profit_factor": 10.27,
                        "cumulative_return": 0.00135,
                        "avg_return_per_trade": 0.00027,
                        "kelly_fraction": 0.25,
                        "trades_observed": 5,
                    },
                    "details": [{"anchor": "2026-01-01 00:00", "success": True}],
                }
            },
        }

    result = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", detail="compact"),
        backtest_impl=fake_backtest_impl,
    )

    row = result["results"]["theta"]
    assert row["avg_rmse"] == 0.0012
    assert row["avg_mae"] == 1.0
    assert row["avg_directional_accuracy"] == 0.6
    assert row["trades_observed"] == 5
    assert row["trading_metrics_available"] is False
    assert row["trading_metrics_reason"] == "spread_and_commission_not_modeled"
    assert "win_rate" not in row
    assert "profit_factor" not in row
    assert "cumulative_return" not in row
    assert "avg_return_per_trade" not in row
    assert "kelly_fraction" not in row
    assert result["cost_assumptions"]["complete"] is False
    assert result["ranked_methods"][0]["trading_metrics_available"] is False
    assert result["ranked_methods"][0]["trading_metrics_reason"] == (
        "spread_and_commission_not_modeled"
    )


def test_run_forecast_backtest_compact_suppresses_low_sample_kelly_metrics():
    def fake_backtest_impl(**kwargs):
        return {
            "success": True,
            "results": {
                "theta": {
                    "success": True,
                    "avg_mae": 1.0,
                    "avg_rmse": 1.2,
                    "metrics": {
                        "win_rate_pct": 50.0,
                        "avg_win_loss_ratio": 20.06,
                        "kelly_fraction": 0.4751,
                        "half_kelly_fraction": 0.2375,
                        "trades_observed": 2,
                        "sample_notice": {
                            "code": "annualization_suppressed_low_sample",
                            "trades_observed": 2,
                            "minimum_trades": 30,
                        },
                    },
                    "details": [{"anchor": "2026-01-01 00:00", "success": True}],
                }
            },
        }

    result = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", detail="compact"),
        backtest_impl=fake_backtest_impl,
    )

    row = result["results"]["theta"]
    assert row["metrics_reliability"] == "low"
    assert row["trades_observed"] == 2
    assert "win_rate_pct" not in row
    assert "avg_win_loss_ratio" not in row
    assert "kelly_fraction" not in row


def test_run_forecast_backtest_omits_trade_metrics_when_unavailable():
    def fake_backtest_impl(**kwargs):
        return {
            "success": True,
            "request": {"detail": "compact"},
            "results": {
                "naive": {
                    "success": True,
                    "avg_mae": 1.0,
                    "avg_rmse": 1.2,
                    "avg_directional_accuracy": 0.0,
                    "successful_tests": 3,
                    "num_tests": 3,
                    "trade_status": "flat",
                    "metrics_available": False,
                    "metrics_reason": "no_non_flat_trades",
                    "metrics": {
                        "avg_return": None,
                        "avg_return_per_trade": None,
                        "win_rate": None,
                        "win_rate_display": None,
                        "max_drawdown": None,
                        "trades_observed": 0,
                    },
                    "details": [{"position": "flat"}],
                }
            },
        }

    result = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", detail="compact"),
        backtest_impl=fake_backtest_impl,
    )

    row = result["results"]["naive"]
    assert row["trade_status"] == "flat"
    assert row["metrics_available"] is False
    assert row["metrics_reason"] == "no_non_flat_trades"
    assert row["metrics_note"] == (
        "No active long/short trades; win_rate and drawdown need at least one trade."
    )
    assert "trades_observed" not in row
    assert "details_count" not in row
    assert "win_rate" not in row
    assert "win_rate_display" not in row
    assert "max_drawdown" not in row
    assert "avg_return" not in row
    assert "avg_return_per_trade" not in row
    assert result["ranking"] == {
        "metric": "avg_rmse",
        "direction": "ascending",
        "scope": "complete_methods_with_finite_avg_rmse",
        "note": (
            "Partial methods are excluded unless metrics are recomputed on an explicit "
            "common-anchor set; trading metrics do not affect rank."
        ),
    }
    assert result["ranked_methods"] == [
        {
            "method": "naive",
            "ranking_status": "ranked",
            "rank": 1,
            "avg_rmse": 1.2,
            "trading_metrics_available": False,
            "selection_warning": (
                "ranking_uses_forecast_error_only; trading metrics are unavailable"
            ),
            "trading_metrics_reason": "no_non_flat_trades",
        }
    ]


def test_run_forecast_backtest_handles_numpy_metrics_available_false():
    def fake_backtest_impl(**kwargs):
        return {
            "success": True,
            "results": {
                "naive": {
                    "success": True,
                    "metrics_available": np.bool_(False),
                    "metrics_reason": "no_non_flat_trades",
                    "metrics": {"win_rate": 1.0, "trades_observed": 1},
                    "details": [{"position": "flat"}],
                }
            },
        }

    result = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", detail="compact"),
        backtest_impl=fake_backtest_impl,
    )

    row = result["results"]["naive"]
    assert row["metrics_available"] == np.bool_(False)
    assert "metrics_note" in row
    assert "win_rate" not in row
    assert "details_count" not in row


def test_compact_backtest_marks_noncausal_rankings_research_only():
    result = forecast_use_cases._compact_backtest_result(
        {
            "success": True,
            "history_policy_ok": False,
            "history_policy_reason": "zero_phase_denoise_uses_future_observations",
            "results": {
                "naive": {
                    "success": True,
                    "avg_rmse": 0.01,
                    "metrics_available": True,
                    "forecast_reliability": "adequate",
                }
            },
        }
    )

    method = result["results"]["naive"]
    assert method["deployment_eligible"] is False
    assert method["forecast_reliability"] == "low"
    assert method["forecast_reliability_reason"] == (
        "zero_phase_denoise_uses_future_observations"
    )
    ranked = result["ranked_methods"][0]
    assert ranked["ranking_status"] == "research_only"
    assert ranked["deployment_eligible"] is False
    assert "noncausal_preprocessing_not_deployable" in ranked["selection_warning"]
    assert result["ranking"]["status"] == "research_only"


def test_run_forecast_backtest_routes_date_range_to_impl():
    captured = {}

    def fake_backtest_impl(**kwargs):
        captured.update(kwargs)
        return {"success": True, "results": {}}

    result = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(
            symbol="EURUSD",
            start="2023-01-01",
            end="2023-12-31",
            detail="full",
        ),
        backtest_impl=fake_backtest_impl,
    )

    assert result["success"] is True
    assert captured["start"] == "2023-01-01"
    assert captured["end"] == "2023-12-31"


def test_run_forecast_generate_routes_date_range_to_impl(monkeypatch):
    captured = {}

    def fake_forecast_impl(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "forecast_price": [1.0, 1.1],
            "volatility_per_bar": 0.01,
            "quantity": "volatility",
        }

    result = forecast_use_cases.run_forecast_generate(
        ForecastGenerateRequest(
            symbol="EURUSD",
            start="2023-01-01",
            end="2023-03-31",
            quantity="volatility",
        ),
        forecast_impl=fake_forecast_impl,
        resolve_sktime_forecaster=lambda _query: None,
    )

    assert result["success"] is True
    assert result["volatility_per_bar"] == pytest.approx(0.01)
    assert captured["start"] == "2023-01-01"
    assert captured["end"] == "2023-03-31"


def test_run_forecast_generate_routes_volatility_proxy_to_impl():
    captured = {}

    def fake_forecast_impl(**kwargs):
        captured.update(kwargs)
        return {"success": True, "method": kwargs["method"], "proxy": kwargs.get("proxy")}

    result = forecast_use_cases.run_forecast_generate(
        ForecastGenerateRequest(
            symbol="EURUSD",
            method="theta",
            quantity="volatility",
            proxy="abs_return",
        ),
        forecast_impl=fake_forecast_impl,
        resolve_sktime_forecaster=lambda _query: None,
    )

    assert result["success"] is True
    assert captured["quantity"] == "volatility"
    assert captured["proxy"] == "abs_return"


def test_run_forecast_generate_defaults_general_volatility_proxy():
    captured = {}

    def fake_forecast_impl(**kwargs):
        captured.update(kwargs)
        return {"success": True, "method": kwargs["method"], "proxy": kwargs.get("proxy")}

    result = forecast_use_cases.run_forecast_generate(
        ForecastGenerateRequest(
            symbol="EURUSD",
            method="theta",
            quantity="volatility",
        ),
        forecast_impl=fake_forecast_impl,
        resolve_sktime_forecaster=lambda _query: None,
    )

    assert result["success"] is True
    assert captured["proxy"] == "squared_return"
    assert any("defaulted proxy=squared_return" in item for item in result["warnings"])


def test_run_forecast_volatility_routes_date_range_to_impl():
    captured = {}

    def fake_volatility_impl(**kwargs):
        captured.update(kwargs)
        return {"success": True, "volatility_per_bar": 0.01}

    result = forecast_use_cases.run_forecast_volatility_estimate(
        ForecastVolatilityEstimateRequest(
            symbol="EURUSD",
            start="2023-01-01",
            end="2023-03-31",
        ),
        forecast_volatility_impl=fake_volatility_impl,
    )

    assert result["success"] is True
    assert captured["start"] == "2023-01-01"
    assert captured["end"] == "2023-03-31"


def test_forecast_generate_converts_typed_forecast_errors(monkeypatch):
    raw = _unwrap(cf.forecast_generate)

    monkeypatch.setattr(cf, "_forecast_impl", lambda **kwargs: (_ for _ in ()).throw(ForecastError("engine exploded")))

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="theta"))

    assert out["success"] is False
    assert out["error"] == "engine exploded"
    assert out["error_code"] == "forecast_generate_error"
    assert out["operation"] == "forecast_generate"
    assert isinstance(out.get("request_id"), str)


def test_forecast_generate_logs_finish_event(caplog, monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(cf, "_forecast_impl", lambda **kwargs: {"success": True, "method": kwargs["method"]})

    with caplog.at_level(logging.DEBUG, logger=cf.logger.name):
        out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="theta"))

    assert out["success"] is True
    assert any(
        "event=finish operation=forecast_generate success=True" in record.message
        for record in caplog.records
    )


def test_forecast_generate_wrapper_emits_single_finish_event(caplog, monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(cf, "_forecast_impl", lambda **kwargs: {"success": True, "method": kwargs["method"]})

    with caplog.at_level(logging.DEBUG):
        out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="theta"))

    assert out["success"] is True
    finish_records = [
        record
        for record in caplog.records
        if "event=finish operation=forecast_generate success=True" in record.message
    ]
    assert len(finish_records) == 1
    assert finish_records[0].name == cf.logger.name


def test_forecast_generate_returns_connection_error_payload(monkeypatch):
    raw = _unwrap(cf.forecast_generate)

    def fail_connection():
        raise MT5ConnectionError("Failed to connect to MetaTrader5. Ensure MT5 terminal is running.")

    monkeypatch.setattr(cf, "ensure_mt5_connection_or_raise", fail_connection)
    monkeypatch.setattr(cf, "_forecast_impl", lambda **kwargs: pytest.fail("forecast implementation should not run"))

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="theta"))

    assert out["success"] is False
    assert out["error"] == "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
    assert out["error_code"] == "mt5_connection_error"
    assert out["operation"] == "mt5_ensure_connection"
    assert isinstance(out.get("request_id"), str)


def test_forecast_tune_genetic_logs_finish_event(caplog, monkeypatch):
    raw = _unwrap(cf.forecast_tune_genetic)
    monkeypatch.setattr(cf, "run_forecast_tune_genetic", lambda request, genetic_search_impl: {"success": True, "best": {}})

    with caplog.at_level(logging.DEBUG, logger=cf.logger.name):
        out = raw(request=ForecastTuneGeneticRequest(symbol="EURUSD", methods=["theta"]))

    assert out["success"] is True
    assert any(
        "event=finish operation=forecast_tune_genetic success=True" in record.message
        for record in caplog.records
    )


def test_forecast_tune_detail_compacts_history_tail():
    def fake_genetic(**kwargs):
        return {
            "success": True,
            "history_count": 2,
            "history_tail": [{"score": 1.0}, {"score": 0.9}],
        }

    compact = forecast_use_cases.run_forecast_tune_genetic(
        ForecastTuneGeneticRequest(symbol="EURUSD", methods=["theta"]),
        genetic_search_impl=fake_genetic,
    )
    assert compact["detail"] == "compact"
    assert "history_tail" not in compact
    assert compact["history_tail_count"] == 2
    assert compact["history_count"] == 2

    full = forecast_use_cases.run_forecast_tune_genetic(
        ForecastTuneGeneticRequest(symbol="EURUSD", methods=["theta"], detail="full"),
        genetic_search_impl=fake_genetic,
    )
    assert full["detail"] == "full"
    assert full["history_tail"] == [{"score": 1.0}, {"score": 0.9}]
    assert "history_tail_count" not in full


def test_forecast_tune_compact_filters_units_and_keeps_data_window():
    def fake_genetic(**kwargs):
        return {
            "success": True,
            "best_score": 0.012,
            "metric": "avg_rmse",
            "units": {
                "best_score": "price",
                "avg_rmse": "price",
                "gross_before_costs_pct": "percent",
                "kelly_fraction": "fraction",
                "annual_return_pct": "percent",
            },
            "analysis_time_window": {
                "history_start": "2026-01-01T00:00Z",
                "history_end": "2026-01-10T00:00Z",
                "first_anchor": "2026-01-08T00:00Z",
                "last_anchor": "2026-01-08T00:00Z",
            },
            "history_bars_used": 100,
            "model_lookback_bars": 100,
            "history_tail": [{"score": 0.012}],
        }

    compact = forecast_use_cases.run_forecast_tune_genetic(
        ForecastTuneGeneticRequest(
            symbol="EURUSD",
            methods=["theta"],
            lookback=100,
        ),
        genetic_search_impl=fake_genetic,
    )
    assert compact["units"] == {"best_score": "price"}
    assert "tuning_context" not in compact
    assert compact["lookback"] == 100
    assert compact["model_lookback_bars"] == 100
    assert compact["history_bars_used"] == 100
    assert compact["analysis_time_window"]["history_start"] == (
        "2026-01-01T00:00Z"
    )
    assert compact["analysis_time_window"]["lookback"] == 100
    assert "history_tail" not in compact

    full = forecast_use_cases.run_forecast_tune_genetic(
        ForecastTuneGeneticRequest(
            symbol="EURUSD",
            methods=["theta"],
            lookback=100,
            detail="full",
        ),
        genetic_search_impl=fake_genetic,
    )
    assert full["units"]["kelly_fraction"] == "fraction"


def test_forecast_tuning_propagates_historical_anchor_and_discloses_window():
    captured = {}

    def fake_genetic(**kwargs):
        captured.update(kwargs)
        return {"success": True, "history_count": 1}

    result = forecast_use_cases.run_forecast_tune_genetic(
        ForecastTuneGeneticRequest(
            symbol="EURUSD",
            methods=["theta"],
            as_of="2025-12-31T21:00:00Z",
            max_search_time_seconds=15.0,
        ),
        genetic_search_impl=fake_genetic,
    )

    assert captured["as_of"] == "2025-12-31T21:00:00Z"
    assert captured["max_search_time_seconds"] == 15.0
    assert result["analysis_time_window"] == {
        "as_of": "2025-12-31T21:00:00Z",
        "timezone": "UTC",
        "input_bar_policy": "closed_bars_only",
        "reference_policy": "historical_candle_close",
    }


def test_forecast_tuning_resolves_direction_costs_and_sample_requirements():
    captured = {}

    def fake_genetic(**kwargs):
        captured.update(kwargs)
        return {"success": True, "best_score": 0.6, "mode": kwargs["mode"]}

    result = forecast_use_cases.run_forecast_tune_genetic(
        ForecastTuneGeneticRequest(
            symbol="EURUSD",
            methods=["drift"],
            metric="avg_rmse",
            slippage_bps=2.0,
        ),
        genetic_search_impl=fake_genetic,
    )

    assert captured["mode"] == "min"
    assert captured["slippage_bps"] == 2.0
    assert result["cost_assumptions"]["score_basis"] == "net_of_configured_slippage"
    assert result["cost_assumptions"]["spread_and_commission"] == "not_modeled"
    assert result["lookback"] is None

    rejected = forecast_use_cases.run_forecast_tune_genetic(
        ForecastTuneGeneticRequest(
            symbol="EURUSD",
            methods=["drift"],
            metric="calmar_ratio",
            steps=5,
        ),
        genetic_search_impl=lambda **kwargs: pytest.fail("search must not start"),
    )
    assert rejected["error_code"] == "insufficient_tuning_sample"
    assert rejected["minimum_steps"] == 30

    rejected_win_rate = forecast_use_cases.run_forecast_tune_genetic(
        ForecastTuneGeneticRequest(
            symbol="EURUSD",
            methods=["drift"],
            metric="win_rate",
            steps=5,
        ),
        genetic_search_impl=lambda **kwargs: pytest.fail("search must not start"),
    )
    assert rejected_win_rate["error_code"] == "insufficient_tuning_sample"
    assert rejected_win_rate["minimum_steps"] == 30

    rejected_costs = forecast_use_cases.run_forecast_tune_genetic(
        ForecastTuneGeneticRequest(
            symbol="EURUSD",
            methods=["drift"],
            metric="win_rate",
            steps=30,
        ),
        genetic_search_impl=lambda **kwargs: pytest.fail("search must not start"),
    )
    assert rejected_costs["error_code"] == "incomplete_cost_model"
    assert rejected_costs["missing_cost_parameters"] == [
        "spread_bps",
        "commission_bps_per_side",
    ]


def test_forecast_tune_optuna_and_optimize_hints_accept_detail():
    def fake_optuna(**kwargs):
        return {"success": True, "history_count": 1, "history_tail": [{"score": 1.0}]}

    def fake_hints(**kwargs):
        return {"success": True, "history_count": 1, "history_tail": [{"best_score": 0.5}]}

    optuna = forecast_use_cases.run_forecast_tune_optuna(
        ForecastTuneOptunaRequest(symbol="EURUSD", methods=["theta"], detail="standard"),
        optuna_search_impl=fake_optuna,
    )
    assert optuna["detail"] == "standard"
    assert "history_tail" not in optuna
    assert optuna["history_tail_count"] == 1

    hints = forecast_use_cases.run_forecast_optimize_hints(
        ForecastOptimizeHintsRequest(
            symbol="EURUSD",
            timeframes=["H1"],
            detail="summary",
            fitness_metric="avg_rmse",
        ),
        optimize_hints_impl=fake_hints,
    )
    assert hints["detail"] == "summary"
    assert "history_tail" not in hints
    assert hints["history_tail_count"] == 1


def test_forecast_tuners_mark_zero_phase_winners_research_only():
    denoise = {"method": "wavelet", "causality": "zero_phase"}
    genetic = forecast_use_cases.run_forecast_tune_genetic(
        ForecastTuneGeneticRequest(
            symbol="EURUSD",
            methods=["fourier_ols"],
            steps=30,
            denoise=denoise,
        ),
        genetic_search_impl=lambda **kwargs: {
            "success": True,
            "best_score": 0.1,
            "best_params": {"terms": 2},
        },
    )
    optuna = forecast_use_cases.run_forecast_tune_optuna(
        ForecastTuneOptunaRequest(
            symbol="EURUSD",
            methods=["fourier_ols"],
            steps=30,
            denoise=denoise,
        ),
        optuna_search_impl=lambda **kwargs: {
            "success": True,
            "best_score": 0.1,
            "best_params": {"terms": 2},
        },
    )
    hints = forecast_use_cases.run_forecast_optimize_hints(
        ForecastOptimizeHintsRequest(
            symbol="EURUSD",
            timeframes=["H1"],
            methods=["theta"],
            steps=30,
            denoise=denoise,
        ),
        optimize_hints_impl=lambda **kwargs: {
            "success": True,
            "hints": [{"method": "theta", "fitness_score": 0.1}],
        },
    )

    for result in (genetic, optuna, hints):
        assert result["denoise_causality"] == "zero_phase"
        assert result["denoise_live_safe"] is False
        assert result["denoise_usage"] == "research_only"
        assert result["selection_status"] == "research_only"
        assert result["deployment_eligible"] is False
        assert result["selection_reliability_reasons"] == [
            "zero_phase_denoise_uses_future_observations"
        ]
        assert any("future observations" in warning for warning in result["warnings"])
    assert hints["hints"][0]["deployment_eligible"] is False
    assert hints["hints"][0]["selection_status"] == "research_only"


@pytest.mark.parametrize("steps", [1, 29])
def test_forecast_tuners_label_small_anchor_searches_exploratory(steps):
    genetic = forecast_use_cases.run_forecast_tune_genetic(
        ForecastTuneGeneticRequest(
            symbol="EURUSD",
            methods=["fourier_ols"],
            steps=steps,
        ),
        genetic_search_impl=lambda **kwargs: {
            "success": True,
            "best_score": 0.1,
            "best_params": {"terms": 2},
        },
    )
    optuna = forecast_use_cases.run_forecast_tune_optuna(
        ForecastTuneOptunaRequest(
            symbol="EURUSD",
            methods=["fourier_ols"],
            steps=steps,
        ),
        optuna_search_impl=lambda **kwargs: {
            "success": True,
            "best_score": 0.1,
            "best_params": {"terms": 2},
        },
    )
    hints = forecast_use_cases.run_forecast_optimize_hints(
        ForecastOptimizeHintsRequest(
            symbol="EURUSD",
            timeframes=["H1"],
            methods=["theta"],
            steps=steps,
        ),
        optimize_hints_impl=lambda **kwargs: {
            "success": True,
            "hints": [{"method": "theta", "fitness_score": 0.1}],
        },
    )

    for result in (genetic, optuna, hints):
        assert result["selection_reliability"] == "low"
        assert result["selection_reliability_reasons"] == ["low_anchor_sample"]
        assert result["selection_status"] == "exploratory"
        assert result["deployment_eligible"] is False
        assert result["selection_sample"] == {
            "anchors_evaluated_per_candidate": steps,
            "minimum_recommended_anchors": 30,
        }
        assert any("exploratory" in warning for warning in result["warnings"])
    assert hints["hints"][0]["selection_status"] == "exploratory"


def test_forecast_tuning_thirty_causal_anchors_has_no_selection_blocker():
    result = forecast_use_cases.run_forecast_tune_genetic(
        ForecastTuneGeneticRequest(
            symbol="EURUSD",
            methods=["fourier_ols"],
            steps=30,
            denoise={"method": "ema", "causality": "causal"},
        ),
        genetic_search_impl=lambda **kwargs: {
            "success": True,
            "best_score": 0.1,
            "best_params": {"terms": 2},
        },
    )

    assert result["denoise_causality"] == "causal"
    assert result["denoise_live_safe"] is True
    assert "selection_reliability" not in result
    assert "deployment_eligible" not in result


def test_forecast_optimize_hints_rejects_unknown_method_before_search():
    result = forecast_use_cases.run_forecast_optimize_hints(
        ForecastOptimizeHintsRequest(
            symbol="EURUSD",
            timeframes=["H1"],
            methods=["theta", "not_a_method"],
            fitness_metric="avg_rmse",
        ),
        optimize_hints_impl=lambda **kwargs: pytest.fail("search must not start"),
    )

    assert result["success"] is False
    assert result["error_code"] == "unsupported_method"
    assert result["method"] == "not_a_method"
    assert result["valid_methods_tool"] == "forecast_list_methods"


@pytest.mark.parametrize(
    ("implementation", "operation"),
    [
        (cf._genetic_search_impl, "forecast_tune_genetic"),
        (cf._optuna_search_impl, "forecast_tune_optuna"),
        (cf._optimize_hints_impl, "forecast_optimize_hints"),
    ],
)
def test_forecast_tuners_reject_unknown_symbol_before_search(
    monkeypatch,
    implementation,
    operation,
):
    gateway = SimpleNamespace(symbol_info=lambda symbol: None)
    monkeypatch.setattr(cf, "create_mt5_gateway", lambda **kwargs: gateway)
    search = Mock(side_effect=AssertionError("candidate search must not start"))
    monkeypatch.setattr(cf, "_forecast_tune_module", lambda: SimpleNamespace(
        genetic_search_forecast_params=search,
        optuna_search_forecast_params=search,
        genetic_search_optimize_hints=search,
    ))

    result = implementation(symbol="NO_SUCH_SYMBOL")

    assert result["success"] is False
    assert result["error_code"] == "symbol_not_found"
    assert result["operation"] == operation
    assert result["history_count"] == 0
    assert result["related_tools"] == ["symbols_list"]
    search.assert_not_called()


def test_forecast_barrier_optimize_logs_finish_event(caplog, monkeypatch):
    raw = _unwrap(cf.forecast_barrier_optimize)
    monkeypatch.setattr(cf, "run_forecast_barrier_optimize", lambda request, parse_kv_or_json, barrier_optimize_impl: {"success": True, "best": {}})

    with caplog.at_level(logging.DEBUG, logger=cf.logger.name):
        out = raw(request=ForecastBarrierOptimizeRequest(symbol="EURUSD"))

    assert out["success"] is True
    assert any(
        "event=finish operation=forecast_barrier_optimize success=True" in record.message
        for record in caplog.records
    )


def test_forecast_barrier_optimize_request_defaults_to_summary_output():
    request = ForecastBarrierOptimizeRequest(symbol="EURUSD")
    assert request.search_profile == "medium"
    assert set(ForecastBarrierOptimizeRequest.model_fields) >= {
        "symbol",
        "timeframe",
        "horizon",
        "method",
        "direction",
        "mode",
        "params",
        "objective",
        "top_k",
        "candidate_filter",
        "min_ev",
        "min_edge",
        "min_kelly",
        "grid_style",
        "preset",
        "search_profile",
        "detail",
        "denoise",
    }
    assert "output_mode" not in ForecastBarrierOptimizeRequest.model_fields
    assert "tp_min" not in ForecastBarrierOptimizeRequest.model_fields
    assert "statistical_robustness" not in ForecastBarrierOptimizeRequest.model_fields
    assert "format" not in ForecastBarrierOptimizeRequest.model_fields


def test_forecast_barrier_requests_normalize_known_direction_aliases_only():
    barrier = {"kind": "tp_sl", "unit": "pct", "take_profit": 0.5, "stop_loss": 0.3}
    assert ForecastBarrierProbRequest(symbol="EURUSD", barrier=barrier, direction="buy").direction == "long"
    assert ForecastBarrierOptimizeRequest(symbol="EURUSD", direction="DOWN").direction == "short"
    with pytest.raises(ValidationError, match="direction"):
        ForecastBarrierProbRequest(symbol="EURUSD", barrier=barrier, direction="weird")


def test_forecast_barrier_optimize_request_rejects_removed_output_field():
    with pytest.raises(ValidationError, match="output was removed; use detail"):
        ForecastBarrierOptimizeRequest(symbol="EURUSD", output="summary")


def test_forecast_barrier_optimize_request_rejects_removed_format_field():
    with pytest.raises(ValidationError, match="format was removed; use json"):
        ForecastBarrierOptimizeRequest(symbol="EURUSD", format="full")


def test_forecast_barrier_optimize_request_rejects_top_level_advanced_params():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ForecastBarrierOptimizeRequest(symbol="EURUSD", tp_min=0.5)


def test_barrier_prob_price_rounding_uses_numeric_precision_without_padding():
    out = forecast_use_cases._round_barrier_prob_payload(
        {
            "success": True,
            "price_precision": 5,
            "last_price": "1.16976000",
            "tp_price": "1.18170000",
            "sl_price": "1.16415000",
            "prob_tp_first": 0.3333333333333,
        }
    )

    assert out["last_price"] == 1.16976
    assert out["tp_price"] == 1.1817
    assert out["sl_price"] == 1.16415
    assert isinstance(out["tp_price"], float)
    assert out["prob_tp_first"] == 0.333333


def test_barrier_symbol_price_precision_reads_cached_digits(monkeypatch):
    monkeypatch.setattr(
        "mtdata.utils.mt5.get_symbol_info_cached",
        lambda _symbol: SimpleNamespace(digits=5),
    )

    assert barriers_shared._symbol_price_precision("EURUSD") == 5


def test_forecast_list_library_models_and_list_methods(monkeypatch):
    stats_mod = ModuleType("statsforecast")
    models_mod = ModuleType("statsforecast.models")
    models_mod.__name__ = "statsforecast.models"
    models_mod.AutoARIMA = type(
        "AutoARIMA",
        (),
        {
            "__module__": "statsforecast.models",
            "fit": lambda self: None,
        },
    )
    models_mod.OtherHelper = 123
    stats_mod.models = models_mod

    monkeypatch.setitem(sys.modules, "statsforecast", stats_mod)
    monkeypatch.setattr(
        cf,
        "_discover_sktime_forecasters",
        lambda: {
            "thetaforecaster": ("ThetaForecaster", "sktime.forecasting.theta.ThetaForecaster"),
            "naiveforecaster": ("NaiveForecaster", "sktime.forecasting.naive.NaiveForecaster"),
        },
    )

    raw_list_models = _unwrap(cf.forecast_list_library_models)
    out_native = raw_list_models("native", detail="full")
    assert out_native["library"] == "native"
    assert out_native["catalog_source"] == "rebuilt"
    assert isinstance(out_native["models"], list)
    assert out_native["models"][0]["method"]
    assert out_native["models"][0]["available"] is True
    assert isinstance(out_native["capabilities"], list)
    assert out_native["capabilities"][0]["execution"]["library"] == "native"

    out_stats = raw_list_models("statsforecast", detail="full")
    assert out_stats["library"] == "statsforecast"
    assert any(
        row["method"] == "AutoARIMA" and row["model"] == "AutoARIMA"
        for row in out_stats["models"]
    )
    assert isinstance(out_stats["usage"], list)
    assert out_stats["capabilities"][0]["execution"]["library"] == "statsforecast"
    assert out_stats["capabilities"][0]["selector"]["key"] == "model_name"

    out_sktime = raw_list_models("sktime", detail="full")
    assert [row["model"] for row in out_sktime["models"]] == [
        "NaiveForecaster",
        "ThetaForecaster",
    ]
    assert out_sktime["capabilities"][0]["selector"]["key"] == "estimator"
    assert out_sktime["capabilities"][0]["execution"]["method"] == "sktime"

    out_ml = raw_list_models("mlforecast", detail="full")
    assert out_ml["library"] == "mlforecast"
    assert out_ml["models"][0]["method"] == "mlforecast"
    assert "note" in out_ml
    assert out_ml["capabilities"][0]["selector"]["key"] == "model"

    out_bad = raw_list_models("other")
    assert "Unsupported library" in out_bad["error"]

    class FalseLike:
        def __bool__(self):
            return False

    monkeypatch.setattr(
        cf,
        "_get_library_forecast_capabilities",
        lambda lib, **_kwargs: [
            {
                "method": "theta",
                "namespace": "native",
                "available": True,
                "execution": {"library": "native", "method": "theta"},
            },
            {
                "method": "unavailable_model",
                "namespace": "native",
                "available": FalseLike(),
                "execution": {"library": "native", "method": "unavailable_model"},
            },
        ]
        if lib == "native"
        else [],
    )
    out_native_available = raw_list_models("native", show_unavailable=False)
    assert out_native_available["models"] == [
        {"method": "theta", "available": True}
    ]
    assert out_native_available["total"] == 2
    assert out_native_available["total_filtered"] == 1
    assert out_native_available["available"] == 1
    assert out_native_available["filters"]["show_unavailable"] is False
    out_native_all = raw_list_models("native")
    assert out_native_all["models"] == [
        {"method": "theta", "available": True},
        {"method": "unavailable_model", "available": False},
    ]
    assert out_native_all["total"] == 2
    assert out_native_all["total_filtered"] == 2
    assert out_native_all["available"] == 1
    assert out_native_all["filters"]["show_unavailable"] is True

    monkeypatch.setattr(
        cf,
        "_get_registered_forecast_capabilities",
        lambda: [
            {
                "method": "theta",
                "supports": {"ci": True},
            },
            {
                "method": "mlf_rf",
                "supports": {"ci": False},
            },
            {
                "method": "sf_autoarima",
                "namespace": "statsforecast",
                "supports": {"ci": True},
            },
            {
                "method": "sf_theta",
                "namespace": "statsforecast",
                "supports": {"ci": True},
            },
            {
                "method": "sf_ets",
                "namespace": "statsforecast",
                "supports": {"ci": True},
            },
            {
                "method": "sf_naive",
                "namespace": "statsforecast",
                "supports": {"ci": True},
            },
        ],
    )

    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_data",
        lambda: {
            "total": 2,
            "categories": {"classical": ["theta"], "ml": ["mlf_rf"]},
            "methods": [
                {
                    "method": "theta",
                    "available": True,
                    "description": "Theta model.",
                    "params": [{"name": "window"}],
                    "requires": [],
                    "supports": {"ci": True},
                    "supports_training": False,
                },
                {
                    "method": "mlf_rf",
                    "available": False,
                    "description": "RF model.",
                    "params": [{"name": "n_estimators"}, {"name": "max_depth"}],
                    "requires": ["mlforecast", "sklearn"],
                    "supports_training": False,
                },
            ],
        },
    )
    compact = _unwrap(cf.forecast_list_methods)(profile="quickstart")
    assert compact["catalog_source"] == "rebuilt"
    assert "detail" not in compact
    assert compact["catalog_total"] == 2
    assert compact["available"] == 1
    assert compact["unavailable"] == 0
    assert compact["methods"][0]["method"] == "theta"
    assert "category_summary" not in compact
    assert "categories" not in compact
    assert "params_count" not in compact["methods"][0]
    assert "description" not in compact["methods"][0]
    assert compact["methods"][0]["supports_ci"] is True
    assert compact["methods"][0]["supports_training"] is False
    assert "namespace" not in compact["methods"][0]
    assert "method_id" not in compact["methods"][0]
    assert "concept" not in compact["methods"][0]
    assert "capability_id" not in compact["methods"][0]
    assert "adapter_method" not in compact["methods"][0]
    assert "selector" not in compact["methods"][0]
    assert "params" not in compact["methods"][0]
    assert all("requires" not in row for row in compact["methods"])
    assert compact["pagination"]["returned"] == 1
    assert compact["count"] == 1
    assert compact["pagination"]["more_available"] == 0
    assert compact["profile"] == "quickstart"
    assert compact["profile_methods_hidden"] == 1
    assert compact["profile_hint"] == "Use profile=all to list all registered methods."
    assert "filters" not in compact
    assert "barrier_methods" not in compact
    assert "note" not in compact
    assert "volatility_methods" not in compact

    standard = _unwrap(cf.forecast_list_methods)(detail="standard", profile="quickstart")
    assert standard["detail"] == "standard"
    assert standard["methods"][0]["description"] == "Theta model."
    assert standard["methods"][0]["params_count"] == 1
    assert "volatility_methods" not in standard
    assert "barrier_methods" not in standard

    volatility_filtered = _unwrap(cf.forecast_list_methods)(
        detail="full",
        profile="all",
        search_term="ewma",
        show_unavailable=True,
    )
    assert "volatility_methods" not in volatility_filtered
    assert "barrier_methods" not in volatility_filtered
    ewma_row = next(
        row
        for row in volatility_filtered["methods"]
        if row["method"] == "ewma"
    )
    assert ewma_row["tool"] == "forecast_volatility_estimate"
    assert any(param.get("name") == "lambda_" for param in ewma_row.get("params") or [])

    barrier_filtered = _unwrap(cf.forecast_list_methods)(
        detail="full",
        profile="all",
        search_term="bootstrap",
        show_unavailable=True,
    )
    bootstrap_row = next(
        row
        for row in barrier_filtered["methods"]
        if row["method"] == "bootstrap"
    )
    assert bootstrap_row["tool"] == "forecast_barrier_prob"
    assert any(param.get("name") == "block_size" for param in bootstrap_row.get("params") or [])

    compact_all = _unwrap(cf.forecast_list_methods)(show_unavailable=True, profile="all")
    unavailable_method = next(row for row in compact_all["methods"] if row["available"] is False)
    assert unavailable_method["unavailable_reason"] == "Requires: mlforecast, sklearn"

    compact_available = _unwrap(cf.forecast_list_methods)(
        profile="all",
        show_unavailable=False,
    )
    assert compact_available["catalog_total"] == 2
    assert compact_available["filtered_total"] == 2
    assert compact_available["available"] == 1
    assert compact_available["unavailable"] == 1
    assert compact_available["unavailable_hidden"] == 1
    assert len(compact_available["methods"]) == 1
    assert compact_available["pagination"]["total"] == 1

    full = _unwrap(cf.forecast_list_methods)(detail="full", show_unavailable=True, profile="all")
    assert full["detail"] == "full"
    assert full["catalog_total"] == 2
    assert full["filtered_total"] == 2
    assert full["available"] == 1
    assert full["unavailable"] == 1
    assert full["pagination"]["total"] == 2
    assert full["pagination"]["returned"] == 2
    assert isinstance(full.get("methods"), list)
    assert "params" in full["methods"][0]
    assert full["methods"][0]["params"] == [{"name": "window"}]
    assert "method_id" not in full["methods"][0]
    assert "capability_id" not in full["methods"][0]
    assert "concept" not in full["methods"][0]
    assert "adapter_method" not in full["methods"][0]
    assert "execution" not in full["methods"][0]
    assert "selector" not in full["methods"][0]
    assert full["methods"][0]["supports_ci"] is True
    assert full["methods"][0]["supports_training"] is False
    assert full["methods"][1]["supports_ci"] is False
    assert full["methods"][1]["supports_training"] is False
    assert full["methods"][1]["library"] == "native"
    assert "barrier_methods" not in full
    assert "volatility_methods" not in full
    assert full["count"] == 2

    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_data",
        lambda: {
            "total": 1,
            "categories": {"classical": ["naive"]},
            "methods": [
                {
                    "method": "naive",
                    "available": True,
                    "description": "naive",
                    "params": [],
                    "requires": [],
                },
            ],
        },
    )
    compact_repeated_description = _unwrap(cf.forecast_list_methods)()
    assert "profile" not in compact_repeated_description
    assert "description" not in compact_repeated_description["methods"][0]

    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_data",
        lambda: {
            "total": 5,
            "categories": {
                "classical": ["theta"],
                "statsforecast": ["sf_autoarima", "sf_theta", "sf_ets", "sf_naive"],
            },
            "methods": [
                {"method": "theta", "available": True, "description": "Theta.", "params": [], "requires": [], "supports_training": False},
                {"method": "sf_autoarima", "available": True, "description": "A", "params": [], "requires": [], "supports_training": True},
                {"method": "sf_theta", "available": True, "description": "B", "params": [], "requires": [], "supports_training": True},
                {"method": "sf_ets", "available": False, "description": "C", "params": [], "requires": [], "supports_training": True},
                {"method": "sf_naive", "available": True, "description": "D", "params": [], "requires": [], "supports_training": True},
            ],
        },
    )
    grouped = _unwrap(cf.forecast_list_methods)(profile="all")
    params = signature(_unwrap(cf.forecast_list_methods)).parameters
    assert "search_term" in params
    assert "search" not in params
    assert "category" in params
    assert "library" in params
    assert "supports_ci" in params
    assert "show_unavailable" in params
    assert "all" not in params
    sf_rows = [r for r in grouped["methods"] if r.get("category") == "statsforecast"]
    assert len(sf_rows) == 4
    assert all(str(r.get("category")) == "statsforecast" for r in sf_rows)
    assert grouped["pagination"]["more_available"] == 0
    filtered = _unwrap(cf.forecast_list_methods)(search_term="theta", limit=1, profile="all")
    assert "filters" not in filtered
    assert len(filtered["methods"]) == 1
    assert filtered["pagination"]["more_available"] >= 1
    assert "theta" in str(filtered["methods"][0]["method"]).lower()
    sf_only = _unwrap(cf.forecast_list_methods)(library="statsforecast", profile="all")
    assert "filters" not in sf_only
    assert all(
        row.get("category") == "statsforecast" for row in sf_only["methods"]
    )
    category_only = _unwrap(cf.forecast_list_methods)(category="statsforecast", profile="all")
    assert category_only["row_key"] == "methods"
    assert "filters" not in category_only
    assert all(row.get("category") == "statsforecast" for row in category_only["methods"])
    invalid_category = _unwrap(cf.forecast_list_methods)(
        category="statsforecast_typo", profile="all"
    )
    assert "Invalid category filter" in invalid_category["error"]
    assert "statsforecast" in invalid_category["error"]
    ci_only = _unwrap(cf.forecast_list_methods)(supports_ci=True, profile="all")
    assert "filters" not in ci_only
    assert ci_only["methods"]
    assert all(row.get("supports_ci") is True for row in ci_only["methods"])
    trainable_auto = _unwrap(cf.forecast_list_methods)(supports_training=True)
    assert [row["method"] for row in trainable_auto["methods"]] == [
        "sf_autoarima",
        "sf_naive",
        "sf_theta",
        "sf_ets",
    ]
    assert "profile" not in trainable_auto
    assert "profile_auto_expanded" not in trainable_auto
    no_ci = _unwrap(cf.forecast_list_methods)(supports_ci=False, show_unavailable=True, profile="all")
    assert "filters" not in no_ci
    assert all(row.get("supports_ci") is False for row in no_ci["methods"])
    with_unavailable = _unwrap(cf.forecast_list_methods)(show_unavailable=True, profile="all")
    assert with_unavailable["unavailable"] == 1
    assert any(row["available"] is False for row in with_unavailable["methods"])
    unavailable_row = next(row for row in with_unavailable["methods"] if row["available"] is False)
    assert unavailable_row["unavailable_reason"] == "Unavailable in the current environment."

    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_data",
        lambda: {
            "total": 25,
            "categories": {"classical": [f"m{i:02d}" for i in range(25)]},
            "methods": [
                {
                    "method": f"m{i:02d}",
                    "available": True,
                    "description": f"Method {i:02d}.",
                    "params": [],
                    "requires": [],
                }
                for i in range(25)
            ],
        },
    )
    default_out = _unwrap(cf.forecast_list_methods)(profile="all")
    assert "filters" not in default_out
    assert default_out["pagination"] == {
        "total": 25,
        "returned": 20,
        "offset": 0,
        "limit": 20,
        "has_more": True,
        "more_available": 5,
    }
    assert default_out["count_by_category"] == {"classical": 25}
    assert default_out["truncation_reason"] == (
        "Limit 20; set limit=25 for all filtered methods."
    )

    page = _unwrap(cf.forecast_list_methods)(limit=5, offset=5, profile="all")
    assert [row["method"] for row in page["methods"]] == [
        "m05",
        "m06",
        "m07",
        "m08",
        "m09",
    ]
    assert page["pagination"] == {
        "total": 25,
        "returned": 5,
        "offset": 5,
        "limit": 5,
        "has_more": True,
        "more_available": 15,
    }
    assert not {
        "total_filtered",
        "methods_shown",
        "methods_hidden",
        "methods_before",
        "offset",
        "has_more",
    } & page.keys()
    assert page["truncation_reason"] == (
        "Limit 5 at offset 5; set offset=10 for more filtered methods."
    )

    filtered_uncapped = _unwrap(cf.forecast_list_methods)(
        category="classical", limit=25, profile="all"
    )
    assert "filters" not in filtered_uncapped
    assert filtered_uncapped["pagination"]["returned"] == 25
    assert filtered_uncapped["pagination"]["more_available"] == 0
    assert "truncation_reason" not in filtered_uncapped

    monkeypatch.setattr(cf, "_get_forecast_methods_data", lambda: {"methods": [1]})
    assert _unwrap(cf.forecast_list_methods)() == {
        "methods": [1],
        "success": True,
        "catalog_source": "rebuilt",
    }
    monkeypatch.setattr(cf, "_get_forecast_methods_data", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert "Error listing forecast methods" in _unwrap(cf.forecast_list_methods)()["error"]


def test_forecast_generate_standard_preserves_requested_volatility_quantity():
    out = forecast_use_cases._apply_forecast_generate_detail(
        {
            "success": True,
            "method": "ewma",
            "horizon": 3,
            "volatility_per_bar": 0.01,
        },
        ForecastGenerateRequest(
            symbol="EURUSD",
            timeframe="H1",
            method="ewma",
            quantity="volatility",
            horizon=3,
            detail="standard",
        ),
    )

    assert out["quantity"] == "volatility"


def test_forecast_list_methods_standard_exposes_ci_method(monkeypatch):
    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_data",
        lambda: {
            "total": 3,
            "categories": {"ets_arima": ["arima"], "monte_carlo": ["mc_gbm"]},
            "methods": [
                {
                    "method": "arima",
                    "available": True,
                    "category": "ets_arima",
                    "description": "ARIMA model.",
                    "params": [],
                    "requires": [],
                    "supports": {"ci": True},
                },
                {
                    "method": "mc_gbm",
                    "available": True,
                    "category": "monte_carlo",
                    "description": "Monte Carlo GBM.",
                    "params": [],
                    "requires": [],
                    "supports": {"ci": True},
                },
                {
                    "method": "naive",
                    "available": True,
                    "category": "classical",
                    "description": "Naive model.",
                    "params": [],
                    "requires": [],
                    "supports": {"ci": False},
                },
            ],
        },
    )

    compact = _unwrap(cf.forecast_list_methods)(supports_ci=True, profile="all")
    assert all("ci_method" not in row for row in compact["methods"])

    standard = _unwrap(cf.forecast_list_methods)(
        detail="standard",
        supports_ci=True,
        profile="all",
    )
    methods = {row["method"]: row for row in standard["methods"]}

    assert set(methods) == {"arima", "mc_gbm"}
    assert methods["arima"]["ci_method"] == "statsmodels_prediction_interval"
    assert methods["mc_gbm"]["ci_method"] == "simulation_quantile"


def test_forecast_list_library_models_defaults_to_compact_page(monkeypatch):
    rows = [
        {
            "method": f"model_{idx}",
            "namespace": "native",
            "available": True,
            "execution": {"library": "native", "method": f"model_{idx}"},
        }
        for idx in range(25)
    ]
    monkeypatch.setattr(
        cf,
        "_get_library_forecast_capabilities",
        lambda lib, **_kwargs: rows if lib == "native" else [],
    )

    compact_page = cf._forecast_list_library_models_impl("native")
    assert "models_shown" not in compact_page
    assert compact_page["total_filtered"] == 25
    assert compact_page["pagination"] == {
        "total": 25,
        "returned": 20,
        "offset": 0,
        "limit": 20,
        "has_more": True,
        "more_available": 5,
    }
    assert "has_more" not in compact_page
    assert compact_page["filters"]["limit"] == 20

    full_page = cf._forecast_list_library_models_impl("native", detail="full")
    assert full_page["models_shown"] == 25
    assert full_page["filters"]["limit"] is None
    assert full_page["pagination"]["has_more"] is False


def test_forecast_list_single_library_does_not_discover_siblings(monkeypatch):
    requested_libraries = []

    def capabilities(library, **_kwargs):
        requested_libraries.append(library)
        return []

    monkeypatch.setattr(cf, "_get_library_forecast_capabilities", capabilities)

    result = cf._forecast_list_library_models_impl("sktime")

    assert result["library"] == "sktime"
    assert requested_libraries == ["sktime"]


def test_forecast_list_all_library_models_uses_one_global_page(monkeypatch):
    def capabilities(library, **_kwargs):
        return [
            {
                "method": f"{library}_{index}",
                "namespace": library,
                "available": True,
                "execution": {"library": library},
            }
            for index in range(3)
        ]

    monkeypatch.setattr(cf, "_get_library_forecast_capabilities", capabilities)

    page = cf._forecast_list_library_models_impl("all", limit=4, offset=2)

    shown = [model for section in page["libraries"] for model in section["models"]]
    assert len(shown) == 4
    assert page["pagination"]["offset"] == 2
    assert page["pagination"]["returned"] == 4
    assert page["page_order"] == "round_robin_by_library"

    default_page = cf._forecast_list_library_models_impl("all")
    assert default_page["pagination"]["returned"] == 15
    assert all(section["models"] for section in default_page["libraries"])
    assert all(section["has_more"] is False for section in default_page["libraries"])


def test_forecast_list_library_models_compact_deduplicates_model_rows(monkeypatch):
    rows = [
        {
            "method": "ARIMA",
            "display_name": "ARIMA",
            "available": True,
            "execution": {"library": "statsforecast"},
        },
        {
            "method": "theta",
            "display_name": "ThetaForecaster",
            "available": True,
            "execution": {"library": "sktime"},
        },
    ]
    monkeypatch.setattr(
        cf,
        "_get_library_forecast_capabilities",
        lambda *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(cf, "import_module", lambda _name: ModuleType("statsforecast"))

    compact = cf._forecast_list_library_models_impl(
        "statsforecast",
        show_unavailable=True,
    )

    assert compact["models"] == [
        {"method": "ARIMA", "available": True},
        {
            "method": "theta",
            "available": True,
            "model": "ThetaForecaster",
            "library": "sktime",
        },
    ]
    assert "unavailable" not in compact
    assert "unavailable_hidden" not in compact
    assert "models_shown" not in compact


def test_forecast_list_library_models_standard_keeps_requirements(monkeypatch):
    rows = [
        {
            "method": "ARIMA",
            "display_name": "ARIMA",
            "available": True,
            "selector": {"key": "model_name", "value": "ARIMA"},
            "execution": {"library": "statsforecast"},
            "requires": ["statsforecast"],
            "notes": "Supports seasonal configuration.",
        }
    ]
    monkeypatch.setattr(
        cf,
        "_get_library_forecast_capabilities",
        lambda *_args, **_kwargs: rows,
    )

    standard = cf._forecast_list_library_models_impl(
        "statsforecast",
        detail="standard",
        limit=5,
        offset=0,
    )

    assert standard["detail"] == "standard"
    assert standard["models"] == [
        {
            "method": "ARIMA",
            "available": True,
            "model": "ARIMA",
            "selector_value": "ARIMA",
            "selector_key": "model_name",
            "library": "statsforecast",
            "requires": ["statsforecast"],
            "notes": "Supports seasonal configuration.",
        }
    ]
    assert "capabilities" not in standard


def test_forecast_list_library_models_reports_missing_statsforecast(monkeypatch):
    raw_list_models = _unwrap(cf.forecast_list_library_models)

    monkeypatch.setattr(
        cf,
        "_get_library_forecast_capabilities",
        lambda *_args, **_kwargs: [],
    )

    def fake_import_module(name):
        if name == "statsforecast":
            raise ModuleNotFoundError("No module named 'statsforecast'")
        return ModuleType(name)

    monkeypatch.setattr(cf, "import_module", fake_import_module)

    out = raw_list_models("statsforecast")

    assert out == {
        "library": "statsforecast",
        "error": "statsforecast import failed: No module named 'statsforecast'",
    }


def test_forecast_list_methods_standard_describes_builtin_methods():
    standard = _unwrap(cf.forecast_list_methods)(
        detail="standard",
        show_unavailable=True,
    )
    missing = [
        row["method"]
        for row in standard["methods"]
        if not row.get("description")
    ]

    assert missing == []


def test_forecast_list_methods_uses_shared_snapshot(monkeypatch):
    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_snapshot",
        lambda: {
            "data": {
                "total": 1,
                "categories": {"statsforecast": ["sf_theta"]},
                "methods": [{"method": "sf_theta", "available": True}],
            },
            "method_to_category": {"sf_theta": "statsforecast"},
            "methods_valid": True,
            "methods": [
                {
                    "method": "sf_theta",
                    "available": True,
                    "category": "statsforecast",
                    "namespace": "statsforecast",
                    "description": "StatsForecast theta.",
                    "params": [{"name": "window"}],
                    "supports": {"ci": True},
                    "supports_training": True,
                    "supports_historical_exog": True,
                    "supports_future_exog": True,
                    "training_category": "moderate",
                    "method_id": "statsforecast:theta",
                    "capability_id": "statsforecast:theta",
                    "adapter_method": "statsforecast",
                    "selector": {"mode": "class_name", "key": "model_name", "value": "Theta"},
                    "execution": {
                        "library": "statsforecast",
                        "method": "statsforecast",
                        "params": {"model_name": "Theta"},
                    },
                }
            ],
        },
    )

    compact = _unwrap(cf.forecast_list_methods)(profile="all")
    standard = _unwrap(cf.forecast_list_methods)(detail="standard", profile="all")
    full = _unwrap(cf.forecast_list_methods)(detail="full", profile="all")

    assert "namespace" not in compact["methods"][0]
    assert compact["methods"][0]["supports_ci"] is True
    assert compact["methods"][0]["supports_training"] is True
    assert compact["methods"][0]["supports_historical_exog"] is True
    assert compact["methods"][0]["supports_future_exog"] is True
    assert standard["methods"][0]["namespace"] == "statsforecast"
    assert standard["methods"][0]["supports_historical_exog"] is True
    assert standard["methods"][0]["supports_future_exog"] is True
    assert standard["methods"][0]["description"] == "StatsForecast theta."
    assert full["methods"][0]["method_id"] == "statsforecast:theta"
    assert full["methods"][0]["training_category"] == "moderate"
    assert full["methods"][0]["supports_historical_exog"] is True
    assert full["methods"][0]["supports_future_exog"] is True
    assert full["methods"][0]["selector"]["key"] == "model_name"
    assert full["methods"][0]["execution"]["method"] == "statsforecast"


def test_forecast_list_methods_theta_supports_ci_from_interval_resolver(monkeypatch):
    monkeypatch.setattr(
        cf,
        "_get_registered_forecast_capabilities",
        lambda: [
            {"method": "theta", "supports": {"ci": False}},
            {"method": "drift", "supports": {"ci": False}},
        ],
    )
    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_data",
        lambda: {
            "total": 2,
            "categories": {"classical": ["theta", "drift"]},
            "methods": [
                {
                    "method": "theta",
                    "available": True,
                    "description": "Theta model.",
                    "params": [],
                    "requires": [],
                    "supports": {"ci": False},
                    "supports_training": False,
                },
                {
                    "method": "drift",
                    "available": True,
                    "description": "Drift model.",
                    "params": [],
                    "requires": [],
                    "supports": {"ci": False},
                    "supports_training": False,
                },
            ],
        },
    )

    compact = _unwrap(cf.forecast_list_methods)(profile="all")
    full = _unwrap(cf.forecast_list_methods)(detail="full", profile="all")
    by_compact = {row["method"]: row for row in compact["methods"]}
    by_full = {row["method"]: row for row in full["methods"]}

    assert cf._forecast_ci_method({"method": "theta"}) is None
    assert by_compact["theta"]["supports_ci"] is False
    assert "ci_method" not in by_compact["theta"]
    assert by_compact["drift"]["supports_ci"] is False
    assert by_full["theta"]["supports_ci"] is False
    assert "ci_method" not in by_full["theta"]
    assert by_full["drift"]["supports_ci"] is False


def test_forecast_generate_full_retains_interval_series(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1", "t2"],
            "forecast_price": [100.0, 101.0],
            "lower_price": [99.0, 99.5],
            "upper_price": [101.0, 102.5],
            "ci_status": "available",
            "ci_alpha": 0.05,
            "last_price": 100.0,
        },
    )

    out = raw(
        request=ForecastGenerateRequest(
            symbol="BTCUSD",
            timeframe="H1",
            method="arima",
            horizon=2,
            detail="full",
        )
    )

    assert out["lower_price"] == [99.0, 99.5]
    assert out["upper_price"] == [101.0, 102.5]
    assert out["forecast"][0]["lower_price"] == 99.0


def test_forecast_list_library_models_derives_pretrained_models_from_capabilities(monkeypatch):
    raw_list_models = _unwrap(cf.forecast_list_library_models)
    pretrained_caps = [
        {
            "method": "custom_pretrained",
            "requires": ["pkg-a", "pkg-b"],
            "params": [{"name": "model_name", "type": "str"}],
            "notes": "registry-backed note",
        }
    ]

    def fake_get_library_capabilities(library, **kwargs):
        if library == "pretrained":
            return pretrained_caps
        return []

    monkeypatch.setattr(cf, "_get_library_forecast_capabilities", fake_get_library_capabilities)

    out = raw_list_models("pretrained", detail="full")

    assert out["library"] == "pretrained"
    assert out["capabilities"] == pretrained_caps
    assert out["models"] == [
        {
            "method": "custom_pretrained",
            "available": True,
            "requires": ["pkg-a", "pkg-b"],
            "notes": "registry-backed note",
        }
    ]


def test_forecast_list_methods_does_not_require_mt5_connection(monkeypatch):
    def fail_connection():
        raise MT5ConnectionError("should not be called")

    monkeypatch.setattr(cf, "ensure_mt5_connection_or_raise", fail_connection)
    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_data",
        lambda: {"total": 1, "categories": {}, "methods": [{"method": "theta", "available": True}]},
    )

    out = _unwrap(cf.forecast_list_methods)()

    assert out["methods"][0]["method"] == "theta"


def test_registered_forecast_capabilities_are_cached(monkeypatch):
    calls = {"count": 0}

    class _FakeCapabilitiesModule:
        @staticmethod
        def get_registered_capabilities():
            calls["count"] += 1
            return [{"method": "theta"}]

    cf._get_registered_forecast_capabilities.cache_clear()
    monkeypatch.setattr(cf, "_forecast_capabilities_module", lambda: _FakeCapabilitiesModule())

    assert cf._get_registered_forecast_capabilities() == [{"method": "theta"}]
    assert cf._get_registered_forecast_capabilities() == [{"method": "theta"}]
    assert calls["count"] == 1

    cf._get_registered_forecast_capabilities.cache_clear()


def test_forecast_list_library_models_logs_finish_event(caplog):
    raw_list_models = _unwrap(cf.forecast_list_library_models)

    with caplog.at_level(logging.DEBUG, logger=cf.logger.name):
        out = raw_list_models("mlforecast")

    assert out["library"] == "mlforecast"
    assert any(
        "event=finish operation=forecast_list_library_models success=True" in record.message
        for record in caplog.records
    )


def test_forecast_conformal_intervals_success_and_errors(monkeypatch):
    raw = _unwrap(cf.forecast_conformal_intervals)

    invalid = raw(
        request=ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            method="not_a_method",
            horizon=2,
        )
    )
    assert invalid["success"] is False
    assert invalid["error_code"] == "invalid_method"
    assert invalid["operation"] == "forecast_conformal_intervals"
    assert invalid["details"] == {"method": "not_a_method"}
    assert invalid["related_tools"] == ["forecast_list_methods"]

    monkeypatch.setattr(
        cf,
        "_forecast_backtest_impl",
        lambda **kwargs: {
            "results": {
                "theta": {
                    "details": [
                        {"forecast": [10.0, 11.0], "actual": [9.0, 12.0]},
                        {"forecast": [13.0, 14.0], "actual": [12.0, 15.0]},
                    ]
                }
            }
        },
    )
    def fake_forecast_impl(symbol, timeframe, method, horizon, params=None, denoise=None):
        return {"forecast_price": [100.0, 101.0]}

    monkeypatch.setattr(cf, "_forecast_impl", fake_forecast_impl)

    out = raw(
        request=ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            method="theta",
            horizon=2,
            ci_alpha=0.1,
            steps=2,
        )
    )

    assert out["ci_alpha"] == 0.1
    assert out["nominal_confidence_level"] == 0.9
    assert out["empirical_coverage"] == 1.0
    assert out["coverage_status"] == "at_or_above_nominal_target"
    assert out["coverage_gap"] == pytest.approx(0.1)
    selected = _select_output_fields(
        out,
        "empirical_coverage,coverage_status,coverage_gap",
    )
    assert selected["coverage_gap"] == pytest.approx(0.1)
    assert "unresolved_output_fields" not in selected
    assert "confidence_level" not in out
    assert out["ci_status"] == "insufficient_calibration"
    assert out["ci_available"] is False
    assert out["interval_usage"] == "diagnostic_only"
    assert out["required_calibration_points"] == 30
    assert out["detail"] == "compact"
    assert out["conformal"]["ci_alpha"] == 0.1
    assert out["conformal"]["empirical_coverage"] == 1.0
    assert out["conformal"]["coverage_gap"] == pytest.approx(0.1)
    assert out["conformal"]["min_calibration_points"] == 2
    assert "lower_price" not in out
    assert "upper_price" not in out
    assert out["diagnostic_bounds"]["lower_price"][0] <= 100.0 <= out["diagnostic_bounds"]["upper_price"][0]
    assert out["trust_level"] == "degraded"
    assert "insufficient_interval_calibration" in out["trust_blockers"]

    monkeypatch.setattr(cf, "_forecast_backtest_impl", lambda **kwargs: {"error": "backtest failed"})
    assert raw(request=ForecastConformalIntervalsRequest(symbol="EURUSD", method="theta", horizon=2))["error"] == "backtest failed"

    monkeypatch.setattr(cf, "_forecast_backtest_impl", lambda **kwargs: {"results": {"theta": {"details": []}}})
    assert "Residual-quantile interval calibration failed" in raw(
        request=ForecastConformalIntervalsRequest(symbol="EURUSD", method="theta", horizon=2)
    )["error"]

    monkeypatch.setattr(cf, "_forecast_backtest_impl", lambda **kwargs: {"results": {"theta": {"details": [{}]}}})
    monkeypatch.setattr(cf, "_forecast_impl", lambda **kwargs: (_ for _ in ()).throw(ForecastError("engine exploded")))
    out = raw(request=ForecastConformalIntervalsRequest(symbol="EURUSD", method="theta", horizon=2))
    assert out["success"] is False
    assert out["error"] == "engine exploded"
    assert out["error_code"] == "forecast_conformal_intervals_error"
    assert out["operation"] == "forecast_conformal_intervals"
    assert isinstance(out.get("request_id"), str)


def test_forecast_conformal_intervals_request_defaults_and_spacing_validation():
    request = ForecastConformalIntervalsRequest(symbol="EURUSD")

    assert request.horizon == 12
    assert request.steps == 50
    assert request.spacing == 20
    assert request.ci_alpha == 0.05
    assert request.detail == "compact"

    assert ForecastConformalIntervalsRequest(
        symbol="EURUSD", ci_alpha=0.10
    ).ci_alpha == 0.10
    with pytest.raises(ValidationError, match="less than or equal to 0.5"):
        ForecastConformalIntervalsRequest(symbol="EURUSD", ci_alpha=0.95)

    with pytest.raises(ValidationError, match="spacing must be greater than or equal to horizon when steps > 1"):
        ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            horizon=12,
            steps=2,
            spacing=10,
        )


def test_conformal_intervals_forward_fixed_lookback_to_validation_and_forecast():
    calls = {}

    def fake_backtest(**kwargs):
        calls["backtest"] = kwargs
        return {
            "results": {
                "theta": {
                    "details": [
                        {"forecast": [10.0], "actual": [9.0]},
                    ]
                }
            }
        }

    def fake_forecast(**kwargs):
        calls["forecast"] = kwargs
        return {"forecast_price": [100.0]}

    result = forecast_use_cases.run_forecast_conformal_intervals(
        ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            method="theta",
            horizon=1,
            steps=1,
            lookback=50,
        ),
        backtest_impl=fake_backtest,
        forecast_impl=fake_forecast,
    )

    assert result["success"] is True
    assert calls["backtest"]["lookback"] == 50
    assert calls["forecast"]["lookback"] == 50


def test_run_forecast_conformal_intervals_routes_method_params_consistently():
    captured = {}

    def fake_backtest(**kwargs):
        captured["backtest"] = kwargs
        return {
            "results": {
                "theta": {
                    "details": [
                        {"forecast": [10.0], "actual": [9.0]},
                    ]
                }
            }
        }

    def fake_forecast(**kwargs):
        captured["forecast"] = kwargs
        return {"forecast_price": [100.0]}

    result = forecast_use_cases.run_forecast_conformal_intervals(
        ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            method="theta",
            horizon=1,
            steps=1,
            spacing=1,
            params={"seasonality": 24},
            detail="full",
        ),
        backtest_impl=fake_backtest,
        forecast_impl=fake_forecast,
    )

    assert captured["backtest"]["params_per_method"] == {"theta": {"seasonality": 24}}
    assert "params" not in captured["backtest"]
    assert captured["forecast"]["params"] == {"seasonality": 24}
    assert "detail" not in captured["forecast"]
    assert result["detail"] == "full"


def test_run_forecast_conformal_intervals_uses_finite_sample_quantile():
    result = forecast_use_cases.run_forecast_conformal_intervals(
        ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            method="theta",
            horizon=1,
            steps=3,
            spacing=1,
            ci_alpha=0.25,
            detail="full",
        ),
        backtest_impl=lambda **kwargs: {
            "results": {
                "theta": {
                    "details": [
                        {"forecast": [10.0], "actual": [9.0]},
                        {"forecast": [10.0], "actual": [8.0]},
                        {"forecast": [10.0], "actual": [7.0]},
                    ]
                }
            }
        },
        forecast_impl=lambda **kwargs: {"forecast_price": [100.0]},
    )

    assert result["conformal"]["per_step_q"] == [3.0]
    assert result["conformal"]["empirical_coverage"] == pytest.approx(2.0 / 3.0)
    assert result["conformal"]["empirical_coverage_per_step"] == [
        pytest.approx(2.0 / 3.0)
    ]
    assert result["conformal"]["calibration_points_per_step"] == [3]
    assert result["nominal_confidence_level"] == 0.75
    assert result["empirical_coverage"] == pytest.approx(2.0 / 3.0)
    assert result["coverage_status"] == "below_nominal_target"
    assert result["coverage_gap"] == pytest.approx(-0.083333)
    assert result["conformal"]["coverage_gap"] == pytest.approx(-0.083333)
    assert "confidence_level" not in result
    assert "lower_price" not in result
    assert "upper_price" not in result
    assert result["diagnostic_bounds"]["lower_price"] == [97.0]
    assert result["diagnostic_bounds"]["upper_price"] == [103.0]
    assert result["ci_status"] == "insufficient_calibration"
    assert result["ci_available"] is False
    assert result["trust_level"] == "degraded"
    assert "insufficient_interval_calibration" in result["trust_blockers"]


def test_run_forecast_conformal_intervals_compact_omits_technical_metadata():
    result = forecast_use_cases.run_forecast_conformal_intervals(
        ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            method="theta",
            horizon=1,
            steps=1,
            spacing=1,
            ci_alpha=0.1,
        ),
        backtest_impl=lambda **kwargs: {
            "results": {
                "theta": {
                    "details": [
                        {"forecast": [10.0], "actual": [9.0]},
                    ]
                }
            }
        },
        forecast_impl=lambda **kwargs: {
            "forecast_time": ["2026-05-29 21:00"],
            "forecast_price": [100.123456789],
            "forecast_epoch": [1780088400.0],
            "forecast_anchor": "next_timeframe_bar_after_last_observation",
            "forecast_step_seconds": 3600,
            "forecast": [{"time": "2026-05-29 21:00", "value": 100.123456789}],
            "params_used": {"alpha": 0.2, "trend_slope": -0.000012493247702752267},
            "last_price": 1.15825,
            "last_price_source": "candle_close",
            "digits": 5,
            "price_precision": 5,
            "last_price_age_seconds": 12.5,
            "last_price_stale": False,
            "history_policy_ok": True,
            "freshness_basis": "last_completed_bar_close",
        },
    )

    assert result["detail"] == "compact"
    assert result["last_price_age_seconds"] == 12.5
    assert result["data_age_seconds"] == 12.5
    assert result["last_price_stale"] is False
    assert result["data_stale"] is False
    assert result["history_policy_ok"] is True
    assert result["last_price"] == 1.15825
    assert result["last_price_source"] == "candle_close"
    assert result["digits"] == 5
    assert result["price_precision"] == 5
    # Compact mode folds point/interval series into forecast rows and drops the
    # parallel technical arrays/metadata fields.
    assert "forecast_time" not in result
    assert "forecast_price" not in result
    assert "lower_price" not in result
    assert "upper_price" not in result
    assert result["forecast"] == [
        {
            "time": "2026-05-29T21:00Z",
            "value": 100.12346,
        }
    ]
    assert result["diagnostic_bounds"]["lower_price"] == [99.12346]
    assert result["diagnostic_bounds"]["upper_price"] == [101.12346]
    assert result["trust_level"] == "degraded"
    assert "insufficient_interval_calibration" in result["trust_blockers"]
    assert result["conformal"] == {
        "ci_alpha": 0.1,
            "calibration_steps": 1,
            "calibration_spacing": 1,
            "calibration_anchor_tests_planned": 1,
            "calibration_anchor_tests_succeeded": 1,
            "calibration_anchor_tests_failed": 0,
            "calibration_complete": True,
            "coverage_target": 0.9,
        "coverage_evaluation": "leave_one_out_calibration_residuals",
        "coverage_note": (
            "Empirical residual quantiles from rolling backtest; not a "
            "finite-sample conformal coverage guarantee."
        ),
        "interval_method": "rolling_residual_quantiles",
        "min_calibration_points": 1,
        "required_calibration_points": 30,
        "calibration_sufficient": False,
        "interval_usage": "diagnostic_only",
    }
    assert "per_step_q" not in result["conformal"]
    for key in (
        "forecast_epoch",
        "forecast_anchor",
        "forecast_step_seconds",
        "params_used",
    ):
        assert key not in result


def test_forecast_conformal_intervals_compact_marks_flat_point_forecast():
    out = forecast_use_cases._apply_conformal_intervals_detail(
        {
            "success": True,
            "method": "theta",
            "horizon": 2,
            "data_as_of": "2026-06-02 19:00",
            "last_observation_time": "2026-06-02 19:00",
            "forecast_time": ["2026-06-02 20:00", "2026-06-02 21:00"],
            "forecast_price": [1.23456, 1.23456],
            "lower_price": [1.23, 1.23],
            "upper_price": [1.24, 1.24],
            "digits": 5,
            "ci_alpha": 0.1,
        },
        ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            timeframe="H1",
            method="theta",
            horizon=2,
            detail="compact",
        ),
    )

    assert out["data_as_of"] == "2026-06-02T19:00Z"
    assert out["last_observation_time"] == "2026-06-02T19:00Z"
    assert out["forecast"] == [
        {"time": "2026-06-02T20:00Z", "value": 1.23456, "lower": 1.23, "upper": 1.24},
        {"time": "2026-06-02T21:00Z", "value": 1.23456, "lower": 1.23, "upper": 1.24},
    ]
    assert out["point_forecast_mode"] == "flat_model_path"


def test_run_forecast_conformal_intervals_rewrites_interval_unavailable_guidance():
    result = forecast_use_cases.run_forecast_conformal_intervals(
        ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            method="theta",
            horizon=1,
            steps=1,
            spacing=1,
        ),
        backtest_impl=lambda **kwargs: {
            "results": {
                "theta": {
                    "details": [
                        {"forecast": [10.0], "actual": [9.0]},
                    ]
                }
            }
        },
        forecast_impl=lambda **kwargs: {
            "forecast_price": [100.0],
            "ci_status": "unavailable",
            "warnings": [
                "Point forecast only for method 'theta'; confidence intervals are unavailable. "
                "Use forecast_conformal_intervals for uncertainty bands.",
                "native theta fallback used",
            ],
        },
    )

    assert result["ci_status"] == "insufficient_calibration"
    assert result["ci_available"] is False
    assert result["calibration_sufficient"] is False
    assert result["required_calibration_points"] == 30
    assert result["interval_usage"] == "diagnostic_only"
    assert "Increase --steps" in result["calibration_remediation"]
    assert "lower_price" not in result
    assert "upper_price" not in result
    assert result["diagnostic_bounds"]["lower_price"] == [99.0]
    assert result["diagnostic_bounds"]["upper_price"] == [101.0]
    assert result["warnings"][0] == "native theta fallback used"
    assert "at least 30" in result["warnings"][1]
    assert result["trust_level"] == "degraded"
    assert "insufficient_interval_calibration" in result["trust_blockers"]


def test_run_forecast_conformal_intervals_raises_typed_error_for_nested_error_payload():
    with pytest.raises(ForecastError, match="backtest failed"):
        forecast_use_cases.run_forecast_conformal_intervals(
            ForecastConformalIntervalsRequest(
                symbol="EURUSD",
                method="theta",
                horizon=1,
                steps=1,
                spacing=1,
            ),
            backtest_impl=lambda **kwargs: {"error": "backtest failed"},
            forecast_impl=lambda **kwargs: {"forecast_price": [100.0]},
        )


@pytest.mark.parametrize("ci_alpha", [0.05, 0.10])
@pytest.mark.parametrize(
    ("sample_size", "expected_status", "expected_available"),
    [
        (29, "insufficient_calibration", False),
        (30, "available", True),
    ],
)
def test_conformal_interval_availability_uses_calibration_threshold(
    ci_alpha, sample_size, expected_status, expected_available
):
    details = [
        {"forecast": [float(index)], "actual": [float(index) + 1.0]}
        for index in range(sample_size)
    ]
    result = forecast_use_cases.run_forecast_conformal_intervals(
        ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            method="theta",
            horizon=1,
            steps=sample_size,
            spacing=1,
            ci_alpha=ci_alpha,
        ),
        backtest_impl=lambda **kwargs: {
            "results": {"theta": {"details": details}}
        },
        forecast_impl=lambda **kwargs: {"forecast_price": [100.0]},
    )

    assert result["ci_status"] == expected_status
    assert result["ci_available"] is expected_available
    assert result["conformal"]["min_calibration_points"] == sample_size
    if expected_available:
        assert "lower_price" in result
        assert "diagnostic_bounds" not in result
    else:
        assert "lower_price" not in result
        assert "upper_price" not in result
        assert result["diagnostic_bounds"]["usage"] == "diagnostic_only"
        assert result["trust_level"] == "degraded"
        assert "insufficient_interval_calibration" in result["trust_blockers"]


def test_forecast_time_normalization_recurses_through_nested_payloads() -> None:
    payload = forecast_use_cases._normalize_forecast_time_fields(
        {
            "timezone": "America/New_York",
            "last_observation_time": "2026-07-15T16:00:00-04:00",
            "forecast_from": {
                "time": "2026-07-15T16:00:00-04:00",
                "anchor": "last_observation",
            },
            "forecast_time": ["2026-07-15T17:00:00-04:00"],
            "diagnostics": {
                "timezone": "America/New_York",
                "history_start_time": "2026-01-15T09:30:00-05:00",
                "history_end_time": "2026-07-15T16:00:00-04:00",
            },
        }
    )

    assert payload["timezone"] == "UTC"
    assert payload["last_observation_time"] == "2026-07-15T20:00Z"
    assert payload["forecast_from"]["time"] == "2026-07-15T20:00Z"
    assert payload["forecast_from"]["anchor"] == "last_observation"
    assert payload["forecast_time"] == ["2026-07-15T21:00Z"]
    assert payload["diagnostics"] == {
        "timezone": "UTC",
        "history_start_time": "2026-01-15T14:30Z",
        "history_end_time": "2026-07-15T20:00Z",
    }


def test_conformal_intervals_expose_actionable_direction_gate():
    details = [
        {"forecast": [100.0], "actual": [100.25]}
        for _ in range(30)
    ]

    result = forecast_use_cases.run_forecast_conformal_intervals(
        ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            method="theta",
            horizon=1,
            steps=30,
            spacing=1,
        ),
        backtest_impl=lambda **kwargs: {
            "results": {"theta": {"details": details}}
        },
        forecast_impl=lambda **kwargs: {
            "success": True,
            "method": "theta",
            "horizon": 1,
            "forecast_price": [101.0],
            "last_price": 100.0,
            "digits": 5,
        },
    )

    context = result["forecast_vs_last_price"]
    assert context["direction"] == "bullish"
    assert context["direction_actionable"] is True
    assert context["direction_status"] == "interval_confirmed"
    assert context["direction_interval_excludes_last_price"] is True
    assert context["direction_interval_basis"] == "horizon_interval_vs_last_price"
    assert result["ci_status"] == "available"


def test_forecast_tune_genetic_and_barrier_prob_routing(monkeypatch):
    raw_tune = _unwrap(cf.forecast_tune_genetic)
    raw_barrier = _unwrap(cf.forecast_barrier_prob)

    captured = {}
    ss_calls = {}

    def fake_genetic(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cf, "_genetic_search_impl", fake_genetic)

    import mtdata.forecast.tune as tune_mod

    def fake_default_search_space(method=None, methods=None):
        ss_calls["method"] = method
        ss_calls["methods"] = methods
        return {"theta": {"window": {"min": 1, "max": 3}}}

    monkeypatch.setattr(tune_mod, "default_search_space", fake_default_search_space)
    out = raw_tune(request=ForecastTuneGeneticRequest(symbol="EURUSD", methods=["theta"], search_space=None))
    assert out["ok"] is True
    assert out["detail"] == "compact"
    assert out["compute_intensity"] == "high"
    assert captured["method"] is None
    assert captured["methods"] == ["theta"]
    assert ss_calls["method"] is None
    assert ss_calls["methods"] == ["theta"]
    assert "theta" in captured["search_space"]

    out = raw_tune(
        request=ForecastTuneGeneticRequest(
            symbol="EURUSD",
            methods=["fourier_ols", "naive"],
            search_space={
                "fourier_ols": {
                    "terms": {"type": "int", "min": 1, "max": 3}
                },
                "naive": {},
            },
        )
    )
    assert out["ok"] is True
    assert out["detail"] == "compact"
    assert out["compute_intensity"] == "high"
    assert captured["method"] is None
    assert out["symbol"] == "EURUSD"
    assert out["timeframe"] == "H1"
    assert out["quantity"] == "price"
    assert out["horizon"] == 12
    assert out["steps"] == 5
    assert out["spacing"] == 20
    assert out["methods"] == ["fourier_ols", "naive"]

    monkeypatch.setattr(cf, "_genetic_search_impl", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fail")))
    assert "Error in genetic tuning" in raw_tune(request=ForecastTuneGeneticRequest(symbol="EURUSD"))["error"]

    monkeypatch.setattr(cf, "_build_barrier_kwargs_from", lambda _: {"tp_abs": 1.2, "sl_abs": 1.1})

    import mtdata.forecast.barriers_probabilities as barriers_mod

    called = {}

    def fake_mc(**kwargs):
        called.update(kwargs)
        return {"kind": "mc", "direction": kwargs["direction"], "method": kwargs["method"]}

    def fake_cf(**kwargs):
        called.update(kwargs)
        return {"kind": "cf", "direction": kwargs["direction"]}

    monkeypatch.setattr(barriers_mod, "forecast_barrier_hit_probabilities", fake_mc)
    monkeypatch.setattr(barriers_mod, "forecast_barrier_closed_form", fake_cf)

    out = raw_barrier(
        request=ForecastBarrierProbRequest(
            symbol="EURUSD",
            method="auto",
            direction="down",
            barrier={"kind": "tp_sl", "unit": "price", "take_profit": 1.2, "stop_loss": 1.1},
        )
    )
    assert out["kind"] == "mc"
    assert out["method"] == "auto"
    assert out["method_source"] == "auto_selection"
    assert out["method_requested"] == "auto"
    assert out["direction"] == "short"
    assert out["detail"] == "compact"

    out = raw_barrier(
        request=ForecastBarrierProbRequest(
            symbol="EURUSD",
            barrier={"kind": "tp_sl", "unit": "price", "take_profit": 1.2, "stop_loss": 1.1},
        )
    )
    assert out["kind"] == "mc"
    assert out["method"] == "mc_gbm_bb"
    assert out["method_source"] == "auto_for_barrier_kind"

    out = raw_barrier(
        request=ForecastBarrierProbRequest(
            symbol="EURUSD",
            barrier={"kind": "single_price", "level": 1.2},
        )
    )
    assert out["kind"] == "cf"
    assert out["method_source"] == "auto_for_barrier_kind"

    out = raw_barrier(
        request=ForecastBarrierProbRequest(
            symbol="EURUSD",
            method="auto",
            barrier={"kind": "single_price", "level": 1.2},
        )
    )
    assert out["kind"] == "cf"
    assert out["method_source"] == "auto_selection"
    assert out["method_requested"] == "auto"
    assert out["method_used"] == "closed_form"
    assert out["auto_reason"] == "auto: single_price barrier; closed_form"

    with pytest.raises(ValidationError, match="direction"):
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            method="closed_form",
            direction="weird",
            barrier={"kind": "single_price", "level": 1.2},
        )

    with pytest.raises(ValidationError, match="method"):
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            method="mystery",
            barrier={"kind": "tp_sl", "unit": "price", "take_profit": 1.2, "stop_loss": 1.1},
        )


def test_forecast_barrier_methods_reject_legacy_aliases():
    barrier = {"kind": "tp_sl", "unit": "pct", "take_profit": 0.5, "stop_loss": 0.3}
    with pytest.raises(ValidationError, match="method"):
        ForecastBarrierProbRequest(symbol="EURUSD", method="monte_carlo", barrier=barrier)
    with pytest.raises(ValidationError, match="method"):
        ForecastBarrierOptimizeRequest(symbol="EURUSD", method="monte_carlo_bb")


def test_forecast_barrier_prob_requires_explicit_barriers(monkeypatch):
    with pytest.raises(ValidationError, match="barrier"):
        ForecastBarrierProbRequest(symbol="EURUSD")


def test_forecast_barrier_prob_kind_string_includes_json_example():
    with pytest.raises(ValidationError, match=r'kind":"tp_sl"'):
        ForecastBarrierProbRequest(symbol="EURUSD", barrier="tp_sl")


def test_forecast_barrier_prob_keeps_partial_barrier_inputs_strict():
    with pytest.raises(ValidationError, match="stop_loss"):
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            barrier={"kind": "tp_sl", "unit": "pct", "take_profit": 0.5},
        )


def test_forecast_barrier_optimize_rejects_unknown_method_without_traceback():
    called = False

    def fake_optimize(**_kwargs):
        nonlocal called
        called = True
        return {"success": True}

    with pytest.raises(ValidationError, match="method"):
        ForecastBarrierOptimizeRequest(symbol="EURUSD", method="mystery")
    assert called is False


def test_forecast_barrier_optimize_rounds_public_float_artifacts():
    def fake_optimize(**_kwargs):
        return {
            "success": True,
            "price_precision": 5,
            "best": {
                "tp": 0.45833333333333337,
                "sl": 0.25,
                "rr": 1.8333333333333335,
                "tp_price": 1.1764675416666668,
                "sl_price": 1.1681722500000001,
                "prob_resolve": 0.46950000000000003,
                "edge": -0.21050000000000002,
                "edge_vs_breakeven": -0.2234411764705882,
                "profit_factor": 0.6982843137254903,
            },
            "results": [
                {
                    "tp": 0.45833333333333337,
                    "sl": 0.25,
                    "sl_price": 1.1681722500000001,
                    "edge": -0.21050000000000002,
                    "edge_vs_breakeven": -0.2234411764705882,
                }
            ],
        }

    out = forecast_use_cases.run_forecast_barrier_optimize(
        ForecastBarrierOptimizeRequest(symbol="EURUSD", method="mc_gbm"),
        parse_kv_or_json=lambda value: value or {},
        barrier_optimize_impl=fake_optimize,
    )

    assert out["best"]["tp"] == 0.458333
    assert out["best"]["rr"] == 1.8333
    assert out["best"]["tp_price"] == 1.17647
    assert out["best"]["sl_price"] == 1.16817
    assert out["best"]["prob_resolve"] == 0.4695
    assert out["best"]["edge"] == -0.2105
    assert out["best"]["edge_vs_breakeven"] == -0.223441
    assert out["best"]["profit_factor"] == 0.698284
    assert "results" not in out
    assert out["ranked_candidates"] == [
        {"rank": 1, "tp": 0.458333, "sl": 0.25, "edge": -0.2105}
    ]
    assert out["barrier_unit"] == "percent"
    assert out["probability_unit"] == "fraction"
    assert out["edge_definition"] == "prob_win - prob_loss (probability fraction)."


def test_forecast_barrier_optimize_uses_tick_unit_context():
    def fake_optimize(**kwargs):
        return {
            "success": True,
            "mode": kwargs["mode"],
            "distance_unit": kwargs["mode"],
            "status": "ok",
            "best": {"ev": 1.0, "prob_tp_first": 0.6, "prob_sl_first": 0.4},
        }

    out = forecast_use_cases.run_forecast_barrier_optimize(
        ForecastBarrierOptimizeRequest(symbol="EURUSD", method="mc_gbm", mode="ticks"),
        parse_kv_or_json=lambda value: value or {},
        barrier_optimize_impl=fake_optimize,
    )

    assert out["distance_unit"] == "ticks"
    assert out["barrier_unit"] == "ticks"
    assert out["barrier_mode"] == "ticks"
    assert out["probability_unit"] == "fraction"


def test_forecast_barrier_optimize_compact_trims_blocked_status_noise():
    reason = "No viable TP/SL candidates satisfied the viability filter."

    def fake_optimize(**_kwargs):
        return {
            "success": True,
            "candidates_evaluated": 0,
            "candidates_viable": 0,
            "candidates_returned": 0,
            "best": None,
            "viable": False,
            "no_candidates": True,
            "status": "non_viable",
            "status_reason": reason,
            "no_action": True,
            "no_action_reason": reason,
            "actionability": "blocked",
            "actionability_reason": reason,
            "actionability_flags": ["status_non_viable", "warning"],
            "mathematically_viable": False,
            "tradable": False,
            "trade_gate_passed": False,
            "warning": reason,
            "output_mode": "concise",
            "viable_only": True,
            "concise": True,
            "reference_price": 1.16606,
            "reference_price_source": "live_tick_ask",
            "usable_for_live_trading": True,
            "usable_for_live_trading_basis": "model_history_and_reference_quote",
            "execution_blockers": [],
        }

    out = forecast_use_cases.run_forecast_barrier_optimize(
        ForecastBarrierOptimizeRequest(symbol="EURUSD", method="mc_gbm"),
        parse_kv_or_json=lambda value: value or {},
        barrier_optimize_impl=fake_optimize,
    )

    assert out["status"] == "non_viable"
    assert out["status_reason"] == reason
    assert out["tradable"] is False
    assert out["candidates_evaluated"] == 0
    assert out["candidates_viable"] == 0
    assert out["candidates_returned"] == 0
    assert out["best"] is None
    assert out["reference_price"] == 1.16606
    assert out["usable_for_live_trading"] is False
    assert out["usable_for_live_trading_basis"] == (
        "model_viability_and_reference_quote"
    )
    assert out["execution_blockers"] == ["optimizer_non_viable"]
    assert out["mathematically_viable"] is False
    for key in (
        "no_action",
        "no_action_reason",
        "actionability",
        "actionability_reason",
        "actionability_flags",
        "trade_gate_passed",
        "viable",
        "no_candidates",
        "warning",
        "output_mode",
        "viable_only",
        "concise",
    ):
        assert key not in out


def test_forecast_barrier_optimize_distinguishes_risk_block_from_non_viability():
    def fake_optimize(**_kwargs):
        return {
            "success": True,
            "status": "ok",
            "best": {"ev": 0.12, "kelly": -0.04},
            "viable": True,
            "mathematically_viable": True,
            "tradable": False,
            "trade_gate_passed": False,
            "actionability_flags": ["phantom_profit_risk"],
            "usable_for_live_trading": True,
            "execution_blockers": [],
        }

    out = forecast_use_cases.run_forecast_barrier_optimize(
        ForecastBarrierOptimizeRequest(symbol="EURUSD", method="mc_gbm"),
        parse_kv_or_json=lambda value: value or {},
        barrier_optimize_impl=fake_optimize,
    )

    assert out["mathematically_viable"] is True
    assert out["tradable"] is False
    assert out["usable_for_live_trading"] is False
    assert out["execution_blockers"] == [
        "risk_actionability_gate_failed",
        "phantom_profit_risk",
    ]
    assert "optimizer_non_viable" not in out["execution_blockers"]


def test_forecast_barrier_prob_closed_form_rejects_tp_sl_inputs_before_generic_error():
    with pytest.raises(ValidationError, match="single_price"):
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            method="closed_form",
            barrier={
                "kind": "tp_sl",
                "unit": "pct",
                "take_profit": 0.5,
                "stop_loss": 0.5,
            },
        )


def test_forecast_barrier_prob_closed_form_rejects_barrier_with_tp_sl_inputs():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            method="closed_form",
            barrier={"kind": "single_price", "level": 1.18},
            tp_pct=0.5,
            sl_pct=0.3,
        )


def test_forecast_barrier_prob_wrapper_emits_single_finish_event(caplog, monkeypatch):
    raw = _unwrap(cf.forecast_barrier_prob)
    monkeypatch.setattr(cf, "_forecast_connection_error", lambda: None)
    monkeypatch.setattr(cf, "_build_barrier_kwargs_from", lambda _: {"tp_abs": 1.2, "sl_abs": 1.1})

    import mtdata.forecast.barriers_probabilities as barriers_mod

    monkeypatch.setattr(
        barriers_mod,
        "forecast_barrier_hit_probabilities",
        lambda **kwargs: {"success": True, "kind": "mc", "direction": kwargs["direction"]},
    )
    monkeypatch.setattr(
        barriers_mod,
        "forecast_barrier_closed_form",
        lambda **kwargs: {"success": True, "kind": "closed_form", "direction": kwargs["direction"]},
    )

    with caplog.at_level(logging.DEBUG):
        out = raw(
            request=ForecastBarrierProbRequest(
                symbol="EURUSD",
                timeframe="H1",
                barrier={
                    "kind": "tp_sl",
                    "unit": "price",
                    "take_profit": 1.2,
                    "stop_loss": 1.1,
                },
            )
        )

    assert out["success"] is True
    finish_records = [
        record
        for record in caplog.records
        if "event=finish operation=forecast_barrier_prob success=True" in record.message
    ]
    assert len(finish_records) == 1


def test_forecast_operation_attaches_mt5_source_when_connection_is_required(
    monkeypatch,
):
    monkeypatch.setattr(cf, "_forecast_connection_error", lambda: None)

    result = cf._run_forecast_operation(
        "forecast_test",
        func=lambda: {"success": True, "symbol": "EURUSD"},
        require_connection=True,
    )

    assert result["source"]["provider"] == "mt5"
    assert "login" not in result["source"]


def test_forecast_barrier_prob_standard_hides_curves_only(monkeypatch):
    raw = _unwrap(cf.forecast_barrier_prob)
    monkeypatch.setattr(cf, "_forecast_connection_error", lambda: None)
    monkeypatch.setattr(cf, "_build_barrier_kwargs_from", lambda _: {"tp_abs": 1.2, "sl_abs": 1.1})

    import mtdata.forecast.barriers_probabilities as barriers_mod

    monkeypatch.setattr(
        barriers_mod,
        "forecast_barrier_hit_probabilities",
        lambda **kwargs: {
            "success": True,
            "symbol": kwargs["symbol"],
            "timeframe": kwargs["timeframe"],
            "method": kwargs["method"],
            "direction": kwargs["direction"],
            "horizon": kwargs["horizon"],
            "last_price": 1.15,
            "tp_price": 1.2,
            "sl_price": 1.1,
            "prob_tp_first": 0.55,
            "prob_sl_first": 0.30,
            "prob_no_hit": 0.15,
            "prob_tp_first_ci95": {"low": 0.5, "high": 0.6},
            "tp_hit_prob_by_t": [0.1, 0.2],
            "sl_hit_prob_by_t": [0.05, 0.1],
            "sim_meta": {"foo": "bar"},
        },
    )
    monkeypatch.setattr(barriers_mod, "forecast_barrier_closed_form", lambda **kwargs: {"success": True})

    out = raw(
        request=ForecastBarrierProbRequest(
            symbol="EURUSD",
            detail="standard",
            barrier={
                "kind": "tp_sl",
                "unit": "price",
                "take_profit": 1.2,
                "stop_loss": 1.1,
            },
        )
    )

    assert out["detail"] == "standard"
    assert "tp_hit_prob_by_t" not in out
    assert "sim_meta" not in out
    assert "prob_tp_first_ci95" in out


def test_forecast_barrier_prob_compact_keeps_confidence_and_history():
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "n_sims": 2000,
        "seed": 42,
        "seed_source": "derived_from_request",
        "prob_tp_first": 0.55,
        "prob_sl_first": 0.30,
        "prob_no_hit": 0.15,
        "prob_tp_first_se": 0.0111,
        "prob_sl_first_se": 0.0102,
        "prob_no_hit_se": 0.008,
        "prob_tp_first_ci95": {"low": 0.5, "high": 0.6},
        "prob_sl_first_ci95": {"low": 0.25, "high": 0.35},
        "prob_no_hit_ci95": {"low": 0.1, "high": 0.2},
        "history_bars_used": 2000,
        "data_as_of": "2026-08-25T14:00Z",
        "timezone": "UTC",
        "history_window": {
            "start": "2026-04-30T06:00Z",
            "end": "2026-08-25T14:00Z",
            "bars_used": 2000,
            "timezone": "UTC",
            "input_bar_policy": "closed_bars_only",
        },
        "intra_bar_hit_detection": "brownian_bridge",
        "bridge_correction": True,
        "bridge_dual_barrier_model": "independent_single_barrier_approximation",
        "bridge_joint_first_passage": False,
        "same_bar_policy": "random",
    }

    out = forecast_use_cases._apply_barrier_prob_detail(
        payload,
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            detail="compact",
            barrier={
                "kind": "tp_sl",
                "unit": "pct",
                "take_profit": 0.5,
                "stop_loss": 0.3,
            },
        ),
    )

    assert out["n_sims"] == 2000
    assert out["seed"] == 42
    assert out["seed_source"] == "derived_from_request"
    assert "confidence" not in out
    assert out["prob_tp_first_ci95"] == {"low": 0.5, "high": 0.6}
    assert out["prob_sl_first_ci95"] == {"low": 0.25, "high": 0.35}
    assert out["prob_no_hit_ci95"] == {"low": 0.1, "high": 0.2}
    assert out["prob_tp_first_se"] == 0.0111
    assert out["prob_sl_first_se"] == 0.0102
    assert out["prob_no_hit_se"] == 0.008
    assert out["history_bars_used"] == 2000
    assert out["timezone"] == "UTC"
    assert out["data_as_of"] == "2026-08-25T14:00Z"
    assert out["history_window"] == {
        "start": "2026-04-30T06:00Z",
        "end": "2026-08-25T14:00Z",
        "bars_used": 2000,
        "timezone": "UTC",
    }
    assert out["intra_bar_hit_detection"] == "brownian_bridge"
    assert out["bridge_correction"] is True
    assert out["bridge_dual_barrier_model"] == (
        "independent_single_barrier_approximation"
    )
    assert out["bridge_joint_first_passage"] is False
    assert out["same_bar_policy"] == "random"


def test_forecast_barrier_prob_compact_keeps_neutral_outcome_partition():
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "same_bar_policy": "neutral",
        "prob_tp_first": 0.088,
        "prob_sl_first": 0.165,
        "prob_no_hit": 0.0,
        "prob_same_bar": 0.747,
        "prob_unresolved": 0.747,
        "prob_resolve": 0.253,
    }

    out = forecast_use_cases._apply_barrier_prob_detail(
        payload,
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            detail="compact",
            same_bar_policy="neutral",
            barrier={
                "kind": "tp_sl",
                "unit": "ticks",
                "take_profit": 20,
                "stop_loss": 10,
            },
        ),
    )

    assert out["prob_same_bar"] == 0.747
    assert out["prob_unresolved"] == 0.747
    assert out["prob_resolve"] == 0.253
    assert sum(out[key] for key in (
        "prob_tp_first", "prob_sl_first", "prob_no_hit", "prob_same_bar"
    )) == pytest.approx(1.0)


def test_compact_barrier_candidate_keeps_neutral_outcome_partition():
    from mtdata.forecast.barriers_optimization import _compact_barrier_candidate

    out = _compact_barrier_candidate(
        {
            "tp": 20,
            "sl": 10,
            "prob_tp_first": 0.088,
            "prob_sl_first": 0.165,
            "prob_no_hit": 0.0,
            "prob_same_bar": 0.747,
            "prob_unresolved": 0.747,
            "prob_resolve": 0.253,
        }
    )

    assert out["prob_same_bar"] == 0.747
    assert out["prob_unresolved"] == 0.747
    assert out["prob_resolve"] == 0.253


def test_forecast_barrier_prob_compact_uses_reference_price_context():
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "last_price": 1.16026,
        "last_price_close": 1.16016,
        "last_price_source": "live_tick_ask",
        "tp_price": 1.16606,
        "sl_price": 1.15678,
        "prob_tp_first": 0.55,
        "prob_sl_first": 0.35,
        "prob_no_hit": 0.10,
    }

    out = forecast_use_cases._apply_barrier_prob_detail(
        payload,
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            detail="compact",
            barrier={
                "kind": "tp_sl",
                "unit": "pct",
                "take_profit": 0.5,
                "stop_loss": 0.3,
            },
        ),
    )

    assert out["reference_price"] == 1.16026
    assert out["reference_price_source"] == "live_tick_ask"
    assert out["tp_pct"] == 0.5
    assert out["sl_pct"] == 0.3
    assert "last_price" not in out
    assert "last_price_close" not in out
    assert "last_price_source" not in out


def test_forecast_barrier_prob_compact_includes_live_quote_freshness():
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "last_price": 1.16026,
        "last_price_source": "live_tick_ask",
        "reference_price_source": "live_tick_ask",
        "tp_price": 1.16606,
        "sl_price": 1.15678,
        "prob_tp_first": 0.55,
        "prob_sl_first": 0.35,
        "prob_no_hit": 0.10,
        "data_as_of": "2026-08-19T19:00:00Z",
        "reference_price_time": "2026-08-19T19:31:24Z",
        "reference_price_age_seconds": 12,
        "reference_price_stale": False,
        "reference_usable_for_live": True,
    }

    out = forecast_use_cases._apply_barrier_prob_detail(
        payload,
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            detail="compact",
            barrier={
                "kind": "tp_sl",
                "unit": "pct",
                "take_profit": 0.5,
                "stop_loss": 0.3,
            },
        ),
    )

    assert out["reference_price_source"] == "live_tick_ask"
    assert out["reference_price_time"] == "2026-08-19T19:31:24Z"
    assert out["reference_price_age_seconds"] == 12
    assert out["reference_price_stale"] is False
    assert out["reference_usable_for_live"] is True
    assert out["data_as_of"] == "2026-08-19T19:00:00Z"


def test_forecast_barrier_prob_closed_form_compact_keeps_reference_source():
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "last_price": 1.16594,
        "last_price_source": "candle_close",
        "barrier": 1.18,
        "prob_hit": 0.15,
        "log_drift_annual": 0.01,
        "sigma_annual": 0.2,
        "bars_per_year": 6240.0,
        "annualization_basis": "260_fx_weekdays_24h",
        "override_units": "annual_decimal_return_fraction",
    }

    out = forecast_use_cases._apply_barrier_prob_detail(
        payload,
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            method="closed_form",
            barrier={"kind": "single_price", "level": 1.18},
            detail="compact",
        ),
    )

    assert out["reference_price"] == 1.16594
    assert out["reference_price_source"] == "candle_close"
    assert out["prob_hit"] == 0.15
    assert out["sigma_annual"] == 0.2
    assert out["bars_per_year"] == 6240.0
    assert out["annualization_basis"] == "260_fx_weekdays_24h"
    assert "last_price" not in out
    assert "last_price_source" not in out


def test_forecast_barrier_prob_closed_form_compact_keeps_already_hit_and_warnings():
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "last_price": 1.18,
        "last_price_source": "candle_close",
        "barrier": 1.17,
        "prob_hit": 1.0,
        "already_hit": True,
        "warnings": ["Denoise request failed; using raw close prices instead: boom"],
        "denoise_status": "failed",
        "denoise_error": "boom",
        "usable_for_live_trading": False,
        "execution_blockers": ["live_reference_quote_not_used"],
    }

    out = forecast_use_cases._apply_barrier_prob_detail(
        payload,
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            method="closed_form",
            barrier={"kind": "single_price", "level": 1.17},
            detail="compact",
        ),
    )

    assert out["already_hit"] is True
    assert out["denoise_status"] == "failed"
    assert "usable_for_live_trading" not in out
    assert "Denoise request failed" in out["warnings"][0]


def test_forecast_barrier_prob_detail_rounds_display_values():
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "last_price": 1.1720124100000001,
        "tp_price": 1.1780124100000001,
        "sl_price": 1.1690124100000001,
        "prob_tp_first": 0.5123456789,
        "prob_sl_first": 0.4876543211,
        "probability_edge": -0.17800000000000005,
        "prob_tp_first_ci95": {"low": 0.5000000001, "high": 0.6000000001},
    }

    out = forecast_use_cases._apply_barrier_prob_detail(
        payload,
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            detail="compact",
            barrier={
                "kind": "tp_sl",
                "unit": "pct",
                "take_profit": 0.5,
                "stop_loss": 0.3,
            },
        ),
    )

    assert out["reference_price"] == 1.17201241
    assert "last_price" not in out
    assert out["tp_price"] == 1.17801241
    assert out["sl_price"] == 1.16901241
    assert out["prob_tp_first"] == 0.512346
    assert out["probability_edge"] == -0.178
    assert out["probability_unit"] == "fraction"
    assert out["probability_edge_definition"] == "prob_tp_first - prob_sl_first"
    assert "edge" not in out
    assert "confidence" not in out
    assert out["prob_tp_first_ci95"] == {"low": 0.5, "high": 0.6}


def test_forecast_barrier_prob_marks_stale_reference_verdict_research_only():
    out = forecast_use_cases._annotate_barrier_prob_context(
        {
            "prob_tp_first": 0.4,
            "prob_sl_first": 0.6,
            "usable_for_live_trading": False,
            "execution_blockers": ["reference_quote_not_live"],
        },
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            barrier={
                "kind": "tp_sl",
                "unit": "ticks",
                "take_profit": 200,
                "stop_loss": 150,
            },
        ),
    )

    assert out["signal_status"] == "not_actionable"
    assert out["verdict"] == "Research only — SL-first probability bias"


def test_forecast_barrier_prob_verdict_keeps_material_tp_sl_bias():
    barrier = {
        "kind": "tp_sl",
        "unit": "pct",
        "take_profit": 0.5,
        "stop_loss": 0.3,
    }
    tp_first = forecast_use_cases._annotate_barrier_prob_context(
        {"prob_tp_first": 0.6, "prob_sl_first": 0.4},
        ForecastBarrierProbRequest(symbol="EURUSD", barrier=barrier),
    )
    sl_first = forecast_use_cases._annotate_barrier_prob_context(
        {"prob_tp_first": 0.4, "prob_sl_first": 0.6},
        ForecastBarrierProbRequest(symbol="EURUSD", barrier=barrier),
    )

    assert tp_first["verdict"] == "TP-first probability bias"
    assert sl_first["verdict"] == "SL-first probability bias"


def test_forecast_barrier_prob_verdict_is_unresolved_when_almost_no_hits():
    out = forecast_use_cases._annotate_barrier_prob_context(
        {
            "prob_tp_first": 0.008,
            "prob_sl_first": 0.0,
            "probability_edge": 0.008,
            "prob_unresolved": 0.992,
            "prob_resolve": 0.008,
        },
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            barrier={
                "kind": "tp_sl",
                "unit": "pct",
                "take_profit": 0.4,
                "stop_loss": 0.6,
            },
        ),
    )

    assert out["verdict"] == "Mostly unresolved; barriers unlikely to be hit"


def test_forecast_barrier_prob_verdict_is_neutral_when_cis_overlap():
    out = forecast_use_cases._annotate_barrier_prob_context(
        {
            "prob_tp_first": 0.031,
            "prob_sl_first": 0.030,
            "probability_edge": 0.001,
            "prob_tp_first_ci95": {"low": 0.020, "high": 0.045},
            "prob_sl_first_ci95": {"low": 0.019, "high": 0.044},
        },
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            barrier={
                "kind": "tp_sl",
                "unit": "pct",
                "take_profit": 0.5,
                "stop_loss": 0.5,
            },
        ),
    )

    assert out["verdict"] == "Neutral first-hit probabilities"


def test_forecast_barrier_prob_verdict_is_neutral_when_edge_inside_se():
    out = forecast_use_cases._annotate_barrier_prob_context(
        {
            "prob_tp_first": 0.034,
            "prob_sl_first": 0.030,
            "probability_edge": 0.004,
            "prob_tp_first_se": 0.006,
            "prob_sl_first_se": 0.006,
        },
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            barrier={
                "kind": "tp_sl",
                "unit": "pct",
                "take_profit": 0.5,
                "stop_loss": 0.5,
            },
        ),
    )

    assert out["verdict"] == "Neutral first-hit probabilities"


def test_forecast_barrier_prob_compact_keeps_cis_after_neutral_verdict():
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "prob_tp_first": 0.031,
        "prob_sl_first": 0.030,
        "probability_edge": 0.001,
        "prob_tp_first_ci95": {"low": 0.020, "high": 0.045},
        "prob_sl_first_ci95": {"low": 0.019, "high": 0.044},
        "prob_tp_first_se": 0.006,
        "prob_sl_first_se": 0.006,
    }

    out = forecast_use_cases._apply_barrier_prob_detail(
        payload,
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            detail="compact",
            barrier={
                "kind": "tp_sl",
                "unit": "pct",
                "take_profit": 0.5,
                "stop_loss": 0.5,
            },
        ),
    )

    assert out["verdict"] == "Neutral first-hit probabilities"
    assert out["prob_tp_first_ci95"] == {"low": 0.02, "high": 0.045}
    assert out["prob_sl_first_ci95"] == {"low": 0.019, "high": 0.044}
    assert out["prob_tp_first_se"] == 0.006
    assert out["prob_sl_first_se"] == 0.006


def test_forecast_barrier_prob_compact_keeps_auto_selection_fields():
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "method": "jump_diffusion",
        "method_source": "auto_selection",
        "method_requested": "auto",
        "method_used": "jump_diffusion",
        "auto_reason": "auto: heavy tails/jump risk",
        "prob_tp_first": 0.2,
        "prob_sl_first": 0.1,
    }

    out = forecast_use_cases._apply_barrier_prob_detail(
        payload,
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            method="auto",
            detail="compact",
            barrier={
                "kind": "tp_sl",
                "unit": "pct",
                "take_profit": 0.2,
                "stop_loss": 0.1,
            },
        ),
    )

    assert out["method"] == "jump_diffusion"
    assert out["method_source"] == "auto_selection"
    assert out["method_requested"] == "auto"
    assert out["method_used"] == "jump_diffusion"
    assert out["auto_reason"] == "auto: heavy tails/jump risk"


def test_compact_barrier_optimize_json_uses_ranked_candidates_not_results():
    best = {
        "tp": 1.0,
        "sl": 0.5,
        "prob_tp_first": 0.55,
        "ev": 0.12,
        "edge": 0.04,
        "ev_ci95": {"low": 0.01, "high": 0.2},
        "kelly": 0.1,
        "profit_factor": 1.4,
        "timeout_mtm_contribution": 0.01,
        "same_bar_contribution": 0.0,
        "phantom_profit_risk": False,
    }
    payload = {
        "success": True,
        "detail": "compact",
        "best": best,
        "results": [
            dict(best),
            {
                "tp": 0.8,
                "sl": 0.4,
                "prob_tp_first": 0.5,
                "ev": 0.08,
                "edge": 0.02,
                "kelly": 0.05,
            },
            {
                "tp": 0.6,
                "sl": 0.3,
                "prob_tp_first": 0.4,
                "ev": 0.01,
                "edge": -0.01,
                "viable": False,
            },
        ],
        "actionability_flags": ["ok"],
        "viable": True,
    }

    compact = forecast_compact._compact_barrier_optimize_payload(payload)

    assert "results" not in compact
    assert compact["best"] == best
    assert [row["rank"] for row in compact["ranked_candidates"]] == [1, 2, 3]
    assert compact["ranked_candidates"][0]["tp"] == 1.0
    assert compact["ranked_candidates"][0]["ci"] == {"low": 0.01, "high": 0.2}
    assert "kelly" not in compact["ranked_candidates"][0]
    assert "timeout_mtm_contribution" not in compact["ranked_candidates"][0]
    assert compact["ranked_candidates"][2]["viable"] is False
    assert compact["ranked_candidates"][0] != compact["best"]


def test_barrier_method_catalog_matches_request_schema(monkeypatch):
    from mtdata.forecast.barrier_constants import (
        BARRIER_PROB_METHODS,
        BARRIER_SAMPLING_CI_METHODS,
        barrier_method_catalog_rows,
    )

    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_data",
        lambda: {"total": 0, "categories": {}, "methods": []},
    )

    schema = ForecastBarrierProbRequest.model_json_schema()
    method_prop = schema["properties"]["method"]
    enums = set()
    if "enum" in method_prop:
        enums.update(value for value in method_prop["enum"] if value is not None)
    for option in method_prop.get("anyOf", method_prop.get("oneOf", [])):
        if isinstance(option, dict) and "enum" in option:
            enums.update(value for value in option["enum"] if value is not None)

    catalog_rows = barrier_method_catalog_rows()
    catalog_names = {row["method"] for row in catalog_rows}
    assert set(BARRIER_PROB_METHODS) == catalog_names
    assert enums == catalog_names

    rows = {row["method"]: row for row in catalog_rows}
    assert rows["auto"]["barrier_kinds"] == ["single_price", "tp_sl"]
    assert rows["closed_form"]["barrier_kinds"] == ["single_price"]
    assert rows["closed_form"]["supports_ci"] is False
    assert rows["mc_gbm_bb"]["supports_ci"] is True
    assert rows["mc_gbm_bb"]["ci_method"] == "simulation_sampling_interval"
    for name in BARRIER_SAMPLING_CI_METHODS:
        assert rows[name]["supports_ci"] is True

    listed = _unwrap(cf.forecast_list_methods)(
        category="barrier",
        profile="all",
        detail="full",
        show_unavailable=True,
        limit=50,
    )
    listed_names = {row["method"] for row in listed["methods"]}
    assert catalog_names <= listed_names
    listed_rows = {row["method"]: row for row in listed["methods"]}
    assert listed_rows["auto"]["barrier_kinds"] == ["single_price", "tp_sl"]
    assert listed_rows["closed_form"]["supports_ci"] is False
    assert listed_rows["mc_gbm_bb"]["supports_ci"] is True
    assert listed_rows["mc_gbm_bb"]["ci_method"] == "simulation_sampling_interval"

    ci_listed = _unwrap(cf.forecast_list_methods)(
        category="barrier",
        profile="all",
        supports_ci=True,
        show_unavailable=True,
        limit=50,
    )
    ci_names = {row["method"] for row in ci_listed["methods"]}
    assert "mc_gbm_bb" in ci_names
    assert "auto" in ci_names
    assert "closed_form" not in ci_names


@pytest.mark.parametrize("detail", ["compact", "full"])
def test_forecast_barrier_prob_keeps_blockers_without_execution_readiness(detail):
    out = forecast_use_cases._apply_barrier_prob_detail(
        {
            "success": True,
            "prob_tp_first": 0.4,
            "prob_sl_first": 0.6,
            "usable_for_live_trading": False,
            "usable_for_live_trading_basis": "model_history_and_reference_quote",
            "execution_blockers": ["live_reference_quote_not_used"],
            "remediation": {"next_steps": ["Fetch a current two-sided quote."]},
        },
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            detail=detail,
            barrier={
                "kind": "tp_sl",
                "unit": "pct",
                "take_profit": 0.5,
                "stop_loss": 0.3,
            },
        ),
    )

    assert "usable_for_live_trading" not in out
    assert "usable_for_live_trading_basis" not in out
    assert out["execution_blockers"] == ["live_reference_quote_not_used"]
    assert out["remediation"] == {
        "next_steps": ["Fetch a current two-sided quote."]
    }


def test_forecast_barrier_optimize_uses_reference_price_context():
    def fake_optimize(**_kwargs):
        return {
            "success": True,
            "last_price": 1.16026,
            "last_price_close": 1.16016,
            "last_price_source": "live_tick_ask",
            "best": {"tp": 0.25, "sl": 0.25},
            "results": [],
        }

    out = forecast_use_cases.run_forecast_barrier_optimize(
        ForecastBarrierOptimizeRequest(symbol="EURUSD", method="mc_gbm"),
        parse_kv_or_json=lambda value: value or {},
        barrier_optimize_impl=fake_optimize,
    )

    assert out["reference_price"] == 1.16026
    assert out["reference_price_source"] == "live_tick_ask"
    assert "last_price" not in out
    assert "last_price_close" not in out
    assert "last_price_source" not in out


def test_forecast_tune_optuna_routing(monkeypatch):
    raw_tune = _unwrap(cf.forecast_tune_optuna)
    captured = {}
    ss_calls = {}

    def fake_optuna(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cf, "_optuna_search_impl", fake_optuna)

    import mtdata.forecast.tune as tune_mod

    def fake_default_search_space(method=None, methods=None):
        ss_calls["method"] = method
        ss_calls["methods"] = methods
        return {"theta": {"window": {"min": 1, "max": 3}}}

    monkeypatch.setattr(tune_mod, "default_search_space", fake_default_search_space)
    out = raw_tune(
        request=ForecastTuneOptunaRequest(
            symbol="EURUSD", methods=["theta"], search_space=None
        )
    )
    assert out["ok"] is True
    assert out["detail"] == "compact"
    assert out["compute_intensity"] == "high"
    assert out["compute_cost"] == {
        "unit": "rolling_backtests",
        "estimated": 200,
        "drivers": "n_trials*steps (method sampled once per trial)",
    }
    assert captured["method"] is None
    assert captured["methods"] == ["theta"]
    assert ss_calls["method"] is None
    assert ss_calls["methods"] == ["theta"]
    assert "theta" in captured["search_space"]

    out = raw_tune(
        request=ForecastTuneOptunaRequest(
            symbol="EURUSD",
            methods=["fourier_ols", "naive"],
            search_space={
                "fourier_ols": {
                    "terms": {"type": "int", "min": 1, "max": 3}
                },
                "naive": {},
            },
        )
    )
    assert out["ok"] is True
    assert out["detail"] == "compact"
    assert out["compute_intensity"] == "high"
    assert "compute_cost" in out
    assert captured["method"] is None
    assert out["symbol"] == "EURUSD"
    assert out["quantity"] == "price"

    monkeypatch.setattr(cf, "_optuna_search_impl", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fail")))
    assert "Error in optuna tuning" in raw_tune(request=ForecastTuneOptunaRequest(symbol="EURUSD"))["error"]


def test_forecast_tuning_rejects_unknown_fourier_parameter_names():
    request = ForecastTuneGeneticRequest(
        symbol="EURUSD",
        methods=["fourier_ols"],
        search_space={"m": {"type": "int", "min": 8, "max": 96}},
    )

    out = forecast_use_cases.run_forecast_tune_genetic(
        request,
        genetic_search_impl=lambda **kwargs: pytest.fail("invalid space was executed"),
    )

    assert out["success"] is False
    assert out["error_code"] == "invalid_search_space"
    assert out["invalid_parameters"] == ["m"]


def _ready_options_provider():
    return {
        "configured_provider": "tradier",
        "effective_provider": "tradier",
        "api_key_configured": True,
        "chain_data_ready": True,
        "chain_request_supported": True,
        "action_required": None,
        "remediation": None,
    }


def test_options_and_quantlib_tool_routing(monkeypatch):
    raw_exp = _unwrap(opt.options_expirations)
    raw_chain = _unwrap(opt.options_chain)
    raw_price = _unwrap(opt.options_barrier_price)
    raw_cal = _unwrap(opt.options_heston_calibrate)

    import mtdata.forecast.quantlib_tools as quantlib_tools
    import mtdata.services.options_service as options_service

    monkeypatch.setattr(options_service, "get_options_expirations", lambda **kwargs: {"kind": "exp", **kwargs})
    monkeypatch.setattr(options_service, "get_options_chain", lambda **kwargs: {"kind": "chain", **kwargs})
    monkeypatch.setattr(quantlib_tools, "price_barrier_option_quantlib", lambda **kwargs: {"kind": "price", **kwargs})
    monkeypatch.setattr(quantlib_tools, "calibrate_heston_quantlib_from_options", lambda **kwargs: {"kind": "cal", **kwargs})
    monkeypatch.setattr(opt, "_options_provider_readiness", _ready_options_provider)

    out = raw_exp(symbol="AAPL")
    assert out["kind"] == "exp"
    assert out["symbol"] == "AAPL"

    out = raw_exp(symbol="AAPL.NAS-24")
    assert out["kind"] == "exp"
    assert out["symbol"] == "AAPL.NAS-24"
    assert out["requested_symbol"] == "AAPL.NAS-24"
    assert out["provider_symbol"] == "AAPL"

    out = raw_chain(symbol="AAPL", expiration="2026-06-19", option_type="call", min_open_interest=10, min_volume=5, limit=20)
    assert out["kind"] == "chain"
    assert out["symbol"] == "AAPL"
    assert out["option_type"] == "call"
    assert out["limit"] == 20

    out = raw_chain(symbol="AAPL.NAS", limit=20)
    assert out["symbol"] == "AAPL.NAS"
    assert out["requested_symbol"] == "AAPL.NAS"
    assert out["provider_symbol"] == "AAPL"

    out = raw_price(
        spot=100.0,
        strike=105.0,
        barrier=120.0,
        maturity_days=30,
        option_type="call",
        barrier_type="up_out",
        risk_free_rate=0.03,
        dividend_yield=0.01,
        volatility=0.25,
        rebate=0.0,
        valuation_date="2026-07-03",
        barrier_already_hit=True,
    )
    assert out["kind"] == "price"
    assert out["spot"] == 100.0
    assert out["valuation_date"] == "2026-07-03"
    assert out["barrier_already_hit"] is True

    rejected = raw_price(
        spot=100.0,
        strike=105.0,
        barrier=120.0,
        maturity_days=0,
    )
    assert rejected["error_code"] == "invalid_parameter"
    assert rejected["details"]["required_minimum"] == 1
    assert rejected["valid_values"] == {"maturity_days": "integer >= 1"}

    rejected_vol = raw_price(
        spot=150.0,
        strike=155.0,
        barrier=140.0,
        maturity_days=30,
        volatility=-0.5,
    )
    assert rejected_vol["error_code"] == "invalid_parameter"
    assert rejected_vol["details"]["parameter"] == "volatility"
    assert rejected_vol["details"]["received"] == -0.5
    assert "options_provider_status" not in str(rejected_vol)

    rejected_strike = raw_price(
        spot=150.0,
        strike=0.0,
        barrier=140.0,
        maturity_days=30,
    )
    assert rejected_strike["error_code"] == "invalid_parameter"
    assert rejected_strike["details"]["parameter"] == "strike"
    assert "options_provider_status" not in str(rejected_strike)

    out = raw_cal(
        symbol="AAPL",
        expiration="2026-06-19",
        option_type="put",
        risk_free_rate=0.03,
        dividend_yield=0.01,
        min_open_interest=10,
        min_volume=2,
        max_contracts=15,
    )
    assert out["kind"] == "cal"
    assert out["symbol"] == "AAPL"
    assert out["option_type"] == "put"


def test_options_tools_resolve_spx_for_default_yahoo_provider(monkeypatch):
    raw_exp = _unwrap(opt.options_expirations)
    raw_chain = _unwrap(opt.options_chain)
    raw_cal = _unwrap(opt.options_heston_calibrate)

    import mtdata.forecast.quantlib_tools as quantlib_tools
    import mtdata.services.options_service as options_service

    monkeypatch.setattr(
        opt,
        "_options_provider_readiness",
        lambda: {
            "configured_provider": "yahoo",
            "effective_provider": "yahoo",
            "chain_request_supported": True,
        },
    )
    monkeypatch.setattr(
        options_service,
        "get_options_expirations",
        lambda **kwargs: {"success": True, **kwargs},
    )
    monkeypatch.setattr(
        options_service,
        "get_options_chain",
        lambda **kwargs: {"success": True, **kwargs},
    )
    monkeypatch.setattr(
        quantlib_tools,
        "calibrate_heston_quantlib_from_options",
        lambda **kwargs: {"success": True, **kwargs},
    )

    for result in (
        raw_exp(symbol="SPX"),
        raw_chain(symbol="SPX"),
        raw_cal(symbol="SPX"),
    ):
        assert result["symbol"] == "SPX"
        assert result["requested_symbol"] == "SPX"
        assert result["provider_symbol"] == "^SPX"


def test_options_heston_compact_preserves_rejected_fit_diagnostics(monkeypatch):
    raw_cal = _unwrap(opt.options_heston_calibrate)

    import mtdata.forecast.quantlib_tools as quantlib_tools

    monkeypatch.setattr(opt, "_options_provider_readiness", _ready_options_provider)
    monkeypatch.setattr(
        quantlib_tools,
        "calibrate_heston_quantlib_from_options",
        lambda **_kwargs: {
            "success": False,
            "error": "Heston calibration is not usable for pricing.",
            "error_code": "heston_calibration_rejected",
            "calibration_status": "rejected",
            "usable_for_pricing": False,
            "pricing_usability_failures": ["kappa_near_zero"],
            "params": {"kappa": 0.00001, "theta": 0.04},
        },
    )

    result = raw_cal(symbol="AAPL", detail="compact")

    assert result["success"] is False
    assert result["error_code"] == "heston_calibration_rejected"
    assert result["params"] == {"kappa": 0.00001, "theta": 0.04}
    assert result["pricing_usability_failures"] == ["kappa_near_zero"]


def test_options_tools_validate_and_normalize_symbols(monkeypatch):
    raw_exp = _unwrap(opt.options_expirations)
    raw_chain = _unwrap(opt.options_chain)
    raw_cal = _unwrap(opt.options_heston_calibrate)

    import mtdata.services.options_service as options_service

    monkeypatch.setattr(opt, "_options_provider_readiness", _ready_options_provider)
    monkeypatch.setattr(
        options_service,
        "get_options_expirations",
        lambda **kwargs: {"success": True, **kwargs},
    )

    for tool in (raw_exp, raw_chain, raw_cal):
        out = tool(symbol="  ")
        assert out["success"] is False
        assert out["error_code"] == "invalid_symbol"
        assert out["error"] == "symbol is required"

        out = tool(symbol="AAPL?query=1")
        assert out["success"] is False
        assert out["error_code"] == "invalid_symbol"

    out = raw_exp(symbol=" brk.b ")
    assert out["success"] is True
    assert out["symbol"] == "BRK.B"
    assert out.get("provider_symbol") == "BRK-B"

    out = raw_exp(symbol="AAPL.NAS")
    assert out["success"] is True
    assert out["symbol"] == "AAPL.NAS"
    assert out.get("provider_symbol") == "AAPL"


def test_options_tools_reject_venue_qualified_non_us_symbols(monkeypatch):
    raw_exp = _unwrap(opt.options_expirations)
    raw_chain = _unwrap(opt.options_chain)
    raw_cal = _unwrap(opt.options_heston_calibrate)

    import mtdata.forecast.quantlib_tools as quantlib_tools
    import mtdata.services.options_service as options_service

    def fail_call(**_kwargs):
        raise AssertionError("venue-qualified symbols must not hit the provider")

    monkeypatch.setattr(options_service, "get_options_expirations", fail_call)
    monkeypatch.setattr(options_service, "get_options_chain", fail_call)
    monkeypatch.setattr(
        quantlib_tools,
        "calibrate_heston_quantlib_from_options",
        fail_call,
    )
    monkeypatch.setattr(opt, "_options_provider_readiness", _ready_options_provider)

    for tool, symbol in (
        (raw_exp, "VOD.L"),
        (raw_chain, "SHOP.TO"),
        (raw_cal, "VOD.L"),
    ):
        out = tool(symbol=symbol)
        assert out["success"] is False
        assert out["error_code"] == "options_unsupported_symbol"
        assert "venue-qualified" in out["error"]
        assert out["symbol"] == symbol


def test_options_tools_reject_fx_symbols_before_provider_calls(monkeypatch):
    raw_exp = _unwrap(opt.options_expirations)
    raw_chain = _unwrap(opt.options_chain)
    raw_cal = _unwrap(opt.options_heston_calibrate)

    import mtdata.forecast.quantlib_tools as quantlib_tools
    import mtdata.services.options_service as options_service

    def fail_call(**_kwargs):
        raise AssertionError("options provider should not be queried for FX")

    monkeypatch.setattr(options_service, "get_options_expirations", fail_call)
    monkeypatch.setattr(options_service, "get_options_chain", fail_call)
    monkeypatch.setattr(
        quantlib_tools,
        "calibrate_heston_quantlib_from_options",
        fail_call,
    )
    monkeypatch.setattr(opt, "_options_provider_readiness", _ready_options_provider)

    for tool in (raw_exp, raw_chain, raw_cal):
        out = tool(symbol="EURUSD")
        assert out["success"] is False
        assert out["error_code"] == "options_unsupported_symbol"
        assert "US-listed" in out["error"]
        assert "options_provider_status" in out["related_tools"]


def test_options_tools_validate_expiration_before_provider_calls(monkeypatch):
    raw_chain = _unwrap(opt.options_chain)
    raw_cal = _unwrap(opt.options_heston_calibrate)

    import mtdata.forecast.quantlib_tools as quantlib_tools
    import mtdata.services.options_service as options_service

    def fail_call(**kwargs):
        raise AssertionError("options provider should not be queried")

    monkeypatch.setattr(options_service, "get_options_chain", fail_call)
    monkeypatch.setattr(
        quantlib_tools,
        "calibrate_heston_quantlib_from_options",
        fail_call,
    )
    monkeypatch.setattr(opt, "_options_provider_readiness", _ready_options_provider)

    for expiration in ("GTC", "2026/07/17", "2026-02-30"):
        for tool in (raw_chain, raw_cal):
            out = tool(symbol="AAPL", expiration=expiration)
            assert out["success"] is False
            assert out["error_code"] == "invalid_expiration"
            assert out["parameter"] == "expiration"
            assert out["value"] == expiration
            assert out["expected_format"] == "YYYY-MM-DD"


def test_options_chain_preserves_unlisted_expiration_contract(monkeypatch):
    raw_chain = _unwrap(opt.options_chain)

    import mtdata.services.options_service as options_service

    monkeypatch.setattr(opt, "_options_provider_readiness", _ready_options_provider)
    monkeypatch.setattr(
        options_service,
        "get_options_chain",
        lambda **_kwargs: {
            "success": False,
            "error": "Requested expiration is not listed.",
            "error_code": "options_expiration_not_listed",
            "expiration": "2000-01-21",
            "expiration_status": "expired",
            "expirations": ["2099-01-16"],
            "remediation": "Choose a listed expiration.",
        },
    )

    out = raw_chain(symbol="AAPL", expiration="2000-01-21")

    assert out["success"] is False
    assert out["error_code"] == "options_expiration_not_listed"
    assert out["expiration_status"] == "expired"
    assert out["expirations"] == ["2099-01-16"]


@pytest.mark.parametrize(
    ("tool_name", "kwargs", "parameter", "error_code"),
    [
        ("chain", {"min_open_interest": -1}, "min_open_interest", "invalid_input"),
        ("chain", {"min_volume": -1}, "min_volume", "invalid_input"),
        ("chain", {"limit": 0}, "limit", "invalid_input"),
        ("calibrate", {"min_open_interest": -1}, "min_open_interest", "invalid_input"),
        ("calibrate", {"min_volume": -1}, "min_volume", "invalid_input"),
        ("calibrate", {"max_contracts": 4}, "max_contracts", "invalid_input"),
        (
            "calibrate",
            {"valuation_date": "2026/07/17"},
            "valuation_date",
            "invalid_valuation_date",
        ),
    ],
)
def test_options_tools_validate_controls_before_provider_gate(
    monkeypatch,
    tool_name,
    kwargs,
    parameter,
    error_code,
):
    raw = _unwrap(
        opt.options_chain
        if tool_name == "chain"
        else opt.options_heston_calibrate
    )

    def fail_readiness():
        raise AssertionError("provider readiness should not be checked")

    monkeypatch.setattr(opt, "_options_provider_readiness", fail_readiness)

    result = raw(symbol="AAPL", **kwargs)

    assert result["success"] is False
    assert result["error_code"] == error_code
    assert result["parameter"] == parameter


def test_options_barrier_price_rejects_relative_valuation_date():
    raw_price = _unwrap(opt.options_barrier_price)

    result = raw_price(
        spot=100.0,
        strike=105.0,
        barrier=90.0,
        maturity_days=30,
        valuation_date="yesterday",
    )

    assert result["success"] is False
    assert result["error_code"] == "invalid_valuation_date"
    assert result["parameter"] == "valuation_date"


def test_options_chain_tools_short_circuit_when_provider_not_ready(monkeypatch):
    raw_exp = _unwrap(opt.options_expirations)
    raw_chain = _unwrap(opt.options_chain)
    raw_cal = _unwrap(opt.options_heston_calibrate)

    import mtdata.forecast.quantlib_tools as quantlib_tools
    import mtdata.services.options_service as options_service

    def fail_call(**kwargs):
        raise AssertionError("options provider should not be queried")

    monkeypatch.setattr(options_service, "get_options_expirations", fail_call)
    monkeypatch.setattr(options_service, "get_options_chain", fail_call)
    monkeypatch.setattr(quantlib_tools, "calibrate_heston_quantlib_from_options", fail_call)
    monkeypatch.setattr(
        opt,
        "_options_provider_readiness",
        lambda: {
            "configured_provider": "tradier",
            "effective_provider": "tradier",
            "api_key_configured": False,
            "chain_data_ready": False,
            "action_required": "configure_options_provider",
            "remediation": "configure Tradier",
        },
    )

    for result in (
        raw_exp(symbol="AAPL"),
        raw_chain(symbol="AAPL"),
        raw_cal(symbol="AAPL"),
    ):
        assert result["success"] is False
        assert result["error_code"] == "options_provider_auth"
        assert result["provider"] == "tradier"
        assert result["next_tool"] == "options_provider_status"
        assert result["chain_data_ready"] is False
        assert result["action_required"] == "configure_options_provider"


def test_options_chain_tools_allow_yahoo_best_effort_provider(monkeypatch):
    raw_exp = _unwrap(opt.options_expirations)

    import mtdata.services.options_service as options_service

    monkeypatch.setattr(
        options_service,
        "get_options_expirations",
        lambda **kwargs: {"success": True, "provider": "yahoo", **kwargs},
    )
    monkeypatch.setattr(
        opt,
        "_options_provider_readiness",
        lambda: {
            "configured_provider": "yahoo",
            "effective_provider": "yahoo",
            "api_key_configured": False,
            "chain_data_ready": False,
            "chain_request_supported": True,
            "provider_mode": "best_effort",
            "action_required": None,
        },
    )

    result = raw_exp(symbol="AAPL")

    assert result["success"] is True
    assert result["provider"] == "yahoo"


def test_options_chain_logs_finish_event(caplog, monkeypatch):
    raw_chain = _unwrap(opt.options_chain)

    import mtdata.services.options_service as options_service

    monkeypatch.setattr(options_service, "get_options_chain", lambda **kwargs: {"success": True, **kwargs})
    monkeypatch.setattr(opt, "_options_provider_readiness", _ready_options_provider)

    with caplog.at_level(logging.DEBUG, logger=opt.logger.name):
        out = raw_chain(symbol="AAPL", expiration="2026-06-19", option_type="call", limit=25)

    assert out["success"] is True
    assert any(
        "event=finish operation=options_chain success=True" in record.message
        for record in caplog.records
    )


def test_options_barrier_compact_keeps_numeric_pricing_inputs():
    payload = {
        "success": True,
        "option_status": "knocked_out",
        "status": "expired",
        "barrier_already_hit": True,
        "barrier_state_source": "explicit_prior_hit",
        "price": 1.23,
        "delta": 0.4,
        "params_used": {
            "risk_free_rate": 0.05,
            "dividend_yield": 0.01,
            "volatility": 0.2,
            "rebate": 0.0,
        },
    }

    result = opt._apply_options_detail(
        payload,
        detail="compact",
        kind="barrier_price",
    )

    assert result["pricing_inputs"] == {
        "risk_free_rate": 0.05,
        "dividend_yield": 0.01,
        "volatility": 0.2,
        "rebate": 0.0,
        "rate_unit": "decimal_fraction",
        "volatility_unit": "decimal_fraction",
    }
    assert result["option_status"] == "knocked_out"
    assert result["status"] == "expired"
    assert result["barrier_already_hit"] is True
    assert result["barrier_state_source"] == "explicit_prior_hit"


def test_options_tools_support_compact_and_full_detail(monkeypatch):
    raw_exp = _unwrap(opt.options_expirations)
    raw_chain = _unwrap(opt.options_chain)
    raw_price = _unwrap(opt.options_barrier_price)
    raw_cal = _unwrap(opt.options_heston_calibrate)

    import mtdata.forecast.quantlib_tools as quantlib_tools
    import mtdata.services.options_service as options_service

    monkeypatch.setattr(opt, "_options_provider_readiness", _ready_options_provider)
    monkeypatch.setattr(
        options_service,
        "get_options_expirations",
        lambda **kwargs: {
            "success": True,
            "symbol": kwargs["symbol"],
            "underlying_price": 100.0,
            "currency": "USD",
            "expirations": ["2026-06-19"],
            "expiration_count": 1,
        },
    )
    monkeypatch.setattr(
        options_service,
        "get_options_chain",
        lambda **kwargs: {
            "success": True,
            "symbol": kwargs["symbol"],
            "expiration": "2026-06-19",
            "underlying_price": 100.0,
            "underlying_as_of": "2026-06-01T20:00:00Z",
            "underlying_data_stale": False,
            "underlying_freshness": "provider_timestamped",
            "option_chain_freshness": "current",
            "option_chain_quality": "live_usable",
            "option_chain_live_usable": True,
            "option_contract_count": 1,
            "option_contract_timestamped_count": 1,
            "option_contract_current_count": 1,
            "option_contract_stale_count": 0,
            "option_contract_quote_usable_count": 1,
            "currency": "USD",
            "option_type": kwargs["option_type"],
            "count": 1,
            "calls_count": 1,
            "puts_count": 0,
            "available_count": 25,
            "available_count_basis": "after_side_and_liquidity_filters",
            "available_calls_count": 13,
            "available_puts_count": 12,
            "pagination": {
                "total": 25,
                "returned": 1,
                "offset": kwargs["offset"],
                "limit": kwargs["limit"],
                "has_more": True,
                "more_available": 24,
            },
            "selection_order": "nearest_strike_to_underlying_balanced_by_side",
            "contract_terms_summary": {
                "provider_classifications": ["REGULAR"],
                "multiplier_statuses": [
                    "standard_from_provider_classification"
                ],
                "uniform_contract_multiplier": 100,
                "uniform_settlement_type": "physical",
                "uniform_terms": {
                    "contract_size": "REGULAR",
                    "contract_multiplier": 100,
                    "multiplier_status": (
                        "standard_from_provider_classification"
                    ),
                    "settlement_type": "physical",
                    "asset_class": "equity_option",
                    "exercise_style": "american",
                    "deliverable": "100 underlying units",
                    "deliverable_status": "standard",
                    "premium_quote_unit": "currency_per_underlying_unit",
                },
                "mixed_fields": [],
                "unresolved_fields": [],
                "mixed_or_unresolved_terms": False,
            },
            "contract_premium_formula": (
                "cash premium = quoted bid/ask/last * contract_multiplier"
            ),
            "units": {
                "option_premium": "currency_per_underlying_unit",
                "contract_multiplier": "underlying_units_per_contract",
            },
            "expirations": ["2026-06-19"],
            "options": [
                {
                    "side": "call",
                    "contract": "AAPL260619C00100000",
                    "strike": 100.0,
                    "last": 2.0,
                    "bid": 1.9,
                    "ask": 2.1,
                    "implied_volatility": 0.2,
                    "in_the_money": True,
                    "last_trade_epoch": 1700000000,
                    "contract_as_of": "2026-06-01T20:00:00Z",
                    "contract_data_age_seconds": 30.0,
                    "contract_data_stale": False,
                    "contract_freshness": "provider_timestamped",
                    "quote_quality": "two_sided",
                    "quote_usable_for_live_analysis": True,
                    "quote_usability_reason": "two_sided_current_quote",
                    "volume": 10,
                    "open_interest": 20,
                    "contract_size": "REGULAR",
                    "contract_multiplier": 100,
                    "multiplier_status": (
                        "standard_from_provider_classification"
                    ),
                    "settlement_type": "physical",
                    "asset_class": "equity_option",
                    "exercise_style": "american",
                    "deliverable": "100 underlying units",
                    "deliverable_status": "standard",
                    "premium_quote_unit": "currency_per_underlying_unit",
                }
            ],
        },
    )
    monkeypatch.setattr(
        quantlib_tools,
        "price_barrier_option_quantlib",
        lambda **kwargs: {
            "success": True,
            "price": 1.23,
            "delta": 0.4,
            "gamma": 0.01,
            "vega": 0.2,
            "valuation_date": kwargs.get("valuation_date") or "2026-07-03",
            "maturity_date": "2026-08-02",
            "time_to_maturity_years": 30 / 365,
            "params_used": {
                "spot": kwargs["spot"],
                "strike": kwargs["strike"],
                "barrier": kwargs["barrier"],
                "option_type": kwargs["option_type"],
                "barrier_type": kwargs["barrier_type"],
                "maturity_days": kwargs["maturity_days"],
                "risk_free_rate": kwargs["risk_free_rate"],
                "dividend_yield": kwargs["dividend_yield"],
                "volatility": kwargs["volatility"],
                "rebate": kwargs["rebate"],
            },
        },
    )
    monkeypatch.setattr(
        quantlib_tools,
        "calibrate_heston_quantlib_from_options",
        lambda **kwargs: {
            "success": True,
            "symbol": kwargs["symbol"],
            "expiration": "2026-06-19",
            "days_to_expiry": 30,
            "contracts_used": 5,
            "spot": 100.0,
            "spot_as_of": "2026-06-01T20:00:00Z",
            "spot_data_age_seconds": 30.0,
            "spot_data_stale": True,
            "spot_freshness": "stale",
            "spot_freshness_reason": "provider_quote_age_exceeds_live_threshold",
            "spot_source": "tradier_last",
            "spot_session": "provider_reported_last",
            "calibration_data_status": "stale",
            "warnings": [
                "Heston calibration used stale options-provider market data."
            ],
            "calibration_error_rmse": 0.01,
            "params": {"kappa": 1.0},
            "sample_contracts": [{"strike": 100.0, "iv": 0.2}],
        },
    )

    assert "underlying_price" not in raw_exp("AAPL", detail="compact")
    assert raw_exp("AAPL", detail="full")["underlying_price"] == 100.0

    compact_chain = raw_chain("AAPL", detail="compact")
    assert compact_chain["detail"] == "compact"
    assert compact_chain["available_count"] == 25
    assert compact_chain["pagination"] == {
        "total": 25,
        "returned": 1,
        "offset": 0,
        "limit": 20,
        "has_more": True,
        "more_available": 24,
    }
    assert "has_more" not in compact_chain
    assert "limit" not in compact_chain
    assert "contract_size" not in compact_chain
    assert compact_chain["contract_terms_summary"][
        "uniform_contract_multiplier"
    ] == 100
    assert compact_chain["contract_terms_summary"][
        "mixed_or_unresolved_terms"
    ] is False
    compact_option = compact_chain["options"][0]
    assert not (
        set(compact_option) & set(opt._OPTIONS_CHAIN_UNIFORM_TERM_FIELDS)
    )
    full_option = raw_chain("AAPL", detail="full")["options"][0]
    assert full_option["contract_size"] == "REGULAR"
    assert full_option["contract_multiplier"] == 100
    assert full_option["premium_quote_unit"] == (
        "currency_per_underlying_unit"
    )
    assert compact_chain["underlying_as_of"] == "2026-06-01T20:00:00Z"
    assert compact_chain["option_chain_quality"] == "live_usable"
    assert compact_chain["option_contract_quote_usable_count"] == 1
    assert compact_chain["options"][0]["contract_as_of"] == (
        "2026-06-01T20:00:00Z"
    )
    assert compact_chain["options"][0]["quote_usable_for_live_analysis"] is True
    assert compact_chain["options"][0]["contract_data_stale"] is False
    assert compact_chain["options"][0]["implied_volatility"] == 0.2
    assert compact_chain["options"][0]["in_the_money"] is True
    assert raw_chain("AAPL", detail="full")["options"][0]["implied_volatility"] == 0.2

    compact_price = raw_price(
        100,
        105,
        120,
        30,
        valuation_date="2026-07-03",
        detail="compact",
    )
    assert compact_price["price"] == 1.23
    assert compact_price["delta"] == 0.4
    assert compact_price["gamma"] == 0.01
    assert compact_price["vega"] == 0.2
    assert compact_price["detail"] == "compact"
    assert compact_price["valuation_date"] == "2026-07-03"
    assert compact_price["maturity_date"] == "2026-08-02"
    assert compact_price["time_to_maturity_years"] == 30 / 365
    assert compact_price["units"] == {
        "price": "premium_per_underlying_unit",
        "delta": "premium_change_per_underlying_price_unit",
        "gamma": "premium_change_per_squared_underlying_price_unit",
        "vega": "premium_change_per_1.0_decimal_volatility",
    }
    assert compact_price["pricing_inputs"] == {
        "risk_free_rate": 0.02,
        "dividend_yield": 0.0,
        "volatility": 0.2,
        "rebate": 0.0,
        "rate_unit": "decimal_fraction",
        "volatility_unit": "decimal_fraction",
    }
    full_price = raw_price(100, 105, 120, 30, detail="full")
    assert full_price["delta"] == 0.4
    assert full_price["units"]["price"] == "premium_per_underlying_unit"
    assert full_price["units"]["gamma"] == (
        "premium_change_per_squared_underlying_price_unit"
    )
    assert full_price["units"]["vega"] == (
        "premium_change_per_1.0_decimal_volatility"
    )
    assert full_price["params_used"]["spot"] == 100

    compact_cal = raw_cal("AAPL", detail="compact")
    assert compact_cal["params"] == {"kappa": 1.0}
    assert compact_cal["spot_as_of"] == "2026-06-01T20:00:00Z"
    assert compact_cal["spot_data_stale"] is True
    assert compact_cal["spot_freshness_reason"] == (
        "provider_quote_age_exceeds_live_threshold"
    )
    assert compact_cal["spot_source"] == "tradier_last"
    assert compact_cal["calibration_data_status"] == "stale"
    assert compact_cal["warnings"] == [
        "Heston calibration used stale options-provider market data."
    ]
    assert "sample_contracts" not in compact_cal
    assert raw_cal("AAPL", detail="full")["sample_contracts"] == [
        {"strike": 100.0, "iv": 0.2}
    ]


def test_options_chain_compact_keeps_terms_when_contracts_are_mixed() -> None:
    payload = {
        "success": True,
        "contract_terms_summary": {
            "uniform_terms": {},
            "mixed_fields": ["contract_size", "contract_multiplier"],
            "unresolved_fields": [],
            "mixed_or_unresolved_terms": True,
        },
        "options": [
            {
                "side": "call",
                "contract": "AAPL-REGULAR",
                "strike": 100.0,
                "contract_size": "REGULAR",
                "contract_multiplier": 100,
            },
            {
                "side": "call",
                "contract": "AAPL-MINI",
                "strike": 100.0,
                "contract_size": "MINI",
                "contract_multiplier": 10,
            },
        ],
    }

    result = opt._apply_options_detail(payload, detail="compact", kind="chain")

    assert [row["contract_size"] for row in result["options"]] == [
        "REGULAR",
        "MINI",
    ]
    assert [row["contract_multiplier"] for row in result["options"]] == [100, 10]


def test_options_chain_uses_detail_aware_default_limits(monkeypatch):
    raw_chain = _unwrap(opt.options_chain)
    captured = []

    import mtdata.services.options_service as options_service

    def fake_chain(**kwargs):
        captured.append((kwargs["limit"], kwargs["offset"]))
        return {
            "success": True,
            "options": [],
            "count": 0,
            "pagination": {
                "total": 0,
                "returned": 0,
                "offset": kwargs["offset"],
                "limit": kwargs["limit"],
                "has_more": False,
                "more_available": 0,
            },
        }

    monkeypatch.setattr(options_service, "get_options_chain", fake_chain)
    monkeypatch.setattr(opt, "_options_provider_readiness", _ready_options_provider)

    compact = raw_chain(symbol="AAPL")
    full = raw_chain(symbol="AAPL", detail="full")

    page = raw_chain(symbol="AAPL", limit=5, offset=10)

    assert captured == [(20, 0), (200, 0), (5, 10)]
    assert compact["pagination"]["limit"] == 20
    assert full["pagination"]["limit"] == 200
    assert page["pagination"]["offset"] == 10


def test_forecast_barrier_optimize_routes_profile_args(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)
    called = {}

    import mtdata.forecast.barriers_optimization as barriers_mod

    def fake_optimize(**kwargs):
        called.update(kwargs)
        return {
            "ok": True,
            "search_profile": kwargs.get("search_profile"),
            "fast_defaults_param": kwargs.get("params", {}).get("fast_defaults"),
        }

    monkeypatch.setattr(barriers_mod, "forecast_barrier_optimize", fake_optimize)
    out = raw_opt(
        request=ForecastBarrierOptimizeRequest(
            symbol="EURUSD",
            search_profile="long",
            params={"fast_defaults": True},
        )
    )
    assert out["ok"] is True
    assert out["fast_defaults_param"] is True
    assert out["search_profile"] == "long"
    assert called["params"]["fast_defaults"] is True
    assert called["fast_defaults"] is False
    assert called["search_profile"] == "long"


def test_forecast_barrier_optimize_routes_statistical_robustness_args(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)
    called = {}

    import mtdata.forecast.barriers_optimization as barriers_mod

    def fake_optimize(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(barriers_mod, "forecast_barrier_optimize", fake_optimize)
    out = raw_opt(
        request=ForecastBarrierOptimizeRequest(
            symbol="EURUSD",
            params={
                "statistical_robustness": True,
                "target_ci_width": 0.02,
                "n_seeds_stability": 4,
                "enable_bootstrap": True,
                "n_bootstrap": 250,
                "enable_convergence_check": False,
                "convergence_window": 80,
                "convergence_threshold": 0.005,
                "enable_power_analysis": True,
                "power_effect_size": 0.02,
                "enable_sensitivity_analysis": True,
                "sensitivity_params": ["tp", "sl"],
            },
        )
    )
    assert out["ok"] is True
    assert called["statistical_robustness"] is False
    assert called["params"]["statistical_robustness"] is True
    assert called["params"]["target_ci_width"] == 0.02
    assert called["params"]["n_seeds_stability"] == 4
    assert called["params"]["enable_bootstrap"] is True
    assert called["params"]["n_bootstrap"] == 250
    assert called["params"]["enable_convergence_check"] is False
    assert called["params"]["convergence_window"] == 80
    assert called["params"]["convergence_threshold"] == 0.005
    assert called["params"]["enable_power_analysis"] is True
    assert called["params"]["power_effect_size"] == 0.02
    assert called["params"]["enable_sensitivity_analysis"] is True
    assert called["params"]["sensitivity_params"] == ["tp", "sl"]


def test_forecast_barrier_optimize_routes_advanced_grid_params(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)
    called = {}

    import mtdata.forecast.barriers_optimization as barriers_mod

    def fake_optimize(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(barriers_mod, "forecast_barrier_optimize", fake_optimize)
    out = raw_opt(
        request=ForecastBarrierOptimizeRequest(
            symbol="EURUSD",
            params={"vol_sl_multiplier": 2.1},
        )
    )

    assert out["ok"] is True
    assert called["params"]["vol_sl_multiplier"] == 2.1
    assert called["vol_sl_multiplier"] == 1.8


def test_forecast_barrier_optimize_keeps_grid_default_path(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)
    called = {}

    import mtdata.forecast.barriers_optimization as barriers_mod

    def fake_optimize(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(barriers_mod, "forecast_barrier_optimize", fake_optimize)
    out = raw_opt(request=ForecastBarrierOptimizeRequest(symbol="BTCUSD"))
    assert out["ok"] is True
    assert out["detail"] == "compact"
    assert called["method"] == "mc_gbm_bb"
    assert called["search_profile"] == "medium"
    assert called["output_mode"] == "summary"
    assert "format" not in called
    assert called["concise"] is True
    assert called["return_grid"] is False
    assert "seed" not in called["params"]
    assert "optimizer" not in called["params"]
    assert "sampler" not in called["params"]
    assert "pruner" not in called["params"]
    assert "n_jobs" not in called["params"]


def test_forecast_barrier_optimize_standard_disables_concise_only(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)
    called = {}

    import mtdata.forecast.barriers_optimization as barriers_mod

    def fake_optimize(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(barriers_mod, "forecast_barrier_optimize", fake_optimize)
    out = raw_opt(request=ForecastBarrierOptimizeRequest(symbol="BTCUSD", detail="standard"))

    assert out["detail"] == "standard"
    assert called["output_mode"] == "summary"
    assert "format" not in called
    assert called["concise"] is False


def test_forecast_barrier_optimize_applies_optuna_defaults_when_requested(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)
    called = {}

    import mtdata.forecast.barriers_optimization as barriers_mod

    def fake_optimize(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(barriers_mod, "forecast_barrier_optimize", fake_optimize)
    out = raw_opt(request=ForecastBarrierOptimizeRequest(symbol="BTCUSD", params={"optimizer": "optuna"}))
    assert out["ok"] is True
    assert called["params"]["optimizer"] == "optuna"
    assert called["params"]["sampler"] == "tpe"
    assert called["params"]["pruner"] == "median"
    assert int(called["params"]["n_jobs"]) >= 1


def test_forecast_barrier_optimize_preserves_explicit_seed(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)
    called = {}

    import mtdata.forecast.barriers_optimization as barriers_mod

    def fake_optimize(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(barriers_mod, "forecast_barrier_optimize", fake_optimize)
    out = raw_opt(request=ForecastBarrierOptimizeRequest(symbol="BTCUSD", params={"seed": 17}))
    assert out["ok"] is True
    assert called["params"]["seed"] == 17


def test_forecast_barrier_optimize_returns_connection_error_payload(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)

    def fail_connection():
        raise MT5ConnectionError("Failed to connect to MetaTrader5. Ensure MT5 terminal is running.")

    monkeypatch.setattr(cf, "ensure_mt5_connection_or_raise", fail_connection)

    out = raw_opt(request=ForecastBarrierOptimizeRequest(symbol="EURUSD"))

    assert out["success"] is False
    assert out["error"] == "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
    assert out["error_code"] == "mt5_connection_error"
    assert out["operation"] == "mt5_ensure_connection"
    assert isinstance(out.get("request_id"), str)
