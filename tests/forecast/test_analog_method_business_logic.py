from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast import forecast_engine as forecast_engine_module
from mtdata.forecast.interface import ForecastCallContext
from mtdata.forecast.methods.analog import AnalogMethod
from mtdata.shared import constants as shared_constants


def test_analog_method_metadata_properties():
    method = AnalogMethod()
    assert method.name == "analog"
    assert method.category == "analog"
    assert "scipy" in method.required_packages
    assert method.supports_features["price"] is True
    assert method.supports_features["return"] is False


def test_analog_method_prepare_forecast_call_injects_history_context():
    method = AnalogMethod()
    history = pd.DataFrame({"time": [1.0, 2.0], "close": [100.0, 101.0]})
    context = ForecastCallContext(
        method="analog",
        symbol="EURUSD",
        timeframe="H1",
        quantity="price",
        horizon=2,
        seasonality=24,
        base_col="close_dn",
        ci_alpha=0.1,
        as_of="2024-01-01",
        denoise_spec_used={"method": "ema"},
        history_df=history,
        target_series=pd.Series([100.0, 101.0], name="close_dn"),
        exog_used=None,
        future_exog=None,
    )

    params, kwargs = method.prepare_forecast_call({"window_size": 32}, {"timeframe": "H1"}, context)

    assert params["symbol"] == "EURUSD"
    assert params["timeframe"] == "H1"
    assert params["base_col"] == "close_dn"
    assert params["as_of"] == "2024-01-01"
    assert params["denoise"] == {"method": "ema"}
    assert kwargs["history_base_col"] == "close_dn"
    assert kwargs["history_denoise_spec"] == {"method": "ema"}
    assert isinstance(kwargs["history_df"], pd.DataFrame)
    assert kwargs["history_df"] is not history


def test_analog_method_rejects_derived_or_missing_series():
    method = AnalogMethod()
    with pytest.raises(ValueError, match="price series only"):
        method.forecast(pd.Series([], dtype=float), horizon=3, seasonality=1, params={"symbol": "EURUSD", "timeframe": "H1"})

    with pytest.raises(ValueError, match="price series only"):
        method.forecast(pd.Series([0.1, 0.2], name="__log_return"), horizon=3, seasonality=1, params={"symbol": "EURUSD", "timeframe": "H1"})


def test_analog_method_requires_symbol_and_timeframe():
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 105.0, 10), name="close")

    with pytest.raises(ValueError, match="requires 'symbol' and 'timeframe'"):
        method.forecast(series, horizon=3, seasonality=1, params={})


def test_analog_method_raises_when_primary_search_fails(monkeypatch):
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 105.0, 80), name="close")

    def fake_run(*args, **kwargs):
        method._timeframe_diagnostics["H1"] = {"reason": "search_failed", "message": "search backend unavailable"}
        return ([], [])

    monkeypatch.setattr(method, "_run_single_timeframe", fake_run)

    with pytest.raises(RuntimeError, match="Primary analog search failed.*search failed.*search backend unavailable"):
        method.forecast(series, horizon=3, seasonality=1, params={"symbol": "EURUSD", "timeframe": "H1"})


def test_analog_method_aggregates_primary_and_secondary_paths(monkeypatch):
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 108.0, 80), name="close")

    def fake_run(symbol, timeframe, horizon, params, query_vector=None, **kwargs):
        if timeframe == "H1":
            return (
                [
                    np.array([101.0, 102.0, 103.0], dtype=float),
                    np.array([100.0, 101.0, 102.0], dtype=float),
                ],
                [{"score": 0.1}, {"score": 0.2}],
            )
        # Secondary horizon for M30 becomes >= 6 with primary horizon=3
        return (
            [np.array([99.0, 99.5, 100.0, 100.5, 101.0, 101.5], dtype=float)],
            [{"score": 0.3}],
        )

    monkeypatch.setattr(method, "_run_single_timeframe", fake_run)

    out = method.forecast(
        series,
        horizon=3,
        seasonality=1,
        params={
            "symbol": "EURUSD",
            "timeframe": "H1",
            "secondary_timeframes": "M30",
            "ci_alpha": 2.0,  # invalid should normalize to 0.05
        },
    )

    assert out.forecast.shape[0] == 3
    assert out.ci_values is None
    assert out.metadata["ci_sample_support"]["status"] == "unavailable"
    assert out.params_used is not None
    assert out.params_used["ci_alpha"] == 0.05
    assert out.params_used["n_paths"] == 3
    assert out.metadata is not None
    assert out.metadata["components"] == ["H1", "M30"]
    assert out.metadata["requested_components"] == ["H1", "M30"]
    assert len(out.metadata["analogs"]) == 2
    assert out.metadata["component_status"][0]["status"] == "contributed"
    assert out.metadata["component_status"][1]["status"] == "contributed"


def test_analog_method_rejects_non_close_base_column():
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 105.0, 10), name="open")

    with pytest.raises(ValueError, match="close-based price series"):
        method.forecast(
            series,
            horizon=3,
            seasonality=1,
            params={"symbol": "EURUSD", "timeframe": "H1", "window_size": 5},
        )


def test_analog_method_requires_denoise_spec_for_close_dn_series():
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 105.0, 10), name="close_dn")

    with pytest.raises(ValueError, match="requires a denoise spec"):
        method.forecast(
            series,
            horizon=3,
            seasonality=1,
            params={"symbol": "EURUSD", "timeframe": "H1", "window_size": 5, "base_col": "close_dn"},
        )


def test_analog_method_rejects_conflicting_denoise_between_params_and_history_context():
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 108.0, 80), name="close_dn")
    history_df = pd.DataFrame(
        {
            "time": np.arange(80, dtype=float),
            "close_dn": np.linspace(100.0, 108.0, 80, dtype=float),
        }
    )

    with pytest.raises(ValueError, match="conflicting denoise specs"):
        method.forecast(
            series,
            horizon=3,
            seasonality=1,
            params={
                "symbol": "EURUSD",
                "timeframe": "H1",
                "base_col": "close_dn",
                "denoise": {"method": "ema", "params": {"span": 5}},
            },
            history_df=history_df,
            history_base_col="close_dn",
            history_denoise_spec={"method": "sma", "params": {"window": 3}},
        )


def test_analog_method_reports_only_contributing_components(monkeypatch):
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 108.0, 80), name="close")

    def fake_run(symbol, timeframe, horizon, params, query_vector=None, **kwargs):
        if timeframe == "H1":
            method._timeframe_diagnostics["H1"] = {"status": "ok", "returned_paths": 1}
            return ([np.array([101.0, 102.0, 103.0], dtype=float)], [{"score": 0.1}])
        method._timeframe_diagnostics[timeframe] = {"status": "ok", "returned_paths": 1}
        return ([np.array([99.0, 100.0, 101.0], dtype=float)], [{"score": 0.3}])

    monkeypatch.setattr(method, "_run_single_timeframe", fake_run)

    out = method.forecast(
        series,
        horizon=3,
        seasonality=1,
        params={
            "symbol": "EURUSD",
            "timeframe": "H1",
            "secondary_timeframes": "H4",
        },
    )

    assert out.metadata is not None
    assert out.metadata["components"] == ["H1"]
    assert out.metadata["requested_components"] == ["H1", "H4"]
    secondary_status = next(item for item in out.metadata["component_status"] if item["timeframe"] == "H4")
    assert secondary_status["status"] == "skipped_insufficient_coverage"


def test_analog_method_reports_search_symbol_universe(monkeypatch):
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 108.0, 80), name="close")

    def fake_run(symbol, timeframe, horizon, params, query_vector=None, **kwargs):
        method._timeframe_diagnostics[timeframe] = {
            "status": "ok",
            "returned_paths": 1,
            "search_symbols_used": ["EURUSD", "GBPUSD"],
        }
        return ([np.array([101.0, 102.0, 103.0], dtype=float)], [{"score": 0.1, "symbol": "GBPUSD"}])

    monkeypatch.setattr(method, "_run_single_timeframe", fake_run)

    out = method.forecast(
        series,
        horizon=3,
        seasonality=1,
        params={
            "symbol": "EURUSD",
            "timeframe": "H1",
            "search_symbols": ["GBPUSD", "USDJPY"],
        },
    )

    assert out.params_used["search_symbols"] == ["EURUSD", "GBPUSD", "USDJPY"]
    assert out.metadata["search_symbols"] == ["EURUSD", "GBPUSD", "USDJPY"]
    assert out.metadata["analogs"][0]["meta"]["symbol"] == "GBPUSD"


def test_analog_method_rejects_primary_scores_above_threshold(monkeypatch):
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 108.0, 80), name="close")

    def fake_run(symbol, timeframe, horizon, params, query_vector=None, **kwargs):
        method._timeframe_diagnostics[timeframe] = {"status": "ok", "returned_paths": 2}
        return (
            [
                np.array([101.0, 102.0, 103.0], dtype=float),
                np.array([100.5, 101.5, 102.5], dtype=float),
            ],
            [{"score": 0.35}, {"score": 0.45}],
        )

    monkeypatch.setattr(method, "_run_single_timeframe", fake_run)

    with pytest.raises(RuntimeError, match="Primary analog search rejected.*primary best score too high"):
        method.forecast(
            series,
            horizon=3,
            seasonality=1,
            params={
                "symbol": "EURUSD",
                "timeframe": "H1",
                "max_primary_best_score": 0.2,
            },
        )

    diag = method._get_timeframe_diagnostic("H1")
    assert diag["reason"] == "primary_best_score_too_high"
    assert diag["quality_gate"]["failed_check"] == "max_primary_best_score"


def test_analog_method_rejects_low_effective_path_count(monkeypatch):
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 108.0, 80), name="close")

    def fake_run(symbol, timeframe, horizon, params, query_vector=None, **kwargs):
        method._timeframe_diagnostics[timeframe] = {"status": "ok", "returned_paths": 2}
        return (
            [
                np.array([101.0, 102.0, 103.0], dtype=float),
                np.array([99.0, 98.0, 97.0], dtype=float),
            ],
            [{"score": 0.0}, {"score": 10.0}],
        )

    monkeypatch.setattr(method, "_run_single_timeframe", fake_run)

    with pytest.raises(RuntimeError, match="Analog ensemble rejected.*insufficient effective paths"):
        method.forecast(
            series,
            horizon=3,
            seasonality=1,
            params={
                "symbol": "EURUSD",
                "timeframe": "H1",
                "weight_temperature": 0.01,
                "min_effective_paths": 1.5,
            },
        )

    diag = method._get_timeframe_diagnostic("H1")
    assert diag["reason"] == "insufficient_effective_paths"
    assert diag["quality_gate"]["failed_check"] == "min_effective_paths"
    assert diag["ensemble_metrics"]["effective_paths"] < 1.5


def test_analog_method_reports_quality_gate_metrics_when_thresholds_pass(monkeypatch):
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 108.0, 80), name="close")

    def fake_run(symbol, timeframe, horizon, params, query_vector=None, **kwargs):
        method._timeframe_diagnostics[timeframe] = {"status": "ok", "returned_paths": 2}
        return (
            [
                np.array([101.0, 102.0, 103.0], dtype=float),
                np.array([100.0, 101.0, 102.0], dtype=float),
            ],
            [{"score": 0.1, "index": 0}, {"score": 0.2, "index": 1}],
        )

    monkeypatch.setattr(method, "_run_single_timeframe", fake_run)

    out = method.forecast(
        series,
        horizon=3,
        seasonality=1,
        params={
            "symbol": "EURUSD",
            "timeframe": "H1",
            "min_primary_paths": 2,
            "min_effective_paths": 1.0,
            "max_primary_best_score": 0.2,
            "max_primary_median_score": 0.2,
        },
    )

    assert out.params_used["min_primary_paths"] == 2
    assert out.params_used["min_effective_paths"] == 1.0
    assert out.metadata is not None
    assert out.metadata["ensemble_metrics"]["effective_paths"] > 1.0
    assert out.metadata["ensemble_metrics"]["quality_gate"]["status"] == "passed"
    assert out.metadata["ensemble_metrics"]["quality_gate"]["primary"]["n_paths"] == 2
    assert out.metadata["timeframe_diagnostics"]["H1"]["quality_gate"]["status"] == "passed"


def test_analog_method_routes_timeframe_specific_history_to_secondary(monkeypatch):
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 108.0, 80), name="close")
    seen_calls = []
    primary_history = pd.DataFrame(
        {
            "time": np.arange(8, dtype=np.int64) * 3600,
            "open": np.linspace(100.0, 107.0, 8),
            "high": np.linspace(100.5, 107.5, 8),
            "low": np.linspace(99.5, 106.5, 8),
            "close": np.linspace(100.2, 107.2, 8),
        }
    )
    secondary_history = pd.DataFrame(
        {
            "time": np.arange(2, dtype=np.int64) * 14400,
            "open": np.array([100.0, 104.0], dtype=float),
            "high": np.array([103.0, 107.0], dtype=float),
            "low": np.array([99.0, 103.0], dtype=float),
            "close": np.array([102.5, 106.5], dtype=float),
        }
    )

    def fake_run(symbol, timeframe, horizon, params, query_vector=None, **kwargs):
        seen_calls.append((timeframe, kwargs))
        method._timeframe_diagnostics[timeframe] = {"status": "ok", "returned_paths": 1}
        return ([np.array([101.0, 102.0, 103.0], dtype=float)], [{"score": 0.1}])

    monkeypatch.setattr(method, "_run_single_timeframe", fake_run)

    method.forecast(
        series,
        horizon=3,
        seasonality=1,
        params={"symbol": "EURUSD", "timeframe": "H1", "secondary_timeframes": "H4"},
        history_df=primary_history,
        history_base_col="close",
        history_by_timeframe={"H4": secondary_history},
        history_base_cols_by_timeframe={"H4": "close"},
        history_denoise_specs_by_timeframe={"H4": {"method": "sma", "params": {"window": 3}}},
    )

    secondary_kwargs = next(kwargs for timeframe, kwargs in seen_calls if timeframe == "H4")
    assert secondary_kwargs["history_df"].equals(secondary_history)
    assert secondary_kwargs["history_base_col"] == "close"
    assert secondary_kwargs["history_denoise_spec"] == {"method": "sma", "params": {"window": 3}}


def test_analog_method_resamples_primary_history_for_coarser_secondary(monkeypatch):
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 108.0, 80), name="close")
    seen_calls = []
    primary_history = pd.DataFrame(
        {
            "time": np.arange(12, dtype=np.int64) * 3600,
            "open": np.linspace(100.0, 111.0, 12),
            "high": np.linspace(100.5, 111.5, 12),
            "low": np.linspace(99.5, 110.5, 12),
            "close": np.linspace(100.2, 111.2, 12),
            "tick_volume": np.ones(12, dtype=float),
        }
    )

    def fake_run(symbol, timeframe, horizon, params, query_vector=None, **kwargs):
        seen_calls.append((timeframe, kwargs))
        method._timeframe_diagnostics[timeframe] = {"status": "ok", "returned_paths": 1}
        return ([np.array([101.0, 102.0, 103.0], dtype=float)], [{"score": 0.1}])

    monkeypatch.setitem(shared_constants.TIMEFRAME_SECONDS, "H1", 3600)
    monkeypatch.setitem(shared_constants.TIMEFRAME_SECONDS, "H4", 14400)
    monkeypatch.setattr(method, "_run_single_timeframe", fake_run)

    method.forecast(
        series,
        horizon=3,
        seasonality=1,
        params={"symbol": "EURUSD", "timeframe": "H1", "secondary_timeframes": "H4"},
        history_df=primary_history,
        history_base_col="close",
        history_denoise_spec={"method": "sma", "params": {"window": 3}},
    )

    secondary_kwargs = next(kwargs for timeframe, kwargs in seen_calls if timeframe == "H4")
    resampled_history = secondary_kwargs["history_df"]
    assert resampled_history is not None
    assert secondary_kwargs["history_base_col"] == "close"
    assert secondary_kwargs["history_denoise_spec"] == {"method": "sma", "params": {"window": 3}}
    assert resampled_history["time"].tolist() == [0, 14400, 28800]
    assert resampled_history["close"].tolist() == pytest.approx([103.2, 107.2, 111.2])


@pytest.mark.parametrize(
    ("name", "value", "suggestion"),
    [
        ("metric", "cosne", "cosine"),
        ("scale", "zsocre", "zscore"),
        ("refine_metric", "dwt", "dtw"),
    ],
)
def test_analog_method_rejects_unknown_similarity_settings_before_search(
    monkeypatch,
    name,
    value,
    suggestion,
):
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 108.0, 80), name="close")
    search = pytest.fail
    monkeypatch.setattr(method, "_run_single_timeframe", search)

    with pytest.raises(
        ValueError,
        match=rf"parameter '{name}'.*Did you mean '{suggestion}'",
    ):
        method.forecast(
            series,
            horizon=3,
            seasonality=1,
            params={"symbol": "EURUSD", "timeframe": "H1", name: value},
        )


def test_analog_method_rejects_incompatible_profile_similarity_settings():
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 108.0, 80), name="close")

    with pytest.raises(ValueError, match="requires metric='euclidean'.*scale='zscore'"):
        method.forecast(
            series,
            horizon=3,
            seasonality=1,
            params={
                "symbol": "EURUSD",
                "timeframe": "H1",
                "search_engine": "mass",
                "scale": "minmax",
            },
        )


@pytest.mark.parametrize(
    ("setting", "value", "suggestion"),
    [
        ("metric", "cosne", "cosine"),
        ("scale", "zsocre", "zscore"),
        ("refine_metric", "dwt", "dtw"),
    ],
)
def test_forecast_engine_rejects_invalid_analog_similarity_before_history_fetch(
    monkeypatch,
    setting,
    value,
    suggestion,
):
    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("history fetch should not run for invalid analog settings")

    monkeypatch.setattr(forecast_engine_module, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(forecast_engine_module, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(
        forecast_engine_module,
        "_get_available_methods",
        lambda: ("analog",),
    )
    monkeypatch.setattr(forecast_engine_module, "_fetch_history", unexpected_fetch)

    out = forecast_engine_module.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="analog",
        horizon=3,
        params={setting: value},
    )

    assert out["error_code"] == "invalid_analog_similarity_settings"
    assert f"parameter '{setting}'" in out["error"]
    assert f"Did you mean '{suggestion}'?" in out["error"]


def test_analog_resampling_drops_trailing_incomplete_bucket(monkeypatch):
    method = AnalogMethod()
    history = pd.DataFrame(
        {
            "time": np.arange(11, dtype=np.int64) * 3600,
            "close": np.arange(11, dtype=float),
        }
    )
    monkeypatch.setitem(shared_constants.TIMEFRAME_SECONDS, "H1", 3600)
    monkeypatch.setitem(shared_constants.TIMEFRAME_SECONDS, "H4", 14400)

    resampled = method._resample_history_df(history, "H1", "H4")

    assert resampled is not None
    assert resampled["time"].tolist() == [0, 14400]
    assert resampled.attrs["analog_resample"]["expected_constituents"] == 4
    assert resampled.attrs["analog_resample"]["constituent_counts"] == [4, 4]


def test_analog_resampling_drops_partial_lead_but_keeps_elapsed_session_gap(
    monkeypatch,
):
    method = AnalogMethod()
    history = pd.DataFrame(
        {
            "time": np.array([1, 2, 3, 4, 5, 7], dtype=np.int64) * 3600,
            "close": np.arange(6, dtype=float),
        }
    )
    monkeypatch.setitem(shared_constants.TIMEFRAME_SECONDS, "H1", 3600)
    monkeypatch.setitem(shared_constants.TIMEFRAME_SECONDS, "H4", 14400)

    resampled = method._resample_history_df(history, "H1", "H4")

    assert resampled is not None
    assert resampled["time"].tolist() == [14400]
    assert resampled.attrs["analog_resample"]["constituent_counts"] == [3]


def test_analog_secondary_resampling_preserves_materialized_denoised_base(
    monkeypatch,
):
    method = AnalogMethod()
    series = pd.Series(np.linspace(1000.0, 1080.0, 80), name="close_dn")
    seen_calls = []
    primary_history = pd.DataFrame(
        {
            "time": np.arange(12, dtype=np.int64) * 3600,
            "close": np.linspace(100.0, 111.0, 12),
            "close_dn": np.linspace(1000.0, 1110.0, 12),
        }
    )

    def fake_run(symbol, timeframe, horizon, params, query_vector=None, **kwargs):
        seen_calls.append((timeframe, kwargs))
        method._timeframe_diagnostics[timeframe] = {
            "status": "ok",
            "returned_paths": 1,
        }
        return ([np.linspace(1000.0, 1010.0, horizon)], [{"score": 0.1}])

    monkeypatch.setitem(shared_constants.TIMEFRAME_SECONDS, "H1", 3600)
    monkeypatch.setitem(shared_constants.TIMEFRAME_SECONDS, "H4", 14400)
    monkeypatch.setattr(method, "_run_single_timeframe", fake_run)

    method.forecast(
        series,
        horizon=3,
        seasonality=1,
        params={
            "symbol": "EURUSD",
            "timeframe": "H1",
            "base_col": "close_dn",
            "secondary_timeframes": "H4",
            "denoise": {"method": "ema"},
        },
        history_df=primary_history,
        history_base_col="close_dn",
        history_denoise_spec={"method": "ema"},
    )

    secondary_kwargs = next(
        kwargs for timeframe, kwargs in seen_calls if timeframe == "H4"
    )
    resampled = secondary_kwargs["history_df"]
    assert secondary_kwargs["history_base_col"] == "close_dn"
    assert resampled["close_dn"].tolist() == pytest.approx(
        [1030.0, 1070.0, 1110.0]
    )
    assert resampled.attrs["denoise_last_application"]["added_columns"] == [
        "close_dn"
    ]


@pytest.mark.parametrize(
    ("horizon", "expected"),
    [
        (4, [100.0, 100.0, 100.0, 104.0]),
        (5, [100.0, 100.0, 100.0, 104.0, 104.0]),
    ],
)
def test_analog_secondary_paths_align_from_future_endpoints(
    monkeypatch,
    horizon,
    expected,
):
    method = AnalogMethod()
    series = pd.Series(np.full(80, 100.0), name="close")

    def fake_run(symbol, timeframe, path_horizon, params, query_vector=None, **kwargs):
        method._timeframe_diagnostics[timeframe] = {
            "status": "ok",
            "returned_paths": 1,
        }
        if timeframe == "H1":
            return ([np.full(path_horizon, 100.0)], [{"score": 0.1}])
        return ([np.array([104.0, 108.0, 112.0])], [{"score": 0.1}])

    monkeypatch.setitem(shared_constants.TIMEFRAME_SECONDS, "H1", 3600)
    monkeypatch.setitem(shared_constants.TIMEFRAME_SECONDS, "H4", 14400)
    monkeypatch.setattr(method, "_run_single_timeframe", fake_run)

    result = method.forecast(
        series,
        horizon=horizon,
        seasonality=1,
        params={
            "symbol": "EURUSD",
            "timeframe": "H1",
            "secondary_timeframes": "H4",
            "component_weights": {"H1": 1.0, "H4": 100.0},
        },
    )

    assert result.forecast.tolist() == expected
    assert result.metadata["components"] == ["H1", "H4"]
    secondary = result.metadata["component_status"][1]
    assert secondary["alignment"] == "step_hold_from_primary_anchor"
    assert result.params_used["secondary_alignment"] == (
        "step_hold_from_primary_anchor"
    )
