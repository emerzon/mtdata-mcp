"""Unknown methods stay typos; registered-but-uninstalled methods name the dependency."""

from __future__ import annotations

from mtdata.core.forecast_tasks import ForecastTrainRequest, forecast_train
from mtdata.forecast.forecast_validation import (
    canonicalize_forecast_methods,
    forecast_method_resolution_error,
)
from mtdata.forecast.requests import ForecastConformalIntervalsRequest
from mtdata.forecast.use_cases.generate import run_forecast_conformal_intervals
from mtdata.utils.minimal_output import format_result_minimal


def _unwrap(fn):
    current = fn
    while hasattr(current, "__wrapped__"):
        current = current.__wrapped__
    return current


def test_resolution_error_distinguishes_typo_from_missing_dependency(monkeypatch):
    monkeypatch.setattr(
        "mtdata.forecast.forecast_validation.get_forecast_methods_snapshot",
        lambda: {
            "methods": [
                {
                    "method": "theta",
                    "available": True,
                    "requires": [],
                },
                {
                    "method": "nbeatsx",
                    "available": False,
                    "requires": ["neuralforecast", "neuralforecast>=1.0.0"],
                },
            ]
        },
    )

    typo = forecast_method_resolution_error("nbeatsxx")
    missing = forecast_method_resolution_error("nbeatsx")

    assert typo is not None
    assert typo["error_code"] == "invalid_method"
    assert typo["error"].startswith("Invalid method: nbeatsxx")
    assert "Did you mean" in typo["error"]

    assert missing is not None
    assert missing["error_code"] == "method_dependency_missing"
    assert missing["unavailable_reason"] == "Requires: neuralforecast, neuralforecast>=1.0.0"
    assert "neuralforecast" in missing["error"]
    assert "Invalid method" not in missing["error"]


def test_canonicalize_rejects_unavailable_registered_method(monkeypatch):
    monkeypatch.setattr(
        "mtdata.forecast.forecast_validation.get_forecast_methods_snapshot",
        lambda: {
            "methods": [
                {
                    "method": "nbeatsx",
                    "available": False,
                    "requires": ["neuralforecast"],
                }
            ]
        },
    )

    _canonical, error = canonicalize_forecast_methods(
        ["nbeatsx"],
        valid_methods=["nbeatsx", "theta"],
    )

    assert _canonical is None
    assert error is not None
    assert error["error_code"] == "method_dependency_missing"
    assert error["unavailable_reason"] == "Requires: neuralforecast"


def test_conformal_intervals_fail_before_backtest_when_dependency_missing(monkeypatch):
    monkeypatch.setattr(
        "mtdata.forecast.forecast_validation.get_forecast_methods_snapshot",
        lambda: {
            "methods": [
                {
                    "method": "nbeatsx",
                    "available": False,
                    "requires": ["neuralforecast>=1.0.0"],
                }
            ]
        },
    )

    def _should_not_run(**_kwargs):
        raise AssertionError("calibration backtest must not start")

    result = run_forecast_conformal_intervals(
        ForecastConformalIntervalsRequest(symbol="EURUSD", method="nbeatsx", horizon=3),
        backtest_impl=_should_not_run,
        forecast_impl=_should_not_run,
    )

    assert result["success"] is False
    assert result["error_code"] == "method_dependency_missing"
    assert result["details"]["unavailable_reason"] == "Requires: neuralforecast>=1.0.0"
    assert "neuralforecast" in result["error"]


def test_forecast_train_reports_missing_dependency_before_submit(monkeypatch):
    monkeypatch.setattr(
        "mtdata.forecast.forecast_validation.get_forecast_methods_snapshot",
        lambda: {
            "methods": [
                {
                    "method": "nbeatsx",
                    "available": False,
                    "requires": ["neuralforecast>=1.0.0"],
                }
            ]
        },
    )

    def _should_not_connect():
        raise AssertionError("missing-dependency train must not open MT5")

    monkeypatch.setattr(
        "mtdata.utils.mt5.ensure_mt5_connection_or_raise",
        _should_not_connect,
    )

    result = _unwrap(forecast_train)(
        ForecastTrainRequest(symbol="EURUSD", method="nbeatsx", horizon=3)
    )

    assert result["success"] is False
    assert result["error_code"] == "method_dependency_missing"
    assert "neuralforecast" in result["error"]


def test_list_methods_toon_includes_unavailable_reason():
    rendered = format_result_minimal(
        {
            "methods": [
                {
                    "method": "nbeatsx",
                    "category": "neural",
                    "available": False,
                    "supports_ci": False,
                    "supports_training": True,
                    "unavailable_reason": "Requires: neuralforecast, neuralforecast>=1.0.0",
                }
            ]
        },
        verbose=False,
        tool_name="forecast_list_methods",
    )

    assert "unavailable_reason" in rendered
    assert "neuralforecast" in rendered
