from __future__ import annotations

import pytest

from mtdata.forecast.requests import ForecastConformalIntervalsRequest
from mtdata.forecast.use_cases.generate import run_forecast_conformal_intervals


@pytest.mark.parametrize("detail", ["compact", "full"])
def test_conformal_intervals_reject_partial_anchor_calibration(detail: str) -> None:
    successful_details = [
        {
            "success": True,
            "forecast": [10.0 + index],
            "actual": [9.0 + index],
        }
        for index in range(30)
    ]

    result = run_forecast_conformal_intervals(
        ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            method="theta",
            horizon=1,
            steps=31,
            spacing=1,
            ci_alpha=0.1,
            detail=detail,
        ),
        backtest_impl=lambda **kwargs: {
            "success": True,
            "complete_success": False,
            "status": "partial",
            "results": {
                "theta": {
                    "success": True,
                    "complete_success": False,
                    "status": "partial",
                    "successful_tests": 30,
                    "failed_tests": 1,
                    "num_tests": 31,
                    "details": successful_details
                    + [{"success": False, "error": "fit failed"}],
                }
            },
        },
        forecast_impl=lambda **kwargs: {
            "success": True,
            "method": "theta",
            "forecast_price": [100.0],
        },
    )

    assert result["ci_status"] == "incomplete_anchor_coverage"
    assert result["coverage_status"] == "incomplete_anchor_coverage"
    assert result["ci_available"] is False
    assert result["calibration_sufficient"] is False
    assert result["calibration_complete"] is False
    assert result["calibration_anchor_tests_planned"] == 31
    assert result["calibration_anchor_tests_succeeded"] == 30
    assert result["calibration_anchor_tests_failed"] == 1
    assert result["interval_usage"] == "diagnostic_only"
    assert "lower_price" not in result
    assert "upper_price" not in result
    assert result["diagnostic_bounds"] == {
        "lower_price": [99.0],
        "upper_price": [101.0],
        "usage": "diagnostic_only",
    }
    assert "incomplete_interval_anchor_coverage" in result["trust_blockers"]
    assert any("1 of 31 anchor tests" in warning for warning in result["warnings"])
    conformal = result["conformal"]
    assert conformal["calibration_anchor_tests_planned"] == 31
    assert conformal["calibration_anchor_tests_succeeded"] == 30
    assert conformal["calibration_anchor_tests_failed"] == 1
    assert conformal["calibration_complete"] is False
