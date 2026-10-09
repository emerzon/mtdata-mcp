"""Return forecasts keep log-return semantics and precision at public boundaries."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mtdata.forecast.forecast_engine import _reconstruct_prices_from_target
from mtdata.forecast.requests import ForecastGenerateRequest
from mtdata.forecast.use_cases.compact import _apply_forecast_generate_detail


@pytest.mark.parametrize("detail", ["summary", "compact", "standard", "full"])
@pytest.mark.parametrize("digits", [0, 2, 5])
def test_return_forecast_explains_per_bar_and_cumulative_log_returns(detail, digits):
    returns = [0.1, -0.04]
    prices = _reconstruct_prices_from_target(
        np.array(returns), np.array([100.0]), {"transform": "log_return"}
    )
    assert prices is not None
    # A 0.1 log return is 10.517%, and the second bar's -0.04 is not the
    # complete horizon return: the two-bar path is up approximately 6.184%.
    assert prices[0] / 100.0 - 1.0 == pytest.approx(0.1051709180756477)
    assert prices[-1] / 100.0 - 1.0 == pytest.approx(0.0618365465453596)
    lower = [0.0987326, -0.0431928]
    upper = [0.1049876, -0.0343877]
    payload = {
        "success": True,
        "method": "arima",
        "quantity": "return",
        "horizon": 2,
        "digits": digits,
        "last_price": 100.0,
        "forecast_time": ["2026-01-01T01:00Z", "2026-01-01T02:00Z"],
        "forecast_return": returns,
        "forecast_price": prices.tolist(),
        "lower_return": lower,
        "upper_return": upper,
        "lower": lower,
        "upper": upper,
        "ci_status": "available",
        "ci_alpha": 0.05,
    }
    result = _apply_forecast_generate_detail(
        payload,
        ForecastGenerateRequest(
            symbol="TEST", method="arima", quantity="return", horizon=2, detail=detail
        ),
    )

    assert result["return_unit"] == "log_return"
    assert "Per-bar log returns" in result["quantity_note"]
    assert "simple_return = expm1(return)" in result["quantity_note"]
    assert "cumulative_simple_return = expm1(sum(returns))" in result["quantity_note"]
    if detail == "summary":
        endpoints = result["forecast_endpoints"]
        assert endpoints["start_return"] == returns[0]
        assert endpoints["end_return"] == returns[-1]
        assert endpoints["end_price"] == round(100.0 * math.exp(0.06), digits)
    else:
        rows = result["forecast"]
        assert [row["return"] for row in rows] == returns
        assert [row["price"] for row in rows] == [round(value, digits) for value in prices]
        assert [row["lower_return"] for row in rows] == lower
        assert [row["upper_return"] for row in rows] == upper
    if detail == "full":
        assert result["lower"] == result["lower_return"] == lower
        assert result["upper"] == result["upper_return"] == upper
        assert "expm1(sum(returns))" in result["interpretation"]["forecast_return"]
    assert payload["lower"] == lower
    assert payload["forecast_price"] == prices.tolist()


@pytest.mark.parametrize("digits", [0, 2, 5])
def test_return_diagnostic_intervals_keep_return_precision(digits):
    lower_return = [-0.0021734, -0.0031928]
    upper_return = [0.0049876, 0.0056123]
    result = _apply_forecast_generate_detail(
        {
            "quantity": "return",
            "digits": digits,
            "ci_status": "insufficient_calibration",
            "ci_available": False,
            "diagnostic_bounds": {
                "lower_return": lower_return,
                "upper_return": upper_return,
                "lower_price": [99.1234567, 98.9876543],
                "upper_price": [100.2345678, 101.3456789],
                "usage": "diagnostic_only",
            },
        },
        ForecastGenerateRequest(symbol="TEST", quantity="return", detail="full"),
    )
    bounds = result["diagnostic_bounds"]
    assert bounds["lower_return"] == lower_return
    assert bounds["upper_return"] == upper_return
    assert bounds["lower_price"] == [round(value, digits) for value in [99.1234567, 98.9876543]]
    assert bounds["upper_price"] == [round(value, digits) for value in [100.2345678, 101.3456789]]
    assert bounds["usage"] == "diagnostic_only"
    assert result["forecast_mode"] == "point_only"


@pytest.mark.parametrize("digits", [0, 2, 5])
def test_price_interval_aliases_still_use_price_precision(digits):
    lower = [99.1234567]
    upper = [100.2345678]
    result = _apply_forecast_generate_detail(
        {
            "quantity": "price",
            "digits": digits,
            "lower_price": lower,
            "upper_price": upper,
            "lower": lower,
            "upper": upper,
            "ci_status": "available",
        },
        ForecastGenerateRequest(symbol="TEST", quantity="price", detail="full"),
    )
    assert result["lower"] == result["lower_price"] == [round(lower[0], digits)]
    assert result["upper"] == result["upper_price"] == [round(upper[0], digits)]
    assert "return_unit" not in result
