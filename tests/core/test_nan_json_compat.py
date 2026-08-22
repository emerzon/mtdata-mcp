"""Test that NaN values in forecast output are properly converted to null in JSON.

Sanitization is owned by the transport boundary (``SafeJSONResponse``), not the
handlers. These tests render handler results through that response class and
assert the JSON the client receives has NaN/inf converted to null.
"""

import json
import math
from unittest.mock import MagicMock, patch

import pytest

from mtdata.core.web_api_handlers import (
    post_backtest_response,
    post_forecast_price_response,
    post_forecast_volatility_response,
)
from mtdata.core.web_api_runtime import SafeJSONResponse
from mtdata.forecast.requests import (
    ForecastBacktestRequest as BacktestBody,
)
from mtdata.forecast.requests import (
    ForecastGenerateRequest as ForecastPriceBody,
)
from mtdata.forecast.requests import (
    ForecastVolatilityEstimateRequest as ForecastVolBody,
)


def _rendered(result):
    """Render a handler result through the Web API response boundary."""
    return json.loads(SafeJSONResponse(content=result).body)


class TestNaNJsonCompat:
    """Verify that NaN/inf values are converted to null for JSON compatibility."""

    def test_backtest_with_nan_exit_price(self):
        """Test that NaN exit_price is converted to null in backtest response."""
        # Simulate a backtest result with NaN values (as would be produced by forecast_backtest)
        backtest_result = {
            "success": True,
            "results": {
                "naive": {
                    "success": True,
                    "details": [
                        {
                            "anchor": "2026-04-15 02:00",
                            "success": True,
                            "mae": 0.00048,
                            "rmse": 0.00059,
                            "entry_price": 1.17835,
                            "exit_price": float("nan"),  # The problem!
                            "position": "flat",
                        }
                    ],
                }
            },
        }

        body = BacktestBody(symbol="EURUSD", timeframe="H1")
        backtest_impl = MagicMock(return_value=backtest_result)

        result = post_backtest_response(body=body, backtest_use_case=backtest_impl)

        # NaN is converted to null when rendered through the response boundary.
        parsed = _rendered(result)
        assert parsed["results"]["naive"]["details"][0]["exit_price"] is None

    def test_backtest_with_nan_returns(self):
        """Test that NaN trade returns are converted to null."""
        backtest_result = {
            "success": True,
            "results": {
                "naive": {
                    "success": True,
                    "details": [
                        {
                            "anchor": "2026-04-15 02:00",
                            "trade_return_gross": float("nan"),
                            "trade_return": float("nan"),
                        }
                    ],
                }
            },
        }

        body = BacktestBody(symbol="EURUSD", timeframe="H1")
        backtest_impl = MagicMock(return_value=backtest_result)

        result = post_backtest_response(body=body, backtest_use_case=backtest_impl)

        parsed = _rendered(result)
        assert parsed["results"]["naive"]["details"][0]["trade_return_gross"] is None
        assert parsed["results"]["naive"]["details"][0]["trade_return"] is None
        assert '"trade_return_gross":null' in SafeJSONResponse(content=result).body.decode()

    def test_backtest_with_inf_values(self):
        """Test that infinity values are converted to null."""
        backtest_result = {
            "success": True,
            "metrics": {
                "max_return": float("inf"),
                "min_return": float("-inf"),
            },
        }

        body = BacktestBody(symbol="EURUSD", timeframe="H1")
        backtest_impl = MagicMock(return_value=backtest_result)

        result = post_backtest_response(body=body, backtest_use_case=backtest_impl)

        parsed = _rendered(result)
        assert parsed["metrics"]["max_return"] is None
        assert parsed["metrics"]["min_return"] is None

    def test_forecast_price_with_nan_values(self):
        """Test that forecast_price response sanitizes NaN values."""
        forecast_result = {
            "success": True,
            "forecast_price": [100.0, 101.0, float("nan"), 103.0],
            "forecast_interval_lower": [99.0, 99.5, float("nan"), 102.0],
            "forecast_interval_upper": [101.0, 102.0, float("inf"), 104.0],
        }

        body = ForecastPriceBody(symbol="EURUSD", timeframe="H1", horizon=4)
        forecast_impl = MagicMock(return_value=forecast_result)

        result = post_forecast_price_response(
            body=body, forecast_generate_use_case=forecast_impl
        )

        parsed = _rendered(result)
        assert parsed["success"] is True
        assert isinstance(parsed["forecast_price"], list)
        # NaN/inf entries become null in the rendered lists.
        assert parsed["forecast_price"][2] is None
        assert parsed["forecast_interval_upper"][2] is None

    def test_forecast_volatility_with_nan(self):
        """Test that forecast_volatility response sanitizes NaN values."""
        forecast_result = {
            "success": True,
            "volatility_horizon": float("nan"),
            "confidence_interval_lower": float("inf"),
            "confidence_interval_upper": float("-inf"),
        }

        body = ForecastVolBody(symbol="EURUSD", timeframe="H1", horizon=12)
        forecast_impl = MagicMock(return_value=forecast_result)

        result = post_forecast_volatility_response(
            body=body, forecast_vol_impl=forecast_impl
        )

        parsed = _rendered(result)
        assert parsed["volatility_horizon"] is None
        assert parsed["confidence_interval_lower"] is None
        assert parsed["confidence_interval_upper"] is None

    def test_nested_nan_in_complex_structure(self):
        """Test that NaN values are handled in deeply nested structures."""
        backtest_result = {
            "success": True,
            "results": {
                "method1": {
                    "details": [
                        {
                            "per_anchor": [
                                {
                                    "metrics": {
                                        "sharpe": float("nan"),
                                        "sortino": float("inf"),
                                    }
                                }
                            ]
                        }
                    ]
                }
            },
        }

        body = BacktestBody(symbol="EURUSD", timeframe="H1")
        backtest_impl = MagicMock(return_value=backtest_result)

        result = post_backtest_response(body=body, backtest_use_case=backtest_impl)

        parsed = _rendered(result)
        # Navigate to the nested NaN value
        nested = parsed["results"]["method1"]["details"][0]["per_anchor"][0]["metrics"]
        assert nested["sharpe"] is None
        assert nested["sortino"] is None

        # Full JSON serialization should work
        json_str = SafeJSONResponse(content=result).body.decode()
        assert "NaN" not in json_str
        assert "Infinity" not in json_str
