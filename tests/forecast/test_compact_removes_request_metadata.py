"""Test that compact detail mode removes request metadata from responses."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from mtdata.forecast.backtest import forecast_backtest, strategy_backtest
from mtdata.forecast.use_cases import _compact_backtest_result
from mtdata.utils.time import _format_time_minimal


def test_forecast_backtest_compact_excludes_request_metadata() -> None:
    """Test that forecast_backtest with detail='compact' doesn't echo request/resolved_request."""
    times = np.arange(1699999980, 1699999980 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 60
    anchor = _format_time_minimal(float(times[idx]))
    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_price": [110.0, 111.0, 112.0]},
    ):
        res_compact = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=3,
            methods=["naive"],
            anchors=[anchor],
            detail="compact",
            slippage_bps=2.5,
        )

    # In compact mode, request and resolved_request should NOT be present
    assert "request" not in res_compact, "compact mode should not include 'request'"
    assert "resolved_request" not in res_compact, "compact mode should not include 'resolved_request'"
    # But the response should still contain results
    assert res_compact["success"] is True
    assert res_compact["symbol"] == "EURUSD"
    assert res_compact["timeframe"] == "H1"
    assert res_compact["units"]["returns"] == "return_fraction"
    assert res_compact["units"]["forecast_error"] == "price"
    assert "results" in res_compact
    assert res_compact["execution_policy"] == {
        "entry": "next_bar_open",
        "exit": "first_close_reaching_terminal_forecast_else_horizon",
        "stop_loss": "none",
    }


def test_forecast_backtest_full_excludes_request_metadata() -> None:
    """Test that forecast_backtest with detail='full' doesn't echo request/resolved_request."""
    times = np.arange(1699999980, 1699999980 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 60
    anchor = _format_time_minimal(float(times[idx]))
    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_price": [110.0, 111.0, 112.0]},
    ):
        res_full = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=3,
            methods=["naive"],
            anchors=[anchor],
            detail="full",
            slippage_bps=2.5,
        )

    assert "request" not in res_full, "full mode should not include 'request'"
    assert "resolved_request" not in res_full, "full mode should not include 'resolved_request'"
    assert res_full["success"] is True
    assert res_full["units"]["forecast_error"] == "price"


def test_strategy_backtest_compact_excludes_request_metadata() -> None:
    """Test that strategy_backtest with detail='compact' doesn't echo request/resolved_request."""
    times = np.arange(1700000000, 1700000000 + 100 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 100, dtype=float)
    high = close + 0.5
    low = close - 0.5
    open_ = np.roll(close, 1)
    df = pd.DataFrame(
        {"time": times, "open": open_, "high": high, "low": low, "close": close}
    )

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df):
        res_compact = strategy_backtest(
            symbol="EURUSD",
            timeframe="H1",
            strategy="sma_cross",
            lookback=50,
            cost_model="fixed",
            spread_bps=0.0,
            detail="compact",
        )

    # In compact mode, request should NOT be present
    assert "request" not in res_compact, "compact mode should not include 'request'"
    assert "resolved_request" not in res_compact, "compact mode should not include 'resolved_request'"
    # But the response should still contain results
    assert res_compact["success"] is True
    assert res_compact["units"]["returns"] == "return_fraction"
    assert "drawdown" not in res_compact["units"]
    assert "summary" in res_compact
    assert res_compact["parameters"]["fast_period"] == 10
    assert res_compact["parameters"]["slow_period"] == 30
    assert "sample_warning" not in res_compact.get("metrics", {})
    if res_compact.get("metrics"):
        assert "sample_notice" not in res_compact["metrics"]
        assert res_compact["metrics"]["metrics_reliability"] == "low"
        assert res_compact["summary"]["sample_status"] == "insufficient_trades"
        assert "trades_per_year" not in res_compact["metrics"]


def test_strategy_backtest_full_includes_request_metadata() -> None:
    """Test that strategy_backtest with detail='full' includes request/resolved_request."""
    times = np.arange(1700000000, 1700000000 + 100 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 100, dtype=float)
    high = close + 0.5
    low = close - 0.5
    open_ = np.roll(close, 1)
    df = pd.DataFrame(
        {"time": times, "open": open_, "high": high, "low": low, "close": close}
    )

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df):
        res_full = strategy_backtest(
            symbol="EURUSD",
            timeframe="H1",
            strategy="sma_cross",
            lookback=50,
            detail="full",
            cost_model="fixed",
            spread_bps=1.0,
        )

    # In full mode, request SHOULD be present
    assert "request" in res_full, "full mode should include 'request'"
    assert res_full["request"]["symbol"] == "EURUSD"
    assert res_full["request"]["strategy"] == "sma_cross"
    assert res_full["request"]["lookback"] == 50


def test_compact_backtest_preserves_integer_count_serialization() -> None:
    compact = _compact_backtest_result(
        {
            "success": True,
            "units": {
                "successful_tests": "count",
                "failed_tests": "count",
                "num_tests": "count",
                "anchor_tests_planned": "count",
                "anchor_tests_succeeded": "count",
                "anchor_tests_failed": "count",
            },
            "results": {
                "theta": {
                    "success": True,
                    "avg_rmse": 0.12,
                    "successful_tests": 5,
                    "num_tests": 5,
                    "details": [{}, {}, {}, {}, {}],
                    "metrics": {"trades_observed": 5},
                    "metrics_available": True,
                }
            },
        }
    )

    method = compact["results"]["theta"]
    assert method["successful_tests"] == 5
    assert method["num_tests"] == 5
    assert method["details_count"] == 5
    assert method["trades_observed"] == 5
    assert isinstance(method["successful_tests"], int)
    assert isinstance(method["num_tests"], int)
    assert isinstance(method["details_count"], int)
    assert isinstance(method["trades_observed"], int)
    assert compact["units"]["anchor_tests_planned"] == "count"
    assert compact["units"]["anchor_tests_succeeded"] == "count"
    assert compact["units"]["anchor_tests_failed"] == "count"


def test_compact_backtest_exposes_partial_anchor_failures() -> None:
    compact = _compact_backtest_result(
        {
            "success": True,
            "results": {
                "arima": {
                    "success": True,
                    "avg_rmse": 0.12,
                    "successful_tests": 3,
                    "num_tests": 5,
                    "details": [
                        {"success": True},
                        {"success": True},
                        {"success": True},
                        {"success": False, "error": "fit failed"},
                        {"success": False, "error": "fit failed"},
                    ],
                    "metrics_available": True,
                }
            },
        }
    )

    assert compact["complete_success"] is False
    assert compact["status"] == "partial"
    assert compact["methods_succeeded"] == 1
    assert compact["methods_complete"] == 0
    assert compact["methods_partial"] == 1
    assert compact["methods_failed"] == 0
    assert compact["partial_methods"] == ["arima"]
    assert compact["anchor_tests_planned"] == 5
    assert compact["anchor_tests_succeeded"] == 3
    assert compact["anchor_tests_failed"] == 2
    assert compact["results"]["arima"]["status"] == "partial"
    assert compact["results"]["arima"]["complete_success"] is False
    assert compact["results"]["arima"]["failed_tests"] == 2
    assert compact["ranked_methods"][0]["ranking_status"] == "unranked"
    assert (
        compact["ranked_methods"][0]["unranked_reason"]
        == "incomplete_anchor_coverage"
    )
    assert compact["ranking"]["scope"] == "complete_methods_with_finite_avg_rmse"


def test_compact_backtest_ranks_low_history_methods() -> None:
    compact = _compact_backtest_result(
        {
            "success": True,
            "results": {
                "theta": {
                    "success": True,
                    "avg_rmse": 0.12,
                    "history_sample_ok": False,
                    "forecast_reliability": "low",
                    "recommended_history_bars": 30,
                    "low_history_anchors": 3,
                    "warnings": ["3 of 3 anchors used fewer than the recommended 30 training bars."],
                    "metrics_available": True,
                }
            },
        }
    )

    ranked = compact["ranked_methods"][0]
    assert ranked["ranking_status"] == "ranked"
    assert ranked["history_sample_ok"] is False
    assert ranked["forecast_reliability"] == "low"
    assert ranked["recommended_history_bars"] == 30
    assert ranked["selection_warning"] == (
        "ranking_uses_forecast_error_only; trading metrics are unavailable; "
        "low_history_sample"
    )
    assert compact["results"]["theta"]["history_sample_ok"] is False
    assert compact["results"]["theta"]["low_history_anchors"] == 3
    assert compact["execution_policy"] == {
        "entry": "next_bar_open",
        "exit": "first_close_reaching_terminal_forecast_else_horizon",
        "stop_loss": "none",
    }
