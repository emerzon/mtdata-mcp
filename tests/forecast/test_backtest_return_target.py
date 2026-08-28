from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from mtdata.forecast.backtest import (
    _compact_metrics_payload,
    _compute_performance_metrics,
    forecast_backtest,
)
from mtdata.utils.time import _format_time_minimal, format_epoch_utc


def test_backtest_return_target_scores_against_returns() -> None:
    times = np.arange(1699999980, 1699999980 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 60
    horizon = 2
    anchor = _format_time_minimal(float(times[idx]))
    actual_returns = np.log(close[idx + 1 : idx + 1 + horizon] / close[idx : idx + horizon])

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_return": [float(v) for v in actual_returns.tolist()]},
    ):
        res = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=horizon,
            methods=["naive"],
            anchors=[anchor],
            quantity="return",
        )

    detail = res["results"]["naive"]["details"][0]
    assert detail["success"] is True
    assert abs(float(detail["mae"])) < 1e-12
    assert abs(float(detail["rmse"])) < 1e-12
    assert res["units"]["avg_mae"] == "log_return"
    assert res["units"]["avg_rmse"] == "log_return"
    reference = res["directional_accuracy_reference"]
    assert reference["value"] == 0.5
    assert reference["basis"] == "balanced_binary_chance"


def test_backtest_volatility_with_return_target_uses_price_truth_windows() -> None:
    times = np.arange(1699999980, 1699999980 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 60
    horizon = 3
    anchor = _format_time_minimal(float(times[idx]))
    truth_path = close[idx : idx + 1 + horizon]
    expected_sigma = float(np.sqrt(np.sum(np.diff(np.log(truth_path)) ** 2)))

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast_volatility",
        return_value={"volatility_horizon": expected_sigma},
    ):
        res = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=horizon,
            methods=["ewma"],
            anchors=[anchor],
            quantity="volatility",
        )

    detail = res["results"]["ewma"]["details"][0]
    assert bool(detail["success"]) is True
    assert abs(float(detail["realized_sigma"]) - expected_sigma) < 1e-12
    assert res["units"]["avg_rmse"] == "return_fraction"
    assert "directional_accuracy_reference" not in res


def test_backtest_aggregates_volatility_rmse_from_squared_errors() -> None:
    times = np.arange(1699999980, 1699999980 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})
    anchor_indices = [60, 61]
    anchors = [_format_time_minimal(float(times[idx])) for idx in anchor_indices]
    realized = []
    for idx in anchor_indices:
        path = close[idx : idx + 2]
        realized.append(float(np.sqrt(np.sum(np.diff(np.log(path)) ** 2))))

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast_volatility",
        side_effect=[
            {"volatility_horizon": realized[0] + 1.0},
            {"volatility_horizon": realized[1] + 5.0},
        ],
    ):
        result = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=1,
            methods=["ewma"],
            anchors=anchors,
            quantity="volatility",
        )

    aggregate = result["results"]["ewma"]
    assert aggregate["avg_mae"] == 3.0
    assert aggregate["avg_rmse"] == np.sqrt(13.0)


def test_backtest_pools_mae_and_rmse_over_the_same_forecast_points() -> None:
    times = np.arange(1699999980, 1699999980 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})
    anchor_indices = [60, 64]
    anchors = [_format_time_minimal(float(times[idx])) for idx in anchor_indices]
    first_actual = close[61:64]
    second_actual = close[65:68]

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        side_effect=[
            {"forecast_price": [float(value + 1.0) for value in first_actual]},
            {"forecast_price": [float(second_actual[0] + 5.0)]},
        ],
    ):
        result = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=3,
            methods=["naive"],
            anchors=anchors,
        )

    aggregate = result["results"]["naive"]
    assert aggregate["avg_mae"] == 2.0
    assert aggregate["avg_rmse"] == np.sqrt(7.0)


def test_backtest_return_target_converts_log_returns_to_simple_trade_returns() -> None:
    times = np.arange(1699999980, 1699999980 + 80 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 140.0, 80, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 70
    horizon = 2
    anchor = _format_time_minimal(float(times[idx]))
    actual_returns = np.log(close[idx + 1 : idx + 1 + horizon] / close[idx : idx + horizon])
    expected_simple = float(np.exp(np.sum(actual_returns)) - 1.0)

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_return": [float(v) for v in actual_returns.tolist()]},
    ):
        res = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=horizon,
            methods=["naive"],
            anchors=[anchor],
            quantity="return",
        )

    detail = res["results"]["naive"]["details"][0]
    assert detail["success"] is True
    assert abs(float(detail["trade_return"]) - expected_simple) < 1e-12


def test_backtest_executes_completed_close_signal_at_next_open() -> None:
    times = np.arange(1699999980, 1699999980 + 80 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 80, dtype=float)
    open_ = close.copy()
    idx = 70
    open_[idx + 1] = close[idx] + 5.0
    df = pd.DataFrame({"time": times, "open": open_, "close": close})
    anchor = _format_time_minimal(float(times[idx]))
    forecast_target = float(open_[idx + 1] + 1.0)

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_price": [forecast_target]},
    ):
        result = forecast_backtest(
            symbol="AAPL.NAS",
            timeframe="H1",
            horizon=1,
            methods=["naive"],
            anchors=[anchor],
        )

    detail = result["results"]["naive"]["details"][0]
    expected = (close[idx + 1] - open_[idx + 1]) / open_[idx + 1]
    assert detail["entry_price"] == open_[idx + 1]
    assert detail["entry_price_source"] == "next_bar_open"
    assert detail["entry_time"] == format_epoch_utc(
        float(times[idx + 1]),
        timespec="seconds",
    )
    assert detail["trade_return_gross"] == expected
    assert result["signal_timing"] == "completed_bar_close"
    assert result["execution_timing"] == "next_bar_open"


def test_performance_metrics_skip_annualization_for_short_samples() -> None:
    metrics = _compute_performance_metrics(
        returns=[0.01, -0.02, 0.015, -0.005, 0.01, 0.0],
        timeframe="M15",
        horizon=12,
        slippage_bps=0.0,
    )
    assert metrics["sharpe_ratio"] is None
    assert metrics["annual_return"] is None
    assert metrics["calmar_ratio"] is None
    assert "sample_warning" in metrics
    assert metrics["sample_notice"]["trades_observed"] == 6
    assert int(metrics["min_trades_for_annualization"]) == 30


def test_compact_metrics_suppresses_low_sample_trading_ratios() -> None:
    metrics = _compute_performance_metrics(
        returns=[0.01, -0.02],
        timeframe="M15",
        horizon=12,
        slippage_bps=0.0,
    )

    compact = _compact_metrics_payload(metrics)

    assert compact["metrics_reliability"] == "low"
    assert compact["trades_observed"] == 2
    assert "win_rate_pct" not in compact
    assert "kelly_fraction" not in compact
    assert "avg_win_loss_ratio" not in compact


def test_performance_metrics_clamps_total_loss_returns() -> None:
    metrics = _compute_performance_metrics(
        returns=[0.01, -1.5, 0.03, -1.0],
        timeframe="H1",
        horizon=1,
        slippage_bps=0.0,
    )

    assert metrics["trades_observed"] == 4
    assert metrics["avg_return_per_trade"] == np.mean([0.01, -0.999, 0.03, -0.999])
    assert 0.0 <= metrics["max_drawdown"] <= 1.0


def test_backtest_price_target_trade_returns_vary_by_forecast_implied_exit() -> None:
    times = np.arange(1699999980, 1699999980 + 90 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 145.0, 90, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 80
    horizon = 3
    anchor = _format_time_minimal(float(times[idx]))

    def _fake_forecast(**kwargs):
        method = str(kwargs.get("method", ""))
        if method == "slow":
            return {"forecast_price": [140.8, 141.0, 141.2]}
        return {"forecast_price": [143.5, 144.0, 144.5]}

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        side_effect=_fake_forecast,
    ):
        res = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=horizon,
            methods=["slow", "aggressive"],
            anchors=[anchor],
        )

    slow_detail = res["results"]["slow"]["details"][0]
    aggressive_detail = res["results"]["aggressive"]["details"][0]

    assert slow_detail["success"] is True
    assert aggressive_detail["success"] is True
    assert slow_detail["position"] == "long"
    assert aggressive_detail["position"] == "long"
    assert float(slow_detail["trade_return"]) != float(aggressive_detail["trade_return"])
    assert int(slow_detail["exit_step"]) < int(aggressive_detail["exit_step"])


def test_backtest_default_detail_is_compact_without_full_series_arrays() -> None:
    times = np.arange(1699999980, 1699999980 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 60
    anchor = _format_time_minimal(float(times[idx]))
    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_price": [110.0, 111.0, 112.0]},
    ):
        res = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=3,
            methods=["naive"],
            anchors=[anchor],
        )

    detail = res["results"]["naive"]["details"][0]
    assert res["detail"] == "compact"
    assert "forecast" not in detail
    assert "actual" not in detail
    assert "forecast_end" in detail
    assert "actual_end" in detail


def test_backtest_full_detail_includes_series_arrays() -> None:
    times = np.arange(1699999980, 1699999980 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 60
    anchor = _format_time_minimal(float(times[idx]))
    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_price": [110.0, 111.0, 112.0]},
    ):
        res = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=3,
            methods=["naive"],
            anchors=[anchor],
            detail="full",
        )

    detail = res["results"]["naive"]["details"][0]
    assert res["detail"] == "full"
    assert isinstance(detail["forecast"], list)
    assert isinstance(detail["actual"], list)


def test_backtest_full_detail_omits_request_metadata_blocks() -> None:
    times = np.arange(1699999980, 1699999980 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 60
    anchor = _format_time_minimal(float(times[idx]))
    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_price": [110.0, 111.0, 112.0]},
    ):
        res = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon="3",  # type: ignore[arg-type]
            methods="naive drift",  # type: ignore[arg-type]
            anchors=[anchor],
            detail="FULL",  # type: ignore[arg-type]
            slippage_bps=2.5,
            trade_threshold=0.01,
        )

    assert res["detail"] == "full"
    assert "request" not in res
    assert "resolved_request" not in res


def test_backtest_full_detail_keeps_actionable_strategy_intent_only() -> None:
    times = np.arange(1699999980, 1699999980 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 60
    anchor = _format_time_minimal(float(times[idx]))
    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_price": [110.0, 111.0, 112.0]},
    ):
        res = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=3,
            methods=["naive"],
            anchors=[anchor],
            detail="full",
            trade_threshold=0.01,
        )

    detail = res["results"]["naive"]["details"][0]
    assert "contracts" not in res
    assert detail["strategy_intent"]["direction"] in {"long", "flat", "short"}
    assert "strategy_context" not in detail
