from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.backtest import (
    _compact_metrics_payload,
    _compute_performance_metrics,
    execute_forecast_backtest,
    forecast_backtest,
)
from mtdata.utils.time import (
    _format_time_minimal,
    bar_close_epoch,
    format_epoch_utc,
)


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


def test_backtest_volatility_full_detail_propagates_effective_params() -> None:
    times = np.arange(1699999980, 1699999980 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 60
    anchor = _format_time_minimal(float(times[idx]))
    params_used = {
        "lookback": 60,
        "lambda_": 0.97,
        "lambda_source": "lambda_",
    }

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast_volatility",
        return_value={
            "volatility_horizon": 0.02,
            "params_used": params_used,
        },
    ) as volatility:
        result = forecast_backtest(
            symbol="BTCUSD",
            timeframe="H1",
            horizon=3,
            lookback=60,
            methods=["ewma"],
            params_per_method={"ewma": {"lambda_": 0.97}},
            anchors=[anchor],
            quantity="volatility",
            detail="full",
        )

    volatility.assert_called_once()
    assert volatility.call_args.kwargs["params"] == {
        "lambda_": 0.97,
        "lookback": 60,
    }
    assert volatility.call_args.kwargs["detail"] == "full"
    detail = result["results"]["ewma"]["details"][0]
    assert detail["params_used"] == params_used


def test_backtest_volatility_compact_detail_omits_effective_params() -> None:
    times = np.arange(1699999980, 1699999980 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 60
    anchor = _format_time_minimal(float(times[idx]))
    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast_volatility",
        return_value={
            "volatility_horizon": 0.02,
            "params_used": {"lookback": 60, "lambda_": 0.97},
        },
    ):
        result = forecast_backtest(
            symbol="BTCUSD",
            timeframe="H1",
            horizon=3,
            lookback=60,
            methods=["ewma"],
            params_per_method={"ewma": {"lambda_": 0.97}},
            anchors=[anchor],
            quantity="volatility",
        )

    detail = result["results"]["ewma"]["details"][0]
    assert "params_used" not in detail


def test_backtest_volatility_copies_effective_params_per_anchor() -> None:
    times = np.arange(1699999980, 1699999980 + 80 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 80, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})
    anchors = [
        _format_time_minimal(float(times[idx]))
        for idx in (60, 64)
    ]
    shared_params = {"lookback": 60, "lambda_": 0.97}

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast_volatility",
        side_effect=[
            {"volatility_horizon": 0.02, "params_used": shared_params},
            {"volatility_horizon": 0.03, "params_used": shared_params},
        ],
    ):
        result = forecast_backtest(
            symbol="BTCUSD",
            timeframe="H1",
            horizon=3,
            lookback=60,
            methods=["ewma"],
            anchors=anchors,
            quantity="volatility",
            detail="full",
        )

    details = result["results"]["ewma"]["details"]
    details[0]["params_used"]["lambda_"] = 0.5
    assert details[1]["params_used"]["lambda_"] == 0.97
    assert shared_params["lambda_"] == 0.97


@pytest.mark.parametrize(
    "window_args",
    [
        {"lookback": 2160},
        {"params": {"lookback": 2160}},
        {"params": {"lookback": None}},
        {"params_per_method": {"har_rv": {"lookback": 2160}}},
        {"params_per_method": {"har_rv": {"lookback": None}}},
    ],
)
def test_backtest_har_rv_rejects_bar_lookback_before_history_fetch(
    window_args,
) -> None:
    with patch("mtdata.forecast.backtest._fetch_history") as fetch:
        result = forecast_backtest(
            symbol="BTCUSD",
            timeframe="H1",
            horizon=3,
            methods=["har_rv"],
            quantity="volatility",
            **window_args,
        )

    assert result["success"] is False
    assert result["error_code"] == "har_rv_lookback_unsupported"
    assert "params.days" in result["error"]
    fetch.assert_not_called()


@pytest.mark.parametrize(
    "window_args",
    [
        {
            "lookback": 2160,
            "params_per_method": {
                "ensemble": {"methods": ["ewma", "har_rv"]},
            },
        },
        {
            "params": {"lookback": 2160},
            "params_per_method": {
                "ensemble": {"methods": "ewma,har_rv"},
            },
        },
        {
            "params_per_method": {
                "ensemble": {
                    "methods": ["ewma", "har_rv"],
                    "lookback": None,
                },
            },
        },
        {
            "params_per_method": {
                "ensemble": {
                    "methods": ["ewma", "har_rv"],
                    "method_params": {"har_rv": {"lookback": 2160}},
                },
            },
        },
        {
            "params_per_method": {
                "ensemble": {
                    "methods": ["ewma", "har_rv"],
                    "method_params": {"har_rv": {"lookback": None}},
                },
            },
        },
    ],
)
def test_backtest_har_rv_ensemble_rejects_lookback_before_history_fetch(
    window_args,
) -> None:
    with patch("mtdata.forecast.backtest._fetch_history") as fetch:
        result = forecast_backtest(
            symbol="BTCUSD",
            timeframe="H1",
            horizon=3,
            methods=["ensemble"],
            quantity="volatility",
            **window_args,
        )

    assert result["success"] is False
    assert result["error_code"] == "har_rv_lookback_unsupported"
    assert "params.days" in result["error"]
    fetch.assert_not_called()


def test_backtest_non_har_ensemble_accepts_lookback() -> None:
    times = np.arange(1699999980, 1699999980 + 70 * 3600, 3600, dtype=float)
    frame = pd.DataFrame(
        {
            "time": times,
            "open": np.linspace(100.0, 120.0, 70, dtype=float),
            "close": np.linspace(100.0, 120.0, 70, dtype=float),
        }
    )
    anchor = _format_time_minimal(float(times[60]))
    ensemble_params = {
        "methods": ["ewma", "rolling_std"],
        "lookback": 60,
    }

    with patch(
        "mtdata.forecast.backtest._fetch_history",
        return_value=frame,
    ) as fetch, patch(
        "mtdata.forecast.backtest.forecast_volatility",
        return_value={"volatility_horizon": 0.02},
    ) as volatility:
        result = forecast_backtest(
            symbol="BTCUSD",
            timeframe="H1",
            horizon=3,
            methods=["ensemble"],
            params_per_method={"ensemble": ensemble_params},
            anchors=[anchor],
            quantity="volatility",
        )

    fetch.assert_called_once()
    assert result["success"] is True
    assert volatility.call_args.kwargs["params"] == ensemble_params


def test_backtest_har_rv_ignores_other_methods_nested_lookback_guard() -> None:
    times = np.arange(1699999980, 1699999980 + 70 * 3600, 3600, dtype=float)
    frame = pd.DataFrame(
        {
            "time": times,
            "open": np.linspace(100.0, 120.0, 70, dtype=float),
            "close": np.linspace(100.0, 120.0, 70, dtype=float),
        }
    )
    anchor = _format_time_minimal(float(times[60]))

    with patch(
        "mtdata.forecast.backtest._fetch_history",
        return_value=frame,
    ) as fetch, patch(
        "mtdata.forecast.backtest.forecast_volatility",
        return_value={"volatility_horizon": 0.02},
    ) as volatility:
        result = forecast_backtest(
            symbol="BTCUSD",
            timeframe="H1",
            horizon=3,
            methods=["har_rv", "ewma"],
            params_per_method={"ewma": {"lookback": 60}},
            anchors=[anchor],
            quantity="volatility",
        )

    fetch.assert_called_once()
    assert result["success"] is True
    assert result.get("error_code") != "har_rv_lookback_unsupported"
    assert [call.kwargs["params"] for call in volatility.call_args_list] == [
        {},
        {"lookback": 60},
    ]


def test_backtest_har_rv_explicit_anchors_expose_bounded_fit_windows() -> None:
    h1_start = datetime(2024, 3, 1, tzinfo=timezone.utc).timestamp()
    h1_times = h1_start + np.arange(100, dtype=float) * 3600.0
    h1_close = 100.0 * np.exp(
        np.cumsum(0.0002 + 0.0001 * np.sin(np.arange(100, dtype=float)))
    )
    h1 = pd.DataFrame(
        {
            "time": h1_times,
            "open": np.concatenate(([100.0], h1_close[:-1])),
            "close": h1_close,
        }
    )
    anchors = [
        format_epoch_utc(float(h1_times[index]), timespec="seconds")
        for index in (60, 72)
    ]
    assert all(anchor is not None for anchor in anchors)

    m5_start = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
    m5_times = m5_start + np.arange(70 * 24 * 12, dtype=float) * 300.0
    m5_returns = (
        0.00005
        + 0.0004 * np.sin(np.arange(len(m5_times), dtype=float) / 17.0)
        + 0.0002 * np.cos(np.arange(len(m5_times), dtype=float) / 31.0)
    )
    m5_close = 100.0 * np.exp(np.cumsum(m5_returns))
    m5 = pd.DataFrame(
        {
            "time": m5_times,
            "open": np.concatenate(([100.0], m5_close[:-1])),
            "high": np.maximum(
                np.concatenate(([100.0], m5_close[:-1])),
                m5_close,
            )
            * 1.0001,
            "low": np.minimum(
                np.concatenate(([100.0], m5_close[:-1])),
                m5_close,
            )
            * 0.9999,
            "close": m5_close,
            "tick_volume": np.full(len(m5_times), 1000),
        }
    )

    def fetch_rates(
        _symbol,
        _mt5_timeframe,
        _count,
        *,
        timeframe,
        **_kwargs,
    ):
        return (m5, None) if timeframe == "M5" else (h1, None)

    with patch(
        "mtdata.forecast.backtest._fetch_history",
        return_value=h1,
    ), patch(
        "mtdata.forecast.volatility._fetch_mt5_rates_guarded",
        side_effect=fetch_rates,
    ):
        result = forecast_backtest(
            symbol="BTCUSD",
            timeframe="H1",
            horizon=3,
            start="2024-01-01T00:00:00Z",
            methods=["har_rv"],
            params_per_method={
                "har_rv": {
                    "days": 40,
                    "rv_timeframe": "M5",
                    "window_w": 3,
                    "window_m": 5,
                }
            },
            anchors=anchors,
            quantity="volatility",
            detail="full",
        )

    assert result["success"] is True
    assert result["backtest_plan"]["model"] == (
        "rolling_origin_method_specific_window"
    )
    details = result["results"]["har_rv"]["details"]
    assert len(details) == 2
    for detail in details:
        anchor_epoch = datetime.fromisoformat(
            detail["anchor"].replace("Z", "+00:00")
        ).timestamp()
        cutoff_epoch = bar_close_epoch(anchor_epoch, "H1")
        start_epoch = cutoff_epoch - 40 * 86400
        assert bool(detail["success"]) is True
        assert detail["training_bars_used"] == 40 * 24 * 12
        assert detail["training_window"] == {
            "start": _format_time_minimal(start_epoch),
            "end": _format_time_minimal(cutoff_epoch - 300),
        }
        params_used = detail["params_used"]
        assert params_used["days"] == 40
        assert params_used["days_semantics"] == (
            "maximum_trailing_calendar_days"
        )
        assert params_used["history_cutoff_epoch"] == cutoff_epoch
        assert params_used["history_start_bound_epoch"] == start_epoch
        assert params_used["history_window_policy"] == (
            "trailing_calendar_days_intersect_requested_start"
        )


def test_execute_entrypoint_preserves_har_rv_lookback_failure() -> None:
    failure = {
        "success": False,
        "error": "HAR-RV does not accept lookback.",
        "error_code": "har_rv_lookback_unsupported",
        "remediation": "Remove lookback and use params.days.",
    }
    with patch(
        "mtdata.forecast.backtest.forecast_backtest",
        return_value=failure,
    ):
        result = execute_forecast_backtest(symbol="BTCUSD")

    assert result == failure


def test_execute_entrypoint_preserves_har_rv_ensemble_lookback_failure() -> None:
    with patch("mtdata.forecast.backtest._fetch_history") as fetch:
        result = execute_forecast_backtest(
            symbol="BTCUSD",
            timeframe="H1",
            horizon=3,
            methods=["ensemble"],
            params_per_method={
                "ensemble": {
                    "methods": ["ewma", "har_rv"],
                    "method_params": {"har_rv": {"lookback": None}},
                },
            },
            quantity="volatility",
        )

    assert result["success"] is False
    assert result["error_code"] == "har_rv_lookback_unsupported"
    assert "params.days" in result["error"]
    fetch.assert_not_called()


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
            symbol="BTCUSD",
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
            symbol="BTCUSD",
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


@pytest.mark.parametrize(
    ("quantity", "entry_open", "forecast_value", "realized_close", "position"),
    [
        ("price", 102.0, 101.0, 103.0, "long"),
        ("price", 98.0, 99.0, 97.0, "short"),
        ("return", 102.0, float(np.log(1.01)), 103.0, "long"),
        ("return", 98.0, float(np.log(0.99)), 97.0, "short"),
    ],
)
def test_backtest_fills_gap_through_targets_at_entry_open(
    quantity: str,
    entry_open: float,
    forecast_value: float,
    realized_close: float,
    position: str,
) -> None:
    times = np.arange(1_699_999_980, 1_699_999_980 + 80 * 3600, 3600, dtype=float)
    close = np.full(80, 100.0)
    open_ = close.copy()
    idx = 70
    open_[idx + 1] = entry_open
    close[idx + 1] = realized_close
    frame = pd.DataFrame({"time": times, "open": open_, "close": close})
    anchor = _format_time_minimal(float(times[idx]))
    forecast_key = "forecast_return" if quantity == "return" else "forecast_price"

    with patch("mtdata.forecast.backtest._fetch_history", return_value=frame), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={forecast_key: [forecast_value]},
    ):
        result = forecast_backtest(
            symbol="BTCUSD",
            timeframe="H1",
            horizon=1,
            methods=["naive"],
            anchors=[anchor],
            quantity=quantity,
        )

    detail = result["results"]["naive"]["details"][0]
    assert detail["position"] == position
    assert detail["entry_price"] == entry_open
    assert detail["exit_price"] == entry_open
    assert detail["exit_price_source"] == "entry_open_target_price_improvement"
    assert detail["exit_step"] == 1
    assert detail["trade_return_gross"] == 0.0
    assert result["execution_policy"]["marketable_at_entry_fill"] == "entry_open"


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
            symbol="BTCUSD",
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
    assert slow_detail["exit_price"] == 141.2
    assert slow_detail["exit_price_source"] == "forecast_target"
    assert aggressive_detail["exit_price_source"] == "horizon_close"
    assert float(slow_detail["trade_return"]) != float(aggressive_detail["trade_return"])
    assert int(slow_detail["exit_step"]) < int(aggressive_detail["exit_step"])


def test_performance_metrics_annualize_over_full_evaluation_window() -> None:
    returns = [0.001, -0.0005] * 20
    metrics = _compute_performance_metrics(
        returns=returns,
        timeframe="H1",
        horizon=12,
        slippage_bps=0.0,
        trade_spacing_bars=10,
        symbol="BTCUSD",
        evaluation_bars=3000,
    )

    assert metrics["annualization_method"] == "evaluation_duration"
    assert metrics["evaluation_years"] == pytest.approx(3000 / 8760)
    assert metrics["trades_per_year"] == pytest.approx(40 / (3000 / 8760))


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
