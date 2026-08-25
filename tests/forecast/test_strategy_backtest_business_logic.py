from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from mtdata.core import forecast as core_forecast
from mtdata.forecast import backtest as forecast_backtest
from mtdata.forecast.requests import StrategyBacktestRequest


def _unwrap(fn):
    current = fn
    while hasattr(current, "__wrapped__"):
        current = current.__wrapped__
    return current


def _history_from_closes(
    closes: list[float],
    *,
    spread_points: float | None = None,
) -> pd.DataFrame:
    rows = []
    for index, close in enumerate(closes):
        open_price = closes[index - 1] if index > 0 else close
        row = {
            "time": 1700000000.0 + (index * 3600.0),
            "open": float(open_price),
            "high": float(max(open_price, close)),
            "low": float(min(open_price, close)),
            "close": float(close),
        }
        if spread_points is not None:
            row["spread"] = float(spread_points)
        rows.append(row)
    return pd.DataFrame(rows)


def test_strategy_backtest_fails_before_evaluation_without_cost_inputs(monkeypatch):
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda symbol, timeframe, need, as_of=None: _history_from_closes(
            [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ),
    )
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info_tick",
        lambda _symbol: type("Tick", (), {"bid": 1.0999, "ask": 1.1001})(),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        timeframe="H1",
        strategy="sma_cross",
        lookback=8,
        fast_period=2,
        slow_period=3,
        detail="full",
        cost_model="historical_bar_spread",
    )

    assert out["success"] is False
    assert out["error_code"] == "cost_model_unavailable"
    assert out["cost_model"]["coverage_pct"] == 0.0
    assert out["cost_model"]["complete"] is False
    assert out["suggested_fixed_spread_bps"] == pytest.approx(1.8182)
    assert out["suggestion_basis"] == "current_bid_ask_snapshot"
    assert "--cost-model fixed --spread-bps 1.8182" in out["remediation"]
    assert "summary" not in out
    assert "metrics" not in out
    assert "trades" not in out


def test_strategy_backtest_explicit_historical_bar_spread(monkeypatch):
    history = _history_from_closes(
        [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        spread_points=10.0,
    )
    monkeypatch.setattr(forecast_backtest, "_fetch_history", lambda *args, **kwargs: history)
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info",
        lambda _symbol: type("Info", (), {"point": 0.0001})(),
    )
    historical = forecast_backtest.strategy_backtest(
        symbol="EURUSD", lookback=8, fast_period=2, slow_period=3, detail="full",
        cost_model="historical_bar_spread",
    )
    fixed = forecast_backtest.strategy_backtest(
        symbol="EURUSD", lookback=8, fast_period=2, slow_period=3, detail="full",
        cost_model="fixed", spread_bps=0.0,
    )

    assert historical["cost_model"]["type"] == "historical_bar_spread"
    assert historical["cost_model"]["requested_type"] == "historical_bar_spread"
    assert historical["cost_model"]["spread_source"] == "mt5_historical_bar_spread"
    assert historical["cost_model"]["historical_spread_coverage_pct"] == 100.0
    assert historical["cost_model"]["spread_observations"] == 1
    assert historical["cost_model"]["spread_bps_round_trip"] == pytest.approx(
        historical["trades"][0]["spread_cost_bps"]
    )
    assert historical["cost_model"]["complete"] is True
    assert "warnings" not in historical
    assert historical["summary"]["net_return"] < fixed["summary"]["net_return"]
    assert historical["trades"][0]["spread_cost_status"] == "included"
    assert "return_net" in historical["trades"][0]
    assert "return_after_known_costs" not in historical["trades"][0]


def test_strategy_backtest_strict_historical_rejects_zero_spread_samples(monkeypatch):
    history = _history_from_closes(
        [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        spread_points=0.0,
    )
    monkeypatch.setattr(forecast_backtest, "_fetch_history", lambda *args, **kwargs: history)
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info",
        lambda _symbol: type("Info", (), {"point": 0.0001})(),
    )
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info_tick",
        lambda _symbol: None,
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        lookback=8,
        fast_period=2,
        slow_period=3,
        detail="full",
        cost_model="historical_bar_spread",
    )

    assert out["success"] is False
    assert out["error_code"] == "cost_model_unavailable"
    assert out["cost_model"]["source"] == "unavailable"
    assert out["cost_model"]["coverage_pct"] == 0.0
    assert out["cost_model"]["complete"] is False
    assert "metrics" not in out


def test_strategy_backtest_default_uses_historical_bar_spread(monkeypatch):
    history = _history_from_closes(
        [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        spread_points=10.0,
    )
    monkeypatch.setattr(forecast_backtest, "_fetch_history", lambda *args, **kwargs: history)
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info",
        lambda _symbol: type("Info", (), {"point": 0.0001})(),
    )
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info_tick",
        lambda _symbol: pytest.fail("default backtest must not read a live quote"),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        lookback=8,
        fast_period=2,
        slow_period=3,
        detail="full",
    )

    assert out["success"] is True
    assert out["cost_model"]["requested_type"] == "auto"
    assert out["cost_model"]["type"] == "historical_bar_spread"
    assert out["cost_model"]["spread_source"] == "mt5_historical_bar_spread"
    assert out["cost_model"]["selection_reason"] == "complete_historical_spread_coverage"
    assert out["cost_model"]["complete"] is True
    assert "spread_bps" not in out["parameters"]
    assert out["metrics"]["metrics_reliability"] in {"low", "medium", "high"}


def test_strategy_backtest_default_auto_falls_back_to_conservative_fixed(
    monkeypatch,
):
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda *args, **kwargs: _history_from_closes(
            [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ),
    )
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info_tick",
        lambda _symbol: type("Tick", (), {"bid": 1.0999, "ask": 1.1001})(),
    )
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info",
        lambda _symbol: type("Info", (), {"point": 0.0001, "spread": 0, "bid": 0, "ask": 0})(),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        lookback=8,
        fast_period=2,
        slow_period=3,
        detail="full",
    )

    assert out["success"] is True
    assert out["cost_model"]["requested_type"] == "auto"
    assert out["cost_model"]["type"] == "fixed"
    assert out["cost_model"]["spread_source"] == "current_bid_ask_snapshot"
    assert out["cost_model"]["selection_reason"] == (
        "incomplete_historical_spread_coverage"
    )
    assert out["cost_model"]["complete"] is True
    assert out["cost_model"]["spread_bps_round_trip"] == pytest.approx(1.8182)
    assert any("conservative fixed spread" in warning for warning in out["warnings"])
    assert "summary" in out


def test_strategy_backtest_auto_fails_without_any_spread_estimate(
    monkeypatch,
):
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda *args, **kwargs: _history_from_closes(
            [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ),
    )
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info_tick",
        lambda _symbol: None,
    )
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info",
        lambda _symbol: None,
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        lookback=8,
        fast_period=2,
        slow_period=3,
        cost_model="auto",
    )

    assert out["error_code"] == "cost_model_unavailable"
    assert out["cost_model"]["requested_type"] == "auto"
    assert out["cost_model"]["source"] == "unavailable"
    assert out["cost_model"]["complete"] is False
    assert "--cost-model fixed --spread-bps" in out["remediation"]


def test_strategy_backtest_fixed_model_requires_explicit_spread_before_io(
    monkeypatch,
):
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda *_args, **_kwargs: pytest.fail("invalid cost model must fail before I/O"),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        cost_model="fixed",
    )

    assert out["success"] is False
    assert out["error_code"] == "invalid_cost_model"
    assert out["cost_model"] == {
        "requested_type": "fixed",
        "source": "missing_explicit_spread",
        "complete": False,
    }
    assert "--spread-bps is required" in out["error"]


def test_strategy_backtest_validates_symbol_before_cost_resolution(monkeypatch):
    def missing_symbol(*_args, **_kwargs):
        raise RuntimeError("Symbol 'NO_SUCH_SYMBOL' was not found in MT5.")

    tick_lookup = pytest.fail
    monkeypatch.setattr(forecast_backtest, "_fetch_history", missing_symbol)
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info_tick",
        lambda _symbol: tick_lookup("spread lookup must not run first"),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="NO_SUCH_SYMBOL",
        lookback=8,
        fast_period=2,
        slow_period=3,
    )

    assert out == {"error": "Symbol 'NO_SUCH_SYMBOL' was not found in MT5."}


def test_strategy_backtest_includes_first_valid_warmup_signal(monkeypatch):
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda *args, **kwargs: _history_from_closes(
            [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        timeframe="H1",
        strategy="sma_cross",
        lookback=10,
        fast_period=2,
        slow_period=3,
        detail="full",
        cost_model="fixed",
        spread_bps=0.0,
        position_mode="long_short",
    )

    assert out["success"] is True
    assert out["trades"][0]["direction"] == "long"


def test_strategy_backtest_compact_mode_excludes_trades(monkeypatch):
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda symbol, timeframe, need, as_of=None: _history_from_closes(
            [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        timeframe="H1",
        strategy="sma_cross",
        lookback=8,
        fast_period=2,
        slow_period=3,
        detail="compact",
        cost_model="fixed",
        spread_bps=0.0,
    )

    assert out["success"] is True
    assert out["summary"]["num_trades"] == 1
    assert out["summary"]["sample_status"] == "insufficient_trades"
    assert out["summary"]["minimum_trades"] == 30
    assert out["is_signal"] is False
    assert out["usage"] == "research_only"
    assert "usable_for_live_trading" not in out
    assert out["price_basis"] == "broker_chart_price"
    assert out["cost_model"] == {
        "type": "fixed",
        "requested_type": "fixed",
        "spread_bps_round_trip": 0.0,
        "spread_source": "explicit",
        "spread_observations": 1,
        "historical_priced_trades": 0,
        "unpriced_trades": 0,
        "priced_trade_coverage_pct": 100.0,
        "slippage_bps_per_side": 1.0,
        "round_trip_cost_bps": 2.0,
        "complete": True,
    }
    assert StrategyBacktestRequest(symbol="EURUSD").slippage_bps == 1.0
    assert StrategyBacktestRequest(symbol="EURUSD").cost_model == "auto"
    assert out["signal_status"] == "not_actionable"
    assert "last_signal" not in out
    assert out["last_historical_signal"]["signal_status"] == "historical_observation_only"
    assert out["last_historical_signal"]["direction"] == "long"
    assert "signal" not in out["last_historical_signal"]
    assert out["summary"]["costs_complete"] is True
    assert out["summary"]["cost_coverage_pct"] == 100.0
    assert "net_return" in out["summary"]
    assert out["metrics"]["trades_observed"] == 1
    assert out["units"]["returns"] == "return_fraction"
    assert "avg_directional_accuracy" not in out["units"]
    assert len(out["units"]) < len(forecast_backtest._backtest_units())
    assert "trades" not in out, "compact mode should not include trades array"
    assert "trade_sample" not in out


def test_strategy_backtest_compact_explains_low_trade_sample(monkeypatch):
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda *args, **kwargs: _history_from_closes(
            [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        lookback=8,
        fast_period=2,
        slow_period=3,
        detail="compact",
        cost_model="fixed",
        spread_bps=1.0,
    )

    assert out["summary"]["sample_status"] == "insufficient_trades"
    assert out["summary"]["minimum_trades"] == 30
    assert out["sample_guidance"]["code"] == "insufficient_trades"
    assert "Increase lookback" in out["sample_guidance"]["recommended_action"]


def test_strategy_backtest_uses_date_range_when_provided(monkeypatch):
    calls = []
    history = _history_from_closes(
        [1.0 + value / 10.0 for value in range(15)],
        spread_points=10.0,
    )

    def fake_fetch_history(symbol, timeframe, need, **kwargs):
        calls.append((need, kwargs))
        if kwargs.get("as_of"):
            return history.iloc[:5].reset_index(drop=True)
        return history.iloc[5:].reset_index(drop=True)

    monkeypatch.setattr(forecast_backtest, "_fetch_history", fake_fetch_history)
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info",
        lambda _symbol: type("Info", (), {"point": 0.0001})(),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        timeframe="H1",
        strategy="sma_cross",
        lookback=5,
        start="2023-01-01",
        end="2023-12-31",
        fast_period=2,
        slow_period=5,
        detail="full",
        cost_model="historical_bar_spread",
    )

    assert out["success"] is True
    assert out["summary"]["bars_used"] == 10
    assert out["summary"]["warmup_history_bars"] == 5
    assert out["summary"]["signal_bars"] == 10
    assert out["summary"]["evaluation_start"] == (
        forecast_backtest._format_time_minimal(float(history["time"].iloc[5]))
    )
    assert calls[0][1]["start"] == "2023-01-01"
    assert calls[0][1]["end"] == "2023-12-31"
    assert calls[1][0] == 5
    assert calls[1][1]["as_of"] == pd.Timestamp(
        float(history["time"].iloc[5]), unit="s", tz="UTC"
    ).isoformat()
    assert all(
        trade["entry_time"] >= out["summary"]["evaluation_start"]
        for trade in out.get("trades", [])
    )
    assert out["parameters"]["start"] == "2023-01-01"
    assert out["parameters"]["end"] == "2023-12-31"


def test_strategy_backtest_rejects_range_without_prestart_warmup(monkeypatch):
    history = _history_from_closes([float(value) for value in range(1, 16)])

    def fake_fetch_history(symbol, timeframe, need, **kwargs):
        if kwargs.get("as_of"):
            return history.iloc[:2].reset_index(drop=True)
        return history.iloc[5:].reset_index(drop=True)

    monkeypatch.setattr(forecast_backtest, "_fetch_history", fake_fetch_history)

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        start="2023-01-01",
        end="2023-12-31",
        lookback=5,
        fast_period=2,
        slow_period=5,
        cost_model="fixed",
        spread_bps=1.0,
    )

    assert out["success"] is False
    assert out["error_code"] == "insufficient_warmup_history"
    assert out["warmup_bars_required"] == 5
    assert out["warmup_bars_available"] == 2


@pytest.mark.parametrize(
    ("strategy", "strategy_kwargs"),
    [
        ("sma_cross", {"fast_period": 2, "slow_period": 5}),
        ("ema_cross", {"fast_period": 2, "slow_period": 5}),
        ("rsi_reversion", {"rsi_length": 4, "oversold": 40.0, "overbought": 60.0}),
    ],
)
def test_range_results_match_same_prefetched_history(
    monkeypatch,
    strategy,
    strategy_kwargs,
):
    closes = [100.0, 90.0, 80.0, 90.0, 100.0] * 5
    history = _history_from_closes(closes)
    warmup_bars = 5
    evaluation = history.iloc[warmup_bars:].reset_index(drop=True)
    prehistory = history.iloc[:warmup_bars].reset_index(drop=True)

    def ranged_fetch(symbol, timeframe, need, **kwargs):
        return prehistory if kwargs.get("as_of") else evaluation

    monkeypatch.setattr(forecast_backtest, "_fetch_history", ranged_fetch)
    ranged = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        strategy=strategy,
        lookback=20,
        start="2023-01-01",
        end="2023-12-31",
        cost_model="fixed",
        spread_bps=1.0,
        slippage_bps=0.0,
        detail="full",
        **strategy_kwargs,
    )

    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda *args, **kwargs: history,
    )
    prefetched = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        strategy=strategy,
        lookback=20,
        cost_model="fixed",
        spread_bps=1.0,
        slippage_bps=0.0,
        detail="full",
        **strategy_kwargs,
    )

    trade_fields = ("direction", "entry_time", "exit_time", "exit_reason")
    assert ranged["summary"]["num_trades"] > 0
    assert [
        tuple(trade[field] for field in trade_fields) for trade in ranged["trades"]
    ] == [
        tuple(trade[field] for field in trade_fields)
        for trade in prefetched["trades"]
    ]
    assert ranged["summary"]["gross_return"] == pytest.approx(
        prefetched["summary"]["gross_return"]
    )
    assert ranged["summary"]["net_return"] == pytest.approx(
        prefetched["summary"]["net_return"]
    )


def test_end_of_data_exit_uses_final_bar_close_across_outputs(monkeypatch):
    history = _history_from_closes([1.0 + value / 100.0 for value in range(10)])
    first_open = datetime(2026, 1, 30, 8, tzinfo=timezone.utc).timestamp()
    history["time"] = [first_open + index * 14_400 for index in range(len(history))]
    persistent = pd.Series([1.0] * len(history))
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda *args, **kwargs: history,
    )
    monkeypatch.setattr(
        forecast_backtest,
        "_build_strategy_signal_series",
        lambda *args, **kwargs: (persistent, {}, 1),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        timeframe="H4",
        lookback=5,
        fast_period=2,
        slow_period=3,
        cost_model="fixed",
        spread_bps=0.0,
        slippage_bps=0.0,
        detail="full",
    )

    expected_exit = forecast_backtest._format_time_minimal(
        float(history["time"].iloc[-1]) + 14_400.0
    )
    trade = out["trades"][0]
    assert trade["exit_reason"] == "end_of_data"
    assert trade["exit_time_basis"] == "bar_close_time"
    assert trade["exit_time"] == expected_exit
    assert trade["bars_held"] == 5
    assert out["equity_curve"][-1]["time"] == expected_exit
    assert out["monthly_breakdown"][0]["month"] == "2026-02"


def test_max_hold_waits_for_fresh_signal_before_same_direction_reentry(monkeypatch):
    history = _history_from_closes([1.0 + value / 100.0 for value in range(15)])
    persistent = pd.Series([1.0] * len(history))
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda *args, **kwargs: history,
    )
    monkeypatch.setattr(
        forecast_backtest,
        "_build_strategy_signal_series",
        lambda *args, **kwargs: (persistent, {}, 1),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        lookback=12,
        fast_period=2,
        slow_period=3,
        max_hold_bars=3,
        cost_model="fixed",
        spread_bps=1.0,
        slippage_bps=1.0,
        detail="full",
    )

    assert out["summary"]["num_trades"] == 1
    assert out["trades"][0]["exit_reason"] == "max_hold"
    assert out["trades"][0]["bars_held"] == 3
    assert out["summary"]["max_hold_reentry_policy"] == "fresh_signal_required"
    assert out["summary"]["longest_continuous_exposure_bars"] == 3
    assert out["cost_model"]["spread_observations"] == 1


def test_max_hold_allows_opposite_signal_at_boundary(monkeypatch):
    history = _history_from_closes([1.0 + value / 100.0 for value in range(12)])
    signals = pd.Series([1.0, 1.0, 1.0, -1.0] + [-1.0] * 8)
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda *args, **kwargs: history,
    )
    monkeypatch.setattr(
        forecast_backtest,
        "_build_strategy_signal_series",
        lambda *args, **kwargs: (signals, {}, 1),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        lookback=12,
        fast_period=2,
        slow_period=3,
        max_hold_bars=3,
        cost_model="fixed",
        spread_bps=0.0,
        slippage_bps=0.0,
        detail="full",
    )

    assert [trade["direction"] for trade in out["trades"]] == ["long", "short"]
    assert out["trades"][0]["exit_reason"] == "signal_reversal"
    assert out["trades"][0]["exit_time"] == out["trades"][1]["entry_time"]


def test_strategy_backtest_exposes_request_metadata_blocks(monkeypatch):
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda symbol, timeframe, need, as_of=None: _history_from_closes(
            [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        timeframe="H1",
        strategy="SMA_CROSS",  # type: ignore[arg-type]
        lookback="8",  # type: ignore[arg-type]
        fast_period="2",  # type: ignore[arg-type]
        slow_period="3",  # type: ignore[arg-type]
        detail="FULL",  # type: ignore[arg-type]
        position_mode="LONG_SHORT",  # type: ignore[arg-type]
        slippage_bps=1.5,
        cost_model="fixed",
        spread_bps=0.0,
    )

    assert out["request"]["detail"] == "FULL"
    assert out["request"]["strategy"] == "SMA_CROSS"
    assert out["request"]["slippage_bps"] == 1.5
    assert out["resolved_request"]["detail"] == "full"
    assert out["resolved_request"]["strategy"] == "sma_cross"
    assert out["resolved_request"]["position_mode"] == "long_short"
    assert out["resolved_request"]["lookback"] == 8
    assert out["resolved_request"]["slippage_bps"] == 1.5
    assert out["parameters"]["slippage_bps"] == 1.5
    strategy_params = out["contracts"]["strategy"]["parameters"]
    assert strategy_params["fast_period"] == 2
    assert strategy_params["slow_period"] == 3
    assert "rsi_length" not in strategy_params
    assert "oversold" not in strategy_params
    assert "overbought" not in strategy_params
    assert out["contracts"]["data_preparation"]["symbol"] == "EURUSD"
    assert out["contracts"]["evaluation"]["detail"] == "full"
    assert out["contracts"]["strategy"]["kind"] == "indicator_strategy"
    assert out["contracts"]["strategy"]["position_mode"] == "long_short"


def test_strategy_backtest_returns_no_action_on_flat_history(monkeypatch):
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda symbol, timeframe, need, as_of=None: _history_from_closes([1.0] * 40),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        timeframe="H1",
        strategy="sma_cross",
        lookback=30,
        fast_period=2,
        slow_period=5,
        cost_model="fixed",
        spread_bps=0.0,
    )

    assert out["success"] is True
    assert out["no_action"] is True
    assert out["summary"]["num_trades"] == 0
    assert out["message"] == "The strategy generated no trades on the requested history."


def test_strategy_backtest_long_only_signal_suppresses_shorts_and_warmup_nan():
    df = _history_from_closes([5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0])

    long_short_signal, _diagnostics, _warmup = forecast_backtest._build_strategy_signal_series(
        df,
        strategy="sma_cross",
        position_mode="long_short",
        fast_period=2,
        slow_period=3,
        rsi_length=14,
        oversold=30.0,
        overbought=70.0,
    )
    long_only_signal, _diagnostics, warmup = forecast_backtest._build_strategy_signal_series(
        df,
        strategy="sma_cross",
        position_mode="long_only",
        fast_period=2,
        slow_period=3,
        rsi_length=14,
        oversold=30.0,
        overbought=70.0,
    )

    assert long_short_signal.isna().any()
    assert (long_short_signal == -1.0).any()
    assert not long_only_signal.isna().any()
    assert (long_only_signal >= 0.0).all()
    assert long_only_signal.iloc[:warmup].eq(0.0).all()


def test_strategy_backtest_request_allows_rsi_reversion_without_ma_constraint():
    request = StrategyBacktestRequest(
        symbol="EURUSD",
        strategy="rsi_reversion",
        fast_period=30,
        slow_period=10,
    )

    assert request.strategy == "rsi_reversion"


def test_core_strategy_backtest_wrapper_routes_request(monkeypatch):
    raw = _unwrap(core_forecast.strategy_backtest)
    monkeypatch.setattr(core_forecast, "ensure_mt5_connection_or_raise", lambda: None)
    monkeypatch.setattr(
        core_forecast,
        "_strategy_backtest_impl",
        lambda **kwargs: {
            "ok": True,
            "strategy": kwargs["strategy"],
            "symbol": kwargs["symbol"],
            "start": kwargs["start"],
            "end": kwargs["end"],
        },
    )

    out = raw(
        request=StrategyBacktestRequest(
            symbol="EURUSD",
            strategy="ema_cross",
            lookback=50,
            start="2023-01-01",
            end="2023-12-31",
        )
    )

    assert out["ok"] is True
    assert out["strategy"] == "ema_cross"
    assert out["symbol"] == "EURUSD"
    assert out["start"] == "2023-01-01"
    assert out["end"] == "2023-12-31"


def test_strategy_backtest_request_rejects_invalid_ma_periods():
    with pytest.raises(ValueError, match="fast_period must be less than slow_period"):
        StrategyBacktestRequest(
            symbol="EURUSD",
            strategy="sma_cross",
            fast_period=20,
            slow_period=10,
        )


def test_strategy_backtest_request_rejects_spread_with_historical_model():
    with pytest.raises(ValueError, match="--spread-bps is only valid"):
        StrategyBacktestRequest(
            symbol="EURUSD",
            cost_model="historical_bar_spread",
            spread_bps=1.0,
        )


def test_strategy_backtest_request_rejects_spread_with_auto_model():
    with pytest.raises(ValueError, match="--spread-bps is only valid"):
        StrategyBacktestRequest(
            symbol="EURUSD",
            cost_model="auto",
            spread_bps=1.0,
        )


def test_strategy_backtest_auto_uses_historical_p75_when_coverage_is_partial(
    monkeypatch,
):
    history = _history_from_closes(
        [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        spread_points=10.0,
    )
    history.loc[0:4, "spread"] = 0.0
    monkeypatch.setattr(forecast_backtest, "_fetch_history", lambda *args, **kwargs: history)
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info",
        lambda _symbol: type("Info", (), {"point": 0.0001, "spread": 0, "bid": 0, "ask": 0})(),
    )
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info_tick",
        lambda _symbol: type("Tick", (), {"bid": 1.0999, "ask": 1.1001})(),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        lookback=8,
        fast_period=2,
        slow_period=3,
        detail="full",
        cost_model="auto",
    )

    assert out["success"] is True
    assert out["cost_model"]["requested_type"] == "auto"
    assert out["cost_model"]["type"] == "fixed"
    assert out["cost_model"]["spread_source"] in {
        "mt5_historical_bar_spread_p75",
        "current_bid_ask_snapshot",
    }
    assert out["cost_model"]["historical_bar_spread_coverage_pct"] < 100.0
    assert out["cost_model"]["complete"] is True
    assert out["cost_model"]["spread_bps_round_trip"] > 0.0


def test_strategy_backtest_request_accepts_auto_cost_model():
    request = StrategyBacktestRequest(symbol="EURUSD", cost_model="auto")

    assert request.cost_model == "auto"
    assert request.spread_bps is None


def test_strategy_backtest_request_requires_explicit_fixed_spread():
    with pytest.raises(ValueError, match="--spread-bps is required"):
        StrategyBacktestRequest(symbol="EURUSD", cost_model="fixed")


def test_strategy_backtest_request_allows_explicit_fixed_spread():
    request = StrategyBacktestRequest(
        symbol="EURUSD",
        cost_model="fixed",
        spread_bps=1.2,
    )

    assert request.spread_bps == 1.2
