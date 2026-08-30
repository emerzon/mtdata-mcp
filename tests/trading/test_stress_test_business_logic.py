from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mtdata.core.trading.requests import TradeStressTestRequest
from mtdata.core.trading.use_cases import run_trade_stress_test


class _Gateway:
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1

    def ensure_connection(self) -> None:
        return None

    def account_info(self):
        return SimpleNamespace(equity=10_000.0, currency="USD")

    def positions_get(self):
        return [
            SimpleNamespace(
                ticket=1,
                symbol="EURUSD",
                type=0,
                volume=1.0,
                price_current=1.1000,
                price_open=1.0900,
            ),
            SimpleNamespace(
                ticket=2,
                symbol="EURUSD",
                type=1,
                volume=0.5,
                price_current=1.1000,
                price_open=1.1200,
            ),
        ]

    def symbol_info(self, symbol):
        return SimpleNamespace(
            trade_tick_size=0.0001,
            trade_tick_value=10.0,
            trade_tick_value_profit=10.0,
            trade_tick_value_loss=10.0,
            point=0.0001,
        )

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=1.0999, ask=1.1001, time=1)


def test_trade_stress_test_rejects_total_loss_shock() -> None:
    with pytest.raises(ValidationError, match="greater than -100"):
        TradeStressTestRequest(shocks={"EURUSD": -100.0})


def test_trade_stress_test_schema_publishes_exclusive_minimum() -> None:
    schema = TradeStressTestRequest.model_json_schema()
    additional = schema["properties"]["shocks"]["additionalProperties"]
    assert additional["exclusiveMinimum"] == -100


def test_trade_stress_test_offsets_long_and_short_positions():
    result = run_trade_stress_test(
        TradeStressTestRequest(shocks={"EURUSD": -1.0}, detail="full"),
        gateway=_Gateway(),
    )

    assert result["success"] is True
    assert result["positions_evaluated"] == 2
    assert result["total_pnl_impact"] == -550.0
    assert result["equity_after"] == 9450.0
    assert result["mark_freshness_status"] == "stale_or_unverified"
    assert result["mark_usability_status"] == "not_live_ready"
    assert "usable_for_live_trading" not in result
    assert result["data_stale"] is True
    assert result["valuation_time"] == "1970-01-01T00:00:01Z"
    assert {item["symbol"] for item in result["mark_freshness"]} == {"EURUSD"}


def test_trade_stress_test_preserves_broker_symbol_case():
    gateway = _Gateway()
    gateway.positions_get = lambda: [
        SimpleNamespace(
            ticket=3,
            symbol="EURUSD.pro",
            type=0,
            volume=1.0,
            price_current=1.1,
            price_open=1.09,
        )
    ]
    looked_up = []
    original_symbol_info = gateway.symbol_info

    def symbol_info(symbol):
        looked_up.append(symbol)
        return original_symbol_info(symbol)

    gateway.symbol_info = symbol_info

    result = run_trade_stress_test(
        TradeStressTestRequest(shocks={"eurusd.PRO": -1.0}),
        gateway=gateway,
    )

    assert result["success"] is True
    assert result["items"][0]["symbol"] == "EURUSD.pro"
    assert looked_up == ["EURUSD.pro"]


def test_trade_stress_test_labels_entry_price_fallback_as_non_live():
    gateway = _Gateway()
    gateway.positions_get = lambda: [
        SimpleNamespace(
            ticket=3,
            symbol="EURUSD",
            type=0,
            volume=1.0,
            price_current=0.0,
            price_open=1.1,
        )
    ]
    gateway.symbol_info_tick = lambda _symbol: SimpleNamespace(
        bid=1.0999, ask=1.1001, time=4_102_444_800
    )

    result = run_trade_stress_test(
        TradeStressTestRequest(shocks={"EURUSD": -1.0}), gateway=gateway
    )

    assert result["items"][0]["valuation_basis"] == "entry_price_fallback"
    assert result["mark_freshness_status"] == "entry_price_fallback"
    assert result["valuation_basis"] == "entry_price_fallback"
    assert result["mark_usability_status"] == "not_live_ready"
    assert "usable_for_live_trading" not in result


def test_trade_stress_test_names_locked_quote_as_usability_blocker():
    gateway = _Gateway()
    gateway.symbol_info_tick = lambda _symbol: SimpleNamespace(
        bid=1.1,
        ask=1.1,
        time=int(time.time()),
    )

    result = run_trade_stress_test(
        TradeStressTestRequest(shocks={"EURUSD": -1.0}),
        gateway=gateway,
    )

    assert result["mark_freshness_status"] in {"live", "stale_or_unverified"}
    assert result["mark_usability_status"] == "not_live_ready"
    assert result["data_stale"] is True
    assert "usable_for_live_trading" not in result
    assert result["valuation_basis"] in {
        "position_marks_quote_not_live_ready",
        "stale_or_unverified_position_marks",
    }
    assert result["unusable_marks"] == [
        {
            "symbol": "EURUSD",
            "reason": "locked_quote",
            "spread_quality": "locked",
            "retry_hint": "Refresh the quote and retry.",
        }
    ]


def test_trade_stress_test_rejects_failed_position_snapshot():
    gateway = _Gateway()
    gateway.positions_get = lambda: None

    result = run_trade_stress_test(
        TradeStressTestRequest(shocks={"EURUSD": -1.0}),
        gateway=gateway,
    )

    assert result["success"] is False
    assert result["error_code"] == "positions_snapshot_unavailable"


def test_trade_stress_test_flat_account_has_no_live_quote_gate():
    gateway = _Gateway()
    gateway.positions_get = lambda: []

    result = run_trade_stress_test(
        TradeStressTestRequest(shocks={"*": -2.0}),
        gateway=gateway,
    )

    assert result["success"] is True
    assert result["empty"] is True
    assert result["status"] == "no_open_positions"
    assert result["portfolio_status"] == "no_open_positions"
    assert result["actionability"] == "informational_no_exposure"
    assert result["positions_evaluated"] == 0
    assert result["mark_freshness_status"] == "not_applicable"
    assert result["valuation_time"].endswith("Z")
    assert "usable_for_live_trading" not in result


def test_trade_stress_test_fails_when_no_position_matches_shocks():
    result = run_trade_stress_test(
        TradeStressTestRequest(shocks={"USDJPY": -1.0}),
        gateway=_Gateway(),
    )

    assert result["success"] is False
    assert result["error_code"] == "stress_no_positions_evaluated"


def test_trade_stress_test_freshness_only_covers_evaluated_positions():
    gateway = _Gateway()
    gateway.positions_get = lambda: [
        SimpleNamespace(
            ticket=1,
            symbol="EURUSD",
            type=0,
            volume=1.0,
            price_current=1.1,
            price_open=1.09,
        ),
        SimpleNamespace(
            ticket=2,
            symbol="AAPL.NAS",
            type=0,
            volume=1.0,
            price_current=200.0,
            price_open=190.0,
        ),
    ]

    result = run_trade_stress_test(
        TradeStressTestRequest(shocks={"EURUSD": -1.0}),
        gateway=gateway,
    )

    assert result["positions_total"] == 2
    assert result["positions_evaluated"] == 1
    assert result["marks_evaluated"] == 1
    assert [item["symbol"] for item in result["unusable_marks"]] == ["EURUSD"]
    assert "mark_freshness" not in result
