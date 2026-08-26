from unittest.mock import patch

import pytest

from mtdata.core.trading.context import (
    _build_quote_quality,
    _build_trade_ready,
    trade_session_context,
)
from mtdata.core.trading.requests import TradeSessionContextRequest


def _raw_trade_session_context(
    symbol: str,
    *,
    detail: str = "compact",
    include_account: bool = True,
):
    return trade_session_context.__wrapped__(
        TradeSessionContextRequest(
            symbol=symbol,
            detail=detail,
            include_account=include_account,
        )
    )


def test_trade_ready_does_not_claim_portfolio_risk_approval() -> None:
    readiness = _build_trade_ready(
        {
            "execution_ready": True,
            "equity": 384.28,
            "margin": 318.97,
            "margin_free": 65.31,
            "margin_level": 120.48,
        },
        {"usable_for_live_trading": True, "data_stale": False},
        {"can_open_new_positions": True},
    )

    assert readiness["execution_preconditions_met"] is True
    assert readiness["execution_preconditions_allow_open"] is True
    assert "can_open_new_positions" not in readiness
    assert readiness["portfolio_risk_assessed"] is False
    assert readiness["margin_level"] == 120.48
    assert readiness["margin_utilization_pct"] == 83.0
    assert readiness["margin_available_positive"] is True
    assert "margin_sufficient_for_min_lot" not in readiness
    assert "not_portfolio_risk_approval" in readiness["readiness_scope"]


def test_trade_ready_blocks_critical_margin_stress() -> None:
    readiness = _build_trade_ready(
        {
            "execution_ready": True,
            "equity": 384.44,
            "margin": 348.96,
            "margin_free": 35.48,
            "margin_level": 110.17,
        },
        {"usable_for_live_trading": True, "data_stale": False},
        {"can_open_new_positions": True},
    )

    assert readiness["execution_preconditions_met"] is False
    assert "critical_margin_stress" in readiness["blockers"]
    assert readiness["margin_stress"]["status"] == "critical"
    assert readiness["trade_mode_allows_opening"] is True
    assert readiness["execution_preconditions_allow_open"] is False
    assert "can_open_new_positions" not in readiness


def test_trade_ready_fails_closed_when_quote_or_market_status_is_unknown() -> None:
    readiness = _build_trade_ready(
        {
            "execution_ready": True,
            "equity": 1000.0,
            "margin": 0.0,
            "margin_free": 1000.0,
        },
        {"bid": 1.1, "ask": 1.1002},
        {"status": "unknown"},
    )

    assert readiness["execution_preconditions_met"] is False
    assert readiness["readiness_status"] == "unknown"
    assert "quote_readiness_unknown" in readiness["blockers"]
    assert "market_opening_status_unknown" in readiness["blockers"]


def test_trade_ready_names_fresh_locked_quote_blocker() -> None:
    quote = {
        "freshness_state": "live",
        "data_stale": False,
        "spread_quality": "locked",
        "usable_for_live_trading": False,
    }

    readiness = _build_trade_ready(
        {
            "execution_ready": True,
            "equity": 1000.0,
            "margin": 0.0,
            "margin_free": 1000.0,
        },
        quote,
        {"can_open_new_positions": True},
    )
    quality = _build_quote_quality(quote)

    assert readiness["blockers"] == ["quote_locked"]
    assert quality["is_live"] is True
    assert quality["usable_for_live_trading"] is False
    assert quality["spread_quality"] == "locked"


@pytest.mark.parametrize(
    ("open_available", "pending_available", "expected_blockers"),
    [
        (False, True, ["open_positions_unavailable"]),
        (True, False, ["pending_orders_unavailable"]),
        (
            False,
            False,
            ["open_positions_unavailable", "pending_orders_unavailable"],
        ),
    ],
)
def test_trade_ready_fails_closed_when_book_snapshot_is_unavailable(
    open_available,
    pending_available,
    expected_blockers,
) -> None:
    readiness = _build_trade_ready(
        {
            "execution_ready": True,
            "equity": 1_000.0,
            "margin": 0.0,
            "margin_free": 1_000.0,
        },
        {"usable_for_live_trading": True, "data_stale": False},
        {"can_open_new_positions": True},
        open_positions_available=open_available,
        pending_orders_available=pending_available,
    )

    assert readiness["execution_preconditions_met"] is False
    assert readiness["readiness_status"] == "unknown"
    assert readiness["execution_preconditions_allow_open"] is False
    assert "can_open_new_positions" not in readiness
    assert readiness["blockers"] == expected_blockers


def test_trade_session_request_normalizes_symbol_once_for_nested_reads() -> None:
    seen = {"ticker": [], "open": [], "pending": []}

    def _ticker(symbol, detail="compact"):
        seen["ticker"].append(symbol)
        return {
            "success": True,
            "symbol": symbol,
            "usable_for_live_trading": True,
            "data_stale": False,
        }

    def _open(request):
        seen["open"].append(request.symbol)
        return {"success": True, "count": 0, "items": []}

    def _pending(request):
        seen["pending"].append(request.symbol)
        return {"success": True, "count": 0, "items": []}

    with patch(
        "mtdata.core.trading.context.trade_account_info",
        new=lambda: {
            "success": True,
            "execution_ready": True,
            "equity": 1_000.0,
            "margin": 0.0,
            "margin_free": 1_000.0,
        },
    ), patch(
        "mtdata.core.trading.context.market_ticker",
        new=_ticker,
    ), patch(
        "mtdata.core.trading.context.trade_get_open",
        new=_open,
    ), patch(
        "mtdata.core.trading.context.trade_get_pending",
        new=_pending,
    ), patch(
        "mtdata.core.trading.context._trade_session_tradability",
        return_value={"can_open_new_positions": True},
    ):
        result = _raw_trade_session_context("  eurusd  ", detail="full")

    assert result["symbol"] == "EURUSD"
    assert seen["ticker"] == ["EURUSD"]
    assert seen["open"] == ["EURUSD", None]
    assert seen["pending"] == ["EURUSD"]
    assert result["state"] == "flat"
    assert result["trade_ready"]["readiness_status"] == "ready"


def test_trade_session_context_compacts_nested_sections_by_default() -> None:
    timezone_meta = {"used": {"tz": "UTC"}}
    ticker_compact = {
        "success": True,
        "symbol": "EURUSD",
        "bid": 1.1,
        "ask": 1.1002,
        "mid": 1.1001,
        "price_currency": "USD",
        "spread": 0.0002,
        "spread_pips": 2.0,
        "freshness_state": "live",
        "usable_for_live_trading": True,
        "usable_for_live_trading_basis": "quote_age_and_market_session",
        "live_max_age_seconds": 30,
        "time": "2023-11-14 22:13",
        "timezone": "UTC",
    }
    open_positions = {
        "success": True,
        "kind": "open_positions",
        "count": 1,
        "items": [
            {
                "symbol": "EURUSD",
                "ticket": 123456,
                "time": "2023-11-14 22:13",
                "type": "BUY",
                "volume": 0.1,
                "price_open": 1.1,
                "price_current": 1.1004,
                "price_current_basis": "bid",
                "sl": 1.095,
                "tp": 1.11,
                "profit": 4.2,
                "comment": "agent-open",
                "magic": 77,
                "timezone": "UTC",
            }
        ],
    }

    with patch(
        "mtdata.core.output_contract.build_runtime_timezone_meta",
        return_value=timezone_meta,
    ), patch(
        "mtdata.core.trading.context.trade_account_info",
        new=lambda: {
            "success": True,
            "login": 123456,
            "balance": 10000.0,
            "equity": 10010.0,
            "profit": 10.0,
            "floating_pnl": 10.0,
            "pnl_basis": "floating_open_positions",
            "equity_balance_delta": 10.0,
            "margin_level": 250.0,
            "terminal_connected": True,
            "execution_ready": True,
            "account_type": "demo",
            "is_demo": True,
            "is_live": False,
        },
    ), patch(
        "mtdata.core.trading.context.market_ticker",
        new=lambda symbol, detail="compact": ticker_compact,
    ), patch(
        "mtdata.core.trading.context.trade_get_open",
        new=lambda request: open_positions,
    ), patch(
        "mtdata.core.trading.context.trade_get_pending",
        new=lambda request: {
            "success": True,
            "kind": "pending_orders",
            "count": 0,
            "message": "No pending orders for EURUSD",
            "no_action": True,
        },
    ):
        out = _raw_trade_session_context("EURUSD")

    assert out["state"] == "open_position"
    assert out["state_scope"] == "symbol"
    assert out["account"]["equity"] == 10010.0
    assert out["account"]["profit"] == 10.0
    assert "floating_pnl" not in out["account"]
    assert "pnl_basis" not in out["account"]
    assert "equity_balance_delta" not in out["account"]
    assert "login" not in out["account"]
    assert out["account"]["account_type"] == "demo"
    assert out["account"]["is_demo"] is True
    assert out["account"]["is_live"] is False
    assert "execution_ready" not in out["account"]
    assert out["quote"] == {
        "bid": 1.1,
        "ask": 1.1002,
        "mid": 1.1001,
        "price_currency": "USD",
        "spread": 0.0002,
        "spread_pips": 2.0,
        "freshness_state": "live",
        "usable_for_live_trading": True,
        "usable_for_live_trading_basis": "quote_age_and_market_session",
        "live_max_age_seconds": 30,
        "time": "2023-11-14 22:13",
        "timezone": "UTC",
    }
    assert out["open_positions"] == [
        {
            "symbol": "EURUSD",
            "ticket": 123456,
            "time": "2023-11-14 22:13",
            "type": "BUY",
            "volume": 0.1,
            "price_open": 1.1,
            "price_current": 1.1004,
            "price_current_basis": "bid",
            "sl": 1.095,
            "tp": 1.11,
            "profit": 4.2,
            "comment": "agent-open",
            "magic": 77,
            "timezone": "UTC",
        }
    ]
    assert out["open_positions_count"] == 1
    assert out["pending_orders"] == []
    assert out["pending_orders_count"] == 0
    assert "hints" not in out
    assert "show_all_hint" not in out
    assert out["meta"]["tool"] == "trade_session_context"
    assert out["meta"]["runtime"]["timezone"] == timezone_meta
    assert out["as_of"].endswith("Z")
    assert out["assembled_at"] == out["as_of"]
    assert out["timezone"] == "UTC"
    assert out["source"]["provider"] == "mt5"


def test_trade_session_context_surfaces_closed_market_tradability() -> None:
    with patch(
        "mtdata.core.trading.context.trade_account_info",
        new=lambda: {
            "success": True,
            "equity": 1000.0,
            "margin_free": 900.0,
            "execution_ready": True,
        },
    ), patch(
        "mtdata.core.trading.context.market_ticker",
        new=lambda symbol, detail="compact": {
            "success": True,
            "bid": 1.1,
            "ask": 1.1002,
            "data_stale": False,
            "freshness": "closed weekend, tick 16h 0m ago",
        },
    ), patch(
        "mtdata.core.trading.context._trade_session_tradability",
        return_value={
            "status": "weekend_closed",
            "reason": "weekend",
            "is_tradable": False,
            "can_open_new_positions": False,
        },
    ), patch(
        "mtdata.core.trading.context.trade_get_open",
        new=lambda request: {
            "success": True,
            "kind": "open_positions",
            "count": 0,
            "items": [],
        },
    ), patch(
        "mtdata.core.trading.context.trade_get_pending",
        new=lambda request: {
            "success": True,
            "kind": "pending_orders",
            "count": 0,
            "items": [],
        },
    ):
        out = _raw_trade_session_context("EURUSD")

    assert out["market_status"] == "weekend_closed"
    assert out["market_status_reason"] == "weekend"
    assert out["is_tradable"] is False
    assert out["execution_preconditions_allow_open"] is False
    assert "can_open_new_positions" not in out
    assert out["trade_ready"]["execution_preconditions_met"] is False
    assert "market_not_open_for_new_positions" in out["trade_ready"]["blockers"]


def test_trade_session_context_compact_surfaces_portfolio_exposure_elsewhere() -> None:
    def open_positions_for_request(request):
        if request.symbol == "EURUSD":
            return {"success": True, "kind": "open_positions", "count": 0, "items": []}
        return {
            "success": True,
            "kind": "open_positions",
            "count": 1,
            "items": [{"symbol": "BTCUSD", "ticket": 77, "profit": -1.73}],
        }

    with patch(
        "mtdata.core.trading.context.trade_account_info",
        new=lambda: {
            "success": True,
            "equity": 998.27,
            "profit": -1.73,
            "account_type": "demo",
        },
    ), patch(
        "mtdata.core.trading.context.market_ticker",
        new=lambda symbol, detail="compact": {
            "success": True,
            "bid": 1.1,
            "ask": 1.1002,
        },
    ), patch(
        "mtdata.core.trading.context.trade_get_open",
        new=open_positions_for_request,
    ), patch(
        "mtdata.core.trading.context.trade_get_pending",
        new=lambda request: {
            "success": True,
            "kind": "pending_orders",
            "count": 0,
            "items": [],
        },
    ):
        out = _raw_trade_session_context("EURUSD")

    assert out["state"] == "flat"
    assert out["state_scope"] == "symbol"
    assert out["open_positions_count"] == 0
    assert out["portfolio_positions_count"] == 1
    assert out["other_positions_count"] == 1
    assert out["account"]["profit"] == -1.73


def test_trade_session_context_compact_keeps_nested_quote_spread_numeric() -> None:
    ticker_compact = {
        "success": True,
        "symbol": "EURUSD",
        "bid": 1.17071,
        "ask": 1.17080,
        "mid": 1.170755,
        "price_currency": "USD",
        "price_precision": 5,
        "spread": 0.00009,
        "spread_points": 9.0,
        "spread_pct": 0.007687,
        "spread_cost_currency": "USD",
        "time": "2026-04-29 02:43",
        "data_stale": True,
        "stale_warning": "quote is stale",
    }

    with patch(
        "mtdata.core.trading.context.trade_account_info",
        new=lambda: {"success": True, "equity": 10010.0},
    ), patch(
        "mtdata.core.trading.context.market_ticker",
        new=lambda symbol, detail="compact": ticker_compact,
    ), patch(
        "mtdata.core.trading.context.trade_get_open",
        new=lambda request: {"success": True, "kind": "open_positions", "count": 0, "items": []},
    ), patch(
        "mtdata.core.trading.context.trade_get_pending",
        new=lambda request: {"success": True, "kind": "pending_orders", "count": 0, "items": []},
    ):
        out = _raw_trade_session_context("EURUSD")

    assert out["quote"]["spread"] == 0.00009
    assert out["quote"]["mid"] == 1.170755
    assert out["quote"]["price_currency"] == "USD"
    assert out["quote"]["price_precision"] == 5
    assert out["quote"]["spread_cost_currency"] == "USD"
    assert out["quote"]["data_stale"] is True
    assert out["quote"]["stale_warning"] == "quote is stale"
    assert out["open_positions"] == []
    assert out["open_positions_count"] == 0
    assert out["pending_orders"] == []
    assert out["pending_orders_count"] == 0


def test_trade_session_context_full_detail_keeps_nested_full_payloads() -> None:
    timezone_meta = {"used": {"tz": "UTC"}}
    ticker = {
        "success": True,
        "symbol": "EURUSD",
        "price_precision": 5,
        "bid": 1.1,
        "ask": 1.1002,
        "time": 1700000000,
        "time_display": "2023-11-14 22:13",
        "meta": {
            "tool": "market_ticker",
            "runtime": {"timezone": timezone_meta},
        },
        "timezone": "UTC",
    }

    with patch(
        "mtdata.core.output_contract.build_runtime_timezone_meta",
        return_value=timezone_meta,
    ), patch(
        "mtdata.core.trading.context.trade_account_info",
        new=lambda: {"success": True, "login": 123456, "balance": 10000.0},
    ), patch(
        "mtdata.core.trading.context.market_ticker",
        new=lambda symbol, detail="compact": ticker,
    ), patch(
        "mtdata.core.trading.context.trade_get_open",
        new=lambda request: {"success": True, "count": 0},
    ), patch(
        "mtdata.core.trading.context.trade_get_pending",
        new=lambda request: {
            "success": True,
            "count": 1,
            "items": [
                {
                    "symbol": "EURUSD",
                    "price_current": 1.1710099999999999,
                    "Current Price": 1.1710099999999999,
                }
            ],
        },
    ):
        out = _raw_trade_session_context("EURUSD", detail="full")

    assert out["state"] == "pending_only"
    # With the fix, nested success and meta are stripped
    assert "success" not in out["quote"]
    assert "meta" not in out["quote"]
    assert out["quote"]["bid"] == 1.1  # data is preserved
    assert out["quote"]["time"] == "2023-11-14 22:13"
    assert out["quote"]["time_epoch"] == 1700000000
    assert "time_display" not in out["quote"]
    assert out["account"]["balance"] == 10000.0
    assert out["account"]["login"] == 123456
    assert "success" not in out["account"]
    assert "meta" not in out["account"]
    assert out["pending_orders"]["count"] == 1
    assert out["pending_orders"]["items"][0]["price_current"] == 1.17101
    assert out["pending_orders"]["items"][0]["Current Price"] == 1.17101
    assert "success" not in out["pending_orders"]
    assert "meta" not in out["pending_orders"]
    assert out["meta"]["tool"] == "trade_session_context"
    assert out["meta"]["runtime"]["timezone"] == timezone_meta


def test_trade_session_context_can_omit_account_section() -> None:
    def fail_account_call():
        raise AssertionError("account info should not be fetched")

    with patch(
        "mtdata.core.trading.context.trade_account_info",
        new=fail_account_call,
    ), patch(
        "mtdata.core.trading.context.market_ticker",
        new=lambda symbol, detail="compact": {"success": True, "bid": 1.1, "ask": 1.1002},
    ), patch(
        "mtdata.core.trading.context.trade_get_open",
        new=lambda request: {"success": True, "count": 0},
    ), patch(
        "mtdata.core.trading.context.trade_get_pending",
        new=lambda request: {"success": True, "count": 0},
    ):
        out = _raw_trade_session_context("EURUSD", include_account=False)

    assert out["success"] is True
    assert out["state"] == "flat"
    assert "account" not in out


def test_trade_session_context_compact_sanitizes_nested_tool_errors() -> None:
    with patch(
        "mtdata.core.trading.context.trade_account_info",
        new=lambda: {"success": True, "balance": 10000.0, "equity": 10010.0},
    ), patch(
        "mtdata.core.trading.context.market_ticker",
        new=lambda symbol, detail="compact": {"success": True, "bid": 1.1, "ask": 1.1002},
    ), patch(
        "mtdata.core.trading.context.trade_get_open",
        new=lambda request: {
            "success": False,
            "error": "'types.SimpleNamespace' object has no attribute '_asdict'",
        },
    ), patch(
        "mtdata.core.trading.context.trade_get_pending",
        new=lambda request: {"success": True, "count": 0, "message": "No pending orders"},
    ):
        out = _raw_trade_session_context("EURUSD")

    assert out["success"] is True
    assert out["partial_failure"] is True
    assert out["state"] == "unknown"
    assert out["open_positions"] == {
        "error": "Unable to fetch open positions.",
    }
    assert "SimpleNamespace" not in str(out["open_positions"])
    assert out["trade_ready"]["execution_preconditions_met"] is False
    assert out["trade_ready"]["readiness_status"] == "unknown"
    assert "open_positions_unavailable" in out["trade_ready"]["blockers"]


def test_trade_session_context_compact_keeps_order_attribution_fields() -> None:
    with patch(
        "mtdata.core.trading.context.trade_account_info",
        new=lambda: {"success": True, "balance": 10000.0, "equity": 10010.0},
    ), patch(
        "mtdata.core.trading.context.market_ticker",
        new=lambda symbol, detail="compact": {"success": True, "bid": 1.1, "ask": 1.1002},
    ), patch(
        "mtdata.core.trading.context.trade_get_open",
        new=lambda request: {
            "success": True,
            "count": 1,
            "items": [
                {
                    "Symbol": "EURUSD",
                    "Ticket": 123,
                    "Type": "BUY",
                    "Volume": 0.1,
                    "Open Price": 1.1,
                    "Comments": "open-agent",
                    "Magic": 7001,
                }
            ],
        },
    ), patch(
        "mtdata.core.trading.context.trade_get_pending",
        new=lambda request: {
            "success": True,
            "count": 1,
            "items": [
                {
                    "Symbol": "EURUSD",
                    "Ticket": 456,
                    "Type": "BUY_LIMIT",
                    "Side": "BUY",
                    "Volume": 0.1,
                    "Open Price": 1.095,
                    "Comments": "pending-agent",
                    "Magic": 7002,
                }
            ],
        },
    ):
        out = _raw_trade_session_context("EURUSD")

    assert out["state"] == "mixed"
    assert out["open_positions"] == [
        {
            "symbol": "EURUSD",
            "ticket": 123,
            "type": "BUY",
            "volume": 0.1,
            "price_open": 1.1,
            "comment": "open-agent",
            "magic": 7001,
        }
    ]
    assert out["pending_orders"] == [
        {
            "symbol": "EURUSD",
            "ticket": 456,
            "type": "BUY_LIMIT",
            "order_type": "BUY_LIMIT",
            "side": "BUY",
            "volume": 0.1,
            "price_open": 1.095,
            "trigger_price": 1.095,
            "entry_price": 1.095,
            "comment": "pending-agent",
            "magic": 7002,
        }
    ]
