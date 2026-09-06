from __future__ import annotations

import importlib
import json
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import call, patch

import pytest

import mtdata.core.market_depth as market_depth_mod
from mtdata.core._mcp_tools import get_tool_functions, registered_tool_catalog
from mtdata.core.market_depth import market_depth_fetch, market_ticker
from mtdata.utils.mt5 import MT5ConnectionError


def _raw_market_depth_fetch(symbol: str, spread: bool = False, require_dom: bool = False):
    raw = getattr(market_depth_fetch, "__wrapped__", market_depth_fetch)
    return raw(symbol, spread=spread, require_dom=require_dom)


def _raw_market_ticker(symbol: str, *, detail: str = "full", price_field=None):
    return market_ticker.__wrapped__(symbol, detail=detail, price_field=price_field)


def _tool_catalog_row(name: str) -> dict:
    catalog = registered_tool_catalog()
    rows = catalog.get("tools")
    assert isinstance(rows, list)
    row = next(
        (
            item
            for item in rows
            if isinstance(item, dict) and item.get("name") == name
        ),
        None,
    )
    assert row is not None
    return row


@pytest.fixture(autouse=True)
def _enable_market_depth(monkeypatch) -> None:
    monkeypatch.setenv("MTDATA_ENABLE_MARKET_DEPTH_FETCH", "1")
    monkeypatch.setattr(
        market_depth_mod, "ensure_mt5_connection_or_raise", lambda: None
    )


def test_market_depth_tick_fallback_includes_price_display() -> None:
    tick = SimpleNamespace(
        bid=65601.0,
        ask=65601.5,
        last=65601.0,
        volume=12,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(digits=2, currency_profit="USD")
        mt5.market_book_get.return_value = []
        mt5.symbol_info_tick.return_value = tick

        out = _raw_market_depth_fetch("BTCUSD")

    assert out["success"] is True
    assert out["type"] == "quote_fallback"
    assert out["depth_status"] == "unavailable"
    assert out["recommended_alternative"] == "market_ticker"
    assert out["capabilities"]["dom_available"] is False
    assert out["capabilities"]["depth_status"] == "unavailable"
    assert out["capabilities"]["depth_source"] == "symbol_info_tick"
    assert out["price_precision"] == 2
    assert out["price_currency"] == "USD"
    assert out["data"]["bid"] == 65601.0
    assert out["data"]["ask"] == 65601.5
    assert out["data"]["last"] == 65601.0
    assert out["data"]["time"] == "2023-11-14T22:13:20Z"
    assert out["data"]["time_epoch"] == 1700000000
    assert out["units"] == {"volume": "mt5_tick_volume"}
    assert out["freshness_state"] == "stale"
    assert out["usable_for_live_trading"] is False
    assert out["source"]["provider"] == "mt5"
    assert out["meta"]["tool"] == "market_depth_fetch"
    assert out["observed_at_epoch"] > 0
    assert isinstance(out.get("query_latency_ms"), float)


def test_market_depth_resolves_public_slash_alias() -> None:
    tick = SimpleNamespace(
        bid=1.1,
        ask=1.1001,
        last=1.1,
        volume=12,
        time=1_700_000_000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz",
        return_value=False,
    ):
        mt5.symbols_get.return_value = [SimpleNamespace(name="EURUSD")]
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            currency_profit="USD",
        )
        mt5.market_book_add.return_value = False
        mt5.market_book_get.return_value = []
        mt5.symbol_info_tick.return_value = tick
        mt5.copy_ticks_range.return_value = []

        out = _raw_market_depth_fetch("EUR/USD")

    assert out["success"] is True
    assert out["symbol"] == "EURUSD"
    assert out["symbol_input"] == "EUR/USD"
    mt5.symbol_select.assert_called_once_with("EURUSD", True)
    mt5.symbol_info.assert_called_with("EURUSD")
    mt5.symbol_info_tick.assert_called_with("EURUSD")


def test_market_depth_tick_fallback_hides_zero_last_display() -> None:
    tick = SimpleNamespace(
        bid=65601.0,
        ask=65601.5,
        last=0.0,
        volume=12,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(digits=2)
        mt5.market_book_get.return_value = []
        mt5.symbol_info_tick.return_value = tick

        out = _raw_market_depth_fetch("BTCUSD")

    assert out["success"] is True
    assert out["data"]["last"] is None
    assert out["data"]["last_unavailable"] is True


def test_market_depth_tick_fallback_marks_fresh_quote_live_ready() -> None:
    tick = SimpleNamespace(
        bid=100.0,
        ask=100.5,
        last=100.25,
        volume=12,
        time=1_700_000_000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth.time.time", return_value=1_700_000_005.0
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(digits=2)
        mt5.market_book_get.return_value = []
        mt5.symbol_info_tick.return_value = tick

        out = _raw_market_depth_fetch("BTCUSD")

    assert out["success"] is True
    assert out["freshness_state"] == "live"
    assert out["data_age_seconds"] == 5.0
    assert out["usable_for_live_trading"] is True


def test_market_depth_tick_fallback_blocks_material_quote_source_conflict() -> None:
    now = 1_700_000_100.0
    cached_tick = SimpleNamespace(
        bid=1.15304,
        ask=1.15326,
        last=1.15315,
        volume=5,
        time=now - 1.0,
        time_msc=(now - 1.0) * 1000.0,
    )
    stream_tick = {
        "bid": 1.15408,
        "ask": 1.15422,
        "last": 1.15415,
        "volume": 6,
        "time": now - 1.0,
        "time_msc": (now - 1.0) * 1000.0,
    }
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth.time.time", return_value=now
    ):
        mt5.COPY_TICKS_ALL = 0
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
        )
        mt5.market_book_add.return_value = False
        mt5.market_book_get.return_value = []
        mt5.symbol_info_tick.return_value = cached_tick
        mt5.copy_ticks_range.return_value = [stream_tick]

        out = _raw_market_depth_fetch("EURUSD")

    assert out["success"] is True
    assert out["type"] == "quote_fallback"
    assert out["quote_source_state"] == "reconciled_equal_timestamp_conflict"
    assert isinstance(out["quote_source_conflict"], dict)
    assert out["usable_for_live_trading"] is False
    assert out["usable_for_live_trading_basis"] == (
        "quote_age_market_session_spread_and_source_agreement"
    )
    assert out["execution_blockers"] == ["quote_source_conflict"]


def test_market_depth_full_depth_includes_price_display() -> None:
    depth = [
        {"price": 65601.0, "volume": 1.0, "volume_real": 1.0, "type": 2},
        {"price": 65602.5, "volume": 2.0, "volume_real": 2.0, "type": 1},
    ]
    with patch("mtdata.core.market_depth.mt5") as mt5:
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(digits=2)
        mt5.market_book_get.return_value = depth

        out = _raw_market_depth_fetch("BTCUSD")

    assert out["success"] is True
    assert out["type"] == "full_depth"
    assert out["capabilities"]["dom_available"] is True
    assert out["data"]["depth_levels"]["total"] == 2
    assert out["price_precision"] == 2
    assert out["data"]["buy_orders"][0]["price_display"] == "65601.00"
    assert out["data"]["sell_orders"][0]["price_display"] == "65602.50"
    assert out["units"]["volume"] == "book_volume"
    assert out["source"]["provider"] == "mt5"
    assert out["meta"]["tool"] == "market_depth_fetch"


def test_market_depth_full_depth_accepts_mt5_bookinfo_volume_dbl() -> None:
    BookInfo = namedtuple("BookInfo", ["type", "price", "volume", "volume_dbl"])
    depth = [
        BookInfo(type=2, price=65601.0, volume=1, volume_dbl=1.25),
        BookInfo(type=1, price=65602.5, volume=2, volume_dbl=2.5),
    ]
    with patch("mtdata.core.market_depth.mt5") as mt5:
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(digits=2)
        mt5.market_book_get.return_value = depth

        out = _raw_market_depth_fetch("BTCUSD")

    assert out["success"] is True
    assert out["data"]["depth_levels"]["total"] == 2
    assert out["data"]["buy_orders"][0]["volume"] == 1.0
    assert out["data"]["buy_orders"][0]["volume_real"] == 1.25
    assert out["data"]["sell_orders"][0]["volume_real"] == 2.5


def test_market_depth_classifies_all_official_mt5_book_types() -> None:
    depth = [
        {"price": 101.0, "volume": 1.0, "volume_real": 1.0, "type": 1},
        {"price": 100.0, "volume": 1.0, "volume_real": 1.0, "type": 2},
        {"price": 102.0, "volume": 1.0, "volume_real": 1.0, "type": 3},
        {"price": 99.0, "volume": 1.0, "volume_real": 1.0, "type": 4},
        {"price": 98.0, "volume": 1.0, "volume_real": 1.0, "type": 99},
    ]
    with patch("mtdata.core.market_depth.mt5") as mt5:
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(digits=2)
        mt5.market_book_get.return_value = depth

        out = _raw_market_depth_fetch("BTCUSD", spread=True)

    assert out["data"]["depth_levels"] == {"buy": 2, "sell": 2, "total": 4}
    assert out["data"]["best_bid"] == 100.0
    assert out["data"]["best_ask"] == 101.0
    assert out["data"]["unclassified_orders"][0]["book_type"] == 99
    assert out["capabilities"]["unclassified_depth_levels"] == 1


def test_market_depth_subscribes_and_releases_book_snapshot() -> None:
    depth = [
        {"price": 65601.0, "volume": 1.0, "volume_real": 1.0, "type": 2},
        {"price": 65602.5, "volume": 2.0, "volume_real": 2.0, "type": 1},
    ]
    with patch("mtdata.core.market_depth.mt5") as mt5:
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(digits=2)
        mt5.market_book_add.return_value = True
        mt5.market_book_get.return_value = depth

        out = _raw_market_depth_fetch("BTCUSD")

    assert out["success"] is True
    mt5.market_book_add.assert_called_once_with("BTCUSD")
    mt5.market_book_get.assert_called_once_with("BTCUSD")
    mt5.market_book_release.assert_called_once_with("BTCUSD")


def test_market_depth_releases_book_after_empty_snapshot() -> None:
    tick = SimpleNamespace(
        bid=65601.0,
        ask=65601.5,
        last=65601.0,
        volume=12,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ), patch("mtdata.core.market_depth.time.sleep"):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(digits=2)
        mt5.market_book_add.return_value = True
        mt5.market_book_get.return_value = []
        mt5.symbol_info_tick.return_value = tick

        out = _raw_market_depth_fetch("BTCUSD")

    assert out["success"] is True
    assert out["type"] == "quote_fallback"
    mt5.market_book_add.assert_called_once_with("BTCUSD")
    mt5.market_book_release.assert_called_once_with("BTCUSD")


def test_market_depth_waits_for_initial_subscribed_snapshot() -> None:
    depth = [
        {"price": 65601.0, "volume": 1.0, "volume_real": 1.0, "type": 2},
    ]
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth.time.sleep"
    ) as sleep:
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(digits=2)
        mt5.market_book_add.return_value = True
        mt5.market_book_get.side_effect = [[], depth]

        out = _raw_market_depth_fetch("BTCUSD")

    assert out["success"] is True
    assert out["type"] == "full_depth"
    assert mt5.market_book_get.call_count == 2
    sleep.assert_called_once_with(0.01)
    mt5.market_book_release.assert_called_once_with("BTCUSD")


def test_market_depth_tick_fallback_includes_spread_metrics_when_requested() -> None:
    tick = SimpleNamespace(
        bid=100.0,
        ask=101.0,
        last=100.5,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
        )
        mt5.market_book_get.return_value = []
        mt5.symbol_info_tick.return_value = tick
        mt5.account_info.return_value = SimpleNamespace(currency="USD")
        out = _raw_market_depth_fetch("BTCUSD", spread=True)

    assert out["success"] is True
    assert out["data"]["spread"] == 1.0
    assert out["data"]["spread_points"] == 100.0
    assert abs(out["data"]["spread_pct"] - (100.0 / 100.5)) < 1e-12
    assert out["data"]["spread_cost_per_lot"] == 100.0
    assert out["capabilities"]["spread_overlay_applied"] is True
    assert out["units"]["spread"] == "absolute_price"
    assert out["units"]["spread_points"] == "broker_points"
    assert out["units"]["spread_pct"] == "percent (1.0 = 1%)"
    assert out["units"]["spread_cost_per_lot"] == (
        "account_currency_per_broker_lot_estimate"
    )


def test_market_depth_compact_mode_fails_fast_without_dom() -> None:
    tick = SimpleNamespace(
        bid=100.0,
        ask=101.0,
        last=100.5,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5:
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(digits=2, point=0.01)
        mt5.market_book_get.return_value = []
        mt5.symbol_info_tick.return_value = tick

        out = _raw_market_depth_fetch("BTCUSD", require_dom=True)

    assert out["error"] == "DOM not available for BTCUSD. Use market_ticker for bid/ask snapshot instead."
    assert out["error_code"] == "dom_unavailable"
    assert out["depth_status"] == "unavailable"
    assert out["capabilities"] == {
        "dom_available": False,
        "depth_status": "unavailable",
        "fallback_reason": "market_book_get returned no levels",
    }
    assert out["recommended_alternative"] == "market_ticker"


def test_market_depth_full_depth_includes_spread_metrics_when_requested() -> None:
    depth = [
        {"price": 100.0, "volume": 1.0, "volume_real": 1.0, "type": 2},
        {"price": 101.0, "volume": 2.0, "volume_real": 2.0, "type": 1},
    ]
    with patch("mtdata.core.market_depth.mt5") as mt5:
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
        )
        mt5.market_book_get.return_value = depth
        out = _raw_market_depth_fetch("BTCUSD", spread=True)

    assert out["success"] is True
    assert out["data"]["best_bid"] == 100.0
    assert out["data"]["best_ask"] == 101.0
    assert out["data"]["spread"] == 1.0
    assert out["capabilities"]["spread_overlay_applied"] is True
    assert out["units"]["spread"] == "absolute_price"
    assert out["units"]["spread_points"] == "broker_points"


def test_market_depth_spread_overlay_skips_all_none_book_prices() -> None:
    depth = [
        {"price": None, "volume": 1.0, "volume_real": 1.0, "type": 2},
        {"price": None, "volume": 2.0, "volume_real": 2.0, "type": 1},
    ]
    with patch("mtdata.core.market_depth.mt5") as mt5:
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
        )
        mt5.market_book_get.return_value = depth
        out = _raw_market_depth_fetch("BTCUSD", spread=True)

    assert out["success"] is True
    assert "best_bid" not in out["data"]
    assert "best_ask" not in out["data"]
    assert "spread_overlay_applied" not in out["capabilities"]


def test_market_ticker_returns_lightweight_spread_snapshot() -> None:
    tick = SimpleNamespace(
        bid=200.0,
        ask=201.0,
        last=200.5,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
            currency_profit="USD",
            trade_contract_size=1.0,
        )
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_ticker("BTCUSD", detail="compact")

    assert out["success"] is True
    assert out["type"] == "quote"
    assert out["price_currency"] == "USD"
    assert out["bid"] == 200.0
    assert out["ask"] == 201.0
    assert out["spread_pct"] == 0.498753
    assert out["point"] == 0.01
    assert "market_state" not in out
    assert out["units"] == {
        "bid": "absolute_price",
        "ask": "absolute_price",
        "spread": "absolute_price",
        "spread_points": "broker_points",
        "spread_pct": "percent (1.0 = 1%)",
    }
    assert "contract_size" not in out
    assert "lot_definition" not in out
    assert "pricing_basis" not in out
    assert "pricing_basis_units" not in out
    assert out["freshness"].startswith("stale, tick ")
    assert out["time"] == "2023-11-14T22:13:20Z"
    assert "time_epoch" not in out
    assert out["spread"] == 1.0
    assert out["spread_points"] == 100.0
    assert "spread_pips" not in out
    assert "spread_pct_display" not in out
    assert out["data_stale"] is True
    assert out["data_age_seconds"] > 300
    assert out["freshness_state"] == "stale"
    assert out["usable_for_live_trading"] is False
    assert "data_age_hours" not in out
    assert out["warning"] == (
        "Tick data may be stale; last tick time is 2023-11-14T22:13:20Z."
    )
    assert "last" not in out
    assert "tick_volume" not in out
    assert "spread_cost_per_lot" not in out
    assert "spread_cost_currency" not in out
    assert "spread_display" not in out
    assert out["meta"]["tool"] == "market_ticker"
    assert "diagnostics" not in out["meta"]


def test_market_ticker_uses_canonical_broker_symbol_case() -> None:
    tick = SimpleNamespace(bid=100.0, ask=100.1, last=100.05, volume=1, time=1700000000)
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth.resolve_broker_symbol_name", return_value="QQQ"
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
            currency_profit="USD",
            trade_contract_size=1.0,
        )
        mt5.symbol_info_tick.return_value = tick

        out = _raw_market_ticker("qqq", detail="compact")

    assert out["symbol"] == "QQQ"
    mt5.symbol_select.assert_called_once_with("QQQ", True)
    mt5.symbol_info_tick.assert_called_once_with("QQQ")


def test_market_ticker_restores_hidden_symbol_visibility() -> None:
    tick = SimpleNamespace(
        bid=100.0, ask=100.1, last=100.05, volume=1, time=1700000000
    )
    with patch("mtdata.core.market_depth.mt5") as mt5:
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            visible=False,
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
            currency_profit="USD",
            trade_contract_size=1.0,
        )
        mt5.symbol_info_tick.return_value = tick

        out = _raw_market_ticker("HIDDEN", detail="compact")

    assert out["success"] is True
    assert mt5.symbol_select.call_args_list == [
        call("HIDDEN", True),
        call("HIDDEN", False),
    ]


def test_market_ticker_compact_detail_omits_verbose_fields() -> None:
    tick = SimpleNamespace(
        bid=200.0,
        ask=201.0,
        last=200.5,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
            currency_profit="USD",
            trade_contract_size=1.0,
        )
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_ticker("BTCUSD", detail="compact")

    assert out["success"] is True
    assert out["type"] == "quote"
    assert out["price_currency"] == "USD"
    assert out["bid"] == 200.0
    assert out["ask"] == 201.0
    assert out["spread_pct"] == 0.498753
    assert "market_state" not in out
    assert out["units"]["spread"] == "absolute_price"
    assert out["units"]["spread_points"] == "broker_points"
    assert out["units"]["spread_pct"] == "percent (1.0 = 1%)"
    assert "contract_size" not in out
    assert out["freshness"].startswith("stale, tick ")
    assert "spread_display" not in out
    assert out["spread"] == 1.0
    assert out["spread_points"] == 100.0
    assert "spread_pips" not in out
    assert "spread_pct_display" not in out
    assert "last" not in out
    assert "tick_volume" not in out
    assert out["time"] == "2023-11-14T22:13:20Z"
    assert "time_display" not in out
    assert out["data_stale"] is True
    assert "stale_after_seconds" not in out
    assert "freshness_basis" not in out
    assert "data_age" not in out
    assert out["warning"] == (
        "Tick data may be stale; last tick time is 2023-11-14T22:13:20Z."
    )
    assert "spread_cost_per_lot" not in out
    assert "spread_cost_currency" not in out
    assert "diagnostics" not in out
    assert out["meta"]["tool"] == "market_ticker"


def test_optional_finite_float_keeps_zero() -> None:
    assert market_depth_mod._optional_finite_float(0.0) == 0.0
    assert market_depth_mod._optional_finite_float(0) == 0.0
    assert market_depth_mod._optional_finite_float(None) is None
    assert market_depth_mod._optional_finite_float(float("nan")) is None


def test_market_ticker_none_detail_uses_compact_output() -> None:
    tick = SimpleNamespace(
        bid=200.0,
        ask=201.0,
        last=200.5,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
            currency_profit="USD",
            trade_contract_size=1.0,
        )
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_ticker("BTCUSD", detail=None)

    assert out["success"] is True
    assert out["spread_pct"] == 0.498753
    assert "market_state" not in out
    assert "contract_size" not in out
    assert out["spread"] == 1.0
    assert "diagnostics" not in out
    assert "spread_cost_per_lot" not in out


def test_market_ticker_price_field_returns_simple_price() -> None:
    tick = SimpleNamespace(
        bid=1.17221,
        ask=1.17237,
        last=1.17230,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth.time.time", return_value=1_700_001_000.0
    ), patch("mtdata.core.market_depth._use_client_tz", return_value=False):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
            currency_profit="USD",
        )
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_ticker("EURUSD", price_field="mid")

    assert out["success"] is True
    assert out["type"] == "price"
    assert out["field"] == "mid"
    assert out["price"] == 1.17229
    assert "1.1722899999999998" not in json.dumps(out)
    assert out["price_precision"] == 6
    assert out["selected_value_executable"] is False
    assert out["quote_usable_for_live_trading"] is False
    assert out["price_currency"] == "USD"
    assert out["price_currency_basis"] == "quote_currency_not_cash_cost"
    assert out["unit"] == "absolute_price"
    assert out["point"] == 0.00001
    assert out["units"] == {
        "price": "absolute_price",
        "point": "price_increment",
    }
    assert out["time"] == "2023-11-14T22:13:20Z"
    assert out["time_epoch"] == 1700000000.0
    assert out["data_age_seconds"] >= 0
    assert out["stale_after_seconds"] == 300
    assert out["data_stale"] is True
    assert out["freshness_basis"] == "absolute_300s"
    assert out["freshness_state"] == "stale"
    assert out["freshness_reason"] == "stale_age"
    assert out["usable_for_live_trading"] is False
    assert out["live_max_age_seconds"] == 10
    assert "bid" not in out
    assert "spread_pips" not in out
    assert out["meta"]["tool"] == "market_ticker"


def test_market_ticker_mid_keeps_quote_readiness_separate_from_executability() -> None:
    now = 1_700_000_100.0
    tick = SimpleNamespace(
        bid=1.17221,
        ask=1.17237,
        last=1.17230,
        volume=5,
        time=now - 1.0,
        time_msc=(now - 1.0) * 1000.0,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth.time.time", return_value=now
    ), patch("mtdata.core.market_depth._use_client_tz", return_value=False):
        mt5.COPY_TICKS_ALL = 0
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
            currency_profit="USD",
        )
        mt5.symbol_info_tick.return_value = tick
        mt5.copy_ticks_range.return_value = []

        out = _raw_market_ticker("EURUSD", price_field="mid")

    assert out["quote_usable_for_live_trading"] is True
    assert out["usable_for_live_trading"] is True
    assert out["usable_for_live_trading_basis"] == (
        "quote_age_market_session_and_positive_spread"
    )
    assert out["selected_value_executable"] is False


def test_market_ticker_spread_field_retains_spread_units() -> None:
    tick = SimpleNamespace(
        bid=1.17221,
        ask=1.17237,
        last=1.17230,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
            currency_profit="USD",
        )
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_ticker("EURUSD", price_field="spread", detail="full")

    assert out["success"] is True
    assert out["type"] == "spread"
    assert out["field"] == "spread"
    assert out["selected_value_executable"] is False
    assert out["usable_for_live_trading"] is False
    assert out["price"] == pytest.approx(0.00016)
    assert out["unit"] == "absolute_price"
    assert out["point"] == 0.00001
    assert out["spread_points"] == 16.0
    assert out["spread_pips"] == 1.6
    assert out["units"] == {
        "price": "absolute_price",
        "point": "price_increment",
        "spread_points": "broker_points",
        "spread_pips": "pips",
        "spread_pct": "percent (1.0 = 1%)",
    }


def test_market_ticker_non_fx_spread_does_not_invent_pips() -> None:
    tick = SimpleNamespace(
        bid=200.0,
        ask=200.05,
        last=200.0,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
            currency_profit="USD",
        )
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_ticker("XAUUSD", price_field="spread", detail="compact")

    assert out["point"] == 0.01
    assert out["unit"] == "absolute_price"
    assert "spread_pips" not in out
    assert "spread_pips" not in out["units"]


def test_market_ticker_refreshes_stale_symbol_tick_from_live_stream() -> None:
    now = 1_700_000_100.0
    cached_tick = SimpleNamespace(
        bid=3976.34,
        ask=3976.84,
        last=3976.5,
        volume=1,
        time=now + 3600.0,
    )
    stream_tick = {
        "bid": 3981.46,
        "ask": 3981.57,
        "last": 3981.5,
        "volume": 2,
        "time": now - 1.0,
        "time_msc": (now - 1.0) * 1000.0,
    }
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth.time.time", return_value=now
    ), patch("mtdata.core.market_depth._use_client_tz", return_value=False):
        mt5.COPY_TICKS_ALL = 0
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
            currency_profit="USD",
        )
        mt5.symbol_info_tick.return_value = cached_tick
        mt5.copy_ticks_range.return_value = [stream_tick]

        out = _raw_market_ticker("XAUUSD", detail="compact")

    assert out["bid"] == 3981.46
    assert out["ask"] == 3981.57
    assert "time_epoch" not in out
    assert out["usable_for_live_trading"] is False
    assert out["send_path_tick_fresh"] is False
    assert out["quote_source"] == "mt5.copy_ticks_range"
    assert out["quote_source_state"] == "refreshed_from_tick_stream"
    assert "quote_refresh_attempted" not in out


def test_market_ticker_prefers_stream_for_equal_timestamp_quote_conflict() -> None:
    now = 1_700_000_100.0
    cached_tick = SimpleNamespace(
        bid=1.15304,
        ask=1.15326,
        time=now - 1.0,
        time_msc=(now - 1.0) * 1000.0,
    )
    stream_tick = {
        "bid": 1.15308,
        "ask": 1.15322,
        "time": now - 1.0,
        "time_msc": (now - 1.0) * 1000.0,
    }
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth.time.time", return_value=now
    ), patch("mtdata.core.market_depth._use_client_tz", return_value=False):
        mt5.COPY_TICKS_ALL = 0
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
        )
        mt5.symbol_info_tick.return_value = cached_tick
        mt5.copy_ticks_range.return_value = [stream_tick]

        out = _raw_market_ticker("EURUSD", detail="full")

    assert out["bid"] == 1.15308
    assert out["ask"] == 1.15322
    assert out["quote_source"] == "mt5.copy_ticks_range"
    assert out["quote_source_state"] == "reconciled_equal_timestamp_conflict"
    assert out["quote_source_conflict"]["symbol_info_tick"] == {
        "bid": 1.15304,
        "ask": 1.15326,
    }


def test_market_ticker_price_field_omits_quote_source_conflict() -> None:
    now = 1_700_000_100.0
    cached_tick = SimpleNamespace(
        bid=1.15304,
        ask=1.15326,
        time=now - 1.0,
        time_msc=(now - 1.0) * 1000.0,
    )
    stream_tick = {
        "bid": 1.15308,
        "ask": 1.15322,
        "time": now - 1.0,
        "time_msc": (now - 1.0) * 1000.0,
    }
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth.time.time", return_value=now
    ), patch("mtdata.core.market_depth._use_client_tz", return_value=False):
        mt5.COPY_TICKS_ALL = 0
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
        )
        mt5.symbol_info_tick.return_value = cached_tick
        mt5.copy_ticks_range.return_value = [stream_tick]

        out = _raw_market_ticker("EURUSD", price_field="bid")

    assert out["success"] is True
    assert out["type"] == "price"
    assert out["field"] == "bid"
    assert out["price"] == 1.15308
    assert out["freshness_state"] == "live"
    assert out["quote_source_state"] == "reconciled_equal_timestamp_conflict"
    assert "quote_source_conflict" not in out
    assert "quote_conflict_pips" not in out
    assert "alternate_bid" not in out
    assert "alternate_ask" not in out


def test_market_ticker_uses_last_stream_tick_when_millisecond_timestamps_tie() -> None:
    now = 1_700_000_100.0
    cached_tick = SimpleNamespace(
        bid=1.15308,
        ask=1.15322,
        time=now - 1.0,
        time_msc=(now - 1.0) * 1000.0,
    )
    earlier_same_millisecond_tick = {
        "bid": 1.15304,
        "ask": 1.15326,
        "time": now - 1.0,
        "time_msc": (now - 1.0) * 1000.0,
    }
    latest_same_millisecond_tick = {
        "bid": 1.15308,
        "ask": 1.15322,
        "time": now - 1.0,
        "time_msc": (now - 1.0) * 1000.0,
    }
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth.time.time", return_value=now
    ), patch("mtdata.core.market_depth._use_client_tz", return_value=False):
        mt5.COPY_TICKS_ALL = 0
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
        )
        mt5.symbol_info_tick.return_value = cached_tick
        mt5.copy_ticks_range.return_value = [
            earlier_same_millisecond_tick,
            latest_same_millisecond_tick,
        ]

        out = _raw_market_ticker("EURUSD", detail="full")

    assert out["bid"] == 1.15308
    assert out["ask"] == 1.15322
    assert out["quote_source"] == "mt5.symbol_info_tick"
    assert out["quote_source_state"] == "current"
    assert "quote_source_conflict" not in out


def test_market_ticker_keeps_positive_cached_quote_over_locked_stream_conflict() -> None:
    now = 1_700_000_100.0
    cached_tick = SimpleNamespace(
        bid=1.15304,
        ask=1.15326,
        time=now - 1.0,
        time_msc=(now - 1.0) * 1000.0,
    )
    stream_tick = {
        "bid": 1.15310,
        "ask": 1.15310,
        "flags": 2,
        "time": now - 1.0,
        "time_msc": (now - 1.0) * 1000.0,
    }
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth.time.time", return_value=now
    ), patch("mtdata.core.market_depth._use_client_tz", return_value=False):
        mt5.COPY_TICKS_ALL = 0
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
        )
        mt5.symbol_info_tick.return_value = cached_tick
        mt5.copy_ticks_range.return_value = [stream_tick]

        out = _raw_market_ticker("EURUSD", detail="compact")

    assert out["bid"] == 1.15304
    assert out["ask"] == 1.15326
    assert out["quote_source"] == "mt5.symbol_info_tick"
    assert out["quote_source_state"] == "reconciled_one_sided_update"
    assert "quote_source_conflict" not in out
    assert out["usable_for_live_trading"] is True
    assert out["spread_valid"] is True
    assert out["spread_quality"] == "two_sided"


def test_market_ticker_marks_locked_quote_unusable() -> None:
    now = 1_700_000_100.0
    tick = SimpleNamespace(bid=1.1, ask=1.1, last=1.1, time=now - 1.0)
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth.time.time", return_value=now
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
        )
        mt5.symbol_info_tick.return_value = tick
        mt5.copy_ticks_range.return_value = []

        out = _raw_market_ticker("EURUSD", detail="compact")

    assert out["spread"] == 0.0
    assert out["spread_valid"] is False
    assert out["spread_quality"] == "locked"
    assert out["usable_for_live_trading"] is False
    assert "Locked quote" in out["warning"]


def test_market_ticker_mid_preserves_half_point_precision() -> None:
    now = 1_700_000_100.0
    tick = SimpleNamespace(
        bid=1.15398,
        ask=1.15399,
        last=1.15398,
        time=now - 1.0,
        time_msc=(now - 1.0) * 1000.0,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth.time.time", return_value=now
    ), patch("mtdata.core.market_depth._use_client_tz", return_value=False):
        mt5.COPY_TICKS_ALL = 0
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
        )
        mt5.symbol_info_tick.return_value = tick
        mt5.copy_ticks_range.return_value = []

        out = _raw_market_ticker("EURUSD", detail="full")

    assert out["mid"] == pytest.approx((out["bid"] + out["ask"]) / 2.0)
    assert out["mid"] == pytest.approx(1.153985)


def test_market_ticker_price_field_mid_uses_normalized_half_point_json() -> None:
    now = 1_700_000_100.0
    tick = SimpleNamespace(
        bid=1.15398,
        ask=1.15399,
        last=1.15398,
        time=now - 1.0,
        time_msc=(now - 1.0) * 1000.0,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth.time.time", return_value=now
    ), patch("mtdata.core.market_depth._use_client_tz", return_value=False):
        mt5.COPY_TICKS_ALL = 0
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
        )
        mt5.symbol_info_tick.return_value = tick
        mt5.copy_ticks_range.return_value = []

        out = _raw_market_ticker("EURUSD", price_field="mid")

    assert out["success"] is True
    assert out["field"] == "mid"
    assert out["price"] == 1.153985
    encoded = json.dumps(out)
    parsed = json.loads(encoded)
    assert parsed["price"] == 1.153985
    assert "1.153984999" not in encoded


def test_market_ticker_compact_explains_unrefreshable_future_tick() -> None:
    now = 1_700_000_100.0
    future_tick = SimpleNamespace(
        bid=1.1,
        ask=1.1002,
        last=1.1001,
        volume=1,
        time=now + 10.0,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth.time.time", return_value=now
    ), patch("mtdata.core.market_depth._use_client_tz", return_value=False):
        mt5.COPY_TICKS_ALL = 0
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
        )
        mt5.symbol_info_tick.return_value = future_tick
        mt5.copy_ticks_range.return_value = []

        out = _raw_market_ticker("EURUSD", detail="compact")

    assert out["freshness"] == "clock skew, tick timestamp 10s ahead of wall clock"
    assert out["freshness_state"] == "clock_skew"
    assert out["freshness_reason"] == "future_timestamp"
    assert out["data_age_seconds"] == 0.0
    assert out["timestamp_ahead_of_wall_clock"] is True
    assert out["timestamp_skew_tolerance_seconds"] == 10
    assert out["timestamp_in_future"] is True
    assert out["timestamp_skew_seconds"] == 10.0
    assert "MT5 time alignment" in out["timestamp_warning"]
    assert out["quote_source_state"] == "unverified_stale"


def test_market_ticker_reports_weekend_relaxed_freshness_basis() -> None:
    tick = SimpleNamespace(bid=1.1, ask=1.2, last=1.15, volume=5, time=100.0)
    freshness = {
        "data_age_seconds": 1000,
        "data_stale": False,
        "stale_after_seconds": 300,
        "freshness_policy_relaxed": True,
        "freshness_basis": "weekend_relaxed_max_3d",
    }
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth.build_tick_freshness_context",
        return_value=freshness,
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
            currency_profit="USD",
        )
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_ticker("EURUSD", price_field="mid")

    assert out["freshness_basis"] == "weekend_relaxed_max_3d"


def test_market_ticker_price_field_reports_unavailable_last() -> None:
    tick = SimpleNamespace(
        bid=1.1,
        ask=1.2,
        last=0.0,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5:
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
        )
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_ticker("EURUSD", price_field="last")

    assert out["success"] is False
    assert out["error_code"] == "market_ticker_price_unavailable"
    assert "last price is unavailable" in out["error"]
    assert "price" not in out
    assert out.get("usable_for_live_trading") is not True


def test_market_ticker_full_quote_marks_zero_last_unavailable() -> None:
    now = 1_700_000_100.0
    tick = SimpleNamespace(
        bid=1.1,
        ask=1.2,
        last=0.0,
        volume=5,
        time=now - 1.0,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth.time.time", return_value=now
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
        )
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_ticker("EURUSD", detail="full")

    assert out["success"] is True
    assert out["last"] is None
    assert out["last_unavailable"] is True
    assert out["bid"] == 1.1
    assert out["ask"] == 1.2


def test_market_ticker_full_detail_preserves_verbose_fields() -> None:
    tick = SimpleNamespace(
        bid=200.0,
        ask=201.0,
        last=200.5,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
            currency_profit="JPY",
            trade_contract_size=1.0,
        )
        mt5.symbol_info_tick.return_value = tick
        mt5.account_info.return_value = SimpleNamespace(currency="USD")
        out = _raw_market_ticker("BTCUSD", detail="full")

    assert out["last"] == 200.5
    assert out["tick_volume"] == 5
    assert out["spread_cost_per_lot"] == 100.0
    assert out["price_currency"] == "JPY"
    assert out["spread_cost_currency"] == "USD"
    assert out["pricing_basis"] == "per_1_lot_estimate"
    assert out["contract_size"] == 1.0
    assert out["pricing_basis_units"] == "broker_lot"
    assert out["units"]["contract_size"] == "contract_units_per_lot"
    assert out["meta"]["diagnostics"]["source"] == "mt5.symbol_info_tick"


def test_market_ticker_full_detail_rounds_age_fields() -> None:
    tick = SimpleNamespace(
        bid=1.17221,
        ask=1.17237,
        last=1.17230,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ), patch("mtdata.core.market_depth.time.time", return_value=1700000034.670966):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
            currency_profit="USD",
        )
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_ticker("EURUSD", detail="full")

    assert out["data_age_seconds"] == 34.7
    assert out["stale_after_seconds"] == 300
    assert "data_age_hours" not in out
    assert out["meta"]["diagnostics"]["data_freshness_seconds"] == 34.7


def test_market_ticker_treats_weekend_gap_as_closed_market() -> None:
    tick = SimpleNamespace(
        bid=1.17221,
        ask=1.17237,
        last=1.17230,
        volume=5,
        time=1779483360,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ), patch("mtdata.core.market_depth.time.time", return_value=1779656340.0):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
            currency_profit="USD",
        )
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_ticker("EURUSD", detail="full")

    assert out["data_stale"] is True
    assert out["usable_for_live_trading"] is False
    assert out["market_status"] == "closed"
    assert out["market_status_reason"] == "weekend"
    assert "latest completed session tick" in out["note"]
    assert "warning" in out


def test_market_ticker_includes_shared_meta_without_dropping_timezone_alias() -> None:
    tick = SimpleNamespace(
        bid=200.0,
        ask=201.0,
        last=200.5,
        volume=5,
        time=1700000000,
    )
    timezone_meta = {"used": {"tz": "UTC"}}
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ), patch(
        "mtdata.core.output_contract.build_runtime_timezone_meta",
        return_value=timezone_meta,
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
        )
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_ticker("BTCUSD")

    assert out["timezone"] == "UTC"
    assert out["meta"]["tool"] == "market_ticker"
    assert out["meta"]["runtime"]["timezone"] == timezone_meta


def test_market_ticker_rounds_tick_precision_noise() -> None:
    tick = SimpleNamespace(
        bid=1.17581,
        ask=1.1758999999999235,
        last=1.175856,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
            currency_profit="USD",
        )
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_ticker("EURUSD", detail="compact")

    assert out["bid"] == 1.17581
    assert out["ask"] == 1.1759
    assert out["spread_pips"] == 0.9
    assert out["units"]["spread"] == "absolute_price"
    assert out["units"]["spread_pips"] == "pips"
    assert out["spread"] == 0.00009
    assert out["spread_points"] == 9.0
    assert "last" not in out
    assert "spread_display" not in out
    assert out["spread_pct"] > 0


def test_market_depth_returns_connection_error_payload() -> None:
    with patch(
        "mtdata.core.market_depth.ensure_mt5_connection_or_raise",
        side_effect=MT5ConnectionError("Failed to connect to MetaTrader5. Ensure MT5 terminal is running."),
    ):
        out = _raw_market_depth_fetch("BTCUSD")

    assert out == {"error": "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."}


def test_market_depth_returns_env_gate_error_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("MTDATA_ENABLE_MARKET_DEPTH_FETCH", raising=False)

    out = _raw_market_depth_fetch("BTCUSD")

    assert out["error"] == (
        "market_depth_fetch is disabled. "
        "Set MTDATA_ENABLE_MARKET_DEPTH_FETCH=1 to enable it."
    )
    assert out["recommended_alternative"] == "market_ticker"


def test_market_depth_help_discloses_enablement_env() -> None:
    assert "MTDATA_ENABLE_MARKET_DEPTH_FETCH=1" in market_depth_fetch.__doc__


def test_tools_catalog_marks_market_depth_disabled(monkeypatch) -> None:
    monkeypatch.delenv("MTDATA_ENABLE_MARKET_DEPTH_FETCH", raising=False)

    row = _tool_catalog_row("market_depth_fetch")

    assert row["category"] == "market"
    assert row["enabled"] is False
    assert row["status"] == "disabled"
    assert row["enable_env"] == "MTDATA_ENABLE_MARKET_DEPTH_FETCH"
    assert row["recommended_alternative"] == "market_ticker"


def test_tools_catalog_marks_market_depth_enabled(monkeypatch) -> None:
    monkeypatch.setenv("MTDATA_ENABLE_MARKET_DEPTH_FETCH", "1")

    row = _tool_catalog_row("market_depth_fetch")

    assert row["enabled"] is True
    assert row["enable_env"] == "MTDATA_ENABLE_MARKET_DEPTH_FETCH"
    assert "status" not in row


def test_market_depth_tool_stays_registered_when_env_disabled(monkeypatch) -> None:
    monkeypatch.delenv("MTDATA_ENABLE_MARKET_DEPTH_FETCH", raising=False)
    reloaded = importlib.reload(market_depth_mod)
    try:
        assert "market_depth_fetch" in get_tool_functions()
        raw = getattr(reloaded.market_depth_fetch, "__wrapped__", reloaded.market_depth_fetch)
        payload = raw("BTCUSD")
        assert payload["error_code"] == "feature_disabled"
    finally:
        monkeypatch.setenv("MTDATA_ENABLE_MARKET_DEPTH_FETCH", "1")
        importlib.reload(reloaded)


def test_market_ticker_logs_finish_event(caplog) -> None:
    tick = SimpleNamespace(
        bid=200.0,
        ask=201.0,
        last=200.5,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ), caplog.at_level("DEBUG", logger="mtdata.core.market_depth"):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
        )
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_ticker("BTCUSD")

    assert out["success"] is True
    assert any(
        "event=finish operation=market_ticker success=True" in record.message
        for record in caplog.records
    )


def test_market_ticker_rewrites_invalid_symbol_selection_error() -> None:
    with patch("mtdata.core.market_depth.mt5") as mt5:
        mt5.symbol_select.return_value = False
        mt5.last_error.return_value = (-1, "Terminal: Call failed")
        mt5.symbols_get.return_value = [
            SimpleNamespace(
                name="FAKESYMBOL.NAS",
                description="Fake Symbol CFD",
                path="Stocks\\NASDAQ",
            )
        ]

        out = _raw_market_ticker("FAKESYMBOL")

    assert out["error"] == "Symbol 'FAKESYMBOL' was not found or is not available in MT5."
    assert out["success"] is False
    assert out["error_code"] == "symbol_not_found"
    assert out["operation"] == "market_ticker"
    assert out["request_id"]
    assert "symbols_list(search_term='FAKESYMBOL')" in out["remediation"]
    assert "symbols_search" not in out["remediation"]
    assert out["details"]["did_you_mean"] == [
        {
            "symbol": "FAKESYMBOL.NAS",
            "description": "Fake Symbol CFD",
            "group": "Stocks\\NASDAQ",
        }
    ]


def test_market_ticker_rejects_empty_quote_snapshot() -> None:
    tick = SimpleNamespace(bid=0.0, ask=0.0, last=0.0, volume=0, time=0)
    with patch("mtdata.core.market_depth.mt5") as mt5:
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
            currency_profit="USD",
        )
        mt5.symbol_info_tick.return_value = tick

        out = _raw_market_ticker("AAPL")

    assert out["success"] is False
    assert out["error_code"] == "market_ticker_quote_unavailable"
    assert "No usable quote data for AAPL" in out["error"]
    assert "symbols_list(search_term='AAPL')" in out["error"]
    assert out["operation"] == "market_ticker"


def test_market_ticker_suggests_live_extended_session_sibling() -> None:
    tick = SimpleNamespace(bid=0.0, ask=0.0, last=0.0, volume=0, time=0)
    related_live_symbols = [
        {
            "symbol": "AAPL-24",
            "session_type": "extended_24h",
            "quote_tool": "market_ticker",
        }
    ]
    with (
        patch("mtdata.core.market_depth.mt5") as mt5,
        patch(
            "mtdata.core.market_depth.find_live_extended_session_symbols",
            return_value=related_live_symbols,
        ),
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
            currency_profit="USD",
        )
        mt5.symbol_info_tick.return_value = tick

        out = _raw_market_ticker("AAPL")

    assert out["success"] is False
    assert out["error_code"] == "market_ticker_quote_unavailable"
    assert out["details"] == {
        "symbol": "AAPL",
        "related_live_symbols": related_live_symbols,
    }
    assert out["remediation"] == (
        "A live extended-session contract is available: call market_ticker for "
        "AAPL-24, then use that exact symbol for current quotes."
    )


def test_market_ticker_stale_success_includes_related_live_symbols() -> None:
    related_live_symbols = [
        {
            "symbol": "AAPL-24",
            "session_type": "extended_24h",
            "quote_tool": "market_ticker",
        }
    ]
    tick = SimpleNamespace(
        bid=100.0,
        ask=100.1,
        last=100.05,
        volume=1,
        time=1_700_000_000,
    )
    with (
        patch("mtdata.core.market_depth.mt5") as mt5,
        patch(
            "mtdata.core.market_depth.find_live_extended_session_symbols",
            return_value=related_live_symbols,
        ),
        patch(
            "mtdata.core.market_depth.enforce_quote_execution_readiness",
            side_effect=lambda payload, **kwargs: payload.update(
                {
                    "usable_for_live_trading": False,
                    "data_stale": True,
                }
            ),
        ),
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
            currency_profit="USD",
        )
        mt5.symbol_info_tick.return_value = tick

        out = _raw_market_ticker("AAPL")

    assert out["success"] is True
    assert out["related_live_symbols"] == related_live_symbols
