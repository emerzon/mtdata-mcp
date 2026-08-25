from __future__ import annotations

import sys
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from mtdata.core.trading import trade_history as _trade_history_tool
from mtdata.core.trading.account import trade_journal_analyze as _trade_journal_tool
from mtdata.core.trading.positions import normalize_trade_history_output
from mtdata.core.trading.requests import (
    TradeGetOpenRequest,
    TradeGetPendingRequest,
    TradeHistoryRequest,
    TradeJournalAnalyzeRequest,
)
from mtdata.core.trading.use_cases import run_trade_history
from mtdata.core.trading.use_cases.common import _trade_rows_to_dataframe
from mtdata.utils.mt5 import MT5ConnectionError, _mt5_epoch_to_utc


def _format_utc_seconds(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def test_trade_rows_to_dataframe_preserves_heterogeneous_fields() -> None:
    import pandas as pd

    frame = _trade_rows_to_dataframe(
        [
            {"ticket": 1, "profit": 100.0, "commission": -5.0},
            {"ticket": 2, "profit": 75.0, "swap": -2.0},
        ],
        pd_module=pd,
    )

    assert list(frame.columns) == ["ticket", "profit", "commission", "swap"]
    assert frame.loc[0, "commission"] == -5.0
    assert frame.loc[1, "swap"] == -2.0


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def trade_history(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradeHistoryRequest(**kwargs)
    with patch(
        "mtdata.core.trading.account.ensure_mt5_connection_or_raise", return_value=None
    ):
        if raw_output:
            return _unwrap(_trade_history_tool)(request=request)
        return _trade_history_tool(request=request, __cli_raw=False)


def trade_journal_analyze(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradeJournalAnalyzeRequest(**kwargs)
    with patch(
        "mtdata.core.trading.account.ensure_mt5_connection_or_raise", return_value=None
    ):
        if raw_output:
            return _unwrap(_trade_journal_tool)(request=request)
        return _trade_journal_tool(request=request, __cli_raw=False)


def _install_mock_mt5() -> tuple[MagicMock, object]:
    prev = sys.modules.get("MetaTrader5")
    mt5 = MagicMock()
    sys.modules["MetaTrader5"] = mt5
    return mt5, prev


def test_trade_history_deals_normalizes_time_to_utc_string() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="EURUSD")
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is True
    assert out["kind"] == "trade_history"
    assert out["history_kind"] == "deals"
    assert out["count"] == 1
    # Compact deals expose fill_time (not raw MT5 `time`).
    assert out["items"][0]["fill_time"] == _format_utc_seconds(
        _mt5_epoch_to_utc(1700000000)
    )
    assert out["timezone"] == "UTC"
    assert out["raw_time_basis"] == "mt5_utc_epoch"
    assert out["time_basis"] == "utc"
    assert out["raw_timestamp_mode"] == "native_utc"
    assert out["time_normalization"] == "mt5_utc_native"
    assert out["source"]["provider"] == "mt5"


def test_trade_history_filters_magic_before_pagination() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "magic"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1_700_000_001, symbol="EURUSD", magic=3001),
        Deal(ticket=2, time=1_700_000_002, symbol="EURUSD", magic=3002),
        Deal(ticket=3, time=1_700_000_003, symbol="EURUSD", magic=3001),
    ]

    try:
        with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
            out = trade_history(
                history_kind="deals",
                magic=3001,
                limit=1,
                detail="full",
                __cli_raw=True,
            )
    finally:
        if prev is not None:
            sys.modules["MetaTrader5"] = prev

    assert out["count"] == 1
    assert out["pagination"]["total"] == 2
    assert out["pagination"]["has_more"] is True
    assert out["items"][0]["magic"] == 3001
    assert out["request_echo"]["magic"] == 3001


def test_trade_history_default_page_returns_twenty_rows() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=index, time=1_700_000_000 + index, symbol="EURUSD")
        for index in range(1, 26)
    ]

    try:
        with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
            out = trade_history(history_kind="deals", __cli_raw=True)
    finally:
        if prev is not None:
            sys.modules["MetaTrader5"] = prev

    assert out["count"] == 20
    assert out["pagination"]["limit"] == 20
    assert out["pagination"]["has_more"] is True
    assert out["pagination"]["more_available"] == 5
    assert out["pagination"]["next_cursor"]


def test_trade_history_flags_future_broker_fill_timestamp() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1_700_001_000, symbol="EURUSD")
    ]

    with (
        patch("mtdata.core.trading.account._use_client_tz", lambda: False),
        patch("mtdata.core.trading.use_cases.history.time.time", return_value=1_700_000_000.0),
    ):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["data_quality"]["timestamp_anomaly_count"] == 1
    assert out["data_quality"]["max_fill_time_ahead_seconds"] == 1000.0
    assert "ahead of the observation clock" in out["warnings"][0]
    assert out["items"][0]["timestamp_anomaly"] is True
    assert out["items"][0]["original_fill_time"] == out["items"][0]["fill_time"]
    assert out["items"][0]["fill_time_future_seconds"] == 1000.0


def test_trade_history_cursor_survives_insertions_and_removals() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="EURUSD"),
        Deal(ticket=2, time=1700000060, symbol="EURUSD"),
        Deal(ticket=3, time=1700000120, symbol="EURUSD"),
        Deal(ticket=4, time=1700000180, symbol="EURUSD"),
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        first = trade_history(history_kind="deals", limit=2, __cli_raw=True)
        cursor = first["pagination"]["next_cursor"]
        mt5.history_deals_get.return_value = [
            Deal(ticket=1, time=1700000000, symbol="EURUSD"),
            Deal(ticket=2, time=1700000060, symbol="EURUSD"),
            Deal(ticket=5, time=1700000240, symbol="EURUSD"),
        ]
        second = trade_history(
            history_kind="deals",
            limit=2,
            cursor=cursor,
            __cli_raw=True,
        )

    assert first["success"] is True
    assert [item["deal_ticket"] for item in first["items"]] == [4, 3]
    assert first["pagination"]["mode"] == "keyset"
    assert first["pagination"]["snapshot_start"]
    assert first["pagination"]["snapshot_end"]
    assert [item["deal_ticket"] for item in second["items"]] == [2, 1]
    assert second["pagination"] == {
        "total": 3,
        "returned": 2,
        "offset": 2,
        "limit": 2,
        "has_more": False,
        "more_available": 0,
        "snapshot_start": first["pagination"]["snapshot_start"],
        "snapshot_end": first["pagination"]["snapshot_end"],
        "mode": "keyset",
    }
    assert not {"total_count", "offset", "limit", "has_more"} & second.keys()

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        ascending = trade_history(
            history_kind="deals",
            limit=2,
            order="asc",
            __cli_raw=True,
        )
    if prev is not None:
        sys.modules["MetaTrader5"] = prev
    assert [item["deal_ticket"] for item in ascending["items"]] == [1, 2]


def test_trade_history_cursor_keeps_equal_millisecond_tickets_complete() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "time_msc", "symbol"])
    mt5.history_deals_get.return_value = [
        Deal(11, 1_700_000_000, 1_700_000_000_500, "EURUSD"),
        Deal(13, 1_700_000_000, 1_700_000_000_500, "EURUSD"),
        Deal(12, 1_700_000_000, 1_700_000_000_500, "EURUSD"),
    ]

    try:
        cursors = [None]
        tickets = []
        bounds = []
        with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
            for _ in range(3):
                page = trade_history(
                    history_kind="deals",
                    limit=1,
                    cursor=cursors[-1],
                    __cli_raw=True,
                )
                tickets.append(page["items"][0]["deal_ticket"])
                call = mt5.history_deals_get.call_args
                bounds.append(call.args[:2])
                next_cursor = page["pagination"].get("next_cursor")
                if next_cursor:
                    cursors.append(next_cursor)
    finally:
        if prev is not None:
            sys.modules["MetaTrader5"] = prev

    assert tickets == [13, 12, 11]
    assert len(cursors) == 3
    assert bounds[0] == bounds[1] == bounds[2]


def test_trade_history_cursor_rejects_filter_mismatch_and_expiry() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1_700_000_000, symbol="EURUSD"),
        Deal(ticket=2, time=1_700_000_001, symbol="EURUSD"),
    ]

    try:
        with (
            patch("mtdata.core.trading.account._use_client_tz", lambda: False),
            patch("mtdata.core.trading.use_cases.history.time.time", return_value=1_000.0),
        ):
            first = trade_history(history_kind="deals", limit=1, __cli_raw=True)
        cursor = first["pagination"]["next_cursor"]
        with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
            mismatch = trade_history(
                history_kind="deals",
                limit=1,
                order="asc",
                cursor=cursor,
                __cli_raw=True,
            )
        with (
            patch("mtdata.core.trading.account._use_client_tz", lambda: False),
            patch("mtdata.core.trading.use_cases.history.time.time", return_value=4_601.0),
        ):
            expired = trade_history(
                history_kind="deals",
                limit=1,
                cursor=cursor,
                __cli_raw=True,
            )
    finally:
        if prev is not None:
            sys.modules["MetaTrader5"] = prev

    assert mismatch["success"] is False
    assert mismatch["error_code"] == "trade_history_invalid_cursor"
    assert "does not match" in mismatch["error"]
    assert expired["success"] is False
    assert expired["error_code"] == "trade_history_cursor_expired"
    assert "fresh snapshot" in expired["error"]


def test_trade_history_sorts_same_second_deals_by_millisecond_time() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "time_msc", "symbol"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=11, time=1_700_000_000, time_msc=1_700_000_000_052, symbol="EURUSD"),
        Deal(ticket=13, time=1_700_000_000, time_msc=1_700_000_000_682, symbol="EURUSD"),
        Deal(ticket=12, time=1_700_000_000, time_msc=1_700_000_000_367, symbol="EURUSD"),
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        descending = trade_history(
            history_kind="deals", limit=3, order="desc", detail="full", __cli_raw=True
        )
        latest = trade_history(
            history_kind="deals", limit=1, order="desc", detail="full", __cli_raw=True
        )
        ascending = trade_history(
            history_kind="deals", limit=3, order="asc", detail="full", __cli_raw=True
        )
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert [item["deal_ticket"] for item in descending["items"]] == [13, 12, 11]
    assert latest["items"][0]["deal_ticket"] == 13
    assert [item["deal_ticket"] for item in ascending["items"]] == [11, 12, 13]


def test_trade_history_sorts_same_second_orders_by_setup_milliseconds() -> None:
    mt5, prev = _install_mock_mt5()
    Order = namedtuple(
        "Order", ["ticket", "time_setup", "time_setup_msc", "time_done", "symbol"]
    )
    mt5.history_orders_get.return_value = [
        Order(21, 1_700_000_000, 1_700_000_000_050, 1_700_000_001, "EURUSD"),
        Order(23, 1_700_000_000, 1_700_000_000_850, 1_700_000_001, "EURUSD"),
        Order(22, 1_700_000_000, 1_700_000_000_400, 1_700_000_001, "EURUSD"),
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        descending = trade_history(
            history_kind="orders", limit=3, order="desc", detail="full", __cli_raw=True
        )
        ascending = trade_history(
            history_kind="orders", limit=3, order="asc", detail="full", __cli_raw=True
        )
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert [item["order_ticket"] for item in descending["items"]] == [23, 22, 21]
    assert [item["order_ticket"] for item in ascending["items"]] == [21, 22, 23]


def test_trade_history_rejects_future_only_explicit_range() -> None:
    mt5, prev = _install_mock_mt5()

    out = trade_history(
        history_kind="deals",
        start="2999-01-01T00:00:00Z",
        end="2999-01-02T00:00:00Z",
        __cli_raw=True,
    )
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is False
    assert out["error_code"] == "future_date_range"
    assert "future" in out["error"].lower()
    assert out["details"]["resolved_start"].startswith("2999-01-01")
    mt5.history_deals_get.assert_not_called()


def test_trade_journal_propagates_future_only_range_error() -> None:
    mt5, prev = _install_mock_mt5()

    out = trade_journal_analyze(
        start="2999-01-01T00:00:00Z",
        end="2999-01-02T00:00:00Z",
        __cli_raw=True,
    )
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is False
    assert out["error_code"] == "future_date_range"
    mt5.history_deals_get.assert_not_called()


def test_trade_history_warns_when_default_window_misses_open_event() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.history_deals_get.return_value = []
    old_epoch = (datetime.now(timezone.utc) - timedelta(days=8)).timestamp()
    mt5.positions_get.return_value = [
        SimpleNamespace(ticket=42, symbol="EURUSD", time=old_epoch)
    ]

    out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["history_incomplete_for_open_positions"] is True
    assert out["open_positions_outside_history_window_count"] == 1
    assert out["open_positions_outside_history_window"][0]["ticket"] == 42
    assert "default 7-day history window" in out["warnings"][0]


def test_trade_history_labels_account_currency_money_fields() -> None:
    out = normalize_trade_history_output(
        [
            {
                "ticket": 1,
                "symbol": "EURUSD",
                "profit": 12.5,
                "commission": -0.25,
                "swap": -0.1,
                "fee": -0.05,
            }
        ],
        request=TradeHistoryRequest(history_kind="deals"),
        account_currency="USD",
    )

    assert out["currency"] == "USD"
    for field in ("profit", "commission", "swap", "fee"):
        assert out["units"][field] == "account_currency"


def test_trade_history_rounds_money_fields_for_display() -> None:
    out = normalize_trade_history_output(
        [
            {
                "ticket": 1,
                "symbol": "EURUSD",
                "profit": -1.6800000000000002,
                "commission": -0.10000000000000002,
            }
        ],
        request=TradeHistoryRequest(history_kind="deals"),
    )

    assert out["items"][0]["profit"] == -1.68
    assert out["items"][0]["commission"] == -0.1

    full = normalize_trade_history_output(
        [
            {
                "ticket": 1,
                "symbol": "EURUSD",
                "profit": -1.6800000000000002,
                "commission": -0.10000000000000002,
            }
        ],
        request=TradeHistoryRequest(history_kind="deals", detail="full"),
    )

    assert full["items"][0]["profit"] == -1.68
    assert full["items"][0]["commission"] == -0.1
    assert "deal_details" not in full["items"][0]


def test_trade_history_deals_accept_simplenamespace_rows() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.history_deals_get.return_value = [
        SimpleNamespace(ticket=1, time=1700000000, symbol="EURUSD")
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is True
    assert out["count"] == 1
    assert out["items"][0]["deal_ticket"] == 1
    assert out["items"][0]["fill_time"] == _format_utc_seconds(
        _mt5_epoch_to_utc(1700000000)
    )


def test_trade_history_orders_normalizes_setup_and_done_times() -> None:
    mt5, prev = _install_mock_mt5()
    Order = namedtuple("Order", ["ticket", "time_setup", "time_done", "symbol"])
    mt5.history_orders_get.return_value = [
        Order(ticket=1, time_setup=1700000000, time_done=1700003600, symbol="EURUSD")
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="orders", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is True
    assert out["history_kind"] == "orders"
    assert out["count"] == 1
    # Compact orders expose placed_time/done_time (not raw time_setup/time_done).
    assert out["items"][0]["placed_time"] == _format_utc_seconds(
        _mt5_epoch_to_utc(1700000000)
    )
    assert out["items"][0]["done_time"] == _format_utc_seconds(
        _mt5_epoch_to_utc(1700003600)
    )
    assert out["timezone"] == "UTC"


def test_trade_history_orders_backfills_filled_zero_open_price() -> None:
    mt5, prev = _install_mock_mt5()
    Order = namedtuple(
        "Order",
        [
            "ticket",
            "time_setup",
            "time_done",
            "symbol",
            "state",
            "price_open",
            "price_current",
        ],
    )
    mt5.history_orders_get.return_value = [
        Order(
            ticket=1,
            time_setup=1700000000,
            time_done=1700003600,
            symbol="BTCUSD",
            state="Filled",
            price_open=0,
            price_current=77474.01,
        )
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="orders", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is True
    assert out["items"][0]["price_open"] == 77474.01
    assert out["items"][0]["price_current"] == 77474.01


def test_trade_history_compact_omits_parallel_normalized_rows() -> None:
    out = normalize_trade_history_output(
        [
            {
                "ticket": 11,
                "order": 22,
                "time": "2024-01-01 12:00:00",
                "time_msc": 1704110400000,
                "symbol": "EURUSD",
                "type": "Buy",
                "entry": "Out",
                "reason": "Expert",
                "volume": 0.5,
                "price": 1.2345,
                "profit": -1.0,
                "position_id": 33,
                "comment_visible_length": 8,
            }
        ],
        request=TradeHistoryRequest(history_kind="deals"),
    )

    row = out["items"][0]
    assert out["row_key"] == "items"
    assert row == {
        "fill_time": "2024-01-01 12:00:00",
        "deal_ticket": 11,
        "order_ticket": 22,
        "position_ticket": 33,
        "symbol": "EURUSD",
        "fill_side": "buy",
        "deal_effect": "close",
        "position_side": "short",
        "position_action": "close_short",
        "volume": 0.5,
        "price": 1.2345,
        "profit": -1.0,
    }
    assert "normalized_items" not in out


def test_trade_history_full_detail_uses_normalized_deal_items() -> None:
    raw_item = {
        "ticket": 11,
        "order": 22,
        "time": "2024-01-01 12:00:00",
        "time_msc": 1704110400000,
        "symbol": "EURUSD",
        "volume": 0.5,
        "price": 1.2345,
        "entry": "Out",
        "entry_code": 1,
        "reason": "Expert",
        "external_id": "diagnostic-noise",
        "comment": "closed",
    }
    out = normalize_trade_history_output(
        [raw_item],
        request=TradeHistoryRequest(history_kind="deals", detail="full"),
    )

    assert "normalized_items" not in out
    assert out["item_schema"] == "trade_history.v3"
    assert out["items"] == [
        {
            "fill_time": "2024-01-01 12:00:00",
            "fill_time_msc": 1704110400000,
            "deal_ticket": 11,
            "order_ticket": 22,
            "symbol": "EURUSD",
            "volume": 0.5,
            "price": 1.2345,
            "deal_effect": "close",
            "comment": "closed",
            "raw": {
                "entry": "Out",
                "entry_code": 1,
                "reason": "Expert",
                "external_id": "diagnostic-noise",
            },
        }
    ]
    assert out["request_echo"]["history_kind"] == "deals"
    assert out["request_echo"]["column_style"] == "snake_case"
    assert out["units"] == {"volume": "broker_lot"}


def test_trade_history_full_detail_uses_top_level_timezone_only() -> None:
    out = normalize_trade_history_output(
        [
            {
                "ticket": 11,
                "time": "2024-01-01 12:00:00",
                "symbol": "EURUSD",
                "volume": 0.5,
                "price": 1.2345,
                "timezone": "UTC",
            }
        ],
        request=TradeHistoryRequest(history_kind="deals", detail="full"),
    )

    assert out["timezone"] == "UTC"
    assert "timezone" not in out["items"][0]
    assert "deal_details" not in out["items"][0]


def test_trade_history_full_detail_applies_humanized_style() -> None:
    out = normalize_trade_history_output(
        [
            {
                "ticket": 11,
                "order": 22,
                "time": "2024-01-01 12:00:00",
                "symbol": "EURUSD",
                "volume": 0.5,
                "price": 1.2345,
                "entry_label": "Out",
                "comment": "closed",
            }
        ],
        request=TradeHistoryRequest(
            history_kind="deals",
            detail="full",
            column_style="humanized",
        ),
    )

    assert out["items"][0]["Deal Ticket"] == 11
    assert out["items"][0]["Symbol"] == "EURUSD"
    assert out["items"][0]["Comments"] == "closed"
    assert "Deal Details" not in out["items"][0]
    assert "deal_ticket" not in out["items"][0]
    assert "normalized_items" not in out
    assert out["request_echo"]["column_style"] == "humanized"


def test_trade_history_full_detail_uses_normalized_order_items() -> None:
    raw_item = {
        "ticket": 33,
        "time_setup": "2024-01-01 12:00:00",
        "time_done": "2024-01-01 12:05:00",
        "symbol": "GBPUSD",
        "volume_initial": 1.0,
        "price_open": 1.25,
        "price_current": 1.251,
        "state_label": "Filled",
        "state_code": 3,
        "provider_order_note": "kept",
    }
    out = normalize_trade_history_output(
        [raw_item],
        request=TradeHistoryRequest(history_kind="orders", detail="full"),
    )

    assert "normalized_items" not in out
    assert out["items"] == [
        {
            "placed_time": "2024-01-01 12:00:00",
            "done_time": "2024-01-01 12:05:00",
            "order_ticket": 33,
            "symbol": "GBPUSD",
            "volume_initial": 1.0,
            "price_open": 1.25,
            "price_current": 1.251,
            "state": "Filled",
            "raw": {"state_code": 3, "provider_order_note": "kept"},
        }
    ]
    assert out["items"][0]["raw"]["state_code"] == 3
    assert out["units"] == {"volume_initial": "broker_lot"}


def test_trade_history_normalizes_price_and_millisecond_artifacts() -> None:
    out = normalize_trade_history_output(
        [
            {
                "ticket": 11,
                "time": "2024-01-01 12:00:00",
                "time_msc": 1778822029181.0,
                "symbol": "EURUSD",
                "volume": 0.5,
                "price": 1.1627399999999999,
                "entry": "Out",
                "exit_trigger_price": 1.1627399999999999,
            }
        ],
        request=TradeHistoryRequest(history_kind="deals", detail="full"),
    )

    row = out["items"][0]
    assert row["price"] == 1.16274
    assert row["fill_time_msc"] == 1778822029181
    assert row["exit_trigger_price"] == 1.16274


def test_trade_history_compact_humanized_column_style_renames_order_times() -> None:
    out = normalize_trade_history_output(
        [
            {
                "ticket": 33,
                "time_setup": "2024-01-01 12:00:00",
                "time_done": "2024-01-01 12:05:00",
                "symbol": "GBPUSD",
                "volume_initial": 1.0,
                "price_current": 1.251,
                "state_label": "Filled",
            }
        ],
        request=TradeHistoryRequest(
            history_kind="orders",
            detail="compact",
            column_style="humanized",
        ),
    )

    # Compact humanized orders rename placed_time/done_time (not raw setup fields).
    assert out["items"][0]["Placed Time"] == "2024-01-01 12:00:00"
    assert out["items"][0]["Done Time"] == "2024-01-01 12:05:00"
    assert out["items"][0]["Initial Volume"] == 1.0
    assert "time_setup" not in out["items"][0]
    assert "Setup Time" not in out["items"][0]
    assert "normalized_items" not in out


def test_trade_history_order_type_uses_canonical_token() -> None:
    out = normalize_trade_history_output(
        [
            {
                "ticket": 33,
                "time_setup": "2024-01-01 12:00:00",
                "symbol": "GBPUSD",
                "type_label": "Sell Limit",
                "volume_initial": 1.0,
                "price_open": 1.25,
            }
        ],
        request=TradeHistoryRequest(history_kind="orders"),
    )

    assert out["items"][0]["order_type"] == "SELL_LIMIT"


def test_trade_history_filters_rows_by_symbol_even_if_mt5_returns_mixed_rows() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="BTCUSD"),
        Deal(ticket=2, time=1700003600, symbol="XAUUSD"),
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", symbol="BTCUSD", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is True
    assert out["scope"] == "symbol"
    assert out["count"] == 1
    assert out["items"][0]["symbol"] == "BTCUSD"


def test_trade_history_distinguishes_fill_and_position_side_filters() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_TYPE_BUY = 0
    mt5.DEAL_TYPE_SELL = 1
    mt5.DEAL_ENTRY_IN = 0
    mt5.DEAL_ENTRY_OUT = 1
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "type", "entry"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="EURUSD", type=0, entry=0),
        Deal(ticket=2, time=1700000060, symbol="EURUSD", type=1, entry=1),
        Deal(ticket=3, time=1700000120, symbol="EURUSD", type=1, entry=0),
        Deal(ticket=4, time=1700000180, symbol="EURUSD", type=0, entry=1),
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        long_out = trade_history(
            history_kind="deals",
            side="long",
            detail="full",
            __cli_raw=True,
        )
        short_out = trade_history(
            history_kind="deals",
            side="short",
            detail="full",
            __cli_raw=True,
        )
        buy_out = trade_history(
            history_kind="deals",
            side="buy",
            detail="full",
            __cli_raw=True,
        )
        sell_out = trade_history(
            history_kind="deals",
            side="sell",
            detail="full",
            __cli_raw=True,
        )
        long_page = trade_history(
            history_kind="deals",
            side="long",
            detail="full",
            limit=1,
            __cli_raw=True,
        )
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert long_out["request_echo"]["side"] == "long"
    assert long_out["side_filter"] == {
        "dimension": "position_side",
        "value": "long",
    }
    assert {row["deal_ticket"] for row in long_out["items"]} == {1, 2}
    assert {row["position_action"] for row in long_out["items"]} == {
        "open_long",
        "close_long",
    }
    assert {row["position_side"] for row in long_out["items"]} == {"long"}

    assert short_out["side_filter"]["dimension"] == "position_side"
    assert {row["deal_ticket"] for row in short_out["items"]} == {3, 4}
    assert {row["position_action"] for row in short_out["items"]} == {
        "open_short",
        "close_short",
    }
    assert {row["position_side"] for row in short_out["items"]} == {"short"}

    assert buy_out["side_filter"] == {
        "dimension": "fill_side",
        "value": "buy",
    }
    assert {row["deal_ticket"] for row in buy_out["items"]} == {1, 4}
    assert {row["fill_side"] for row in buy_out["items"]} == {"buy"}

    assert sell_out["side_filter"] == {
        "dimension": "fill_side",
        "value": "sell",
    }
    assert sell_out["filters_applied"]["side"] == "sell"
    assert sell_out["request_echo"]["side"] == "sell"
    assert {row["deal_ticket"] for row in sell_out["items"]} == {2, 3}
    assert {row["fill_side"] for row in sell_out["items"]} == {"sell"}

    assert long_page["count"] == 1
    assert long_page["pagination"]["total"] == 2
    assert long_page["pagination"]["has_more"] is True
    assert long_page["items"][0]["position_side"] == "long"


def test_trade_history_request_normalizes_buy_sell_aliases() -> None:
    assert TradeHistoryRequest(side="buy").side == "BUY"
    assert TradeHistoryRequest(side="sell").side == "SELL"
    with pytest.raises(
        ValidationError,
        match="side must be BUY, SELL, LONG, or SHORT",
    ):
        TradeHistoryRequest(side="weird")
    assert TradeHistoryRequest(side="long").side == "LONG"
    assert TradeHistoryRequest(side="short").side == "SHORT"
    assert TradeJournalAnalyzeRequest(side="long").side == "LONG"
    assert TradeGetOpenRequest(side="long").side == "BUY"
    assert TradeGetPendingRequest(side="short").side == "SELL"
    assert TradeHistoryRequest().detail == "compact"
    assert TradeHistoryRequest().limit == 20
    assert TradeJournalAnalyzeRequest().detail == "compact"
    assert TradeGetOpenRequest().detail == "compact"
    assert TradeGetPendingRequest().detail == "compact"


@pytest.mark.parametrize("legacy_pagination", [{"offset": 1}, {"page": 2}])
def test_trade_history_rejects_unstable_legacy_pagination(
    legacy_pagination,
) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TradeHistoryRequest(**legacy_pagination)


@pytest.mark.parametrize(
    "request_type",
    [TradeHistoryRequest, TradeGetOpenRequest, TradeGetPendingRequest],
)
@pytest.mark.parametrize("limit", [0, -1, 1.5])
def test_trade_query_requests_reject_invalid_limits(request_type, limit) -> None:
    with pytest.raises(ValidationError):
        request_type(limit=limit)


def test_trade_history_filters_orders_by_side_prefix() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.ORDER_TYPE_BUY_LIMIT = 2
    mt5.ORDER_TYPE_SELL_STOP = 5
    Order = namedtuple("Order", ["ticket", "time_setup", "symbol", "type"])
    mt5.history_orders_get.return_value = [
        Order(ticket=11, time_setup=1700000000, symbol="EURUSD", type=2),
        Order(ticket=12, time_setup=1700003600, symbol="EURUSD", type=5),
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="orders", side="sell", detail="full", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is True
    assert out["request_echo"]["side"] == "sell"
    assert out["count"] == 1
    assert out["items"][0]["order_ticket"] == 12
    assert out["items"][0]["order_type"] == "SELL_STOP"
    assert out["items"][0]["raw"]["type_code"] == 5


def test_trade_history_rejects_position_side_filter_for_orders() -> None:
    with pytest.raises(
        ValidationError,
        match="LONG/SHORT side filters require history_kind='deals'",
    ):
        TradeHistoryRequest(history_kind="orders", side="long")


def test_trade_history_deals_decodes_enum_codes_to_labels() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_TYPE_BUY = 0
    mt5.DEAL_ENTRY_IN = 0
    mt5.DEAL_REASON_CLIENT = 0
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "type", "entry", "reason"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="EURUSD", type=0, entry=0, reason=0)
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    row = out["items"][0]
    assert row["fill_side"] == "buy"
    assert "type" not in row
    assert "action" not in row
    assert row["deal_effect"] == "open"
    assert row["position_side"] == "long"
    assert row["position_action"] == "open_long"
    assert "entry" not in row
    assert "reason" not in row
    assert "type_code" not in row
    assert "entry_code" not in row
    assert "reason_code" not in row


def test_trade_history_deals_reports_closed_position_side() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_TYPE_SELL = 1
    mt5.DEAL_ENTRY_OUT = 1
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "type", "entry"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="EURUSD", type=1, entry=1)
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    row = out["items"][0]
    assert row["fill_side"] == "sell"
    assert "type" not in row
    assert "action" not in row
    assert row["deal_effect"] == "close"
    assert row["position_side"] == "long"
    assert row["position_action"] == "close_long"


def test_trade_history_deals_extracts_exit_trigger_from_comment() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_ENTRY_OUT = 1
    mt5.DEAL_REASON_SL = 4
    Deal = namedtuple(
        "Deal", ["ticket", "time", "symbol", "entry", "reason", "comment"]
    )
    mt5.history_deals_get.return_value = [
        Deal(
            ticket=1,
            time=1700000000,
            symbol="EURUSD",
            entry=1,
            reason=4,
            comment="[sl 64654.92]",
        )
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    row = out["items"][0]
    assert row["exit_trigger"] == "SL"
    assert row["exit_trigger_price"] == 64654.92
    assert row["deal_effect"] == "close"
    assert "exit_trigger_source" not in row


def test_trade_history_full_reports_comment_exit_trigger_source() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_ENTRY_OUT = 1
    mt5.DEAL_REASON_CLIENT = 0
    Deal = namedtuple(
        "Deal", ["ticket", "time", "symbol", "entry", "reason", "comment"]
    )
    mt5.history_deals_get.return_value = [
        Deal(
            ticket=1,
            time=1700000000,
            symbol="EURUSD",
            entry=1,
            reason=0,
            comment="[tp 1.12345]",
        )
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", detail="full", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    row = out["items"][0]
    assert row["exit_trigger"] == "TP"
    assert row["exit_trigger_price"] == 1.12345
    assert row["exit_trigger_source"] == "comment_tag"


def test_trade_history_deals_extracts_exit_trigger_from_reason_when_comment_missing() -> (
    None
):
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_ENTRY_OUT = 1
    mt5.DEAL_REASON_TP = 5
    Deal = namedtuple(
        "Deal", ["ticket", "time", "symbol", "entry", "reason", "comment"]
    )
    mt5.history_deals_get.return_value = [
        Deal(
            ticket=1,
            time=1700000000,
            symbol="EURUSD",
            entry=1,
            reason=5,
            comment="manual close",
        )
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", detail="full", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    row = out["items"][0]
    assert row["exit_trigger"] == "TP"
    assert "exit_trigger_price" not in row
    assert row["deal_effect"] == "close"
    assert row["exit_trigger_source"] == "mt5_reason"


def test_trade_history_deals_drops_non_informative_noise_columns() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_TYPE_BUY = 0
    mt5.DEAL_ENTRY_IN = 0
    mt5.DEAL_REASON_CLIENT = 0
    Deal = namedtuple(
        "Deal",
        [
            "ticket",
            "time",
            "symbol",
            "type",
            "entry",
            "reason",
            "time_msc",
            "external_id",
            "fee",
        ],
    )
    mt5.history_deals_get.return_value = [
        Deal(
            ticket=1,
            time=1700000000,
            symbol="EURUSD",
            type=0,
            entry=0,
            reason=0,
            time_msc=0,
            external_id="",
            fee=0.0,
        )
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    row = out["items"][0]
    assert "time_msc" not in row
    assert "external_id" not in row
    assert "fee" not in row


def test_trade_history_deals_keeps_fee_when_non_zero() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_TYPE_BUY = 0
    mt5.DEAL_ENTRY_IN = 0
    mt5.DEAL_REASON_CLIENT = 0
    Deal = namedtuple(
        "Deal",
        [
            "ticket",
            "time",
            "symbol",
            "type",
            "entry",
            "reason",
            "time_msc",
            "external_id",
            "fee",
        ],
    )
    mt5.history_deals_get.return_value = [
        Deal(
            ticket=1,
            time=1700000000,
            symbol="EURUSD",
            type=0,
            entry=0,
            reason=0,
            time_msc=0,
            external_id="",
            fee=1.25,
        )
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    row = out["items"][0]
    assert row["fee"] == 1.25


def test_trade_history_replaces_non_finite_values_with_none() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "profit"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="EURUSD", profit=float("nan"))
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert "profit" not in out["items"][0]


def test_run_trade_history_logs_finish_event(caplog) -> None:
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        history_deals_get=lambda from_dt, to_dt, symbol=None: [
            Deal(ticket=1, time=1700000000, symbol="EURUSD")
        ],
    )

    with caplog.at_level("DEBUG", logger="mtdata.core.trading.use_cases"):
        out = run_trade_history(
            TradeHistoryRequest(history_kind="deals"),
            gateway=gateway,
            use_client_tz=lambda: False,
            format_time_minimal=lambda ts: f"t{int(ts)}",
            format_time_minimal_local=lambda ts: f"lt{int(ts)}",
            mt5_epoch_to_utc=lambda ts: ts,
            parse_end_datetime=lambda value: None,
            parse_start_datetime=lambda value: None,
            normalize_limit=lambda value: value,
            comment_row_metadata=lambda comment: {},
            normalize_ticket_filter=lambda value, name: (None, None),
            normalize_minutes_back=lambda value: (None, None),
            decode_mt5_enum_label=lambda gateway, value, prefix=None: None,
            mt5_config=SimpleNamespace(get_client_tz=lambda: "UTC"),
        )

    assert isinstance(out, list)
    assert any(
        "event=finish operation=trade_history success=True" in record.message
        for record in caplog.records
    )


def test_run_trade_history_resolves_client_timezone_once_per_request() -> None:
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        history_deals_get=lambda from_dt, to_dt, symbol=None: [
            Deal(ticket=1, time=1700000000, symbol="EURUSD"),
            Deal(ticket=2, time=1700000060, symbol="EURUSD"),
        ],
    )
    mt5_config = SimpleNamespace(get_client_tz=MagicMock(return_value=timezone.utc))

    out = run_trade_history(
        TradeHistoryRequest(history_kind="deals"),
        gateway=gateway,
        use_client_tz=lambda: True,
        format_time_minimal=lambda ts: f"t{int(ts)}",
        format_time_minimal_local=lambda ts: f"lt{int(ts)}",
        mt5_epoch_to_utc=lambda ts: ts,
        parse_end_datetime=lambda value: None,
        parse_start_datetime=lambda value: None,
        normalize_limit=lambda value: value,
        comment_row_metadata=lambda comment: {},
        normalize_ticket_filter=lambda value, name: (None, None),
        normalize_minutes_back=lambda value: (None, None),
        decode_mt5_enum_label=lambda gateway, value, prefix=None: None,
        mt5_config=mt5_config,
    )

    assert len(out) == 2
    mt5_config.get_client_tz.assert_called_once_with()


def test_trade_history_filters_deals_by_position_ticket() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "position_id"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="BTCUSD", position_id=111),
        Deal(ticket=2, time=1700003600, symbol="BTCUSD", position_id=222),
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(
            history_kind="deals", symbol="BTCUSD", position_ticket=222, __cli_raw=True
        )
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["scope"] == "ticket"
    assert out["count"] == 1
    assert out["items"][0]["deal_ticket"] == 2
    assert out["items"][0]["position_ticket"] == 222


def test_trade_history_without_range_uses_full_history_start() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="EURUSD")
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is True
    from_dt, to_dt = mt5.history_deals_get.call_args.args[:2]
    assert abs((to_dt - from_dt) - (7 * 24 * 60 * 60)) < 1.0
    assert to_dt >= from_dt


def test_trade_history_surfaces_mt5_history_exception_with_actionable_hint() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.history_deals_get.side_effect = RuntimeError(
        "<built-in function history_deals_get> returned a result with an exception set"
    )

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["error"] == (
        "Failed to fetch deal history from MT5. "
        "Try narrowing the range with --minutes-back, --days, --start, or --end."
    )


def test_trade_history_rejects_start_with_minutes_back() -> None:
    out = trade_history(
        history_kind="deals",
        start="2026-03-01",
        minutes_back=30,
        __cli_raw=True,
    )

    assert out["error"] == "Use either start or minutes_back, not both."


def test_trade_history_rejects_invalid_side_filter() -> None:
    with pytest.raises(
        ValidationError,
        match="side must be BUY, SELL, LONG, or SHORT",
    ):
        TradeHistoryRequest(history_kind="deals", side="flat", detail="full")


def test_trade_history_compact_hides_comment_limit_metadata() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "comment"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="BTCUSD", comment="audit short"),
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", symbol="BTCUSD", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    row = out["items"][0]
    assert row["comment"] == "audit short"
    assert "comment_max_length" not in row
    assert "comment_visible_length" not in row
    assert "comment_may_be_truncated" not in row


def test_trade_history_compact_flags_possibly_truncated_comment() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "comment"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="BTCUSD", comment="x" * 31),
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", symbol="BTCUSD", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    row = out["items"][0]
    assert row["comment"] == "x" * 31
    assert row["comment_truncated"] is True
    assert "comment_max_length" not in row
    assert "comment_may_be_truncated" not in row


def test_trade_history_empty_deals_message_includes_orders_hint() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.history_deals_get.return_value = []

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["message"].startswith("No deals found")
    assert "--history-kind orders" in out["message"]


def test_trade_history_small_window_empty_orders_message_includes_propagation_note() -> (
    None
):
    mt5, prev = _install_mock_mt5()
    mt5.history_orders_get.return_value = []

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="orders", minutes_back=5, __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["message"].startswith("No orders found")
    assert (
        "may take up to a few minutes to reflect very recent events" in out["message"]
    )


def test_trade_history_minutes_back_empty_deals_message_mentions_window_not_orders() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.history_deals_get.return_value = []

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", minutes_back=60, __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert "in the last 60 minute(s)" in out["message"]
    assert "--history-kind orders" not in out["message"]


def test_trade_history_queries_minutes_back_as_absolute_mt5_epoch_window() -> None:
    captured: dict[str, object] = {}
    parsed_end = datetime(2026, 3, 1, 11, 0, 0)

    def history_deals_get(from_dt, to_dt, symbol=None):
        captured["from_dt"] = from_dt
        captured["to_dt"] = to_dt
        captured["symbol"] = symbol
        return []

    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        history_deals_get=history_deals_get,
    )

    out = run_trade_history(
        TradeHistoryRequest(
            history_kind="deals",
            symbol="BTCUSD",
            end="2026-03-01 11:00",
            minutes_back=60,
        ),
        gateway=gateway,
        use_client_tz=lambda: False,
        format_time_minimal=lambda ts: f"t{int(ts)}",
        format_time_minimal_local=lambda ts: f"lt{int(ts)}",
        mt5_epoch_to_utc=lambda ts: ts,
        parse_end_datetime=lambda value: parsed_end if value == "2026-03-01 11:00" else None,
        parse_start_datetime=lambda value: parsed_end if value == "2026-03-01 11:00" else None,
        normalize_limit=lambda value: value,
        comment_row_metadata=lambda comment: {},
        normalize_ticket_filter=lambda value, name: (None, None),
        normalize_minutes_back=lambda value: (value, None),
        decode_mt5_enum_label=lambda gateway, value, prefix=None: None,
        mt5_config=SimpleNamespace(
            get_client_tz=lambda: "UTC",
            get_time_offset_seconds=lambda at_time=None: 3 * 60 * 60,
        ),
    )

    assert captured["from_dt"] == datetime(
        2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc
    ).timestamp()
    assert captured["to_dt"] == datetime(
        2026, 3, 1, 11, 0, 0, tzinfo=timezone.utc
    ).timestamp()
    assert float(captured["to_dt"]) - float(captured["from_dt"]) == 60 * 60
    assert captured["symbol"] is None


def test_trade_history_deals_include_quote_price_currency() -> None:
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "price"])
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        history_deals_get=lambda *args, **kwargs: [
            Deal(ticket=1, time=1_700_000_000, symbol="EURUSD", price=1.1)
        ],
        symbol_info=lambda _symbol: SimpleNamespace(currency_profit="USD"),
        account_info=lambda: SimpleNamespace(currency="EUR"),
    )

    raw = run_trade_history(
        TradeHistoryRequest(history_kind="deals", minutes_back=60),
        gateway=gateway,
        use_client_tz=lambda: False,
        format_time_minimal=lambda ts: f"t{int(ts)}",
        format_time_minimal_local=lambda ts: f"lt{int(ts)}",
        mt5_epoch_to_utc=lambda ts: ts,
        parse_end_datetime=lambda value: None,
        parse_start_datetime=lambda value: None,
        normalize_limit=lambda value: value,
        comment_row_metadata=lambda comment: {},
        normalize_ticket_filter=lambda value, name: (None, None),
        normalize_minutes_back=lambda value: (value, None),
        decode_mt5_enum_label=lambda gateway, value, prefix=None: None,
        mt5_config=SimpleNamespace(
            get_client_tz=lambda: "UTC",
            get_time_offset_seconds=lambda at_time=None: 0,
        ),
    )
    out = normalize_trade_history_output(
        raw,
        request=TradeHistoryRequest(history_kind="deals", minutes_back=60),
        account_currency="EUR",
    )

    row = out["items"][0]
    assert out["currency"] == "EUR"
    assert row["price"] == 1.1
    assert row["price_currency"] == "USD"
    assert row["price_basis"] == "executed_fill"
    assert "price_currency_unavailable" not in row


def test_trade_history_says_when_quote_currency_is_unavailable() -> None:
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "price"])
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        history_deals_get=lambda *args, **kwargs: [
            Deal(ticket=1, time=1_700_000_000, symbol="EURUSD", price=1.1)
        ],
        symbol_info=lambda _symbol: None,
        account_info=lambda: SimpleNamespace(currency="EUR"),
    )

    raw = run_trade_history(
        TradeHistoryRequest(history_kind="deals", minutes_back=60),
        gateway=gateway,
        use_client_tz=lambda: False,
        format_time_minimal=lambda ts: f"t{int(ts)}",
        format_time_minimal_local=lambda ts: f"lt{int(ts)}",
        mt5_epoch_to_utc=lambda ts: ts,
        parse_end_datetime=lambda value: None,
        parse_start_datetime=lambda value: None,
        normalize_limit=lambda value: value,
        comment_row_metadata=lambda comment: {},
        normalize_ticket_filter=lambda value, name: (None, None),
        normalize_minutes_back=lambda value: (value, None),
        decode_mt5_enum_label=lambda gateway, value, prefix=None: None,
        mt5_config=SimpleNamespace(
            get_client_tz=lambda: "UTC",
            get_time_offset_seconds=lambda at_time=None: 0,
        ),
    )
    out = normalize_trade_history_output(
        raw,
        request=TradeHistoryRequest(history_kind="deals", minutes_back=60),
        account_currency="EUR",
    )

    row = out["items"][0]
    assert "price_currency" not in row
    assert row["price_basis"] == "executed_fill"
    assert row["price_currency_unavailable"] is True


@pytest.mark.parametrize("history_kind", ["deals", "orders"])
def test_trade_history_computes_epochs_without_signature_error(history_kind: str) -> None:
    captured: dict[str, object] = {}

    def history_get(from_dt, to_dt, symbol=None):
        captured["from_dt"] = from_dt
        captured["to_dt"] = to_dt
        return []

    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        history_deals_get=history_get,
        history_orders_get=history_get,
        symbol_info=lambda _symbol: SimpleNamespace(name="EURUSD"),
    )

    out = run_trade_history(
        TradeHistoryRequest(history_kind=history_kind, minutes_back=60),
        gateway=gateway,
        use_client_tz=lambda: False,
        format_time_minimal=lambda ts: f"t{int(ts)}",
        format_time_minimal_local=lambda ts: f"lt{int(ts)}",
        mt5_epoch_to_utc=lambda ts: ts,
        parse_end_datetime=lambda value: None,
        parse_start_datetime=lambda value: None,
        normalize_limit=lambda value: value,
        comment_row_metadata=lambda comment: {},
        normalize_ticket_filter=lambda value, name: (None, None),
        normalize_minutes_back=lambda value: (value, None),
        decode_mt5_enum_label=lambda gateway, value, prefix=None: None,
        mt5_config=SimpleNamespace(
            get_client_tz=lambda: "UTC",
            get_time_offset_seconds=lambda at_time=None: 3 * 60 * 60,
        ),
    )

    assert "unexpected keyword argument" not in str(out.get("error") or "")
    assert out.get("error_code") != "tool_error"
    assert "error" not in out
    assert isinstance(captured["from_dt"], (int, float))
    assert isinstance(captured["to_dt"], (int, float))
    assert float(captured["to_dt"]) - float(captured["from_dt"]) == 60 * 60
    assert "No " in out["message"]


def test_trade_journal_analyze_reaches_history_without_epoch_signature_error() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.history_deals_get.return_value = []
    mt5.account_info.return_value = SimpleNamespace(currency="USD")

    try:
        with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
            out = trade_journal_analyze(minutes_back=43200, __cli_raw=True)
    finally:
        if prev is not None:
            sys.modules["MetaTrader5"] = prev

    assert "unexpected keyword argument" not in str(out.get("error") or "")
    assert out.get("error_code") != "tool_error"
    mt5.history_deals_get.assert_called()


def test_trade_history_returns_connection_error_payload() -> None:
    with patch(
        "mtdata.core.trading.account.ensure_mt5_connection_or_raise",
        side_effect=MT5ConnectionError(
            "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
        ),
    ):
        out = _unwrap(_trade_history_tool)(
            request=TradeHistoryRequest(history_kind="deals")
        )

    assert (
        out["error"]
        == "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
    )
    assert out["success"] is False
    assert out["error_code"] == "mt5_connection_error"
    assert out["operation"] == "trade_history"
    assert not {"kind", "scope", "count", "items", "row_key"} & set(out)


def test_normalize_trade_history_output_preserves_upstream_error_metadata() -> None:
    request = TradeHistoryRequest(history_kind="deals", symbol="EURUSD")

    out = normalize_trade_history_output(
        {
            "error": "history lookup failed",
            "error_code": "trade_history_lookup_failed",
            "request_id": "broker-123",
            "details": {"range": "7d"},
            "checked_scopes": ["history"],
        },
        request=request,
    )

    assert out["success"] is False
    assert out["error"] == "history lookup failed"
    assert out["error_code"] == "trade_history_lookup_failed"
    assert out["request_id"] == "broker-123"
    assert out["details"] == {"range": "7d"}
    assert out["checked_scopes"] == ["history"]
    assert not {"kind", "scope", "count", "items", "row_key"} & set(out)


def test_trade_history_compact_detail_omits_echoed_filters() -> None:
    out = normalize_trade_history_output(
        [{"ticket": 1, "symbol": "EURUSD"}],
        request=TradeHistoryRequest(
            history_kind="deals",
            detail="compact",
            symbol="EURUSD",
            side="buy",
            limit=5,
        ),
    )

    assert out["history_kind"] == "deals"
    assert out["scope"] == "symbol"
    assert out["count"] == 1
    assert "symbol" not in out
    assert "side" not in out
    assert "limit" not in out


def test_trade_history_compact_includes_period_context() -> None:
    out = normalize_trade_history_output(
        [{"ticket": 1, "symbol": "EURUSD"}],
        request=TradeHistoryRequest(history_kind="deals", detail="compact"),
    )

    assert out["period_source"] == "default_lookback"
    assert out["minutes_back_effective"] == 10_080
    assert out["period_start"]
    assert out["period_end"]
    assert out["defaults_applied"] == {"lookback_minutes": 10_080}
    assert out["order"] == "desc"
    assert out["order_basis"] == "history_time"
    assert "note" in out
    keys = list(out)
    assert keys.index("items") < len(keys)


def test_trade_history_standard_period_context_precedes_items() -> None:
    out = normalize_trade_history_output(
        [{"ticket": 1, "symbol": "EURUSD"}],
        request=TradeHistoryRequest(history_kind="deals", detail="standard"),
    )

    assert out["period_source"] == "default_lookback"
    assert out["minutes_back_effective"] == 10080
    assert out["defaults_applied"] == {"lookback_minutes": 10080}
    assert out["period_timezone"] == "UTC"
    assert "note" in out
    keys = list(out)
    assert keys.index("period_start") < keys.index("items")


def test_trade_history_reports_explicit_period_context() -> None:
    out = normalize_trade_history_output(
        [{"ticket": 1, "symbol": "EURUSD"}],
        request=TradeHistoryRequest(
            history_kind="deals",
            detail="standard",
            start="2026-01-01 00:00",
            end="2026-01-03 00:00",
        ),
    )

    assert out["period_start"] == "2026-01-01T00:00:00Z"
    assert out["period_end"] == "2026-01-03T00:00:00Z"
    assert out["period_timezone"] == "UTC"
    assert out["period_source"] == "explicit_range"
    assert "minutes_back_effective" not in out
    assert "note" not in out


def test_trade_history_date_only_end_covers_full_day() -> None:
    out = normalize_trade_history_output(
        [{"ticket": 1, "symbol": "EURUSD"}],
        request=TradeHistoryRequest(
            history_kind="deals",
            detail="standard",
            start="2026-01-01",
            end="2026-01-03",
        ),
    )

    assert out["period_start"] == "2026-01-01T00:00:00Z"
    assert out["period_end"] == "2026-01-03T23:59:59Z"


def test_trade_history_empty_message_uses_enveloped_contract() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.history_deals_get.return_value = []

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is True
    assert out["kind"] == "trade_history"
    assert out["history_kind"] == "deals"
    assert out["count"] == 0
    assert out["items"] == []
    assert out["no_action"] is True
    assert out["message"].startswith("No deals found")


def test_trade_journal_analyze_summarizes_realized_exit_deals() -> None:
    history_rows = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "entry": "In",
            "type": "Buy",
            "position_ticket": 101,
            "profit": 0.0,
            "commission": -1.0,
            "swap": 0.0,
            "time": "2026-01-01 10:00",
            "volume": 0.1,
        },
        {
            "ticket": 2,
            "symbol": "EURUSD",
            "entry": "Out",
            "type": "Buy",
            "position_ticket": 101,
            "profit": 25.000000000000004,
            "commission": -1.0,
            "swap": -0.5,
            "exit_trigger": "TP",
            "time": "2026-01-01 12:00",
            "volume": 0.1,
        },
        {
            "ticket": 3,
            "symbol": "GBPUSD",
            "entry": "Out",
            "type": "Sell",
            "profit": -10.0,
            "commission": -0.5,
            "swap": 0.0,
            "exit_trigger": "SL",
            "time": "2026-01-02 09:00",
            "volume": 0.2,
        },
    ]

    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={
            "success": True,
            "currency": "USD",
            "count": len(history_rows),
            "items": history_rows,
        },
    ):
        out = trade_journal_analyze(
            start="2026-01-01 00:00",
            end="2026-01-03 00:00",
            detail="full",
            __cli_raw=True,
        )

    assert out["success"] is True
    assert out["currency"] == "USD"
    assert out["units"]["net_pnl"] == "account_currency"
    assert out["units"]["commission"] == "account_currency"
    assert out["period_start"] == "2026-01-01T00:00:00Z"
    assert out["period_end"] == "2026-01-03T00:00:00Z"
    assert out["period_timezone"] == "UTC"
    assert out["period_source"] == "explicit_range"
    assert "minutes_back_effective" not in out
    assert out["summary"]["closed_deals"] == 2
    assert out["summary"]["wins"] == 1
    assert out["summary"]["losses"] == 1
    assert out["summary"]["net_pnl"] == 12.0
    assert out["summary"]["profit_factor"] == 2.1429
    assert out["summary"]["expectancy"] == 6.0
    assert out["summary"]["sample_notice"]["code"] == "low_sample_unreliable_metrics"
    assert "avg_pnl" not in out["summary"]
    assert out["breakdowns"]["by_symbol"][0]["symbol"] == "EURUSD"
    assert out["pnl_basis"] == "mixed_round_trip_and_exit_only_costs"
    assert out["entry_cost_coverage"] == {
        "status": "partial",
        "method": "position_ticket_volume_pro_rata",
        "exit_deals": 2,
        "exit_deals_with_entry_cost_coverage": 1,
        "exit_deals_without_entry_cost_coverage": 1,
        "entry_commission_included": -1.0,
        "entry_fees_included": 0.0,
        "entry_costs_included": -1.0,
    }
    assert out["item_schema"] == "trade_journal_analyzed_exit.v3"
    assert [item["deal_ticket"] for item in out["items"]] == [2, 3]
    assert out["items"][0]["fill_time"] == "2026-01-01 12:00"
    assert out["items"][0]["exit_net_pnl"] == 23.5
    assert out["items"][0]["entry_commission"] == -1.0
    assert out["items"][0]["net_pnl"] == 22.5
    assert out["best_trades"][0]["deal_ticket"] == 2
    assert out["best_trades"][0]["profit"] == 25.0
    assert out["worst_trades"][0]["deal_ticket"] == 3


def test_trade_journal_magic_scope_excludes_other_strategies() -> None:
    history_rows = [
        {
            "deal_ticket": 1,
            "symbol": "EURUSD",
            "entry": "Out",
            "magic": 3001,
            "profit": 12.0,
            "volume": 0.1,
        },
        {
            "deal_ticket": 2,
            "symbol": "EURUSD",
            "entry": "Out",
            "magic": 3002,
            "profit": -50.0,
            "volume": 0.1,
        },
    ]
    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={"success": True, "items": history_rows},
    ) as history_mock:
        out = trade_journal_analyze(
            magic=3001,
            min_sample=1,
            detail="full",
            __cli_raw=True,
        )

    assert history_mock.call_args.args[0].magic == 3001
    assert out["magic"] == 3001
    assert out["summary"]["closed_deals"] == 1
    assert out["summary"]["net_pnl"] == 12.0
    assert [row["deal_ticket"] for row in out["items"]] == [1]


def test_trade_journal_allocates_entry_costs_across_partial_exits() -> None:
    history_rows = [
        {
            "deal_ticket": 10,
            "position_ticket": 100,
            "symbol": "EURUSD",
            "deal_effect": "open",
            "commission": -2.0,
            "fee": -0.4,
            "volume": 0.2,
        },
        {
            "deal_ticket": 11,
            "position_ticket": 100,
            "symbol": "EURUSD",
            "deal_effect": "close",
            "profit": 5.0,
            "commission": -0.5,
            "fee": -0.1,
            "volume": 0.1,
        },
        {
            "deal_ticket": 12,
            "position_ticket": 100,
            "symbol": "EURUSD",
            "deal_effect": "close",
            "profit": -1.0,
            "commission": -0.5,
            "fee": -0.1,
            "volume": 0.1,
        },
    ]

    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={"success": True, "items": history_rows},
    ):
        out = trade_journal_analyze(detail="full", __cli_raw=True)

    assert out["pnl_basis"] == "round_trip_allocated_entry_and_exit_costs"
    assert out["entry_cost_coverage"] == {
        "status": "complete",
        "method": "position_ticket_volume_pro_rata",
        "exit_deals": 2,
        "exit_deals_with_entry_cost_coverage": 2,
        "exit_deals_without_entry_cost_coverage": 0,
        "entry_commission_included": -2.0,
        "entry_fees_included": -0.4,
        "entry_costs_included": -2.4,
    }
    assert out["summary"]["net_pnl"] == 0.4
    assert [row["exit_net_pnl"] for row in out["items"]] == [4.4, -1.6]
    assert [row["entry_costs"] for row in out["items"]] == [-1.2, -1.2]
    assert [row["net_pnl"] for row in out["items"]] == [3.2, -2.8]
    assert all(row["pnl_cost_basis"] == "round_trip_allocated" for row in out["items"])
    assert "warnings" not in out


def test_trade_journal_side_filter_keeps_matching_entry_costs() -> None:
    history_rows = [
        {
            "deal_ticket": 10,
            "position_ticket": 100,
            "symbol": "EURUSD",
            "deal_effect": "open",
            "position_side": "long",
            "fill_side": "buy",
            "commission": -1.0,
            "volume": 0.1,
        },
        {
            "deal_ticket": 11,
            "position_ticket": 100,
            "symbol": "EURUSD",
            "deal_effect": "close",
            "position_side": "long",
            "fill_side": "sell",
            "profit": 5.0,
            "commission": -0.5,
            "volume": 0.1,
        },
        {
            "deal_ticket": 12,
            "position_ticket": 200,
            "symbol": "EURUSD",
            "deal_effect": "close",
            "position_side": "short",
            "fill_side": "buy",
            "profit": 9.0,
            "commission": -0.5,
            "volume": 0.1,
        },
    ]
    captured = {}

    def _fake_history(request):
        captured["request"] = request
        return {"success": True, "items": history_rows}

    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        side_effect=_fake_history,
    ):
        out = trade_journal_analyze(side="long", detail="full", __cli_raw=True)

    assert captured["request"].side is None
    assert out["sample_size"] == 1
    assert out["summary"]["net_pnl"] == 3.5
    assert out["entry_cost_coverage"]["status"] == "complete"
    assert out["items"][0]["deal_ticket"] == 11


def test_trade_journal_deal_filter_keeps_matching_entry_costs() -> None:
    history_rows = [
        {
            "deal_ticket": 10,
            "position_ticket": 100,
            "symbol": "EURUSD",
            "deal_effect": "open",
            "commission": -1.0,
            "volume": 0.1,
        },
        {
            "deal_ticket": 11,
            "position_ticket": 100,
            "symbol": "EURUSD",
            "deal_effect": "close",
            "profit": 5.0,
            "commission": -0.5,
            "volume": 0.1,
        },
        {
            "deal_ticket": 12,
            "position_ticket": 200,
            "symbol": "EURUSD",
            "deal_effect": "close",
            "profit": 9.0,
            "commission": -0.5,
            "volume": 0.1,
        },
    ]
    captured = {}

    def _fake_history(request):
        captured["request"] = request
        return {"success": True, "items": history_rows}

    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        side_effect=_fake_history,
    ):
        out = trade_journal_analyze(
            deal_ticket=11,
            detail="full",
            __cli_raw=True,
        )

    assert captured["request"].deal_ticket is None
    assert out["sample_size"] == 1
    assert out["summary"]["net_pnl"] == 3.5
    assert out["entry_cost_coverage"]["status"] == "complete"
    assert out["items"][0]["deal_ticket"] == 11


def test_trade_journal_includes_canonical_manual_close_deals() -> None:
    history_rows = [
        {
            "fill_time": "2026-07-16T20:26:44Z",
            "deal_ticket": 101,
            "order_ticket": 201,
            "position_ticket": 301,
            "symbol": "TSLA.NAS-24",
            "deal_effect": "close",
            "position_action": "close_long",
            "position_side": "long",
            "profit": 0.92,
            "commission": -0.02,
            "volume": 0.2,
            "comment": "MCP close",
        }
    ]

    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={
            "success": True,
            "count": 1,
            "items": history_rows,
        },
    ):
        out = trade_journal_analyze(detail="full", __cli_raw=True)

    assert out["summary"]["closed_deals"] == 1
    assert out["summary"]["net_pnl"] == 0.9
    assert out["meta"]["exit_deals"] == 1
    assert out["items"] == [
        {
            "deal_ticket": 101,
            "order_ticket": 201,
            "position_ticket": 301,
            "symbol": "TSLA.NAS-24",
            "fill_time": "2026-07-16T20:26:44Z",
            "side": "long",
            "exit_trigger": "Unspecified",
            "net_pnl": 0.9,
            "exit_net_pnl": 0.9,
            "entry_commission": None,
            "entry_fee": None,
            "entry_costs": None,
            "pnl_cost_basis": "exit_deal_only_entry_cost_unavailable",
            "profit": 0.92,
            "commission": -0.02,
            "swap": None,
            "fee": None,
            "volume": 0.2,
        }
    ]


def test_trade_journal_analyze_compact_returns_summary_only() -> None:
    history_rows = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "entry": "Out",
            "type": "Buy",
            "profit": 25.0,
            "commission": -1.0,
            "swap": -0.5,
            "exit_trigger": "TP",
            "time": "2026-01-01 12:00",
        },
        {
            "ticket": 2,
            "symbol": "GBPUSD",
            "entry": "Out",
            "type": "Sell",
            "profit": -10.0,
            "commission": -0.5,
            "swap": 0.0,
            "exit_trigger": "SL",
            "time": "2026-01-02 09:00",
        },
    ]

    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={
            "success": True,
            "count": len(history_rows),
            "items": history_rows,
        },
    ):
        out = trade_journal_analyze(__cli_raw=True)

    assert out["success"] is True
    assert out["period_source"] == "default_lookback"
    assert out["minutes_back_effective"] == 10_080
    assert out["period_start"]
    assert out["period_end"]
    assert "7-day" in out["note"]
    assert out["timezone"] == "UTC"
    assert out["summary"]["closed_deals"] == 2
    assert out["units"]["win_rate"] == "fraction"
    assert "items" not in out
    assert "item_schema" not in out
    assert "breakdowns" not in out
    assert "best_trades" not in out
    assert "worst_trades" not in out
    assert out["sample_provenance"] == {
        "output_item_limit": 50,
        "history_rows_scanned": 2,
        "period_exit_deals_analyzed": 2,
        "analysis_complete": True,
        "items_returned": 0,
        "items_truncated": False,
        "history_has_more": False,
    }


def test_trade_journal_pages_until_period_history_is_complete() -> None:
    pages = {
        None: [
            {"ticket": 1, "symbol": "EURUSD", "entry": "Out", "profit": 3.0},
            {"ticket": 2, "symbol": "GBPUSD", "entry": "Out", "profit": 4.0},
        ],
        "page-2": [
            {"ticket": 3, "symbol": "EURUSD", "entry": "Out", "profit": 2.0},
            {"ticket": 4, "symbol": "GBPUSD", "entry": "Out", "profit": -1.0},
        ],
    }
    observed_cursors = []
    observed_bounds = []

    def _fake_history(request):
        observed_cursors.append(request.cursor)
        observed_bounds.append((request.start, request.end))
        rows = pages[request.cursor]
        return {
            "success": True,
            "count": len(rows),
            "items": rows,
            "pagination": {
                "total": 4,
                "returned": len(rows),
                "offset": 0 if request.cursor is None else 2,
                "limit": request.limit,
                "has_more": request.cursor is None,
                "more_available": 2 if request.cursor is None else 0,
                **(
                    {"next_cursor": "page-2"}
                    if request.cursor is None
                    else {}
                ),
            },
        }

    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        side_effect=_fake_history,
    ):
        out = trade_journal_analyze(limit=2, __cli_raw=True)

    assert observed_cursors == [None, "page-2"]
    assert observed_bounds[0] == observed_bounds[1]
    assert all(start and end for start, end in observed_bounds)
    assert out["sample_size"] == 4
    assert out["sample_provenance"] == {
        "output_item_limit": 2,
        "history_rows_scanned": 4,
        "period_exit_deals_analyzed": 4,
        "analysis_complete": True,
        "items_returned": 0,
        "items_truncated": False,
        "history_has_more": False,
        "history_rows_available": 4,
    }


def test_trade_journal_limit_caps_items_not_period_statistics() -> None:
    rows = [
        {
            "ticket": index,
            "symbol": "EURUSD",
            "entry": "Out",
            "profit": float(index),
        }
        for index in range(1, 61)
    ]
    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={
            "success": True,
            "count": len(rows),
            "items": rows,
            "pagination": {
                "total": len(rows),
                "returned": len(rows),
                "offset": 0,
                "limit": 100,
                "has_more": False,
            },
        },
    ):
        out = trade_journal_analyze(limit=50, detail="full", __cli_raw=True)

    assert out["summary"]["closed_deals"] == 60
    assert out["summary"]["net_pnl"] == 1830.0
    assert out["sample_size"] == 60
    assert len(out["items"]) == 50
    assert out["sample_provenance"] == {
        "output_item_limit": 50,
        "history_rows_scanned": 60,
        "period_exit_deals_analyzed": 60,
        "analysis_complete": True,
        "items_returned": 50,
        "items_truncated": True,
        "history_has_more": False,
        "history_rows_available": 60,
    }


def test_trade_journal_excludes_future_timestamp_anomalies() -> None:
    rows = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "entry": "Out",
            "profit": 99.0,
            "timestamp_anomaly": True,
        },
        {"ticket": 2, "symbol": "EURUSD", "entry": "Out", "profit": 2.0},
    ]
    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={"success": True, "count": 2, "items": rows},
    ):
        out = trade_journal_analyze(limit=2, __cli_raw=True)

    assert out["sample_size"] == 1
    assert out["summary"]["net_pnl"] == 2.0
    assert "Excluded 1 future-dated" in out["warnings"][0]


def test_trade_journal_analyze_standard_uses_lite_symbol_breakdown() -> None:
    history_rows = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "entry": "Out",
            "type": "Buy",
            "position_side": "short",
            "profit": 25.0,
            "commission": -1.0,
            "swap": -0.5,
            "exit_trigger": "TP",
            "time": "2026-01-01 12:00",
        },
        {
            "ticket": 2,
            "symbol": "GBPUSD",
            "entry": "Out",
            "type": "Sell",
            "position_side": "long",
            "profit": -10.0,
            "commission": -0.5,
            "swap": 0.0,
            "exit_trigger": "SL",
            "time": "2026-01-02 09:00",
        },
    ]

    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={
            "success": True,
            "count": len(history_rows),
            "items": history_rows,
        },
    ):
        out = trade_journal_analyze(detail="standard", __cli_raw=True)

    assert list(out["breakdowns"]) == ["by_symbol"]
    assert set(out["breakdowns"]["by_symbol"][0]) == {
        "symbol",
        "closed_deals",
        "win_rate",
        "win_rate_pct",
        "net_pnl",
        "expectancy",
    }
    assert "by_side" not in out["breakdowns"]
    assert "by_exit_trigger" not in out["breakdowns"]
    assert "best_trades" not in out
    assert "worst_trades" not in out


def test_trade_journal_analyze_summary_adds_lite_side_breakdown() -> None:
    history_rows = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "entry": "Out",
            "type": "Buy",
            "position_side": "short",
            "profit": 25.0,
            "commission": -1.0,
            "swap": -0.5,
            "exit_trigger": "TP",
            "time": "2026-01-01 12:00",
        },
        {
            "ticket": 2,
            "symbol": "GBPUSD",
            "entry": "Out",
            "type": "Sell",
            "position_side": "long",
            "profit": -10.0,
            "commission": -0.5,
            "swap": 0.0,
            "exit_trigger": "SL",
            "time": "2026-01-02 09:00",
        },
    ]

    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={
            "success": True,
            "count": len(history_rows),
            "items": history_rows,
        },
    ):
        out = trade_journal_analyze(detail="summary", __cli_raw=True)

    assert list(out["breakdowns"]) == ["by_symbol", "by_side"]
    assert set(out["breakdowns"]["by_side"][0]) == {
        "side",
        "closed_deals",
        "win_rate",
        "win_rate_pct",
        "net_pnl",
        "expectancy",
    }
    assert {row["side"] for row in out["breakdowns"]["by_side"]} == {"long", "short"}
    assert "by_exit_trigger" not in out["breakdowns"]
    assert "best_trades" not in out
    assert "worst_trades" not in out


def test_trade_journal_analyze_reports_explicit_minutes_back_window() -> None:
    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={
            "success": True,
            "count": 0,
            "items": [],
        },
    ):
        out = trade_journal_analyze(minutes_back=60, detail="full", __cli_raw=True)

    assert out["success"] is True
    assert out["period_source"] == "minutes_back"
    assert out["timezone"] == "UTC"
    assert out["minutes_back_requested"] == 60
    assert out["minutes_back_effective"] == 60
    assert "note" not in out
    assert "Only 0 realized exit deal" in out["sample_warning"]
    assert "Increase minutes_back" in out["sample_warning"]
    assert "Increase limit" not in out["sample_warning"]


@pytest.mark.parametrize("minutes_back", [0, -1])
def test_trade_journal_analyze_rejects_non_positive_minutes_back(
    minutes_back: int,
) -> None:
    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
    ) as history:
        out = trade_journal_analyze(minutes_back=minutes_back, __cli_raw=True)

    assert out["success"] is False
    assert out["error"] == "minutes_back must be a positive integer."
    assert "minutes_back_effective" not in out
    history.assert_not_called()


def test_trade_journal_analyze_filters_best_worst_by_pnl_sign() -> None:
    """Verify that best_trades only contains wins and worst_trades only contains losses.
    
    This test validates the fix for the logic error where best_trades and worst_trades
    were mixed together, just sorted differently.
    """
    history_rows = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "entry": "Out",
            "type": "Buy",
            "profit": 0.82,
            "commission": 0.0,
            "swap": 0.0,
            "exit_trigger": "TP",
            "time": "2026-01-01 10:00",
        },
        {
            "ticket": 2,
            "symbol": "USDJPY",
            "entry": "Out",
            "type": "Buy",
            "profit": 0.04,
            "commission": 0.0,
            "swap": 0.0,
            "exit_trigger": "TP",
            "time": "2026-01-01 11:00",
        },
        {
            "ticket": 3,
            "symbol": "EURUSD",
            "entry": "Out",
            "type": "Sell",
            "profit": -0.23,
            "commission": 0.0,
            "swap": 0.0,
            "exit_trigger": "SL",
            "time": "2026-01-01 12:00",
        },
    ]

    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={
            "success": True,
            "count": len(history_rows),
            "items": history_rows,
        },
    ):
        out = trade_journal_analyze(detail="full", __cli_raw=True)

    # Verify metrics are correct
    assert out["summary"]["wins"] == 2
    assert out["summary"]["losses"] == 1
    assert out["summary"]["win_rate"] == 0.6667
    assert "win_rate_display" not in out["summary"]
    assert out["units"]["win_rate"] == "fraction"
    assert out["summary"]["best_trade"] == 0.82
    assert out["summary"]["worst_trade"] == -0.23

    # Verify best_trades only contains winners (positive P&L)
    assert len(out["best_trades"]) == 2
    for trade in out["best_trades"]:
        assert trade["net_pnl"] > 0, (
            "best_trades should only contain wins, but found deal "
            f"{trade['deal_ticket']} with net_pnl {trade['net_pnl']}"
        )

    # Verify worst_trades only contains losers (negative P&L)
    assert len(out["worst_trades"]) == 1
    for trade in out["worst_trades"]:
        assert trade["net_pnl"] < 0, (
            "worst_trades should only contain losses, but found deal "
            f"{trade['deal_ticket']} with net_pnl {trade['net_pnl']}"
        )

    # Verify specific tickets in correct lists
    best_tickets = {trade["deal_ticket"] for trade in out["best_trades"]}
    worst_tickets = {trade["deal_ticket"] for trade in out["worst_trades"]}
    
    assert 1 in best_tickets  # EURUSD +0.82
    assert 2 in best_tickets  # USDJPY +0.04
    assert 3 in worst_tickets  # EURUSD -0.23


def test_trade_journal_analyze_rounds_float_noise() -> None:
    history_rows = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "entry": "Out",
            "type": "Buy",
            "profit": 1.9100000000000001,
            "commission": 0.0,
            "swap": 0.0,
            "exit_trigger": "TP",
            "time": "2026-01-01 10:00",
        },
        {
            "ticket": 2,
            "symbol": "EURUSD",
            "entry": "Out",
            "type": "Buy",
            "profit": -7.4399999999999995,
            "commission": 0.0,
            "swap": 0.0,
            "exit_trigger": "SL",
            "time": "2026-01-01 11:00",
        },
    ]

    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={
            "success": True,
            "count": len(history_rows),
            "items": history_rows,
        },
    ):
        out = trade_journal_analyze(detail="full", __cli_raw=True)

    assert out["summary"]["net_pnl"] == -5.53
    assert out["summary"]["gross_loss"] == 7.44
    assert out["summary"]["profit_factor"] == 0.2567
    assert out["summary"]["expectancy"] == -2.765
    assert out["summary"]["avg_win"] == 1.91
    assert out["summary"]["avg_loss"] == 7.44
    assert out["worst_trades"][0]["net_pnl"] == -7.44


def test_trade_journal_analyze_returns_message_when_no_exit_deals_found() -> None:
    history_rows = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "entry": "In",
            "type": "Buy",
            "profit": 0.0,
            "commission": -1.0,
            "time": "2026-01-01 10:00",
        },
    ]

    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={
            "success": True,
            "count": len(history_rows),
            "items": history_rows,
        },
    ):
        out = trade_journal_analyze(__cli_raw=True)

    assert out["success"] is True
    assert out["summary"]["closed_deals"] == 0
    assert "No realized exit deals found" in out["message"]
    assert "Only 0 realized exit deal" in out["sample_warning"]


def test_trade_journal_analyze_propagates_history_errors() -> None:
    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={"error": "boom"},
    ):
        out = trade_journal_analyze(__cli_raw=True)

    assert out == {"error": "boom"}


def test_trade_journal_analyze_applies_side_filter_after_history_reconciliation() -> None:
    captured = {}

    def _fake_history(request):
        captured["request"] = request
        return {"success": True, "count": 0, "items": [], "message": "No deals found"}

    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        side_effect=_fake_history,
    ):
        out = trade_journal_analyze(side="long", __cli_raw=True)

    assert captured["request"].side is None
    assert out["success"] is True
    assert out["summary"]["closed_deals"] == 0
    assert out["side_filter"] == {
        "dimension": "position_side",
        "value": "long",
    }


def test_trade_journal_position_side_filter_never_returns_opposite_exits() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_TYPE_BUY = 0
    mt5.DEAL_TYPE_SELL = 1
    mt5.DEAL_ENTRY_OUT = 1
    Deal = namedtuple(
        "Deal",
        [
            "ticket",
            "time",
            "symbol",
            "type",
            "entry",
            "profit",
            "commission",
            "swap",
            "fee",
            "volume",
        ],
    )
    mt5.history_deals_get.return_value = [
        Deal(1, 1700000000, "EURUSD", 1, 1, 10.0, -0.5, 0.0, 0.0, 0.1),
        Deal(2, 1700000060, "EURUSD", 0, 1, -5.0, -0.5, 0.0, 0.0, 0.1),
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        long_out = trade_journal_analyze(
            side="long",
            detail="full",
            limit=10,
            __cli_raw=True,
        )
        short_out = trade_journal_analyze(
            side="short",
            detail="full",
            limit=10,
            __cli_raw=True,
        )
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert long_out["side_filter"]["dimension"] == "position_side"
    assert long_out["sample_size"] == 1
    assert {row["side"] for row in long_out["items"]} == {"long"}
    assert short_out["sample_size"] == 1
    assert {row["side"] for row in short_out["items"]} == {"short"}


def test_trade_journal_request_rejects_invalid_side() -> None:
    with pytest.raises(
        ValidationError,
        match="side must be BUY, SELL, LONG, or SHORT",
    ):
        TradeJournalAnalyzeRequest(side="sideways")
