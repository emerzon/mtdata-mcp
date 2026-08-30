from __future__ import annotations

import sys
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mtdata.core.output_profiles import apply_public_output_profile
from mtdata.core.trading import trade_history as _trade_history_tool
from mtdata.core.trading.account import trade_journal_analyze as _journal_tool
from mtdata.core.trading.positions import normalize_trade_history_output
from mtdata.core.trading.requests import TradeHistoryRequest, TradeJournalAnalyzeRequest


def _unwrap(function):
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    return function


def test_origin_magic_keeps_manual_exit_in_history_and_journal() -> None:
    previous_mt5 = sys.modules.get("MetaTrader5")
    mt5 = MagicMock()
    mt5.DEAL_TYPE_BUY = 0
    mt5.DEAL_TYPE_SELL = 1
    mt5.DEAL_ENTRY_IN = 0
    mt5.DEAL_ENTRY_OUT = 1
    mt5.DEAL_REASON_CLIENT = 0
    mt5.DEAL_REASON_EXPERT = 3
    mt5.account_info.return_value = SimpleNamespace(currency="USD")
    mt5.symbol_info.return_value = SimpleNamespace(currency_profit="USD")
    mt5.positions_get.return_value = []
    Deal = namedtuple(
        "Deal",
        [
            "ticket",
            "order",
            "time",
            "position_id",
            "symbol",
            "type",
            "entry",
            "reason",
            "magic",
            "volume",
            "price",
            "profit",
            "commission",
            "swap",
            "fee",
            "comment",
        ],
    )
    mt5.history_deals_get.return_value = [
        Deal(
            1,
            101,
            1_700_000_000,
            900,
            "EURUSD",
            0,
            0,
            3,
            3001,
            0.1,
            1.1,
            0.0,
            -1.0,
            0.0,
            0.0,
            "EA entry",
        ),
        Deal(
            2,
            102,
            1_700_000_060,
            900,
            "EURUSD",
            1,
            1,
            0,
            0,
            0.1,
            1.2,
            10.0,
            -0.5,
            -0.2,
            -0.1,
            "manual exit",
        ),
    ]
    sys.modules["MetaTrader5"] = mt5

    try:
        with (
            patch(
                "mtdata.core.trading.account.ensure_mt5_connection_or_raise",
                return_value=None,
            ),
            patch("mtdata.core.trading.account._use_client_tz", lambda: False),
        ):
            history = _unwrap(_trade_history_tool)(
                request=TradeHistoryRequest(
                    history_kind="deals",
                    magic=3001,
                    position_ticket=900,
                    detail="full",
                    limit=10,
                )
            )
            journal = _unwrap(_journal_tool)(
                request=TradeJournalAnalyzeRequest(
                    magic=3001,
                    position_ticket=900,
                    detail="full",
                    limit=10,
                    min_sample=1,
                )
            )
    finally:
        if previous_mt5 is None:
            sys.modules.pop("MetaTrader5", None)
        else:
            sys.modules["MetaTrader5"] = previous_mt5

    assert history["pagination"]["total"] == 2
    assert history["item_schema"] == "trade_history.v4"
    history_by_ticket = {row["deal_ticket"]: row for row in history["items"]}
    assert history_by_ticket[1]["deal_magic"] == 3001
    assert history_by_ticket[2]["deal_magic"] == 0
    assert {row["attributed_magic"] for row in history["items"]} == {3001}
    assert {row["attribution_method"] for row in history["items"]} == {
        "position_origin_entry"
    }
    assert history_by_ticket[2]["net_pnl"] == 9.2
    assert history_by_ticket[2]["profit_basis"] == (
        "broker_profit_excluding_cost_components"
    )

    assert journal["summary"]["closed_deals"] == 1
    assert journal["items"][0]["deal_ticket"] == 2
    assert journal["items"][0]["deal_magic"] == 0
    assert journal["items"][0]["attributed_magic"] == 3001
    assert journal["items"][0]["exit_net_pnl"] == 9.2
    assert journal["items"][0]["net_pnl"] == 8.2


def test_compact_history_keeps_versioned_stable_monetary_columns() -> None:
    normalized = normalize_trade_history_output(
        [
            {
                "ticket": 1,
                "order": 100,
                "position_id": 100,
                "symbol": "EURUSD",
                "entry": "In",
                "volume": 0.1,
                "profit": 0.0,
                "commission": 0.0,
                "swap": 0.0,
                "fee": 0.0,
            },
            {
                "ticket": 2,
                "order": 101,
                "position_id": 100,
                "symbol": "EURUSD",
                "entry": "Out",
                "volume": 0.1,
                "profit": 10.0,
                "commission": -1.0,
                "swap": -0.5,
                "fee": -0.25,
            },
        ],
        request=TradeHistoryRequest(history_kind="deals", detail="compact"),
        account_currency="USD",
    )

    compact = apply_public_output_profile(
        normalized,
        tool_name="trade_history",
        detail="compact",
    )

    assert compact["item_schema"] == "trade_history.v4"
    money_fields = {"profit", "commission", "swap", "fee", "net_pnl"}
    assert all(money_fields.issubset(row) for row in compact["items"])
    assert compact["items"][0]["profit"] == 0.0
    assert compact["items"][0]["net_pnl"] == 0.0
    assert compact["items"][0]["order_ticket"] == 100
    assert compact["items"][1]["net_pnl"] == 8.25
    assert compact["units"] == {
        "volume": "broker_lot",
        "net_pnl": "account_currency",
    }
