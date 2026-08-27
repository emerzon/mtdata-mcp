"""Tests for trade close operations in mtdata.core.trading.

Covers:
- trade_close (dispatch logic)
- _close_positions (MT5)
- _cancel_pending (MT5)
- _execute_single_close
- _sort_close_positions
- _deal_history_sort_key
- _resolve_closed_deal_from_history
"""

import importlib
import os
import sys
from collections import namedtuple
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable and stub MetaTrader5 before importing the module
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

_mt5_stub = MagicMock()
_mt5_stub.ORDER_TYPE_BUY = 0
_mt5_stub.ORDER_TYPE_SELL = 1
_mt5_stub.ORDER_TYPE_BUY_LIMIT = 2
_mt5_stub.ORDER_TYPE_SELL_LIMIT = 3
_mt5_stub.ORDER_TYPE_BUY_STOP = 4
_mt5_stub.ORDER_TYPE_SELL_STOP = 5
_mt5_stub.TRADE_ACTION_DEAL = 1
_mt5_stub.TRADE_ACTION_PENDING = 5
_mt5_stub.TRADE_ACTION_SLTP = 6
_mt5_stub.TRADE_ACTION_MODIFY = 7
_mt5_stub.TRADE_ACTION_REMOVE = 8
_mt5_stub.TRADE_RETCODE_DONE = 10009
_mt5_stub.TRADE_RETCODE_DONE_PARTIAL = 10010
_mt5_stub.TRADE_RETCODE_PRICE_CHANGED = 10020
_mt5_stub.ORDER_TIME_GTC = 0
_mt5_stub.ORDER_TIME_SPECIFIED = 2
_mt5_stub.ORDER_FILLING_IOC = 1
_mt5_stub.ORDER_FILLING_FOK = 0
_mt5_stub.ORDER_FILLING_RETURN = 2
_mt5_stub.POSITION_TYPE_BUY = 0
_mt5_stub.POSITION_TYPE_SELL = 1
sys.modules["MetaTrader5"] = _mt5_stub

from mtdata.core.trading import (
    trade_close as _trade_close_tool,
)
from mtdata.core.trading.execution import (
    _deal_history_sort_key,
    _execute_single_close,
    _resolve_close_dry_run_target,
    _resolve_closed_deal_from_history,
    _sort_close_positions,
)
from mtdata.core.trading.gateway import MT5TradingGateway
from mtdata.core.trading.requests import (
    TradeCloseRequest,
)

# ===================================================================
# Helpers
# ===================================================================

def trade_close(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradeCloseRequest(**kwargs)
    return _trade_close_tool(request=request, __cli_raw=raw_output)


def _unwrap_mcp(result):
    """MCP tool decorator serialises dicts to strings; return the raw dict
    when the underlying function is patched to return one, otherwise return
    the string for assertion with ``in``."""
    if isinstance(result, dict):
        return result
    return str(result)


def _tick(bid=1.10000, ask=1.10020, time=None, time_msc=None):
    # Close paths require a usable tick timestamp for freshness checks.
    import time as _time

    now = int(_time.time()) if time is None else int(time)
    msc = int(now * 1000) if time_msc is None else int(time_msc)
    return SimpleNamespace(bid=bid, ask=ask, time=now, time_msc=msc)


def _order_result(retcode=10009, deal=1, order=1, volume=0.01, price=1.1,
                  bid=1.1, ask=1.1002, comment="done", request_id=1):
    return SimpleNamespace(
        retcode=retcode, deal=deal, order=order, volume=volume,
        price=price, bid=bid, ask=ask, comment=comment, request_id=request_id,
    )


def _position(ticket=1, symbol="EURUSD", type_=0, volume=0.01,
              price_open=1.1, sl=1.09, tp=1.12, profit=50.0,
              price_current=1.105, comment="", magic=234000,
              time=1700000000, time_update=1700000000, swap=0.0,
              identifier=None, position_id=None, order=None, deal=None):
    return SimpleNamespace(
        ticket=ticket, symbol=symbol, type=type_, volume=volume,
        price_open=price_open, sl=sl, tp=tp, profit=profit,
        price_current=price_current, comment=comment, magic=magic,
        time=time, time_update=time_update, swap=swap,
        identifier=identifier, position_id=position_id, order=order, deal=deal,
        _asdict=lambda: {
            "ticket": ticket, "symbol": symbol, "type": type_, "volume": volume,
            "price_open": price_open, "sl": sl, "tp": tp, "profit": profit,
            "price_current": price_current, "comment": comment, "magic": magic,
            "time": time, "time_update": time_update, "swap": swap,
            "identifier": identifier, "position_id": position_id, "order": order, "deal": deal,
        },
    )


def _pending_order(ticket=100, symbol="EURUSD", type_=2, volume=0.01,
                   price_open=1.09, sl=1.08, tp=1.11, time_setup=1700000000,
                   time_expiration=0, type_time=0, comment="", magic=234000):
    return SimpleNamespace(
        ticket=ticket, symbol=symbol, type=type_, volume=volume,
        price_open=price_open, sl=sl, tp=tp, time_setup=time_setup,
        time_expiration=time_expiration, type_time=type_time,
        comment=comment, magic=magic,
        _asdict=lambda: {
            "ticket": ticket, "symbol": symbol, "type": type_, "volume": volume,
            "price_open": price_open, "sl": sl, "tp": tp,
            "time_setup": time_setup, "time_expiration": time_expiration,
            "comment": comment, "magic": magic,
        },
    )


# Patch MT5 connection helpers to avoid real terminal access in tests
@pytest.fixture(autouse=True)
def _bypass_auto_connect(monkeypatch):
    """Neutralize MT5 connection guards so no real terminal access is needed."""
    passthrough = lambda fn=None, **kw: fn if fn else (lambda f: f)
    monkeypatch.setattr("mtdata.core.trading.gateway.ensure_mt5_connection_or_raise", lambda: None)
    monkeypatch.setattr("mtdata.core.trading.gateway.ensure_mt5_connection_or_raise", lambda: None)
    for module_name in [
        "mtdata.core.trading.account",
        "mtdata.core.trading.execution",
        "mtdata.core.trading.orders",
        "mtdata.core.trading.positions",
        "mtdata.core.trading.risk",
        "mtdata.core.trading.validation",
    ]:
        module = importlib.import_module(module_name)
        if hasattr(module, "ensure_mt5_connection_or_raise"):
            monkeypatch.setattr(f"{module_name}.ensure_mt5_connection_or_raise", lambda: None)
    return passthrough


# ===================================================================
#  deal_history helpers (standalone tests)
# ===================================================================

def test_deal_history_sort_key_normalizes_millisecond_timestamps():
    earlier = SimpleNamespace(time_msc=1744876700000)
    later = SimpleNamespace(time=1744876800)

    assert _deal_history_sort_key(earlier) == 1744876700.0
    assert _deal_history_sort_key(later) == 1744876800.0
    assert max([earlier, later], key=_deal_history_sort_key) is later


def test_resolve_closed_deal_from_history_queries_recent_window_as_mt5_epoch():
    matched_row = SimpleNamespace(
        ticket=101,
        order=202,
        position_id=303,
        position_by_id=None,
        position=None,
        time=1700000000,
    )
    mt5 = SimpleNamespace(history_deals_get=MagicMock(return_value=[matched_row]))

    with patch(
        "mtdata.core.trading.execution._utc_epoch_seconds",
        side_effect=[111.0, 222.0],
    ):
        result = _resolve_closed_deal_from_history(
            mt5,
            result=SimpleNamespace(deal=101, order=202),
            position=SimpleNamespace(ticket=303),
            closed_at_utc=datetime(2026, 3, 1, 11, 0, tzinfo=timezone.utc),
        )

    called_from, called_to = mt5.history_deals_get.call_args.args
    assert called_from == 111.0
    assert called_to == 222.0
    assert result is matched_row


def test_resolve_closed_deal_from_history_tiebreaks_missing_timestamps_by_ticket():
    older_ticket = SimpleNamespace(
        ticket=101,
        order=None,
        position_id=303,
        position_by_id=None,
        position=None,
        time=0,
        time_msc=None,
        time_update=0,
        time_update_msc=None,
    )
    newer_ticket = SimpleNamespace(
        ticket=102,
        order=None,
        position_id=303,
        position_by_id=None,
        position=None,
        time=0,
        time_msc=None,
        time_update=0,
        time_update_msc=None,
    )
    mt5 = SimpleNamespace(
        history_deals_get=MagicMock(return_value=[older_ticket, newer_ticket])
    )

    with patch(
        "mtdata.core.trading.execution._utc_epoch_seconds",
        side_effect=[111.0, 222.0],
    ):
        result = _resolve_closed_deal_from_history(
            mt5,
            result=SimpleNamespace(deal=None, order=None),
            position=SimpleNamespace(ticket=303),
            closed_at_utc=datetime(2026, 3, 1, 11, 0, tzinfo=timezone.utc),
        )

    assert result is newer_ticket


# ===================================================================
#  trade_close (dispatch)
# ===================================================================

class TestTradeClose:

    def test_dry_run_partial_close_rejects_volume_above_position(self):
        adapter = MagicMock()
        adapter.positions_get.return_value = [
            _position(ticket=123, symbol="EURUSD", volume=0.1)
        ]
        adapter.symbol_info.return_value = SimpleNamespace(
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
        )
        gateway = MT5TradingGateway(
            adapter=adapter,
            ensure_connection_impl=lambda: None,
        )

        out = _resolve_close_dry_run_target(
            ticket=123,
            volume=0.2,
            gateway=gateway,
        )

        assert out["error"] == "volume must be <= open position volume (0.1)"
        assert out["requested_volume"] == 0.2
        assert out["position_volume"] == 0.1

    def test_dry_run_missing_ticket_does_not_select_unrelated_trade(self):
        adapter = MagicMock()

        def positions_get(**kwargs):
            if kwargs.get("ticket") == 999:
                return ()
            return [_position(ticket=123, symbol="EURUSD", volume=0.1)]

        def orders_get(**kwargs):
            if kwargs.get("ticket") == 999:
                return ()
            return [_pending_order(ticket=456, symbol="EURUSD")]

        adapter.positions_get.side_effect = positions_get
        adapter.orders_get.side_effect = orders_get
        gateway = MT5TradingGateway(
            adapter=adapter,
            ensure_connection_impl=lambda: None,
        )

        out = _resolve_close_dry_run_target(ticket=999, gateway=gateway)

        assert "not found" in out["error"]
        assert out["checked_scopes"] == ["positions"]
        adapter.orders_get.assert_not_called()

    def test_dry_run_pending_target_queries_orders_only(self):
        adapter = MagicMock()
        adapter.orders_get.return_value = [
            _pending_order(ticket=456, symbol="EURUSD")
        ]
        gateway = MT5TradingGateway(
            adapter=adapter,
            ensure_connection_impl=lambda: None,
        )

        out = _resolve_close_dry_run_target(
            ticket=456,
            target="pending",
            gateway=gateway,
        )

        assert out["success"] is True
        assert out["target_scope"] == "pending_orders"
        assert out["target_kind"] == "pending_order"
        assert out["resolved_ticket"] == 456
        adapter.positions_get.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_profit_filter_routes_to_positions_only(self, mock_close, mock_cancel):
        mock_close.return_value = {"closed_count": 1}
        trade_close(ticket=123, pnl_filter="profit", dry_run=False)
        mock_close.assert_called_once()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_loss_filter_routes_to_positions_only(self, mock_close, mock_cancel):
        mock_close.return_value = {"closed_count": 1}
        trade_close(ticket=123, pnl_filter="loss", dry_run=False)
        mock_close.assert_called_once()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_ticket_pending_target_only_cancels_pending(self, mock_close, mock_cancel):
        mock_cancel.return_value = {"cancelled_count": 1}
        out = trade_close(
            ticket=123,
            target="pending",
            dry_run=False,
            __cli_raw=True,
        )
        mock_close.assert_not_called()
        mock_cancel.assert_called_once_with(
            ticket=123,
            symbol=None,
            comment=None,
            dry_run=False,
        )
        assert out["cancelled_count"] == 1

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_position_ticket_missing_reports_position_scope_only(self, mock_close, mock_cancel):
        mock_close.return_value = {"error": "Position 123 not found"}
        result = trade_close(ticket=123, dry_run=False, __cli_raw=True)

        assert result["error"] == "Position 123 not found."
        assert result["checked_scopes"] == ["positions"]
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_dispatches_to_close_positions(self, mock_close, mock_cancel):
        mock_close.return_value = {"closed_count": 1}
        trade_close(ticket=123, dry_run=False)
        mock_close.assert_called_once()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_partial_close_by_ticket_passes_volume(self, mock_close, mock_cancel):
        mock_close.return_value = {"closed_count": 1}
        trade_close(ticket=123, volume=0.05, dry_run=False)
        mock_close.assert_called_once_with(
            ticket=123,
            symbol=None,
            volume=0.05,
            profit_only=False,
            loss_only=False,
            close_priority=None,
            comment=None,
            deviation=20,
            dry_run=False,
        )
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._resolve_close_dry_run_target")
    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_dry_run_ticket_preview_skips_execution(
        self, mock_close, mock_cancel, mock_resolve
    ):
        mock_resolve.return_value = {
            "success": True,
            "target_scope": "positions",
            "target_kind": "open_position",
            "resolved_ticket": 123,
            "target_symbol": "EURUSD",
            "target_volume": 0.1,
            "preview_ok": True,
            "matched_count": 1,
            "market_readiness": {"usable_for_live_trading": True},
        }

        out = trade_close(ticket=123, volume=0.05, dry_run=True, __cli_raw=True)

        assert out["success"] is True
        assert out["dry_run"] is True
        assert out["actionability"] == "preview_only"
        assert out["operation"] == "partial_close_position"
        assert out["ticket"] == 123
        assert out["volume"] == 0.05
        assert out["would_send_order"] is False
        assert out["target_scope"] == "positions"
        assert out["target_kind"] == "open_position"
        assert out["resolved_ticket"] == 123
        assert out["preview_ok"] is True
        assert out["matched_count"] == 1
        assert out["market_readiness"]["usable_for_live_trading"] is True
        assert "realized_pnl" in out["not_estimated"]
        mock_resolve.assert_called_once_with(
            ticket=123,
            target="positions",
            symbol=None,
            volume=0.05,
            magic=None,
            profit_only=False,
            loss_only=False,
            close_priority=None,
        )
        mock_close.assert_not_called()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._resolve_close_dry_run_target")
    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_dry_run_ticket_preview_copies_live_ready_blockers(
        self, mock_close, mock_cancel, mock_resolve
    ):
        mock_resolve.return_value = {
            "success": False,
            "error_code": "preview_blocked",
            "error": "One or more position quotes are not live-ready.",
            "blockers": ["market_not_live_ready"],
            "preview_ok": False,
            "target_scope": "positions",
            "target_kind": "open_position",
            "resolved_ticket": 123,
            "target_symbol": "TSLA.NAS-24",
            "target_volume": 50.0,
            "total_profit": -310.0,
            "matched_positions": [
                {"ticket": 123, "symbol": "TSLA.NAS-24", "side": "BUY", "volume": 50.0}
            ],
            "market_readiness": {"usable_for_live_trading": False},
        }

        out = trade_close(ticket=123, dry_run=True, __cli_raw=True)

        assert out["success"] is False
        assert out["preview_ok"] is False
        assert out["error_code"] == "preview_blocked"
        assert out["blockers"] == ["market_not_live_ready"]
        assert out.get("symbol") == "TSLA.NAS-24" or out.get("target_symbol") == (
            "TSLA.NAS-24"
        )
        assert out.get("volume") == 50.0 or out.get("target_volume") == 50.0
        assert out["total_profit"] == -310.0
        mock_close.assert_not_called()

    @patch("mtdata.core.trading._resolve_close_dry_run_target")
    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_dry_run_ticket_validates_ticket_exists(
        self, mock_close, mock_cancel, mock_resolve
    ):
        mock_resolve.return_value = {
            "error": "Position 999 not found.",
            "checked_scopes": ["positions"],
        }

        out = trade_close(ticket=999, dry_run=True, __cli_raw=True)

        assert out["error"] == "Position 999 not found."
        assert out["error_code"] == "ticket_not_found"
        assert out["ticket"] == 999
        assert out["checked_scopes"] == ["positions"]
        mock_resolve.assert_called_once_with(
            ticket=999,
            target="positions",
            symbol=None,
            volume=None,
            magic=None,
            profit_only=False,
            loss_only=False,
            close_priority=None,
        )
        mock_close.assert_not_called()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_partial_close_requires_ticket(self, mock_close, mock_cancel):
        out = _unwrap_mcp(trade_close(symbol="EURUSD", volume=0.05, dry_run=False))
        if isinstance(out, dict):
            assert "volume is only supported" in str(out.get("error", "")).lower()
        else:
            assert "volume is only supported" in out.lower()
        mock_close.assert_not_called()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_partial_close_ticket_not_found_does_not_cancel_pending(self, mock_close, mock_cancel):
        mock_close.return_value = {"error": "Position 123 not found"}
        out = _unwrap_mcp(trade_close(ticket=123, volume=0.05, dry_run=False))
        if isinstance(out, dict):
            assert "partial close volume only applies to open positions" in str(out.get("error", "")).lower()
            assert out.get("checked_scopes") == ["positions"]
        else:
            assert "partial close volume only applies to open positions" in out.lower()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_symbol_close_requires_close_all_confirmation(self, mock_close, mock_cancel):
        out = _unwrap_mcp(trade_close(symbol="EURUSD", dry_run=False))
        if isinstance(out, dict):
            error = str(out.get("error", "")).lower()
            assert "live bulk operation requires explicit confirmation" in error
            assert "--confirm-close-all true" in error
        else:
            assert "live bulk operation requires explicit confirmation" in out.lower()
        mock_close.assert_not_called()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_global_close_requires_close_all_confirmation(self, mock_close, mock_cancel):
        out = _unwrap_mcp(trade_close(dry_run=False))
        if isinstance(out, dict):
            error = str(out.get("error", "")).lower()
            assert "bulk close requires explicit confirmation" in error
            assert "--close-all true" in error
        else:
            assert "bulk close requires explicit confirmation" in out.lower()
        mock_close.assert_not_called()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_global_close_dry_run_requires_explicit_scope(self, mock_close, mock_cancel):
        out = trade_close(dry_run=True, __cli_raw=True)

        assert out["success"] is False
        assert out["error_code"] == "close_scope_required"
        assert "--close-all true" in out["error"]
        assert "--ticket <ticket>" in out["error"]
        assert out.get("remediation")
        mock_close.assert_not_called()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_symbol_bulk_close_dry_run_succeeds_without_live_confirmation(
        self, mock_close, mock_cancel
    ):
        mock_close.return_value = {
            "success": True,
            "matched_count": 0,
            "matched_positions": [],
            "would_send_orders": 0,
            "message": "No open positions for EURUSD",
        }
        out = _unwrap_mcp(trade_close(symbol="EURUSD", dry_run=True, __cli_raw=True))
        assert out["success"] is True
        assert out["dry_run"] is True
        assert out["preview_ok"] is True
        assert out["authorization_status"] == "required"
        assert out["required_confirmation"] == "--confirm-close-all true"
        assert out["validation"]["live_submission_eligible"] is False
        assert "confirmation_required" in out["validation"]["blockers"]
        assert out.get("error_code") != "preview_blocked"
        assert out["would_send_order"] is False
        assert out["operation"] == "close_symbol_positions"
        assert out["matched_positions_count"] == 0
        assert out["empty"] is True
        assert out["no_action"] is True
        assert out["symbol"] == "EURUSD"
        mock_close.assert_called_once_with(
            symbol="EURUSD",
            volume=None,
            profit_only=False,
            loss_only=False,
            close_priority=None,
            comment=None,
            deviation=20,
            dry_run=True,
        )
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_close_all_dry_run_succeeds_without_live_confirmation(
        self, mock_close, mock_cancel
    ):
        mock_close.return_value = {
            "success": True,
            "preview_ok": True,
            "matched_count": 2,
            "would_send_orders": 2,
        }
        out = trade_close(close_all=True, dry_run=True, __cli_raw=True)

        assert out["success"] is True
        assert out["preview_ok"] is True
        assert out["authorization_status"] == "required"
        assert out["required_confirmation"] == "--confirm-close-all true"
        assert out["validation"]["live_submission_eligible"] is False
        assert out.get("error_code") != "preview_blocked"
        mock_close.assert_called_once()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_close_all_cannot_be_combined_with_ticket(self, mock_close, mock_cancel):
        out = _unwrap_mcp(trade_close(ticket=123, close_all=True, dry_run=False))
        if isinstance(out, dict):
            assert "close_all cannot be combined with ticket" in str(out.get("error", "")).lower()
        else:
            assert "close_all cannot be combined with ticket" in out.lower()
        mock_close.assert_not_called()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_all_exposure_rejects_ticket_scope(self, mock_close, mock_cancel):
        out = trade_close(
            ticket=123,
            target="all_exposure",
            dry_run=True,
            __cli_raw=True,
        )

        assert out["success"] is False
        assert out["error_code"] == "invalid_close_target_options"
        assert "bulk symbol, magic, or account scope" in out["error"]
        mock_close.assert_not_called()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_pending_target_rejects_position_only_filters(
        self,
        mock_close,
        mock_cancel,
    ):
        volume_out = trade_close(
            ticket=123,
            target="pending",
            volume=0.01,
            dry_run=True,
            __cli_raw=True,
        )
        pnl_out = trade_close(
            magic=3001,
            target="pending",
            pnl_filter="profit",
            dry_run=True,
            __cli_raw=True,
        )

        assert volume_out["error"] == "volume is valid only with target=positions."
        assert pnl_out["error"] == "pnl_filter is valid only with target=positions."
        mock_close.assert_not_called()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_symbol_positions_only_no_match_marks_no_action(self, mock_close, mock_cancel):
        mock_close.return_value = {"message": "No open positions for EURUSD"}
        mock_cancel.return_value = {"message": "No pending orders for EURUSD"}
        out = _unwrap_mcp(
            trade_close(
                symbol="EURUSD",
                close_all=True,
                confirm_close_all=True,
                dry_run=False,
            )
        )
        if isinstance(out, dict):
            assert out.get("message") == "No open positions for EURUSD"
            assert out.get("no_action") is True
        else:
            assert "No open positions for EURUSD" in out
            assert "no_action" in out.lower()
        mock_close.assert_called_once()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_global_positions_only_no_match_marks_no_action(self, mock_close, mock_cancel):
        mock_close.return_value = {"message": "No open positions"}
        mock_cancel.return_value = {"message": "No pending orders"}
        out = _unwrap_mcp(
            trade_close(close_all=True, confirm_close_all=True, dry_run=False)
        )
        if isinstance(out, dict):
            assert out.get("message") == "No open positions"
            assert out.get("no_action") is True
        else:
            assert "No open positions" in out
            assert "no_action" in out.lower()
        mock_close.assert_called_once()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_all_exposure_live_runs_both_legs_for_mixed_book(
        self,
        mock_close,
        mock_cancel,
    ):
        mock_close.return_value = {
            "success": True,
            "closed_count": 2,
            "results": [{"ticket": 11}, {"ticket": 12}],
        }
        mock_cancel.return_value = {
            "success": True,
            "cancelled_count": 1,
            "results": [{"ticket": 21}],
        }

        out = trade_close(
            symbol="EURUSD",
            target="all_exposure",
            confirm_close_all=True,
            dry_run=False,
            __cli_raw=True,
        )

        assert out["success"] is True
        assert out["target"] == "all_exposure"
        assert out["closed_count"] == 2
        assert out["cancelled_count"] == 1
        assert out["closed_positions"]["results"] == [
            {"ticket": 11},
            {"ticket": 12},
        ]
        assert out["cancelled_pending_orders"]["results"] == [{"ticket": 21}]
        assert out["leg_status"] == {
            "positions": "ok",
            "pending_orders": "ok",
        }
        mock_close.assert_called_once()
        mock_cancel.assert_called_once_with(
            symbol="EURUSD",
            comment=None,
            dry_run=False,
        )

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_all_exposure_surfaces_per_leg_partial_failure(
        self,
        mock_close,
        mock_cancel,
    ):
        mock_close.return_value = {"error": "position close rejected"}
        mock_cancel.return_value = {"success": True, "cancelled_count": 1}

        out = trade_close(
            magic=3001,
            target="all_exposure",
            confirm_close_all=True,
            dry_run=False,
            __cli_raw=True,
        )

        assert out["success"] is False
        assert out["error_code"] == "all_exposure_partial_failure"
        assert out["partial_failure"] is True
        assert out["failed_legs"] == ["positions"]
        assert out["leg_status"]["pending_orders"] == "ok"
        assert out["cancelled_count"] == 1
        assert out["closed_positions"]["error"] == "position close rejected"
        assert mock_close.call_args.kwargs["magic"] == 3001
        assert mock_cancel.call_args.kwargs["magic"] == 3001

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_magic_is_a_standalone_positions_preview_scope(
        self,
        mock_close,
        mock_cancel,
    ):
        mock_close.return_value = {
            "success": True,
            "matched_count": 2,
            "matched_positions": [
                {
                    "ticket": 11,
                    "magic": 3001,
                    "quote_context": {
                        "usable_for_live_trading": False,
                        "quote_source_conflict": {"reason": "test"},
                    },
                },
                {"ticket": 12, "magic": 3001},
            ],
        }

        out = trade_close(
            magic=3001,
            dry_run=True,
            confirm_close_all=True,
            __cli_raw=True,
        )

        assert out["success"] is True
        assert out["preview_ok"] is True
        assert out["target"] == "positions"
        assert out["matched_count"] == 2
        assert {row["magic"] for row in out["matched_positions"]} == {3001}
        assert out["matched_positions"][0]["quote_usable"] is False
        assert out["matched_positions"][0]["quote_readiness_reason"] == (
            "quote_source_conflict"
        )
        assert "quote_context" not in out["matched_positions"][0]
        assert mock_close.call_args.kwargs["magic"] == 3001
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_all_exposure_dry_run_previews_both_legs(
        self,
        mock_close,
        mock_cancel,
    ):
        mock_close.return_value = {"success": True, "matched_count": 2}
        mock_cancel.return_value = {
            "success": True,
            "matched_pending_count": 1,
        }

        out = trade_close(
            symbol="EURUSD",
            target="all_exposure",
            dry_run=True,
            confirm_close_all=True,
            __cli_raw=True,
        )

        assert out["success"] is True
        assert out["preview_ok"] is True
        assert out["would_send_orders"] == 2
        assert out["would_cancel_pending_orders"] == 1
        assert out["closed_count"] == 0
        assert out["cancelled_count"] == 0
        assert out["matched_positions_count"] == 2
        assert out["matched_pending_count"] == 1
        assert out["closed_positions"]["matched_count"] == 2
        assert out["cancelled_pending_orders"]["matched_pending_count"] == 1
        mock_close.assert_called_once()
        mock_cancel.assert_called_once()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_all_exposure_dry_run_ands_child_preview_ok(
        self,
        mock_close,
        mock_cancel,
    ):
        mock_close.return_value = {
            "success": True,
            "matched_count": 1,
            "preview_ok": False,
        }
        mock_cancel.return_value = {
            "success": True,
            "matched_pending_count": 0,
            "preview_ok": True,
        }

        out = trade_close(
            symbol="EURUSD",
            target="all_exposure",
            dry_run=True,
            confirm_close_all=True,
            __cli_raw=True,
        )

        assert out["success"] is False
        assert out["preview_ok"] is False
        assert out["error_code"] == "preview_blocked"

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_empty_pending_dry_run_does_not_claim_cancel(
        self,
        mock_close,
        mock_cancel,
    ):
        mock_cancel.return_value = {
            "success": True,
            "matched_pending_count": 0,
            "would_cancel_pending_orders": 0,
            "message": "No pending orders",
        }

        out = trade_close(
            close_all=True,
            target="pending",
            dry_run=True,
            confirm_close_all=True,
            __cli_raw=True,
        )

        assert out["would_cancel_pending_order"] is False
        assert out["would_cancel_pending_orders"] == 0
        mock_close.assert_not_called()


# ===================================================================
#  _close_positions (MT5 mocked)
# ===================================================================

class TestClosePositions:

    def _setup_mt5(self, mt5):
        mt5.ORDER_TYPE_BUY = 0
        mt5.ORDER_TYPE_SELL = 1
        mt5.POSITION_TYPE_BUY = 0
        mt5.POSITION_TYPE_SELL = 1
        mt5.TRADE_ACTION_DEAL = 1
        mt5.TRADE_RETCODE_DONE = 10009
        mt5.TRADE_RETCODE_DONE_PARTIAL = 10010
        mt5.ORDER_TIME_GTC = 0
        mt5.ORDER_FILLING_IOC = 1
        mt5.ORDER_FILLING_FOK = 0
        mt5.ORDER_FILLING_RETURN = 2

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_ticket_not_found(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = None
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=999)
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_no_open_positions(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = []
        from mtdata.core.trading import _close_positions
        result = _close_positions()
        assert "message" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_snapshot_failure_is_not_reported_as_flat(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = None
        from mtdata.core.trading import _close_positions

        result = _close_positions()

        assert result["success"] is False
        assert result["error_code"] == "positions_snapshot_unavailable"

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_invalid_symbol_is_not_reported_as_snapshot_outage(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.symbol_info.return_value = None
        mt5.symbol_select.return_value = False
        mt5.positions_get.return_value = None
        mt5.last_error.return_value = (1, "Success")
        from mtdata.core.trading import _close_positions

        result = _close_positions(symbol="NOTREAL", dry_run=True)

        assert result["success"] is False
        assert result["error_code"] == "symbol_not_found"
        assert "NOTREAL" in result["error"]
        assert result.get("related_tools") == ["symbols_list"]
        assert all(
            call.kwargs.get("symbol") != "NOTREAL"
            for call in mt5.positions_get.call_args_list
        )

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_valid_symbol_empty_book_is_not_an_outage(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.symbol_info.return_value = SimpleNamespace(name="EURUSD")
        mt5.positions_get.return_value = []
        from mtdata.core.trading import _close_positions

        result = _close_positions(symbol="EURUSD", dry_run=True)

        assert result.get("error_code") != "positions_snapshot_unavailable"
        assert "No open positions" in str(result.get("message") or "")

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_snapshot_outage_omits_success_last_error(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.symbol_info.return_value = SimpleNamespace(name="EURUSD")
        mt5.positions_get.return_value = None
        mt5.last_error.return_value = (1, "Success")
        from mtdata.core.trading import _close_positions

        result = _close_positions(symbol="EURUSD")

        assert result["error_code"] == "positions_snapshot_unavailable"
        assert "last_error" not in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_profit_only_filter(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        pos_profit = _position(ticket=1, profit=50)
        pos_loss = _position(ticket=2, profit=-30)
        mt5.positions_get.return_value = [pos_profit, pos_loss]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _close_positions
        result = _close_positions(profit_only=True)
        assert result["attempted_count"] == 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_loss_only_filter(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        pos_profit = _position(ticket=1, profit=50)
        pos_loss = _position(ticket=2, profit=-30)
        mt5.positions_get.return_value = [pos_profit, pos_loss]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _close_positions
        result = _close_positions(loss_only=True)
        assert result["attempted_count"] == 1

    @patch("mtdata.core.trading.execution._execute_single_close")
    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_bulk_all_fail_sets_top_level_failure(self, mock_execute):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=1), _position(ticket=2)]
        mock_execute.side_effect = [{"error": "rejected"}, {"error": "timeout"}]
        from mtdata.core.trading import _close_positions

        result = _close_positions()

        assert result["success"] is False
        assert result["closed_count"] == 0
        assert result["attempted_count"] == 2
        assert result["partial_failure"] is False
        assert "error" in result

    @patch("mtdata.core.trading.execution._execute_single_close")
    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_bulk_mixed_result_sets_partial_failure(self, mock_execute):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=1), _position(ticket=2)]
        mock_execute.side_effect = [
            {"retcode": mt5.TRADE_RETCODE_DONE},
            {"error": "timeout"},
        ]
        from mtdata.core.trading import _close_positions

        result = _close_positions()

        assert result["success"] is False
        assert result["closed_count"] == 1
        assert result["partial_failure"] is True

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_profit_only_excludes_breakeven_positions(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=3, profit=0.0)]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _close_positions
        result = _close_positions(profit_only=True)
        assert result["message"] == "No positions matched criteria"

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_loss_only_excludes_breakeven_positions(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=4, profit=0.0)]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _close_positions
        result = _close_positions(loss_only=True)
        assert result["message"] == "No positions matched criteria"

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_profit_only_skips_positions_with_unreadable_profit(self):
        class _BrokenProfitPosition:
            ticket = 7
            symbol = "EURUSD"
            type = 0
            volume = 0.01
            price_current = 1.105
            price_open = 1.1
            comment = ""
            magic = 234000
            time = 1700000000
            time_update = 1700000000
            swap = 0.0

            @property
            def profit(self):
                raise RuntimeError("bad profit")

        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_BrokenProfitPosition(), _position(ticket=8, profit=50.0)]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _close_positions
        result = _close_positions(profit_only=True)
        assert result["attempted_count"] == 1
        assert mt5.order_send.call_count == 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_conflicting_profit_and_loss_filters_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=5, profit=10.0)]
        from mtdata.core.trading import _close_positions
        result = _close_positions(profit_only=True, loss_only=True)
        assert result["error"] == "profit_only and loss_only cannot both be true."

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_tick_failure_for_position(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position()]
        mt5.symbol_info_tick.return_value = None
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=1)
        assert "error" in result and "tick" in result["error"].lower()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_close_rejects_position_with_unreadable_side(self):
        class _BrokenTypePosition:
            ticket = 11
            symbol = "EURUSD"
            volume = 0.01
            price_open = 1.1
            sl = 1.09
            tp = 1.12
            profit = 50.0
            price_current = 1.105
            comment = ""
            magic = 234000
            time = 1700000000
            time_update = 1700000000
            swap = 0.0

            @property
            def type(self):
                raise RuntimeError("bad type")

        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_BrokenTypePosition()]
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=11)
        assert result["error"] == "Unable to determine position side for close request."
        mt5.order_send.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_none(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position()]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = None
        mt5.last_error.return_value = (10006, "No connection")
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=1)
        assert "error" in result
        assert "attempts" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_close_does_not_retry_after_ambiguous_none_response(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=77)]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.side_effect = [None, _order_result()]
        mt5.last_error.return_value = (10006, "No connection")
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=77)
        assert result["ticket"] == 77
        assert "outcome is unknown" in result["error"]
        assert result["error_code"] == "order_send_ambiguous"
        assert result["ambiguous"] is True
        assert isinstance(result.get("attempts"), list)
        assert len(result.get("attempts") or []) == 1
        assert mt5.order_send.call_count == 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_partial_close_does_not_resubmit_after_ambiguous_comment_fallback(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=77, volume=1.0)]
        mt5.symbol_info.return_value = SimpleNamespace(
            volume_min=0.01,
            volume_step=0.01,
        )
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.side_effect = [
            _order_result(retcode=10013, comment="Invalid comment"),
            None,
            _order_result(volume=0.5),
        ]
        mt5.last_error.side_effect = [(0, ""), (1, "IPC timeout")]
        from mtdata.core.trading import _close_positions

        result = _close_positions(
            ticket=77,
            volume=0.5,
            comment="broker-rejected",
        )

        assert "outcome is unknown" in result["error"]
        assert result["error_code"] == "order_send_ambiguous"
        assert result["ambiguous"] is True
        assert result["comment_fallback"]["ambiguous"] is True
        assert mt5.order_send.call_count == 2
        assert all(
            call.args[0]["volume"] == pytest.approx(0.5)
            for call in mt5.order_send.call_args_list
        )

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_close_retries_only_after_invalid_fill(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.TRADE_RETCODE_INVALID_FILL = 10030
        mt5.positions_get.return_value = [_position(ticket=77)]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.side_effect = [
            _order_result(retcode=10030, comment="Unsupported filling mode"),
            _order_result(),
        ]
        from mtdata.core.trading import _close_positions

        result = _close_positions(ticket=77)

        assert int(result["retcode"]) == 10009
        assert mt5.order_send.call_count == 2

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_close_does_not_retry_unrelated_rejection_across_fill_modes(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=77)]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.side_effect = [
            _order_result(retcode=10006, comment="Rejected"),
            _order_result(),
        ]
        from mtdata.core.trading import _close_positions

        result = _close_positions(ticket=77)

        assert result["error"] == "Failed to send close order"
        assert mt5.order_send.call_count == 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_no_positions_matched_criteria(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        # All profitable, but loss_only filter
        mt5.positions_get.return_value = [_position(profit=50)]
        from mtdata.core.trading import _close_positions
        result = _close_positions(loss_only=True)
        assert "message" in result and "No positions matched" in result["message"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_single_ticket_returns_single_result(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=42)]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=42)
        assert result["success"] is True
        assert result["ticket"] == 42
        assert result["open_price"] == 1.1
        assert result["close_price"] == 1.1
        assert "pnl" in result
        assert "duration_seconds" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_single_ticket_with_two_open_positions_sends_one_exact_close(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        requested = _position(ticket=42, symbol="EURUSD", volume=0.01)
        other = _position(ticket=43, symbol="EURUSD", volume=0.01)
        mt5.positions_get.side_effect = [[requested], [other, requested]]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = _order_result()

        from mtdata.core.trading import _close_positions

        result = _close_positions(ticket=42)

        assert result["ticket"] == 42
        assert mt5.order_send.call_count == 1
        close_request = mt5.order_send.call_args.args[0]
        assert close_request["position"] == 42
        assert close_request["volume"] == 0.01

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_partial_close_uses_requested_volume(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=42, volume=0.10)]
        mt5.symbol_info.return_value = SimpleNamespace(volume_min=0.01, volume_step=0.01)
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = _order_result(volume=0.05)
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=42, volume=0.05)
        request = mt5.order_send.call_args[0][0]
        assert request["volume"] == 0.05
        assert result["requested_volume"] == 0.05
        assert result["position_volume_before"] == 0.10
        assert result["position_volume_remaining_estimate"] == 0.05
        assert result["filled_volume"] == 0.05

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_partial_fill_uses_broker_volume_for_remaining_estimate(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=42, volume=0.10)]
        mt5.symbol_info.return_value = SimpleNamespace(volume_min=0.01, volume_step=0.01)
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = _order_result(retcode=10010, volume=0.03)
        from mtdata.core.trading import _close_positions

        result = _close_positions(ticket=42, volume=0.05)

        assert result["requested_volume"] == 0.05
        assert result["filled_volume"] == 0.03
        assert result["position_volume_remaining_estimate"] == pytest.approx(0.07)
        assert result["partial_fill"] is True
        assert result["success"] is False
        assert result["execution_status"] == "partial"
        assert result["remaining_volume"] == pytest.approx(0.02)

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_partial_close_rejects_volume_greater_than_position(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=42, volume=0.10)]
        mt5.symbol_info.return_value = SimpleNamespace(volume_min=0.01, volume_step=0.01)
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=42, volume=0.20)
        assert "volume must be <=" in result["error"]
        assert result["position_volume"] == 0.10
        mt5.order_send.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_partial_close_rejects_invalid_remaining_volume(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=42, volume=0.03)]
        mt5.symbol_info.return_value = SimpleNamespace(volume_min=0.02, volume_step=0.01)
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=42, volume=0.02)
        assert "remaining position volume would be invalid" in result["error"]
        assert result["remaining_volume"] == pytest.approx(0.01)
        mt5.order_send.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_symbol_no_positions(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = []
        from mtdata.core.trading import _close_positions
        result = _close_positions(symbol="GBPUSD")
        assert "message" in result

    # -----------------------------------------------------------------
    # _execute_single_close tests (from consolidated test_trading_business_logic)
    # -----------------------------------------------------------------

    def test_execute_single_close_success(self):
        """Successful close returns result with PnL metadata."""
        from mtdata.core.trading.gateway import create_trading_gateway
        mt5 = MagicMock()
        mt5.ORDER_FILLING_IOC = 1
        mt5.ORDER_TIME_GTC = 0
        mt5.TRADE_RETCODE_DONE = 10009

        position = SimpleNamespace(
            ticket=42, symbol="EURUSD", volume=0.1, type=0,
            price_open=1.04000, time=1700000000, magic=100,
        )
        mt5.order_send.return_value = SimpleNamespace(
            retcode=10009, deal=500, order=600, volume=0.1,
            price=1.05000, comment="close", profit=10.0,
        )
        mt5.symbol_info_tick.return_value = _tick(bid=1.05000, ask=1.05010)

        gw = create_trading_gateway(
            adapter=mt5,
            ensure_connection_impl=lambda: None,
        )
        fill_modes = [mt5.ORDER_FILLING_IOC]

        result = _execute_single_close(
            gw, position,
            requested_volume=None, position_volume_before=0.1,
            remaining_volume_estimate=None, deviation=20,
            comment="test", fill_modes=fill_modes,
        )
        assert result["ticket"] == 42
        assert result["retcode"] == 10009
        assert result.get("error") is None
        assert result["pnl"] == 10.0

    def test_execute_single_close_retries_same_fill_mode_after_price_change(self):
        from mtdata.core.trading.gateway import create_trading_gateway
        mt5 = MagicMock()
        mt5.ORDER_FILLING_FOK = 0
        mt5.ORDER_TIME_GTC = 0
        mt5.TRADE_RETCODE_DONE = 10009
        mt5.TRADE_RETCODE_PRICE_CHANGED = 10020

        position = SimpleNamespace(
            ticket=42, symbol="EURUSD", volume=0.1, type=0,
            price_open=1.04000, time=1700000000, magic=100,
        )
        mt5.order_send.side_effect = [
            SimpleNamespace(
                retcode=10020, deal=0, order=0, volume=0.1,
                price=1.05000, comment="price changed", profit=None,
            ),
            SimpleNamespace(
                retcode=10009, deal=500, order=600, volume=0.1,
                price=1.04990, comment="close", profit=9.0,
            ),
        ]
        mt5.symbol_info_tick.side_effect = [
            _tick(bid=1.05000, ask=1.05010),
            _tick(bid=1.04990, ask=1.05000),
        ]

        gw = create_trading_gateway(
            adapter=mt5,
            ensure_connection_impl=lambda: None,
        )

        result = _execute_single_close(
            gw, position,
            requested_volume=None, position_volume_before=0.1,
            remaining_volume_estimate=None, deviation=20,
            comment="test", fill_modes=[mt5.ORDER_FILLING_FOK],
        )

        assert result["ticket"] == 42
        assert result["retcode"] == 10009
        assert mt5.order_send.call_count == 2
        first_req = mt5.order_send.call_args_list[0].args[0]
        second_req = mt5.order_send.call_args_list[1].args[0]
        assert first_req["type_filling"] == mt5.ORDER_FILLING_FOK
        assert second_req["type_filling"] == mt5.ORDER_FILLING_FOK
        assert first_req["price"] == pytest.approx(1.05000)
        assert second_req["price"] == pytest.approx(1.04990)

    def test_execute_single_close_no_tick(self):
        """Returns error when tick data unavailable."""
        from mtdata.core.trading.gateway import create_trading_gateway
        mt5 = MagicMock()
        mt5.ORDER_FILLING_IOC = 1
        mt5.ORDER_TIME_GTC = 0

        position = SimpleNamespace(
            ticket=42, symbol="XYZUSD", volume=0.1, type=0,
            price_open=1.0, time=1700000000, magic=100,
        )
        mt5.symbol_info_tick.return_value = None
        mt5.last_error.return_value = (10006, "No tick")

        gw = create_trading_gateway(
            adapter=mt5,
            ensure_connection_impl=lambda: None,
        )
        fill_modes = [1]

        result = _execute_single_close(
            gw, position,
            requested_volume=None, position_volume_before=0.1,
            remaining_volume_estimate=None, deviation=20,
            comment=None, fill_modes=fill_modes,
        )
        assert result["ticket"] == 42
        assert "tick data" in result["error"].lower()

    def test_execute_single_close_rejects_stale_tick(self):
        from mtdata.core.trading.gateway import create_trading_gateway

        mt5 = MagicMock()
        mt5.ORDER_FILLING_IOC = 1
        mt5.ORDER_TIME_GTC = 0
        mt5.TRADE_RETCODE_DONE = 10009
        position = SimpleNamespace(
            ticket=42, symbol="EURUSD", volume=0.1, type=0, magic=100,
        )
        mt5.symbol_info_tick.return_value = SimpleNamespace(
            bid=1.05, ask=1.0501, time=1
        )
        gw = create_trading_gateway(
            adapter=mt5,
            ensure_connection_impl=lambda: None,
        )

        result = _execute_single_close(
            gw,
            position,
            requested_volume=None,
            position_volume_before=0.1,
            remaining_volume_estimate=None,
            deviation=20,
            comment=None,
            fill_modes=[mt5.ORDER_FILLING_IOC],
        )

        assert result["error_code"] == "tick_stale"
        assert result["tick_age_seconds"] > result["tick_max_age_seconds"]
        mt5.order_send.assert_not_called()

    def test_execute_single_close_rejects_invalid_full_close_volume(self):
        from mtdata.core.trading.gateway import create_trading_gateway
        mt5 = MagicMock()
        mt5.ORDER_FILLING_IOC = 1
        mt5.ORDER_TIME_GTC = 0

        position = SimpleNamespace(
            ticket=42, symbol="EURUSD", volume=0.0, type=0, magic=100,
        )

        gw = create_trading_gateway(
            adapter=mt5,
            ensure_connection_impl=lambda: None,
        )

        result = _execute_single_close(
            gw, position,
            requested_volume=None, position_volume_before=None,
            remaining_volume_estimate=None, deviation=20,
            comment=None, fill_modes=[mt5.ORDER_FILLING_IOC],
        )

        assert result["ticket"] == 42
        assert "invalid volume" in result["error"].lower()
        mt5.symbol_info_tick.assert_not_called()
        mt5.order_send.assert_not_called()

    def test_execute_single_close_unknown_side(self):
        """Returns error when position side cannot be determined."""
        from mtdata.core.trading.gateway import create_trading_gateway
        mt5 = MagicMock()
        mt5.ORDER_FILLING_IOC = 1
        mt5.ORDER_TIME_GTC = 0

        # Position with no type attribute at all
        position = SimpleNamespace(ticket=42, symbol="EURUSD", volume=0.1, magic=100)

        gw = create_trading_gateway(
            adapter=mt5,
            ensure_connection_impl=lambda: None,
        )
        fill_modes = [1]

        result = _execute_single_close(
            gw, position,
            requested_volume=None, position_volume_before=0.1,
            remaining_volume_estimate=None, deviation=20,
            comment=None, fill_modes=fill_modes,
        )
        assert result["ticket"] == 42
        assert "side" in result["error"].lower()

    def test_execute_single_close_none_position_returns_structured_error(self):
        mt5 = MagicMock()

        result = _execute_single_close(
            mt5,
            None,
            requested_volume=1.0,
            position_volume_before=None,
            remaining_volume_estimate=None,
            deviation=20,
            comment=None,
            fill_modes=[],
        )

        assert result == {
            "ticket": None,
            "error": "Unable to determine position side for close request.",
        }

    # -----------------------------------------------------------------
    # _sort_close_positions tests
    # -----------------------------------------------------------------

    def test_sort_loss_first(self):
        """loss_first sorts by ascending profit (most negative first)."""
        positions = [
            SimpleNamespace(ticket=1, profit=50.0, volume=0.1),
            SimpleNamespace(ticket=2, profit=-100.0, volume=0.2),
            SimpleNamespace(ticket=3, profit=-20.0, volume=0.3),
        ]
        result = _sort_close_positions(positions, "loss_first")
        assert [p.ticket for p in result] == [2, 3, 1]

    def test_sort_profit_first(self):
        """profit_first sorts by descending profit (most positive first)."""
        positions = [
            SimpleNamespace(ticket=1, profit=-10.0, volume=0.1),
            SimpleNamespace(ticket=2, profit=100.0, volume=0.2),
            SimpleNamespace(ticket=3, profit=50.0, volume=0.3),
        ]
        result = _sort_close_positions(positions, "profit_first")
        assert [p.ticket for p in result] == [2, 3, 1]

    def test_sort_largest_first(self):
        """largest_first sorts by descending volume."""
        positions = [
            SimpleNamespace(ticket=1, profit=10.0, volume=0.01),
            SimpleNamespace(ticket=2, profit=20.0, volume=1.0),
            SimpleNamespace(ticket=3, profit=-5.0, volume=0.5),
        ]
        result = _sort_close_positions(positions, "largest_first")
        assert [p.ticket for p in result] == [2, 3, 1]

    def test_sort_none_preserves_order(self):
        """None priority preserves discovery order."""
        positions = [
            SimpleNamespace(ticket=3, profit=10.0, volume=0.1),
            SimpleNamespace(ticket=1, profit=-10.0, volume=0.5),
        ]
        result = _sort_close_positions(positions, None)
        assert [p.ticket for p in result] == [3, 1]

    def test_sort_missing_profit(self):
        """Handles positions with missing profit attribute gracefully."""
        positions = [
            SimpleNamespace(ticket=1, volume=0.1),
            SimpleNamespace(ticket=2, profit=-50.0, volume=0.2),
        ]
        result = _sort_close_positions(positions, "loss_first")
        # Missing profit defaults to 0.0, so -50 sorts first
        assert [p.ticket for p in result] == [2, 1]

    # -----------------------------------------------------------------
    # Abort policy tests
    # -----------------------------------------------------------------

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_attempts_every_position_after_consecutive_failures(self):
        """Bulk close keeps trying distinct positions after earlier failures."""
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)

        positions = [
            SimpleNamespace(ticket=i, symbol="EURUSD", volume=0.1, type=0, profit=1.0, magic=0)
            for i in range(1, 6)
        ]
        mt5.positions_get.return_value = positions

        # All order_sends fail
        mt5.order_send.return_value = None
        mt5.last_error.return_value = (10004, "Connection lost")
        mt5.symbol_info_tick.return_value = _tick()

        # positions_get with ticket= returns the matching position
        def _positions_get(*args, **kwargs):
            t = kwargs.get("ticket")
            if t is not None:
                return [p for p in positions if p.ticket == t] or None
            return positions
        mt5.positions_get.side_effect = _positions_get

        from mtdata.core.trading import _close_positions
        res = _close_positions(symbol="EURUSD")

        results = res.get("results", [res])
        assert len(results) == 5
        assert mt5.order_send.call_count == 5
        assert not any(r.get("aborted") for r in results)

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_single_ticket_exception_returns_structured_error(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)

        position = _position(ticket=42, symbol="EURUSD")

        def _positions_get(*args, **kwargs):
            if kwargs.get("ticket") is not None:
                return [position]
            return [position]

        mt5.positions_get.side_effect = _positions_get
        mt5.symbol_info.return_value = None
        mt5.last_error.return_value = (10006, "No connection")

        from mtdata.core.trading import _close_positions

        with patch(
            "mtdata.core.trading.execution._execute_single_close",
            side_effect=RuntimeError("boom"),
        ):
            result = _close_positions(ticket=42)

        assert result["error"] == "boom"
        assert result["ticket"] == 42
        assert result["last_error"] == (10006, "No connection")

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_refresh_failure_is_not_reported_as_position_closed(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)

        position = _position(ticket=42, symbol="EURUSD")

        def _positions_get(*args, **kwargs):
            if kwargs.get("ticket") is not None:
                raise RuntimeError("transport down")
            return [position]

        mt5.positions_get.side_effect = _positions_get
        mt5.last_error.return_value = (10006, "No connection")

        from mtdata.core.trading import _close_positions

        result = _close_positions(symbol="EURUSD")

        row = result["results"][0]
        assert row["ticket"] == 42
        assert row["error"] == "Failed to refresh open position before close."
        assert row["last_error"] == (10006, "No connection")

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_refresh_must_return_the_exact_position_ticket(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        requested = _position(ticket=42, symbol="EURUSD")
        different = _position(ticket=43, symbol="EURUSD")
        mt5.positions_get.side_effect = [[requested], [different]]

        from mtdata.core.trading import _close_positions

        result = _close_positions(ticket=42)

        assert result["ticket"] == 42
        assert "Exact position ticket" in result["error"]
        mt5.order_send.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_resets_abort_counter_on_success(self):
        """A successful close resets the consecutive failure counter."""
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)

        positions = [
            SimpleNamespace(ticket=i, symbol="EURUSD", volume=0.1, type=0, profit=1.0, magic=0, price_open=1.04, time=1700000000)
            for i in range(1, 7)
        ]

        # Position 3 succeeds, all others fail
        def _order_send(request):
            if request.get("position") == 3:
                return _order_result()
            return None

        mt5.order_send.side_effect = _order_send
        mt5.last_error.return_value = (10004, "err")
        mt5.symbol_info_tick.return_value = _tick()

        def _positions_get(*args, **kwargs):
            t = kwargs.get("ticket")
            if t is not None:
                return [p for p in positions if p.ticket == t] or None
            return positions
        mt5.positions_get.side_effect = _positions_get

        from mtdata.core.trading import _close_positions
        res = _close_positions(symbol="EURUSD")

        results = res.get("results", [res])
        aborted = [r for r in results if r.get("aborted")]
        # Reset prevents aborts
        assert len(aborted) == 0

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_loss_first_integration(self):
        """close_priority=loss_first closes losers before winners."""
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)

        positions = [
            SimpleNamespace(ticket=1, symbol="EURUSD", volume=0.1, type=0, profit=100.0, magic=0, price_open=1.04, time=1700000000),
            SimpleNamespace(ticket=2, symbol="EURUSD", volume=0.1, type=0, profit=-50.0, magic=0, price_open=1.06, time=1700000000),
            SimpleNamespace(ticket=3, symbol="EURUSD", volume=0.1, type=0, profit=-200.0, magic=0, price_open=1.07, time=1700000000),
        ]

        mt5.order_send.return_value = _order_result()
        mt5.symbol_info_tick.return_value = _tick()

        def _positions_get(*args, **kwargs):
            t = kwargs.get("ticket")
            if t is not None:
                return [p for p in positions if p.ticket == t] or None
            return positions
        mt5.positions_get.side_effect = _positions_get

        from mtdata.core.trading import _close_positions
        res = _close_positions(symbol="EURUSD", close_priority="loss_first")

        assert res["close_priority"] == "loss_first"
        tickets_in_order = [r["ticket"] for r in res["results"]]
        # Losers closed first (ticket 3, then 2), then winner (ticket 1)
        assert tickets_in_order == [3, 2, 1]


# ===================================================================
#  _cancel_pending (MT5 mocked)
# ===================================================================

class TestCancelPending:

    def _setup_mt5(self, mt5):
        mt5.TRADE_ACTION_REMOVE = 8
        mt5.TRADE_RETCODE_DONE = 10009
        mt5.TRADE_RETCODE_DONE_PARTIAL = 10010

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_ticket_not_found(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = None
        from mtdata.core.trading import _cancel_pending
        result = _cancel_pending(ticket=999)
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_no_pending_orders(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = []
        from mtdata.core.trading import _cancel_pending
        result = _cancel_pending()
        assert "message" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_snapshot_failure_is_not_reported_as_no_pending_orders(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = None
        from mtdata.core.trading import _cancel_pending

        result = _cancel_pending()

        assert result["success"] is False
        assert result["error_code"] == "orders_snapshot_unavailable"

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_single_cancel_success(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order(ticket=100)]
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _cancel_pending
        result = _cancel_pending(ticket=100)
        assert result["success"] is True
        assert result["ticket"] == 100

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_single_cancel_rejection_is_not_success(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order(ticket=100)]
        mt5.order_send.return_value = _order_result(
            retcode=10006,
            comment="Rejected",
        )
        from mtdata.core.trading import _cancel_pending

        result = _cancel_pending(ticket=100)

        assert result["success"] is False
        assert result["execution_status"] == "rejected"
        assert "not completed" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_cancel_order_send_none(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order(ticket=100)]
        mt5.order_send.return_value = None
        from mtdata.core.trading import _cancel_pending
        result = _cancel_pending(ticket=100)
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_bulk_cancel(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [
            _pending_order(ticket=100),
            _pending_order(ticket=101),
        ]
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _cancel_pending
        result = _cancel_pending()
        assert result["attempted_count"] == 2

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_symbol_no_pending_orders(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = []
        from mtdata.core.trading import _cancel_pending
        result = _cancel_pending(symbol="GBPUSD")
        assert "message" in result
