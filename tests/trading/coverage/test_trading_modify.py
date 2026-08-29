"""Tests for trade modify operations in mtdata.core.trading.

Covers:
- trade_modify (dispatch logic)
- _modify_position (MT5)
- _modify_pending_order (MT5)
"""

import importlib
import os
import sys
import time
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
_mt5_stub.TRADE_RETCODE_NO_CHANGES = 10025
_mt5_stub.ORDER_TIME_GTC = 0
_mt5_stub.ORDER_TIME_SPECIFIED = 2
_mt5_stub.ORDER_FILLING_IOC = 1
_mt5_stub.POSITION_TYPE_BUY = 0
_mt5_stub.POSITION_TYPE_SELL = 1
sys.modules["MetaTrader5"] = _mt5_stub

from mtdata.core.trading import (
    trade_modify as _trade_modify_tool,
)
from mtdata.core.trading.requests import (
    TradeModifyRequest,
)

# ===================================================================
# Helpers
# ===================================================================

def trade_modify(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        kwargs.setdefault("dry_run", False)
        request = TradeModifyRequest(**kwargs)
    return _trade_modify_tool(request=request, __cli_raw=raw_output)


def _unwrap_mcp(result):
    """MCP tool decorator serialises dicts to strings; return the raw dict
    when the underlying function is patched to return one, otherwise return
    the string for assertion with ``in``."""
    if isinstance(result, dict):
        return result
    return str(result)


def _sym(
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
    point=0.00001,
    digits=5,
    visible=True,
    trade_contract_size=100000,
    trade_tick_value=1.0,
    trade_tick_size=0.00001,
):
    """Create a mock symbol_info object."""
    return SimpleNamespace(
        volume_min=volume_min,
        volume_max=volume_max,
        volume_step=volume_step,
        point=point,
        digits=digits,
        visible=visible,
        trade_contract_size=trade_contract_size,
        trade_tick_value=trade_tick_value,
        trade_tick_size=trade_tick_size,
        symbol="EURUSD",
    )


def _tick(bid=1.10000, ask=1.10020):
    return SimpleNamespace(bid=bid, ask=ask, time=time.time())


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
#  trade_modify (dispatch)
# ===================================================================

class TestTradeModify:

    def test_request_accepts_dry_run(self):
        request = TradeModifyRequest(ticket=100, stop_loss=1.08, dry_run=True)

        assert request.dry_run is True

    @patch("mtdata.core.trading._modify_pending_order")
    def test_price_routes_to_pending(self, mock_pending):
        mock_pending.return_value = {"success": True}
        trade_modify(ticket=100, price=1.09)
        mock_pending.assert_called_once()

    @patch("mtdata.core.trading._modify_pending_order")
    def test_expiration_routes_to_pending(self, mock_pending):
        mock_pending.return_value = {"success": True}
        trade_modify(ticket=100, expiration="GTC")
        mock_pending.assert_called_once()

    @patch("mtdata.core.trading._modify_position")
    def test_sl_tp_only_tries_position_first(self, mock_pos):
        mock_pos.return_value = {"success": True}
        trade_modify(ticket=100, stop_loss=1.08)
        mock_pos.assert_called_once()

    @patch("mtdata.core.trading._modify_pending_order")
    @patch("mtdata.core.trading._modify_position")
    def test_dry_run_position_preview_skips_execution(self, mock_pos, mock_pend):
        mock_pos.return_value = {
            "success": True,
            "dry_run": True,
            "actionability": "preview_only",
        }

        out = trade_modify(ticket=100, stop_loss=1.08, dry_run=True, __cli_raw=True)

        assert out["success"] is True
        assert out["dry_run"] is True
        assert out["actionability"] == "preview_only"
        assert out["preview_ok"] is True
        assert out["would_send_order"] is False
        mock_pos.assert_called_once_with(
            ticket=100,
            stop_loss=1.08,
            take_profit=None,
            dry_run=True,
        )
        mock_pend.assert_not_called()

    def test_comment_modifications_are_rejected(self):
        with pytest.raises(Exception) as exc_info:
            TradeModifyRequest(ticket=100, comment="retag")
        assert "comment" in str(exc_info.value)

    @patch("mtdata.core.trading._modify_pending_order")
    @patch("mtdata.core.trading._modify_position")
    def test_position_not_found_falls_back_to_pending(self, mock_pos, mock_pend):
        mock_pos.return_value = {"error": "Position 100 not found"}
        mock_pend.return_value = {"success": True}
        result = _unwrap_mcp(trade_modify(ticket=100, stop_loss=1.08))
        assert "success" in result or "True" in result

    @patch("mtdata.core.trading._modify_pending_order")
    @patch("mtdata.core.trading._modify_position")
    def test_both_not_found(self, mock_pos, mock_pend):
        mock_pos.return_value = {"error": "Position 100 not found"}
        mock_pend.return_value = {"error": "Pending order 100 not found"}
        result = _unwrap_mcp(trade_modify(ticket=100, stop_loss=1.08))
        assert "not found" in result

    @patch("mtdata.core.trading._modify_pending_order")
    def test_pending_not_found_with_price(self, mock_pend):
        mock_pend.return_value = {"error": "Pending order 100 not found"}
        result = _unwrap_mcp(trade_modify(ticket=100, price=1.09))
        assert "Pending order 100 not found" in result

    def test_bad_expiration_returns_error(self):
        result = _unwrap_mcp(trade_modify(ticket=100, expiration="not-a-date-xyz-abc"))
        assert "error" in result.lower() or "unsupported" in result.lower()


# ===================================================================
#  _modify_position (MT5 mocked)
# ===================================================================

class TestModifyPosition:

    def _setup_mt5(self, mt5):
        mt5.TRADE_ACTION_SLTP = 6
        mt5.TRADE_RETCODE_DONE = 10009
        mt5.symbol_info_tick.return_value = _tick(bid=1.1050, ask=1.1052)

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_position_not_found(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = []
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=999)
        assert "error" in result and "not found" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_success(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.side_effect = [
            [_position()],
            [_position(sl=1.081, tp=1.129)],
        ]
        mt5.symbol_info.return_value = _sym()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08, take_profit=1.13)
        assert result.get("success") is True
        assert result["verification_status"] == "verified"
        assert result["applied_sl"] == pytest.approx(1.081)
        assert result["applied_tp"] == pytest.approx(1.129)
        assert result["broker_adjusted"] is True

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_dry_run_validates_and_skips_order_send(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position()]
        mt5.symbol_info.return_value = _sym()
        from mtdata.core.trading import _modify_position

        result = _modify_position(
            ticket=1,
            stop_loss=1.08,
            take_profit=1.13,
            dry_run=True,
        )

        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["operation"] == "modify_position"
        assert result["would_send_order"] is False
        assert result["position_ticket"] == 1
        assert result["applied_sl"] == 1.08
        assert result["applied_tp"] == pytest.approx(1.13)
        mt5.order_send.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_no_changes_returns_success_when_levels_already_match(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.TRADE_RETCODE_NO_CHANGES = 10025
        mt5.positions_get.return_value = [_position(sl=1.08, tp=1.13)]
        mt5.symbol_info.return_value = _sym()
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08, take_profit=1.13)
        assert result.get("success") is True
        assert result.get("no_change") is True
        assert result.get("retcode") == 10025
        mt5.order_send.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_resolves_position_by_identifier_when_ticket_lookup_misses(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.side_effect = [
            [],
            [_position(ticket=777, identifier=123)],
        ]
        mt5.symbol_info.return_value = _sym()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=123, stop_loss=1.08)
        assert result.get("success") is True
        req = mt5.order_send.call_args.args[0]
        assert req.get("symbol") == "EURUSD"
        assert req.get("position") == 777
        assert result.get("position_ticket") == 777

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_none(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position()]
        mt5.symbol_info.return_value = _sym()
        mt5.order_send.return_value = None
        mt5.last_error.return_value = (10006, "err")
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08)
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_retcode_fail(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position()]
        mt5.symbol_info.return_value = _sym()
        mt5.order_send.return_value = _order_result(retcode=10004)
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08)
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_broker_no_changes_retcode_is_success_when_live_position_matches(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.TRADE_RETCODE_NO_CHANGES = 10025
        mt5.positions_get.side_effect = [
            [_position(sl=1.09, tp=1.12)],
            [_position(sl=1.08, tp=1.12)],
        ]
        mt5.symbol_info.return_value = _sym()
        mt5.order_send.return_value = _order_result(retcode=10025, comment="No changes")
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08)
        assert result.get("success") is True
        assert result.get("no_change") is True
        assert result.get("retcode") == 10025

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_broker_no_changes_retcode_errors_when_live_position_still_differs(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.TRADE_RETCODE_NO_CHANGES = 10025
        mt5.positions_get.side_effect = [
            [_position(sl=1.09, tp=1.12)],
            [_position(sl=1.09, tp=1.12)],
        ]
        mt5.symbol_info.return_value = _sym()
        mt5.order_send.return_value = _order_result(retcode=10025, comment="No changes")
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08)
        assert result.get("error")
        assert result.get("no_change") is True
        assert result.get("desired_sl") == 1.08
        assert result.get("actual_sl") == 1.09

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_symbol_info_none(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position()]
        mt5.symbol_info.return_value = None
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08)
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_only_tp_change_revalidates_existing_sl(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(sl=1.10495, tp=1.12)]
        symbol_info = _sym()
        symbol_info.trade_stops_level = 30
        symbol_info.trade_freeze_level = 0
        mt5.symbol_info.return_value = symbol_info
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, take_profit=1.13)
        assert "error" in result
        assert "stop_loss is too close" in result["error"]
        mt5.order_send.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_digits_none_uses_safe_default_when_modifying_position(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position()]
        mt5.symbol_info.return_value = _sym(point=0.0, digits=None)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08001)
        assert result.get("success") is True

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_returns_error_when_position_side_cannot_be_resolved(self):
        class _BrokenTypePosition:
            ticket = 1
            symbol = "EURUSD"
            sl = 1.09
            tp = 1.12

            @property
            def type(self):
                raise RuntimeError("boom")

        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_BrokenTypePosition()]
        mt5.symbol_info.return_value = _sym()
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08)
        assert result["error"] == "Unable to determine position side for protection validation."
        mt5.order_send.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_tiny_stop_loss_is_treated_as_explicit_remove(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(sl=1.09, tp=1.12)]
        mt5.symbol_info.return_value = _sym()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1e-12)
        assert result.get("success") is True
        request = mt5.order_send.call_args.args[0]
        assert request["sl"] == 0.0
        assert request["tp"] == pytest.approx(1.12)

    def test_zero_stop_loss_is_rejected_without_clear_flag(self):
        out = trade_modify(ticket=100, stop_loss=0, dry_run=True, __cli_raw=True)
        assert out["success"] is False
        assert out["preview_ok"] is False
        assert out["error_code"] == "protection_clear_requires_flag"
        assert "clear_stop_loss" in out["error"]

    @patch("mtdata.core.trading._modify_pending_order")
    @patch("mtdata.core.trading._modify_position")
    def test_clear_stop_loss_flag_forwards_zero(self, mock_pos, mock_pend):
        mock_pos.return_value = {
            "success": True,
            "dry_run": True,
            "applied_sl": 0.0,
            "protection_removed": ["stop_loss"],
            "preview_ok": True,
        }
        out = trade_modify(
            ticket=100,
            clear_stop_loss=True,
            dry_run=True,
            __cli_raw=True,
        )
        assert out["success"] is True
        assert out["protection_removed"] == ["stop_loss"]
        mock_pos.assert_called_once_with(
            ticket=100,
            stop_loss=0.0,
            take_profit=None,
            dry_run=True,
        )
        mock_pend.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_stale_tick_modify_preview_allows_protection_edit(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(sl=1.09, tp=1.12)]
        mt5.symbol_info.return_value = _sym()
        mt5.symbol_info_tick.return_value = SimpleNamespace(
            bid=1.1050,
            ask=1.1052,
            time=time.time() - 60,
        )
        from mtdata.core.trading import _modify_position

        result = _modify_position(
            ticket=1,
            stop_loss=1.08,
            take_profit=1.13,
            dry_run=True,
        )
        assert result["success"] is True
        assert result["preview_ok"] is True
        assert result["applied_sl"] == pytest.approx(1.08)
        assert result["applied_tp"] == pytest.approx(1.13)
        mt5.order_send.assert_not_called()


# ===================================================================
#  _modify_pending_order (MT5 mocked)
# ===================================================================

class TestModifyPendingOrder:

    def _setup_mt5(self, mt5):
        mt5.TRADE_ACTION_MODIFY = 7
        mt5.TRADE_RETCODE_DONE = 10009
        mt5.ORDER_TIME_GTC = 0
        mt5.ORDER_TIME_SPECIFIED = 2
        mt5.ORDER_TYPE_BUY_LIMIT = 2
        mt5.ORDER_TYPE_SELL_LIMIT = 3
        mt5.ORDER_TYPE_BUY_STOP = 4
        mt5.ORDER_TYPE_SELL_STOP = 5
        mt5.symbol_info.return_value = _sym()
        mt5.symbol_info_tick.return_value = _tick()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_not_found(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = []
        from mtdata.core.trading import _modify_pending_order
        result = _modify_pending_order(ticket=999)
        assert "error" in result and "not found" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_success(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.side_effect = [
            [_pending_order()],
            [_pending_order(price_open=1.0948)],
        ]
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_pending_order
        result = _modify_pending_order(ticket=100, price=1.095)
        assert result.get("success") is True
        assert result["verification_status"] == "verified"
        assert result["applied_price"] == pytest.approx(1.0948)
        assert result["broker_adjusted"] is True

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_allows_untimestamped_tick_for_pending_modify(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.side_effect = [
            [_pending_order()],
            [_pending_order(price_open=1.0948)],
        ]
        mt5.symbol_info_tick.return_value = SimpleNamespace(bid=1.1, ask=1.1002)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_pending_order

        result = _modify_pending_order(ticket=100, price=1.095)

        assert result["success"] is True
        mt5.order_send.assert_called_once()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_broker_no_changes_retcode_is_pending_modify_success(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.TRADE_RETCODE_NO_CHANGES = 10025
        mt5.orders_get.return_value = [_pending_order(price_open=1.095)]
        mt5.order_send.return_value = SimpleNamespace(
            retcode=10025,
            comment="No changes",
            request_id=123,
        )
        from mtdata.core.trading import _modify_pending_order

        result = _modify_pending_order(ticket=100, price=1.095)

        assert result.get("success") is True
        assert result.get("no_change") is True
        assert result["retcode"] == 10025
        assert result["pending_order_ticket"] == 100

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_dry_run_validates_and_skips_order_send(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order()]
        from mtdata.core.trading import _modify_pending_order

        result = _modify_pending_order(
            ticket=100,
            price=1.09,
            stop_loss=1.08,
            take_profit=1.11,
            dry_run=True,
        )

        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["operation"] == "modify_pending_order"
        assert result["would_send_order"] is False
        assert result["pending_order_ticket"] == 100
        assert result["applied_price"] == 1.09
        assert result["applied_sl"] == 1.08
        assert result["applied_tp"] == 1.11
        assert result["expiration_policy"] == "preserve_existing"
        assert result["expiration_explicit"] is False
        mt5.order_send.assert_not_called()

    @pytest.mark.parametrize(
        "expiration",
        [0, -1, float("nan"), float("inf"), float("-inf"), "0", "-1"],
    )
    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_invalid_expiration_cannot_modify_order_to_gtc(self, expiration):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order()]
        from mtdata.core.trading import _modify_pending_order

        result = _modify_pending_order(ticket=100, expiration=expiration)

        assert result["error_code"] == "invalid_pending_expiration"
        mt5.order_send.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_gtc_expiration_sets_type_time(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order()]
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_pending_order
        result = _modify_pending_order(ticket=100, expiration="GTC")
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["type_time"] == 0  # GTC

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_none(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order()]
        mt5.order_send.return_value = None
        mt5.last_error.return_value = (10006, "err")
        from mtdata.core.trading import _modify_pending_order
        result = _modify_pending_order(ticket=100)
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_preserves_existing_expiration(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        order = _pending_order(type_time=2, time_expiration=1717200000)
        mt5.orders_get.return_value = [order]
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_pending_order
        _modify_pending_order(ticket=100, price=1.095)
        call_args = mt5.order_send.call_args[0][0]
        assert call_args.get("expiration") == 1717200000

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_modify_request_includes_symbol_type_and_volume(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order(symbol="EURUSD", type_=2, volume=0.07)]
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_pending_order
        result = _modify_pending_order(ticket=100, price=1.095)
        assert result.get("success") is True
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["symbol"] == "EURUSD"
        assert call_args["type"] == mt5.ORDER_TYPE_BUY_LIMIT
        assert call_args["volume"] == pytest.approx(0.07)

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_modify_request_uses_live_pending_order_volume_fields(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        order = _pending_order(symbol="EURUSD", type_=2, volume=0.5)
        del order.volume
        order.volume_current = 0.5
        order.volume_initial = 0.5
        mt5.orders_get.return_value = [order]
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_pending_order

        result = _modify_pending_order(ticket=100, price=1.095)

        assert result.get("success") is True
        request = mt5.order_send.call_args.args[0]
        assert request["volume"] == pytest.approx(0.5)

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_modify_rejects_pending_order_without_positive_remaining_volume(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        order = _pending_order(volume=0.5)
        order.volume_current = 0.0
        order.volume_initial = 0.5
        mt5.orders_get.return_value = [order]
        from mtdata.core.trading import _modify_pending_order

        result = _modify_pending_order(ticket=100, price=1.095)

        assert result["error_code"] == "invalid_pending_order_volume"
        assert "no positive remaining volume" in result["error"]
        mt5.order_send.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_normalizes_pending_modify_prices(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order(price_open=1.0945, sl=1.0845, tp=1.1145)]
        mt5.symbol_info.return_value = _sym(point=0.0001, digits=4)
        mt5.symbol_info_tick.return_value = _tick(bid=1.1000, ask=1.1002)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_pending_order
        result = _modify_pending_order(ticket=100, price=1.09456, stop_loss=1.08456, take_profit=1.11456)
        assert result.get("success") is True
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["price"] == pytest.approx(1.0946)
        assert call_args["sl"] == pytest.approx(1.0846)
        assert call_args["tp"] == pytest.approx(1.1146)

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_modify_rejects_invalid_take_profit_side(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order(type_=3, price_open=1.1050, sl=1.1100, tp=1.0950)]
        mt5.symbol_info.return_value = _sym()
        mt5.symbol_info_tick.return_value = _tick(bid=1.1000, ask=1.1002)
        from mtdata.core.trading import _modify_pending_order
        result = _modify_pending_order(ticket=100, take_profit=1.1100)
        assert "error" in result
        assert "take_profit must be below entry for SELL orders" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_modify_rejects_entry_inside_broker_stop_distance(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order(type_=2, price_open=1.0950, sl=1.0850, tp=1.1150)]
        mt5.symbol_info.return_value = _sym(point=0.00001, digits=5)
        mt5.symbol_info.return_value.trade_stops_level = 30
        mt5.symbol_info.return_value.trade_freeze_level = 0
        mt5.symbol_info_tick.return_value = _tick(bid=1.1000, ask=1.1002)
        from mtdata.core.trading import _modify_pending_order
        result = _modify_pending_order(ticket=100, price=1.1000)
        assert "error" in result
        assert "pending entry is too close" in result["error"]
        assert "min_distance_points=30" in result["error"]
