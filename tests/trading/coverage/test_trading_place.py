"""Tests for trade place operations in mtdata.core.trading.

Covers:
- trade_place (dispatch logic)
- _place_market_order (MT5)
- _place_pending_order (MT5)
"""

import importlib
import math
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

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
_mt5_stub.ORDER_TIME_GTC = 0
_mt5_stub.ORDER_TIME_SPECIFIED = 2
_mt5_stub.ORDER_FILLING_IOC = 1
_mt5_stub.POSITION_TYPE_BUY = 0
_mt5_stub.POSITION_TYPE_SELL = 1
sys.modules["MetaTrader5"] = _mt5_stub

from mtdata.core.trading import trade_place as _trade_place_tool
from mtdata.core.trading.orders import (
    _prepare_order_symbol_context,
    _send_order_with_fill_mode_retry,
    _submit_order_request,
)
from mtdata.core.trading.requests import TradePlaceRequest
from mtdata.core.trading.use_cases.common import _should_persist_idempotency_outcome

# ===================================================================
# Helpers
# ===================================================================

def trade_place(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradePlaceRequest(**kwargs)
    return _trade_place_tool(request=request, __cli_raw=raw_output)


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
    return SimpleNamespace(bid=bid, ask=ask, time_msc=time.time() * 1000.0)


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


def test_place_does_not_retry_fill_mode_after_timeout():
    mt5 = MagicMock()
    mt5.TRADE_RETCODE_DONE = 10009
    mt5.TRADE_RETCODE_DONE_PARTIAL = 10010
    mt5.TRADE_RETCODE_PLACED = 10008
    mt5.TRADE_RETCODE_TIMEOUT = 10012
    mt5.TRADE_RETCODE_INVALID_FILL = 10030
    mt5.retcode_name.side_effect = lambda value: str(value)
    mt5.order_send.side_effect = [
        _order_result(retcode=10012, comment="timeout"),
        _order_result(),
    ]

    result, _, attempts, _ = _send_order_with_fill_mode_retry(
        mt5, {"type_filling": 1}, symbol_info=None
    )

    assert result.retcode == 10012
    assert len(attempts) == 1
    assert mt5.order_send.call_count == 1


def test_place_retries_fill_mode_only_after_invalid_fill():
    mt5 = MagicMock()
    mt5.TRADE_RETCODE_DONE = 10009
    mt5.TRADE_RETCODE_DONE_PARTIAL = 10010
    mt5.TRADE_RETCODE_PLACED = 10008
    mt5.TRADE_RETCODE_INVALID_FILL = 10030
    mt5.ORDER_FILLING_FOK = 0
    mt5.ORDER_FILLING_IOC = 1
    mt5.ORDER_FILLING_RETURN = 2
    mt5.retcode_name.side_effect = lambda value: str(value)
    mt5.order_send.side_effect = [
        _order_result(retcode=10030, comment="invalid fill"),
        _order_result(),
    ]

    result, _, attempts, _ = _send_order_with_fill_mode_retry(
        mt5, {"type_filling": 1}, symbol_info=None
    )

    assert result.retcode == 10009
    assert len(attempts) == 2
    assert mt5.order_send.call_count == 2


def test_ambiguous_order_send_outcome_is_idempotency_safe():
    assert _should_persist_idempotency_outcome(
        {"error_code": "order_send_ambiguous", "ambiguous": True}
    )


def test_dry_run_preview_is_not_persisted_as_idempotent_outcome():
    assert not _should_persist_idempotency_outcome(
        {"success": True, "dry_run": True, "no_action": True}
    )


def test_order_preflight_blocks_critical_account_margin_before_symbol_lookup():
    gateway = MagicMock()
    gateway.build_trade_preflight.return_value = {"execution_ready_strict": True}
    gateway.account_info.return_value = SimpleNamespace(
        equity=1000.0, margin=950.0, margin_free=50.0, margin_level=105.26
    )

    context, error = _prepare_order_symbol_context(
        gateway, symbol="EURUSD", volume=0.01
    )

    assert context is None
    assert error["error_code"] == "trade_blocked_account_state"
    assert "critical_margin_stress" in error["blockers"]
    gateway.symbol_info.assert_not_called()


def test_order_preflight_blocks_when_account_snapshot_is_unavailable():
    gateway = MagicMock()
    gateway.build_trade_preflight.return_value = {"execution_ready_strict": True}
    gateway.account_info.return_value = None

    context, error = _prepare_order_symbol_context(
        gateway, symbol="EURUSD", volume=0.01
    )

    assert context is None
    assert error["error_code"] == "account_snapshot_unavailable"
    assert error["blockers"] == ["account_snapshot_unavailable"]
    gateway.symbol_info.assert_not_called()


def _critical_margin_gateway(*, margin_mode: int, positions):
    from mtdata.core.trading.gateway import MT5TradingGateway

    adapter = MagicMock()
    adapter.account_info.return_value = SimpleNamespace(
        equity=1000.0,
        margin=1000.0,
        margin_free=0.0,
        margin_level=100.0,
        margin_mode=margin_mode,
        trade_allowed=True,
    )
    adapter.positions_get.return_value = list(positions)
    adapter.symbol_info.return_value = _sym()
    adapter.symbol_info_tick.return_value = _tick()
    adapter.symbol_select.return_value = True
    adapter.order_send.return_value = _order_result(volume=0.5)
    adapter.ORDER_TYPE_BUY = 0
    adapter.ORDER_TYPE_SELL = 1
    adapter.POSITION_TYPE_BUY = 0
    adapter.POSITION_TYPE_SELL = 1
    adapter.TRADE_ACTION_DEAL = 1
    adapter.TRADE_ACTION_SLTP = 6
    adapter.TRADE_RETCODE_DONE = 10009
    adapter.TRADE_RETCODE_DONE_PARTIAL = 10010
    adapter.TRADE_RETCODE_PLACED = 10008
    adapter.ORDER_TIME_GTC = 0
    adapter.ORDER_FILLING_IOC = 1
    return MT5TradingGateway(
        adapter=adapter,
        ensure_connection_impl=lambda: None,
        build_trade_preflight_impl=lambda *_args, **_kwargs: {
            "execution_ready_strict": True
        },
        retcode_name_impl=lambda _adapter, retcode: str(retcode),
    )


def test_critical_margin_allows_proven_netting_reduction():
    from mtdata.core.trading.orders import _place_market_order

    gateway = _critical_margin_gateway(
        margin_mode=0,
        positions=[_position(type_=0, volume=1.0)],
    )

    result = _place_market_order(
        "EURUSD",
        0.5,
        "SELL",
        gateway=gateway,
    )

    assert result["success"] is True
    assert gateway.adapter.order_send.call_count == 1


def test_critical_margin_still_blocks_hedging_order():
    from mtdata.core.trading.orders import _place_market_order

    gateway = _critical_margin_gateway(
        margin_mode=2,
        positions=[_position(type_=0, volume=1.0)],
    )

    result = _place_market_order(
        "EURUSD",
        0.5,
        "SELL",
        gateway=gateway,
    )

    assert result["error_code"] == "trade_blocked_account_state"
    gateway.adapter.order_send.assert_not_called()


def test_netting_order_protection_cannot_mutate_existing_position():
    from mtdata.core.trading.orders import _place_market_order

    gateway = _critical_margin_gateway(
        margin_mode=0,
        positions=[_position(type_=0, volume=1.0)],
    )
    gateway.adapter.account_info.return_value = SimpleNamespace(
        equity=1000.0,
        margin=100.0,
        margin_free=900.0,
        margin_level=1000.0,
        margin_mode=0,
        trade_allowed=True,
    )

    result = _place_market_order(
        "EURUSD",
        0.1,
        "BUY",
        stop_loss=1.0,
        take_profit=1.2,
        gateway=gateway,
    )

    assert result["error_code"] == "netting_position_protection_conflict"
    gateway.adapter.order_send.assert_not_called()


def test_failed_order_omits_success_last_error_sentinel():
    gateway = MagicMock()
    gateway.TRADE_RETCODE_DONE = 10009
    gateway.TRADE_RETCODE_DONE_PARTIAL = 10010
    gateway.TRADE_RETCODE_PLACED = 10008
    gateway.TRADE_RETCODE_INVALID_FILL = 10030
    gateway.ORDER_FILLING_FOK = 0
    gateway.ORDER_FILLING_IOC = 1
    gateway.ORDER_FILLING_RETURN = 2
    gateway.order_send.return_value = _order_result(retcode=10004)
    gateway.last_error.return_value = (1, "Success")
    gateway.retcode_name.side_effect = lambda value: str(value)

    outcome, error = _submit_order_request(
        gateway,
        {"type_filling": 1},
        base_error="Order failed",
        invalid_comment_error="Invalid comment",
    )

    assert outcome is None
    assert error["retcode"] == 10004
    assert "last_error" not in error


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
#  trade_place  (dispatch / validation – no MT5 calls for validation errors)
# ===================================================================

class TestTradePlace:

    def test_missing_symbol(self):
        with pytest.raises(ValidationError, match="symbol"):
            trade_place(symbol=None, volume=0.01, order_type="BUY")

    def test_missing_volume(self):
        with pytest.raises(ValidationError, match="volume"):
            trade_place(symbol="EURUSD", volume=None, order_type="BUY")

    def test_missing_order_type(self):
        with pytest.raises(ValidationError, match="order_type"):
            trade_place(symbol="EURUSD", volume=0.01, order_type=None)

    def test_all_missing(self):
        with pytest.raises(ValidationError, match="Field required"):
            trade_place()

    def test_empty_order_type_string(self):
        with pytest.raises(ValidationError, match="order_type"):
            trade_place(symbol="EURUSD", volume=0.01, order_type="  ")

    def test_invalid_order_type(self):
        with pytest.raises(ValidationError, match="order_type"):
            trade_place(symbol="EURUSD", volume=0.01, order_type="GARBAGE")

    def test_pending_without_price_returns_error(self):
        result = _unwrap_mcp(trade_place(symbol="EURUSD", volume=0.01, order_type="BUY_LIMIT", price=None))
        assert "price" in result.lower()

    def test_invalid_expiration_returns_error(self):
        result = _unwrap_mcp(trade_place(
            symbol="EURUSD", volume=0.01, order_type="BUY_LIMIT",
            price=1.09, expiration="not-a-date-at-all-xyz",
        ))
        assert "error" in result.lower() or "unsupported" in result.lower()

    @patch("mtdata.core.trading._place_market_order")
    def test_dispatches_to_market(self, mock_market):
        mock_market.return_value = {"retcode": 10009}
        trade_place(
            symbol="EURUSD",
            volume=0.01,
            order_type="BUY",
            require_sl_tp=False,
            dry_run=False,
        )
        mock_market.assert_called_once()

    @patch("mtdata.core.trading._place_pending_order")
    def test_dispatches_to_pending_explicit_type(self, mock_pending):
        mock_pending.return_value = {"success": True}
        trade_place(
            symbol="EURUSD",
            volume=0.01,
            order_type="BUY_LIMIT",
            price=1.09,
            require_sl_tp=False,
            dry_run=False,
        )
        mock_pending.assert_called_once()

    def test_rejects_market_order_with_price(self):
        with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
            "mtdata.core.trading._place_pending_order"
        ) as mock_pending:
            result = trade_place(
                symbol="EURUSD",
                volume=0.01,
                order_type="BUY",
                price=1.09,
                __cli_raw=True,
            )
        assert "Conflicting arguments" in result["error"]
        assert "BUY_LIMIT/BUY_STOP" in result["error"]
        mock_market.assert_not_called()
        mock_pending.assert_not_called()

    def test_rejects_market_order_with_price_and_expiration(self):
        with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
            "mtdata.core.trading._place_pending_order"
        ) as mock_pending:
            result = trade_place(
                symbol="EURUSD",
                volume=0.01,
                order_type="BUY",
                price=1.09,
                expiration="GTC",
                __cli_raw=True,
            )
        assert "Conflicting arguments" in result["error"]
        mock_market.assert_not_called()
        mock_pending.assert_not_called()


# ===================================================================
#  _place_market_order (MT5 mocked)
# ===================================================================

class TestPlaceMarketOrder:

    def _setup_mt5(self, mock_mt5):
        mock_mt5.symbol_info.return_value = _sym()
        mock_mt5.symbol_info_tick.return_value = _tick()
        mock_mt5.symbol_select.return_value = True
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.ORDER_TYPE_SELL = 1
        mock_mt5.TRADE_ACTION_DEAL = 1
        mock_mt5.TRADE_ACTION_SLTP = 6
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TIME_GTC = 0
        mock_mt5.ORDER_FILLING_IOC = 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_symbol_not_found(self):
        import sys
        mt5 = sys.modules["MetaTrader5"]
        mt5.symbol_info.return_value = None
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("INVALID", 0.01, "BUY")
        assert "error" in result and "not found" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_returns_none(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = None
        mt5.last_error.return_value = (10006, "No connection")
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_retcode_fail(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result(retcode=10004)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        assert "error" in result and result["retcode"] == 10004

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_invisible_symbol_selected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.symbol_info.return_value = _sym(visible=False)
        mt5.symbol_select.return_value = True
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        mt5.symbol_select.assert_called_with("EURUSD", True)

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_invisible_symbol_select_fails(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.symbol_info.return_value = _sym(visible=False)
        mt5.symbol_select.return_value = False
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        assert "error" in result and "Failed to select" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_buy_sl_above_price_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.2)
        assert "error" in result and "below" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sell_sl_below_price_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "SELL", stop_loss=1.0)
        assert "error" in result and "above" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_buy_tp_below_price_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", take_profit=1.0)
        assert "error" in result and "above" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sell_tp_above_price_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "SELL", take_profit=1.2)
        assert "error" in result and "below" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_success_with_sl_tp_modification(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        mt5.positions_get.return_value = [_position()]
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.09, take_profit=1.12)
        sl_tp_result = result.get("sl_tp_result") or {}
        assert sl_tp_result.get("status") == "applied"
        assert sl_tp_result.get("requested") == {"sl": pytest.approx(1.09), "tp": pytest.approx(1.12)}
        assert sl_tp_result.get("applied") == {"sl": pytest.approx(1.09), "tp": pytest.approx(1.12)}
        assert sl_tp_result.get("broker_adjusted") is None
        assert sl_tp_result.get("adjustment") is None
        assert sl_tp_result.get("error") is None
        assert result.get("protection_status") == "protected"
        assert mt5.order_send.call_count == 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_follow_up_modify_verifies_missing_protection(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.side_effect = [_order_result(), _order_result()]
        mt5.positions_get.side_effect = [
            [_position(sl=0.0, tp=0.0)],
            [_position(sl=1.09, tp=1.12)],
        ]
        from mtdata.core.trading import _place_market_order

        result = _place_market_order(
            "EURUSD",
            0.01,
            "BUY",
            stop_loss=1.09,
            take_profit=1.12,
        )

        sl_tp_result = result.get("sl_tp_result") or {}
        assert sl_tp_result.get("status") == "applied"
        assert sl_tp_result.get("applied") == {"sl": pytest.approx(1.09), "tp": pytest.approx(1.12)}
        assert result.get("protection_status") == "protected"
        assert mt5.order_send.call_count == 2

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_follow_up_preserves_existing_take_profit(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.side_effect = [_order_result(), _order_result()]
        mt5.positions_get.side_effect = [
            [_position(sl=0.0, tp=1.12)],
            [_position(sl=1.09, tp=1.12)],
        ]
        from mtdata.core.trading import _place_market_order

        result = _place_market_order(
            "EURUSD",
            0.01,
            "BUY",
            stop_loss=1.09,
        )

        modify_request = mt5.order_send.call_args_list[1].args[0]
        assert modify_request["sl"] == pytest.approx(1.09)
        assert modify_request["tp"] == pytest.approx(1.12)
        assert (result.get("sl_tp_result") or {}).get("status") == "applied"

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_tp_follow_up_preserves_existing_stop_loss(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.side_effect = [_order_result(), _order_result()]
        mt5.positions_get.side_effect = [
            [_position(sl=1.09, tp=0.0)],
            [_position(sl=1.09, tp=1.12)],
        ]
        from mtdata.core.trading import _place_market_order

        result = _place_market_order(
            "EURUSD",
            0.01,
            "BUY",
            take_profit=1.12,
        )

        modify_request = mt5.order_send.call_args_list[1].args[0]
        assert modify_request["sl"] == pytest.approx(1.09)
        assert modify_request["tp"] == pytest.approx(1.12)
        assert (result.get("sl_tp_result") or {}).get("status") == "applied"

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_follow_up_does_not_retry_ambiguous_none(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.side_effect = [_order_result(), None]
        mt5.positions_get.return_value = [_position(sl=0.0, tp=0.0)]
        from mtdata.core.trading import _place_market_order

        result = _place_market_order(
            "EURUSD",
            0.01,
            "BUY",
            stop_loss=1.09,
            take_profit=1.12,
        )

        assert mt5.order_send.call_count == 2
        assert (result.get("sl_tp_result") or {}).get("status") == "unverified"
        assert "not retried" in (result.get("sl_tp_result") or {}).get("error", "")

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_resolution_rejects_same_symbol_heuristic_match(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result(order=10, deal=20)
        mt5.history_deals_get.return_value = []
        mt5.positions_get.return_value = [
            _position(ticket=99, sl=0.0, tp=0.0, time=int(time.time()))
        ]
        from mtdata.core.trading import _place_market_order

        with patch(
            "mtdata.core.trading.orders._POSITION_RESOLUTION_WAIT_SCHEDULE_SECONDS",
            (),
        ):
            result = _place_market_order(
                "EURUSD",
                0.01,
                "BUY",
                stop_loss=1.09,
                take_profit=1.12,
            )

        assert mt5.order_send.call_count == 1
        assert (result.get("sl_tp_result") or {}).get("status") == "unverified"
        assert result.get("position_ticket") is None

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_readback_rejects_different_position_ticket(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.side_effect = [_order_result(), _order_result()]
        mt5.positions_get.side_effect = [
            [_position(ticket=1, sl=0.0, tp=0.0)],
            [_position(ticket=99, sl=1.09, tp=1.12)],
        ]
        from mtdata.core.trading import _place_market_order

        result = _place_market_order(
            "EURUSD", 0.01, "BUY", stop_loss=1.09, take_profit=1.12
        )

        assert (result.get("sl_tp_result") or {}).get("status") == "unverified"
        assert result.get("protection_status") == "protection_unverified"

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_verification_tolerates_missing_symbol_point(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.symbol_info.return_value = _sym(point=None)
        mt5.order_send.side_effect = [_order_result(), _order_result()]
        mt5.positions_get.return_value = [_position()]
        from mtdata.core.trading import _place_market_order

        result = _place_market_order(
            "EURUSD",
            0.01,
            "BUY",
            stop_loss=1.09,
            take_profit=1.12,
        )

        assert (result.get("sl_tp_result") or {}).get("status") == "applied"

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_uses_resolved_position_ticket_without_follow_up(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.side_effect = [
            _order_result(order=1001, deal=2002),
        ]
        mt5.positions_get.side_effect = [
            [],  # candidate order ticket lookup
            [_position(ticket=3003, deal=2002, sl=1.09, tp=1.12)],  # candidate deal ticket lookup
        ]
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.09, take_profit=1.12)
        assert (result.get("sl_tp_result") or {}).get("status") == "applied"
        assert result.get("position_ticket") == 3003
        assert mt5.order_send.call_count == 1
        request = mt5.order_send.call_args.args[0]
        assert request.get("sl") == pytest.approx(1.09)
        assert request.get("tp") == pytest.approx(1.12)

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_attach_failure_surfaces_broker_diagnostics(self):
        """When the post-fill SL/TP attach never returns DONE, the error must carry
        the broker retcode/comment rather than a generic 'Failed to set TP/SL'."""
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)

        calls = {"n": 0}

        def _send(_req):
            calls["n"] += 1
            if calls["n"] == 1:
                return _order_result()  # market order fill
            return _order_result(retcode=10004, comment="Requote")

        mt5.order_send.side_effect = _send
        # Position has no protection yet so the attach loop (Phase 2) runs.
        mt5.positions_get.return_value = [_position(sl=0.0, tp=0.0)]
        with patch("mtdata.core.trading.orders._stdlib_time.sleep", lambda *_a, **_k: None):
            from mtdata.core.trading import _place_market_order
            result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.09, take_profit=1.12)
        sl_tp_result = result.get("sl_tp_result") or {}
        assert sl_tp_result.get("status") == "failed"
        error_text = str(sl_tp_result.get("error") or "")
        assert "10004" in error_text
        assert "Requote" in error_text
        assert sl_tp_result.get("last_retcode") == 10004

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_attach_stops_after_ambiguous_none_return(self):
        """A missing write response must not trigger another modify request."""
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)

        # place -> REQUOTE -> None (ambiguous); the later DONE must not be sent.
        results = [
            _order_result(),                                # market order fill
            _order_result(retcode=10004, comment="Requote"),
            None,
            _order_result(retcode=10009, comment="done"),
        ]
        calls = {"n": 0}

        def _send(_req):
            idx = calls["n"]
            calls["n"] += 1
            return results[idx] if idx < len(results) else _order_result()

        mt5.order_send.side_effect = _send
        mt5.positions_get.side_effect = [
            [_position(sl=0.0, tp=0.0)],   # initial resolution: no protection yet
            [_position(sl=1.09, tp=1.12)],  # readback after DONE
        ]
        with patch("mtdata.core.trading.orders._stdlib_time.sleep", lambda *_a, **_k: None):
            from mtdata.core.trading import _place_market_order
            result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.09, take_profit=1.12)
        # Placement + REQUOTE + ambiguous None => no further write.
        assert calls["n"] == 3
        assert (result.get("sl_tp_result") or {}).get("status") == "unverified"

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_initial_protected_order_failure_returns_error(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result(retcode=10016, comment="Invalid stops")
        mt5.positions_get.return_value = [_position()]
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.09, take_profit=1.12)
        assert result.get("error") == "Failed to send order"
        assert result.get("retcode") == 10016
        assert "sl_tp_result" not in result
        assert "fallback_used" not in result
        assert mt5.order_send.call_count == 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_position_not_found_does_not_mark_protected_order_failed(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        mt5.positions_get.return_value = []
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.09)
        sl_tp_result = result.get("sl_tp_result") or {}
        assert sl_tp_result.get("requested") == {"sl": pytest.approx(1.09)}
        assert sl_tp_result.get("status") == "unverified"
        assert sl_tp_result.get("applied") is None
        assert result.get("position_ticket") is None
        assert result.get("position_ticket_candidates") == [1]
        assert result.get("protection_status") == "protection_unverified"
        assert any("could not be verified" in str(w).lower() for w in result.get("warnings", []))
        assert not any("CRITICAL" in str(w) for w in result.get("warnings", []))

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_not_requested_state_is_explicit(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        assert result.get("sl_tp_result") == {"status": "not_requested"}

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_queued_market_order_is_not_reported_as_filled(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result(retcode=10008, volume=0.0)
        from mtdata.core.trading import _place_market_order

        result = _place_market_order("EURUSD", 0.01, "BUY")

        assert result["success"] is False
        assert result["execution_status"] == "queued"
        assert result["requested_volume"] == 0.01
        assert result["filled_volume"] == 0.0
        assert result["remaining_volume"] == 0.01

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_partial_market_order_exposes_remaining_volume(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result(retcode=10010, volume=0.004)
        from mtdata.core.trading import _place_market_order

        result = _place_market_order("EURUSD", 0.01, "BUY")

        assert result["success"] is False
        assert result["execution_status"] == "partial"
        assert result["filled_volume"] == 0.004
        assert result["remaining_volume"] == pytest.approx(0.006)

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_comment_truncation_is_reported(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_market_order
        long_comment = "X" * 60
        result = _place_market_order("EURUSD", 0.01, "BUY", comment=long_comment)
        trunc = result.get("comment_truncation")
        assert isinstance(trunc, dict)
        assert trunc.get("requested") == long_comment
        assert trunc.get("max_length") == 31
        assert len(str(trunc.get("applied") or "")) <= 31
        assert any("Comment truncated" in str(w) for w in result.get("warnings", []))

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_comment_rejection_returns_error_without_fallback(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.side_effect = [
            _order_result(retcode=10013, comment='Invalid "comment" argument'),
        ]
        from mtdata.core.trading import _place_market_order

        result = _place_market_order(
            "BTCUSD",
            0.01,
            "SELL",
            comment="Short: ETS bearish, barrier EV+, R:R 9.65",
        )

        assert "broker rejected the comment field" in result["error"]
        assert "comment_fallback" not in result
        assert mt5.order_send.call_count == 1
        first_request = mt5.order_send.call_args_list[0].args[0]
        assert first_request["comment"] == "Short ETS bearish barrier EV R"

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_atomic_sl_tp_result_does_not_report_readback_adjustment(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.side_effect = [_order_result(), _order_result()]
        mt5.positions_get.side_effect = [
            [_position(sl=1.09, tp=1.12)],
            [_position(sl=1.0895, tp=1.1203)],
        ]
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.09, take_profit=1.12)
        sl_tp_result = result.get("sl_tp_result") or {}
        assert sl_tp_result.get("status") == "applied"
        assert sl_tp_result.get("broker_adjusted") is None
        assert sl_tp_result.get("adjustment") is None
        assert mt5.order_send.call_count == 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_type_rejected_for_market(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY_LIMIT")
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_tick_none_returns_error(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.symbol_info_tick.return_value = None
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        assert "error" in result and "current price" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_stale_tick_returns_error(self):
        import time as _time_module

        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.symbol_info_tick.return_value = _tick()
        mt5.symbol_info_tick.return_value.time_msc = (_time_module.time() - 60.0) * 1000.0
        from mtdata.core.trading import _place_market_order

        result = _place_market_order("EURUSD", 0.01, "BUY")

        assert "stale" in result.get("error", "").lower()
        assert result.get("tick_age_seconds", 0) >= 50

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_stale_send_tick_returns_error(self):
        import time as _time_module

        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        fresh_tick = _tick()
        fresh_tick.time_msc = (_time_module.time() - 1.0) * 1000.0
        stale_tick = _tick()
        stale_tick.time_msc = (_time_module.time() - 60.0) * 1000.0
        mt5.symbol_info_tick.side_effect = [fresh_tick, stale_tick]
        from mtdata.core.trading import _place_market_order

        result = _place_market_order("EURUSD", 0.01, "BUY")

        assert "stale" in result.get("error", "").lower()
        assert result.get("tick_age_seconds", 0) >= 50

    def test_accepts_injected_gateway(self):
        """Market order placement accepts an injected MT5TradingGateway."""
        from mtdata.core.trading import _place_market_order
        from mtdata.core.trading.gateway import MT5TradingGateway

        adapter = MagicMock()
        adapter.symbol_info.return_value = _sym()
        adapter.symbol_info_tick.side_effect = [
            _tick(bid=1.05000, ask=1.05010),
            _tick(bid=1.05000, ask=1.05010),
        ]
        adapter.positions_get.return_value = [MagicMock(sl=1.04000, tp=1.06000)]
        adapter.order_send.side_effect = [
            MagicMock(
                retcode=10009,
                deal=123,
                order=456,
                volume=0.1,
                price=1.05010,
                bid=1.05000,
                ask=1.05010,
                comment="",
                request_id=789,
            ),
            MagicMock(retcode=10009, comment="", request_id=790),
        ]
        adapter.ORDER_TYPE_BUY = 0
        adapter.ORDER_TYPE_SELL = 1
        adapter.TRADE_ACTION_DEAL = 1
        adapter.TRADE_ACTION_SLTP = 6
        adapter.TRADE_RETCODE_DONE = 10009
        adapter.ORDER_TIME_GTC = 0
        adapter.ORDER_FILLING_IOC = 1
        ensure_connection = MagicMock()
        gateway = MT5TradingGateway(
            adapter=adapter,
            ensure_connection_impl=ensure_connection,
            build_trade_preflight_impl=lambda mt5, **_: {
                "execution_ready": True,
                "execution_ready_strict": True,
            },
            retcode_name_impl=lambda mt5, retcode: "TRADE_RETCODE_DONE",
        )

        with patch(
            "mtdata.core.trading.orders._resolve_open_position",
            return_value=(MagicMock(sl=0.0, tp=0.0), 456, {}),
        ), patch("mtdata.core.trading.orders.trade_guardrails_config") as mock_guard_config:
            mock_guard_config.enabled = False
            mock_guard_config.trading_enabled = True
            mock_guard_config.max_volume_by_symbol = {}
            mock_guard_config.is_enabled.return_value = False

            result = _place_market_order(
                symbol="EURUSD",
                volume=0.1,
                order_type="BUY",
                stop_loss=1.04000,
                take_profit=1.06000,
                gateway=gateway,
            )

        assert "error" not in result
        ensure_connection.assert_called_once_with()
        assert adapter.order_send.call_count == 2
        initial_request = adapter.order_send.call_args_list[0].args[0]
        assert math.isclose(initial_request["sl"], 1.04000)
        assert math.isclose(initial_request["tp"], 1.06000)
        follow_up_request = adapter.order_send.call_args_list[1].args[0]
        assert follow_up_request["position"] == 456
        assert "comment_fallback" not in result


# ===================================================================
#  _place_pending_order (MT5 mocked)
# ===================================================================

class TestPlacePendingOrder:

    def _setup_mt5(self, mock_mt5):
        mock_mt5.symbol_info.return_value = _sym()
        mock_mt5.symbol_info_tick.return_value = _tick(bid=1.10000, ask=1.10020)
        mock_mt5.symbol_select.return_value = True
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.ORDER_TYPE_SELL = 1
        mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
        mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
        mock_mt5.ORDER_TYPE_BUY_STOP = 4
        mock_mt5.ORDER_TYPE_SELL_STOP = 5
        mock_mt5.ORDER_TYPE_BUY_STOP_LIMIT = 6
        mock_mt5.ORDER_TYPE_SELL_STOP_LIMIT = 7
        mock_mt5.TRADE_ACTION_PENDING = 5
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TIME_GTC = 0
        mock_mt5.ORDER_TIME_SPECIFIED = 2
        mock_mt5.ORDER_FILLING_IOC = 1
        mock_mt5.ORDER_FILLING_RETURN = 2

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_buy_limit_success(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09)
        assert result.get("success") is True
        assert result.get("requested_price") == 1.09
        assert result.get("requested_sl") is None
        assert result.get("requested_tp") is None
        request = mt5.order_send.call_args.args[0]
        assert request["type_filling"] == mt5.ORDER_FILLING_RETURN

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_accepted_pending_order_is_not_a_fill(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result(
            deal=0, price=0.0, bid=0.0, ask=0.0, comment="Request executed"
        )
        from mtdata.core.trading import _place_pending_order

        result = _place_pending_order(
            "EURUSD", 0.01, "BUY_LIMIT", price=1.09, comment="audit-pending"
        )

        assert result["success"] is True
        assert result["execution_status"] == "accepted"
        assert result["fill_status"] == "not_filled"
        assert result["order_status"] == "working"
        assert result["price"] is None
        assert result["bid"] is None
        assert result["ask"] is None
        assert result["applied_comment"] == "audit-pending"
        assert result["broker_message"] == "Request executed"

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_allows_stale_tick_for_pending_staging(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.symbol_info_tick.return_value = SimpleNamespace(
            bid=1.1000, ask=1.1002, time=1
        )
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order

        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09)

        assert result["success"] is True
        mt5.order_send.assert_called_once()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sell_stop_success(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "SELL_STOP", price=1.09)
        assert result.get("success") is True

    @pytest.mark.parametrize(
        ("order_type", "trigger", "limit_price", "expected_type"),
        [
            ("BUY_STOP_LIMIT", 1.1010, 1.10083, 6),
            ("SELL_STOP_LIMIT", 1.0990, 1.09917, 7),
        ],
    )
    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_stop_limit_submission_includes_normalized_second_price(
        self, order_type, trigger, limit_price, expected_type
    ):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order

        result = _place_pending_order(
            "EURUSD",
            0.01,
            order_type,
            price=trigger,
            stop_limit_price=limit_price,
        )

        assert result["success"] is True
        request = mt5.order_send.call_args.args[0]
        assert request["type"] == expected_type
        assert request["price"] == pytest.approx(trigger)
        assert request["stoplimit"] == pytest.approx(limit_price)
        assert result["requested_stop_limit_price"] == pytest.approx(limit_price)

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_buy_limit_price_above_ask_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.20)
        assert "error" in result and "below ask" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sell_limit_price_below_bid_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "SELL_LIMIT", price=1.05)
        assert "error" in result and "above bid" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_buy_stop_price_below_ask_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_STOP", price=1.05)
        assert "error" in result and "above ask" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sell_stop_price_above_bid_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "SELL_STOP", price=1.20)
        assert "error" in result and "below bid" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_sl_tp_buy_validation(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        # SL above entry for BUY_LIMIT should fail
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09, stop_loss=1.10)
        assert "error" in result and "below entry" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_tp_buy_validation(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        # TP below entry for BUY_LIMIT should fail
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09, take_profit=1.05)
        assert "error" in result and "above entry" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_sl_sell_validation(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        # SL below entry for SELL_LIMIT should fail
        result = _place_pending_order("EURUSD", 0.01, "SELL_LIMIT", price=1.12, stop_loss=1.10)
        assert "error" in result and "above entry" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_tp_sell_validation(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        # TP above entry for SELL_LIMIT should fail
        result = _place_pending_order("EURUSD", 0.01, "SELL_LIMIT", price=1.12, take_profit=1.15)
        assert "error" in result and "below entry" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_implicit_buy_below_ask_becomes_buy_limit(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY", price=1.09)
        # Verify the request used BUY_LIMIT (type=2)
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["type"] == mt5.ORDER_TYPE_BUY_LIMIT

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_implicit_sell_above_bid_becomes_sell_limit(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "SELL", price=1.12)
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["type"] == mt5.ORDER_TYPE_SELL_LIMIT

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_none_returns_error(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = None
        mt5.last_error.return_value = (10006, "err")
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09)
        assert result.get("error_code") == "order_send_ambiguous"
        assert result.get("ambiguous") is True
        assert mt5.order_send.call_count == 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_retcode_fail(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result(retcode=10004)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09)
        assert "error" in result

    @pytest.mark.parametrize(
        "price",
        [0, -1, float("nan"), float("inf"), float("-inf")],
    )
    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_invalid_pending_price_rejected_before_order_send(self, price):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=price)
        assert result["error_code"] == "invalid_pending_price"
        assert "strictly positive finite" in result["error"]
        mt5.order_send.assert_not_called()

    @pytest.mark.parametrize(
        "expiration",
        [0, -1, float("nan"), float("inf"), float("-inf"), "0", "-1"],
    )
    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_invalid_expiration_cannot_create_gtc_request(self, expiration):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order

        result = _place_pending_order(
            "EURUSD",
            0.01,
            "BUY_LIMIT",
            price=1.09,
            expiration=expiration,
        )

        assert result["error_code"] == "invalid_pending_expiration"
        mt5.order_send.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_explicit_gtc_token_sets_gtc_request(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order

        result = _place_pending_order(
            "EURUSD",
            0.01,
            "BUY_LIMIT",
            price=1.09,
            expiration="GTC",
        )

        assert result["success"] is True
        request = mt5.order_send.call_args.args[0]
        assert request["type_time"] == mt5.ORDER_TIME_GTC
        assert "expiration" not in request

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_with_expiration_sets_type_time_specified(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order
        # Use a numeric expiration (epoch seconds)
        with patch("mtdata.core.trading.time._normalize_pending_expiration", return_value=(1717200000, True)):
            result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09, expiration=1717200000)
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["type_time"] == 2  # ORDER_TIME_SPECIFIED
        assert call_args["expiration"] == 1717200000
