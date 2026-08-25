"""Tests for account info and risk analysis in mtdata.core.trading.

Covers:
- trade_account_info (MT5 mocked)
- trade_risk_analyze (MT5 mocked)
"""

import importlib
import os
import sys
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
_mt5_stub.ORDER_TIME_GTC = 0
_mt5_stub.ORDER_TIME_SPECIFIED = 2
_mt5_stub.ORDER_FILLING_IOC = 1
_mt5_stub.POSITION_TYPE_BUY = 0
_mt5_stub.POSITION_TYPE_SELL = 1
sys.modules["MetaTrader5"] = _mt5_stub

from mtdata.core.trading import (
    trade_account_info,
)
from mtdata.core.trading import (
    trade_risk_analyze as _trade_risk_analyze_tool,
)
from mtdata.core.trading.requests import (
    TradeRiskAnalyzeRequest,
)

# ===================================================================
# Helpers
# ===================================================================

def trade_risk_analyze(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradeRiskAnalyzeRequest(**kwargs)
    return _trade_risk_analyze_tool(request=request, __cli_raw=raw_output)


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
#  trade_account_info (MT5 mocked)
# ===================================================================

class TestTradeAccountInfo:

    def test_compact_account_info_keeps_execution_identity(self):
        from mtdata.core.trading.account import _trade_account_payload_for_mode

        payload = {
            "success": True,
            "login": 123456,
            "account_context_id": "abc123def4567890",
            "server": "Broker-Demo",
            "company": "Broker Inc.",
            "account_type": "demo",
            "is_demo": True,
            "is_live": False,
            "trade_mode": "demo",
            "balance": 10000.0,
            "execution_ready": True,
        }

        result = _trade_account_payload_for_mode(payload, mode="compact")

        assert "login" not in result
        assert result["account_context_id"] == "abc123def4567890"
        assert result["server"] == "Broker-Demo"
        assert result["company"] == "Broker Inc."
        assert result["account_type"] == "demo"
        assert result["is_demo"] is True
        assert result["is_live"] is False
        assert result["trade_mode"] == "demo"
        assert result["execution_ready"] is True

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_account_info_success(self):
        mt5 = sys.modules["MetaTrader5"]
        mt5.account_info.return_value = SimpleNamespace(
            balance=10000, equity=10000, profit=0, margin=100,
            margin_free=9900, margin_level=10000, currency="USD",
            leverage=100, trade_allowed=True, trade_expert=True,
        )
        result = _unwrap_mcp(trade_account_info())
        assert "10000" in result or "balance" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_account_info_none(self):
        mt5 = sys.modules["MetaTrader5"]
        mt5.account_info.return_value = None
        result = _unwrap_mcp(trade_account_info())
        assert "error" in result.lower()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_margin_level_is_null_when_no_open_margin(self):
        mt5 = sys.modules["MetaTrader5"]
        mt5.account_info.return_value = SimpleNamespace(
            balance=10000, equity=10000, profit=0, margin=0,
            margin_free=10000, margin_level=0, currency="USD",
            leverage=100, trade_allowed=True, trade_expert=True,
        )
        # Call the raw wrapped function so assertions are not formatter-dependent.
        result = trade_account_info.__wrapped__()
        assert result.get("margin_level") is None
        assert "N/A" in str(result.get("margin_level_note", ""))


# ===================================================================
#  trade_risk_analyze (MT5 mocked)
# ===================================================================

class TestTradeRiskAnalyze:

    def _setup_mt5(self, mt5):
        mt5.TRADE_RETCODE_DONE = 10009

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_account_info_none(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = None
        result = _unwrap_mcp(trade_risk_analyze())
        assert "error" in result.lower()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_no_positions(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = SimpleNamespace(equity=10000, currency="USD")
        mt5.positions_get.return_value = ()
        result = _unwrap_mcp(trade_risk_analyze())
        assert "success: true" in result
        assert "positions_snapshot_unavailable" not in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_positions_with_sl(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = SimpleNamespace(equity=10000, currency="USD")
        pos = SimpleNamespace(
            ticket=1, symbol="EURUSD", type=0, volume=0.1,
            price_open=1.1, price_current=1.1, sl=1.09, tp=1.12, profit=50,
        )
        mt5.positions_get.return_value = [pos]
        mt5.symbol_info.return_value = _sym()
        result = _unwrap_mcp(trade_risk_analyze())
        assert "defined" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_positions_without_sl(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = SimpleNamespace(equity=10000, currency="USD")
        pos = SimpleNamespace(
            ticket=1, symbol="EURUSD", type=0, volume=0.1,
            price_open=1.1, price_current=1.1, sl=0, tp=0, profit=50,
        )
        mt5.positions_get.return_value = [pos]
        mt5.symbol_info.return_value = _sym()
        result = _unwrap_mcp(trade_risk_analyze())
        assert "unlimited" in result.lower()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_position_sizing(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = SimpleNamespace(equity=10000, currency="USD")
        mt5.positions_get.return_value = []
        mt5.symbol_info.return_value = _sym()
        result = _unwrap_mcp(trade_risk_analyze(
            symbol="EURUSD",
            sizing={"method": "fixed_fraction", "risk_pct": 2.0},
            entry=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
        ))
        assert "position_sizing" in result or "suggested_volume" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_position_sizing_no_symbol(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = SimpleNamespace(equity=10000, currency="USD")
        mt5.positions_get.return_value = []
        result = _unwrap_mcp(trade_risk_analyze(
            sizing={"method": "fixed_fraction", "risk_pct": 2.0},
            entry=1.1,
            stop_loss=1.09,
        ))
        assert "error" in result.lower()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_position_sizing_symbol_not_found(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = SimpleNamespace(equity=10000, currency="USD")
        mt5.positions_get.return_value = []
        mt5.symbol_info.return_value = None
        result = _unwrap_mcp(trade_risk_analyze(
            symbol="INVALID",
            sizing={"method": "fixed_fraction", "risk_pct": 2.0},
            entry=1.1, stop_loss=1.09,
        ))
        assert "error" in result.lower() or "not found" in result.lower()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_risk_level_high(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = SimpleNamespace(equity=1000, currency="USD")
        pos = SimpleNamespace(
            ticket=1, symbol="EURUSD", type=0, volume=10.0,
            price_open=1.1, price_current=1.1, sl=1.0, tp=1.2, profit=0,
        )
        mt5.positions_get.return_value = [pos]
        mt5.symbol_info.return_value = _sym()
        result = _unwrap_mcp(trade_risk_analyze())
        assert "high" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_position_sizing_zero_sl_distance(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = SimpleNamespace(equity=10000, currency="USD")
        mt5.positions_get.return_value = []
        mt5.symbol_info.return_value = _sym()
        result = _unwrap_mcp(trade_risk_analyze(
            symbol="EURUSD",
            sizing={"method": "fixed_fraction", "risk_pct": 2.0},
            entry=1.1, stop_loss=1.1,
        ))
        assert "position_sizing_error" in result or "SL distance" in result
