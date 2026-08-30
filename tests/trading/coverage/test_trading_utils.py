"""Tests for utilities and edge cases in mtdata.core.trading.

Covers:
- _tick_age_seconds and _validate_tick_freshness
- Edge cases and integration tests
"""

import importlib
import os
import sys
from datetime import datetime
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

from mtdata.core.trading.comments import (
    _normalize_trade_comment,
)
from mtdata.core.trading.time import (
    _GTC_EXPIRATION_TOKENS,
    _normalize_pending_expiration,
    _server_time_naive_to_mt5_timestamp,
)
from mtdata.core.trading.validation import (
    _DEFAULT_TICK_MAX_AGE_SECONDS,
    _SUPPORTED_ORDER_TYPES,
    _normalize_order_type_input,
    _tick_age_seconds,
    _tick_clock_reference,
    _validate_tick_freshness,
    _validate_volume,
)

# ===================================================================
# Helpers
# ===================================================================

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
#  Tick utility tests
# ===================================================================

class TestTickUtils:
    """Tests for _tick_age_seconds and _validate_tick_freshness."""

    def test_tick_age_seconds_from_time_msc(self):
        """Prefers time_msc (millisecond epoch) when available."""
        import time as _time_module
        now_ms = _time_module.time() * 1000.0
        tick = SimpleNamespace(time_msc=now_ms - 5000, time=0)
        age = _tick_age_seconds(tick)
        assert age is not None
        assert 4.5 <= age <= 6.0

    def test_tick_age_seconds_from_time_seconds(self):
        """Falls back to time (seconds) when time_msc missing."""
        import time as _time_module
        now_s = _time_module.time()
        tick = SimpleNamespace(time=int(now_s) - 10)
        age = _tick_age_seconds(tick)
        assert age is not None
        assert 9.0 <= age <= 12.0

    def test_tick_age_seconds_no_timestamp(self):
        """Returns None when tick carries no usable timestamp."""
        tick = SimpleNamespace()
        assert _tick_age_seconds(tick) is None
        tick2 = SimpleNamespace(time_msc=None, time=None)
        assert _tick_age_seconds(tick2) is None

    def test_validate_tick_freshness_fresh(self):
        """Fresh tick passes validation."""
        import time as _time_module
        now_ms = _time_module.time() * 1000.0
        tick = SimpleNamespace(time_msc=now_ms - 1000)
        result = _validate_tick_freshness(tick, symbol="EURUSD")
        assert result is None

    def test_validate_tick_freshness_stale(self):
        """Stale tick returns error dict with metadata."""
        import time as _time_module
        now_ms = _time_module.time() * 1000.0
        tick = SimpleNamespace(time_msc=now_ms - 60_000)  # 60s old
        result = _validate_tick_freshness(tick, symbol="EURUSD")
        assert result is not None
        assert "stale" in result["error"].lower()
        assert result["tick_age_seconds"] > 50
        assert result["tick_max_age_seconds"] == _DEFAULT_TICK_MAX_AGE_SECONDS

    def test_validate_tick_freshness_custom_threshold(self):
        """Custom threshold narrows acceptance window."""
        import time as _time_module
        now_ms = _time_module.time() * 1000.0
        tick = SimpleNamespace(time_msc=now_ms - 3000)  # 3s old
        assert _validate_tick_freshness(tick, symbol="GOLD", max_age_seconds=2.0) is not None
        assert _validate_tick_freshness(tick, symbol="GOLD", max_age_seconds=5.0) is None

    def test_validate_tick_freshness_no_timestamp(self):
        """Tick without timestamp fails closed because age is unknown."""
        tick = SimpleNamespace()
        result = _validate_tick_freshness(tick, symbol="EURUSD")
        assert result is not None
        assert result["tick_age_status"] == "unknown"
        assert "freshness cannot be verified" in result["error"]

    def test_validate_tick_freshness_rejects_future_timestamp(self):
        import time as _time_module

        now_ms = _time_module.time() * 1000.0
        tick = SimpleNamespace(time_msc=now_ms + 50_000)
        result = _validate_tick_freshness(tick, symbol="EURUSD")
        assert result is not None
        assert result["tick_age_status"] == "future"
        assert result["timestamp_in_future"] is True

    def test_validate_tick_freshness_reconciles_bounded_broker_clock_lead(self):
        import time as _time_module

        now_s = _time_module.time()
        tick = SimpleNamespace(time_msc=(now_s + 13.9) * 1000.0)

        clock = _tick_clock_reference(tick, wall_clock_epoch=now_s)
        assert clock["clock_reconciled"] is True
        assert clock["clock_reference"] == "broker_tick_at_acquisition"
        assert clock["local_clock_lag_seconds"] == pytest.approx(13.9, abs=0.01)
        assert _validate_tick_freshness(tick, symbol="BTCUSD") is None

    def test_validate_tick_freshness_optional_for_risk_reducing_actions(self):
        import time as _time_module

        now_ms = _time_module.time() * 1000.0
        stale = SimpleNamespace(time_msc=now_ms - 60_000)
        future = SimpleNamespace(time_msc=now_ms + 50_000)
        assert _validate_tick_freshness(stale, symbol="EURUSD", required=False) is None
        assert _validate_tick_freshness(future, symbol="EURUSD", required=False) is None
        assert _validate_tick_freshness(stale, symbol="EURUSD", required=True) is not None


# ===================================================================
#  Edge cases / integration
# ===================================================================

class TestEdgeCases:

    def test_supported_order_types_are_canonical_trade_place_names(self):
        assert _SUPPORTED_ORDER_TYPES == {
            "BUY",
            "SELL",
            "BUY_LIMIT",
            "BUY_STOP",
            "BUY_STOP_LIMIT",
            "SELL_LIMIT",
            "SELL_STOP",
            "SELL_STOP_LIMIT",
        }

    def test_gtc_tokens_all_uppercase(self):
        for token in _GTC_EXPIRATION_TOKENS:
            assert token == token.upper()

    def test_normalize_order_type_with_double_underscores(self):
        t, err = _normalize_order_type_input("BUY__LIMIT")
        assert t == "BUY_LIMIT"

    def test_normalize_order_type_with_dashes(self):
        t, err = _normalize_order_type_input("sell-limit")
        assert t == "SELL_LIMIT"

    def test_validate_volume_exactly_at_min(self):
        vol, err = _validate_volume(0.01, _sym(volume_min=0.01))
        assert err is None and vol == 0.01

    def test_validate_volume_exactly_at_max(self):
        vol, err = _validate_volume(100.0, _sym(volume_max=100.0))
        assert err is None

    def test_trade_comment_with_non_string_input(self):
        result = _normalize_trade_comment(123, default="x")
        assert result == "123"

    def test_trade_comment_preserves_zero_input(self):
        result = _normalize_trade_comment(0, default="x")
        assert result == "0"

    def test_server_time_naive_large_timestamp(self):
        dt = datetime(2099, 12, 31, 23, 59, 59)
        result = _server_time_naive_to_mt5_timestamp(dt)
        assert result > 0

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_expiration_relative_minutes_variants(self, _mock):
        for variant in ("30min", "30 mins", "30 minute", "30 minutes"):
            val, explicit = _normalize_pending_expiration(variant)
            assert isinstance(val, int) and explicit is True, f"Failed for: {variant}"

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_expiration_relative_hours_variants(self, _mock):
        for variant in ("2h", "2 hr", "2 hrs", "2 hour", "2 hours"):
            val, explicit = _normalize_pending_expiration(variant)
            assert isinstance(val, int) and explicit is True, f"Failed for: {variant}"

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_expiration_relative_seconds_variants(self, _mock):
        for variant in ("60s", "60 sec", "60 secs", "60 second", "60 seconds"):
            val, explicit = _normalize_pending_expiration(variant)
            assert isinstance(val, int) and explicit is True, f"Failed for: {variant}"

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_expiration_relative_days_variants(self, _mock):
        for variant in ("1d", "1 day", "1 days"):
            val, explicit = _normalize_pending_expiration(variant)
            assert isinstance(val, int) and explicit is True, f"Failed for: {variant}"

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_expiration_relative_weeks_variants(self, _mock):
        for variant in ("1w", "1 wk", "1 weeks"):
            val, explicit = _normalize_pending_expiration(variant)
            assert isinstance(val, int) and explicit is True, f"Failed for: {variant}"

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_expiration_in_prefix(self, _mock):
        val, explicit = _normalize_pending_expiration("in 5 minutes")
        assert isinstance(val, int) and explicit is True
