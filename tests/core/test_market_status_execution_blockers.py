"""Compact status distinguishes displayed quotes from blocked submission ticks."""

from datetime import datetime, timezone
from inspect import unwrap
from types import SimpleNamespace

import pytest

from mtdata.core import market_status


@pytest.mark.parametrize("detail", ["compact", "full"])
@pytest.mark.parametrize("future_cached_quote", [False, True])
def test_symbol_status_preserves_submission_quote_blocker(monkeypatch, detail, future_cached_quote):
    now = datetime(2026, 7, 30, 12, tzinfo=timezone.utc)
    epoch = now.timestamp()

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.replace(tzinfo=None) if tz is None else now.astimezone(tz)

    gateway = SimpleNamespace(
        COPY_TICKS_ALL=0,
        TIMEFRAME_M1=1,
        SYMBOL_TRADE_MODE_FULL=4,
        SYMBOL_TRADE_MODE_DISABLED=0,
        SYMBOL_TRADE_MODE_CLOSEONLY=3,
        SYMBOL_TRADE_MODE_LONGONLY=1,
        SYMBOL_TRADE_MODE_SHORTONLY=2,
        ensure_connection=lambda: None,
        symbol_info=lambda symbol: SimpleNamespace(name=symbol, visible=True, trade_mode=4),
        symbol_info_tick=lambda _symbol: SimpleNamespace(
            time=epoch + 45 if future_cached_quote else epoch - 1,
            bid=100.1, ask=100.2,
        ),
        copy_ticks_range=lambda *_args: [{"time": epoch - 1, "bid": 100.1, "ask": 100.2}],
        copy_rates_range=lambda *_args: [],
    )
    monkeypatch.setattr(market_status, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status, "create_mt5_gateway", lambda **kwargs: gateway)
    result = unwrap(market_status.market_status)(symbol="BTCUSD", detail=detail)
    quote = result["tick"] if detail == "full" else result
    assert quote["tick_freshness"] == "live"
    assert quote["send_path_tick_fresh"] is (not future_cached_quote)
    assert quote["usable_for_live_trading"] is (not future_cached_quote)
    if future_cached_quote:
        assert result["reason"] == "submission_tick_not_fresh"
        assert "ahead of the wall clock" in quote["send_path_freshness_error"]
        assert quote["send_path_freshness_error"] in quote["warning"]
        assert quote["quote_source"] == "mt5.copy_ticks_range"
        assert quote["symbol_info_tick_time_epoch"] == epoch + 45
        assert quote["usable_for_live_trading_basis"] == "submission_tick_freshness_required"
    else:
        assert "send_path_freshness_error" not in quote
