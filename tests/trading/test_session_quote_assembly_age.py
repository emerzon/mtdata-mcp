from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from mtdata.core.trading import context
from mtdata.core.trading.requests import TradeSessionContextRequest


@pytest.mark.parametrize("detail", ["compact", "full"])
def test_delayed_positions_age_quote_before_session_readiness(monkeypatch, detail):
    now = [1788512400.0]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(now[0], tz=timezone.utc)

    quote = {"success": True, "time": "2026-09-04T09:00:00Z", "bid": 100, "ask": 101,
             "data_age_seconds": 0, "freshness_state": "live", "usable_for_live_trading": True}
    quote["time"] = datetime.fromtimestamp(now[0], timezone.utc).isoformat()

    def positions(**kwargs):
        now[0] += 10
        return {"success": True, "count": 0, "items": []}

    monkeypatch.setattr(context, "datetime", Clock)
    monkeypatch.setattr(context, "create_trading_gateway", lambda: SimpleNamespace())
    monkeypatch.setattr(context, "resolve_trading_symbol_request", lambda request, gateway: (request, None))
    monkeypatch.setattr(context, "market_ticker", lambda **kwargs: quote)
    monkeypatch.setattr(context, "trade_get_open", positions)
    monkeypatch.setattr(context, "trade_get_pending", lambda **kwargs: {"success": True, "count": 0, "items": []})
    monkeypatch.setattr(context, "_trade_session_tradability", lambda symbol: {"now_tradable": True})
    result = context.trade_session_context.__wrapped__(TradeSessionContextRequest(symbol="BTCUSD", detail=detail, include_account=False))
    assert result["quote"]["data_age_seconds"] == 20
    assert result["quote"]["usable_for_live_trading"] is False
    assert result["quote"]["data_age_as_of"] == result["assembled_at"]
    assert result["snapshot_span_seconds"] == 20
    assert result["quote_quality"]["freshness_is_live"] is False
    assert quote["data_age_seconds"] == 0


def test_assembly_cannot_clear_existing_quote_veto():
    now = datetime(2026, 9, 4, 9, tzinfo=timezone.utc)
    result = context._age_session_quote({"time": now.isoformat(), "usable_for_live_trading": False}, symbol="BTCUSD", observed_at=now, assembled_at=now)
    assert result["data_age_seconds"] == 0
    assert result["usable_for_live_trading"] is False
