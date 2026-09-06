from types import SimpleNamespace

import pytest

from mtdata.core.trading import common, validation
from mtdata.utils import quote


@pytest.mark.parametrize("age", [43.65, -60, None, 0, 15, 30])
def test_reconciled_quote_readiness_includes_raw_submission_policy(monkeypatch, age):
    now = 1788512400.0
    raw = SimpleNamespace(time_msc=(now + age) * 1000 if age is not None else 0, bid=100., ask=101.)
    streamed = SimpleNamespace(time_msc=now * 1000, bid=100., ask=101.)
    monkeypatch.setattr(quote, "_latest_stream_ticks", lambda *args, **kwargs: (streamed, streamed))
    monkeypatch.setattr(quote, "_symbol_point", lambda *args: .01)
    monkeypatch.setattr(validation._time_module, "time", lambda: now)
    selected, metadata = quote.resolve_quote_tick(SimpleNamespace(), "BTCUSD", raw, now_epoch=now)
    error = validation._validate_tick_freshness(raw, symbol="BTCUSD")
    assert metadata["send_path_tick_fresh"] is (error is None)
    context = common.build_trade_quote_context("BTCUSD", selected, now_epoch=now, source_metadata=metadata)
    if error:
        assert context["usable_for_live_trading"] is False
        assert context["usable_for_live_trading_basis"] == "submission_tick_freshness_required"
        assert error["error"] in context["warning"]
    else:
        assert context["usable_for_live_trading"] is True
