from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mtdata.analytics.engine_common import _rates
from mtdata.analytics.relative_strength import rank_relative_strength
from mtdata.core.analytics_requests import MarketRelativeStrengthRequest
from mtdata.utils.time import bar_close_epoch


class HistoricalGateway:
    def __init__(self):
        self.cutoff = datetime(2025, 1, 15, 12, tzinfo=timezone.utc).timestamp()
        self.frames = {}
        for idx, symbol in enumerate(["EURUSD", "GBPUSD", "USDJPY"]):
            close = 100 * np.exp(np.cumsum(np.random.default_rng(idx).normal(.0001 * idx, .002, 240)))
            self.frames[symbol] = pd.DataFrame({"time": self.cutoff + (np.arange(240) - 220) * 3600, "close": close, "tick_volume": 100})
        self.symbol_info_tick = Mock(side_effect=AssertionError("Historical ranking must not query live quotes"))
        self.copy_rates_from_pos = Mock(side_effect=AssertionError("Historical ranking must not fetch latest history"))

    def symbols_get(self):
        return [SimpleNamespace(name=name, visible=True, path="Forex") for name in self.frames]

    def copy_rates_from(self, symbol, timeframe, end, count):
        frame = self.frames[symbol]
        return frame[frame.time <= end.timestamp()].tail(count)


@pytest.mark.parametrize("cutoff", ["2025-01-15T12:00:00Z", "2025-01-15T12:30:00Z"])
def test_rankings_are_replayable_and_ignore_future_prices_and_live_quotes(cutoff):
    gateway = HistoricalGateway()
    request = MarketRelativeStrengthRequest(symbols="EURUSD,GBPUSD,USDJPY", as_of=cutoff, detail="full")
    before = rank_relative_strength(request, gateway)
    assert before["success"] is True
    assert before["status"] == "ranked"
    assert before["history_mode"] == "historical"
    assert before["quote_policy"] == "not_queried_historical"
    assert before["analysis_as_of"] == cutoff
    for frame in gateway.frames.values():
        frame.loc[frame.time >= gateway.cutoff, "close"] *= 10
    after = rank_relative_strength(request, gateway)
    assert before["rankings"] == after["rankings"]
    assert before["breadth"] == after["breadth"]
    assert all(row["spread_pct"] is None for row in before["rankings"])
    gateway.symbol_info_tick.assert_not_called()
    gateway.copy_rates_from_pos.assert_not_called()
    rates = _rates(gateway, "EURUSD", "H1", 100, end=cutoff)
    assert len(rates) == 100
    assert bar_close_epoch(float(rates.time.iloc[-1]), "H1") == gateway.cutoff


@pytest.mark.parametrize("kwargs,match", [
    ({"as_of": "invalid", "symbols": "EURUSD,GBPUSD"}, "valid date"),
    ({"as_of": "2999-01-01", "symbols": "EURUSD,GBPUSD"}, "future"),
    ({"as_of": "2025-01-01"}, "explicit symbols"),
    ({"as_of": "2025-01-01", "symbols": "EURUSD,GBPUSD", "max_spread_pct": .1}, "current quotes"),
])
def test_historical_controls_reject_unreproducible_filters(kwargs, match):
    with pytest.raises((ValueError, ValidationError), match=match):
        MarketRelativeStrengthRequest(**kwargs)


def test_cutoff_normalizes_utc_offset():
    request = MarketRelativeStrengthRequest(symbols="EURUSD,GBPUSD", as_of="2025-01-15T07:00:00-05:00")
    assert request.as_of == "2025-01-15T12:00:00Z"
