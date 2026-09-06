from contextlib import contextmanager
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from mtdata.core.regime.api import _fetch_history as regime_history
from mtdata.forecast.common import fetch_history as forecast_history
from mtdata.forecast.volatility import fetch_history_frame as volatility_history
from mtdata.services.data_service import candles, query
from mtdata.utils.time import bar_close_epoch


@pytest.mark.parametrize("fetch", [forecast_history, volatility_history, regime_history])
@pytest.mark.parametrize("timeframe,opens,date", [
    ("W1", ["2026-08-15T21:00Z", "2026-08-22T21:00Z", "2026-08-29T21:00Z"], "2026-08-24"),
    ("MN1", ["2026-06-30T21:00Z", "2026-07-31T21:00Z", "2026-08-31T21:00Z"], "2026-08-01"),
])
@pytest.mark.parametrize("bound", ["as_of", "end"])
def test_analytical_dates_exclude_finalized_prices_after_cutoff(monkeypatch, fetch, timeframe, opens, date, bound):
    zone = ZoneInfo("Europe/Nicosia")
    monkeypatch.setattr(query, "_broker_calendar_timezone", lambda *a: zone)
    monkeypatch.setattr("mtdata.utils.time._broker_calendar_timezone", lambda *a: zone)
    rates = [{"time": pd.Timestamp(value).timestamp(), "open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100 + i} for i, value in enumerate(opens)]
    captured = []

    @contextmanager
    def ready(*a, **kw):
        yield None, None

    def provider(*args, **kw):
        captured.append(args[6])
        return rates, None

    monkeypatch.setattr(candles, "_symbol_ready_guard", ready)
    monkeypatch.setattr(candles, "get_symbol_info_cached", lambda *a: None)
    monkeypatch.setattr(candles, "resolve_broker_symbol_name", lambda value: value)
    monkeypatch.setattr(candles, "_fetch_rates_with_warmup", provider)
    cutoff = query._parse_fetch_datetime_arg(date, timeframe=timeframe, end_bound=True)[0]
    result = fetch("EURUSD", timeframe, 3, **{bound: date})
    explicit = fetch("EURUSD", timeframe, 3, **{bound: cutoff.isoformat()})
    pd.testing.assert_frame_equal(result, explicit)
    assert result.close.tolist() == [100.0]
    assert bar_close_epoch(result.time.iloc[-1], timeframe) <= cutoff.timestamp()
    assert all("T" in value for value in captured)
