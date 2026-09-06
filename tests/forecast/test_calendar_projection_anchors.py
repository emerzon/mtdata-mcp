from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from mtdata.forecast.common import (
    describe_forecast_calendar_treatment,
    next_times_from_last,
)
from mtdata.shared.constants import TIMEFRAME_SECONDS


@pytest.mark.parametrize("timeframe,base,expected", [
    ("W1", "2026-08-15T21:00Z", ["2026-08-22T21:00Z", "2026-08-29T21:00Z"]),
    ("W1", "2026-10-17T21:00Z", ["2026-10-24T21:00Z", "2026-10-31T22:00Z"]),
    ("MN1", "2026-07-31T21:00Z", ["2026-08-31T21:00Z", "2026-09-30T21:00Z"]),
    ("MN1", "2025-01-31T22:00Z", ["2025-02-28T22:00Z", "2025-03-31T21:00Z"]),
])
@pytest.mark.parametrize("symbol", ["EURUSD", "AAPL", "BTCUSD"])
def test_calendar_period_keys_preserve_broker_anchor(monkeypatch, timeframe, base, expected, symbol):
    monkeypatch.setattr("mtdata.utils.time._broker_calendar_timezone", lambda *a: ZoneInfo("Europe/Nicosia"))
    result = next_times_from_last(pd.Timestamp(base).timestamp(), TIMEFRAME_SECONDS[timeframe], 2,
                                  timeframe=timeframe, skip_weekends=True, symbol=symbol)
    assert result == [pd.Timestamp(value).timestamp() for value in expected]
    assert describe_forecast_calendar_treatment(symbol, TIMEFRAME_SECONDS[timeframe], calendar_timeframe=True) == "broker_calendar_period_boundaries"
