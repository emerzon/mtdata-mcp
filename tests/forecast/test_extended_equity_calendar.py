import numpy as np
import pandas as pd
import pytest

from mtdata.forecast import common as fc


def _extended_observations(start="2026-08-23", end="2026-09-04", friday_close=17):
    observed = []
    for day in pd.date_range(start, end):
        hours = (
            range(17, 24) if day.weekday() == 6 else
            range(friday_close) if day.weekday() == 4 else
            range(0) if day.weekday() == 5 else range(24)
        )
        for hour in hours:
            observed.append(pd.Timestamp(
                f"{day.date()} {hour:02d}:00", tz="America/New_York"
            ).timestamp())
    return observed


@pytest.mark.parametrize("symbol", ["AAPL.NAS-24", "AAPL.NAS"])
@pytest.mark.parametrize("friday_close", [14, 17])
def test_extended_weekday_slots_keep_friday_closure(symbol, friday_close):
    observed = _extended_observations(friday_close=friday_close)
    forecast = fc.next_times_from_last(
        observed[-1], 3600, 10, timeframe="H1", symbol=symbol, observed_times=observed,
    )
    # Sunday reopening is observed; Labor Day Monday remains a broker-session
    # estimate rather than being suppressed by the cash exchange's holiday.
    expected = pd.date_range("2026-09-06T21:00Z", periods=10, freq="h")
    assert forecast == [value.timestamp() for value in expected]
    assert not any(fc.is_standard_weekend_closed_epoch(epoch) for epoch in forecast)
    assert fc.describe_forecast_calendar_treatment(
        symbol, 3600, calendar_timeframe=False, observed_times=observed,
    ) == "broker_observed_weekday_slots_standard_weekend_holidays_unknown"


def test_partial_current_friday_does_not_infer_premature_weekly_close():
    observed = _extended_observations()[:-8]
    forecast = fc.next_times_from_last(
        observed[-1], 3600, 3, timeframe="H1", symbol="AAPL.NAS-24", observed_times=observed,
    )
    assert forecast == [observed[-1] + step * 3600 for step in (1, 2, 3)]


def test_short_extended_history_uses_weekend_policy_without_exchange_holiday_claim():
    observed = _extended_observations(start="2026-09-04", end="2026-09-04")
    forecast = fc.next_times_from_last(
        observed[-1], 3600, 2, timeframe="H1", symbol="AAPL.NAS-24",
        observed_times=observed, skip_weekends=True,
    )
    assert forecast == [pd.Timestamp(value).timestamp() for value in ("2026-09-06T21:00Z", "2026-09-06T22:00Z")]
    assert not fc.uses_exchange_intraday_projection("AAPL.NAS-24", 3600, observed_times=observed)


def test_extended_weekend_boundary_follows_new_york_dst():
    observed = _extended_observations(start="2026-10-18", end="2026-10-30")
    forecast = fc.next_times_from_last(
        observed[-1], 3600, 2, timeframe="H1", symbol="AAPL.NAS-24", observed_times=observed,
    )
    assert forecast == [pd.Timestamp(value).timestamp() for value in ("2026-11-01T22:00Z", "2026-11-01T23:00Z")]


def test_extended_projection_handles_invalid_history_values():
    observed = [None, float("nan"), True, "invalid", *_extended_observations()]
    forecast = fc.next_times_from_last(
        observed[-1], 3600, 2, timeframe="H1", symbol="AAPL.NAS-24", observed_times=observed,
    )
    assert len(forecast) == 2
    assert np.isfinite(forecast).all()
