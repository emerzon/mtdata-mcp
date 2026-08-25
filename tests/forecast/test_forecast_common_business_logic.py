from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast import common as fc


def test_parse_as_of_bound_includes_date_only_day_and_preserves_timestamp():
    assert fc._parse_as_of_bound("2026-08-13", timeframe="H1") == datetime(
        2026,
        8,
        13,
        23,
        59,
        59,
        999999,
    )
    assert fc._parse_as_of_bound(
        "2026-08-13T10:15:00Z",
        timeframe="H1",
    ) == datetime(2026, 8, 13, 10, 15)


def test_parse_as_of_bound_uses_broker_calendar_for_daily_timeframe(monkeypatch):
    expected = datetime(2026, 8, 13, 20, 59, 59)
    monkeypatch.setattr(
        fc,
        "_parse_candle_calendar_bound",
        lambda value, *, timeframe, end_bound: expected,
    )

    assert fc._parse_as_of_bound("2026-08-13", timeframe="D1") == expected


def test_describe_forecast_calendar_treatment_labels_fx_crypto_and_unknown():
    hour = 3600
    day = 86400

    assert fc.describe_forecast_calendar_treatment(
        "EURUSD", hour, calendar_timeframe=False
    ) == "forex_weekend_skipped"
    assert fc.describe_forecast_calendar_treatment(
        "EURUSD", day, calendar_timeframe=True
    ) == "broker_calendar_boundaries_and_forex_weekend_skipped"
    assert fc.describe_forecast_calendar_treatment(
        "BTCUSD", hour, calendar_timeframe=False
    ) == "continuous_no_weekend_skip"
    assert fc.describe_forecast_calendar_treatment(
        "BTCUSD", day, calendar_timeframe=True
    ) == "broker_calendar_boundaries_continuous_crypto"
    assert fc.describe_forecast_calendar_treatment(
        "US500", day, calendar_timeframe=True
    ) == "broker_calendar_boundaries_and_weekend_skipped_holidays_unknown"
    assert fc.describe_forecast_calendar_treatment(
        "US500", hour, calendar_timeframe=False
    ) == "standard_weekend_skipped_session_hours_unknown"

    assert fc.describe_forecast_calendar_treatment(
        "AAPL.NAS", day, calendar_timeframe=True
    ) == "broker_calendar_boundaries_and_xnys_holidays_skipped"
    assert fc.describe_forecast_calendar_treatment(
        "AAPL.NAS", hour, calendar_timeframe=False
    ) == "xnys_exchange_regular_session_fallback_holidays_and_early_closes_applied"


def _xnys_hourly_observations(*session_dates: str) -> list[float]:
    return [
        pd.Timestamp(f"{session_date} {hour:02d}:00", tz="America/New_York").timestamp()
        for session_date in session_dates
        for hour in range(9, 16)
    ]


def test_equity_intraday_projection_uses_observed_broker_session_slots() -> None:
    observed = _xnys_hourly_observations("2026-08-17", "2026-08-18", "2026-08-19")

    result = fc.next_times_from_last(
        observed[-1],
        3600,
        3,
        skip_weekends=True,
        timeframe="H1",
        symbol="TSLA.NAS",
        observed_times=observed,
    )

    assert [pd.Timestamp(value, unit="s", tz="UTC").isoformat() for value in result] == [
        "2026-08-20T13:00:00+00:00",
        "2026-08-20T14:00:00+00:00",
        "2026-08-20T15:00:00+00:00",
    ]
    assert fc.describe_forecast_calendar_treatment(
        "TSLA.NAS",
        3600,
        calendar_timeframe=False,
        observed_times=observed,
    ) == "xnys_observed_broker_slots_holidays_and_early_closes_applied"


def test_equity_intraday_projection_applies_holidays_and_early_closes() -> None:
    july_observed = _xnys_hourly_observations("2026-06-30", "2026-07-01", "2026-07-02")
    after_independence_day = fc.next_times_from_last(
        july_observed[-1],
        3600,
        1,
        timeframe="H1",
        symbol="AAPL.NAS",
        observed_times=july_observed,
    )
    assert pd.Timestamp(after_independence_day[0], unit="s", tz="UTC").isoformat() == (
        "2026-07-06T13:00:00+00:00"
    )

    november_observed = _xnys_hourly_observations(
        "2026-11-23",
        "2026-11-24",
        "2026-11-25",
    )
    after_thanksgiving = fc.next_times_from_last(
        november_observed[-1],
        3600,
        5,
        timeframe="H1",
        symbol="AAPL.NAS",
        observed_times=november_observed,
    )
    assert [
        pd.Timestamp(value, unit="s", tz="America/New_York").strftime("%Y-%m-%d %H:%M")
        for value in after_thanksgiving
    ] == [
        "2026-11-27 09:00",
        "2026-11-27 10:00",
        "2026-11-27 11:00",
        "2026-11-27 12:00",
        "2026-11-30 09:00",
    ]


def test_equity_intraday_projection_converts_dst_in_exchange_timezone() -> None:
    observed = _xnys_hourly_observations("2026-03-04", "2026-03-05", "2026-03-06")

    result = fc.next_times_from_last(
        observed[-1],
        3600,
        1,
        timeframe="H1",
        symbol="AAPL.NAS",
        observed_times=observed,
    )

    assert pd.Timestamp(result[0], unit="s", tz="UTC").isoformat() == (
        "2026-03-09T13:00:00+00:00"
    )


def test_future_as_of_is_rejected_against_wall_clock():
    now = datetime(2026, 8, 12, 14, 0).timestamp()

    assert fc.future_as_of_error("2030-01-01T00:00:00Z", now_epoch=now) == (
        "as_of must not be in the future."
    )
    assert fc.future_as_of_error("2024-01-01T00:00:00Z", now_epoch=now) is None


def test_extract_forecast_values_requires_exact_horizon():
    yf_standard = pd.DataFrame({"pred": [1.0, 2.0]})
    out = fc._extract_forecast_values(yf_standard, fh=2, method_name="m")
    assert out.tolist() == [1.0, 2.0]

    yf_alt = pd.DataFrame({"unique_id": ["ts"], "ds": [0], "pred": [9.0]})
    with pytest.raises(ValueError, match="requested 3, received 1"):
        fc._extract_forecast_values(yf_alt, fh=3, method_name="m")

    yf_with_actuals = pd.DataFrame(
        {
            "unique_id": ["ts"] * 2,
            "ds": [0, 1],
            "y": [1.0, 2.0],
            "pred": [9.0, 10.0],
        }
    )
    out = fc._extract_forecast_values(yf_with_actuals, fh=2, method_name="m")
    assert out.tolist() == [9.0, 10.0]

    with pytest.raises(RuntimeError, match="refusing to use actuals column 'y'"):
        fc._extract_forecast_values(pd.DataFrame({"y": [1.0, 2.0]}), fh=2, method_name="m")

    yf_with_auxiliary = pd.DataFrame(
        {
            "unique_id": ["ts"] * 2,
            "ds": [0, 1],
            "cutoff": ["2026-04-09T00:00:00Z"] * 2,
            "NHITS": [7.5, 8.0],
        }
    )
    out = fc._extract_forecast_values(yf_with_auxiliary, fh=2, method_name="nhits")
    assert out.tolist() == [7.5, 8.0]

    with pytest.raises(RuntimeError, match="prediction columns not found"):
        fc._extract_forecast_values(pd.DataFrame({"unique_id": ["ts"], "ds": [0]}), fh=1, method_name="demo")


def test_create_training_dataframes_with_and_without_exog():
    series = np.array([1.0, 2.0, 3.0], dtype=float)
    exog = np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]], dtype=float)
    exog_future = np.array([[13.0, 23.0], [14.0, 24.0]], dtype=float)

    y_df, x_df, xf_df = fc._create_training_dataframes(series, fh=2, exog_used=exog, exog_future=exog_future)
    assert list(y_df.columns) == ["unique_id", "ds", "y"]
    assert y_df["y"].tolist() == [1.0, 2.0, 3.0]
    assert x_df is not None and list(x_df.columns) == ["unique_id", "ds", "x0", "x1"]
    assert xf_df is not None and list(xf_df.columns) == ["unique_id", "ds", "x0", "x1"]
    assert xf_df["x0"].tolist() == [13.0, 14.0]

    y_df, x_df, xf_df = fc._create_training_dataframes(series, fh=2, exog_used=None, exog_future=None)
    assert x_df is None
    assert xf_df is None
    assert len(y_df) == 3


def test_create_training_dataframes_accepts_1d_single_column_exog():
    series = np.array([1.0, 2.0, 3.0], dtype=float)
    exog = np.array([10.0, 11.0, 12.0], dtype=float)
    exog_future = np.array([13.0, 14.0], dtype=float)

    _, x_df, xf_df = fc._create_training_dataframes(
        series,
        fh=2,
        exog_used=exog,
        exog_future=exog_future,
    )

    assert x_df is not None
    assert list(x_df.columns) == ["unique_id", "ds", "x0"]
    assert x_df["x0"].tolist() == [10.0, 11.0, 12.0]
    assert xf_df is not None
    assert list(xf_df.columns) == ["unique_id", "ds", "x0"]
    assert xf_df["x0"].tolist() == [13.0, 14.0]


def test_timeframe_helpers_cover_key_branches():
    assert fc.default_seasonality("H1") == 24
    assert fc.default_seasonality("D1") == 5
    assert fc.default_seasonality("W1") == 52
    assert fc.default_seasonality("MN1") == 12
    assert fc.default_seasonality("NOPE") == 0


def test_default_seasonality_uses_observed_session_bar_count() -> None:
    sessions = pd.DatetimeIndex(
        [
            *pd.date_range("2026-01-05 14:30", periods=7, freq="h", tz="UTC"),
            *pd.date_range("2026-01-06 14:30", periods=7, freq="h", tz="UTC"),
            *pd.date_range("2026-01-07 14:30", periods=7, freq="h", tz="UTC"),
        ]
    )

    assert fc.default_seasonality("H1", sessions) == 7

    assert fc.next_times_from_last(100.0, 60, 3) == [160.0, 220.0, 280.0]
    assert fc.pd_freq_from_timeframe("H4") == "4h"
    assert fc.pd_freq_from_timeframe("x") == "D"


def test_resolve_forecast_symbol_maps_slash_alias(monkeypatch):
    class _Info:
        def __init__(self, name):
            self.name = name

    class _Gateway:
        def symbols_get(self):
            return [_Info("EURUSD"), _Info("GBPUSD")]

    from mtdata.utils import mt5 as mt5_mod

    monkeypatch.setattr(mt5_mod, "mt5", _Gateway())

    canonical, requested = fc.resolve_forecast_symbol("EUR/USD")
    assert canonical == "EURUSD"
    assert requested == "EUR/USD"
    unchanged, alias = fc.resolve_forecast_symbol("EURUSD")
    assert unchanged == "EURUSD"
    assert alias is None


def test_fetch_history_delegates_to_canonical_gateway(monkeypatch):
    expected = pd.DataFrame({"time": [100.0], "close": [1.2]})
    gateway = MagicMock(return_value=expected)
    monkeypatch.setattr(fc, "fetch_history_frame", gateway)

    actual = fc.fetch_history(
        "EURUSD",
        "H1",
        need=25,
        as_of="2026-08-20T12:00:00Z",
        drop_last_live=False,
    )

    assert actual is expected
    gateway.assert_called_once_with(
        "EURUSD",
        "H1",
        25,
        "2026-08-20T12:00:00Z",
        start=None,
        end=None,
        include_incomplete=True,
    )


def test_fetch_history_delegates_ranges_and_closed_bar_policy(monkeypatch):
    expected = pd.DataFrame({"time": [100.0, 200.0]})
    gateway = MagicMock(return_value=expected)
    monkeypatch.setattr(fc, "fetch_history_frame", gateway)

    actual = fc.fetch_history(
        "EURUSD",
        "H4",
        need=50,
        start="2026-08-01",
        end="2026-08-10",
    )

    assert actual is expected
    gateway.assert_called_once_with(
        "EURUSD",
        "H4",
        50,
        None,
        start="2026-08-01",
        end="2026-08-10",
        include_incomplete=False,
    )


def test_fetch_history_preserves_gateway_errors(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("Invalid as_of time.")

    monkeypatch.setattr(fc, "fetch_history_frame", fail)

    with pytest.raises(RuntimeError, match="Invalid as_of time"):
        fc.fetch_history("EURUSD", "H1", need=2, as_of="bad")
