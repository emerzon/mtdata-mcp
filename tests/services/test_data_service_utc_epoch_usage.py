from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd

from mtdata.services import data_service
from mtdata.services.data_service import query as data_service_query


def test_trim_df_to_target_uses_utc_epoch_seconds() -> None:
    df = pd.DataFrame({"__epoch": [100.0, 200.0, 250.0, 300.0], "close": [1.0, 2.0, 2.5, 3.0]})
    with patch("mtdata.services.data_service.candles._parse_start_datetime") as mock_parse, patch(
        "mtdata.services.data_service.candles._utc_epoch_seconds"
    ) as mock_epoch:
        mock_parse.side_effect = [datetime(2025, 1, 1, 0, 0), datetime(2025, 1, 1, 1, 0)]
        mock_epoch.side_effect = [150.0, 250.0]
        out = data_service.candles._trim_df_to_target(df, "2025-01-01 00:00", "2025-01-01 01:00", candles=100)

    assert mock_epoch.call_count == 2
    # End bound is inclusive: epochs in [150, 250] are kept.
    assert out["__epoch"].tolist() == [200.0, 250.0]


def test_trim_df_to_target_includes_entire_date_only_end() -> None:
    epochs = [
        datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp(),
        datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp(),
        datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc).timestamp(),
    ]
    df = pd.DataFrame({"__epoch": epochs, "close": [1.0, 2.0, 3.0]})

    out = data_service.candles._trim_df_to_target(
        df,
        "2025-01-01",
        "2025-01-01",
        candles=100,
    )

    assert out["close"].tolist() == [1.0, 2.0]


def test_trim_df_to_target_uses_bar_close_time_for_historical_end() -> None:
    df = pd.DataFrame(
        {
            "__epoch": [
                pd.Timestamp("2025-01-01 11:00", tz="UTC").timestamp(),
                pd.Timestamp("2025-01-01 12:00", tz="UTC").timestamp(),
            ]
        }
    )

    out = data_service.candles._trim_df_to_target(
        df,
        None,
        "2025-01-01 12:30",
        candles=100,
        timeframe="H1",
    )

    assert out["__epoch"].tolist() == [
        pd.Timestamp("2025-01-01 11:00", tz="UTC").timestamp(),
    ]


def test_trim_df_to_target_excludes_bar_opening_at_equal_bounds() -> None:
    epoch = pd.Timestamp("2025-01-01 12:00", tz="UTC").timestamp()
    df = pd.DataFrame({"__epoch": [epoch], "close": [1.1]})

    out = data_service.candles._trim_df_to_target(
        df,
        "2025-01-01 12:00",
        "2025-01-01 12:00",
        candles=100,
        timeframe="H1",
    )

    assert out.empty


def test_trim_df_to_target_includes_bar_closing_at_end_bound() -> None:
    epoch = pd.Timestamp("2025-01-01 11:00", tz="UTC").timestamp()
    df = pd.DataFrame({"__epoch": [epoch], "close": [1.1]})

    out = data_service.candles._trim_df_to_target(
        df,
        "2025-01-01 11:00",
        "2025-01-01 12:00",
        candles=100,
        timeframe="H1",
    )

    assert out["__epoch"].tolist() == [epoch]


def test_trim_daily_date_only_range_uses_broker_session_date() -> None:
    df = pd.DataFrame(
        {
            "__epoch": [
                pd.Timestamp("2026-08-09 21:00", tz="UTC").timestamp(),
                pd.Timestamp("2026-08-10 21:00", tz="UTC").timestamp(),
                pd.Timestamp("2026-08-11 21:00", tz="UTC").timestamp(),
            ],
            "close": [1.0, 2.0, 3.0],
        }
    )

    with patch.object(
        data_service.candles.mt5_config,
        "get_server_tz",
        return_value=ZoneInfo("Europe/Nicosia"),
    ):
        out = data_service.candles._trim_df_to_target(
            df,
            "2026-08-11",
            "2026-08-11",
            candles=100,
            timeframe="D1",
        )

    assert out["close"].tolist() == [2.0]


def test_calendar_timeframe_instant_end_uses_bar_close() -> None:
    df = pd.DataFrame(
        {
            "__epoch": [
                pd.Timestamp("2026-08-19 21:00", tz="UTC").timestamp(),
                pd.Timestamp("2026-08-20 21:00", tz="UTC").timestamp(),
            ],
            "close": [1.0, 2.0],
        }
    )

    with patch.object(
        data_service.candles.mt5_config,
        "get_server_tz",
        return_value=ZoneInfo("Europe/Nicosia"),
    ):
        out = data_service.candles._trim_df_to_target(
            df,
            "2026-08-20",
            "2026-08-21T00:00:00Z",
            candles=100,
            timeframe="D1",
        )

    assert out["close"].tolist() == [1.0]


def test_daily_date_only_provider_range_starts_at_broker_session_open() -> None:
    broker_tz = ZoneInfo("Europe/Nicosia")
    captured: dict[str, datetime] = {}
    expected_open = datetime(2026, 8, 12, 21, tzinfo=timezone.utc)

    def copy_rates(_symbol, _timeframe, start, end):
        captured.update(start=start, end=end)
        return [{"time": expected_open.timestamp()}]

    with (
        patch.object(data_service.candles.mt5_config, "get_server_tz", return_value=broker_tz),
        patch.object(data_service.candles, "_mt5_copy_rates_range", side_effect=copy_rates),
    ):
        rates, error = data_service.candles._fetch_rates_with_warmup(
            symbol="EURUSD",
            mt5_timeframe=1,
            timeframe="D1",
            candles=10,
            warmup_bars=0,
            start_datetime="2026-08-13",
            end_datetime="2026-08-13",
            include_incomplete=True,
            retry=False,
            sanity_check=False,
        )

    assert error is None
    assert rates == [{"time": expected_open.timestamp()}]
    assert captured["start"] == expected_open
    assert captured["end"] == datetime(
        2026, 8, 13, 20, 59, 59, 999999, tzinfo=timezone.utc
    )


def test_higher_timeframe_query_metadata_echoes_broker_session_bounds() -> None:
    with patch.object(
        data_service.candles.mt5_config,
        "get_server_tz",
        return_value=ZoneInfo("Europe/Nicosia"),
    ):
        query = data_service.candles._candle_query_applied(
            timeframe="D1",
            start="2026-08-13",
            end="2026-08-13",
            limit=10,
        )

    assert query["bound_basis"] == "broker_session_calendar"
    assert query["resolved_start"] == "2026-08-12T21:00:00Z"
    assert query["resolved_end"] == "2026-08-13T20:59:59.999999Z"


def test_daily_date_bounds_localize_named_zone_without_lmt_shift() -> None:
    with patch.object(
        data_service.candles.mt5_config,
        "get_server_tz",
        return_value=ZoneInfo("Europe/Nicosia"),
    ):
        query = data_service.candles._candle_query_applied(
            timeframe="D1",
            start="2026-08-10",
            end="2026-08-13",
            limit=10,
        )

    assert query["resolved_start"] == "2026-08-09T21:00:00Z"
    assert query["resolved_end"] == "2026-08-13T20:59:59.999999Z"


def test_daily_date_only_bounds_fail_without_broker_timezone() -> None:
    config = data_service.candles.mt5_config
    original_offset = config.time_offset_minutes
    with patch.object(config, "get_server_tz", return_value=None):
        config.time_offset_minutes = 0
        try:
            parsed, error = data_service.candles._parse_fetch_datetime_arg(
                "2026-08-13",
                timeframe="D1",
            )
        finally:
            config.time_offset_minutes = original_offset

    assert parsed is None
    assert error is not None
    assert "MT5_SERVER_TZ" in error
    assert "MT5_TIME_OFFSET_MINUTES" in error


def test_daily_date_only_bounds_succeed_with_static_offset() -> None:
    config = data_service.candles.mt5_config
    original_offset = config.time_offset_minutes
    with patch.object(config, "get_server_tz", return_value=None):
        config.time_offset_minutes = 180
        try:
            parsed, error = data_service.candles._parse_fetch_datetime_arg(
                "2026-08-13",
                timeframe="D1",
            )
        finally:
            config.time_offset_minutes = original_offset

    assert error is None
    assert parsed == datetime(2026, 8, 12, 21, tzinfo=timezone.utc)


def test_daily_date_bounds_prefer_static_offset_over_named_zone() -> None:
    config = data_service.candles.mt5_config
    original_offset = config.time_offset_minutes
    with patch.object(config, "get_server_tz", return_value=ZoneInfo("Europe/Athens")):
        config.time_offset_minutes = 120
        try:
            query = data_service.candles._candle_query_applied(
                timeframe="D1",
                start="2026-08-13",
                end="2026-08-13",
                limit=10,
            )
        finally:
            config.time_offset_minutes = original_offset

    assert query["resolved_start"] == "2026-08-12T22:00:00Z"
    assert query["resolved_end"] == "2026-08-13T21:59:59.999999Z"


def test_natural_week_and_month_bounds_use_broker_calendar() -> None:
    broker_tz = ZoneInfo("Europe/Nicosia")
    fixed_now = datetime(2026, 8, 13, 12, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.replace(tzinfo=None) if tz is None else fixed_now.astimezone(tz)

    with (
        patch.object(data_service_query, "datetime", FixedDateTime),
        patch.object(data_service.candles.mt5_config, "get_server_tz", return_value=broker_tz),
    ):
        week_start, week_error = data_service.candles._parse_fetch_datetime_arg(
            "this week",
            timeframe="W1",
        )
        month_start, month_error = data_service.candles._parse_fetch_datetime_arg(
            "2026-08-01",
            timeframe="MN1",
        )

    assert week_error is None
    assert week_start == datetime(2026, 8, 9, 21, tzinfo=timezone.utc)
    assert month_error is None
    assert month_start == datetime(2026, 7, 31, 21, tzinfo=timezone.utc)


def test_trim_weekly_and_monthly_date_only_ranges_match_containing_period() -> None:
    broker_tz = ZoneInfo("Europe/Nicosia")
    cases = [
        (
            "W1",
            ["2026-06-27 21:00", "2026-07-04 21:00"],
            "2026-07-01",
            1.0,
        ),
        (
            "MN1",
            ["2026-06-30 21:00", "2026-07-31 21:00"],
            "2026-07-15",
            1.0,
        ),
    ]
    for timeframe, timestamps, requested_date, expected_close in cases:
        df = pd.DataFrame(
            {
                "__epoch": [
                    pd.Timestamp(value, tz="UTC").timestamp() for value in timestamps
                ],
                "close": [1.0, 2.0],
            }
        )
        with patch.object(
            data_service.candles.mt5_config, "get_server_tz", return_value=broker_tz
        ):
            out = data_service.candles._trim_df_to_target(
                df,
                requested_date,
                requested_date,
                candles=100,
                timeframe=timeframe,
            )
        assert out["close"].tolist() == [expected_close]


def test_fetch_rates_with_warmup_uses_utc_epoch_seconds_for_end_ts() -> None:
    rates = [{"time": 1000.0}]
    with patch("mtdata.services.data_service.candles._parse_start_datetime") as mock_parse, patch(
        "mtdata.services.data_service.candles._utc_epoch_seconds", return_value=1000.0
    ) as mock_epoch, patch("mtdata.services.data_service.candles._mt5_copy_rates_range", return_value=rates):
        mock_parse.side_effect = [datetime(2025, 1, 1, 0, 0), datetime(2025, 1, 1, 1, 0)]
        out_rates, out_err = data_service.candles._fetch_rates_with_warmup(
            symbol="EURUSD",
            mt5_timeframe=1,
            timeframe="H1",
            candles=10,
            warmup_bars=2,
            start_datetime="2025-01-01 00:00",
            end_datetime="2025-01-01 01:00",
            retry=False,
            sanity_check=False,
        )

    assert out_err is None
    assert out_rates == rates
    assert mock_epoch.called


def test_weekly_range_safety_budget_does_not_overflow_datetime() -> None:
    rates = [{"time": 1_700_000_000.0}]
    captured = {}

    def _copy_rates(_symbol, _timeframe, start, end):
        captured.update(start=start, end=end)
        return rates

    with (
        patch(
            "mtdata.services.data_service.candles._mt5_copy_rates_range",
            side_effect=_copy_rates,
        ),
        patch.object(
            data_service.candles.mt5_config,
            "get_server_tz",
            return_value=ZoneInfo("Europe/Nicosia"),
        ),
    ):
        out_rates, out_err = data_service.candles._fetch_rates_with_warmup(
            symbol="EURUSD",
            mt5_timeframe=1,
            timeframe="W1",
            candles=100_000,
            warmup_bars=0,
            start_datetime="2026-07-01",
            end_datetime="2026-08-16",
            include_incomplete=True,
            retry=False,
            sanity_check=False,
        )

    assert out_err is None
    assert out_rates == rates
    assert captured["start"].year == 2026
    assert captured["start"] < captured["end"]


def test_fetch_candles_exposes_time_normalization_metadata() -> None:
    rates = [
        {"time": 1_700_000_000.0, "open": 1.10, "high": 1.12, "low": 1.09, "close": 1.11},
        {"time": 1_700_003_600.0, "open": 1.11, "high": 1.13, "low": 1.10, "close": 1.12},
    ]

    @contextmanager
    def _guard(*args, **kwargs):
        yield None, MagicMock(digits=5)

    def _fake_fetch(*args, diagnostics=None, **kwargs):
        return rates, None

    with patch("mtdata.services.data_service.candles.get_symbol_info_cached", return_value=MagicMock(digits=5)), patch(
        "mtdata.services.data_service.candles._symbol_ready_guard",
        _guard,
    ), patch(
        "mtdata.services.data_service.candles._fetch_rates_with_warmup",
        side_effect=_fake_fetch,
    ), patch(
        "mtdata.services.data_service.candles._resolve_client_tz",
        return_value=None,
    ), patch(
        "mtdata.services.data_service.candles.mt5_config.server_tz_name",
        "Europe/Nicosia",
    ), patch(
        "mtdata.services.data_service.candles.mt5_config.time_offset_minutes",
        0,
    ):
        result = data_service.fetch_candles(
            "EURUSD",
            timeframe="H1",
            limit=2,
            include_incomplete=True,
        )

    assert result["time_basis"] == "utc"
    assert result["raw_time_basis"] == "mt5_utc_epoch"
    assert result["time_normalization"] == "mt5_utc_native"
    assert result["broker_server_tz"] == "Europe/Nicosia"
    assert "request bounds and returned epochs use native UTC" in result["timezone_note"]
    assert (
        result["meta"]["diagnostics"]["time_normalization"]["broker_server_tz"]
        == "Europe/Nicosia"
    )
    assert "timezone_note" in result["meta"]["diagnostics"]["time_normalization"]
