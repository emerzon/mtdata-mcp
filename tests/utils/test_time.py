from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import mtdata.utils.time as time_utils
from mtdata.utils.time import (
    _format_datetime_explicit,
    as_utc,
    format_datetime_utc,
    format_epoch_utc,
    format_relative_date,
    format_relative_time,
    parse_iso_utc,
)


def test_as_utc_treats_naive_values_as_utc_and_converts_aware_values() -> None:
    naive = datetime(2026, 8, 7, 13, 30, 15)
    central = naive.replace(tzinfo=timezone(timedelta(hours=-5)))

    assert as_utc(naive) == naive.replace(tzinfo=timezone.utc)
    assert as_utc(central) == datetime(2026, 8, 7, 18, 30, 15, tzinfo=timezone.utc)


def test_format_epoch_utc_uses_second_resolution() -> None:
    assert format_epoch_utc(1000.75) == "1970-01-01T00:16:40Z"


def test_format_epoch_utc_rejects_invalid_values() -> None:
    assert format_epoch_utc(None) is None
    assert format_epoch_utc("not-an-epoch") is None


def test_format_epoch_utc_supports_minute_precision() -> None:
    assert format_epoch_utc(0, timespec="minutes") == "1970-01-01T00:00Z"


def test_parse_iso_utc_accepts_z_offsets_and_naive_values() -> None:
    assert parse_iso_utc("2024-01-01T00:00:00Z") == datetime(
        2024, 1, 1, tzinfo=timezone.utc
    )
    assert parse_iso_utc("2024-01-01T01:00:00+01:00") == datetime(
        2024, 1, 1, tzinfo=timezone.utc
    )
    assert parse_iso_utc("2024-01-01T00:00:00") == datetime(
        2024, 1, 1, tzinfo=timezone.utc
    )


def test_format_datetime_utc_normalizes_offsets_and_resolution() -> None:
    local_value = datetime(
        2026,
        8,
        7,
        8,
        30,
        15,
        123456,
        tzinfo=timezone(timedelta(hours=-5)),
    )

    assert format_datetime_utc(local_value) == "2026-08-07T13:30:15Z"
    assert (
        format_datetime_utc(local_value, timespec="microseconds")
        == "2026-08-07T13:30:15.123456Z"
    )
    assert format_datetime_utc(local_value.replace(tzinfo=None)) == (
        "2026-08-07T08:30:15Z"
    )


def test_client_local_formatters_share_resolved_timezone(monkeypatch) -> None:
    client_tz = timezone(timedelta(hours=5, minutes=30))
    monkeypatch.setattr(time_utils, "_resolve_client_tz", lambda: client_tz)

    assert time_utils._use_client_tz() is True
    assert time_utils._format_time_minimal_local(0) == "1970-01-01T05:30+05:30"


def test_format_relative_time_handles_past_future_and_large_units() -> None:
    now = datetime(2026, 1, 31, 12, tzinfo=timezone.utc)

    assert format_relative_time(now - timedelta(minutes=5), now=now) == "5 minutes ago"
    assert format_relative_time(now + timedelta(hours=3), now=now) == "in 3 hours"
    assert format_relative_time(now - timedelta(days=14), now=now) == "2 weeks ago"
    assert format_relative_time(now - timedelta(days=60), now=now) == "2 months ago"


def test_format_relative_date_uses_calendar_days_not_midnight_countdown() -> None:
    now = datetime(2026, 8, 26, 20, 42, tzinfo=timezone.utc)

    assert format_relative_date(date(2026, 8, 27), now=now) == "tomorrow"
    assert format_relative_date(date(2026, 8, 26), now=now) == "today"


def test_format_datetime_explicit_keeps_named_zone_offset_at_gmt() -> None:
    london = datetime(2026, 1, 15, 12, 0, tzinfo=ZoneInfo("Europe/London"))
    utc = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)

    assert _format_datetime_explicit(london, timespec="seconds") == "2026-01-15T12:00:00+00:00"
    assert _format_datetime_explicit(utc, timespec="seconds") == "2026-01-15T12:00:00Z"
