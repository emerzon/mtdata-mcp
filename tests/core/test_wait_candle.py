from datetime import datetime, timezone

import pytest

from mtdata.core.trading import time
from mtdata.core.trading.time import (
    _next_candle_close_server_time,
    _next_candle_wait_payload,
    _sleep_until_next_candle,
)


@pytest.fixture()
def utc_server_clock(monkeypatch):
    monkeypatch.setattr(time.mt5_config, "get_server_tz", lambda: None)
    monkeypatch.setattr(time.mt5_config, "get_time_offset_seconds", lambda: 0)
    monkeypatch.setattr(time.mt5_config, "server_tz_name", None)


def test_next_candle_close_skips_weekend_closure(utc_server_clock) -> None:
    friday_night = datetime(2026, 8, 21, 21, 38, tzinfo=timezone.utc)

    result = _next_candle_close_server_time(
        "H1",
        now_utc=friday_night,
        symbol="EURUSD",
    )

    next_utc = time._server_time_naive_to_utc(result)
    assert next_utc >= datetime(2026, 8, 23, 21, tzinfo=timezone.utc)


def test_next_candle_close_server_time_rounds_intraday_frame(utc_server_clock) -> None:
    now_utc = datetime(2026, 3, 13, 10, 2, 10, tzinfo=timezone.utc)

    result = _next_candle_close_server_time("M5", now_utc=now_utc)

    assert result == datetime(2026, 3, 13, 10, 5, 0)


def test_next_candle_close_server_time_handles_weekly_boundary(utc_server_clock) -> None:
    now_utc = datetime(2026, 3, 13, 10, 2, 10, tzinfo=timezone.utc)

    result = _next_candle_close_server_time("W1", now_utc=now_utc)

    assert result == datetime(2026, 3, 16, 0, 0, 0)


def test_next_candle_close_server_time_uses_shared_unsupported_timeframe_error(
    utc_server_clock,
    monkeypatch,
) -> None:
    monkeypatch.setitem(time.TIMEFRAME_SECONDS, "M5", 0)
    monkeypatch.setattr(
        time,
        "unsupported_timeframe_seconds_error",
        lambda timeframe: f"custom unsupported {timeframe}",
    )

    with pytest.raises(ValueError, match="custom unsupported M5"):
        _next_candle_close_server_time("M5", now_utc=datetime(2026, 3, 13, 10, 2, 10, tzinfo=timezone.utc))


def test_sleep_until_next_candle_returns_expected_wait(utc_server_clock) -> None:
    slept = []

    payload = _sleep_until_next_candle(
        "M5",
        buffer_seconds=1.0,
        sleep_impl=lambda seconds: slept.append(seconds),
        now_utc=datetime(2026, 3, 13, 10, 2, 10, tzinfo=timezone.utc),
    )

    assert slept == [171.0]
    assert payload["sleep_seconds"] == 171.0
    assert payload["slept"] is True
    assert payload["status"] == "completed"
    assert payload["next_candle_close_utc"] == "2026-03-13T10:05:00Z"


@pytest.mark.parametrize(
    ("now_utc", "expected_utc", "expected_server"),
    [
        (
            datetime(2026, 3, 29, 0, 54, tzinfo=timezone.utc),
            "2026-03-29T01:00:00Z",
            "2026-03-29T04:00:00+03:00",
        ),
        (
            datetime(2026, 10, 25, 0, 54, tzinfo=timezone.utc),
            "2026-10-25T01:00:00Z",
            "2026-10-25T03:00:00+02:00",
        ),
    ],
)
def test_next_candle_wait_payload_handles_dst_transitions(
    monkeypatch,
    now_utc,
    expected_utc,
    expected_server,
) -> None:
    from zoneinfo import ZoneInfo

    monkeypatch.setattr(time.mt5_config, "get_server_tz", lambda: ZoneInfo("Europe/Nicosia"))
    monkeypatch.setattr(time.mt5_config, "get_time_offset_seconds", lambda: 7200)
    monkeypatch.setattr(time.mt5_config, "server_tz_name", "Europe/Nicosia")

    payload = _next_candle_wait_payload(
        "M15",
        buffer_seconds=1.0,
        now_utc=now_utc,
        symbol="BTCUSD",
    )

    assert payload["next_candle_close_server"] == expected_server
    assert datetime.fromisoformat(payload["next_candle_close_server"]).astimezone(
        timezone.utc
    ) == datetime.fromisoformat(payload["next_candle_close_utc"])
    assert payload["next_candle_close_utc"] == expected_utc
    assert payload["sleep_seconds"] == 361.0


def test_next_candle_wait_payload_without_symbol_has_no_market_session(
    utc_server_clock,
) -> None:
    payload = _next_candle_wait_payload(
        "M1",
        buffer_seconds=1.0,
        now_utc=datetime(2026, 8, 29, 2, 30, tzinfo=timezone.utc),
    )

    assert "market_status" not in payload
    assert payload["sleep_seconds"] == 61.0


