from datetime import datetime, timezone

from mtdata.utils import time as time_utils
from mtdata.utils.freshness import (
    closed_session_context,
    completed_bar_freshness_fields,
    format_freshness_label,
    is_standard_weekend_closure,
    round_age_seconds,
)
from mtdata.utils.market_metadata import build_tick_freshness_context
from mtdata.utils.time import bar_close_epoch


def test_closed_session_context_marks_weekend_fx_but_not_crypto():
    saturday = datetime(2026, 6, 6, 12, tzinfo=timezone.utc).timestamp()

    assert closed_session_context("EURUSD", now_epoch=saturday) == {
        "market_status": "closed",
        "market_status_reason": "weekend",
        "market_status_source": "standard_weekend_hours",
        "note": "Market is closed; showing the latest completed session tick.",
    }
    assert closed_session_context("BTCUSD", now_epoch=saturday) is None


def test_closed_session_context_marks_other_non_crypto_weekend_markets() -> None:
    saturday = datetime(2026, 6, 6, 12, tzinfo=timezone.utc).timestamp()

    assert closed_session_context("US500", now_epoch=saturday)["market_status"] == "closed"
    assert closed_session_context("XAUUSD", now_epoch=saturday)["market_status"] == "closed"


def test_completed_bar_freshness_uses_close_time_and_shared_policy() -> None:
    last_bar_open = datetime(2026, 8, 19, 19, tzinfo=timezone.utc).timestamp()
    stale_now = datetime(2026, 8, 20, 0, tzinfo=timezone.utc).timestamp()

    stale = completed_bar_freshness_fields(
        "TSLA.NAS",
        "H1",
        last_bar_open,
        now_epoch=stale_now,
    )
    at_boundary = completed_bar_freshness_fields(
        "TSLA.NAS",
        "H1",
        last_bar_open,
        now_epoch=datetime(2026, 8, 19, 23, tzinfo=timezone.utc).timestamp(),
    )

    assert stale["data_as_of"] == "2026-08-19T20:00:00Z"
    assert stale["data_age_seconds"] == 4 * 60 * 60
    assert stale["stale_after_seconds"] == 3 * 60 * 60
    assert stale["data_stale"] is True
    assert stale["history_policy_ok"] is False
    assert stale["freshness"] == "stale, bar 4h 0m ago"
    assert at_boundary["data_stale"] is False
    assert at_boundary["history_policy_ok"] is True


def test_closed_session_context_allows_fx_after_sunday_utc_reopen() -> None:
    sunday_reopen = datetime(2026, 6, 14, 21, 0, tzinfo=timezone.utc).timestamp()

    assert closed_session_context("EURUSD", now_epoch=sunday_reopen) is None


def test_weekend_boundary_tracks_new_york_daylight_saving_time() -> None:
    winter_before_reopen = datetime(2026, 1, 4, 21, 30, tzinfo=timezone.utc)
    winter_after_reopen = datetime(2026, 1, 4, 22, 30, tzinfo=timezone.utc)

    assert is_standard_weekend_closure(winter_before_reopen)
    assert not is_standard_weekend_closure(winter_after_reopen)


def test_monthly_bar_close_uses_calendar_month_boundary(monkeypatch) -> None:
    monkeypatch.setattr(time_utils, "_broker_calendar_timezone", lambda at_time: timezone.utc)
    opened = datetime(2026, 2, 1, tzinfo=timezone.utc).timestamp()
    expected = datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp()

    assert bar_close_epoch(opened, "MN1") == expected


def test_daily_bar_close_uses_broker_calendar_across_dst(monkeypatch) -> None:
    from zoneinfo import ZoneInfo

    monkeypatch.setattr(
        time_utils,
        "_broker_calendar_timezone",
        lambda at_time: ZoneInfo("Europe/Helsinki"),
    )
    opened = datetime(2026, 3, 28, 22, 0, tzinfo=timezone.utc).timestamp()
    expected = datetime(2026, 3, 29, 21, 0, tzinfo=timezone.utc).timestamp()

    assert bar_close_epoch(opened, "D1") == expected


def test_monthly_bar_close_uses_broker_local_month(monkeypatch) -> None:
    from zoneinfo import ZoneInfo

    monkeypatch.setattr(
        time_utils,
        "_broker_calendar_timezone",
        lambda at_time: ZoneInfo("Europe/Helsinki"),
    )
    opened = datetime(2026, 2, 28, 22, 0, tzinfo=timezone.utc).timestamp()
    expected = datetime(2026, 3, 31, 21, 0, tzinfo=timezone.utc).timestamp()

    assert bar_close_epoch(opened, "MN1") == expected


def test_standard_weekend_closure_uses_new_york_close_boundaries() -> None:
    assert is_standard_weekend_closure(
        datetime(2026, 6, 14, 20, 59, tzinfo=timezone.utc)
    )
    assert not is_standard_weekend_closure(
        datetime(2026, 6, 14, 21, 0, tzinfo=timezone.utc)
    )
    assert not is_standard_weekend_closure(
        datetime(2026, 6, 12, 20, 59, tzinfo=timezone.utc)
    )
    assert is_standard_weekend_closure(
        datetime(2026, 6, 12, 21, 0, tzinfo=timezone.utc)
    )


def test_closed_session_context_does_not_relax_very_old_data():
    saturday = datetime(2026, 6, 6, 12, tzinfo=timezone.utc).timestamp()

    result = closed_session_context(
        "EURUSD",
        now_epoch=saturday,
        data_age_seconds=4 * 24 * 60 * 60,
    )

    assert result is not None
    assert result["freshness_policy_relaxed"] is False
    assert result["assumed_closure_start"] == "2026-06-05T21:00:00Z"
    assert result["assumed_closure_end"] == "2026-06-07T21:00:00Z"
    assert result["assumed_closure_seconds"] == 48 * 60 * 60


def test_round_age_seconds_uses_integer_seconds() -> None:
    assert round_age_seconds(18055.499814987183) == 18055
    assert round_age_seconds(0.25) == 0.25
    assert round_age_seconds(0.4) == 0.4
    assert round_age_seconds(1.4) == 1


def test_tick_freshness_rounds_age_to_integer_seconds() -> None:
    result = build_tick_freshness_context(
        "EURUSD",
        tick_epoch=1_000.0,
        now_epoch=1_000.0 + 18055.499814987183,
    )

    assert result["data_age_seconds"] == 18055
    assert isinstance(result["data_age_seconds"], int)


def test_closed_session_does_not_restore_unrounded_age() -> None:
    saturday = datetime(2026, 6, 6, 12, tzinfo=timezone.utc).timestamp()
    friday = datetime(2026, 6, 5, 20, tzinfo=timezone.utc).timestamp() + 0.499814987183

    result = build_tick_freshness_context(
        "EURUSD",
        tick_epoch=friday,
        now_epoch=saturday,
        stale_after_seconds=300,
    )

    assert result["data_age_seconds"] == int(round(saturday - friday))
    assert isinstance(result["data_age_seconds"], int)


def test_weekend_tick_keeps_absolute_stale_flag() -> None:
    saturday = datetime(2026, 6, 6, 12, tzinfo=timezone.utc).timestamp()
    friday = datetime(2026, 6, 5, 20, tzinfo=timezone.utc).timestamp()

    result = build_tick_freshness_context(
        "EURUSD",
        tick_epoch=friday,
        now_epoch=saturday,
        stale_after_seconds=300,
    )

    assert result["data_stale"] is True
    assert result["freshness_policy_relaxed"] is True
    assert result["usable_for_live_trading"] is False
    assert result["freshness_basis"] == "absolute_300s"


def test_future_tick_is_not_accepted_as_fresh() -> None:
    result = build_tick_freshness_context(
        "TSLA.NAS-24",
        tick_epoch=10_800.0,
        now_epoch=0.0,
        stale_after_seconds=300,
    )

    assert result["data_age_seconds"] == 0.0
    assert result["data_stale"] is True
    assert result["usable_for_live_trading"] is False
    assert result["timestamp_in_future"] is True
    assert result["timestamp_skew_seconds"] == 10_800.0
    assert result["freshness_state"] == "clock_skew"
    assert result["freshness"] == "clock skew, tick timestamp 3h 0m ahead of wall clock"


def test_small_future_clock_skew_is_disclosed_without_zero_age() -> None:
    result = build_tick_freshness_context(
        "EURUSD",
        tick_epoch=1_008.0,
        now_epoch=1_000.0,
    )

    assert result["data_age_seconds"] == 0.0
    assert result["data_stale"] is False
    assert result["usable_for_live_trading"] is True
    assert result["timestamp_ahead_of_wall_clock"] is True
    assert result.get("timestamp_in_future") is not True
    assert result["timestamp_skew_seconds"] == 8.0
    assert result["timestamp_skew_tolerance_seconds"] == 10
    assert result["freshness_reason"] == "clock_skew_within_tolerance"


def test_quote_at_shared_execution_threshold_is_live() -> None:
    result = build_tick_freshness_context(
        "EURUSD",
        tick_epoch=990.0,
        now_epoch=1_000.0,
    )

    assert result["data_stale"] is False
    assert result["freshness_state"] == "live"
    assert result["freshness"] == "fresh, tick 10s ago"
    assert result["live_max_age_seconds"] == 10
    assert result["usable_for_live_trading"] is True
    assert result["usable_for_live_trading_basis"] == "quote_age_and_market_session"


def test_live_tick_is_usable_for_execution() -> None:
    result = build_tick_freshness_context(
        "EURUSD",
        tick_epoch=995.0,
        now_epoch=1_000.0,
    )

    assert result["freshness_state"] == "live"
    assert result["usable_for_live_trading"] is True


class _FalseLike:
    def __bool__(self):
        return False


class _TrueLike:
    def __bool__(self):
        return True


def test_format_freshness_label_accepts_bool_like_stale_flags():
    assert format_freshness_label(data_stale=_TrueLike()) == "stale"
    assert format_freshness_label(data_stale=_FalseLike()) == "fresh"


def test_format_freshness_label_ignores_textual_stale_flags():
    assert format_freshness_label(data_stale="false") is None
