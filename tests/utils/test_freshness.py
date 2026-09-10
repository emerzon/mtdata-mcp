from datetime import datetime, timezone

from mtdata.utils import time as time_utils
from mtdata.utils.freshness import (
    closed_session_context,
    completed_bar_freshness_fields,
    format_freshness_label,
    freshness_hole_explained_by_session,
    freshness_hole_explained_by_weekend,
    is_standard_weekend_closure,
    round_age_seconds,
    standard_weekend_overlap_seconds,
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
        "assumed_closure_start": "2026-06-05T21:00:00Z",
        "assumed_closure_end": "2026-06-07T21:00:00Z",
        "assumed_closure_seconds": 48 * 60 * 60,
    }
    assert closed_session_context("BTCUSD", now_epoch=saturday) is None


def test_equity_weekend_uses_cash_session_until_monday_open() -> None:
    friday_after_close = datetime(
        2026, 9, 11, 20, 30, tzinfo=timezone.utc
    ).timestamp()
    sunday_evening = datetime(
        2026, 9, 13, 21, 30, tzinfo=timezone.utc
    ).timestamp()
    monday_open = datetime(
        2026, 9, 14, 13, 30, tzinfo=timezone.utc
    ).timestamp()

    friday = closed_session_context("AAPL.NAS", now_epoch=friday_after_close)
    sunday = closed_session_context("AAPL.NAS", now_epoch=sunday_evening)
    bare_ticker = closed_session_context("AAPL", now_epoch=sunday_evening)

    assert friday is not None
    assert friday["market_status_reason"] == "outside_regular_session"
    assert friday["market_venue"] == "NASDAQ"
    assert friday["session_calendar"] == "XNYS"
    assert sunday is not None
    assert sunday["market_status_reason"] == "weekend"
    assert bare_ticker is not None
    assert bare_ticker["market_venue"] == "NYSE"
    assert bare_ticker["assumed_closure_end"] == "2026-09-14T13:30:00Z"
    assert closed_session_context("AAPL.NAS", now_epoch=monday_open) is None


def test_equity_holiday_uses_existing_exchange_calendar() -> None:
    observed_independence_day = datetime(
        2026, 7, 3, 16, 0, tzinfo=timezone.utc
    ).timestamp()

    result = closed_session_context(
        "AAPL.NAS",
        now_epoch=observed_independence_day,
        data_age_seconds=23 * 60 * 60,
    )

    assert result is not None
    assert result["market_status_reason"] == "holiday"
    assert "Independence Day" in result["holiday"]
    assert result["market_status_source"] == "exchange_calendar"
    assert result["assumed_closure_start"] == "2026-07-02T17:00:00Z"
    assert result["assumed_closure_end"] == "2026-07-06T13:30:00Z"
    assert result["freshness_policy_relaxed"] is True


def test_equity_holiday_does_not_hide_preclosure_missing_bar() -> None:
    holiday_cutoff = datetime(
        2026, 7, 3, 16, 0, tzinfo=timezone.utc
    ).timestamp()
    early_close = datetime(
        2026, 7, 2, 17, 0, tzinfo=timezone.utc
    ).timestamp()

    assert freshness_hole_explained_by_session(
        "AAPL.NAS",
        last_completed_epoch=early_close,
        cutoff_epoch=holiday_cutoff,
        bar_seconds=60 * 60,
    )
    assert not freshness_hole_explained_by_session(
        "AAPL.NAS",
        last_completed_epoch=early_close - 60 * 60,
        cutoff_epoch=holiday_cutoff,
        bar_seconds=60 * 60,
    )


def test_fx_and_crypto_keep_their_distinct_weekend_sessions() -> None:
    friday_before_fx_close = datetime(
        2026, 9, 11, 20, 30, tzinfo=timezone.utc
    ).timestamp()
    friday_fx_close = datetime(
        2026, 9, 11, 21, 0, tzinfo=timezone.utc
    ).timestamp()

    assert closed_session_context(
        "EURUSD",
        now_epoch=friday_before_fx_close,
    ) is None
    assert closed_session_context(
        "EURUSD",
        now_epoch=friday_fx_close,
    )["market_status_reason"] == "weekend"
    assert closed_session_context(
        "BTCUSD",
        now_epoch=friday_fx_close,
    ) is None


def test_closed_session_context_marks_other_non_crypto_weekend_markets() -> None:
    saturday = datetime(2026, 6, 6, 12, tzinfo=timezone.utc).timestamp()

    assert closed_session_context("US500", now_epoch=saturday)["market_status"] == "closed"
    assert closed_session_context("XAUUSD", now_epoch=saturday)["market_status"] == "closed"


def test_completed_bar_freshness_uses_close_time_and_shared_policy() -> None:
    last_bar_open = datetime(2026, 8, 19, 19, tzinfo=timezone.utc).timestamp()
    stale_now = datetime(2026, 8, 20, 0, tzinfo=timezone.utc).timestamp()

    stale = completed_bar_freshness_fields(
        "EURUSD",
        "H1",
        last_bar_open,
        now_epoch=stale_now,
    )
    at_boundary = completed_bar_freshness_fields(
        "EURUSD",
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


def test_completed_bar_future_close_is_flagged_and_stale() -> None:
    result = completed_bar_freshness_fields(
        "EURUSD",
        "H1",
        1_000.0,
        now_epoch=4_000.0,
    )

    assert result["data_age_seconds"] == 0
    assert result["data_stale"] is True
    assert result["timestamp_ahead_of_wall_clock"] is True
    assert result["timestamp_in_future"] is True
    assert result["timestamp_skew_seconds"] == 600.0
    assert result["freshness"] == (
        "clock skew, bar timestamp 10m 0s ahead of wall clock"
    )


def test_completed_bar_small_future_skew_is_disclosed(monkeypatch) -> None:
    monkeypatch.setattr(
        time_utils,
        "_broker_calendar_timezone",
        lambda _at_time: timezone.utc,
    )
    result = completed_bar_freshness_fields(
        "EURUSD",
        "H1",
        3_600.0,
        now_epoch=7_192.0,
    )

    assert result["data_stale"] is False
    assert result["timestamp_ahead_of_wall_clock"] is True
    assert result.get("timestamp_in_future") is not True
    assert result["timestamp_skew_seconds"] == 8.0
    assert result["timestamp_skew_basis"] == "latest_completed_bar_close"
    assert result["freshness"] == (
        "fresh with tolerated clock skew, bar timestamp 8s ahead of wall clock"
    )


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


def test_h4_bar_grid_handles_short_and_long_dst_bars(monkeypatch) -> None:
    from zoneinfo import ZoneInfo

    monkeypatch.setattr(
        time_utils,
        "_broker_calendar_timezone",
        lambda at_time: ZoneInfo("Europe/Nicosia"),
    )
    spring_open = datetime(2026, 3, 28, 22, 0, tzinfo=timezone.utc).timestamp()
    spring_close = datetime(2026, 3, 29, 1, 0, tzinfo=timezone.utc).timestamp()
    fall_open = datetime(2026, 10, 24, 21, 0, tzinfo=timezone.utc).timestamp()
    fall_close = datetime(2026, 10, 25, 2, 0, tzinfo=timezone.utc).timestamp()
    fall_reference = datetime(
        2026, 10, 25, 1, 30, tzinfo=timezone.utc
    ).timestamp()

    assert bar_close_epoch(spring_open, "H4") == spring_close
    assert bar_close_epoch(fall_open, "H4") == fall_close
    assert time_utils.timeframe_bar_open_epoch(
        fall_reference,
        "H4",
    ) == fall_open


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


def test_weekend_overlap_covers_friday_close_to_saturday_cutoff() -> None:
    last_completed = datetime(2026, 8, 28, 21, tzinfo=timezone.utc).timestamp()
    cutoff = datetime(2026, 8, 29, 21, tzinfo=timezone.utc).timestamp()

    assert standard_weekend_overlap_seconds(last_completed, cutoff) == 24 * 60 * 60
    assert freshness_hole_explained_by_weekend(
        last_completed_epoch=last_completed,
        cutoff_epoch=cutoff,
        bar_seconds=86400,
    )


def test_weekday_freshness_hole_is_not_explained_by_weekend() -> None:
    last_completed = datetime(2026, 8, 24, 21, tzinfo=timezone.utc).timestamp()
    cutoff = datetime(2026, 8, 25, 21, tzinfo=timezone.utc).timestamp()

    assert standard_weekend_overlap_seconds(last_completed, cutoff) == 0.0
    assert not freshness_hole_explained_by_weekend(
        last_completed_epoch=last_completed,
        cutoff_epoch=cutoff,
        bar_seconds=86400,
    )


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


def test_weekend_tick_before_close_is_not_session_relaxed() -> None:
    saturday = datetime(2026, 6, 6, 12, tzinfo=timezone.utc).timestamp()
    friday = datetime(2026, 6, 5, 20, tzinfo=timezone.utc).timestamp()

    result = build_tick_freshness_context(
        "EURUSD",
        tick_epoch=friday,
        now_epoch=saturday,
        stale_after_seconds=300,
    )

    assert result["data_stale"] is True
    assert result["freshness_policy_relaxed"] is False
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
