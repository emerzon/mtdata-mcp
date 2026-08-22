"""Tests for market-status early-close calendar rule types."""
from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

import mtdata.core.market_status as ms_mod


@pytest.fixture(autouse=True)
def _clear_holiday_cache():
    ms_mod._get_holidays.cache_clear()
    yield
    ms_mod._get_holidays.cache_clear()


def _fake_holidays_factory(mapping: dict[date, str]):
    """Return a fake country_holidays function with explicit date→name map."""
    def fake_country_holidays(country: str, years):
        return {d: n for d, n in mapping.items()}
    return fake_country_holidays


# ---- _check_market_status: same-day half-holiday ----

class TestSameDayEarlyClose:
    def test_same_day_early_close_returns_early_close_time(self, monkeypatch):
        """A holiday listed in early_close_holidays should return early close,
        not full closure."""
        # Set up a market with same-day early close
        test_market = {
            "name": "Test Exchange",
            "country": "XX",
            "timezone": "UTC",
            "open": (9, 0),
            "close": (16, 0),
            "early_close": (13, 0),
            "early_close_holidays": ["Half Day"],
        }
        monkeypatch.setitem(ms_mod.MARKET_SESSIONS, "TEST", test_market)
        monkeypatch.setattr(
            ms_mod.holidays, "country_holidays",
            _fake_holidays_factory({date(2030, 6, 17): "Half Day"}),
        )

        # 10 AM on the half-holiday (Monday) → should be open with early close
        now = datetime(2030, 6, 17, 10, 0, tzinfo=timezone.utc)
        result = ms_mod._check_market_status("TEST", now)
        assert result["status"] == "open"
        # Close time should be 13:00, not 16:00
        assert result["minutes_until_close"] <= 3 * 60

    def test_same_day_full_holiday_still_closed(self, monkeypatch):
        """A holiday NOT in early_close_holidays should still be full closure."""
        test_market = {
            "name": "Test Exchange",
            "country": "XX",
            "timezone": "UTC",
            "open": (9, 0),
            "close": (16, 0),
            "early_close": (13, 0),
            "early_close_holidays": ["Half Day"],
        }
        monkeypatch.setitem(ms_mod.MARKET_SESSIONS, "TEST", test_market)
        monkeypatch.setattr(
            ms_mod.holidays, "country_holidays",
            _fake_holidays_factory({date(2030, 6, 17): "Full Holiday"}),
        )

        now = datetime(2030, 6, 17, 10, 0, tzinfo=timezone.utc)
        result = ms_mod._check_market_status("TEST", now)
        assert result["status"] == "closed"
        assert result["reason"] == "holiday"


class TestUsProductionEarlyCloseCalendar:
    @pytest.mark.parametrize("market_id", ["NYSE", "NASDAQ"])
    def test_christmas_eve_2026_closes_at_one(self, market_id):
        now = datetime(2026, 12, 24, 12, 0, tzinfo=ZoneInfo("America/New_York"))

        result = ms_mod._check_market_status(market_id, now)

        assert result["status"] == "open"
        assert result["minutes_until_close"] == 60
        assert result["early_close"] is True
        assert result["early_close_time"] == "13:00"

    @pytest.mark.parametrize("market_id", ["NYSE", "NASDAQ"])
    def test_independence_day_eve_2025_closes_at_one(self, market_id):
        now = datetime(2025, 7, 3, 12, 0, tzinfo=ZoneInfo("America/New_York"))

        result = ms_mod._check_market_status(market_id, now)

        assert result["status"] == "open"
        assert result["minutes_until_close"] == 60
        assert result["early_close"] is True

    @pytest.mark.parametrize("market_id", ["NYSE", "NASDAQ"])
    def test_observed_independence_day_2026_remains_closed(self, market_id):
        now = datetime(2026, 7, 3, 12, 0, tzinfo=ZoneInfo("America/New_York"))

        result = ms_mod._check_market_status(market_id, now)

        assert result["status"] == "closed"
        assert result["reason"] == "holiday"
        assert "early_close" not in result


class TestLseProductionEarlyCloseCalendar:
    @pytest.mark.parametrize(
        ("session_time", "expected_date"),
        [
            (datetime(2026, 12, 24, 13, 0, tzinfo=ZoneInfo("Europe/London")), "2026-12-29"),
            (datetime(2028, 12, 22, 13, 0, tzinfo=ZoneInfo("Europe/London")), "2028-12-27"),
            (datetime(2026, 12, 31, 13, 0, tzinfo=ZoneInfo("Europe/London")), "2027-01-04"),
        ],
    )
    def test_last_business_day_before_christmas_or_new_year_closes_at_1230(
        self,
        session_time,
        expected_date,
    ):
        result = ms_mod._check_market_status("LSE", session_time)

        assert result["status"] == "closed"
        assert result["reason"] == "post_close"
        assert result["early_close"] is True
        assert result["early_close_time"] == "12:30"
        assert result["next_open"].startswith(expected_date)

    def test_upcoming_holidays_includes_lse_shortened_session(self):
        upcoming = ms_mod._get_upcoming_holidays(
            ["LSE"],
            days_ahead=10,
            now_utc=datetime(2026, 12, 20, 12, 0, tzinfo=timezone.utc),
        )

        shortened = next(
            row
            for row in upcoming
            if row["date"] == "2026-12-24" and row["impact"] == "early_close"
        )
        assert shortened["early_close_time"] == "12:30"


# ---- _check_market_status: day-after early close ----

class TestDayAfterEarlyClose:
    def test_day_after_thanksgiving_is_early_close(self, monkeypatch):
        """Day after Thanksgiving (Black Friday) should use early close time."""
        test_market = {
            "name": "Test Exchange",
            "country": "XX",
            "timezone": "UTC",
            "open": (9, 0),
            "close": (16, 0),
            "early_close": (13, 0),
            "early_close_holidays": [],
            "early_close_day_after": ["Thanksgiving"],
        }
        monkeypatch.setitem(ms_mod.MARKET_SESSIONS, "TEST", test_market)
        # Thanksgiving on Thursday Nov 28, Black Friday is Nov 29
        monkeypatch.setattr(
            ms_mod.holidays, "country_holidays",
            _fake_holidays_factory({date(2030, 11, 28): "Thanksgiving"}),
        )

        # 10 AM on Black Friday
        now = datetime(2030, 11, 29, 10, 0, tzinfo=timezone.utc)
        result = ms_mod._check_market_status("TEST", now)
        assert result["status"] == "open"
        # Should close at 13:00 (3 hours from 10 AM)
        assert result["minutes_until_close"] == 180

    def test_day_after_non_matching_holiday_normal_close(self, monkeypatch):
        """Day after a holiday not in early_close_day_after → normal hours."""
        test_market = {
            "name": "Test Exchange",
            "country": "XX",
            "timezone": "UTC",
            "open": (9, 0),
            "close": (16, 0),
            "early_close": (13, 0),
            "early_close_holidays": [],
            "early_close_day_after": ["Thanksgiving"],
        }
        monkeypatch.setitem(ms_mod.MARKET_SESSIONS, "TEST", test_market)
        monkeypatch.setattr(
            ms_mod.holidays, "country_holidays",
            _fake_holidays_factory({date(2030, 7, 4): "Independence Day"}),
        )

        # Day after Independence Day
        now = datetime(2030, 7, 5, 10, 0, tzinfo=timezone.utc)
        result = ms_mod._check_market_status("TEST", now)
        assert result["status"] == "open"
        assert result["minutes_until_close"] == 360  # 6 hours to 16:00


# ---- _check_market_status: eve early close ----

class TestEveEarlyClose:
    def test_eve_of_christmas_is_early_close(self, monkeypatch):
        """Day before Christmas Day should use early close time."""
        test_market = {
            "name": "Test Exchange",
            "country": "XX",
            "timezone": "UTC",
            "open": (9, 0),
            "close": (16, 0),
            "early_close": (12, 0),
            "early_close_holidays": [],
            "early_close_eves": ["Christmas Day"],
        }
        monkeypatch.setitem(ms_mod.MARKET_SESSIONS, "TEST", test_market)
        monkeypatch.setattr(
            ms_mod.holidays, "country_holidays",
            _fake_holidays_factory({date(2030, 12, 25): "Christmas Day"}),
        )

        # 10 AM on Dec 24 (eve)
        now = datetime(2030, 12, 24, 10, 0, tzinfo=timezone.utc)
        result = ms_mod._check_market_status("TEST", now)
        assert result["status"] == "open"
        assert result["minutes_until_close"] == 120  # 2 hours to 12:00

    def test_eve_non_matching_holiday_normal_close(self, monkeypatch):
        """Eve of a holiday not in early_close_eves → normal hours."""
        test_market = {
            "name": "Test Exchange",
            "country": "XX",
            "timezone": "UTC",
            "open": (9, 0),
            "close": (16, 0),
            "early_close": (12, 0),
            "early_close_holidays": [],
            "early_close_eves": ["Christmas Day"],
        }
        monkeypatch.setitem(ms_mod.MARKET_SESSIONS, "TEST", test_market)
        monkeypatch.setattr(
            ms_mod.holidays, "country_holidays",
            _fake_holidays_factory({date(2030, 1, 1): "New Year's Day"}),
        )

        # Dec 31 — eve of New Year's Day but not in early_close_eves
        now = datetime(2030, 12, 31, 10, 0, tzinfo=timezone.utc)
        result = ms_mod._check_market_status("TEST", now)
        assert result["status"] == "open"
        assert result["minutes_until_close"] == 360


class TestNextOpenSkipsAndAllowsHolidaySessions:
    def test_weekend_skips_full_holiday_monday(self, monkeypatch):
        test_market = {
            "name": "Test Exchange",
            "country": "XX",
            "timezone": "UTC",
            "open": (9, 0),
            "close": (16, 0),
        }
        monkeypatch.setitem(ms_mod.MARKET_SESSIONS, "TEST", test_market)
        monkeypatch.setattr(
            ms_mod.holidays, "country_holidays",
            _fake_holidays_factory({date(2030, 6, 17): "Full Holiday"}),
        )

        now = datetime(2030, 6, 15, 10, 0, tzinfo=timezone.utc)  # Saturday
        result = ms_mod._check_market_status("TEST", now)

        assert result["status"] == "closed"
        assert result["reason"] == "weekend"
        assert result["next_open"] == "2030-06-18T09:00:00+00:00"

    def test_after_hours_keeps_next_day_early_close_session(self, monkeypatch):
        test_market = {
            "name": "Test Exchange",
            "country": "XX",
            "timezone": "UTC",
            "open": (9, 0),
            "close": (16, 0),
            "early_close": (13, 0),
            "early_close_holidays": ["Half Day"],
        }
        monkeypatch.setitem(ms_mod.MARKET_SESSIONS, "TEST", test_market)
        monkeypatch.setattr(
            ms_mod.holidays, "country_holidays",
            _fake_holidays_factory({date(2030, 12, 25): "Half Day"}),
        )

        now = datetime(2030, 12, 24, 17, 0, tzinfo=timezone.utc)
        result = ms_mod._check_market_status("TEST", now)

        assert result["status"] == "closed"
        assert result["reason"] == "post_close"
        assert result["next_open"] == "2030-12-25T09:00:00+00:00"


# ---- _get_upcoming_holidays: impact field ----

class TestUpcomingHolidayImpact:
    def test_shared_derived_sessions_include_every_market(self):
        now = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)

        upcoming = ms_mod._get_upcoming_holidays(
            ["NYSE", "NASDAQ"], days_ahead=7, now_utc=now
        )

        black_friday = next(
            row
            for row in upcoming
            if row["date"] == "2026-11-27" and row["impact"] == "early_close"
        )
        assert black_friday["markets_affected"] == ["NASDAQ", "NYSE"]

    def test_day_after_holiday_appears_as_early_close(self, monkeypatch):
        """Upcoming holidays should include day-after early close entries."""
        test_market = {
            "name": "Test Exchange",
            "country": "XX",
            "timezone": "UTC",
            "open": (9, 0),
            "close": (16, 0),
            "early_close": (13, 0),
            "early_close_holidays": [],
            "early_close_day_after": ["Thanksgiving"],
        }
        monkeypatch.setitem(ms_mod.MARKET_SESSIONS, "TEST", test_market)
        # Thursday Nov 28 Thanksgiving, Black Friday Nov 29
        monkeypatch.setattr(
            ms_mod.holidays, "country_holidays",
            _fake_holidays_factory({date(2030, 11, 28): "Thanksgiving"}),
        )

        class FixedDT(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2030, 11, 25, 12, 0, tzinfo=tz or timezone.utc)

        monkeypatch.setattr(ms_mod, "datetime", FixedDT)

        upcoming = ms_mod._get_upcoming_holidays(["TEST"], days_ahead=7)
        dates = {e["date"]: e for e in upcoming}
        assert "2030-11-28" in dates
        assert dates["2030-11-28"]["impact"] == "closed"
        assert "2030-11-29" in dates
        assert dates["2030-11-29"]["impact"] == "early_close"
        assert dates["2030-11-29"]["early_close_time"] == "13:00"

    def test_eve_holiday_appears_as_early_close(self, monkeypatch):
        """Upcoming holidays should include eve early close entries."""
        test_market = {
            "name": "Test Exchange",
            "country": "XX",
            "timezone": "UTC",
            "open": (9, 0),
            "close": (16, 0),
            "early_close": (12, 0),
            "early_close_holidays": [],
            "early_close_eves": ["Christmas Day"],
        }
        monkeypatch.setitem(ms_mod.MARKET_SESSIONS, "TEST", test_market)
        monkeypatch.setattr(
            ms_mod.holidays, "country_holidays",
            _fake_holidays_factory({date(2030, 12, 25): "Christmas Day"}),
        )

        class FixedDT(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2030, 12, 20, 12, 0, tzinfo=tz or timezone.utc)

        monkeypatch.setattr(ms_mod, "datetime", FixedDT)

        upcoming = ms_mod._get_upcoming_holidays(["TEST"], days_ahead=7)
        dates = {e["date"]: e for e in upcoming}
        assert "2030-12-25" in dates
        assert dates["2030-12-25"]["impact"] == "closed"
        assert "2030-12-24" in dates
        assert dates["2030-12-24"]["impact"] == "early_close"
        assert dates["2030-12-24"]["early_close_time"] == "12:00"
