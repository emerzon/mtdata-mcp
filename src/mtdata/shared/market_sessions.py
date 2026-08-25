"""Canonical exchange-session definitions and early-close evaluation."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple

import holidays

MARKET_SESSIONS: Dict[str, Dict[str, Any]] = {
    "NYSE": {
        "name": "New York Stock Exchange",
        "country": "US",
        "exchange_calendar": "XNYS",
        "timezone": "America/New_York",
        "pre_open": (4, 0),
        "open": (9, 30),
        "close": (16, 0),
        "after_hours_close": (20, 0),
        "early_close": (13, 0),
        "early_close_holidays": [],
        "early_close_day_after": ["Thanksgiving"],
        "early_close_eves": ["Independence Day", "Christmas Day"],
    },
    "NASDAQ": {
        "name": "NASDAQ",
        "country": "US",
        "exchange_calendar": "XNYS",
        "timezone": "America/New_York",
        "pre_open": (4, 0),
        "open": (9, 30),
        "close": (16, 0),
        "after_hours_close": (20, 0),
        "early_close": (13, 0),
        "early_close_holidays": [],
        "early_close_day_after": ["Thanksgiving"],
        "early_close_eves": ["Independence Day", "Christmas Day"],
    },
    "LSE": {
        "name": "London Stock Exchange",
        "country": "UK",
        "timezone": "Europe/London",
        "open": (8, 0),
        "close": (16, 30),
        "early_close": (12, 30),
        "early_close_holidays": [],
        "early_close_last_business_day_before": ["Christmas Day", "New Year's Day"],
    },
    "XETRA": {
        "name": "Xetra (Frankfurt)",
        "country": "DE",
        "exchange_calendar": "XETR",
        "timezone": "Europe/Berlin",
        "open": (9, 0),
        "close": (17, 30),
        "early_close": None,
        "early_close_holidays": [],
    },
    "EURONEXT": {
        "name": "Euronext Paris",
        "country": "FR",
        "exchange_calendar": "EURONEXT",
        "timezone": "Europe/Paris",
        "open": (9, 0),
        "close": (17, 30),
        "early_close": (14, 0),
        "early_close_holidays": [],
        "early_close_eves": ["Christmas Day", "New Year's Day"],
    },
    "TSE": {
        "name": "Tokyo Stock Exchange",
        "country": "JP",
        "exchange_calendar": "XJPX",
        "timezone": "Asia/Tokyo",
        "open": (9, 0),
        "close": (15, 30),
        "lunch_start": (11, 30),
        "lunch_end": (12, 30),
        "early_close": None,
        "early_close_holidays": [],
    },
    "HKEX": {
        "name": "Hong Kong Stock Exchange",
        "country": "HK",
        "exchange_calendar": "XHKG",
        "timezone": "Asia/Hong_Kong",
        "open": (9, 30),
        "close": (16, 0),
        "lunch_start": (12, 0),
        "lunch_end": (13, 0),
        "early_close": (12, 0),
        "early_close_holidays": [],
        "early_close_eves": ["Christmas Day", "New Year's Day"],
    },
    "SSE": {
        "name": "Shanghai Stock Exchange",
        "country": "CN",
        "exchange_calendar": "XSHG",
        "timezone": "Asia/Shanghai",
        "open": (9, 30),
        "close": (15, 0),
        "lunch_start": (11, 30),
        "lunch_end": (13, 0),
        "early_close": None,
        "early_close_holidays": [],
    },
    "ASX": {
        "name": "Australian Securities Exchange",
        "country": "AU",
        "timezone": "Australia/Sydney",
        "open": (10, 0),
        "pre_open": (7, 0),
        "close": (16, 0),
        "early_close": (14, 0),
        "early_close_holidays": [],
        "early_close_eves": ["Christmas Day"],
    },
}

HolidayResolver = Callable[[str, Any, Optional[str]], Tuple[bool, Optional[str]]]

# Official Euronext Paris closed days: New Year's Day, Good Friday, Easter
# Monday, Labour Day (1 May), and Christmas Day. Bastille Day (14 July) is a
# French national holiday but the cash market remains open. Christmas Eve and
# New Year's Eve are half days when they fall on a weekday.
_EURONEXT_CALENDAR_KEYS = frozenset({"EURONEXT", "XPAR"})


def _gregorian_easter(year: int) -> date:
    """Return Easter Sunday for *year* using the Anonymous Gregorian algorithm."""
    year_value = int(year)
    a = year_value % 19
    b = year_value // 100
    c = year_value % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year_value, month, day)


def euronext_paris_holidays(year: int) -> Dict[date, str]:
    """Return Euronext Paris full-day closures for *year*."""
    easter = _gregorian_easter(year)
    return {
        date(int(year), 1, 1): "New Year's Day",
        easter - timedelta(days=2): "Good Friday",
        easter + timedelta(days=1): "Easter Monday",
        date(int(year), 5, 1): "Labour Day",
        date(int(year), 12, 25): "Christmas Day",
    }


@lru_cache(maxsize=128)
def exchange_holidays(exchange: str, year: int) -> holidays.HolidayBase:
    """Return the venue trading calendar for *exchange* and *year*."""
    key = str(exchange or "").strip().upper()
    if key in _EURONEXT_CALENDAR_KEYS:
        return euronext_paris_holidays(int(year))  # type: ignore[return-value]
    return holidays.financial_holidays(exchange, years=[int(year)])


def market_for_exchange_calendar(calendar: str) -> Optional[Dict[str, Any]]:
    """Return the canonical market definition for an exchange calendar."""
    normalized = str(calendar or "").strip().upper()
    return next(
        (
            market
            for market in MARKET_SESSIONS.values()
            if str(market.get("exchange_calendar") or "").upper() == normalized
        ),
        None,
    )


def is_early_close_session(
    market: Dict[str, Any],
    country: str,
    session_dt: Any,
    *,
    holiday_resolver: HolidayResolver,
) -> bool:
    """Return whether a session uses its configured shortened hours."""
    exchange = market.get("exchange_calendar")
    is_holiday, holiday_name = holiday_resolver(country, session_dt, exchange)

    if is_holiday and holiday_name and market.get("early_close_holidays"):
        if any(
            name.lower() in holiday_name.lower()
            for name in market["early_close_holidays"]
        ):
            return True

    if is_holiday:
        return False

    if market.get("early_close_day_after"):
        _, yesterday_holiday = holiday_resolver(
            country,
            session_dt - timedelta(days=1),
            exchange,
        )
        if yesterday_holiday and any(
            name.lower() in yesterday_holiday.lower()
            for name in market["early_close_day_after"]
        ):
            return True

    if market.get("early_close_eves"):
        _, tomorrow_holiday = holiday_resolver(
            country,
            session_dt + timedelta(days=1),
            exchange,
        )
        if tomorrow_holiday and any(
            name.lower() in tomorrow_holiday.lower()
            for name in market["early_close_eves"]
        ):
            return True

    target_names = market.get("early_close_last_business_day_before")
    if target_names:
        for days_ahead in range(1, 8):
            next_day = session_dt + timedelta(days=days_ahead)
            next_is_holiday, next_holiday = holiday_resolver(
                country,
                next_day,
                exchange,
            )
            if next_is_holiday and next_holiday:
                if any(
                    target.lower() in next_holiday.lower() for target in target_names
                ):
                    return True
                continue
            if next_day.weekday() < 5:
                break

    return False
