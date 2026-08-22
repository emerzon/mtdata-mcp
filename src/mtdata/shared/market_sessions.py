"""Canonical exchange-session definitions and early-close evaluation."""

from __future__ import annotations

from datetime import timedelta
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
        "timezone": "Europe/Paris",
        "open": (9, 0),
        "close": (17, 30),
        "early_close": None,
        "early_close_holidays": [],
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


@lru_cache(maxsize=128)
def exchange_holidays(exchange: str, year: int) -> holidays.HolidayBase:
    """Return the venue trading calendar supplied by python-holidays."""
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
