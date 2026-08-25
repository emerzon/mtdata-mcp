"""DST-aware geographic market-session classification."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

EQUITY_SESSION_DEFINITION = {
    "basis": "dst_aware_market_sessions",
    "calendar": "equity",
    "asia": "Tokyo open to London open",
    "london": "London open to New York open",
    "london_ny_overlap": "New York open to London close",
    "ny": "London close to New York close",
    "off_session": "New York close to next Tokyo open",
    "market_timezones": {
        "tokyo": "Asia/Tokyo",
        "london": "Europe/London",
        "new_york": "America/New_York",
    },
    "market_local_hours": {
        "tokyo_open": "09:00",
        "london_open": "08:00",
        "london_close": "16:00",
        "new_york_open": "09:30",
        "new_york_close": "16:00",
    },
}
FX_SESSION_DEFINITION = {
    **EQUITY_SESSION_DEFINITION,
    "calendar": "fx",
    "market_local_hours": {
        "tokyo_open": "09:00",
        "london_open": "08:00",
        "london_close": "17:00",
        "new_york_open": "08:00",
        "new_york_close": "17:00",
    },
}
CONTINUOUS_24_7_SESSION_DEFINITION = {
    **{key: value for key, value in FX_SESSION_DEFINITION.items() if key != "off_session"},
    "calendar": "continuous_24_7",
    "basis": "continuous_market",
    "off_hours": (
        "Hours outside the Asia/London/NY geographic windows. This is a 24/7 "
        "liquidity partition, not an exchange close."
    ),
}
_FX_SESSION_WINDOWS = frozenset({"fx", "continuous_24_7"})

_TOKYO_TZ = ZoneInfo("Asia/Tokyo")
_LONDON_TZ = ZoneInfo("Europe/London")
_NEW_YORK_TZ = ZoneInfo("America/New_York")


def _session_boundary(
    day: date,
    *,
    market_tz: ZoneInfo,
    hour: int,
    minute: int,
    analysis_tz: Any,
) -> datetime:
    return datetime(
        day.year,
        day.month,
        day.day,
        hour,
        minute,
        tzinfo=market_tz,
    ).astimezone(analysis_tz)


def _uses_fx_session_windows(session_calendar: str) -> bool:
    return session_calendar in _FX_SESSION_WINDOWS


def _leftover_session_label(session_calendar: str) -> str:
    if session_calendar == "continuous_24_7":
        return "off_hours"
    return "off_session"


def session_boundaries_for_day(
    day: date,
    analysis_tz: Any,
    session_calendar: str = "equity",
) -> Dict[str, datetime]:
    fx = _uses_fx_session_windows(session_calendar)
    return {
        "asia_open": _session_boundary(
            day, market_tz=_TOKYO_TZ, hour=9, minute=0, analysis_tz=analysis_tz
        ),
        "london_open": _session_boundary(
            day, market_tz=_LONDON_TZ, hour=8, minute=0, analysis_tz=analysis_tz
        ),
        "ny_open": _session_boundary(
            day,
            market_tz=_NEW_YORK_TZ,
            hour=8 if fx else 9,
            minute=0 if fx else 30,
            analysis_tz=analysis_tz,
        ),
        "london_close": _session_boundary(
            day,
            market_tz=_LONDON_TZ,
            hour=17 if fx else 16,
            minute=0,
            analysis_tz=analysis_tz,
        ),
        "ny_close": _session_boundary(
            day,
            market_tz=_NEW_YORK_TZ,
            hour=17 if fx else 16,
            minute=0,
            analysis_tz=analysis_tz,
        ),
    }


def market_session_label(
    value: Any,
    *,
    analysis_tz: Any = timezone.utc,
    boundary_cache: Optional[Dict[date, Dict[str, datetime]]] = None,
    session_calendar: str = "equity",
) -> str:
    if not isinstance(value, datetime):
        return "unknown"
    dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    try:
        dt_analysis = dt.astimezone(analysis_tz or timezone.utc)
    except Exception:
        return "unknown"
    cache = boundary_cache if boundary_cache is not None else {}
    anchor_date = dt_analysis.date()
    for day in (
        anchor_date - timedelta(days=1),
        anchor_date,
        anchor_date + timedelta(days=1),
    ):
        boundaries = cache.get(day)
        if boundaries is None:
            boundaries = session_boundaries_for_day(
                day, analysis_tz or timezone.utc, session_calendar
            )
            cache[day] = boundaries
        if boundaries["asia_open"] <= dt_analysis < boundaries["london_open"]:
            return "asia"
        if boundaries["london_open"] <= dt_analysis < boundaries["ny_open"]:
            return "london"
        if boundaries["ny_open"] <= dt_analysis < boundaries["london_close"]:
            return "london_ny_overlap"
        if boundaries["london_close"] <= dt_analysis < boundaries["ny_close"]:
            return "ny"
    return _leftover_session_label(session_calendar)


def session_definition_for_clock(
    clock_name: str, session_calendar: str = "equity"
) -> Dict[str, Any]:
    if session_calendar == "fx":
        source = FX_SESSION_DEFINITION
    elif session_calendar == "continuous_24_7":
        source = CONTINUOUS_24_7_SESSION_DEFINITION
    else:
        source = EQUITY_SESSION_DEFINITION
    out = dict(source)
    out["clock"] = clock_name or "UTC"
    return out
