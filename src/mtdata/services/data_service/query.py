"""Query-bound parsing for candle history requests."""

import re
from datetime import datetime
from datetime import timezone as dt_timezone
from typing import Any, Dict, Optional

from ...bootstrap.settings import mt5_config
from ...shared.schema import TimeframeLiteral
from ...utils.time import (
    _broker_calendar_timezone as _resolve_broker_calendar_timezone,
)
from ...utils.time import (
    _localize_broker_calendar_time,
    as_utc,
    format_datetime_utc,
)
from ...utils.utils import (
    _calendar_period_bounds,
    _iana_timezone_datetime_issue,
    _is_calendar_period_expression,
    _parse_end_datetime,
    _parse_start_datetime,
)

_DATE_FORMAT_HINT = (
    "Accepted examples: '2026-01-15', '2026-01-15 14:30', "
    "'2026-01-15T14:30:00Z', '2026-01-15 09:30 America/New_York', "
    "'yesterday', '2 days ago', 'last Friday'."
)


def _is_iso_date_only(value: Optional[str]) -> bool:
    return bool(
        value is not None
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value).strip())
    )


def _is_calendar_query_bound(value: Optional[str]) -> bool:
    return _is_iso_date_only(value) or _is_calendar_period_expression(value)


def _broker_calendar_timezone(at_time: Optional[datetime] = None) -> Any:
    """Return the shared broker calendar zone, or None if none is configured.

    D1/W1/MN1 date-only bounds fail closed when neither ``MT5_SERVER_TZ`` nor a
    non-zero ``MT5_TIME_OFFSET_MINUTES`` is set. When configured, priority
    matches ``utils.time._broker_calendar_timezone``: static offset first.
    """
    try:
        named_tz = mt5_config.get_server_tz()
    except Exception:
        named_tz = None
    try:
        static_offset_minutes = int(getattr(mt5_config, "time_offset_minutes", 0) or 0)
    except (TypeError, ValueError):
        static_offset_minutes = 0
    if named_tz is None and not static_offset_minutes:
        return None
    if at_time is None:
        at_time = datetime.now(dt_timezone.utc)
    return _resolve_broker_calendar_timezone(at_time)


def _missing_broker_session_timezone_error(
    timeframe: Optional[str],
    value: Optional[str],
) -> Optional[str]:
    if timeframe not in {"D1", "W1", "MN1"} or not _is_calendar_query_bound(value):
        return None
    if _broker_calendar_timezone() is not None:
        return None
    return (
        f"{timeframe} date-only and calendar bounds need MT5_SERVER_TZ or a "
        "non-zero MT5_TIME_OFFSET_MINUTES so they resolve broker-local session "
        "days. Set MT5_SERVER_TZ (for example Europe/Nicosia) or pass an "
        "explicit UTC timestamp."
    )


def _parse_candle_calendar_bound(
    value: Optional[str],
    *,
    timeframe: Optional[str],
    end_bound: bool,
) -> Optional[datetime]:
    """Resolve D1/W1/MN1 calendar labels at broker-local midnight."""
    if timeframe not in {"D1", "W1", "MN1"} or not _is_calendar_query_bound(value):
        return None
    tz_error = _missing_broker_session_timezone_error(timeframe, value)
    if tz_error:
        raise ValueError(tz_error)
    broker_tz = _broker_calendar_timezone()
    if broker_tz is None:
        raise ValueError(
            _missing_broker_session_timezone_error(timeframe, value)
            or "Broker session timezone is not configured."
        )
    text = str(value or "").strip()
    if _is_iso_date_only(text):
        local_date = datetime.strptime(text, "%Y-%m-%d").date()
        local_bound = datetime.combine(
            local_date,
            datetime.max.time() if end_bound else datetime.min.time(),
        )
    else:
        period = _calendar_period_bounds(
            text,
            now=datetime.now(dt_timezone.utc),
            calendar_timezone=broker_tz,
        )
        if period is None:
            return None
        local_bound = period[1] if end_bound else period[0]
    return _localize_broker_calendar_time(
        broker_tz,
        local_bound,
    ).astimezone(dt_timezone.utc)


def _parse_fetch_datetime_arg(
    value: str,
    *,
    end_bound: bool = False,
    timeframe: Optional[str] = None,
) -> tuple[Optional[datetime], Optional[str]]:
    try:
        parsed = _parse_candle_calendar_bound(
            value,
            timeframe=timeframe,
            end_bound=end_bound,
        )
    except ValueError as exc:
        return None, str(exc)
    if parsed is None:
        parsed = _parse_end_datetime(value) if end_bound else _parse_start_datetime(value)
    if parsed is None:
        issue = _iana_timezone_datetime_issue(value)
        if issue is not None:
            return None, f"{issue['error']} {issue['remediation']}"
        return None, f"Could not parse date {value!r}. {_DATE_FORMAT_HINT}"
    return as_utc(parsed), None


def _format_resolved_query_bound(value: datetime) -> str:
    """Format a parsed query bound without hiding an inclusive day-end."""
    resolved = as_utc(value)
    timespec = "microseconds" if resolved.microsecond else "seconds"
    return format_datetime_utc(resolved, timespec=timespec)


def _candle_query_applied(
    *,
    timeframe: TimeframeLiteral,
    start: Optional[str],
    end: Optional[str],
    limit: Optional[int],
) -> Dict[str, Any]:
    query: Dict[str, Any] = {"mode": "range", "timeframe": timeframe}
    if limit is not None:
        query["limit"] = int(limit)
    calendar_session_bounds = timeframe in {"D1", "W1", "MN1"} and (
        _is_calendar_query_bound(start) or _is_calendar_query_bound(end)
    )
    if calendar_session_bounds:
        query["bound_basis"] = "broker_session_calendar"
    elif _is_calendar_query_bound(start) or _is_calendar_query_bound(end):
        query["bound_basis"] = "utc_calendar"
    if end not in (None, ""):
        query["end_filter"] = (
            "bar_open_calendar_period"
            if _is_calendar_query_bound(end)
            else "bar_close"
        )

    for name, raw_value, end_bound in (
        ("start", start, False),
        ("end", end, True),
    ):
        if raw_value in (None, ""):
            continue
        query[name] = raw_value
        resolved, _ = _parse_fetch_datetime_arg(
            raw_value,
            end_bound=end_bound,
            timeframe=timeframe,
        )
        if resolved is None:
            continue
        query[f"resolved_{name}"] = _format_resolved_query_bound(resolved)
        is_iso_day = _is_iso_date_only(raw_value)
        is_natural_period = _is_calendar_period_expression(raw_value)
        if calendar_session_bounds and (is_iso_day or is_natural_period):
            bound_mode = "inclusive_broker_session_period"
        elif is_iso_day:
            bound_mode = "inclusive_day_end" if end_bound else "inclusive_day_start"
        elif is_natural_period:
            period = _calendar_period_bounds(str(raw_value))
            period_kind = period[2] if period is not None else "day"
            bound_mode = f"inclusive_{period_kind}_{'end' if end_bound else 'start'}"
        else:
            bound_mode = "inclusive_instant"
        query[f"{name}_bound"] = bound_mode
    return query
