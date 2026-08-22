"""Canonical time formatting and client-timezone helpers."""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from ..shared.constants import TIME_DISPLAY_FORMAT


def bar_close_epoch(open_epoch: Any, timeframe: str) -> float:
    """Return the UTC end epoch for a bar opened at *open_epoch*.

    Daily, weekly, and monthly bars close on broker-calendar boundaries so a
    configured broker timezone remains correct across daylight-saving changes.
    """
    opened = float(open_epoch)
    normalized_timeframe = str(timeframe).upper()
    if normalized_timeframe in {"D1", "W1", "MN1"}:
        opened_at = datetime.fromtimestamp(opened, tz=timezone.utc)
        broker_tz = _broker_calendar_timezone(opened_at)
        opened_local = opened_at.astimezone(broker_tz)
        local_naive = opened_local.replace(tzinfo=None)
        if normalized_timeframe == "D1":
            closed_local_naive = local_naive + timedelta(days=1)
        elif normalized_timeframe == "W1":
            closed_local_naive = local_naive + timedelta(days=7)
        elif local_naive.month == 12:
            closed_local_naive = local_naive.replace(
                year=local_naive.year + 1, month=1, day=1
            )
        else:
            closed_local_naive = local_naive.replace(
                month=local_naive.month + 1, day=1
            )
        closed_at = _localize_broker_calendar_time(
            broker_tz,
            closed_local_naive,
        )
        return float(closed_at.astimezone(timezone.utc).timestamp())

    from ..shared.constants import TIMEFRAME_SECONDS

    seconds = TIMEFRAME_SECONDS.get(normalized_timeframe)
    if seconds is None:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    return opened + float(seconds)


def _broker_calendar_timezone(at_time: datetime):
    from ..bootstrap.settings import mt5_config

    static_offset_minutes = int(getattr(mt5_config, "time_offset_minutes", 0) or 0)
    if static_offset_minutes:
        return timezone(timedelta(minutes=static_offset_minutes))
    server_tz = mt5_config.get_server_tz()
    if server_tz is not None:
        return server_tz
    offset_seconds = int(mt5_config.get_time_offset_seconds(at_time=at_time) or 0)
    return timezone(timedelta(seconds=offset_seconds))


def _localize_broker_calendar_time(broker_tz: Any, value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value
    return value.replace(tzinfo=broker_tz)


def format_epoch_utc(value: Any) -> Optional[str]:
    """Format epoch seconds as second-resolution RFC 3339 UTC."""
    try:
        timestamp = float(value)
        return format_datetime_utc(
            datetime.fromtimestamp(timestamp, timezone.utc),
            timespec="seconds",
        )
    except (OSError, OverflowError, TypeError, ValueError):
        return None


def as_utc(value: datetime) -> datetime:
    """Convert a datetime to UTC, treating a naive value as already UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def format_datetime_utc(value: datetime, *, timespec: str = "seconds") -> str:
    """Format a datetime as RFC 3339 UTC, treating naive values as UTC."""
    resolved = as_utc(value)
    return resolved.isoformat(timespec=timespec).replace("+00:00", "Z")


def format_relative_time(value: datetime, *, now: Optional[datetime] = None) -> str:
    """Format a datetime as a compact past- or future-relative label."""
    timestamp = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    current = now or datetime.now(timezone.utc)
    current = (
        current.astimezone(timezone.utc)
        if current.tzinfo
        else current.replace(tzinfo=timezone.utc)
    )
    delta_seconds = int(round((current - timestamp).total_seconds()))
    if abs(delta_seconds) < 60:
        return "just now"

    seconds = abs(delta_seconds)
    for unit_seconds, unit_name in (
        (30 * 86400, "month"),
        (7 * 86400, "week"),
        (86400, "day"),
        (3600, "hour"),
        (60, "minute"),
    ):
        if seconds >= unit_seconds:
            amount = max(1, seconds // unit_seconds)
            label = f"{amount} {unit_name}{'' if amount == 1 else 's'}"
            return f"in {label}" if delta_seconds < 0 else f"{label} ago"
    return "just now"


def _format_time_minimal(epoch_seconds: float) -> str:
    """Format epoch seconds as a minute-resolution RFC 3339 UTC string."""
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return dt.strftime(TIME_DISPLAY_FORMAT)


def _format_time_minimal_local(epoch_seconds: float) -> str:
    """Format epoch seconds in client-local time with an explicit offset."""
    try:
        tz = _resolve_client_tz()
        dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).astimezone(tz)
        return _format_datetime_minute_explicit(dt)
    except Exception:
        return _format_time_minimal(epoch_seconds)


def _format_time_explicit(epoch_seconds: float) -> str:
    """Format UTC epoch seconds with an embedded timezone marker."""
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return _format_datetime_minute_explicit(dt)


def _format_time_explicit_local(epoch_seconds: float) -> str:
    """Format epoch seconds in local/client time with an embedded offset."""
    try:
        tz = _resolve_client_tz()
        dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).astimezone(tz)
        return _format_datetime_minute_explicit(dt)
    except Exception:
        return _format_time_explicit(epoch_seconds)


def _format_time_second_explicit(epoch_seconds: float) -> str:
    """Format UTC epoch seconds at quote/event precision."""
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return _format_datetime_second_explicit(dt)


def _format_time_second_explicit_local(epoch_seconds: float) -> str:
    """Format local/client epoch seconds at quote/event precision."""
    try:
        tz = _resolve_client_tz()
        dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).astimezone(tz)
        return _format_datetime_second_explicit(dt)
    except Exception:
        return _format_time_second_explicit(epoch_seconds)


def _format_datetime_minute_explicit(dt: datetime) -> str:
    return _format_datetime_explicit(dt, timespec="minutes")


def _format_datetime_second_explicit(dt: datetime) -> str:
    return _format_datetime_explicit(dt, timespec="seconds")


def _format_datetime_explicit(dt: datetime, *, timespec: str) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    text = dt.isoformat(timespec=timespec)
    return f"{text[:-6]}Z" if text.endswith("+00:00") else text


def _use_client_tz() -> bool:
    """Return True when a client timezone is configured."""
    return _resolve_client_tz() is not None


def _resolve_client_tz():
    """Return the configured client timezone, if any."""
    from ..bootstrap.settings import mt5_config

    try:
        return mt5_config.get_client_tz()
    except Exception:
        return None
