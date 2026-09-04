"""Canonical time formatting and client-timezone helpers."""

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from ..shared.constants import CALENDAR_TIMEFRAMES, TIMEFRAME_SECONDS

# Broker history lookbacks longer than this overflow datetime arithmetic or
# exceed any useful MT5 archive. 20 years is 10_512_000 minutes.
MAX_TRADING_MINUTES_BACK = 20 * 365 * 24 * 60


def bar_close_epoch(open_epoch: Any, timeframe: str) -> float:
    """Return the UTC end epoch for a bar opened at *open_epoch*.

    Daily, weekly, and monthly bars close on broker-calendar boundaries so a
    configured broker timezone remains correct across daylight-saving changes.
    """
    opened = float(open_epoch)
    normalized_timeframe = str(timeframe).upper()
    if normalized_timeframe in CALENDAR_TIMEFRAMES:
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


def format_epoch_utc(value: Any, *, timespec: str = "seconds") -> Optional[str]:
    """Format epoch seconds as RFC 3339 UTC at the requested precision."""
    try:
        timestamp = float(value)
        return format_datetime_utc(
            datetime.fromtimestamp(timestamp, timezone.utc),
            timespec=timespec,
        )
    except (OSError, OverflowError, TypeError, ValueError):
        return None


def as_utc(value: datetime) -> datetime:
    """Convert a datetime to UTC, treating a naive value as already UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_iso_utc(value: Any) -> datetime:
    """Parse an ISO datetime and normalize it to UTC.

    Python's ISO parser accepts the RFC 3339 ``Z`` suffix. Naive values are
    treated as UTC, matching the repository's existing timestamp contracts.
    """
    if isinstance(value, datetime):
        return as_utc(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a non-empty ISO datetime string.")
    return as_utc(datetime.fromisoformat(value.strip()))


def format_datetime_utc(value: datetime, *, timespec: str = "seconds") -> str:
    """Format a datetime as RFC 3339 UTC, treating naive values as UTC."""
    resolved = as_utc(value)
    return resolved.isoformat(timespec=timespec).replace("+00:00", "Z")


def parse_relative_time(value: str, *, now: Optional[datetime] = None) -> Optional[datetime]:
    """Parse strings like ``5 minutes ago`` into an aware UTC datetime."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    current = now or datetime.now(timezone.utc)
    current = (
        current.astimezone(timezone.utc)
        if current.tzinfo
        else current.replace(tzinfo=timezone.utc)
    )
    if text == "just now":
        return current
    if text == "yesterday":
        return current - timedelta(days=1)
    if text.startswith("-"):
        return None
    match = re.fullmatch(
        r"(\d+)\s+(minute|hour|day|week|month)s?\s+ago",
        text,
    )
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    try:
        if unit == "minute":
            return current - timedelta(minutes=amount)
        if unit == "hour":
            return current - timedelta(hours=amount)
        if unit == "day":
            return current - timedelta(days=amount)
        if unit == "week":
            return current - timedelta(weeks=amount)
        return current - timedelta(days=30 * amount)
    except OverflowError:
        return None


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


def format_relative_date(
    value: date | datetime,
    *,
    now: Optional[datetime] = None,
    tz: Optional[ZoneInfo] = None,
) -> str:
    """Format a calendar date as today/tomorrow/yesterday, not a clock countdown."""
    calendar_tz = tz or ZoneInfo("America/New_York")
    if isinstance(value, datetime):
        stamp = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        event_date = stamp.astimezone(calendar_tz).date()
    else:
        event_date = value
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    today = current.astimezone(calendar_tz).date()
    delta_days = (event_date - today).days
    if delta_days == 0:
        return "today"
    if delta_days == 1:
        return "tomorrow"
    if delta_days == -1:
        return "yesterday"
    if delta_days > 1:
        return f"in {delta_days} days"
    return f"{abs(delta_days)} days ago"


def _format_time_minimal(epoch_seconds: float) -> str:
    """Format epoch seconds as a minute-resolution RFC 3339 UTC string."""
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return format_datetime_utc(dt, timespec="minutes")


def _format_time_minimal_local(epoch_seconds: float) -> str:
    """Format epoch seconds in client-local time with an explicit offset."""
    try:
        tz = _resolve_client_tz()
        dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).astimezone(tz)
        return _format_datetime_minute_explicit(dt)
    except Exception:
        return _format_time_minimal(epoch_seconds)


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


def _timezone_uses_zulu_suffix(tzinfo: Any) -> bool:
    """Keep ``Z`` for UTC; preserve named-zone offsets even at GMT."""
    if tzinfo is None or tzinfo is timezone.utc:
        return True
    key = str(getattr(tzinfo, "key", "") or "").upper()
    if key in {"UTC", "ETC/UTC"}:
        return True
    if key:
        return False
    try:
        return tzinfo.utcoffset(None) == timedelta(0)
    except Exception:
        return False


def _format_datetime_explicit(dt: datetime, *, timespec: str) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    text = dt.isoformat(timespec=timespec)
    if text.endswith("+00:00") and _timezone_uses_zulu_suffix(dt.tzinfo):
        return f"{text[:-6]}Z"
    return text


def coerce_time_epoch_seconds(values: Any) -> Any:
    """Convert datetime or numeric time values to UTC epoch seconds.

    Numeric magnitudes above ``1e16`` are treated as nanoseconds and above
    ``1e12`` as milliseconds. Returns an empty float array when conversion
    fails.
    """
    import numpy as np
    import pandas as pd

    try:
        series = values if isinstance(values, pd.Series) else pd.Series(values)
    except Exception:
        return np.asarray([], dtype=float)
    if pd.api.types.is_datetime64_any_dtype(series) or isinstance(
        getattr(series, "dtype", None), pd.DatetimeTZDtype
    ):
        try:
            dt = pd.to_datetime(series, utc=True, errors="coerce")
            ns = dt.astype("int64", copy=False).to_numpy(dtype=float)
            seconds = ns / 1e9
            valid = dt.notna().to_numpy()
            return np.asarray(np.where(valid, seconds, np.nan), dtype=float)
        except Exception:
            return np.asarray([], dtype=float)
    try:
        times = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    except (TypeError, ValueError):
        try:
            dt = pd.to_datetime(series, utc=True, errors="coerce")
            ns = dt.astype("int64", copy=False).to_numpy(dtype=float)
            return np.asarray(ns / 1e9, dtype=float)
        except Exception:
            return np.asarray([], dtype=float)
    finite = times[np.isfinite(times)]
    if finite.size == 0:
        return times
    typical = float(np.nanmedian(np.abs(finite)))
    if typical > 1e16:
        times = times / 1e9
    elif typical > 1e12:
        times = times / 1e3
    return times


def timezone_label(tz: Any, default: str = "UTC") -> str:
    if tz is None:
        return default
    name = getattr(tz, "key", None) or getattr(tz, "zone", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    try:
        text = str(tz).strip()
    except Exception:
        return default
    return text or default


def display_timezone_label(
    *,
    use_client_tz: bool,
    fallback: str = "client_local",
    resolve_client_tz: Any = None,
    client_tz: Any = None,
) -> str:
    """Return a display label for UTC or the resolved client timezone."""
    if not use_client_tz:
        return "UTC"
    try:
        resolved = client_tz
        if resolved is None:
            if resolve_client_tz is None:
                resolve_client_tz = _resolve_client_tz
            resolved = resolve_client_tz()
        return timezone_label(resolved, default=fallback)
    except Exception:
        return fallback


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
