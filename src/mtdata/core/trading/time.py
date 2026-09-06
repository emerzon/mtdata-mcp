"""Trading expiration and broker-time normalization helpers."""

from __future__ import annotations

import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Union

from ...bootstrap.settings import mt5_config
from ...shared.constants import TIMEFRAME_SECONDS
from ...shared.validators import unsupported_timeframe_seconds_error
from ...utils.freshness import closed_session_context
from ...utils.time import format_datetime_utc, format_epoch_utc

ExpirationValue = Union[int, float, str, datetime]
_GTC_EXPIRATION_TOKENS = {"GTC"}
_SIMPLE_RELATIVE_PATTERN = re.compile(r"^(?:in\s+)?(\d+(?:\.\d+)?)\s*([a-zA-Z]+)$", re.IGNORECASE)


class PendingExpirationValidationError(ValueError):
    """Stable validation failure for an explicit pending-order expiration."""

    error_code = "invalid_pending_expiration"

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        expiration: object,
        resolved_epoch: Optional[int] = None,
        observed_epoch: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.context = {
            "reason": reason,
            "expiration_received": str(expiration),
        }
        if resolved_epoch is not None:
            self.context["expiration_resolved_utc"] = _format_expiration_utc(
                resolved_epoch
            )
        if observed_epoch is not None:
            self.context["validation_observed_utc"] = _format_expiration_utc(
                observed_epoch
            )


def _format_expiration_utc(epoch_seconds: int) -> str:
    return format_epoch_utc(epoch_seconds) or ""


def _validate_pending_expiration_timestamp(
    timestamp: int,
    *,
    expiration: object,
) -> int:
    observed_epoch = int(datetime.now(timezone.utc).timestamp())
    if timestamp <= observed_epoch:
        resolved_utc = _format_expiration_utc(timestamp)
        observed_utc = _format_expiration_utc(observed_epoch)
        raise PendingExpirationValidationError(
            (
                "expiration must resolve to a future UTC instant; "
                f"resolved_utc={resolved_utc}, validation_observed_utc={observed_utc}"
            ),
            reason="not_in_future",
            expiration=expiration,
            resolved_epoch=timestamp,
            observed_epoch=observed_epoch,
        )
    return timestamp


def _invalid_pending_expiration(
    expiration: object,
    *,
    reason: str,
    message: str,
) -> PendingExpirationValidationError:
    return PendingExpirationValidationError(
        message,
        reason=reason,
        expiration=expiration,
    )


def _validate_expiration_local_time(
    value: datetime,
    *,
    expiration: object,
) -> None:
    """Reject naive client-local wall times that are ambiguous or nonexistent."""
    if value.tzinfo is not None and value.utcoffset() is not None:
        return
    try:
        client_tz = mt5_config.get_client_tz()
    except Exception:
        client_tz = None
    if client_tz is None:
        return

    candidates: set[datetime] = set()
    for fold in (0, 1):
        try:
            aware = value.replace(tzinfo=client_tz, fold=fold)
            utc_value = aware.astimezone(timezone.utc)
            roundtrip = utc_value.astimezone(client_tz)
        except Exception:
            continue
        if roundtrip.replace(tzinfo=None) == value:
            candidates.add(utc_value)

    timezone_name = str(client_tz)
    remediation = "Pass an ISO-8601 expiration with Z or an explicit numeric UTC offset."
    if not candidates:
        raise _invalid_pending_expiration(
            expiration,
            reason="nonexistent_local_time",
            message=(
                f"expiration {value.isoformat(sep=' ')} does not exist in client "
                f"timezone {timezone_name} because of a daylight-saving transition. "
                f"{remediation}"
            ),
        )
    if len(candidates) > 1:
        raise _invalid_pending_expiration(
            expiration,
            reason="ambiguous_local_time",
            message=(
                f"expiration {value.isoformat(sep=' ')} occurs twice in client "
                f"timezone {timezone_name} because of a daylight-saving transition. "
                f"{remediation}"
            ),
        )


def _expiration_to_server_time_naive(
    value: datetime,
    *,
    expiration: object,
) -> datetime:
    _validate_expiration_local_time(value, expiration=expiration)
    return _to_server_time_naive(value)


def _to_server_time_naive(dt: datetime) -> datetime:
    """Convert a datetime into broker/server-local naive time."""
    try:
        server_tz = mt5_config.get_server_tz()
        client_tz = mt5_config.get_client_tz()
    except Exception:
        server_tz = None
        client_tz = None

    aware = dt
    try:
        if dt.tzinfo is None:
            if client_tz is not None:
                aware = dt.replace(tzinfo=client_tz)
            else:
                aware = dt.replace(tzinfo=timezone.utc)
    except Exception:
        aware = dt.replace(tzinfo=timezone.utc)

    if server_tz is not None:
        try:
            server_aware = aware.astimezone(server_tz)
            return server_aware.replace(tzinfo=None)
        except Exception:
            pass

    try:
        offset_sec = int(mt5_config.get_time_offset_seconds())
    except Exception:
        offset_sec = 0
    try:
        utc_dt = aware.astimezone(timezone.utc)
    except Exception:
        utc_dt = aware if aware.tzinfo is not None else aware.replace(tzinfo=timezone.utc)
    server_dt = utc_dt + timedelta(seconds=offset_sec)
    return server_dt.replace(tzinfo=None)


def _server_time_naive_to_mt5_timestamp(dt: datetime) -> int:
    """Convert a server-local naive datetime into an MT5-compatible timestamp."""
    utc_dt = _server_time_naive_to_utc(dt.replace(microsecond=0))
    return int(utc_dt.timestamp())


def _server_time_naive_to_utc(dt: datetime) -> datetime:
    """Convert a server-local naive datetime into UTC."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc)

    try:
        server_tz = mt5_config.get_server_tz()
    except Exception:
        server_tz = None

    if server_tz is not None:
        return dt.replace(tzinfo=server_tz).astimezone(timezone.utc)

    try:
        offset_sec = int(mt5_config.get_time_offset_seconds())
    except Exception:
        offset_sec = 0
    return (dt - timedelta(seconds=offset_sec)).replace(tzinfo=timezone.utc)


def _next_candle_close_utc(
    timeframe: str,
    *,
    now_utc: Optional[datetime] = None,
    symbol: Optional[str] = None,
) -> datetime:
    """Return the next real candle-close instant on the UTC timeline."""
    tf = str(timeframe or "").upper().strip()
    if tf not in TIMEFRAME_SECONDS:
        valid = ", ".join(sorted(TIMEFRAME_SECONDS.keys()))
        raise ValueError(f"Invalid timeframe '{timeframe}'. Valid options: {valid}")

    current_utc = now_utc or datetime.now(timezone.utc)
    if current_utc.tzinfo is None:
        current_utc = current_utc.replace(tzinfo=timezone.utc)
    else:
        current_utc = current_utc.astimezone(timezone.utc)

    if tf not in {"D1", "W1", "MN1"}:
        interval_seconds = int(TIMEFRAME_SECONDS[tf])
        if interval_seconds <= 0:
            raise ValueError(unsupported_timeframe_seconds_error(tf))
        current_epoch = current_utc.timestamp()
        next_epoch = (
            math.floor(current_epoch / float(interval_seconds)) + 1
        ) * interval_seconds
        next_utc = datetime.fromtimestamp(next_epoch, tz=timezone.utc)
    else:
        server_now = _to_server_time_naive(current_utc).replace(tzinfo=None)
        if tf == "MN1":
            if server_now.month == 12:
                result = server_now.replace(year=server_now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                result = server_now.replace(month=server_now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
        elif tf == "W1":
            days_until_next_monday = (7 - server_now.weekday()) % 7
            if days_until_next_monday == 0:
                days_until_next_monday = 7
            result = (server_now + timedelta(days=days_until_next_monday)).replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        else:
            result = (server_now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        next_utc = _server_time_naive_to_utc(result)
    if symbol:
        closed = closed_session_context(
            symbol,
            now_epoch=current_utc.timestamp(),
            item="bar",
            data_age_seconds=max(0.0, (next_utc - current_utc).total_seconds()),
        )
        if closed and closed.get("assumed_closure_end"):
            try:
                closure_end = datetime.fromisoformat(str(closed["assumed_closure_end"]))
            except ValueError:
                closure_end = None
            if closure_end is not None:
                if closure_end.tzinfo is None:
                    closure_end = closure_end.replace(tzinfo=timezone.utc)
                else:
                    closure_end = closure_end.astimezone(timezone.utc)
            if closure_end is not None and next_utc <= closure_end:
                return _next_candle_close_utc(
                    timeframe,
                    now_utc=closure_end,
                    symbol=None,
                )
    return next_utc


def _next_candle_close_server_time(
    timeframe: str,
    *,
    now_utc: Optional[datetime] = None,
    symbol: Optional[str] = None,
) -> datetime:
    """Return the next candle close in server-local naive time."""
    next_utc = _next_candle_close_utc(
        timeframe,
        now_utc=now_utc,
        symbol=symbol,
    )
    try:
        server_tz = mt5_config.get_server_tz()
    except Exception:
        server_tz = None
    if server_tz is not None:
        return next_utc.astimezone(server_tz).replace(tzinfo=None)
    try:
        offset_seconds = int(mt5_config.get_time_offset_seconds())
    except Exception:
        offset_seconds = 0
    return (next_utc + timedelta(seconds=offset_seconds)).replace(tzinfo=None)


def _format_utc_offset(offset_seconds: int) -> str:
    sign = "+" if offset_seconds >= 0 else "-"
    total_seconds = abs(int(offset_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    return f"{sign}{hours:02d}:{minutes:02d}"


def _next_candle_wait_payload(
    timeframe: str,
    *,
    buffer_seconds: float = 1.0,
    now_utc: Optional[datetime] = None,
    symbol: Optional[str] = None,
) -> dict:
    """Build timing metadata for the next candle close without sleeping."""
    current_utc = now_utc or datetime.now(timezone.utc)
    if current_utc.tzinfo is None:
        current_utc = current_utc.replace(tzinfo=timezone.utc)
    else:
        current_utc = current_utc.astimezone(timezone.utc)

    next_close_utc = _next_candle_close_utc(
        timeframe,
        now_utc=current_utc,
        symbol=symbol,
    )
    try:
        server_tz = mt5_config.get_server_tz()
    except Exception:
        server_tz = None
    if server_tz is not None:
        server_close_aware = next_close_utc.astimezone(server_tz)
        offset = server_close_aware.utcoffset()
        server_offset_seconds = int(offset.total_seconds()) if offset else 0
    else:
        try:
            server_offset_seconds = int(
                mt5_config.get_time_offset_seconds(at_time=next_close_utc)
            )
        except TypeError:
            server_offset_seconds = int(mt5_config.get_time_offset_seconds())
        except Exception:
            server_offset_seconds = 0
        server_close_aware = next_close_utc.astimezone(
            timezone(timedelta(seconds=server_offset_seconds))
        )
    server_utc_offset = _format_utc_offset(server_offset_seconds)
    wait_seconds = max(
        0.0,
        float((next_close_utc - current_utc).total_seconds()) + max(0.0, float(buffer_seconds)),
    )

    try:
        server_tz_name = getattr(mt5_config, "server_tz_name", None) or f"UTC{int(mt5_config.get_time_offset_seconds()) / 3600:+g}"
    except Exception:
        server_tz_name = "UTC"

    payload = {
        "timeframe": str(timeframe).upper().strip(),
        "buffer_seconds": float(buffer_seconds),
        "sleep_seconds": float(wait_seconds),
        "started_at_utc": format_datetime_utc(current_utc, timespec="auto"),
        "next_candle_close_utc": format_datetime_utc(next_close_utc, timespec="auto"),
        "next_candle_close_server": server_close_aware.isoformat(),
        "server_timezone": str(server_tz_name),
        "server_utc_offset": server_utc_offset,
    }
    closed = None
    if symbol:
        closed = closed_session_context(
            symbol,
            now_epoch=current_utc.timestamp(),
            item="bar",
            data_age_seconds=max(0.0, (next_close_utc - current_utc).total_seconds()),
        )
    if closed:
        payload.update(
            {
                key: closed[key]
                for key in (
                    "market_status",
                    "market_status_reason",
                    "assumed_closure_start",
                    "assumed_closure_end",
                )
                if key in closed
            }
        )
    return payload


def _sleep_until_next_candle(
    timeframe: str,
    *,
    buffer_seconds: float = 1.0,
    sleep_impl=time.sleep,
    now_utc: Optional[datetime] = None,
    symbol: Optional[str] = None,
) -> dict:
    """Sleep until the next candle closes and return timing metadata."""
    payload = _next_candle_wait_payload(
        timeframe,
        buffer_seconds=buffer_seconds,
        now_utc=now_utc,
        symbol=symbol,
    )
    sleep_seconds = float(payload.get("sleep_seconds", 0.0) or 0.0)
    sleep_impl(sleep_seconds)
    payload["status"] = "completed"
    payload["slept"] = True
    payload["slept_seconds"] = sleep_seconds
    payload["remaining_seconds"] = 0.0
    return payload


def _relative_expiration_base(*, now_utc: Optional[datetime] = None) -> datetime:
    """Return a naive dateparser base in the configured client timezone."""
    current_utc = now_utc or datetime.now(timezone.utc)
    if current_utc.tzinfo is None:
        current_utc = current_utc.replace(tzinfo=timezone.utc)
    else:
        current_utc = current_utc.astimezone(timezone.utc)
    try:
        client_tz = mt5_config.get_client_tz()
    except Exception:
        client_tz = None
    if client_tz is not None:
        try:
            return current_utc.astimezone(client_tz).replace(tzinfo=None)
        except Exception:
            pass
    return current_utc.replace(tzinfo=None)


def _normalize_pending_expiration(  # noqa: C901
    expiration: Optional[ExpirationValue],
) -> Tuple[Optional[int], bool]:
    """Convert user-supplied expiration data into an MT5-compatible timestamp."""
    if expiration is None:
        return None, False

    if isinstance(expiration, datetime):
        server_dt = _expiration_to_server_time_naive(
            expiration,
            expiration=expiration,
        )
        timestamp = _server_time_naive_to_mt5_timestamp(server_dt)
        return _validate_pending_expiration_timestamp(
            timestamp,
            expiration=expiration,
        ), True

    if isinstance(expiration, (int, float)):
        if not math.isfinite(expiration) or expiration <= 0:
            raise _invalid_pending_expiration(
                expiration,
                reason="nonpositive_or_nonfinite",
                message="expiration must be a positive finite UTC epoch timestamp or the GTC token.",
            )
        try:
            server_dt = _to_server_time_naive(datetime.fromtimestamp(expiration, tz=timezone.utc))
            timestamp = _server_time_naive_to_mt5_timestamp(server_dt)
            return _validate_pending_expiration_timestamp(
                timestamp,
                expiration=expiration,
            ), True
        except (OverflowError, OSError) as exc:
            raise _invalid_pending_expiration(
                expiration,
                reason="timestamp_out_of_range",
                message=f"Expiration timestamp out of range: {expiration}",
            ) from exc

    if isinstance(expiration, str):
        cleaned = expiration.strip().strip('"').strip("'")
        if cleaned == "":
            return None, False

        upper_cleaned = cleaned.upper()
        if upper_cleaned in _GTC_EXPIRATION_TOKENS:
            return None, True

        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned):
            try:
                end_of_client_day = datetime.fromisoformat(cleaned).replace(
                    hour=23,
                    minute=59,
                    second=59,
                    microsecond=999999,
                )
            except ValueError as exc:
                raise _invalid_pending_expiration(
                    expiration,
                    reason="unsupported_format",
                    message=f"Unsupported expiration format: {expiration}",
                ) from exc
            server_dt = _expiration_to_server_time_naive(
                end_of_client_day,
                expiration=expiration,
            )
            timestamp = _server_time_naive_to_mt5_timestamp(server_dt)
            return _validate_pending_expiration_timestamp(
                timestamp,
                expiration=expiration,
            ), True

        match = _SIMPLE_RELATIVE_PATTERN.match(cleaned)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            delta = None
            if unit in ("s", "sec", "secs", "second", "seconds"):
                delta = timedelta(seconds=value)
            elif unit in ("m", "min", "mins", "minute", "minutes"):
                delta = timedelta(minutes=value)
            elif unit in ("h", "hr", "hrs", "hour", "hours"):
                delta = timedelta(hours=value)
            elif unit in ("d", "day", "days"):
                delta = timedelta(days=value)
            elif unit in ("w", "wk", "weeks"):
                delta = timedelta(weeks=value)
            if delta is not None:
                server_dt = _to_server_time_naive(datetime.now(timezone.utc) + delta)
                timestamp = _server_time_naive_to_mt5_timestamp(server_dt)
                return _validate_pending_expiration_timestamp(
                    timestamp,
                    expiration=expiration,
                ), True

        try:
            numeric = float(cleaned)
        except ValueError:
            numeric = None
        if numeric is not None:
            if not math.isfinite(numeric) or numeric <= 0:
                raise _invalid_pending_expiration(
                    expiration,
                    reason="nonpositive_or_nonfinite",
                    message="expiration must be a positive finite UTC epoch timestamp or the GTC token.",
                )
            try:
                server_dt = _to_server_time_naive(
                    datetime.fromtimestamp(numeric, tz=timezone.utc)
                )
                timestamp = _server_time_naive_to_mt5_timestamp(server_dt)
                return _validate_pending_expiration_timestamp(
                    timestamp,
                    expiration=expiration,
                ), True
            except (OverflowError, OSError) as exc:
                raise _invalid_pending_expiration(
                    expiration,
                    reason="timestamp_out_of_range",
                    message=f"Expiration timestamp out of range: {expiration}",
                ) from exc

        try:
            iso_datetime = datetime.fromisoformat(cleaned)
        except ValueError:
            iso_datetime = None
        if iso_datetime is not None:
            server_dt = _expiration_to_server_time_naive(
                iso_datetime,
                expiration=expiration,
            )
            timestamp = _server_time_naive_to_mt5_timestamp(server_dt)
            return _validate_pending_expiration_timestamp(
                timestamp,
                expiration=expiration,
            ), True

        try:
            import dateparser  # type: ignore
        except Exception:
            dateparser = None
        if dateparser is not None:
            try:
                dt = dateparser.parse(
                    cleaned,
                    settings={
                        "RETURN_AS_TIMEZONE_AWARE": False,
                        "PREFER_DATES_FROM": "future",
                        "RELATIVE_BASE": _relative_expiration_base(),
                    },
                )
            except Exception:
                dt = None
            if dt is not None:
                server_dt = _expiration_to_server_time_naive(
                    dt,
                    expiration=expiration,
                )
                timestamp = _server_time_naive_to_mt5_timestamp(server_dt)
                return _validate_pending_expiration_timestamp(
                    timestamp,
                    expiration=expiration,
                ), True

        raise _invalid_pending_expiration(
            expiration,
            reason="unsupported_format",
            message=f"Unsupported expiration format: {expiration}",
        )

    raise TypeError(f"Unsupported expiration type: {type(expiration).__name__}")
