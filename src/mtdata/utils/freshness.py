from __future__ import annotations

import math
import re
from datetime import datetime, time, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from ..shared.constants import SANITY_BARS_TOLERANCE, TIMEFRAME_SECONDS
from ..shared.market_sessions import (
    MARKET_SESSIONS,
    exchange_holidays,
    is_early_close_session,
)
from ..shared.symbols import (
    EQUITY_BROKER_SUFFIXES,
    is_probably_crypto_symbol,
    is_probably_fx_session_symbol,
)
from .time import bar_close_epoch, format_epoch_utc

# Keep market-data readiness aligned with the pre-trade validator so the same
# quote age is not rejected by one public surface and accepted by another.
QUOTE_LIVE_SECONDS = 10
QUOTE_RECENT_SECONDS = 60
QUOTE_STALE_SECONDS = 300
TIMESTAMP_FUTURE_TOLERANCE_SECONDS = 10
_NEW_YORK = ZoneInfo("America/New_York")
_EQUITY_VENUE_SUFFIXES = {
    "AMEX": "NYSE",
    "ARCA": "NYSE",
    "ASE": "NYSE",
    "BATS": "NYSE",
    "L": "LSE",
    "NAS": "NASDAQ",
    "NASDAQ": "NASDAQ",
    "NQ": "NASDAQ",
    "NY": "NYSE",
    "NYSE": "NYSE",
    "NYS": "NYSE",
    "NYQ": "NYSE",
    "O": "NASDAQ",
    "US": "NYSE",
}
_VENUE_TEXT_HINTS = {
    "NASDAQ": "NASDAQ",
    "NYSE": "NYSE",
    "NEW YORK STOCK EXCHANGE": "NYSE",
    "LONDON STOCK EXCHANGE": "LSE",
    "LSE": "LSE",
    "XETRA": "XETRA",
    "EURONEXT": "EURONEXT",
    "TOKYO STOCK EXCHANGE": "TSE",
    "TSE": "TSE",
    "HONG KONG": "HKEX",
    "HKEX": "HKEX",
    "SHANGHAI": "SSE",
    "SSE": "SSE",
    "AUSTRALIAN SECURITIES": "ASX",
    "ASX": "ASX",
}

COMPLETED_BAR_FRESHNESS_KEYS = (
    "data_as_of",
    "data_as_of_epoch",
    "data_age_seconds",
    "data_stale",
    "stale_after_seconds",
    "freshness_basis",
    "freshness_age_metric",
    "history_policy_ok",
    "freshness",
    "timestamp_ahead_of_wall_clock",
    "timestamp_in_future",
    "timestamp_skew_seconds",
    "timestamp_skew_tolerance_seconds",
    "timestamp_skew_basis",
    "timestamp_warning",
    "market_status",
    "market_status_reason",
    "market_status_source",
    "market_venue",
    "session_calendar",
    "holiday",
    "freshness_policy_relaxed",
    "assumed_closure_start",
    "assumed_closure_end",
    "assumed_closure_seconds",
    "note",
    "stale_warning",
)


def standard_weekend_window(now_utc: datetime) -> Optional[tuple[datetime, datetime]]:
    """Return the DST-aware FX weekend window containing ``now_utc``."""
    utc_value = now_utc if now_utc.tzinfo else now_utc.replace(tzinfo=timezone.utc)
    new_york = utc_value.astimezone(_NEW_YORK)
    days_since_friday = (new_york.weekday() - 4) % 7
    friday = new_york.date() - timedelta(days=days_since_friday)
    close_local = datetime.combine(friday, time(17), tzinfo=_NEW_YORK)
    open_local = datetime.combine(friday + timedelta(days=2), time(17), tzinfo=_NEW_YORK)
    close_utc = close_local.astimezone(timezone.utc)
    open_utc = open_local.astimezone(timezone.utc)
    if close_utc <= utc_value < open_utc:
        return close_utc, open_utc
    return None


def is_standard_weekend_closure(now_utc: datetime) -> bool:
    return standard_weekend_window(now_utc) is not None


def _friday_session_close_on_or_before(utc_value: datetime) -> datetime:
    """Return the NY 17:00 Friday close at or before *utc_value*."""
    new_york = utc_value.astimezone(_NEW_YORK)
    days_since_friday = (new_york.weekday() - 4) % 7
    friday = new_york.date() - timedelta(days=days_since_friday)
    close_local = datetime.combine(friday, time(17), tzinfo=_NEW_YORK)
    close_utc = close_local.astimezone(timezone.utc)
    if close_utc > utc_value:
        previous_friday = friday - timedelta(days=7)
        close_local = datetime.combine(previous_friday, time(17), tzinfo=_NEW_YORK)
        close_utc = close_local.astimezone(timezone.utc)
    return close_utc


def standard_weekend_overlap_seconds(start_epoch: float, end_epoch: float) -> float:
    """Return seconds of ``[start_epoch, end_epoch]`` that fall in FX weekend closures."""
    try:
        start_value = float(start_epoch)
        end_value = float(end_epoch)
    except (TypeError, ValueError):
        return 0.0
    if not (math.isfinite(start_value) and math.isfinite(end_value)):
        return 0.0
    if end_value <= start_value:
        return 0.0
    start_dt = datetime.fromtimestamp(start_value, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_value, tz=timezone.utc)
    overlap = 0.0
    close_utc = _friday_session_close_on_or_before(start_dt)
    for _ in range(8):
        close_local = close_utc.astimezone(_NEW_YORK)
        open_local = datetime.combine(
            close_local.date() + timedelta(days=2),
            time(17),
            tzinfo=_NEW_YORK,
        )
        open_utc = open_local.astimezone(timezone.utc)
        overlap_start = max(start_dt, close_utc)
        overlap_end = min(end_dt, open_utc)
        if overlap_end > overlap_start:
            overlap += (overlap_end - overlap_start).total_seconds()
        next_friday = close_local.date() + timedelta(days=7)
        close_utc = datetime.combine(
            next_friday, time(17), tzinfo=_NEW_YORK
        ).astimezone(timezone.utc)
        if close_utc >= end_dt:
            break
    return overlap


def _symbol_info_text(symbol_info: Any, field: str) -> str:
    try:
        value = (
            symbol_info.get(field)
            if isinstance(symbol_info, dict)
            else getattr(symbol_info, field, None)
        )
    except Exception:
        value = None
    return str(value or "").strip()


def _equity_venue_for_symbol(
    symbol: Any,
    *,
    symbol_info: Any = None,
) -> Optional[str]:
    metadata_text = " ".join(
        (
            _symbol_info_text(symbol_info, "path"),
            _symbol_info_text(symbol_info, "description"),
            _symbol_info_text(symbol_info, "exchange"),
        )
    ).upper()
    for hint, venue in _VENUE_TEXT_HINTS.items():
        if hint in metadata_text:
            return venue

    normalized = re.sub(
        r"(?:[._-]24)$",
        "",
        str(symbol or "").strip().upper(),
    )
    suffix_match = re.fullmatch(r".+[._-]([A-Z0-9]+)", normalized)
    if suffix_match is not None:
        suffix = suffix_match.group(1)
        venue = _EQUITY_VENUE_SUFFIXES.get(suffix)
        if venue:
            return venue
        if suffix in EQUITY_BROKER_SUFFIXES:
            return "NYSE"

    explicit_equity = any(
        hint in metadata_text
        for hint in ("STOCK", "EQUITY", "SHARE", "ETF")
    )
    if explicit_equity or re.fullmatch(r"[A-Z]{1,5}(?:[./-][A-Z])?", normalized):
        # A bare ticker cannot identify its listing venue. XNYS supplies the
        # shared US cash-session and holiday calendar used by both US venues.
        return "NYSE"
    return None


def _symbol_session_profile(
    symbol: Any,
    *,
    symbol_info: Any = None,
) -> tuple[str, Optional[str], Optional[dict[str, Any]]]:
    path = _symbol_info_text(symbol_info, "path")
    description = _symbol_info_text(symbol_info, "description")
    metadata = f"{path} {description}".lower()
    if is_probably_crypto_symbol(symbol) or "crypto" in metadata:
        return "continuous_24_7", None, None
    if is_probably_fx_session_symbol(symbol, path=path) or any(
        hint in metadata
        for hint in ("forex", "foreign exchange", "metal", "commodity", "index")
    ):
        return "fx", None, None
    venue = _equity_venue_for_symbol(symbol, symbol_info=symbol_info)
    if venue and venue in MARKET_SESSIONS:
        return "equity", venue, MARKET_SESSIONS[venue]
    # Preserve the established near-24/5 fallback for unclassified broker CFDs.
    return "fx", None, None


def _exchange_holiday(
    market: dict[str, Any],
    session_value: Any,
) -> tuple[bool, Optional[str]]:
    exchange = str(market.get("exchange_calendar") or "").strip()
    if not exchange:
        return False, None
    session_date = (
        session_value.date()
        if isinstance(session_value, datetime)
        else session_value
    )
    try:
        calendar = exchange_holidays(exchange, int(session_date.year))
        if session_date in calendar:
            return True, str(calendar[session_date])
    except Exception:
        return False, None
    return False, None


def _market_active_intervals(
    market: dict[str, Any],
    session_date: Any,
) -> list[tuple[datetime, datetime]]:
    if session_date.weekday() >= 5:
        return []
    is_holiday, _ = _exchange_holiday(market, session_date)
    if is_holiday:
        return []

    market_tz = ZoneInfo(str(market["timezone"]))
    session_dt = datetime.combine(session_date, time(12), tzinfo=market_tz)
    early_close = is_early_close_session(
        market,
        str(market.get("country") or ""),
        session_dt,
        holiday_resolver=lambda _country, value, _exchange=None: (
            _exchange_holiday(market, value)
        ),
    )
    open_hour, open_minute = market["open"]
    close_value = (
        market.get("early_close")
        if early_close and market.get("early_close")
        else market["close"]
    )
    close_hour, close_minute = close_value
    opened = datetime.combine(
        session_date,
        time(int(open_hour), int(open_minute)),
        tzinfo=market_tz,
    )
    closed = datetime.combine(
        session_date,
        time(int(close_hour), int(close_minute)),
        tzinfo=market_tz,
    )
    if closed <= opened:
        return []

    lunch_start = market.get("lunch_start")
    lunch_end = market.get("lunch_end")
    if lunch_start and lunch_end:
        pause = datetime.combine(
            session_date,
            time(int(lunch_start[0]), int(lunch_start[1])),
            tzinfo=market_tz,
        )
        resume = datetime.combine(
            session_date,
            time(int(lunch_end[0]), int(lunch_end[1])),
            tzinfo=market_tz,
        )
        if opened < pause < resume < closed:
            return [(opened, pause), (resume, closed)]
    return [(opened, closed)]


def _equity_closure_window(
    market: dict[str, Any],
    now_utc: datetime,
) -> Optional[tuple[datetime, datetime, str, Optional[str]]]:
    market_tz = ZoneInfo(str(market["timezone"]))
    now_local = now_utc.astimezone(market_tz)
    intervals: list[tuple[datetime, datetime]] = []
    for offset in range(-14, 15):
        intervals.extend(
            _market_active_intervals(
                market,
                now_local.date() + timedelta(days=offset),
            )
        )
    if any(opened <= now_local < closed for opened, closed in intervals):
        return None

    previous_closes = [
        closed for _opened, closed in intervals if closed <= now_local
    ]
    next_opens = [
        opened for opened, _closed in intervals if opened > now_local
    ]
    if not previous_closes or not next_opens:
        return None

    holiday, holiday_name = _exchange_holiday(market, now_local)
    if holiday:
        reason = "holiday"
    elif now_local.weekday() >= 5:
        reason = "weekend"
    elif any(
        closed <= now_local < next_open
        for (_opened, closed), (next_open, _next_closed) in zip(
            intervals,
            intervals[1:],
        )
        if closed.date() == next_open.date()
    ):
        reason = "midday_break"
    else:
        reason = "outside_regular_session"
    return (
        max(previous_closes).astimezone(timezone.utc),
        min(next_opens).astimezone(timezone.utc),
        reason,
        holiday_name,
    )


def _equity_closed_overlap_seconds(
    market: dict[str, Any],
    start_epoch: float,
    end_epoch: float,
) -> float:
    start_utc = datetime.fromtimestamp(float(start_epoch), tz=timezone.utc)
    end_utc = datetime.fromtimestamp(float(end_epoch), tz=timezone.utc)
    duration = (end_utc - start_utc).total_seconds()
    if duration <= 0 or duration > 62 * 86_400:
        return 0.0

    market_tz = ZoneInfo(str(market["timezone"]))
    first_date = start_utc.astimezone(market_tz).date() - timedelta(days=1)
    last_date = end_utc.astimezone(market_tz).date() + timedelta(days=1)
    open_seconds = 0.0
    day = first_date
    while day <= last_date:
        for opened, closed in _market_active_intervals(market, day):
            overlap_start = max(start_utc, opened.astimezone(timezone.utc))
            overlap_end = min(end_utc, closed.astimezone(timezone.utc))
            if overlap_end > overlap_start:
                open_seconds += (overlap_end - overlap_start).total_seconds()
        day += timedelta(days=1)
    return max(0.0, duration - open_seconds)


def freshness_hole_explained_by_weekend(
    *,
    last_completed_epoch: float,
    cutoff_epoch: float,
    bar_seconds: float,
) -> bool:
    """True when the latest-N freshness hole is a standard weekend closure."""
    try:
        last_epoch = float(last_completed_epoch)
        cutoff = float(cutoff_epoch)
        seconds_per_bar = float(bar_seconds)
    except (TypeError, ValueError):
        return False
    if not (
        math.isfinite(last_epoch)
        and math.isfinite(cutoff)
        and math.isfinite(seconds_per_bar)
        and seconds_per_bar > 0
    ):
        return False
    hole = cutoff - last_epoch
    if hole <= 0:
        return False
    overlap = standard_weekend_overlap_seconds(last_epoch, cutoff)
    if overlap <= 0:
        return False
    unexplained = hole - overlap
    slack = min(3600.0, max(1.0, seconds_per_bar * 0.25))
    return unexplained <= slack


def freshness_hole_explained_by_session(
    symbol: Any,
    *,
    last_completed_epoch: float,
    cutoff_epoch: float,
    bar_seconds: float,
    symbol_info: Any = None,
) -> bool:
    """True when scheduled closed time explains a latest-bar freshness hole."""
    try:
        last_epoch = float(last_completed_epoch)
        cutoff = float(cutoff_epoch)
        seconds_per_bar = float(bar_seconds)
    except (TypeError, ValueError):
        return False
    if not (
        math.isfinite(last_epoch)
        and math.isfinite(cutoff)
        and math.isfinite(seconds_per_bar)
        and seconds_per_bar > 0
        and cutoff > last_epoch
    ):
        return False

    session_kind, _venue, market = _symbol_session_profile(
        symbol,
        symbol_info=symbol_info,
    )
    if session_kind == "continuous_24_7":
        return False
    if session_kind == "equity" and market is not None:
        overlap = _equity_closed_overlap_seconds(market, last_epoch, cutoff)
    else:
        overlap = standard_weekend_overlap_seconds(last_epoch, cutoff)
    if overlap <= 0:
        return False
    unexplained = cutoff - last_epoch - overlap
    slack = min(3600.0, max(1.0, seconds_per_bar * 0.25))
    return unexplained <= slack


def closed_session_context(
    symbol: Any,
    *,
    now_epoch: Any,
    item: str = "tick",
    data_age_seconds: Any = None,
    symbol_info: Any = None,
) -> Optional[dict[str, Any]]:
    if not str(symbol or "").strip():
        return None
    try:
        now_utc = datetime.fromtimestamp(float(now_epoch), tz=timezone.utc)
    except Exception:
        return None

    session_kind, venue, market = _symbol_session_profile(
        symbol,
        symbol_info=symbol_info,
    )
    if session_kind == "continuous_24_7":
        return None
    holiday_name: Optional[str] = None
    if session_kind == "equity" and market is not None:
        equity_closure = _equity_closure_window(market, now_utc)
        if equity_closure is None:
            return None
        close_utc, open_utc, reason, holiday_name = equity_closure
        source = "exchange_calendar"
    else:
        closure_window = standard_weekend_window(now_utc)
        if closure_window is None:
            return None
        close_utc, open_utc = closure_window
        reason = "weekend"
        source = "standard_weekend_hours"

    item_label = str(item or "data").strip() or "data"
    out = {
        "market_status": "closed",
        "market_status_reason": reason,
        "market_status_source": source,
        "note": f"Market is closed; showing the latest completed session {item_label}.",
        "assumed_closure_start": close_utc.isoformat().replace("+00:00", "Z"),
        "assumed_closure_end": open_utc.isoformat().replace("+00:00", "Z"),
        "assumed_closure_seconds": round_age_seconds(
            (open_utc - close_utc).total_seconds()
        ),
    }
    if venue and market is not None:
        out["market_venue"] = venue
        out["session_calendar"] = market.get("exchange_calendar") or venue
    if holiday_name:
        out["holiday"] = holiday_name
    if data_age_seconds is not None:
        try:
            raw_age_seconds = float(data_age_seconds)
            age_seconds = max(0.0, raw_age_seconds)
        except (TypeError, ValueError):
            raw_age_seconds = float("inf")
            age_seconds = float("inf")
        data_epoch = now_utc.timestamp() - raw_age_seconds
        rounded_age = round_age_seconds(age_seconds)
        out.update(
            {
                "data_age_seconds": rounded_age,
                "freshness_policy_relaxed": (
                    math.isfinite(data_epoch)
                    and raw_age_seconds >= 0
                    and data_epoch
                    >= close_utc.timestamp() - float(QUOTE_STALE_SECONDS)
                ),
            }
        )
    return out


def is_derived_age_seconds_key(key: Any) -> bool:
    """Return True for computed freshness ages, not filter or policy thresholds."""
    name = str(key or "").strip().lower()
    if not name.endswith("_age_seconds"):
        return False
    if name.startswith("max_"):
        return False
    if "stale_after" in name or "live_max" in name:
        return False
    return True


def round_age_seconds(seconds: Any) -> Optional[float]:
    """Round a wall-clock age without IEEE-754 noise.

    Sub-second ages stay at millisecond resolution so live-tick gates can
    still distinguish a 250ms quote from a missing one. Older ages use
    integer seconds.
    """
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if abs(value) < 1.0:
        return round(value, 3)
    return max(0, int(round(value)))


def format_age_seconds(seconds: Any) -> Optional[str]:
    try:
        total = max(0, int(round(float(seconds))))
    except Exception:
        return None
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _clean_status(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def _coerce_bool_flag(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, bytes, bytearray, list, tuple, dict, set)):
        return None
    if not hasattr(value, "__bool__"):
        return None
    try:
        return bool(value)
    except Exception:
        return None


def format_freshness_label(
    *,
    data_stale: Any = None,
    market_status: Any = None,
    market_status_reason: Any = None,
    age_seconds: Any = None,
    age_text: Any = None,
    item: str = "data",
    delayed: bool = False,
    delay_minutes: Any = None,
    timestamp_available: bool = True,
) -> Optional[str]:
    if delayed:
        delay = None
        try:
            numeric_delay = float(delay_minutes)
            if math.isfinite(numeric_delay) and numeric_delay > 0:
                delay = int(round(numeric_delay))
        except Exception:
            delay = None
        label = f"delayed {delay}m" if delay else "delayed"
        if not timestamp_available:
            return f"{label}, timestamp unavailable"
        return label

    status = _clean_status(market_status)
    reason = _clean_status(market_status_reason)
    if status == "closed":
        label = "closed"
        if reason:
            label = f"{label} {reason}"
    elif status and status not in {"open", "live"}:
        label = status
    else:
        stale_flag = _coerce_bool_flag(data_stale)
        if stale_flag is True:
            label = "stale"
        elif stale_flag is False:
            label = "fresh"
        else:
            return None

    age = str(age_text or "").strip() or format_age_seconds(age_seconds)
    if age:
        item_label = str(item or "data").strip() or "data"
        return f"{label}, {item_label} {age} ago"
    return label


def completed_bar_freshness_fields(
    symbol: Any,
    timeframe: Any,
    last_bar_epoch: Any,
    *,
    now_epoch: Any = None,
    item: str = "bar",
    tolerance_bars: Any = SANITY_BARS_TOLERANCE,
) -> dict[str, Any]:
    """Build the shared latest-completed-bar freshness contract."""
    timeframe_value = str(timeframe or "").strip().upper()
    try:
        opened_at = float(last_bar_epoch)
        step_seconds = int(TIMEFRAME_SECONDS[timeframe_value])
        observed_at = (
            datetime.now(timezone.utc).timestamp()
            if now_epoch is None
            else float(now_epoch)
        )
        if not all(math.isfinite(value) for value in (opened_at, observed_at)):
            return {}
        completed_at = bar_close_epoch(opened_at, timeframe_value)
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return {}

    signed_age_seconds = observed_at - completed_at
    age_seconds = max(0, int(round(signed_age_seconds)))
    future_skew_seconds = max(0.0, -signed_age_seconds)
    timestamp_ahead = future_skew_seconds > 0.0
    timestamp_in_future = (
        future_skew_seconds >= float(TIMESTAMP_FUTURE_TOLERANCE_SECONDS)
    )
    try:
        policy_bars = max(1, int(tolerance_bars))
    except (TypeError, ValueError, OverflowError):
        policy_bars = max(1, int(SANITY_BARS_TOLERANCE))
    stale_after = int(step_seconds * policy_bars)
    data_stale = age_seconds > stale_after or timestamp_in_future
    out: dict[str, Any] = {
        "data_as_of": format_epoch_utc(completed_at),
        "data_as_of_epoch": completed_at,
        "data_age_seconds": age_seconds,
        "data_stale": data_stale,
        "stale_after_seconds": stale_after,
        "freshness_basis": "last_completed_bar_close",
        "freshness_age_metric": "latest_completed_bar_close_age_seconds",
    }
    if timestamp_ahead:
        out["timestamp_ahead_of_wall_clock"] = True
        out["timestamp_skew_seconds"] = round(future_skew_seconds, 3)
        out["timestamp_skew_tolerance_seconds"] = int(
            TIMESTAMP_FUTURE_TOLERANCE_SECONDS
        )
        out["timestamp_skew_basis"] = "latest_completed_bar_close"
    if timestamp_in_future:
        out["timestamp_in_future"] = True
        out["timestamp_warning"] = (
            "Latest completed-bar timestamp is ahead of the wall clock; "
            "the history is not safe for live decisions."
        )
    closed_session = closed_session_context(
        symbol,
        now_epoch=observed_at,
        item=item,
        data_age_seconds=None if timestamp_in_future else age_seconds,
    )
    if closed_session:
        out.update(closed_session)
    out["history_policy_ok"] = not data_stale and not bool(closed_session)
    policy_relaxed = out.get("freshness_policy_relaxed") is not False
    if timestamp_in_future:
        label = (
            f"clock skew, {item} timestamp "
            f"{format_age_seconds(future_skew_seconds)} ahead of wall clock"
        )
    elif timestamp_ahead:
        label = (
            f"fresh with tolerated clock skew, {item} timestamp "
            f"{format_age_seconds(future_skew_seconds)} ahead of wall clock"
        )
    else:
        label = format_freshness_label(
            data_stale=data_stale,
            market_status=out.get("market_status") if policy_relaxed else None,
            market_status_reason=(
                out.get("market_status_reason") if policy_relaxed else None
            ),
            age_seconds=age_seconds,
            item=item,
        )
    if label:
        out["freshness"] = label
    if data_stale:
        out["stale_warning"] = (
            "Latest completed bar is outside the freshness policy window; "
            "market may be closed or broker data may be stale."
        )
    return out
