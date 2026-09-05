"""Market status and trading hours MCP tool."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Tuple
from zoneinfo import ZoneInfo

import holidays

from ..shared.market_sessions import (
    MARKET_SESSIONS,
    exchange_holidays,
)
from ..shared.market_sessions import (
    is_early_close_session as evaluate_early_close_session,
)
from ..shared.schema import DetailLiteral
from ..shared.symbols import is_probably_crypto_symbol, is_probably_forex_symbol
from ..utils.coercion import UNPARSED_BOOL, coerce_optional_bool, parse_bool_like
from ..utils.freshness import is_standard_weekend_closure
from ..utils.market_metadata import build_tick_freshness_context
from ..utils.mt5 import (
    MT5ConnectionError,
    _normalize_times_in_struct,
    ensure_mt5_connection_or_raise,
    resolve_broker_symbol_name,
)
from ..utils.mt5_enums import decode_mt5_enum_label
from ..utils.quote import (
    enforce_quote_execution_readiness,
    resolve_quote_tick,
    tick_epoch,
    tick_value,
)
from ..utils.time import format_datetime_utc, format_epoch_utc
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway
from .output_contract import normalize_output_verbosity_detail
from .runtime_metadata import attach_mt5_source, build_runtime_timezone_meta

logger = logging.getLogger(__name__)

_SYMBOL_SCHEDULE_LOOKBACK_DAYS = 7
_M1_TIMEFRAME_FALLBACK = 1
_WEEKDAY_NAMES = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)



VenueLiteral = Literal[
    "NYSE",
    "NASDAQ",
    "LSE",
    "XETRA",
    "EURONEXT",
    "TSE",
    "HKEX",
    "SSE",
    "ASX",
]
_VENUE_REGIONS = {
    "NYSE": "us",
    "NASDAQ": "us",
    "LSE": "europe",
    "XETRA": "europe",
    "EURONEXT": "europe",
    "TSE": "asia",
    "HKEX": "asia",
    "SSE": "asia",
    "ASX": "asia",
}


@lru_cache(maxsize=64)
def _get_holidays(country: str, year: int) -> holidays.HolidayBase:
    """Get the holiday calendar for a country/year pair."""
    return holidays.country_holidays(country, years=[int(year)])




def _is_holiday(
    country: str,
    dt: datetime,
    exchange: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Check if date is a holiday and return holiday name if so."""
    h = (
        exchange_holidays(exchange, dt.year)
        if exchange
        else _get_holidays(country, dt.year)
    )
    date_key = dt.date()
    if date_key in h:
        return True, str(h[date_key])
    return False, None


def _get_local_time(tz_name: str) -> datetime:
    """Get current time in specified timezone."""
    return datetime.now(ZoneInfo(tz_name))


def _normalize_time(dt: datetime) -> datetime:
    """Normalize datetime for comparison."""
    return dt.replace(second=0, microsecond=0)


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = parse_bool_like(value, allow_none=True)
        if parsed is UNPARSED_BOOL or parsed is None:
            return None
        return bool(parsed)
    numeric = coerce_optional_bool(value)
    if numeric is not None:
        return numeric
    try:
        return bool(value)
    except Exception:
        return None


def _format_local_iso(dt: datetime) -> str:
    return dt.replace(second=0, microsecond=0).isoformat()


def _format_duration(minutes: int) -> str:
    """Format minutes into human-readable duration."""
    if minutes < 60:
        return f"{minutes}min{'s' if minutes != 1 else ''}"
    hours = minutes // 60
    mins = minutes % 60
    if mins == 0:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    return f"{hours}h {mins}min{'s' if mins != 1 else ''}"


def _normalize_timezone_display(
    value: Optional[str],
    *,
    symbol_mode: bool = False,
) -> Optional[str]:
    normalized = str(value or "auto").strip().lower()
    if normalized in {"", "auto"}:
        return "server" if symbol_mode else "local"
    if normalized in {"local", "utc", "server"}:
        return normalized
    return None


def _apply_market_timezone_display(
    status: Dict[str, Any],
    *,
    now_local: datetime,
    display: str,
    server_tzinfo: Any = None,
    server_label: Optional[str] = None,
    exchange_timezone: Optional[str] = None,
) -> Dict[str, Any]:
    out = dict(status)
    exchange_local_time = out.get("local_time")
    out["local_time"] = exchange_local_time
    if display == "local":
        out["display_time"] = exchange_local_time
        out["display_timezone"] = exchange_timezone or "exchange_local"
        return out
    target_tz = timezone.utc if display == "utc" else server_tzinfo
    if target_tz is None:
        target_tz = timezone.utc
    display_time = (
        format_datetime_utc(now_local)
        if display == "utc"
        else now_local.astimezone(target_tz).replace(microsecond=0).isoformat()
    )
    out["exchange_local_time"] = exchange_local_time
    out["display_time"] = display_time
    out["display_timezone"] = (
        "UTC" if display == "utc" else server_label or "UTC"
    )
    for key in ("next_open", "next_close"):
        value = out.get(key)
        if not isinstance(value, str) or not value:
            continue
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            continue
        if parsed.tzinfo is not None:
            out[key] = (
                format_datetime_utc(parsed)
                if display == "utc"
                else parsed.astimezone(target_tz).replace(microsecond=0).isoformat()
            )
    return out


def _apply_global_weekend_reason(status: Dict[str, Any], *, now_local: datetime) -> Dict[str, Any]:
    if now_local.weekday() < 5:
        return status
    if status.get("status") != "closed" or status.get("reason") not in {
        "after_hours",
        "before_open",
        "overnight",
        "post_close",
    }:
        return status
    out = dict(status)
    out["reason"] = "weekend"
    return out


def _runtime_meta_tzinfo(
    meta: Dict[str, Any],
    *,
    allow_offset: bool = False,
) -> tuple[Optional[Any], Optional[str]]:
    tz_name = meta.get("tz")
    if isinstance(tz_name, str) and tz_name.strip():
        try:
            return ZoneInfo(tz_name.strip()), tz_name.strip()
        except Exception:
            pass
    if allow_offset:
        offset_seconds = meta.get("offset_seconds")
        if offset_seconds is not None:
            try:
                tzinfo = timezone(timedelta(seconds=int(offset_seconds)))
                return tzinfo, tzinfo.tzname(None) or "server"
            except Exception:
                pass
    return None, None




def _is_early_close_session(
    market: Dict[str, Any],
    country: str,
    session_dt: datetime,
) -> bool:
    return evaluate_early_close_session(
        market,
        country,
        session_dt,
        holiday_resolver=_is_holiday,
    )


def _next_market_open_datetime(
    market: Dict[str, Any],
    country: str,
    now_local: datetime,
) -> datetime:
    """Return the next tradable session open after *now_local*."""
    next_open = now_local + timedelta(days=1)
    while True:
        if next_open.weekday() >= 5:
            next_open += timedelta(days=1)
            continue
        is_holiday_result, _holiday_name = _is_holiday(
            country, next_open, market.get("exchange_calendar")
        )
        if is_holiday_result and not _is_early_close_session(market, country, next_open):
            next_open += timedelta(days=1)
            continue
        return next_open.replace(
            hour=market["open"][0],
            minute=market["open"][1],
            second=0,
            microsecond=0,
        )


def _check_market_status(market_id: str, now_local: datetime) -> Dict[str, Any]:
    """Check status for a single market."""
    market = MARKET_SESSIONS[market_id]
    country = market["country"]
    
    # Check weekend
    weekday = now_local.weekday()
    if weekday >= 5:  # Saturday or Sunday
        next_open = _next_market_open_datetime(market, country, now_local)
        minutes_until = int((next_open.astimezone(timezone.utc) - _normalize_time(now_local).astimezone(timezone.utc)).total_seconds() // 60)
        return {
            "venue": market_id,
            "name": market["name"],
            "status": "closed",
            "reason": "weekend",
            "local_time": _format_local_iso(now_local),
            "message": f"{market_id}: Closed (opening in {_format_duration(minutes_until)})",
            "next_open": next_open.isoformat(),
            "minutes_until_open": minutes_until,
        }
    
    # Check holidays
    is_holiday_result, holiday_name = _is_holiday(
        country, now_local, market.get("exchange_calendar")
    )

    # Determine early close BEFORE the holiday return so same-day
    # half-holidays are not treated as full closures.
    is_early_close = _is_early_close_session(market, country, now_local)

    # Full holiday (not a half-day session) → closed
    if is_holiday_result and not is_early_close:
        next_open = _next_market_open_datetime(market, country, now_local)
        minutes_until = int((next_open.astimezone(timezone.utc) - _normalize_time(now_local).astimezone(timezone.utc)).total_seconds() // 60)
        return {
            "venue": market_id,
            "name": market["name"],
            "status": "closed",
            "reason": "holiday",
            "holiday": holiday_name,
            "local_time": _format_local_iso(now_local),
            "message": f"{market_id}: Closed - Holiday ({holiday_name}, opening in {_format_duration(minutes_until)})",
            "next_open": next_open.isoformat(),
            "minutes_until_open": minutes_until,
        }
    
    open_hour, open_minute = market["open"]
    close_hour, close_minute = market["close"]
    
    if is_early_close and market.get("early_close"):
        close_hour, close_minute = market["early_close"]
    
    open_time = now_local.replace(hour=open_hour, minute=open_minute, second=0, microsecond=0)
    close_time = now_local.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0)
    session_fields = (
        {
            "early_close": True,
            "early_close_time": f"{close_hour:02d}:{close_minute:02d}",
        }
        if is_early_close
        else {}
    )
    
    # Check pre-market (before open)
    now_norm = _normalize_time(now_local)
    if now_norm < open_time:
        minutes_until_open = int((open_time.astimezone(timezone.utc) - now_norm.astimezone(timezone.utc)).total_seconds() // 60)
        pre_open = market.get("pre_open")
        if pre_open:
            pre_open_time = now_local.replace(
                hour=pre_open[0],
                minute=pre_open[1],
                second=0,
                microsecond=0,
            )
            if now_norm >= pre_open_time:
                return {
                    "venue": market_id,
                    "name": market["name"],
                    "status": "pre_market",
                    "local_time": _format_local_iso(now_local),
                    "message": f"{market_id}: Pre-market (opening in {_format_duration(minutes_until_open)})",
                    "next_open": open_time.isoformat(),
                    "minutes_until_open": minutes_until_open,
                    **session_fields,
                }
        return {
            "venue": market_id,
            "name": market["name"],
            "status": "closed",
            "reason": "overnight" if pre_open else "before_open",
            "local_time": _format_local_iso(now_local),
            "message": (
                f"{market_id}: Closed "
                f"(opening in {_format_duration(minutes_until_open)})"
            ),
            "next_open": open_time.isoformat(),
            "minutes_until_open": minutes_until_open,
            **session_fields,
        }
    
    # A shortened session can end exactly when the normal lunch interval starts.
    # Resolve the effective close before considering a midday break.
    if now_norm >= close_time:
        after_hours_close = market.get("after_hours_close")
        if after_hours_close:
            after_hours_close_time = now_local.replace(
                hour=after_hours_close[0],
                minute=after_hours_close[1],
                second=0,
                microsecond=0,
            )
            if now_norm < after_hours_close_time:
                minutes_until_close = int(
                    (after_hours_close_time.astimezone(timezone.utc) - now_norm.astimezone(timezone.utc)).total_seconds() // 60
                )
                return {
                    "venue": market_id,
                    "name": market["name"],
                    "status": "after_hours",
                    "local_time": _format_local_iso(now_local),
                    "message": (
                        f"{market_id}: After-hours; electronic session ends "
                        f"{after_hours_close_time.strftime('%H:%M')} "
                        f"({_format_duration(minutes_until_close)} remaining)"
                    ),
                    "next_close": after_hours_close_time.isoformat(),
                    "next_after_hours_close": after_hours_close_time.isoformat(),
                    "minutes_until_close": minutes_until_close,
                    "minutes_until_after_hours_close": minutes_until_close,
                    **session_fields,
                }

        next_open = _next_market_open_datetime(market, country, now_local)
        minutes_until = int((next_open.astimezone(timezone.utc) - now_norm.astimezone(timezone.utc)).total_seconds() // 60)
        return {
            "venue": market_id,
            "name": market["name"],
            "status": "closed",
            "reason": "overnight" if after_hours_close else "post_close",
            "local_time": _format_local_iso(now_local),
            "message": (
                f"{market_id}: Closed "
                f"(opening in {_format_duration(minutes_until)})"
            ),
            "next_open": next_open.isoformat(),
            "minutes_until_open": minutes_until,
            **session_fields,
        }

    # Check if during lunch break.
    if market.get("lunch_start") and market.get("lunch_end"):
        lunch_start = now_local.replace(hour=market["lunch_start"][0], minute=market["lunch_start"][1], second=0, microsecond=0)
        lunch_end = now_local.replace(hour=market["lunch_end"][0], minute=market["lunch_end"][1], second=0, microsecond=0)

        effective_lunch_end = min(lunch_end, close_time)
        if lunch_start < close_time and lunch_start <= now_norm < effective_lunch_end:
            minutes_until_resume = int((effective_lunch_end.astimezone(timezone.utc) - now_norm.astimezone(timezone.utc)).total_seconds() // 60)
            return {
                "venue": market_id,
                "name": market["name"],
                "status": "lunch_break",
                "local_time": _format_local_iso(now_local),
                "message": f"{market_id}: Lunch break (resuming in {_format_duration(minutes_until_resume)})",
                "next_open": effective_lunch_end.isoformat(),
                "minutes_until_open": minutes_until_resume,
                **session_fields,
            }
    
    # Check if market is open
    if now_norm < close_time:
        minutes_until_close = int((close_time.astimezone(timezone.utc) - now_norm.astimezone(timezone.utc)).total_seconds() // 60)
        return {
            "venue": market_id,
            "name": market["name"],
            "status": "open",
            "local_time": _format_local_iso(now_local),
            "message": f"{market_id}: Open (closing in {_format_duration(minutes_until_close)})",
            "next_close": close_time.isoformat(),
            "minutes_until_close": minutes_until_close,
            **session_fields,
        }
    
    raise RuntimeError("Market status interval resolution failed.")


def _get_upcoming_holidays(
    market_ids: List[str],
    days_ahead: int = 14,
    *,
    now_utc: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Get venue-local closure and shortened-session events."""
    events: Dict[Tuple[str, str, str, Optional[str], str], Dict[str, Any]] = {}
    current_utc = now_utc or datetime.now(timezone.utc)

    def _upsert(
        *,
        market_id: str,
        market: Dict[str, Any],
        event_date: Any,
        holiday_name: str,
        impact: str,
        early_close_time: Optional[str],
        days_away: int,
    ) -> None:
        country = str(market["country"])
        key = (
            event_date.isoformat(),
            country,
            impact,
            early_close_time,
            holiday_name,
        )
        row = events.get(key)
        if row is None:
            row = {
                "date": event_date.isoformat(),
                "holiday": holiday_name,
                "country": country,
                "markets_affected": [],
                "impact": impact,
                "early_close_time": early_close_time,
                "days_away": int(days_away),
                "calendar_source": (
                    "exchange_calendar"
                    if market.get("exchange_calendar")
                    else "country_calendar_fallback"
                ),
            }
            events[key] = row
        if market_id not in row["markets_affected"]:
            row["markets_affected"].append(market_id)

    for market_id in market_ids:
        market = MARKET_SESSIONS.get(market_id)
        if market is None:
            continue
        country = str(market["country"])
        exchange = market.get("exchange_calendar")
        now_local = current_utc.astimezone(ZoneInfo(str(market["timezone"])))
        try:
            for days_away in range(0, days_ahead + 1):
                check_local = now_local + timedelta(days=days_away)
                is_holiday_result, holiday_name = _is_holiday(
                    country, check_local, exchange
                )
                if not is_holiday_result or holiday_name is None:
                    continue
                event_date = check_local.date()
                same_day_early = any(
                    name.lower() in holiday_name.lower()
                    for name in market.get("early_close_holidays", [])
                )
                early_time = (
                    f"{market['early_close'][0]:02d}:{market['early_close'][1]:02d}"
                    if same_day_early and market.get("early_close")
                    else None
                )
                _upsert(
                    market_id=market_id,
                    market=market,
                    event_date=event_date,
                    holiday_name=holiday_name,
                    impact="early_close" if same_day_early else "closed",
                    early_close_time=early_time,
                    days_away=days_away,
                )

                derived_time = (
                    f"{market['early_close'][0]:02d}:{market['early_close'][1]:02d}"
                    if market.get("early_close")
                    else None
                )
                if any(
                    name.lower() in holiday_name.lower()
                    for name in market.get("early_close_day_after", [])
                ):
                    after_date = event_date + timedelta(days=1)
                    if after_date.weekday() < 5:
                        _upsert(
                            market_id=market_id,
                            market=market,
                            event_date=after_date,
                            holiday_name=f"Day after {holiday_name}",
                            impact="early_close",
                            early_close_time=derived_time,
                            days_away=days_away + 1,
                        )
                if any(
                    name.lower() in holiday_name.lower()
                    for name in market.get("early_close_eves", [])
                ):
                    eve_date = event_date - timedelta(days=1)
                    if eve_date.weekday() < 5:
                        _upsert(
                            market_id=market_id,
                            market=market,
                            event_date=eve_date,
                            holiday_name=f"Eve of {holiday_name}",
                            impact="early_close",
                            early_close_time=derived_time,
                            days_away=max(0, days_away - 1),
                        )
                if any(
                    name.lower() in holiday_name.lower()
                    for name in market.get("early_close_last_business_day_before", [])
                ):
                    previous_session = event_date - timedelta(days=1)
                    while previous_session.weekday() >= 5 or _is_holiday(
                        country,
                        datetime.combine(previous_session, datetime.min.time()).replace(
                            tzinfo=ZoneInfo(str(market["timezone"]))
                        ),
                        exchange,
                    )[0]:
                        previous_session -= timedelta(days=1)
                    _upsert(
                        market_id=market_id,
                        market=market,
                        event_date=previous_session,
                        holiday_name=f"Last business day before {holiday_name}",
                        impact="early_close",
                        early_close_time=derived_time,
                        days_away=max(0, (previous_session - now_local.date()).days),
                    )
        except Exception as exc:
            logger.warning("Failed to get exchange holidays for %s: %s", market_id, exc)

    upcoming = list(events.values())
    for row in upcoming:
        row["markets_affected"].sort()
    upcoming.sort(key=lambda row: (row["date"], row["country"], row["impact"]))
    return upcoming


def normalize_market_status_output(
    result: Dict[str, Any],
    *,
    detail: Any = None,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return dict(result)

    detail_mode = normalize_output_verbosity_detail(detail)
    out = dict(result)
    if detail_mode == "full":
        return out

    markets = out.get("markets")
    if out.get("mode") == "global" and detail_mode in {"compact", "summary"}:
        out.pop("markets", None)
    elif isinstance(markets, list):
        compact_markets = []
        for market in markets:
            if isinstance(market, dict):
                market = {key: value for key, value in market.items() if key != "message"}
            compact_markets.append(market)
        out["markets"] = compact_markets
    out.pop("message", None)
    out.pop("upcoming_holidays", None)
    out.pop("upcoming_holidays_count", None)
    out.pop("upcoming_holidays_summary", None)
    return out


def _symbol_trade_mode_status(gateway: Any, trade_mode: Any) -> Dict[str, Any]:
    label = decode_mt5_enum_label(
        gateway,
        trade_mode,
        prefix="SYMBOL_TRADE_MODE_",
    )
    label_text = str(label or "").strip()
    normalized = label_text.lower().replace("symbol_trade_mode_", "")
    if not normalized:
        normalized = str(trade_mode).strip().lower()

    full_values = {
        getattr(gateway, "SYMBOL_TRADE_MODE_FULL", object()),
    }
    disabled_values = {
        getattr(gateway, "SYMBOL_TRADE_MODE_DISABLED", object()),
    }
    close_only_values = {
        getattr(gateway, "SYMBOL_TRADE_MODE_CLOSEONLY", object()),
    }
    long_only_values = {
        getattr(gateway, "SYMBOL_TRADE_MODE_LONGONLY", object()),
    }
    short_only_values = {
        getattr(gateway, "SYMBOL_TRADE_MODE_SHORTONLY", object()),
    }

    if trade_mode in disabled_values or "disabled" in normalized:
        status = "disabled"
        can_open = False
        is_tradable = False
    elif trade_mode in close_only_values or "close" in normalized:
        status = "close_only"
        can_open = False
        is_tradable = True
    elif trade_mode in long_only_values or "long" in normalized:
        status = "long_only"
        can_open = True
        is_tradable = True
    elif trade_mode in short_only_values or "short" in normalized:
        status = "short_only"
        can_open = True
        is_tradable = True
    elif trade_mode in full_values or "full" in normalized:
        status = "tradable"
        can_open = True
        is_tradable = True
    else:
        status = "unknown"
        can_open = None
        is_tradable = None

    return {
        "trade_mode": trade_mode,
        "trade_mode_label": label_text or None,
        "status": status,
        "can_open_new_positions": can_open,
        "is_tradable": is_tradable,
    }


def _symbol_tick_snapshot(
    symbol: str,
    tick: Any,
    *,
    now_utc: datetime,
    source_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if tick is None:
        return {
            "tick_available": False,
            "tick_freshness": "missing",
        }

    out: Dict[str, Any] = {
        "tick_available": True,
    }
    if source_metadata:
        out.update(source_metadata)
    quote_epoch = tick_epoch(tick)
    if quote_epoch is not None:
        try:
            quote_as_of = format_datetime_utc(
                datetime.fromtimestamp(quote_epoch, tz=timezone.utc)
            )
            out["quote_as_of"] = quote_as_of
            out["last_tick_time"] = quote_as_of
            freshness = build_tick_freshness_context(
                symbol,
                tick_epoch=quote_epoch,
                now_epoch=now_utc.timestamp(),
                item="tick",
                age_rounder=lambda value: round(value, 3),
            )
            if freshness:
                out["data_age_seconds"] = freshness["data_age_seconds"]
                out["last_tick_age_seconds"] = freshness["data_age_seconds"]
                out["tick_freshness"] = freshness.get("freshness_state", "unknown")
                for key in (
                    "data_stale",
                    "usable_for_live_trading",
                    "usable_for_live_trading_basis",
                    "freshness_reason",
                    "timestamp_ahead_of_wall_clock",
                    "timestamp_in_future",
                    "timestamp_skew_seconds",
                    "timestamp_skew_tolerance_seconds",
                    "timestamp_warning",
                    "market_status",
                    "market_status_reason",
                    "market_status_source",
                    "freshness_policy_relaxed",
                    "note",
                ):
                    if freshness.get(key) is not None:
                        out[key] = freshness.get(key)
            else:
                out["tick_freshness"] = "unknown"
        except (OSError, OverflowError, TypeError, ValueError):
            out["tick_freshness"] = "unknown"
    else:
        out["tick_freshness"] = "unknown"

    for field in ("bid", "ask", "last", "volume"):
        value = tick_value(tick, field)
        if value is not None:
            out[field] = value
    enforce_quote_execution_readiness(
        out,
        bid=tick_value(tick, "bid"),
        ask=tick_value(tick, "ask"),
        quote_source_conflict=out.get("quote_source_conflict"),
    )
    return out


def _rate_epoch_seconds(row: Any) -> Optional[float]:
    value = None
    if isinstance(row, dict):
        value = row.get("time")
    else:
        try:
            value = row["time"]
        except (IndexError, KeyError, TypeError, ValueError):
            value = getattr(row, "time", None)

    try:
        epoch = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(epoch):
        return None
    return epoch


def _minute_bounds(minutes: List[int], *, max_missing_minutes: int = 1) -> List[Tuple[int, int]]:
    """Return inferred half-open intervals while bridging small tick gaps."""
    normalized = sorted(
        {int(minute) for minute in minutes if 0 <= int(minute) < 24 * 60}
    )
    if not normalized:
        return []

    bounds: List[Tuple[int, int]] = []
    start = normalized[0]
    previous = normalized[0]
    for minute in normalized[1:]:
        if minute <= previous + int(max_missing_minutes) + 1:
            previous = minute
            continue
        bounds.append((start, previous + 1))
        start = previous = minute
    bounds.append((start, previous + 1))
    return bounds


def _minute_ranges(minutes: List[int]) -> List[str]:
    """Format inferred UTC minute-of-day intervals."""

    def _label(minute: int) -> str:
        hour, minute_of_hour = divmod(minute, 60)
        return f"{hour:02d}:{minute_of_hour:02d}"

    return [
        f"{_label(start)}-{_label(end)}"
        for start, end in _minute_bounds(minutes)
    ]


def _infer_symbol_schedule_from_recent_candles(
    symbol: str,
    gateway: Any,
    *,
    now_utc: datetime,
) -> Dict[str, Any]:
    lookback_days = _SYMBOL_SCHEDULE_LOOKBACK_DAYS
    base: Dict[str, Any] = {
        "source": "recent_m1_candles",
        "lookback_days": lookback_days,
        "timeframe": "M1",
    }
    timeframe = getattr(gateway, "TIMEFRAME_M1", _M1_TIMEFRAME_FALLBACK)
    # Include the full matching weekday from the prior week. An exact rolling
    # timestamp would discard later session minutes just when today reaches
    # them, leaving no complete prior-day schedule to compare against.
    start_utc = (now_utc - timedelta(days=lookback_days)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    try:
        rates = gateway.copy_rates_range(symbol, timeframe, start_utc, now_utc)
    except Exception as exc:
        logger.warning("Failed to infer market schedule for %s from candles: %s", symbol, exc)
        return {
            **base,
            "confidence": "unavailable",
            "candles_analyzed": 0,
            "error": str(exc),
        }

    if rates is None or len(rates) == 0:
        return {
            **base,
            "confidence": "unavailable",
            "candles_analyzed": 0,
        }
    rates = _normalize_times_in_struct(rates)

    slots: set[Tuple[int, int]] = set()
    active_weekdays: set[int] = set()
    weekend_candles = 0
    saturday_candles = 0
    sunday_candles = 0
    candle_count = 0
    for row in rates:
        epoch = _rate_epoch_seconds(row)
        if epoch is None:
            continue
        try:
            candle_time = datetime.fromtimestamp(epoch, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            continue
        candle_count += 1
        weekday = candle_time.weekday()
        minute_of_day = candle_time.hour * 60 + candle_time.minute
        slots.add((weekday, minute_of_day))
        active_weekdays.add(weekday)
        if weekday >= 5:
            weekend_candles += 1
        if weekday == 5:
            saturday_candles += 1
        elif weekday == 6:
            sunday_candles += 1

    if candle_count == 0:
        return {
            **base,
            "confidence": "unavailable",
            "candles_analyzed": 0,
        }

    current_weekday = now_utc.weekday()
    current_minute = now_utc.hour * 60 + now_utc.minute
    active_intervals_by_day: Dict[str, List[str]] = {}
    current_time_in_active_session = False
    for weekday in sorted(active_weekdays):
        minutes = [minute for day, minute in slots if day == weekday]
        active_intervals_by_day[_WEEKDAY_NAMES[weekday]] = _minute_ranges(minutes)
        if weekday == current_weekday:
            current_time_in_active_session = any(
                start <= current_minute < end
                for start, end in _minute_bounds(minutes)
            )

    active_slot_count = len(slots)
    active_minute_coverage = active_slot_count / (7 * 24 * 60)
    inferred_24_7 = active_minute_coverage >= 0.90 and all(
        weekday in active_weekdays for weekday in range(7)
    )
    confidence = "medium" if candle_count >= 20 and active_slot_count >= 2 else "low"
    if inferred_24_7 and candle_count >= 100:
        confidence = "high"

    return {
        **base,
        "confidence": confidence,
        "candles_analyzed": candle_count,
        "active_weekdays": [_WEEKDAY_NAMES[weekday] for weekday in sorted(active_weekdays)],
        "active_intervals_utc": active_intervals_by_day,
        "active_minute_coverage": round(active_minute_coverage, 6),
        # Sunday evening bars are the normal FX weekly reopen. Saturday
        # activity is the reliable discriminator for true weekend trading.
        "trades_on_weekends": saturday_candles > 0,
        "weekend_candles": weekend_candles,
        "saturday_candles": saturday_candles,
        "sunday_candles": sunday_candles,
        "current_time_in_active_session": current_time_in_active_session,
        "inferred_24_7": inferred_24_7,
    }


def _symbol_market_now(
    now_utc: datetime,
    *,
    display: str,
    server: Dict[str, Any],
    client: Dict[str, Any],
) -> tuple[str, str, str]:
    display_mode = str(display or "server").strip().lower()
    if display_mode == "utc":
        return "utc", "UTC", format_datetime_utc(now_utc)
    if display_mode == "local":
        client_tzinfo, client_label = _runtime_meta_tzinfo(client)
        if client_tzinfo is not None:
            market_now = now_utc.astimezone(client_tzinfo).replace(microsecond=0).isoformat()
            return "client", client_label or "local", market_now
        return "utc", "UTC", format_datetime_utc(now_utc)
    server_tzinfo, server_label = _runtime_meta_tzinfo(server, allow_offset=True)
    if server_tzinfo is not None:
        market_now = now_utc.astimezone(server_tzinfo).replace(microsecond=0).isoformat()
        return "server", server_label or "server", market_now
    return "utc", "UTC", format_datetime_utc(now_utc)


def _check_symbol_market_status(
    symbol: str,
    *,
    detail: str,
    timezone_display: str = "server",
    gateway: Any = None,
) -> Dict[str, Any]:
    symbol_input = str(symbol or "").strip()
    if not symbol_input:
        return {"error": "symbol cannot be empty."}

    mt5_gateway = gateway if gateway is not None else create_mt5_gateway(
        ensure_connection_impl=ensure_mt5_connection_or_raise,
    )
    try:
        mt5_gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return {"error": str(exc)}

    symbol_name = resolve_broker_symbol_name(
        symbol_input.upper(),
        gateway=mt5_gateway,
    )

    info = mt5_gateway.symbol_info(symbol_name)
    if info is None:
        return {"error": f"Symbol {symbol_name} not found"}

    query_started_utc = datetime.now(timezone.utc)
    trade_mode = getattr(info, "trade_mode", None)
    mode_status = _symbol_trade_mode_status(mt5_gateway, trade_mode)
    raw_tick = mt5_gateway.symbol_info_tick(symbol_name)
    tick, quote_source = resolve_quote_tick(
        mt5_gateway,
        symbol_name,
        raw_tick,
        now_epoch=query_started_utc.timestamp(),
    )
    now_utc = datetime.now(timezone.utc)
    tick_status = _symbol_tick_snapshot(
        symbol_name,
        tick,
        now_utc=now_utc,
        source_metadata=quote_source,
    )
    schedule_status = _infer_symbol_schedule_from_recent_candles(
        symbol_name,
        mt5_gateway,
        now_utc=now_utc,
    )

    trade_mode_can_open = _coerce_optional_bool(mode_status["can_open_new_positions"])
    can_open = trade_mode_can_open
    live_ready = _coerce_optional_bool(tick_status.get("usable_for_live_trading"))
    tick_freshness = tick_status.get("tick_freshness")
    reason = None
    is_crypto_symbol = is_probably_crypto_symbol(symbol_name)
    recent_schedule_allows_now = (
        _coerce_optional_bool(schedule_status.get("current_time_in_active_session"))
        is True
    )
    schedule_match = _coerce_optional_bool(
        schedule_status.get("current_time_in_active_session")
    )
    if schedule_match is False or live_ready is False:
        local_session_open: Optional[bool] = False
    elif schedule_match is True and live_ready is True:
        local_session_open = True
    else:
        local_session_open = None
    weekend_closed_now = is_standard_weekend_closure(now_utc)
    if (
        can_open is True
        and weekend_closed_now
        and not is_crypto_symbol
        and not recent_schedule_allows_now
    ):
        open_state = "weekend_closed"
        can_open = False
        reason = "weekend"
    elif can_open is True and live_ready is not True:
        can_open = False
        if schedule_match is False:
            open_state = "session_closed"
            reason = "not_in_recent_session"
        else:
            open_state = "quote_not_live_ready"
            reason = (
                "quote_source_conflict"
                if isinstance(tick_status.get("quote_source_conflict"), dict)
                else f"quote_{tick_status['spread_quality']}"
                if str(tick_status.get("spread_quality") or "") not in {"", "two_sided"}
                else str(tick_status.get("freshness_reason") or "quote_not_live_ready")
            )
    elif can_open is True and tick_freshness == "live":
        open_state = "probably_open"
    elif can_open is True:
        open_state = "trade_mode_allows_opening"
    elif can_open is False:
        open_state = mode_status["status"]
    else:
        open_state = "unknown"

    if reason == "weekend":
        message = (
            f"{symbol_name}: closed for the standard Friday 17:00 through Sunday "
            "17:00 America/New_York weekend window even though MT5 trade_mode "
            "allows opening."
        )
    else:
        message = (
            f"{symbol_name}: {open_state.replace('_', ' ')} "
            "(heuristic from MT5 trade_mode and tick freshness)."
        )

    observed_epoch = now_utc.timestamp()
    fetched_epoch = observed_epoch
    result: Dict[str, Any] = {
        "success": True,
        "mode": "symbol",
        "symbol": symbol_name,
        "status": open_state,
        "status_source": "trade_mode_and_tick_freshness",
        "status_confidence": "heuristic",
        "heuristic_note": _symbol_status_heuristic_note(symbol_name),
        "can_open_new_positions": can_open,
        "is_tradable": _coerce_optional_bool(mode_status["is_tradable"]),
        "is_tradable_confidence": "broker_trade_mode",
        "is_tradable_means": "broker_trade_mode",
        "tradable_now": can_open,
        "trade_mode_allows_opening": trade_mode_can_open,
        "trade_mode_label": mode_status.get("trade_mode_label"),
        "tick_freshness": tick_freshness,
        "schedule_source": schedule_status["source"],
        "schedule_confidence": schedule_status["confidence"],
        "current_time_in_recent_session": local_session_open,
        "trades_on_weekends": schedule_status.get("trades_on_weekends"),
        "inferred_24_7": schedule_status.get("inferred_24_7"),
        "session_context": {
            "source": "recent_m1_candles_and_quote_readiness",
            "schedule_source": schedule_status["source"],
            "confidence": schedule_status["confidence"],
            "schedule_match": schedule_match,
            "quote_live_ready": live_ready,
            "local_session_open": local_session_open,
            "trades_on_weekends": schedule_status.get("trades_on_weekends"),
            "inferred_24_7": schedule_status.get("inferred_24_7"),
        },
        "message": message,
        "data_fetched_at": format_epoch_utc(fetched_epoch),
        "wall_clock_observed_at": format_datetime_utc(now_utc),
        "data_fetched_at_basis": "wall_clock",
        "timezone": "UTC",
        "timezone_context": _symbol_market_status_timezone_context(
            timezone_display,
            now_utc=now_utc,
        ),
    }
    if symbol_name != symbol_input:
        result["symbol_input"] = symbol_input
    if reason:
        result["reason"] = reason
    if detail == "full":
        result["trade_mode"] = trade_mode
        result["symbol_info"] = {
            key: getattr(info, key, None)
            for key in (
                "name",
                "description",
                "visible",
                "select",
                "session_deals",
                "session_buy_orders",
                "session_sell_orders",
                "start_time",
                "expiration_time",
            )
            if getattr(info, key, None) is not None
        }
        result["tick"] = tick_status
        result["inferred_schedule"] = schedule_status
    else:
        result.pop("message", None)
        timezone_context = result.pop("timezone_context", {})
        if isinstance(timezone_context, dict):
            market_now = timezone_context.get("market_now")
            if market_now is not None:
                result["market_clock"] = market_now
            status_timezone = timezone_context.get("status_timezone")
            if status_timezone is not None:
                result["market_clock_timezone"] = status_timezone
            authoritative_clock = timezone_context.get("authoritative_clock")
            if authoritative_clock is not None:
                result["authoritative_clock"] = authoritative_clock
        for key in (
            "tick_available",
            "quote_as_of",
            "last_tick_time",
            "data_age_seconds",
            "last_tick_age_seconds",
            "data_stale",
            "usable_for_live_trading",
            "usable_for_live_trading_basis",
            "spread_valid",
            "spread_quality",
            "warning",
            "freshness_reason",
            "timestamp_ahead_of_wall_clock",
            "timestamp_in_future",
            "timestamp_skew_seconds",
            "timestamp_skew_tolerance_seconds",
            "timestamp_warning",
            "quote_source",
            "quote_source_state",
        ):
            if key in tick_status:
                result[key] = tick_status[key]
    return result


def _symbol_status_heuristic_note(symbol_name: str) -> str:
    note = (
        "Symbol status is inferred from MT5 trade_mode, tick freshness, "
        "and recent broker M1 candles; it is not an exchange-calendar guarantee."
    )
    if is_probably_forex_symbol(symbol_name):
        note += (
            " FX weekly sessions typically run Sun 17:00-Fri 17:00 "
            "America/New_York, "
            "subject to broker holidays and session gaps."
        )
    return note


def _symbol_market_status_timezone_context(
    timezone_display: Any,
    *,
    now_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    runtime = build_runtime_timezone_meta({}, include_now=True)
    server = runtime.get("server") if isinstance(runtime.get("server"), dict) else {}
    client = runtime.get("client") if isinstance(runtime.get("client"), dict) else {}
    clock_now = now_utc or datetime.now(timezone.utc)
    authoritative_clock, status_timezone, market_now = _symbol_market_now(
        clock_now,
        display=str(timezone_display or "server"),
        server=server,
        client=client,
    )
    return {
        "timezone_display": str(timezone_display or "server"),
        "authoritative_clock": authoritative_clock,
        "market_now": market_now,
        "status_timezone": status_timezone,
        "server_tz": server.get("tz"),
        "server_now": server.get("now"),
        "client_tz": client.get("tz"),
        "client_now": client.get("now"),
    }


def _split_market_status_symbols(symbols: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for part in str(symbols or "").split(","):
        symbol = part.strip().upper()
        if symbol and symbol not in seen:
            out.append(symbol)
            seen.add(symbol)
    return out


_SYMBOL_STATUS_SHARED_COMPACT_KEYS = (
    "heuristic_note",
    "market_clock",
    "market_clock_timezone",
    "authoritative_clock",
)


def _compact_symbol_market_status(row: Dict[str, Any], *, detail: str) -> Dict[str, Any]:
    if detail == "full":
        return row
    keys = (
        "success",
        "mode",
        "symbol",
        "symbol_input",
        "source",
        "timezone",
        "quote_as_of",
        "data_age_seconds",
        "data_stale",
        "status",
        "status_source",
        "status_confidence",
        "heuristic_note",
        "usable_for_live_trading",
        "tick_freshness",
        "freshness_reason",
        "tick_available",
        "data_fetched_at",
        "data_fetched_at_basis",
        "wall_clock_observed_at",
        "last_tick_time",
        "timestamp_in_future",
        "timestamp_ahead_of_wall_clock",
        "timestamp_warning",
        "market_clock",
        "market_clock_timezone",
        "authoritative_clock",
        "reason",
        "message",
    )
    return {key: row.get(key) for key in keys if row.get(key) is not None}


def _hoist_shared_symbol_status_fields(
    rows: List[Dict[str, Any]],
    *,
    detail: str,
) -> Dict[str, Any]:
    """Move identical methodology/clock fields to the basket payload once."""
    shared: Dict[str, Any] = {}
    if detail == "full" or len(rows) < 2:
        return shared
    for key in _SYMBOL_STATUS_SHARED_COMPACT_KEYS:
        values = [row.get(key) for row in rows if isinstance(row, dict)]
        if len(values) != len(rows):
            continue
        first = values[0]
        if first is None or any(value != first for value in values[1:]):
            continue
        shared[key] = first
        for row in rows:
            row.pop(key, None)
    return shared


def _check_symbol_market_status_batch(
    symbols: List[str],
    *,
    detail: str,
    timezone_display: str,
    allow_partial: bool = True,
    gateway: Any = None,
) -> Dict[str, Any]:
    mt5_gateway = gateway or create_mt5_gateway(
        ensure_connection_impl=ensure_mt5_connection_or_raise
    )
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for symbol in symbols:
        result = _check_symbol_market_status(
            symbol,
            detail=detail,
            timezone_display=timezone_display,
            gateway=mt5_gateway,
        )
        if result.get("error"):
            failure = {"symbol": symbol, "error": result.get("error")}
            if result.get("error_code") not in (None, ""):
                failure["error_code"] = result["error_code"]
            errors.append(failure)
            continue
        rows.append(_compact_symbol_market_status(result, detail=detail))

    shared_fields = _hoist_shared_symbol_status_fields(rows, detail=detail)

    status_counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    can_open_count = sum(1 for row in rows if row.get("can_open_new_positions") is True)
    total = len(symbols)
    failed_count = len(errors)
    succeeded_count = len(rows)
    cannot_open_count = max(0, succeeded_count - can_open_count)
    partial_failure = bool(succeeded_count and failed_count)
    success = bool(succeeded_count) and (bool(allow_partial) or not failed_count)
    if succeeded_count:
        summary = (
            f"{can_open_count}/{succeeded_count} evaluated symbol(s) can open new positions."
        )
        if failed_count:
            summary = (
                f"{summary} {failed_count} symbol(s) unavailable."
            )
    elif failed_count:
        summary = f"0 evaluated symbol(s) can open new positions. {failed_count} symbol(s) unavailable."
    else:
        summary = "0/0 evaluated symbol(s) can open new positions."
    payload: Dict[str, Any] = {
        "success": success,
        "mode": "symbols",
        "symbols": symbols,
        "data": rows,
        "count": succeeded_count,
        "errors": errors if errors else None,
        "failed_items": errors,
        "requested_count": total,
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "can_open_count": can_open_count,
        "cannot_open_count": cannot_open_count,
        "unknown_count": failed_count,
        "partial_failure": partial_failure,
        "allow_partial": bool(allow_partial),
        "summary": summary,
        "status_counts": status_counts,
        "timezone_context": _symbol_market_status_timezone_context(timezone_display),
        **shared_fields,
    }
    if failed_count == total:
        payload.update(
            {
                "error": "Market status failed for all requested symbols.",
                "error_code": "market_status_all_symbols_failed",
            }
        )
    elif partial_failure and not allow_partial:
        payload.update(
            {
                "error": (
                    "Market status was incomplete and allow_partial=false requires "
                    "every requested symbol to succeed."
                ),
                "error_code": "market_status_partial_failure",
            }
        )
    return payload


def _market_status_symbol_mode_warnings(
    *,
    region: Any,
) -> List[str]:
    warnings: List[str] = []
    region_value = str(region or "all").strip().lower()
    if region_value not in {"", "all"}:
        warnings.append("region is ignored when symbol is provided; symbol mode checks broker symbol tradability directly.")
    return warnings


@mcp.tool()
def market_status(  # noqa: C901
    symbol: Optional[str] = None,
    venue: Optional[VenueLiteral] = None,
    region: Literal["us", "europe", "asia", "all"] = "all",
    timezone_display: Literal["local", "utc", "server", "auto"] = "auto",
    detail: DetailLiteral = "compact",
    allow_partial: bool = True,
) -> Dict[str, Any]:
    """Get exchange-calendar status or MT5 symbol tradability.

    Returns the current status (open/closed/pre-market/after-hours/lunch break) for major
    markets including NYSE, NASDAQ, LSE, Xetra, Euronext, Tokyo, Hong Kong,
    Shanghai, and ASX. Handles weekends and holidays correctly.

    Parameters
    ----------
    symbol : str, optional
        Broker symbol to check via MT5 trade mode and tick freshness. When
        supplied, returns a heuristic symbol status instead of the exchange
        overview.
    venue : str, optional
        Exact major-equity venue ID for a single static calendar, such as ASX
        or NYSE. Mutually exclusive with symbol.
    region : str, optional
        Filter by region: "us", "europe", "asia", or "all" (default: "all")
    timezone_display : str, optional
        Time display format: "local" (market's local time), "utc", "server"
        for MT5 symbol mode, or "auto" (default). Auto uses local exchange
        time in global mode and broker/server time in symbol mode.
    detail : {"compact", "standard", "summary", "full"}, optional
        Response detail level. `compact`, `standard`, and `summary` use the
        concise view without per-market messages or upcoming holiday details;
        `full` preserves them.
    allow_partial : bool, optional
        For comma-separated symbol batches, keep usable rows when some symbols
        fail. Explicit lists default permissive. Set false to return
        `success=false` unless every symbol succeeds.

    Returns
    -------
    dict
        Response containing:
        - `data_fetched_at`: Current UTC time (ISO 8601, `Z` suffix)
        - `day_of_week`: Current day name (e.g., "Tuesday")
        - `summary`: Human-readable summary of market statuses (e.g., "1 market open: NYSE; 3 pre-market: LSE, XETRA, EURONEXT; 5 closed")
        - `markets_open`: Count of markets currently open
        - `markets_pre_market`: Count of markets in pre-market
        - `markets_lunch_break`: Count of markets in lunch break
        - `markets_closed`: Count of markets currently closed
        - `upcoming_holidays`: Full holiday rows when `detail='full'`
            - `date`: Holiday date (ISO format)
            - `holiday`: Holiday name
            - `markets_affected`: List of market codes that will be closed
            - `impact`: "closed" or "early_close"
            - `early_close_time`: If early close, the close time (HH:MM)
            - `days_away`: Days from now
        - `markets`: List of market status objects with:
            - `venue`: Market code (e.g., "NYSE")
            - `name`: Full market name
            - `status`: "open", "closed", "pre_market", "after_hours", "lunch_break"
            - `reason`: Reason if closed ("weekend", "holiday", "post_close", "overnight", "before_open")
            - `local_time`: Current exchange-local time (stable across display modes)
            - `exchange_local_time`: Same as `local_time` when display conversion is used
            - `display_time`: Current time in the requested presentation timezone
            - `display_timezone`: IANA or UTC label for `display_time`
            - `message`: Human-readable status in `detail="full"`
            - `next_open` / `next_close`: ISO timestamp of next event
            - `minutes_until_open` / `minutes_until_close`: Minutes until the
              named next event.
            - `early_close`: True on a shortened session.
            - `early_close_time`: Effective local close time (HH:MM) on a
              shortened session.
    """

    detail_mode = normalize_output_verbosity_detail(detail)
    venue_id = str(venue or "").strip().upper()
    symbol_mode = symbol not in (None, "")
    venue_mode = venue_id != ""
    if symbol_mode and venue_mode:
        return {
            "error": "symbol and venue are mutually exclusive.",
            "error_code": "invalid_market_status_scope",
            "symbol": symbol,
            "venue": venue_id,
        }
    if symbol_mode and not venue_mode:
        positional = str(symbol).strip().upper()
        if positional in MARKET_SESSIONS and "," not in str(symbol):
            return {
                "error": (
                    f"'{positional}' is a venue ID, not an MT5 symbol. "
                    f"Use --venue {positional} for the equity session calendar."
                ),
                "error_code": "invalid_market_status_scope",
                "venue": positional,
                "remediation": (
                    f"Use --venue {positional} or pass a broker symbol such as EURUSD."
                ),
            }
    if venue_mode and venue_id not in MARKET_SESSIONS:
        return {
            "error": (
                f"Unknown venue '{venue}'. Valid venues: "
                + ", ".join(MARKET_SESSIONS)
                + "."
            ),
            "error_code": "invalid_venue",
            "venue": venue_id,
            "valid_venues": list(MARKET_SESSIONS),
        }
    region_value = str(region or "all").strip().lower()
    if venue_mode and region_value not in {"", "all"}:
        venue_region = _VENUE_REGIONS.get(venue_id)
        if venue_region is not None and venue_region != region_value:
            return build_error_payload(
                (
                    f"venue '{venue_id}' is in region '{venue_region}', not "
                    f"'{region_value}'."
                ),
                code="incompatible_parameters",
                operation="market_status",
                details={
                    "invalid": ["venue", "region"],
                    "venue": venue_id,
                    "requested_region": region_value,
                    "effective_region": venue_region,
                },
                valid_values={"region": ["all", venue_region]},
                remediation=(
                    f"Omit --region, pass --region {venue_region}, or drop --venue."
                ),
            )
    timezone_display_mode = _normalize_timezone_display(
        timezone_display,
        symbol_mode=symbol_mode,
    )
    if timezone_display_mode is None:
        return {"error": "Invalid timezone_display. Use 'local', 'utc', 'server', or 'auto'."}

    def _run() -> Dict[str, Any]:
        if symbol_mode:
            mt5_gateway = create_mt5_gateway(
                ensure_connection_impl=ensure_mt5_connection_or_raise
            )
            symbol_warnings = _market_status_symbol_mode_warnings(
                region=region,
            )
            symbol_list = _split_market_status_symbols(str(symbol))
            if len(symbol_list) > 1:
                batch_result = _check_symbol_market_status_batch(
                    symbol_list,
                    detail=detail_mode,
                    timezone_display=timezone_display_mode,
                    allow_partial=allow_partial,
                    gateway=mt5_gateway,
                )
                if symbol_warnings:
                    batch_result["warnings"] = symbol_warnings
                return attach_mt5_source(batch_result, gateway=mt5_gateway)
            result = _check_symbol_market_status(
                str(symbol),
                detail=detail_mode,
                timezone_display=timezone_display_mode,
                gateway=mt5_gateway,
            )
            if not result.get("error"):
                result = _compact_symbol_market_status(result, detail=detail_mode)
            if symbol_warnings and not result.get("error"):
                result["warnings"] = symbol_warnings
            return attach_mt5_source(result, gateway=mt5_gateway)

        # Map regions to markets
        region_map = {
            "us": ["NYSE", "NASDAQ"],
            "europe": ["LSE", "XETRA", "EURONEXT"],
            "asia": ["TSE", "HKEX", "SSE", "ASX"],
        }
        
        if venue_mode:
            markets_to_check = [venue_id]
        elif region == "all" or region is None:
            markets_to_check = list(MARKET_SESSIONS.keys())
        else:
            markets_to_check = region_map.get(region, list(MARKET_SESSIONS.keys()))
        
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        
        now_utc = datetime.now(timezone.utc)
        runtime_timezone = build_runtime_timezone_meta({}, include_now=False)
        server_timezone_meta = runtime_timezone.get("server", {})
        server_tzinfo, server_tz_label = _runtime_meta_tzinfo(
            server_timezone_meta,
            allow_offset=True,
        )

        for market_id in markets_to_check:
            if market_id not in MARKET_SESSIONS:
                continue
            
            market = MARKET_SESSIONS[market_id]
            try:
                local_now = _get_local_time(market["timezone"])
                status = _check_market_status(market_id, local_now)
                status["exchange_day_of_week"] = local_now.strftime("%A")
                status = _apply_global_weekend_reason(status, now_local=local_now)
                status = _apply_market_timezone_display(
                    status,
                    now_local=local_now,
                    display=timezone_display_mode,
                    server_tzinfo=server_tzinfo,
                    server_label=server_tz_label,
                    exchange_timezone=str(market.get("timezone") or ""),
                )
                results.append(status)
            except Exception as exc:
                logger.warning(f"Failed to check status for {market_id}: {exc}")
                errors.append({
                    "venue": market_id,
                    "error": str(exc),
                })
        
        # Sort results: open first, then by region
        def _sort_key(item: Dict[str, Any]) -> Tuple[int, str]:
            status_priority = {
                "open": 0,
                "pre_market": 1,
                "after_hours": 2,
                "lunch_break": 3,
                "closed": 4,
            }
            return (status_priority.get(item["status"], 4), item["venue"])
        
        results.sort(key=_sort_key)
        
        # Build summary messages with status breakdown
        status_counts = {
            "open": sum(1 for m in results if m["status"] == "open"),
            "pre_market": sum(1 for m in results if m["status"] == "pre_market"),
            "after_hours": sum(1 for m in results if m["status"] == "after_hours"),
            "lunch_break": sum(1 for m in results if m["status"] == "lunch_break"),
            "closed": sum(1 for m in results if m["status"] == "closed"),
        }
        
        summary_messages = []
        
        # Add open markets (always list them if any)
        if status_counts["open"] > 0:
            open_markets = [m["venue"] for m in results if m["status"] == "open"]
            summary_messages.append(f"{status_counts['open']} market{'s' if status_counts['open'] != 1 else ''} open: {', '.join(open_markets)}")
        
        # Add pre-market markets (always list if any)
        if status_counts["pre_market"] > 0:
            pre_markets = [m["venue"] for m in results if m["status"] == "pre_market"]
            summary_messages.append(f"{status_counts['pre_market']} pre-market: {', '.join(pre_markets)}")

        if status_counts["after_hours"] > 0:
            after_hours_markets = [
                m["venue"] for m in results if m["status"] == "after_hours"
            ]
            summary_messages.append(
                f"{status_counts['after_hours']} after-hours: "
                + ", ".join(after_hours_markets)
            )
        
        # Add lunch break markets (always list if any)
        if status_counts["lunch_break"] > 0:
            lunch_markets = [m["venue"] for m in results if m["status"] == "lunch_break"]
            summary_messages.append(f"{status_counts['lunch_break']} lunch break: {', '.join(lunch_markets)}")
        
        # Add closed markets (list if <= 3, otherwise just count)
        if status_counts["closed"] > 0:
            closed_markets = [m["venue"] for m in results if m["status"] == "closed"]
            if status_counts["closed"] <= 3:
                summary_messages.append(f"{status_counts['closed']} closed: {', '.join(closed_markets)}")
            else:
                summary_messages.append(f"{status_counts['closed']} closed")
        
        reason_counts: Dict[str, int] = {}
        for market in results:
            if market.get("status") == "closed" and market.get("reason"):
                reason = str(market.get("reason"))
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        global_status = None
        if results and status_counts["closed"] == len(results) and reason_counts.get("weekend") == len(results):
            global_status = "weekend"

        # Get upcoming holidays impacting these markets
        upcoming_holidays = _get_upcoming_holidays(markets_to_check)

        exchange_days = {
            str(market.get("exchange_day_of_week"))
            for market in results
            if market.get("exchange_day_of_week")
        }
        uses_country_fallback = any(
            not (MARKET_SESSIONS.get(market_id) or {}).get("exchange_calendar")
            for market_id in markets_to_check
        )
        euronext_only = bool(markets_to_check) and all(
            str((MARKET_SESSIONS.get(market_id) or {}).get("exchange_calendar") or "")
            .strip()
            .upper()
            in {"EURONEXT", "XPAR"}
            for market_id in markets_to_check
        )
        payload = {
            "success": True,
            "source": {
                "provider": (
                    "mtdata_market_sessions"
                    if uses_country_fallback
                    else "mtdata_exchange_calendar"
                ),
                "holiday_provider": (
                    "mtdata_euronext_paris"
                    if euronext_only
                    else (
                        "python_holidays.country_fallback"
                        if uses_country_fallback
                        else "python_holidays"
                    )
                ),
                "context_available": True,
            },
            "mode": "equity_venue" if venue_mode else "equity_exchanges",
            "market_scope": (
                "single_equity_venue" if venue_mode else "major_equity_exchanges"
            ),
            "scope_note": (
                f"Explicit static exchange-calendar view for venue {venue_id}."
                if venue_mode
                else (
                    "This no-symbol view covers major equity exchanges only; pass a "
                    "broker symbol for MT5 tradability and quote-freshness status."
                )
            ),
            "requested_venue": venue_id if venue_mode else None,
            "data_fetched_at": format_datetime_utc(now_utc),
            "timezone": "UTC",
            "timezone_display": timezone_display_mode,
            "display_timezone": (
                server_tz_label
                if timezone_display_mode == "server" and server_tz_label
                else "UTC"
                if timezone_display_mode in {"utc", "server"}
                else "market_local"
            ),
            "day_of_week": next(iter(exchange_days)) if len(exchange_days) == 1 else "mixed",
            "day_of_week_basis": "exchange_local",
            "region": region or "all",
            "summary": "; ".join(summary_messages) if summary_messages else "No market data available",
            "markets_open": status_counts["open"],
            "markets_closed": status_counts["closed"],
            "markets_pre_market": status_counts["pre_market"],
            "markets_after_hours": status_counts["after_hours"],
            "markets_lunch_break": status_counts["lunch_break"],
            "markets": results,
            "upcoming_holidays": upcoming_holidays if upcoming_holidays else None,
            "errors": errors if errors else None,
        }
        if reason_counts:
            payload["closed_reason_counts"] = reason_counts
        if global_status:
            payload["global_status"] = global_status
        return normalize_market_status_output(
            payload,
            detail=detail_mode,
        )

    return run_logged_operation(
        logger,
        operation="market_status",
            symbol=symbol,
            venue=venue_id or None,
            region=region,
            timezone_display=timezone_display_mode,
            detail=detail_mode,
            func=_run,
        )
