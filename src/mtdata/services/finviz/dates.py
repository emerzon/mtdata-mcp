"""Date normalization utilities for Finviz service."""
import datetime
import re
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

FINVIZ_CALENDAR_TIMEZONE = "America/New_York"
_FINVIZ_CALENDAR_TZ = ZoneInfo(FINVIZ_CALENDAR_TIMEZONE)


def _finviz_market_date(now: Optional[datetime.datetime] = None) -> datetime.date:
    """Return the current Finviz calendar date in US Eastern time."""
    current = now or datetime.datetime.now(datetime.timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=datetime.timezone.utc)
    return current.astimezone(_FINVIZ_CALENDAR_TZ).date()


def _finviz_market_time(now: Optional[datetime.datetime] = None) -> datetime.time:
    """Return the current wall-clock time in the Finviz calendar timezone."""
    current = now or datetime.datetime.now(datetime.timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=datetime.timezone.utc)
    return current.astimezone(_FINVIZ_CALENDAR_TZ).time().replace(tzinfo=None)


def parse_iso_date_input(value: Any, *, field_name: str) -> datetime.date:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Invalid {field_name} '{value}'. Expected YYYY-MM-DD or ISO datetime")
    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        return datetime.date.fromisoformat(normalized)
    except ValueError:
        pass
    try:
        return datetime.datetime.fromisoformat(normalized).date()
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field_name} '{value}'. Expected YYYY-MM-DD or ISO datetime"
        ) from exc


def normalize_finviz_date_string(value: Any) -> Any:
    """Normalize Finviz short dates like `Nov 07 '25` to ISO 8601."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    text = text.replace("’", "'")
    for fmt in ("%b %d '%y", "%b %d %Y"):
        try:
            return datetime.datetime.strptime(text, fmt).date().isoformat()
        except Exception:
            continue
    try:
        return parse_iso_date_input(text, field_name="date").isoformat()
    except ValueError:
        pass
    return value


def parse_finviz_publication_date(
    value: Any,
    *,
    now: Optional[datetime.datetime] = None,
) -> Optional[datetime.date]:
    """Parse provider values that contain a date but no publication time.

    Yearless values resolve to their most recent occurrence in the Finviz
    market timezone. They remain dates and must not be promoted to midnight
    timestamps by callers.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return datetime.date.fromisoformat(text)
        except ValueError:
            return None
    reference = _finviz_market_date(now)
    for separator, fmt in (("-", "%b-%d-%Y"), (" ", "%b %d %Y")):
        try:
            candidate = datetime.datetime.strptime(
                f"{text}{separator}{reference.year}",
                fmt,
            ).date()
        except ValueError:
            continue
        if candidate > reference:
            candidate = candidate.replace(year=candidate.year - 1)
        return candidate
    return None


def parse_finviz_datetime(
    value: Any,
    *,
    allow_fuzzy: bool = False,
) -> Optional[datetime.datetime]:
    """Parse a Finviz wall-clock value and return an aware UTC datetime."""
    if isinstance(value, datetime.datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text or re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
            return None
        try:
            parsed = datetime.datetime.fromisoformat(text)
        except ValueError:
            parsed = None
            for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",
                "%b %d '%y",
                "%b %d %Y",
            ):
                try:
                    parsed = datetime.datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            if parsed is None and allow_fuzzy:
                try:
                    from dateutil import parser as date_parser

                    parsed = date_parser.parse(text)
                except Exception:
                    return None
            if parsed is None:
                return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_FINVIZ_CALENDAR_TZ)
    return parsed.astimezone(datetime.timezone.utc)


def normalize_finviz_dates_in_rows(
    rows: List[Dict[str, Any]], *keys: str
) -> List[Dict[str, Any]]:
    """Normalize date strings in specified columns of row dictionaries."""
    out: List[Dict[str, Any]] = []
    wanted = set(keys)
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_out = dict(row)
        for key in wanted:
            if key in row_out:
                row_out[key] = normalize_finviz_date_string(row_out.get(key))
        out.append(row_out)
    return out


def finviz_earnings_period_window(
    period_key: str,
    reference_date: datetime.date,
) -> tuple[datetime.date, datetime.date]:
    """Resolve the calendar bounds used to interpret yearless earnings dates."""
    week_start = reference_date - datetime.timedelta(days=reference_date.weekday())
    if period_key == "next-week":
        start = week_start + datetime.timedelta(days=7)
        return start, start + datetime.timedelta(days=6)
    if period_key == "previous-week":
        start = week_start - datetime.timedelta(days=7)
        return start, start + datetime.timedelta(days=6)
    if period_key == "this-month":
        start = reference_date.replace(day=1)
        next_month = (
            start.replace(year=start.year + 1, month=1)
            if start.month == 12
            else start.replace(month=start.month + 1)
        )
        return start, next_month - datetime.timedelta(days=1)
    return week_start, week_start + datetime.timedelta(days=6)


def parse_finviz_earnings_date(
    value: Any,
    *,
    reference_date: Optional[datetime.date] = None,
    period_window: Optional[tuple[datetime.date, datetime.date]] = None,
) -> Optional[datetime.date]:
    """Parse a Finviz earnings token such as ``Aug 12/a`` into a date."""
    if value in (None, ""):
        return None
    date_part = str(value).strip().split("/", 1)[0].strip()
    if not date_part:
        return None
    try:
        return datetime.date.fromisoformat(date_part)
    except ValueError:
        pass

    reference = reference_date or _finviz_market_date()
    candidates: List[datetime.date] = []
    for year in (reference.year - 1, reference.year, reference.year + 1):
        for fmt in ("%b %d %Y", "%B %d %Y"):
            try:
                parsed = datetime.datetime.strptime(
                    f"{date_part} {year}", fmt
                ).date()
            except ValueError:
                continue
            candidates.append(parsed)
            break
    if period_window is not None:
        start, end = period_window
        within = [candidate for candidate in candidates if start <= candidate <= end]
        if within:
            return min(within, key=lambda candidate: abs(candidate - reference))
        return None
    if candidates:
        return min(candidates, key=lambda candidate: abs(candidate - reference))
    return None


def strip_string_fields_in_rows(
    rows: List[Dict[str, Any]], *keys: str
) -> List[Dict[str, Any]]:
    """Strip whitespace from specified string columns in row dictionaries."""
    out: List[Dict[str, Any]] = []
    wanted = set(keys)
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_out = dict(row)
        for key in wanted:
            value = row_out.get(key)
            if isinstance(value, str):
                row_out[key] = value.strip()
        out.append(row_out)
    return out


def resolve_date_range(
    *, date_from: Optional[str], date_to: Optional[str], default_days: int = 7
) -> tuple[str, str]:
    """Resolve an ISO date range for Finviz API calls.

    Defaults: ``date_from`` to the current America/New_York date when omitted and
    ``date_to`` to ``date_from + default_days`` when omitted. Enforces
    ``date_to >= date_from``. Error messages use the public ``start``/``end``
    names rather than provider field aliases.
    """
    if date_from:
        df = parse_iso_date_input(date_from, field_name="start")
        from_str = df.isoformat()
    else:
        df = _finviz_market_date()
        from_str = df.isoformat()

    if date_to:
        dt = parse_iso_date_input(date_to, field_name="end")
        to_str = dt.isoformat()
    else:
        dt = df + datetime.timedelta(days=int(default_days))
        to_str = dt.isoformat()

    if dt < df:
        raise ValueError("end must be on or after start")

    return from_str, to_str


def align_to_next_monday_if_weekend(date_from: str) -> str:
    """Align date to next Monday if it falls on weekend.

    Finviz economic calendar API appears to anchor by week; weekend anchors
    often return the prior week.
    """
    d = parse_iso_date_input(date_from, field_name="date_from")
    # If Saturday (5) or Sunday (6), move to next Monday
    if d.weekday() == 5:  # Saturday
        d = d + datetime.timedelta(days=2)
    elif d.weekday() == 6:  # Sunday
        d = d + datetime.timedelta(days=1)
    return d.isoformat()


__all__ = [
    "FINVIZ_CALENDAR_TIMEZONE",
    "parse_iso_date_input",
    "normalize_finviz_date_string",
    "normalize_finviz_dates_in_rows",
    "parse_finviz_publication_date",
    "parse_finviz_datetime",
    "finviz_earnings_period_window",
    "parse_finviz_earnings_date",
    "strip_string_fields_in_rows",
    "resolve_date_range",
    "align_to_next_monday_if_weekend",
]
