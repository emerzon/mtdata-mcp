import math
import re
from datetime import datetime, timedelta, timezone
from numbers import Number
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import dateparser
import numpy as np
import pandas as pd

from .coercion import coerce_cli_scalar
from .formatting import (
    format_float,
    format_number,
    optimal_decimals,
)
from .time import as_utc


def _positive_float_attr(obj: Any, *names: str) -> Optional[float]:
    """Return the first finite, strictly-positive float among *names* on *obj*.

    Tries each attribute name in order, accepting only real numeric values
    (``int``/``float``, excluding ``bool``) and skipping non-numeric,
    non-finite, and non-positive values.  Returns ``None`` when no attribute
    yields a positive float.
    """
    if obj is None:
        return None
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric) and numeric > 0.0:
            return numeric
    return None


def _normalize_ohlcv_arg(ohlcv: Optional[str]) -> Optional[Set[str]]:
    """Normalize user-provided OHLCV selection into a set of letters.

    Accepts forms like: 'close', 'price', 'ohlc', 'ohlcv', 'all', 'cl', 'OHLCV',
    or names 'open,high,low,close,volume'. Returns None when not specified.
    """
    if ohlcv is None:
        return None
    text = str(ohlcv).strip()
    if text == "":
        return None
    t = text.lower()
    if t in ("all", "ohlcv"):
        return {"O", "H", "L", "C", "V"}
    if t in ("ohlc",):
        return {"O", "H", "L", "C"}
    if t in ("price", "close"):
        return {"C"}
    mapping = {
        "o": "O", "open": "O",
        "h": "H", "high": "H",
        "l": "L", "low": "L",
        "c": "C", "close": "C", "price": "C",
        "v": "V", "vol": "V", "volume": "V", "tick_volume": "V",
    }
    if t in mapping:
        return {mapping[t]}
    # Compact letters like 'cl', 'oh', etc.
    if all(ch in "ohlcv" for ch in t):
        return {ch.upper() for ch in t}
    # Comma separated names
    parts = [p.strip().lower() for p in t.replace(";", ",").split(",") if p.strip() != ""]
    if not parts:
        return None
    out: Set[str] = set()
    unknown: list[str] = []
    for p in parts:
        key = mapping.get(p)
        if key:
            out.add(key)
        else:
            unknown.append(p)
    if unknown:
        valid = "open, high, low, close, volume (or o,h,l,c,v)"
        raise ValueError(
            f"Invalid ohlcv token(s): {', '.join(unknown)}. Use {valid}."
        )
    return out or None


def _normalize_limit(limit: Optional[Any]) -> Optional[int]:
    try:
        if limit is None:
            return None
        if isinstance(limit, str):
            limit = limit.strip()
            if not limit:
                return None
        value = int(float(limit))
        return value if value > 0 else None
    except Exception:
        return None


def _table_from_rows(headers: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    """Build a normalized tabular payload for results.

    Returns a dict with at least:
    - data: list[dict] rows (keys follow the provided headers order)
    - success: True
    - count: number of data rows
    """
    cols = [str(h) for h in (headers or [])]
    items: List[Dict[str, Any]] = []
    for row in rows or []:
        item: Dict[str, Any] = {}
        for idx, col in enumerate(cols):
            item[col] = row[idx] if idx < len(row) else None
        items.append(item)
    return {
        "data": items,
        "row_key": "data",
        "success": True,
        "count": len(items),
    }

def parse_kv_or_json(obj: Any) -> Dict[str, Any]:
    """Parse params/features provided as dict, JSON string, or k=v pairs into a dict.

    - Dict: shallow-copied and returned
    - JSON-like string: parsed via json.loads (dict or list-of-pairs)
    - Plain string: split on whitespace/commas into k=v assignments
    """
    import json

    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return {}
        if s.startswith(('{', '[')):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return dict(parsed)
                # Accept list-of-pairs JSON (e.g., [["k","v"],["k2","v2"]])
                if isinstance(parsed, list):
                    out_pairs: Dict[str, Any] = {}
                    ok = True
                    for item in parsed:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            out_pairs[str(item[0])] = item[1]
                        else:
                            ok = False
                            break
                    if ok:
                        return out_pairs
                raise ValueError("JSON mapping input must be an object or list of pairs.")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON mapping: {exc.msg}.") from exc
        # Parse k=v / k:v assignments. Commas split assignments only when a new key follows.
        import re
        if re.fullmatch(r"[A-Za-z]:[\\/].*", s):
            return {}
        out: Dict[str, Any] = {}
        pair_pattern = re.compile(
            r'(?:^|[\s,])([A-Za-z_][\w.\-]*)\s*([=:])\s*(.*?)\s*(?=(?:[\s,]+[A-Za-z_][\w.\-]*\s*[=:])|$)'
        )
        matches = list(pair_pattern.finditer(s))
        if not matches:
            raise ValueError("Malformed mapping; use JSON or complete key=value pairs.")

        cursor = 0
        for m in matches:
            unmatched = s[cursor:m.start()]
            if unmatched.strip(" \t\r\n,"):
                raise ValueError(
                    f"Malformed mapping fragment: {unmatched.strip()!r}; "
                    "use complete key=value pairs."
                )
            k = str(m.group(1) or '').strip()
            v = str(m.group(3) or '').strip().strip(',')
            # Avoid Windows drive paths like "C:\foo".
            if len(k) == 1 and v.startswith(("\\", "/")):
                raise ValueError(f"Malformed mapping fragment: {m.group(0).strip()!r}.")
            if not v or v.startswith(("=", ":")):
                raise ValueError(
                    f"Malformed value for {k!r}; use exactly one assignment operator."
                )
            if k in out:
                raise ValueError(f"Duplicate mapping key: {k!r}.")
            out[k] = coerce_cli_scalar(v)
            cursor = m.end()
        trailing = s[cursor:]
        if trailing.strip(" \t\r\n,"):
            raise ValueError(
                f"Malformed mapping fragment: {trailing.strip()!r}; "
                "use complete key=value pairs."
            )
        if trailing.count(",") > 1 or ",," in s:
            raise ValueError("Malformed mapping delimiter; remove duplicate commas.")
        return out
    return {}


def _format_numeric_rows_from_df(
    df: pd.DataFrame,
    headers: List[str],
    *,
    stringify: bool = True,
) -> List[List[Any]]:
    if not stringify:
        # Public numeric row modes do not need adaptive display decimals. Keep
        # the conversion columnar while preserving NaN/Inf sentinels used by
        # indicator bands and internal callers.
        return df.loc[:, headers].to_numpy(dtype=object).tolist()

    # Precompute per-column decimals to trim numeric noise without losing precision.
    col_decimals: Dict[str, int] = {}
    for col in headers:
        if col == 'time' or col not in df.columns:
            continue
        try:
            series = pd.to_numeric(df[col], errors="coerce")
            values = [
                float(v)
                for v in series
                if v is not None and not pd.isna(v) and math.isfinite(v)
            ]
        except Exception:
            values = []
        if values:
            col_decimals[col] = optimal_decimals(values)

    out_rows: List[List[Any]] = []
    for row_values in df[headers].itertuples(index=False, name=None):
        out_row: List[Any] = []
        for col, val in zip(headers, row_values):
            if col == 'time':
                out_row.append(str(val) if stringify else val)
            elif val is None or isinstance(val, bool):
                out_row.append(format_number(val) if stringify else val)
            elif isinstance(val, Number):
                try:
                    num = float(val)
                except Exception:
                    out_row.append(str(val) if stringify else val)
                    continue
                if not math.isfinite(num):
                    out_row.append(format_number(num) if stringify else num)
                    continue
                decimals = col_decimals.get(col)
                if decimals is None:
                    out_row.append(format_number(num))
                else:
                    out_row.append(format_float(num, decimals))
            else:
                out_row.append(str(val))
        out_rows.append(out_row)
    return out_rows

def to_float_np(
    values: Any,
    *,
    coerce: bool = True,
    drop_na: bool = False,
    finite_only: bool = False,
    return_mask: bool = False,
) -> "np.ndarray | Tuple[np.ndarray, 'np.ndarray']":
    """Convert a pandas Series/array-like to a float NumPy array.

    - coerce=True uses `pd.to_numeric(errors='coerce')` to convert invalids to NaN.
    - drop_na=True removes NaN entries from the returned array (mask applied).
    - finite_only=True removes non-finite entries (NaN, inf, -inf).
    - return_mask=True returns (array, mask) where mask marks kept elements.

    Notes: When both drop_na and finite_only are False, the original length is preserved.
    """
    try:
        ser = pd.Series(values)

        arr = (
            pd.to_numeric(ser, errors="coerce").astype(float).to_numpy()
            if coerce
            else ser.astype(float).to_numpy()
        )

        mask = None
        if drop_na or finite_only:
            if finite_only:
                mask = np.isfinite(arr)
            else:
                mask = ~pd.isna(arr)
            arr = arr[mask]
        if return_mask:
            if mask is None:
                mask = np.ones(arr.shape, dtype=bool)
            return arr, mask
        return arr
    except Exception:
        # Fallbacks
        try:
            arr = np.asarray(values, dtype=float)
            if drop_na or finite_only:
                m = np.isfinite(arr) if finite_only else ~pd.isna(arr)
                arr = arr[m]
                if return_mask:
                    return arr, m
            elif return_mask:
                return arr, np.ones(arr.shape, dtype=bool)
            return arr
        except Exception:
            empty = np.asarray([], dtype=float)
            if return_mask:
                return empty, np.asarray([], dtype=bool)
            return empty


def align_finite(*arrays: Any) -> Tuple["np.ndarray", ...]:
    """Convert arrays to float and align them by keeping only rows where all are finite.

    Returns a tuple of filtered arrays, all of equal length.
    """
    conv = [to_float_np(a) for a in arrays]
    if not conv:
        return tuple()
    mask = np.ones_like(conv[0], dtype=bool)
    for a in conv:
        mask &= np.isfinite(a)
    return tuple(a[mask] for a in conv)


def _resolve_iana_timezone_datetime(
    value: str,
) -> tuple[Optional[datetime], Optional[Dict[str, Any]]]:
    """Resolve an IANA-zone local time or describe its DST transition conflict."""
    try:
        local_text, timezone_name = str(value).strip().rsplit(maxsplit=1)
    except ValueError:
        return None, None
    if "/" not in timezone_name:
        return None, None
    try:
        local_zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        return None, None
    local_time = dateparser.parse(
        local_text,
        settings={
            "RETURN_AS_TIMEZONE_AWARE": False,
            "PREFER_DAY_OF_MONTH": "first",
        },
    )
    if local_time is None or local_time.tzinfo is not None:
        return None, None

    # A named local time at a daylight-saving transition can be nonexistent or
    # ambiguous. Do not silently select a different instant for either case.
    fold_zero = local_time.replace(tzinfo=local_zone, fold=0)
    fold_one = local_time.replace(tzinfo=local_zone, fold=1)
    roundtrip_zero = fold_zero.astimezone(timezone.utc).astimezone(local_zone)
    roundtrip_one = fold_one.astimezone(timezone.utc).astimezone(local_zone)
    zero_is_valid = roundtrip_zero.replace(tzinfo=None) == local_time
    one_is_valid = roundtrip_one.replace(tzinfo=None) == local_time
    if fold_zero.utcoffset() != fold_one.utcoffset():
        value_text = str(value).strip()
        if zero_is_valid and one_is_valid:
            choices = [fold_zero.isoformat(), fold_one.isoformat()]
            return None, {
                "success": False,
                "error": (
                    f"Local time {local_text!r} is ambiguous in {timezone_name}: "
                    "it occurs twice when daylight-saving time ends."
                ),
                "error_code": "ambiguous_local_time",
                "details": {
                    "value": value_text,
                    "timezone": timezone_name,
                    "offset_choices": choices,
                },
                "remediation": (
                    "Choose the intended instant with an explicit ISO 8601 offset, "
                    f"for example {choices[0]} or {choices[1]}."
                ),
            }
        nearby = sorted(
            {
                roundtrip_zero.replace(fold=0).isoformat(),
                roundtrip_one.replace(fold=0).isoformat(),
            }
        )
        return None, {
            "success": False,
            "error": (
                f"Local time {local_text!r} does not exist in {timezone_name}: "
                "the clock skips it when daylight-saving time starts."
            ),
            "error_code": "nonexistent_local_time",
            "details": {
                "value": value_text,
                "timezone": timezone_name,
                "nearest_valid_local_times": nearby,
            },
            "remediation": (
                "Choose an existing local time with an explicit ISO 8601 offset, "
                f"for example {nearby[-1]}."
            ),
        }
    utc_time = fold_zero.astimezone(timezone.utc)
    if not zero_is_valid:
        return None, None
    return utc_time.replace(tzinfo=None), None


def _iana_timezone_datetime_issue(value: str) -> Optional[Dict[str, Any]]:
    """Return a structured DST issue for an otherwise valid IANA local time."""
    _, issue = _resolve_iana_timezone_datetime(value)
    return issue


def _parse_iana_timezone_datetime(value: str) -> Optional[datetime]:
    """Parse a local datetime ending in an IANA zone name into naive UTC."""
    parsed, _ = _resolve_iana_timezone_datetime(value)
    return parsed


def _calendar_period_bounds(
    value: str,
    *,
    now: Optional[datetime] = None,
    calendar_timezone: Any = timezone.utc,
) -> Optional[tuple[datetime, datetime, str]]:
    """Resolve supported natural-language calendar periods to inclusive bounds."""
    text = " ".join(str(value or "").strip().lower().split())
    if not text:
        return None
    current = now or datetime.now(timezone.utc)
    current_date = current.astimezone(calendar_timezone or timezone.utc).date()
    period_start = None
    kind = "day"
    if text in {"today", "yesterday", "tomorrow"}:
        offset = {"yesterday": -1, "today": 0, "tomorrow": 1}[text]
        period_start = datetime.combine(
            current_date + timedelta(days=offset),
            datetime.min.time(),
        )
    elif text in {"last week", "this week", "next week"}:
        offset_weeks = {"last week": -1, "this week": 0, "next week": 1}[text]
        monday = current_date - timedelta(days=current_date.weekday())
        period_start = datetime.combine(
            monday + timedelta(weeks=offset_weeks),
            datetime.min.time(),
        )
        kind = "week"
    elif text in {"last month", "this month", "next month"}:
        month_offset = {"last month": -1, "this month": 0, "next month": 1}[text]
        month_index = current_date.year * 12 + current_date.month - 1 + month_offset
        year, zero_based_month = divmod(month_index, 12)
        period_start = datetime(year, zero_based_month + 1, 1)
        kind = "month"
    elif text in {"last year", "this year", "next year"}:
        year_offset = {"last year": -1, "this year": 0, "next year": 1}[text]
        period_start = datetime(current_date.year + year_offset, 1, 1)
        kind = "year"
    else:
        parts = text.split()
        weekdays = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        if len(parts) != 2 or parts[0] not in {"last", "next"} or parts[1] not in weekdays:
            return None
        target_weekday = weekdays[parts[1]]
        if parts[0] == "next":
            days = (target_weekday - current_date.weekday()) % 7 or 7
        else:
            days = -((current_date.weekday() - target_weekday) % 7 or 7)
        period_start = datetime.combine(
            current_date + timedelta(days=days),
            datetime.min.time(),
        )
    if kind == "week":
        period_end_exclusive = period_start + timedelta(weeks=1)
    elif kind == "month":
        next_month_index = period_start.year * 12 + period_start.month
        next_year, next_zero_based_month = divmod(next_month_index, 12)
        period_end_exclusive = datetime(next_year, next_zero_based_month + 1, 1)
    elif kind == "year":
        period_end_exclusive = datetime(period_start.year + 1, 1, 1)
    else:
        period_end_exclusive = period_start + timedelta(days=1)
    return period_start, period_end_exclusive - timedelta(microseconds=1), kind


def _is_calendar_period_expression(value: Optional[str]) -> bool:
    return bool(value and _calendar_period_bounds(str(value)) is not None)


def _parse_start_datetime(value: str) -> Optional[datetime]:
    """Parse a date/time string, including IANA zone names, into naive UTC."""
    if not value:
        return None
    text = str(value).strip()
    calendar_period = _calendar_period_bounds(text)
    if calendar_period is not None:
        return calendar_period[0]
    named_timezone_datetime = _parse_iana_timezone_datetime(text)
    if named_timezone_datetime is not None:
        return named_timezone_datetime
    # ISO-shaped values are an automation contract, not natural language.
    # Validate their calendar token before dateparser can reinterpret an
    # impossible month as a day (for example, 2026-13-01 -> 2026-01-13).
    if re.match(r"^\d{4}-\d{2}-\d{2}(?:$|[T ])", text):
        try:
            datetime.fromisoformat(text[:10])
        except ValueError:
            return None
        normalized_iso = re.sub(r"\s+UTC$", "+00:00", text, flags=re.IGNORECASE)
        if normalized_iso.endswith(("Z", "z")):
            normalized_iso = normalized_iso[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(normalized_iso)
        except ValueError:
            return None
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    dt = dateparser.parse(
        value,
        settings={
            'RETURN_AS_TIMEZONE_AWARE': True,
            'TIMEZONE': 'UTC',
            'TO_TIMEZONE': 'UTC',
            'PREFER_DAY_OF_MONTH': 'first',
        },
    )
    if not dt:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _parse_end_datetime(value: str) -> Optional[datetime]:
    """Parse an inclusive range end, expanding calendar periods to their end."""
    calendar_period = _calendar_period_bounds(value)
    if calendar_period is not None:
        return calendar_period[1]
    parsed = _parse_start_datetime(value)
    if parsed is None:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value).strip()):
        return parsed + timedelta(days=1) - timedelta(microseconds=1)
    return parsed


def _is_in_progress_calendar_day_end(
    value: Optional[str],
    end_dt: datetime,
    now_naive: datetime,
) -> bool:
    """True for an inclusive current-day bound such as YYYY-MM-DD or 'today'."""
    if end_dt.date() != now_naive.date():
        return False
    text = str(value or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return True
    if re.search(r"T\d{1,2}:\d{2}", text) or re.search(r"\s\d{1,2}:\d{2}", text):
        return False
    return bool(text)


def validate_historical_range(
    start: Optional[str],
    end: Optional[str],
    *,
    now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """Return a stable validation error for invalid or future-only history bounds."""
    start_dt = _parse_start_datetime(start) if start else None
    end_dt = _parse_end_datetime(end) if end else None
    invalid_fields = []
    if start and start_dt is None:
        if issue := _iana_timezone_datetime_issue(start):
            issue = dict(issue)
            issue["details"] = {
                **dict(issue.get("details") or {}),
                "field": "start",
            }
            return issue
        invalid_fields.append({"field": "start", "value": str(start)[:200]})
    if end and end_dt is None:
        if issue := _iana_timezone_datetime_issue(end):
            issue = dict(issue)
            issue["details"] = {
                **dict(issue.get("details") or {}),
                "field": "end",
            }
            return issue
        invalid_fields.append({"field": "end", "value": str(end)[:200]})
    if invalid_fields:
        invalid_text = ", ".join(
            f"{item['field']}={item['value']!r}" for item in invalid_fields
        )
        return {
            "success": False,
            "error": (
                f"Could not parse historical datetime bound(s): {invalid_text}. "
                "Accepted formats include YYYY-MM-DD, ISO 8601 timestamps such as "
                "2026-08-12T14:30:00Z, and supported natural calendar periods."
            ),
            "error_code": "invalid_datetime",
            "details": {"invalid_fields": invalid_fields},
            "remediation": (
                "Correct the listed start/end value using an ISO 8601 date or "
                "timestamp."
            ),
        }
    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        return {
            "success": False,
            "error": "start must be before or equal to end.",
            "error_code": "invalid_date_range",
        }
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    now_naive = now_utc.replace(tzinfo=None)
    if start_dt is not None and start_dt > now_naive:
        return {
            "success": False,
            "error": (
                f"start datetime {start_dt.isoformat()}Z is in the future; "
                "no historical data is available for future dates."
            ),
            "error_code": "future_date_range",
            "details": {
                "resolved_start": f"{start_dt.isoformat()}Z",
                "resolved_end": f"{end_dt.isoformat()}Z" if end_dt else None,
                "current_time": now_utc.isoformat(),
            },
            "remediation": "Choose a start datetime at or before the current time.",
        }
    if end_dt is not None and end_dt > now_naive:
        if not _is_in_progress_calendar_day_end(end, end_dt, now_naive):
            return {
                "success": False,
                "error": (
                    f"end datetime {end_dt.isoformat()}Z is in the future; "
                    "historical ranges must have elapsed."
                ),
                "error_code": "future_date_range",
                "details": {
                    "resolved_start": f"{start_dt.isoformat()}Z" if start_dt else None,
                    "resolved_end": f"{end_dt.isoformat()}Z",
                    "current_time": now_utc.isoformat(),
                },
                "remediation": "Choose an end datetime at or before the current time.",
            }
    return None


def _utc_epoch_seconds(dt: datetime) -> float:
    """Convert a datetime to UTC epoch seconds, treating naive values as UTC.

    Python's `datetime.timestamp()` interprets naive datetimes as *local time*,
    which can silently shift values when the host isn't running in UTC.
    """
    return as_utc(dt).timestamp()

