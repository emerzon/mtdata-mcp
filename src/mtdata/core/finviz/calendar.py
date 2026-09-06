"""Finviz economic, earnings, and dividend calendar adapters."""

import re
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
)
from zoneinfo import ZoneInfo

from pydantic import Field

from mtdata.core.error_envelope import build_error_payload
from mtdata.core.finviz.common import (
    _FINVIZ_CALENDAR_LOCAL_TIMEZONE,
    _FINVIZ_CALENDAR_LOCAL_TZ,
    _FINVIZ_NUMERIC_SUFFIX_MULTIPLIERS,
    _append_finviz_warning,
    _apply_finviz_pagination_contract,
    _attach_finviz_delayed_root_metadata,
    _build_tool_contract_meta,
    _canonicalize_finviz_market_row,
    _finviz_percent_value,
    _finviz_screen_units_for_rows,
    _format_finviz_large_number,
    _mark_finviz_delayed_price,
    _normalize_finviz_fundamental_value,
    _normalize_finviz_output_key,
    _normalize_finviz_output_rows,
    _parse_finviz_numeric_value,
    _run_logged_tool,
    _validate_finviz_detail,
)
from mtdata.core.output_contract import (
    normalize_output_detail,
    normalize_output_verbosity_detail,
)
from mtdata.services.finviz import (
    get_dividends_calendar_api,
    get_earnings_calendar,
    get_earnings_calendar_api,
    get_economic_calendar,
)
from mtdata.services.finviz.dates import (
    finviz_earnings_period_window,
    finviz_timestamp_is_date_only,
    parse_finviz_earnings_date,
)
from mtdata.shared.schema import DetailLiteral
from mtdata.utils.time import format_datetime_utc
from mtdata.utils.utils import (
    _parse_end_datetime,
    _parse_start_datetime,
)


def _finviz_earnings_error_code(message: str) -> str:
    text = str(message or "")
    if "rate limit" in text.lower():
        return "finviz_rate_limited"
    if "Invalid period" in text:
        return "finviz_earnings_invalid_period"
    if "No earnings calendar data available" in text:
        return "finviz_earnings_no_data"
    return "finviz_earnings_failed"


_FINVIZ_EARNINGS_PERIODS = {
    "this-week": "This Week",
    "next-week": "Next Week",
    "previous-week": "Previous Week",
    "this-month": "This Month",
}
_FINVIZ_EARNINGS_PERIOD_ALIASES = {
    value.lower(): key for key, value in _FINVIZ_EARNINGS_PERIODS.items()
}


def _normalize_finviz_earnings_period(value: Any) -> Optional[tuple[str, str]]:
    text = str(value or "").strip()
    key = text.lower()
    key = key.replace("_", "-")
    if key in _FINVIZ_EARNINGS_PERIODS:
        return key, _FINVIZ_EARNINGS_PERIODS[key]
    label_key = text.lower()
    if label_key in _FINVIZ_EARNINGS_PERIOD_ALIASES:
        canonical = _FINVIZ_EARNINGS_PERIOD_ALIASES[label_key]
        return canonical, _FINVIZ_EARNINGS_PERIODS[canonical]
    return None


_FINVIZ_EARNINGS_COMPACT_FIELDS = (
    "symbol",
    "company",
    "earnings_date",
    "earnings",
    "earnings_timing",
    "eps_estimate",
    "eps_actual",
    "eps_surprise",
    "eps_basis",
    "eps_reported_surprise",
    "eps_reported_basis",
    "eps_surprise_direction_conflict",
    "market_cap",
    "price",
    "price_change_pct",
    "price_change_basis",
    "volume",
    "price_source",
    "data_delayed",
    "nominal_provider_delay_minutes_min",
    "nominal_provider_delay_minutes_max",
)
_FINVIZ_EARNINGS_TIMING_SUFFIXES = {
    "/b": "before_market",
    "/a": "after_market",
}
_FINVIZ_EARNINGS_SESSION_TIMES = {
    (8, 30, 0): "before_market",
    (16, 30, 0): "after_market",
}
_FINVIZ_CALENDAR_TRADER_COMPACT_FIELDS = (
    "calendar_id",
    "symbol",
    "country",
    "country_code",
    "country_attribution",
    "event",
    "scheduled_at",
    "local_time",
    "local_timezone",
    "earnings_date",
    "earnings_timing",
    "event_time_precision",
    "is_earning_date_estimate",
    "ex_dividend_date",
    "exdate",
    "ex_date",
    "pay_date",
    "record_date",
    "actual_value",
    "previous_value",
    "forecast_value",
    "unit",
    "currency",
    "eps_estimate",
    "eps_actual",
    "eps_surprise",
    "eps_basis",
    "eps_reported_surprise",
    "eps_reported_basis",
    "eps_surprise_direction_conflict",
    "sales_estimate",
    "sales_actual",
    "sales_surprise",
    "one_day_price_reaction",
    "dividend",
    "amount",
    "dividend_amount",
    "ordinary_amount",
    "special_amount",
    "yield_pct",
    "impact",
)
_FINVIZ_CALENDAR_STANDARD_FIELDS = (
    "calendar_id",
    "symbol",
    "country",
    "country_code",
    "country_attribution",
    "event",
    "category",
    "scheduled_at",
    "local_time",
    "local_timezone",
    "earnings_date",
    "earnings_timing",
    "event_time_precision",
    "is_earning_date_estimate",
    "ex_dividend_date",
    "exdate",
    "ex_date",
    "pay_date",
    "record_date",
    "reference",
    "reference_date",
    "actual",
    "actual_value",
    "previous",
    "previous_value",
    "forecast",
    "forecast_value",
    "unit",
    "scale",
    "currency",
    "value_parse_status",
    "eps_estimate",
    "eps_actual",
    "eps_surprise",
    "eps_basis",
    "eps_reported_surprise",
    "eps_reported_basis",
    "eps_surprise_direction_conflict",
    "sales_estimate",
    "sales_actual",
    "sales_surprise",
    "one_day_price_reaction",
    "dividend",
    "amount",
    "dividend_amount",
    "ordinary_amount",
    "special_amount",
    "yield_pct",
    "impact",
    "provider_conflicts",
)
_FINVIZ_CALENDAR_COMPACT_FIELDS = _FINVIZ_CALENDAR_STANDARD_FIELDS
_FINVIZ_CALENDAR_IMPORTANCE_LABELS = {
    1: "low",
    2: "medium",
    3: "high",
}


def _finviz_earnings_period_window(
    period_key: str, reference_date: Any
) -> tuple[Any, Any]:
    return finviz_earnings_period_window(period_key, reference_date)


def _normalize_finviz_earnings_rows(
    rows: Any,
    *,
    period_key: str = "this-week",
    reference_date: Any = None,
) -> List[Any]:
    normalized = _normalize_finviz_output_rows(rows)
    if not isinstance(normalized, list):
        return []
    reference = reference_date or datetime.now(timezone.utc).astimezone(
        ZoneInfo("America/New_York")
    ).date()
    period_window = _finviz_earnings_period_window(period_key, reference)
    for index, row in enumerate(normalized):
        if not isinstance(row, dict):
            continue
        row = _canonicalize_finviz_market_row(row)
        normalized[index] = row
        if row.get("dividend") not in (None, ""):
            dividend_yield = _finviz_percent_value(
                row.get("dividend"),
                fraction_input=True,
            )
            if dividend_yield is not None:
                row["dividend_yield"] = dividend_yield
            row.pop("dividend", None)
        if row.get("change_pct") not in (None, ""):
            row["price_change_pct"] = row.pop("change_pct")
            row["price_change_basis"] = "daily_market_move"
        if row.get("price") not in (None, ""):
            row.update(_mark_finviz_delayed_price(row))
        earnings_text = str(row.get("earnings") or "").strip().lower()
        for suffix, timing in _FINVIZ_EARNINGS_TIMING_SUFFIXES.items():
            if earnings_text.endswith(suffix):
                row["earnings_timing"] = timing
                break
        earnings_date = _finviz_earnings_date_from_token(
            row.get("earnings"),
            reference_date=reference,
            period_window=period_window,
        )
        if earnings_date:
            row["earnings_date"] = earnings_date
            date_part = str(row.get("earnings") or "").split("/", 1)[0].strip()
            if len(date_part) < 10 or date_part[4:5] != "-":
                row["earnings_date_year_inferred"] = True
        if "market_cap" not in row:
            continue
        market_cap_source = row.get("market_cap")
        market_cap = _normalize_finviz_fundamental_value(
            "market_cap",
            market_cap_source,
        )
        market_cap_formatted = _format_finviz_large_number(market_cap_source)
        if market_cap is not None:
            row["market_cap"] = market_cap
        if market_cap_formatted:
            row["market_cap_formatted"] = market_cap_formatted
    return normalized


def _finviz_earnings_date_from_token(
    value: Any,
    *,
    reference_date: Any = None,
    period_window: Optional[tuple[Any, Any]] = None,
) -> Optional[str]:
    parsed = parse_finviz_earnings_date(
        value,
        reference_date=reference_date,
        period_window=period_window,
    )
    return parsed.isoformat() if parsed is not None else None


def _calendar_input_has_clock_time(value: str) -> bool:
    text = str(value or "").strip()
    if re.search(r"T\d{1,2}:\d{2}", text):
        return True
    return bool(
        re.search(
            r"(?:^|\s)\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:\s*(?:Z|UTC|[+-]\d{2}(?::?\d{2})?))?",
            text,
            flags=re.IGNORECASE,
        )
    )


def _parse_calendar_bound(
    value: Optional[str],
    *,
    inclusive_end: bool,
) -> Optional[Dict[str, Any]]:
    text = str(value or "").strip()
    if not text:
        return None
    ny = ZoneInfo("America/New_York")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            date_ny = datetime.strptime(text, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(
                f"Invalid calendar date {text!r}. Use a real YYYY-MM-DD date."
            ) from exc
        return {
            "date_ny": date_ny.isoformat(),
            "instant_utc": None,
            "has_time": False,
            "original": text,
        }
    relative_day = text.lower()
    if relative_day in {"today", "yesterday", "tomorrow"}:
        day_offset = {"yesterday": -1, "today": 0, "tomorrow": 1}[relative_day]
        current_ny = datetime.now(timezone.utc).astimezone(ny)
        return {
            "date_ny": (current_ny.date() + timedelta(days=day_offset)).isoformat(),
            "instant_utc": None,
            "has_time": False,
            "original": text,
        }
    parsed = (
        _parse_end_datetime(text) if inclusive_end else _parse_start_datetime(text)
    )
    if parsed is None:
        raise ValueError(
            f"Invalid calendar date {text!r}. Use YYYY-MM-DD, an ISO "
            "datetime, or a relative expression such as '2 days ago'."
        )
    aware_utc = parsed.replace(tzinfo=timezone.utc)
    has_time = _calendar_input_has_clock_time(text)
    return {
        "date_ny": aware_utc.astimezone(ny).date().isoformat(),
        "instant_utc": aware_utc if has_time else None,
        "has_time": has_time,
        "original": text,
    }


def _stamp_calendar_requested_bounds(
    payload: Dict[str, Any],
    *,
    start_bound: Optional[Dict[str, Any]],
    end_bound: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("success") is False:
        return payload
    out = dict(payload)
    date_only_bounds = False
    if start_bound is not None:
        instant = start_bound.get("instant_utc")
        if instant is not None:
            out["start"] = format_datetime_utc(instant)
            out["start_precision"] = "timestamp"
        else:
            out["start"] = start_bound["date_ny"]
            out["start_precision"] = "date"
            date_only_bounds = True
    if end_bound is not None:
        instant = end_bound.get("instant_utc")
        if instant is not None:
            out["end"] = format_datetime_utc(instant)
            out["end_precision"] = "timestamp"
        else:
            out["end"] = end_bound["date_ny"]
            out["end_precision"] = "date"
            date_only_bounds = True
    if date_only_bounds:
        out["timezone"] = _FINVIZ_CALENDAR_LOCAL_TIMEZONE
    return out


def _parse_finviz_calendar_time(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_FINVIZ_CALENDAR_LOCAL_TZ)
    return parsed


def _normalize_finviz_economic_calendar_time(item: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(item)
    raw_value = normalized.get("date")
    parsed = _parse_finviz_calendar_time(raw_value)
    if parsed is None:
        return normalized
    if finviz_timestamp_is_date_only(raw_value, parsed):
        event_date = parsed.astimezone(timezone.utc).date().isoformat()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(raw_value or "").strip()):
            event_date = str(raw_value).strip()
        normalized["event_time_precision"] = "date_only"
        normalized["scheduled_at"] = event_date
        normalized["date"] = event_date
        normalized.pop("local_time", None)
        normalized["local_timezone"] = _FINVIZ_CALENDAR_LOCAL_TIMEZONE
        return normalized
    local_dt = parsed.astimezone(_FINVIZ_CALENDAR_LOCAL_TZ)
    normalized["local_time"] = local_dt.replace(microsecond=0).isoformat()
    normalized["local_timezone"] = _FINVIZ_CALENDAR_LOCAL_TIMEZONE
    utc_time = parsed.astimezone(timezone.utc)
    scheduled_at = format_datetime_utc(utc_time)
    normalized["date"] = scheduled_at
    normalized["scheduled_at"] = scheduled_at
    return normalized


def _normalize_finviz_reference_date(item: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve an MM/DD provider label as a calendar date, not a UTC instant."""
    normalized = dict(item)
    match = re.fullmatch(
        r"\s*(\d{1,2})/(\d{1,2})\s*",
        str(normalized.get("reference") or ""),
    )
    if match is None:
        return normalized
    try:
        month, day = (int(match.group(1)), int(match.group(2)))
        event_time = _parse_finviz_calendar_time(normalized.get("date"))
        if event_time is None:
            return normalized
        event_date = event_time.astimezone(_FINVIZ_CALENDAR_LOCAL_TZ).date()
        candidates = [
            datetime(year, month, day).date()
            for year in (event_date.year - 1, event_date.year, event_date.year + 1)
        ]
    except ValueError:
        return normalized
    reference_date = min(
        candidates,
        key=lambda candidate: abs((candidate - event_date).days),
    )
    normalized["reference_date"] = reference_date.isoformat()
    return normalized


def _normalize_finviz_earnings_calendar_time(item: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(item)
    raw_value = normalized.get("earnings_date")
    raw_text = str(raw_value or "").strip()
    if len(raw_text) <= 10:
        if raw_text:
            normalized["event_time_precision"] = "date_only"
        return normalized
    parsed = _parse_finviz_calendar_time(raw_value)
    if parsed is None:
        return normalized
    local_dt = parsed.astimezone(_FINVIZ_CALENDAR_LOCAL_TZ)
    session = _FINVIZ_EARNINGS_SESSION_TIMES.get(
        (local_dt.hour, local_dt.minute, local_dt.second)
    )
    if session is not None:
        # Finviz uses these two values as before-/after-market buckets. Publishing
        # them as second-precision release instants would invent timing precision.
        local_date = local_dt.date().isoformat()
        normalized["earnings_date"] = local_date
        normalized["date"] = local_date
        normalized["earnings_timing"] = session
        normalized["event_time_precision"] = "session_bucket"
        normalized["local_timezone"] = _FINVIZ_CALENDAR_LOCAL_TIMEZONE
        normalized.pop("local_time", None)
        return normalized
    utc_text = format_datetime_utc(parsed)
    normalized["earnings_date"] = utc_text
    normalized["date"] = utc_text
    normalized["local_time"] = local_dt.replace(microsecond=0).isoformat()
    normalized["local_timezone"] = _FINVIZ_CALENDAR_LOCAL_TIMEZONE
    normalized["event_time_precision"] = "exact"
    return normalized


def _finviz_calendar_importance_label(value: Any) -> Optional[str]:
    try:
        importance = int(value)
    except (TypeError, ValueError):
        return None
    return _FINVIZ_CALENDAR_IMPORTANCE_LABELS.get(importance)


def _add_finviz_calendar_impact_label(item: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(item)
    if normalized.get("impact") in (None, ""):
        impact = _finviz_calendar_importance_label(normalized.get("importance"))
        if impact is not None:
            normalized["impact"] = impact
    return normalized


_ECONOMIC_RELEASE_SUFFIX_UNITS = {
    suffix: ("count", multiplier)
    for suffix, multiplier in _FINVIZ_NUMERIC_SUFFIX_MULTIPLIERS.items()
}
_ECONOMIC_RELEASE_CURRENCY_MARKERS = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
}


def parse_economic_release_value(value: Any) -> Dict[str, Any]:
    """Parse a provider economic print into a numeric value plus unit/scale."""
    missing = {
        "value": None,
        "unit": None,
        "scale": None,
        "currency": None,
        "parse_status": "missing",
    }
    if value in (None, ""):
        return dict(missing)
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "N/A", "n/a", "None", "none", "null"}:
        return dict(missing)
    currency_code = None
    remainder = text
    sign = ""
    if remainder[:1] in "+-":
        sign = remainder[0]
        remainder = remainder[1:].lstrip()
    marker = remainder[:1] if remainder else ""
    if marker in _ECONOMIC_RELEASE_CURRENCY_MARKERS:
        currency_code = _ECONOMIC_RELEASE_CURRENCY_MARKERS[marker]
        remainder = remainder[1:].lstrip()
    if sign:
        remainder = f"{sign}{remainder}"
    unparseable = {
        "value": None,
        "unit": None,
        "scale": None,
        "currency": currency_code,
        "parse_status": "unparseable",
    }
    if remainder.endswith("%"):
        parsed = _parse_finviz_numeric_value(remainder)
        if parsed is None:
            return unparseable
        return {
            "value": parsed,
            "unit": "percent",
            "scale": 1.0,
            "currency": None,
            "parse_status": "ok",
        }
    parsed = _parse_finviz_numeric_value(remainder)
    if parsed is None:
        return unparseable
    suffix = remainder[-1].upper() if remainder else ""
    unit: Optional[str]
    scale: Optional[float]
    if suffix in _ECONOMIC_RELEASE_SUFFIX_UNITS:
        unit, scale = _ECONOMIC_RELEASE_SUFFIX_UNITS[suffix]
    else:
        unit, scale = None, 1.0
    if currency_code:
        unit = "currency"
    return {
        "value": parsed,
        "unit": unit,
        "scale": scale,
        "currency": currency_code,
        "parse_status": "ok",
    }


_CURRENCY_EVENT_HINTS = (
    "trade balance",
    "budget",
    "deficit",
    "surplus",
    "current account",
    "capital account",
    "account balance",
)


def _event_implies_currency_unit(item: Dict[str, Any]) -> bool:
    haystack = " ".join(
        str(item.get(field) or "")
        for field in ("event", "category")
    ).casefold()
    return any(hint in haystack for hint in _CURRENCY_EVENT_HINTS)


def _attach_economic_release_values(item: Dict[str, Any]) -> Dict[str, Any]:
    """Keep raw actual/previous/forecast strings and add parsed numerics."""
    normalized = dict(item)
    if all(
        normalized.get(field) in (None, "")
        for field in ("actual", "previous", "forecast")
    ):
        if _event_implies_currency_unit(normalized):
            normalized["unit"] = "currency"
        return normalized
    parsed_fields = {
        "actual": parse_economic_release_value(normalized.get("actual")),
        "previous": parse_economic_release_value(normalized.get("previous")),
        "forecast": parse_economic_release_value(normalized.get("forecast")),
    }
    statuses: List[str] = []
    units: List[str] = []
    scales: List[float] = []
    currencies: List[str] = []
    for field, parsed in parsed_fields.items():
        normalized[f"{field}_value"] = parsed["value"]
        status = str(parsed.get("parse_status") or "missing")
        if status != "missing":
            statuses.append(status)
        if status != "ok":
            continue
        unit = parsed.get("unit")
        scale = parsed.get("scale")
        currency = parsed.get("currency")
        if unit not in (None, ""):
            units.append(str(unit))
        if scale not in (None, ""):
            scales.append(float(scale))
        if currency not in (None, ""):
            currencies.append(str(currency))
    unique_units = list(dict.fromkeys(units))
    unique_scales = list(dict.fromkeys(scales))
    unique_currencies = list(dict.fromkeys(currencies))
    if len(unique_units) == 1:
        normalized["unit"] = unique_units[0]
    elif not unique_units and _event_implies_currency_unit(normalized):
        normalized["unit"] = "currency"
    if len(unique_scales) == 1:
        normalized["scale"] = unique_scales[0]
    if len(unique_currencies) == 1:
        normalized["currency"] = unique_currencies[0]
    if not statuses:
        normalized["value_parse_status"] = "missing"
    elif all(status == "ok" for status in statuses):
        normalized["value_parse_status"] = "ok"
    elif all(status == "unparseable" for status in statuses):
        normalized["value_parse_status"] = "unparseable"
    else:
        normalized["value_parse_status"] = "partial"
    return normalized


def _compact_finviz_earnings_items(items: Any) -> List[Any]:
    if not isinstance(items, list):
        return []
    compact_rows: List[Any] = []
    for item in items:
        if not isinstance(item, dict):
            compact_rows.append(item)
            continue
        compact_fields = _FINVIZ_EARNINGS_COMPACT_FIELDS
        if (
            item.get("eps_surprise_direction_conflict") is True
            and _finviz_signed_direction(item.get("eps_reported_surprise")) == 0
            and _finviz_signed_direction(item.get("eps_surprise")) not in (None, 0)
        ):
            compact_fields = tuple(
                field
                for field in _FINVIZ_EARNINGS_COMPACT_FIELDS
                if field != "eps_reported_surprise"
            )
        row = {
            field: item[field]
            for field in compact_fields
            if field in item
        }
        if "market_cap" in row:
            market_cap_formatted = _format_finviz_large_number(row.get("market_cap"))
            if market_cap_formatted:
                row["market_cap"] = market_cap_formatted
        compact_rows.append(row)
    return compact_rows


_FINVIZ_CALENDAR_COUNTRY_PREFIXES = (
    ("HOLIDAYSUNITED", "United States", "US"),
    ("UNITEDSTA", "United States", "US"),
    ("USA", "United States", "US"),
    ("USD", "United States", "US"),
    ("US", "United States", "US"),
    ("CANADA", "Canada", "CA"),
    ("CAD", "Canada", "CA"),
    ("GERMANY", "Germany", "DE"),
    ("DEU", "Germany", "DE"),
    ("EUROZONE", "Eurozone", "EU"),
    ("EUR", "Eurozone", "EU"),
    ("JAPAN", "Japan", "JP"),
    ("JPY", "Japan", "JP"),
    ("UNITEDKINGDOM", "United Kingdom", "GB"),
    ("UK", "United Kingdom", "GB"),
    ("GBP", "United Kingdom", "GB"),
    ("AUSTRALIA", "Australia", "AU"),
    ("AUD", "Australia", "AU"),
    ("NEWZEALAND", "New Zealand", "NZ"),
    ("NZD", "New Zealand", "NZ"),
    ("SWITZERLAND", "Switzerland", "CH"),
    ("CHF", "Switzerland", "CH"),
    ("CHINA", "China", "CN"),
    ("CNY", "China", "CN"),
)
_FINVIZ_CALENDAR_EVENT_COUNTRY_KEYWORDS = (
    ("FEDERAL RESERVE", "United States", "US"),
    ("FOMC", "United States", "US"),
    ("FED ", "United States", "US"),
    ("INITIAL JOBLESS CLAIMS", "United States", "US"),
    ("UNEMPLOYMENT RATE", "United States", "US"),
    ("NON FARM", "United States", "US"),
    ("NONFARM", "United States", "US"),
    ("ISM MANUFACTURING", "United States", "US"),
    ("ISM SERVICES", "United States", "US"),
    ("JOLTS", "United States", "US"),
    ("PCE PRICE", "United States", "US"),
    ("CORE PCE", "United States", "US"),
    ("API CRUDE OIL", "United States", "US"),
    ("BAKER HUGHES", "United States", "US"),
    ("WEEK BILL AUCTION", "United States", "US"),
    ("YEAR BOND AUCTION", "United States", "US"),
)
_FINVIZ_CALENDAR_SOURCE_ID_COUNTRIES = {
    # Finviz uses indicator identifiers rather than ISO/currency prefixes for
    # several US releases. Keep these exact mappings auditable and conservative.
    "USURTOT": ("United States", "US"),
    "CPIYOY": ("United States", "US"),
    "RSTAMOM": ("United States", "US"),
    "CONCCONF": ("United States", "US"),
    "FDTR": ("United States", "US"),
    "GDPCQOQ": ("United States", "US"),
    "NFPTCH": ("United States", "US"),
    "NFP": ("United States", "US"),
    "NAPMPMI": ("United States", "US"),
    "NAPMNMI": ("United States", "US"),
    "NAPM": ("United States", "US"),
    "JOLTSJOL": ("United States", "US"),
    "JOLTS": ("United States", "US"),
    "PCEPI": ("United States", "US"),
    "PCEPILFE": ("United States", "US"),
    "PCE": ("United States", "US"),
    "TBIMTOT": ("United States", "US"),
    "TBEXTOT": ("United States", "US"),
}
_FINVIZ_CALENDAR_CURRENCY_TO_COUNTRY_CODE = {
    "USD": "US",
    "EUR": "EU",
    "GBP": "GB",
    "JPY": "JP",
    "CAD": "CA",
    "AUD": "AU",
    "NZD": "NZ",
    "CHF": "CH",
    "CNY": "CN",
}


def _resolve_finviz_calendar_country_filter(
    *,
    country: Optional[str],
    currency: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    country_text = str(country or "").strip()
    currency_text = str(currency or "").strip().upper()
    resolved_code: Optional[str] = None
    if currency_text:
        resolved_code = _FINVIZ_CALENDAR_CURRENCY_TO_COUNTRY_CODE.get(currency_text)
        if resolved_code is None:
            return None, f"Unsupported currency filter '{currency}'."
    if country_text:
        compact_country = re.sub(r"[^A-Za-z]", "", country_text).upper()
        country_code = None
        for prefix, country_name, code in _FINVIZ_CALENDAR_COUNTRY_PREFIXES:
            compact_name = re.sub(r"[^A-Za-z]", "", country_name).upper()
            if compact_country in {prefix, compact_name, code}:
                country_code = code
                break
        if country_code is None:
            return None, f"Unsupported country filter '{country}'."
        if resolved_code is not None and country_code != resolved_code:
            return None, "country and currency filters refer to different regions."
        resolved_code = country_code
    return resolved_code, None


def _infer_finviz_calendar_country(item: Dict[str, Any]) -> tuple[Any, Any]:
    existing_country = item.get("country")
    existing_code = item.get("country_code")
    if existing_country not in (None, "") or existing_code not in (None, ""):
        return existing_country, existing_code

    source_id = str(item.get("symbol") or item.get("source_id") or "").strip()
    compact_source = re.sub(r"[^A-Za-z]", "", source_id).upper()
    exact_country = _FINVIZ_CALENDAR_SOURCE_ID_COUNTRIES.get(compact_source)
    if exact_country is not None:
        return exact_country
    for prefix, country, code in _FINVIZ_CALENDAR_COUNTRY_PREFIXES:
        if compact_source.startswith(prefix):
            return country, code
    event_text = " ".join(
        str(item.get(field) or "").upper()
        for field in ("event", "title", "category")
    )
    for keyword, country, code in _FINVIZ_CALENDAR_EVENT_COUNTRY_KEYWORDS:
        if keyword in event_text:
            return country, code
    return None, None


def _compact_finviz_calendar_item(
    item: Any,
    *,
    source_id_only: bool = True,
    mode: str = "compact",
) -> Any:
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    source_id = normalized.get("source_id") or normalized.get("symbol")
    if source_id_only and source_id not in (None, ""):
        normalized["source_id"] = source_id
        normalized.pop("symbol", None)
    country, country_code = _infer_finviz_calendar_country(normalized)
    if country not in (None, ""):
        normalized["country"] = country
    if country_code not in (None, ""):
        normalized["country_code"] = country_code
    if normalized.get("scheduled_at") in (None, "") and normalized.get("date") not in (
        None,
        "",
    ):
        normalized["scheduled_at"] = normalized["date"]
    trader_compact = str(mode or "compact") in {"compact", "summary"}
    fields = (
        _FINVIZ_CALENDAR_TRADER_COMPACT_FIELDS
        if trader_compact
        else _FINVIZ_CALENDAR_STANDARD_FIELDS
    )
    if (
        normalized.get("category") not in (None, "")
        and normalized.get("category") == normalized.get("event")
    ):
        normalized = dict(normalized)
        normalized.pop("category", None)
    omit_reported_zero = (
        normalized.get("eps_surprise_direction_conflict") is True
        and _finviz_signed_direction(normalized.get("eps_reported_surprise")) == 0
        and _finviz_signed_direction(normalized.get("eps_surprise")) not in (None, 0)
    )
    out = {
        field: normalized[field]
        for field in fields
        if field in normalized
        and normalized[field] not in (None, "")
        and not (omit_reported_zero and field == "eps_reported_surprise")
    }
    if (
        trader_compact
        and normalized.get("category") not in (None, "")
        and normalized.get("category") != normalized.get("event")
    ):
        out["category"] = normalized["category"]
    if trader_compact:
        parse_status = str(normalized.get("value_parse_status") or "").strip()
        if parse_status in {"partial", "unparseable"}:
            out["value_parse_status"] = parse_status
            out["parse_warning"] = {
                "code": "calendar_value_parse_lossy",
                "status": parse_status,
            }
            for field in ("actual", "previous", "forecast"):
                raw = normalized.get(field)
                if raw not in (None, ""):
                    out[f"raw_{field}"] = raw
    return out


def _normalize_finviz_dividend_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    for source, target in (
        ("amount", "dividend_amount"),
        ("ordinary", "ordinary_amount"),
        ("special", "special_amount"),
        ("yield", "yield_pct"),
    ):
        if source in normalized and target not in normalized:
            normalized[target] = normalized.pop(source)
    return normalized


def _normalize_finviz_earnings_amounts(item: Any) -> Any:
    """Convert provider million-scaled earnings amounts to base currency units."""
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    for field in ("market_cap", "sales_estimate", "sales_actual"):
        value = normalized.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if numeric == numeric and numeric not in (float("inf"), float("-inf")):
            normalized[field] = numeric * 1_000_000.0
    return normalized


def _enrich_finviz_calendar_country(item: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(item)
    had_country = normalized.get("country") not in (None, "")
    had_country_code = normalized.get("country_code") not in (None, "")
    country, country_code = _infer_finviz_calendar_country(normalized)
    if country not in (None, ""):
        normalized["country"] = country
    if country_code not in (None, ""):
        normalized["country_code"] = country_code
    if (country not in (None, "") or country_code not in (None, "")) and not (
        had_country or had_country_code
    ):
        normalized["country_inferred"] = True
        normalized["country_attribution"] = "inferred"
    elif had_country or had_country_code:
        normalized["country_attribution"] = "provider"
    else:
        normalized["country_attribution"] = "unknown"
    return normalized


def _normalize_finviz_earnings_percentages(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    for field in (
        "eps_surprise",
        "eps_reported_surprise",
        "sales_surprise",
        "one_day_price_reaction",
    ):
        if field not in normalized:
            continue
        value = _finviz_percent_value(normalized.get(field), fraction_input=False)
        if value is not None:
            normalized[field] = value
    return normalized


def _finviz_signed_direction(value: Any) -> Optional[int]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric > 0:
        return 1
    if numeric < 0:
        return -1
    return 0


def _normalize_finviz_earnings_items(
    items: List[Any],
    payload: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    normalized_items = [
        _normalize_finviz_earnings_calendar_time(item)
        if isinstance(item, dict)
        else item
        for item in items
    ]
    normalized_items = [
        _normalize_finviz_earnings_amounts(item)
        for item in normalized_items
    ]
    normalized_items = [
        _normalize_finviz_earnings_percentages(item)
        for item in normalized_items
    ]
    normalized_items = [
        _annotate_finviz_earnings_eps_bases(item)
        for item in normalized_items
    ]
    conflict_count = sum(
        1
        for item in normalized_items
        if isinstance(item, dict)
        and item.get("eps_surprise_direction_conflict") is True
    )
    if payload is not None and conflict_count:
        _append_finviz_warning(
            payload,
            (
                f"{conflict_count} earnings row(s) have conflicting EPS "
                "surprise directions across the two provider families."
            ),
        )
    return normalized_items


def _annotate_finviz_earnings_eps_bases(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    primary_present = any(
        normalized.get(field) not in (None, "")
        for field in ("eps_estimate", "eps_actual", "eps_surprise")
    )
    reported_present = normalized.get("eps_reported_surprise") not in (None, "")
    if primary_present:
        normalized["eps_basis"] = "provider_unspecified"
    if reported_present:
        normalized["eps_reported_basis"] = "provider_unspecified"
    primary_direction = _finviz_signed_direction(normalized.get("eps_surprise"))
    if primary_direction is None:
        estimate = normalized.get("eps_estimate")
        actual = normalized.get("eps_actual")
        try:
            primary_direction = _finviz_signed_direction(
                float(actual) - float(estimate)
            )
        except (TypeError, ValueError):
            primary_direction = None
    reported_direction = _finviz_signed_direction(
        normalized.get("eps_reported_surprise")
    )
    if (
        primary_direction is not None
        and reported_direction is not None
        and primary_direction != reported_direction
    ):
        normalized["eps_surprise_direction_conflict"] = True
    return normalized


def _finviz_calendar_excluded_event(item: Dict[str, Any]) -> Dict[str, Any]:
    source_id = item.get("source_id") or item.get("symbol")
    return {
        key: value
        for key, value in {
            "event": item.get("event") or item.get("title"),
            "date": item.get("date") or item.get("earnings_date"),
            "source_id": source_id,
            "reason": "unknown_country_attribution",
        }.items()
        if value not in (None, "")
    }


def _finviz_calendar_item_is_upcoming(
    item: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
) -> bool:
    if item.get("actual") not in (None, ""):
        return False
    raw_date = item.get("scheduled_at") or item.get("date") or item.get("earnings_date")
    if raw_date in (None, ""):
        return False
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    if str(item.get("event_time_precision") or "").strip().lower() == "date_only":
        raw_text = str(raw_date).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_text):
            try:
                event_date = datetime.strptime(raw_text, "%Y-%m-%d").date()
            except ValueError:
                return False
            today_ny = reference.astimezone(_FINVIZ_CALENDAR_LOCAL_TZ).date()
            return event_date >= today_ny
    try:
        event_time = datetime.fromisoformat(str(raw_date))
    except (TypeError, ValueError):
        return False
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)
    return event_time.astimezone(timezone.utc) >= reference.astimezone(timezone.utc)


def _apply_finviz_calendar_empty_hint(
    out: Dict[str, Any],
    *,
    calendar_type: str,
    upcoming_only: bool,
    filtered_released_count: int,
) -> None:
    cal_type = str(calendar_type or "economic").strip().lower()
    if cal_type == "earnings":
        out["message"] = "No detailed earnings calendar rows matched the date range."
        out["hint"] = (
            "Use calendar --kind earnings --view period for the compact "
            "earnings window with price/volume context."
        )
        return
    if cal_type == "dividends":
        if out.get("range_complete") is False:
            out["message"] = (
                "No dividend rows matched the supported current-forward portion; "
                "the earlier requested portion was not represented by the provider."
            )
        else:
            out["message"] = "No dividend calendar rows matched the date range."
        return
    out["message"] = "No economic calendar events matched the filters."
    if upcoming_only:
        out["hint"] = (
            "upcoming_only filtered "
            f"{filtered_released_count} released event(s); "
            "pass --upcoming false to include prints that already have an actual."
        )
        out["filtered_released_count"] = int(filtered_released_count)
        return
    if str(out.get("country_filter") or "").strip().upper() not in {"", "US"}:
        out["hint"] = (
            "Finviz economic calendar is US-centric; an empty result for a "
            "non-US country/currency filter does not mean that region has no "
            "releases. Relax the country/currency filter or use a US/USD filter."
        )
        return
    out["hint"] = "Relax impact, country, currency, start, or end filters."


def _calendar_item_instant_utc(item: Any) -> Optional[datetime]:
    if not isinstance(item, dict):
        return None
    for field in ("scheduled_at", "date", "earnings_date", "ex_dividend_date"):
        parsed = _parse_finviz_calendar_time(item.get(field))
        if parsed is not None:
            return parsed.astimezone(timezone.utc)
    return None


def _finviz_calendar_row_items(
    items: List[Any],
    *,
    calendar_type: str,
    shape_detail: str,
) -> List[Any]:
    if shape_detail == "full":
        return items
    economic = str(calendar_type or "economic").strip().lower() == "economic"
    return [
        _compact_finviz_calendar_item(
            item,
            source_id_only=economic,
            mode=shape_detail,
        )
        for item in items
    ]


def _economic_calendar_numeric_unit_label(items: Any) -> str:
    row_units: list[str] = []
    has_untyped_numeric_row = False
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            unit = str(item.get("unit") or "").strip().lower()
            if unit:
                row_units.append(unit)
            elif any(
                isinstance(item.get(key), (int, float))
                for key in ("actual_value", "previous_value", "forecast_value")
            ):
                has_untyped_numeric_row = True
    unique_units = list(dict.fromkeys(row_units))
    if has_untyped_numeric_row:
        return "parsed_numeric (per-row unit; some unknown)"
    if len(unique_units) == 1:
        unit = unique_units[0]
        if unit == "percent":
            return "parsed_numeric (percent: 1.0 = 1%)"
        if unit == "count":
            return "parsed_numeric (count)"
        if unit == "currency":
            return "parsed_numeric (listing currency)"
        return f"parsed_numeric ({unit})"
    return "parsed_numeric (per-row unit)"


def _apply_finviz_calendar_units(
    out: Dict[str, Any],
    *,
    calendar_type: str,
    shape_detail: str,
) -> None:
    kind = str(calendar_type or "economic").strip().lower()
    if kind == "dividends":
        out["currency_status"] = "unavailable"
        out["units"] = {
            "dividend_amount": "unspecified_listing_currency_per_share",
            "ordinary_amount": "unspecified_listing_currency_per_share",
            "special_amount": "unspecified_listing_currency_per_share",
            "yield_pct": "percent (1.0 = 1%)",
        }
        return
    if kind == "economic":
        units = dict(out.get("units") or {})
        numeric_label = _economic_calendar_numeric_unit_label(out.get("items"))
        units.update(
            {
                "actual_value": numeric_label,
                "previous_value": numeric_label,
                "forecast_value": numeric_label,
            }
        )
        if shape_detail not in {"compact", "summary"}:
            units.update(
                {
                    "actual": "provider_text",
                    "previous": "provider_text",
                    "forecast": "provider_text",
                    "scale": "numeric_multiplier_applied_to_provider_suffix",
                }
            )
        out["units"] = units
        return
    if kind == "earnings":
        out["currency_status"] = "unavailable"
        out["amount_source_scale"] = "provider_millions_normalized_to_base_units"
        out["units"] = {
            "market_cap": "unspecified_listing_currency_base_units",
            "sales_estimate": "unspecified_listing_currency_base_units",
            "sales_actual": "unspecified_listing_currency_base_units",
            "eps_estimate": "unspecified_listing_currency_per_share",
            "eps_actual": "unspecified_listing_currency_per_share",
            "eps_surprise": "percent (1.0 = 1%)",
            "eps_reported_surprise": "percent (1.0 = 1%)",
            "sales_surprise": "percent (1.0 = 1%)",
            "one_day_price_reaction": "percent (1.0 = 1%)",
        }


def _filter_finviz_calendar_items_by_instant(
    items: List[Any],
    *,
    start_instant: Optional[datetime],
    end_instant: Optional[datetime],
) -> List[Any]:
    if start_instant is None and end_instant is None:
        return items
    kept: List[Any] = []
    for item in items:
        instant = _calendar_item_instant_utc(item)
        if instant is None:
            continue
        if start_instant is not None and instant < start_instant:
            continue
        if end_instant is not None and instant > end_instant:
            continue
        kept.append(item)
    return kept


def _normalize_finviz_calendar_payload(
    result: Dict[str, Any],
    *,
    detail: str = "compact",
    calendar_type: str = "economic",
    country_code_filter: Optional[str] = None,
    upcoming_only: bool = False,
    source_is_unpaged: bool = False,
    limit: int = 20,
    page: int = 1,
    start_instant: Optional[datetime] = None,
    end_instant: Optional[datetime] = None,
) -> Dict[str, Any]:
    if not isinstance(result, dict) or result.get("error"):
        return result
    shape_detail = normalize_output_detail(detail, default="compact")
    out: Dict[str, Any] = {}
    for key, value in result.items():
        normalized_key = _normalize_finviz_output_key(key)
        out[normalized_key] = value
    filtered_total: Optional[int] = None
    source_complete = True
    if isinstance(out.get("items"), list):
        source_item_count = len(out["items"])
        try:
            source_total = int(result.get("total"))
        except (TypeError, ValueError):
            source_total = source_item_count
        source_complete = source_total <= source_item_count
        calendar_mode = str(calendar_type or "economic").strip().lower()
        normalized_items = _normalize_finviz_output_rows(out["items"])
        if calendar_mode == "economic":
            normalized_items = [
                _enrich_finviz_calendar_country(item)
                for item in normalized_items
            ]
        if calendar_mode == "dividends":
            normalized_items = [
                _normalize_finviz_dividend_item(item)
                for item in normalized_items
            ]
        if calendar_mode == "economic":
            normalized_items = [
                _normalize_finviz_economic_calendar_time(item)
                if isinstance(item, dict)
                else item
                for item in normalized_items
            ]
            normalized_items = [
                _add_finviz_calendar_impact_label(item)
                if isinstance(item, dict)
                else item
                for item in normalized_items
            ]
            normalized_items = [
                _normalize_finviz_reference_date(item)
                if isinstance(item, dict)
                else item
                for item in normalized_items
            ]
            normalized_items = [
                _attach_economic_release_values(item)
                if isinstance(item, dict)
                else item
                for item in normalized_items
            ]
        elif calendar_mode == "earnings":
            normalized_items = _normalize_finviz_earnings_items(
                normalized_items,
                out,
            )
        if country_code_filter:
            unclassified_items = [
                item
                for item in normalized_items
                if isinstance(item, dict)
                and not str(item.get("country_code") or "").strip()
            ]
            high_impact_unclassified = [
                item
                for item in unclassified_items
                if str(item.get("impact") or "").strip().lower() == "high"
            ]
            excluded_unclassified = [
                item
                for item in unclassified_items
                if item not in high_impact_unclassified
            ]
            matched_items = [
                item
                for item in normalized_items
                if isinstance(item, dict)
                and str(item.get("country_code") or "").upper()
                == str(country_code_filter).upper()
            ]
            if (
                str(country_code_filter).strip().upper() != "US"
                and not matched_items
            ):
                _append_finviz_warning(
                    out,
                    "Finviz economic calendar currently covers US releases. "
                    f"No events were attributed to country_filter="
                    f"{str(country_code_filter).upper()}; an empty match does "
                    "not mean that region has no scheduled releases.",
                )
            retained_high_impact = []
            for item in high_impact_unclassified:
                retained = dict(item)
                retained.setdefault("country_attribution", "unknown")
                retained_high_impact.append(retained)
            normalized_items = matched_items + retained_high_impact
            if excluded_unclassified:
                out["unclassified_events_count"] = int(len(excluded_unclassified))
                out["excluded_events"] = [
                    _finviz_calendar_excluded_event(item)
                    for item in excluded_unclassified
                ]
                warnings_out = list(out.get("warnings") or [])
                warnings_out.append(
                    f"{len(excluded_unclassified)} event(s) had unknown country "
                    "attribution and were excluded by the country/currency filter; "
                    "see excluded_events for names and times."
                )
                out["warnings"] = warnings_out
            if retained_high_impact:
                out["high_impact_unattributed_count"] = int(len(retained_high_impact))
                warnings_out = list(out.get("warnings") or [])
                warnings_out.append(
                    f"{len(retained_high_impact)} high-impact event(s) had unknown "
                    "country attribution and were retained; see "
                    "high_impact_unattributed_count."
                )
                out["warnings"] = warnings_out
        filtered_released_count = 0
        if upcoming_only and calendar_mode == "economic":
            kept_items = []
            for item in normalized_items:
                if isinstance(item, dict) and _finviz_calendar_item_is_upcoming(item):
                    kept_items.append(item)
                elif isinstance(item, dict):
                    filtered_released_count += 1
            normalized_items = kept_items
            normalized_items.sort(
                key=lambda item: str(item.get("date") or item.get("earnings_date") or "")
            )
        normalized_items = _filter_finviz_calendar_items_by_instant(
            normalized_items,
            start_instant=start_instant,
            end_instant=end_instant,
        )
        filtered_total = len(normalized_items)
        if source_is_unpaged:
            page_value = max(1, int(page or 1))
            limit_value = max(1, int(limit))
            offset_value = (page_value - 1) * limit_value
            normalized_items = normalized_items[
                offset_value : offset_value + limit_value
            ]
        out["items"] = _finviz_calendar_row_items(
            normalized_items,
            calendar_type=calendar_type,
            shape_detail=shape_detail,
        )
        out["count"] = len(out["items"])
        out["row_key"] = "items"
        if country_code_filter:
            out["country_filter"] = str(country_code_filter).upper()
        if out["count"] == 0:
            _apply_finviz_calendar_empty_hint(
                out,
                calendar_type=calendar_type,
                upcoming_only=upcoming_only,
                filtered_released_count=filtered_released_count,
            )
    if country_code_filter:
        out["country_filter"] = str(country_code_filter).upper()
    if upcoming_only:
        out["upcoming_only"] = True
        out["sort"] = "scheduled_time_ascending"
    if str(calendar_type or "economic").strip().lower() in {"economic", "earnings"}:
        out["timezone"] = "UTC"
    else:
        out.setdefault("timezone", _FINVIZ_CALENDAR_LOCAL_TIMEZONE)
    _apply_finviz_calendar_units(
        out,
        calendar_type=calendar_type,
        shape_detail=shape_detail,
    )
    page_value = int(page if source_is_unpaged else result.get("page") or page or 1)
    if source_is_unpaged:
        offset_value = (max(1, page_value) - 1) * max(1, int(limit))
        has_more = bool(
            (filtered_total is not None and offset_value + len(out.get("items") or []) < filtered_total)
            or not source_complete
        )
        pagination_total = filtered_total if source_complete else None
        pagination_lower_bound = filtered_total
    else:
        pages = result.get("pages")
        has_more = bool(
            result.get("has_more")
            or (pages not in (None, "") and page_value < int(pages))
        )
        pagination_total = result.get("total")
        pagination_lower_bound = result.get("total_lower_bound")
    _apply_finviz_pagination_contract(
        out,
        returned=len(out.get("items") or []),
        limit=limit,
        page=page_value,
        total=pagination_total,
        total_lower_bound=pagination_lower_bound,
        has_more=has_more,
    )
    out.pop("omitted_item_count", None)
    out["detail"] = shape_detail
    return out


def run_finviz_calendar(
    calendar: str = "economic",
    impact: Optional[str] = None,
    country: Optional[str] = None,
    currency: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    upcoming: Optional[bool] = None,
    limit: int = 20,
    page: int = 1,
    detail: str = "compact",
) -> Dict[str, Any]:
    """Fetch a Finviz calendar payload without MCP/CLI wrapping."""

    try:
        start_bound = _parse_calendar_bound(start, inclusive_end=False)
        end_bound = _parse_calendar_bound(end, inclusive_end=True)
    except ValueError as exc:
        return build_error_payload(
            str(exc),
            code="invalid_date",
            operation="calendar",
            remediation=(
                "Pass YYYY-MM-DD, an ISO timestamp, or a dateparser "
                "expression such as start='2 days ago', end='today'."
            ),
        )
    start_value = None if start_bound is None else str(start_bound["date_ny"])
    end_value = None if end_bound is None else str(end_bound["date_ny"])
    start_instant = None if start_bound is None else start_bound.get("instant_utc")
    end_instant = None if end_bound is None else end_bound.get("instant_utc")

    cal = (calendar or "economic").strip().lower()
    upcoming_only = (
        bool(upcoming)
        if upcoming is not None
        else cal == "economic" and start_value is None and end_value is None
    )

    country_filter, filter_error = _resolve_finviz_calendar_country_filter(
        country=country,
        currency=currency,
    )
    if filter_error:
        return build_error_payload(
            str(filter_error),
            code="invalid_parameter",
            operation="calendar",
            remediation="Use a supported country or currency code, such as US or USD.",
        )
    if (
        start_instant is not None
        and end_instant is not None
        and end_instant < start_instant
    ):
        return build_error_payload(
            "end must be on or after start",
            code="invalid_date_range",
            operation="calendar",
            details={
                "start": format_datetime_utc(start_instant),
                "end": format_datetime_utc(end_instant),
            },
            remediation="Set end to a timestamp on or after start.",
        )
    if start_value and end_value and end_value < start_value:
        return build_error_payload(
            "end must be on or after start",
            code="invalid_date_range",
            operation="calendar",
            details={"start": start_value, "end": end_value},
            remediation="Set end to a date on or after start.",
        )
    if cal != "economic" and country_filter:
        return build_error_payload(
            "country/currency filters are only supported for economic calendar.",
            code="incompatible_parameters",
            operation="calendar",
            details={"invalid": ["country", "currency"], "kind": cal},
            valid_values={"kind": ["economic"]},
            remediation="Drop country/currency, or set kind=economic.",
        )
    if cal != "economic" and upcoming is not None:
        return build_error_payload(
            "upcoming is only supported for economic calendar.",
            code="incompatible_parameters",
            operation="calendar",
            details={"invalid": ["upcoming"], "kind": cal},
            valid_values={"kind": ["economic"]},
            remediation="Drop upcoming, or set kind=economic.",
        )
    if cal != "economic" and impact is not None:
        return build_error_payload(
            "impact is only supported for economic calendar.",
            code="incompatible_parameters",
            operation="calendar",
            details={"invalid": ["impact"], "kind": cal},
            valid_values={"kind": ["economic"]},
            remediation="Drop impact, or set kind=economic.",
        )

    if cal == "economic":
        payload = _normalize_finviz_calendar_payload(
            get_economic_calendar(
                impact=impact,
                limit=500,
                page=1,
                date_from=start_value,
                date_to=end_value,
            ),
            detail=detail,
            calendar_type=cal,
            country_code_filter=country_filter,
            upcoming_only=upcoming_only,
            source_is_unpaged=True,
            limit=limit,
            page=page,
            start_instant=start_instant,
            end_instant=end_instant,
        )
    elif cal == "earnings":
        payload = _normalize_finviz_calendar_payload(
            get_earnings_calendar_api(
                limit=limit,
                page=page,
                date_from=start_value,
                date_to=end_value,
            ),
            detail=detail,
            calendar_type=cal,
            limit=limit,
            page=page,
            start_instant=start_instant,
            end_instant=end_instant,
        )
    elif cal == "dividends":
        payload = _normalize_finviz_calendar_payload(
            get_dividends_calendar_api(
                limit=limit,
                page=page,
                date_from=start_value,
                date_to=end_value,
            ),
            detail=detail,
            calendar_type=cal,
            limit=limit,
            page=page,
            start_instant=start_instant,
            end_instant=end_instant,
        )
    else:
        return {
            "error": (
                f"Unsupported calendar '{calendar}'. Expected economic, "
                "earnings, or dividends."
            )
        }
    return _stamp_calendar_requested_bounds(
        payload,
        start_bound=start_bound,
        end_bound=end_bound,
    )


def finviz_calendar(
    calendar: Literal["economic", "earnings", "dividends"] = "economic",  # type: ignore
    impact: Optional[str] = None,
    country: Optional[str] = None,
    currency: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    upcoming: Optional[bool] = None,
    limit: Annotated[int, Field(ge=1)] = 20,
    page: Annotated[int, Field(ge=1)] = 1,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Get detailed Finviz calendar data (economic, earnings, or dividends).

    Internal adapter. Public callers use ``calendar``. Use
    ``calendar(kind='earnings', view='period')`` for the compact period view.

    Parameters
    ----------
    calendar : str
        Calendar type: "economic", "earnings", or "dividends".
    impact : str, optional
        Economic only: filter by impact level. Accepts low, medium, high, or a
        comma-separated list such as high,medium.
    country : str, optional
        Economic only: filter by country name or code (for example "US").
    currency : str, optional
        Economic only: filter by affected currency (for example "USD").
    start : str, optional
        Start date in ISO format: YYYY-MM-DD. Defaults to the current
        America/New_York date, including when only end is supplied.
    end : str, optional
        Inclusive end date in ISO format: YYYY-MM-DD. May be supplied without
        start; otherwise defaults from the resolved start date.
    upcoming : bool, optional
        Economic only: keep unreleased future events. Defaults to true for the
        live default window and false when start or end is supplied.
    limit : int
        Max upcoming or date-range events per page, sorted by scheduled time.
    page : int
        Page number for pagination (default 1)
    detail : str
        Use "compact" for trader-facing fields or "full" for raw upstream fields.

    Returns
    -------
    dict
        Calendar entries (schema depends on calendar type).
    """
    fields = {
        "calendar": calendar,
        "impact": impact,
        "country": country,
        "currency": currency,
        "start": start,
        "end": end,
        "upcoming": upcoming,
        "limit": limit,
        "page": page,
        "detail": detail,
    }
    return _run_logged_tool(
        "finviz_calendar",
        fields,
        lambda: run_finviz_calendar(
            calendar=calendar,
            impact=impact,
            country=country,
            currency=currency,
            start=start,
            end=end,
            upcoming=upcoming,
            limit=limit,
            page=page,
            detail=detail,
        ),
    )


def finviz_earnings(
    period: Literal["this-week", "next-week", "previous-week", "this-month"] = "this-week",
    limit: Annotated[int, Field(ge=1)] = 10,
    page: Annotated[int, Field(ge=1)] = 1,
    include_elapsed: bool = False,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Get the period-based earnings calendar from Finviz.
    
    Internal adapter. Public callers use
    ``calendar(kind='earnings', view='period')``.
    
    Parameters
    ----------
    period : str
        Calendar period: this-week, next-week, previous-week, or this-month.
        Current periods return upcoming dates by default. Previous-week is
        always an archive.
    limit : int
        Max items per page (default 10)
    page : int
        Page number for pagination (default 1)
    include_elapsed : bool
        Include events that have elapsed in New York time. Before-market events
        elapse at 09:30 and after-market events at 16:00 (default false).
    detail : {"compact", "full"}
        Response detail level. Compact returns calendar-focused rows; full keeps
        all normalized provider columns and adds the tool metadata block.
    
    Returns
    -------
    dict
        Earnings calendar data
    """
    def _run() -> Dict[str, Any]:
        normalized_period = _normalize_finviz_earnings_period(period)
        if normalized_period is None:
            return {
                "success": False,
                "error": (
                    "Invalid period. Use one of: "
                    + ", ".join(sorted(_FINVIZ_EARNINGS_PERIODS))
                    + "."
                ),
                "error_code": "finviz_earnings_invalid_period",
                "meta": _build_tool_contract_meta(
                    tool="finviz_earnings",
                    request={"period": period, "limit": limit, "page": page, "detail": detail},
                ),
            }
        period_key, period_value = normalized_period
        request = {
            "period": period_key,
            "limit": limit,
            "page": page,
            "include_elapsed": include_elapsed,
            "detail": detail,
        }
        detail_error = _validate_finviz_detail(detail, operation="finviz_earnings")
        if detail_error is not None:
            return detail_error
        result = get_earnings_calendar(
            period=period_value,
            limit=limit,
            page=page,
            include_elapsed=include_elapsed,
        )
        if not isinstance(result, dict):
            return {
                "success": False,
                "error": "Unexpected earnings calendar response.",
                "error_code": "finviz_earnings_failed",
                "meta": _build_tool_contract_meta(
                    tool="finviz_earnings",
                    request=request,
                ),
            }
        if result.get("error"):
            error_out = {
                "success": False,
                "error": str(result.get("error")),
                "error_code": str(result.get("error_code") or "").strip()
                or _finviz_earnings_error_code(str(result.get("error"))),
                "meta": _build_tool_contract_meta(
                    tool="finviz_earnings",
                    request=request,
                ),
            }
            for key in (
                "retryable",
                "retry_after_seconds",
                "remediation",
                "provider",
                "endpoint",
            ):
                if result.get(key) not in (None, ""):
                    error_out[key] = result[key]
            return error_out

        items = result.get("earnings")
        if not isinstance(items, list):
            items = []
        reference_date_text = str(result.get("calendar_reference_date") or "")
        try:
            reference_date = datetime.strptime(reference_date_text, "%Y-%m-%d").date()
        except ValueError:
            reference_date = datetime.now(timezone.utc).astimezone(
                ZoneInfo("America/New_York")
            ).date()
        normalized_items = _normalize_finviz_earnings_rows(
            items,
            period_key=period_key,
            reference_date=reference_date,
        )
        detail_mode = normalize_output_verbosity_detail(detail, default="compact")
        output_items = (
            normalized_items
            if detail_mode == "full"
            else _compact_finviz_earnings_items(normalized_items)
        )
        stats = {
            "truncated": result.get("truncated"),
        }
        out: Dict[str, Any] = {
            "success": True,
            "period": period_key,
            "detail": detail_mode,
            "items": output_items,
            "row_key": "items",
            "count": len(output_items),
            "total": result.get("total"),
            "page": result.get("page"),
            "pages": result.get("pages"),
            "has_more": bool(result.get("has_more")),
            "truncated": bool(result.get("truncated")),
            "calendar_reference_date": reference_date.isoformat(),
            "calendar_timezone": str(
                result.get("calendar_timezone") or "America/New_York"
            ),
            "calendar_order": (
                "upcoming_date_ascending"
                if result.get("elapsed_filter_applied") is True
                else "period_start_ascending"
            ),
            "include_elapsed": bool(include_elapsed),
            "includes_elapsed_dates": bool(
                period_key == "previous-week"
                or (
                    period_key in {"this-week", "this-month"}
                    and result.get("elapsed_filter_applied") is not True
                )
            ),
        }
        if result.get("calendar_reference_at") not in (None, ""):
            out["calendar_reference_at"] = result["calendar_reference_at"]
        if result.get("elapsed_filter_applied") is True:
            out["elapsed_cutoff_date"] = reference_date.isoformat()
            if result.get("calendar_reference_at") not in (None, ""):
                out["elapsed_cutoff_at"] = result["calendar_reference_at"]
        units = _finviz_screen_units_for_rows(output_items)
        if units:
            out["units"] = units
        if any(
            isinstance(item, dict) and item.get("data_delayed") is True
            for item in output_items
        ):
            _attach_finviz_delayed_root_metadata(out)
        if out["count"] == 0 and not include_elapsed:
            out["message"] = (
                "No unelapsed earnings remain in this period after the "
                "include_elapsed=false filter."
            )
            out["hint"] = (
                "Pass --include-elapsed true to include already-released "
                "prints, or --period next-week for the next window."
            )
        elif out["detail"] != "full":
            start = result.get("period_start") or "<date>"
            end = result.get("period_end") or "<date>"
            out["hint"] = (
                "Period-based earnings view; use "
                "calendar --kind earnings --view range "
                f"--start {start} --end {end} "
                "for date-range EPS/sales actuals and surprises."
            )
        if out["detail"] == "full":
            out["meta"] = _build_tool_contract_meta(
                tool="finviz_earnings",
                request=request,
                stats=stats,
            )
        for key in (
            "source_incomplete",
            "partial",
            "period_filter_applied",
            "period_start",
            "period_end",
            "period_rows_rejected",
            "warnings",
            "related_tools",
        ):
            if result.get(key) not in (None, "", [], {}):
                out[key] = result[key]
        page_value = int(result.get("page") or page or 1)
        pages = result.get("pages")
        has_more = bool(
            result.get("has_more")
            or (pages not in (None, "") and page_value < int(pages))
        )
        effective_limit = int(result.get("limit") or limit)
        if effective_limit != int(limit):
            out["requested_limit"] = int(limit)
            _append_finviz_warning(
                out,
                f"limit was capped from {int(limit)} to {effective_limit} by the Finviz provider adapter.",
            )
        _apply_finviz_pagination_contract(
            out,
            returned=len(output_items),
            limit=effective_limit,
            page=page_value,
            total=result.get("total"),
            total_lower_bound=result.get("total_lower_bound"),
            has_more=has_more,
        )
        return out

    return _run_logged_tool(
        "finviz_earnings",
        {
            "period": period,
            "limit": limit,
            "page": page,
            "include_elapsed": include_elapsed,
            "detail": detail,
        },
        _run,
    )
