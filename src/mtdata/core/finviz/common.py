"""Shared Finviz adapter helpers: identity, pagination, logging, delayed metadata."""

import logging
import re
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)
from zoneinfo import ZoneInfo

from mtdata.core.error_envelope import build_error_payload
from mtdata.core.execution_logging import run_logged_operation
from mtdata.core.output_contract import (
    build_pagination_meta,
    normalize_output_verbosity_detail,
)
from mtdata.services.finviz.symbols import (
    looks_like_non_equity_symbol,
    normalize_finviz_equity_symbol,
)
from mtdata.services.finviz.utils import finviz_percent_value
from mtdata.shared.symbols import finviz_forex_symbol_to_mt5
from mtdata.utils.time import format_datetime_utc

logger = logging.getLogger("mtdata.core.finviz")


def _finviz_error_payload(
    message: Any,
    *,
    code: str,
    operation: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return build_error_payload(
        message,
        code=code,
        operation=operation,
        details=details,
    )


def _validate_positive_finviz_limit(
    limit: Any,
    *,
    operation: str,
) -> Optional[Dict[str, Any]]:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = 0
    if value >= 1:
        return None
    return _finviz_error_payload(
        "limit must be greater than or equal to 1.",
        code=f"{operation}_invalid_limit",
        operation=operation,
        details={"limit": limit, "minimum": 1},
    )


def _normalize_equity_symbol(symbol: str, *, tool_name: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    symbol_norm = normalize_finviz_equity_symbol(symbol)
    if not symbol_norm:
        return None, _finviz_error_payload(
            f"{tool_name} requires a symbol.",
            code="finviz_symbol_required",
            operation=tool_name,
            details={"tool": tool_name},
        )
    if looks_like_non_equity_symbol(symbol_norm):
        return None, _finviz_error_payload(
            (
                f"{symbol_norm} is not a Finviz-supported equity ticker. "
                f"{tool_name} only supports US equities."
            ),
            code="finviz_unsupported_symbol",
            operation=tool_name,
            details={"symbol": symbol_norm, "tool": tool_name},
        )
    return symbol_norm, None


def _require_equity_symbol(
    symbol: str,
    *,
    tool_name: str,
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    symbol_norm, error = _normalize_equity_symbol(symbol, tool_name=tool_name)
    if error is not None:
        return None, error
    if symbol_norm is None:
        return None, _finviz_error_payload(
            f"{tool_name} could not normalize symbol.",
            code="finviz_symbol_invalid",
            operation=tool_name,
            details={"tool": tool_name},
        )
    return symbol_norm, None


def _attach_finviz_symbol_identity(
    payload: Dict[str, Any],
    *,
    requested_symbol: Any,
    finviz_ticker: str,
) -> Dict[str, Any]:
    """Expose the provider rewrite without replacing the broker identifier."""
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    out["requested_symbol"] = str(requested_symbol)
    out["finviz_ticker"] = finviz_ticker
    return out


def _attach_finviz_fetch_timestamp(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in payload or payload.get("success") is False:
        return payload
    out = dict(payload)
    fetched_at = datetime.now(timezone.utc)
    out.setdefault("data_fetched_at", format_datetime_utc(fetched_at))
    rows = out.get("items")
    delayed_values = bool(
        out.get("data_delayed") is True
        or out.get("price_source") == _FINVIZ_DELAYED_FRESHNESS
        or out.get("freshness") == _FINVIZ_DELAYED_FRESHNESS
        or (
            isinstance(rows, list)
            and any(
                isinstance(row, dict) and row.get("data_delayed") is True
                for row in rows
            )
        )
    )
    if delayed_values:
        out.setdefault("observation_time_status", "provider_timestamp_unavailable")
        out.setdefault(
            "nominal_provider_delay_minutes",
            {
                "minimum": _FINVIZ_DELAY_MINUTES_MIN,
                "maximum": _FINVIZ_DELAY_MINUTES_MAX,
            },
        )
        out.setdefault(
            "observation_time_note",
            "data_fetched_at is transport time. Finviz does not provide the "
            "observation timestamp, so no observation instant or window is inferred.",
        )
    return out


def _run_logged_tool(
    operation: str,
    fields: Dict[str, Any],
    fn: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    return run_logged_operation(
        logger,
        operation=operation,
        func=lambda: _attach_finviz_fetch_timestamp(fn()),
        **fields,
    )


_FINVIZ_MARKET_KEY_ALIASES = {
    "perf_quart": "perf_quarter",
    "perf_half": "perf_half_year",
}


def _snake_finviz_market_key(value: Any) -> str:
    key = str(value).strip().lower()
    for old, new in (("%", "pct"), ("/", "_"), ("&", "and"), ("-", "_")):
        key = key.replace(old, new)
    normalized = "_".join(part for part in key.replace(".", "").split() if part)
    return _FINVIZ_MARKET_KEY_ALIASES.get(normalized, normalized)


_FOREX_CURRENCY_NAMES = {
    "AUD": "Australian Dollar",
    "CAD": "Canadian Dollar",
    "CHF": "Swiss Franc",
    "EUR": "Euro",
    "GBP": "British Pound",
    "JPY": "Japanese Yen",
    "NZD": "New Zealand Dollar",
    "USD": "US Dollar",
}
_FINVIZ_MARKET_COMPACT_FIELDS = (
    "symbol",
    "display_symbol",
    "name",
    "price",
    "price_status",
    "price_currency",
    "price_source",
    "data_delayed",
    "nominal_provider_delay_minutes_min",
    "nominal_provider_delay_minutes_max",
    "group",
    "perf_5min_pct",
    "perf_hour_pct",
    "perf_day_pct",
    "perf_week_pct",
    "perf_month_pct",
    "perf_quarter_pct",
    "perf_half_year_pct",
    "perf_year_pct",
    "perf_ytd_pct",
)
_FINVIZ_MARKET_PERFORMANCE_PERIOD_FIELDS = (
    ("5_minutes", "perf_5min_pct"),
    ("hour", "perf_hour_pct"),
    ("day", "perf_day_pct"),
    ("week", "perf_week_pct"),
    ("month", "perf_month_pct"),
    ("quarter", "perf_quarter_pct"),
    ("half_year", "perf_half_year_pct"),
    ("year", "perf_year_pct"),
    ("year_to_date", "perf_ytd_pct"),
)
_FINVIZ_SCREEN_COMPACT_FIELDS_BY_VIEW = {
    "overview": (
        "symbol",
        "price",
        "change_pct",
        "volume",
        "pe_ratio",
    ),
    "valuation": (
        "symbol",
        "price",
        "market_cap",
        "pe_ratio",
        "forward_pe",
        "peg",
        "price_to_sales",
        "price_to_book",
    ),
    "financial": (
        "symbol",
        "profit_margin",
        "operating_margin",
        "gross_margin",
        "return_on_assets",
        "return_on_equity",
        "return_on_investment",
        "return_on_invested_capital",
        "current_ratio",
        "debt_to_equity",
    ),
    "ownership": (
        "symbol",
        "insider_own",
        "insider_trans",
        "inst_own",
        "inst_trans",
        "short_float",
        "short_ratio",
    ),
    "performance": (
        "symbol",
        "performance_week",
        "performance_month",
        "performance_quarter",
        "performance_half_year",
        "performance_year",
        "performance_ytd",
    ),
    "technical": (
        "symbol",
        "price",
        "change_pct",
        "volume",
        "rsi_14",
        "high_52w_distance_pct",
        "low_52w_distance_pct",
        "sma20_distance_pct",
        "sma50_distance_pct",
        "sma200_distance_pct",
        "atr_14",
        "beta",
    ),
}
_FINVIZ_SCREEN_FRACTION_PERCENT_FIELDS = frozenset(
    {
        "profit_margin",
        "operating_margin",
        "gross_margin",
        "return_on_assets",
        "return_on_equity",
        "return_on_investment",
        "return_on_invested_capital",
        "insider_own",
        "insider_trans",
        "inst_own",
        "inst_trans",
        "performance_week",
        "performance_month",
        "performance_quarter",
        "performance_half_year",
        "performance_year",
        "performance_ytd",
        "performance_3y",
        "performance_5y",
        "performance_10y",
        "high_52w_distance_pct",
        "low_52w_distance_pct",
        "sma20_distance_pct",
        "sma50_distance_pct",
        "sma200_distance_pct",
        "volatility_w_pct",
        "volatility_m_pct",
        "change_from_open_pct",
        "gap_pct",
        "short_float",
        "dividend_yield",
        "payout",
        "eps_this_year_growth_pct",
        "eps_next_5y_growth_pct",
    }
)
_FINVIZ_SCREEN_PERCENT_FIELDS = _FINVIZ_SCREEN_FRACTION_PERCENT_FIELDS | frozenset(
    {
        "change_pct",
        "price_change_pct",
        "perf_day_pct",
        "perf_week_pct",
        "perf_month_pct",
        "perf_quarter_pct",
        "perf_year_pct",
        "perf_5min_pct",
        "perf_hour_pct",
        "perf_half_year_pct",
        "perf_ytd_pct",
    }
)
_FINVIZ_DETAIL_ERROR = (
    "detail must be one of: compact, standard, summary, full. "
    "Finviz standard/summary output uses the compact shape."
)
_FINVIZ_DELAYED_FRESHNESS = "finviz_delayed"
_FINVIZ_DELAY_MINUTES_MIN = 15
_FINVIZ_DELAY_MINUTES_MAX = 20
_FINVIZ_DELAYED_DATA_QUALITY = "delayed"
_FINVIZ_PERFORMANCE_RANK_BY = {
    "5min": "perf_5min_pct",
    "hour": "perf_hour_pct",
    "day": "perf_day_pct",
    "week": "perf_week_pct",
    "month": "perf_month_pct",
    "quarter": "perf_quarter_pct",
    "half": "perf_half_year_pct",
    "year": "perf_year_pct",
    "ytd": "perf_ytd_pct",
}
_FINVIZ_FOREX_DELAYED_PRICE_WARNING = (
    "Finviz forex prices are delayed web quotes, not executable MT5 bid/ask; "
    "use market_ticker before order placement."
)
_FINVIZ_FUTURES_DELAYED_WARNING = (
    "Finviz futures rows are delayed generic provider series with unknown "
    "contract and roll identity. Use symbols_list to choose a broker contract, "
    "then market_ticker before order placement."
)
_FINVIZ_USD_PRICE_CURRENCY = "USD"
_FINVIZ_CALENDAR_LOCAL_TIMEZONE = "America/New_York"
_FINVIZ_CALENDAR_LOCAL_TZ = ZoneInfo(_FINVIZ_CALENDAR_LOCAL_TIMEZONE)


def _derive_forex_pair_name(symbol: Any) -> Optional[str]:
    text = str(symbol or "").strip().upper()
    if "/" in text:
        left, right = text.split("/", 1)
    elif len(text) == 6:
        left, right = text[:3], text[3:]
    else:
        return None
    left_name = _FOREX_CURRENCY_NAMES.get(left)
    right_name = _FOREX_CURRENCY_NAMES.get(right)
    if left_name and right_name:
        return f"{left_name} / {right_name}"
    return None


def _forex_pair_currencies(symbol: Any) -> Optional[tuple[str, str]]:
    text = str(symbol or "").strip().upper()
    if "/" in text:
        left, right = text.split("/", 1)
    elif len(text) == 6:
        left, right = text[:3], text[3:]
    else:
        return None
    if left in _FOREX_CURRENCY_NAMES and right in _FOREX_CURRENCY_NAMES:
        return left, right
    return None


def _normalize_finviz_forex_symbol(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    source_symbol = str(row.get("symbol") or "").strip().upper()
    mt5_symbol = finviz_forex_symbol_to_mt5(source_symbol)
    if mt5_symbol is not None:
        out["symbol"] = mt5_symbol
        if source_symbol and source_symbol != mt5_symbol:
            out["display_symbol"] = source_symbol
    currencies = (
        _forex_pair_currencies(source_symbol or out.get("symbol"))
        if out.get("price") not in (None, "")
        else None
    )
    if currencies is not None:
        base_currency, quote_currency = currencies
        out["base_currency"] = base_currency
        out["price_currency"] = quote_currency
    return out


def _append_finviz_warning(payload: Dict[str, Any], warning: str) -> None:
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    if warning not in warnings:
        warnings.append(warning)
    payload["warnings"] = warnings


def _finviz_percent_value(
    value: Any,
    *,
    fraction_input: bool = True,
) -> Optional[float]:
    return finviz_percent_value(value, fraction_input=fraction_input)


def _normalize_finviz_market_performance_fields(
    row: Dict[str, Any],
    *,
    rows_key: str,
) -> Dict[str, Any]:
    """Use percentage-point performance fields for every market detail level."""
    out = dict(row)
    if "perf_day" not in out and "perf_pct" in out:
        out["perf_day"] = out["perf_pct"]
    out.pop("perf_pct", None)
    if "perf_week" not in out and out.get("perf_wtd") not in (None, ""):
        out["perf_week"] = out["perf_wtd"]
        out["perf_week_basis"] = "week_to_date"
    out.pop("perf_wtd", None)

    for field, value in tuple(out.items()):
        if (
            not field.startswith("perf_")
            or field.endswith("_pct")
            or field == "perf_week_basis"
        ):
            continue
        pct_value = _finviz_percent_value(
            value,
            fraction_input=rows_key != "futures",
        )
        if pct_value is not None:
            out[f"{field}_pct"] = pct_value
        out.pop(field, None)
    return out


def _compact_finviz_market_row(row: Dict[str, Any], *, rows_key: str) -> Dict[str, Any]:
    compact = _normalize_finviz_market_performance_fields(row, rows_key=rows_key)
    if rows_key == "pairs" and not compact.get("name"):
        derived_name = _derive_forex_pair_name(compact.get("symbol"))
        if derived_name is not None:
            compact["name"] = derived_name
    fields = _FINVIZ_MARKET_COMPACT_FIELDS
    out = {
        field: compact[field]
        for field in fields
        if field in compact and compact[field] not in (None, "")
    }
    if compact.get("perf_week_basis") and "perf_week_pct" in out:
        out["perf_week_basis"] = compact["perf_week_basis"]
    if rows_key == "futures":
        for field in ("contract_identity_status", "series_basis"):
            if compact.get(field) not in (None, ""):
                out[field] = compact[field]
    return out


def _is_known_forex_pair_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    return _derive_forex_pair_name(row.get("symbol")) is not None


def _compact_finviz_market_symbol(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def _finviz_market_row_matches_symbol(row: Any, requested: Any) -> bool:
    if not isinstance(row, dict):
        return False
    needle = _compact_finviz_market_symbol(requested)
    if not needle:
        return False
    candidates: list[str] = []
    for field in ("symbol", "ticker", "name"):
        compact = _compact_finviz_market_symbol(row.get(field))
        if compact:
            candidates.append(compact)
    for token in candidates:
        if token == needle:
            return True
        shorter, longer = (needle, token) if len(needle) <= len(token) else (token, needle)
        if len(shorter) >= 3 and longer.startswith(shorter):
            return True
    return False


def _finviz_market_performance_periods(rows: Any) -> List[str]:
    if not isinstance(rows, list):
        return []
    periods: List[str] = []
    for period, field in _FINVIZ_MARKET_PERFORMANCE_PERIOD_FIELDS:
        if any(
            isinstance(row, dict) and row.get(field) not in (None, "")
            for row in rows
        ):
            periods.append(period)
    return periods


def _finviz_screen_compact_fields(view: Any) -> tuple[str, ...]:
    view_key = str(view or "overview").strip().lower()
    return _FINVIZ_SCREEN_COMPACT_FIELDS_BY_VIEW.get(
        view_key,
        _FINVIZ_SCREEN_COMPACT_FIELDS_BY_VIEW["overview"],
    )


def _compact_finviz_screen_row(
    row: Dict[str, Any],
    *,
    view: str = "overview",
) -> Dict[str, Any]:
    out = {
        field: row[field]
        for field in _finviz_screen_compact_fields(view)
        if field in row
        and (
            row[field] not in (None, "")
            or field in _FINVIZ_FUNDAMENTAL_NUMERIC_KEYS
        )
    }
    for field in (
        "price_source",
        "data_delayed",
        "nominal_provider_delay_minutes_min",
        "nominal_provider_delay_minutes_max",
    ):
        if row.get(field) not in (None, ""):
            out[field] = row[field]
    return out


def _mark_finviz_delayed_price(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    price = _parse_finviz_numeric_value(out.get("price"))
    if price is None:
        return out
    out["price"] = price
    out["price_source"] = _FINVIZ_DELAYED_FRESHNESS
    out["data_delayed"] = True
    out["nominal_provider_delay_minutes_min"] = _FINVIZ_DELAY_MINUTES_MIN
    out["nominal_provider_delay_minutes_max"] = _FINVIZ_DELAY_MINUTES_MAX
    return out


def _attach_finviz_delayed_root_metadata(out: Dict[str, Any]) -> None:
    out["price_source"] = _FINVIZ_DELAYED_FRESHNESS
    out["freshness"] = _FINVIZ_DELAYED_FRESHNESS
    out["data_quality"] = _FINVIZ_DELAYED_DATA_QUALITY
    out["data_delayed"] = True
    out.setdefault(
        "nominal_provider_delay_minutes",
        {
            "minimum": _FINVIZ_DELAY_MINUTES_MIN,
            "maximum": _FINVIZ_DELAY_MINUTES_MAX,
        },
    )


def _finviz_screen_units_for_rows(
    rows: Any,
    *,
    rows_key: Optional[str] = None,
) -> Dict[str, str]:
    if not isinstance(rows, list):
        return {}
    seen_fields = {
        key
        for row in rows
        if isinstance(row, dict)
        for key, value in row.items()
        if value not in (None, "")
    }
    units = {
        key: "percent (1.0 = 1%)"
        for key in seen_fields
        if key in _FINVIZ_SCREEN_PERCENT_FIELDS or key.endswith("_pct")
    }
    if "short_ratio" in seen_fields:
        units["short_ratio"] = "days_to_cover"
    if rows_key == "stocks":
        if "price" in seen_fields:
            units["price"] = "USD_per_share"
        if "market_cap" in seen_fields:
            units["market_cap"] = "USD"
        if "volume" in seen_fields:
            units["volume"] = "shares (provider delayed snapshot)"
    return units


def _resolve_finviz_performance_rank(
    rank_by: Any,
    order: Any,
    *,
    operation: str,
) -> tuple[Optional[str], str, Optional[str], Optional[Dict[str, Any]]]:
    rank_key = None if rank_by in (None, "") else str(rank_by).strip().lower()
    order_key = None if order in (None, "") else str(order).strip().lower()
    if rank_key is None:
        if order_key is not None:
            return None, "desc", None, _finviz_error_payload(
                "order requires rank_by.",
                code=f"{operation}_incompatible_parameters",
                operation=operation,
                details={"invalid": ["order"]},
            )
        return None, "desc", None, None
    field = _FINVIZ_PERFORMANCE_RANK_BY.get(rank_key)
    if field is None:
        return None, "desc", None, _finviz_error_payload(
            "rank_by must be one of: "
            + ", ".join(_FINVIZ_PERFORMANCE_RANK_BY)
            + ".",
            code=f"{operation}_invalid_rank_by",
            operation=operation,
            details={"rank_by": rank_by},
        )
    if order_key is None:
        order_key = "desc"
    aliases = {
        "desc": "desc",
        "descending": "desc",
        "asc": "asc",
        "ascending": "asc",
    }
    resolved_order = aliases.get(order_key)
    if resolved_order is None:
        return None, "desc", None, _finviz_error_payload(
            "order must be one of: desc, asc.",
            code=f"{operation}_invalid_order",
            operation=operation,
            details={"order": order},
        )
    return rank_key, resolved_order, field, None


def _available_finviz_rank_by(rows: List[Any]) -> List[str]:
    available: List[str] = []
    for key, field in _FINVIZ_PERFORMANCE_RANK_BY.items():
        if any(
            isinstance(row, dict) and row.get(field) not in (None, "")
            for row in rows
        ):
            available.append(key)
    return available


def _sort_finviz_performance_rows(
    rows: List[Any],
    *,
    field: str,
    descending: bool,
) -> List[Any]:
    def _key(row: Any) -> tuple[int, float]:
        if not isinstance(row, dict):
            return (1, 0.0)
        value = row.get(field)
        if value in (None, ""):
            return (1, 0.0)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return (1, 0.0)
        return (0, -numeric if descending else numeric)

    return sorted(rows, key=_key)


def _normalize_finviz_market_payload(  # noqa: C901
    result: Dict[str, Any],
    *,
    rows_key: str,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    detail: str = "compact",
    tool: str,
    request: Dict[str, Any],
    symbol_filter: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(result, dict) or "error" in result:
        return result
    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    rows = result.get(rows_key, [])
    key_normalizer = (
        _normalize_finviz_output_key
        if rows_key == "stocks"
        else _snake_finviz_market_key
    )
    normalized_rows = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            normalized_rows.append(row)
            continue
        normalized_row = _canonicalize_finviz_market_row(
            {key_normalizer(key): value for key, value in row.items()}
        )
        if rows_key == "stocks":
            normalized_row = _normalize_finviz_screen_numeric_fields(
                normalized_row
            )
        if rows_key in {"pairs", "coins", "futures"}:
            normalized_row = _normalize_finviz_market_performance_fields(
                normalized_row,
                rows_key=rows_key,
            )
        normalized_rows.append(normalized_row)
    upstream_count = len(normalized_rows)
    symbol_filter_norm: Optional[str] = None
    if rows_key == "pairs":
        normalized_rows = [
            row for row in normalized_rows if _is_known_forex_pair_row(row)
        ]
        if symbol_filter not in (None, ""):
            symbol_filter_norm = finviz_forex_symbol_to_mt5(symbol_filter)
            if symbol_filter_norm is not None:
                normalized_rows = [
                    row
                    for row in normalized_rows
                    if str(row.get("symbol") or "").upper() == symbol_filter_norm
                ]
    elif rows_key in {"coins", "futures"} and symbol_filter not in (None, ""):
        symbol_filter_norm = str(symbol_filter).strip() or None
        if symbol_filter_norm is not None:
            normalized_rows = [
                row
                for row in normalized_rows
                if _finviz_market_row_matches_symbol(row, symbol_filter_norm)
            ]
    rank_by_value: Optional[str] = None
    rank_order_value = "desc"
    rank_field: Optional[str] = None
    if rows_key in {"pairs", "coins", "futures"}:
        rank_by_value, rank_order_value, rank_field, rank_error = (
            _resolve_finviz_performance_rank(
                request.get("rank_by"),
                request.get("order"),
                operation=tool,
            )
        )
        if rank_error is not None:
            return rank_error
        if rank_field is not None:
            available_rank_by = _available_finviz_rank_by(normalized_rows)
            if rank_by_value not in available_rank_by:
                return _finviz_error_payload(
                    (
                        f"rank_by={rank_by_value!r} is not present in this "
                        "provider snapshot."
                    ),
                    code=f"{tool}_rank_by_unavailable",
                    operation=tool,
                    details={
                        "rank_by": rank_by_value,
                        "available": available_rank_by,
                    },
                )
            normalized_rows = _sort_finviz_performance_rows(
                normalized_rows,
                field=rank_field,
                descending=rank_order_value == "desc",
            )
    requested_limit = _coerce_finviz_limit(limit, default=len(normalized_rows))
    effective_limit = _coerce_finviz_limit(
        result.get("limit"),
        default=requested_limit,
    )
    limit_value = min(requested_limit, effective_limit)
    offset_value = _coerce_finviz_offset(offset)
    limited_rows = normalized_rows[offset_value : offset_value + limit_value]
    if detail_mode != "full" and rows_key in {"pairs", "coins", "futures"}:
        output_rows = [
            _compact_finviz_market_row(row, rows_key=rows_key)
            if isinstance(row, dict)
            else row
            for row in limited_rows
        ]
    elif detail_mode != "full" and rows_key == "stocks":
        view = str(request.get("view") or result.get("view") or "overview")
        output_rows = [
            _compact_finviz_screen_row(row, view=view)
            if isinstance(row, dict)
            else row
            for row in limited_rows
        ]
    else:
        output_rows = limited_rows
    out = {key: value for key, value in result.items() if key != rows_key}
    out["items"] = output_rows
    out["row_key"] = "items"
    out["count"] = len(output_rows)
    if symbol_filter_norm is not None:
        out["symbol"] = symbol_filter_norm
    if effective_limit != requested_limit:
        out["requested_limit"] = requested_limit
        _append_finviz_warning(
            out,
            f"limit was capped from {requested_limit} to {effective_limit} by the Finviz provider adapter.",
        )
    available = len(normalized_rows)
    pagination_total = out.get("total") if rows_key == "stocks" else available
    pagination_lower_bound = (
        (out.get("total_lower_bound") or available)
        if rows_key == "stocks"
        else None
    )
    pagination_has_more = bool(
        out.get("has_more") or available > offset_value + len(limited_rows)
    )
    out.pop("omitted_item_count", None)
    _apply_finviz_pagination_contract(
        out,
        returned=len(output_rows),
        limit=limit_value,
        page=int(out.get("page") or request.get("page") or 1),
        offset=offset_value if offset is not None else None,
        total=pagination_total,
        total_lower_bound=pagination_lower_bound,
        has_more=pagination_has_more,
    )
    out["detail"] = detail_mode
    has_price = any(
        isinstance(row, dict) and row.get("price") not in (None, "")
        for row in normalized_rows
    )
    if has_price and rows_key in {"stocks", "coins"}:
        out["price_currency"] = _FINVIZ_USD_PRICE_CURRENCY
    if has_price and rows_key == "pairs":
        out["price_currency_basis"] = "quote_currency"
    if has_price and rows_key in {"stocks", "pairs", "coins"}:
        out["price_source"] = _FINVIZ_DELAYED_FRESHNESS
        out["freshness"] = _FINVIZ_DELAYED_FRESHNESS
    if has_price and rows_key in {"pairs", "coins"}:
        _attach_finviz_delayed_root_metadata(out)
    if has_price and rows_key == "pairs":
        _append_finviz_warning(out, _FINVIZ_FOREX_DELAYED_PRICE_WARNING)
    if rows_key == "pairs" and symbol_filter_norm is not None and not output_rows:
        _append_finviz_warning(
            out,
            f"No Finviz forex row matched symbol {symbol_filter_norm}.",
        )
    if rows_key in {"coins", "futures"} and symbol_filter_norm is not None:
        out["requested_symbol"] = symbol_filter_norm
        provider_symbol = None
        for row in output_rows:
            if isinstance(row, dict) and row.get("symbol") not in (None, ""):
                provider_symbol = str(row.get("symbol"))
                break
        if provider_symbol:
            out["provider_symbol"] = provider_symbol
            out["symbol"] = provider_symbol
        else:
            _append_finviz_warning(
                out,
                (
                    "No Finviz "
                    + ("crypto" if rows_key == "coins" else "futures")
                    + f" row matched symbol {symbol_filter_norm}."
                ),
            )
    if rows_key == "futures":
        for row in output_rows:
            if isinstance(row, dict):
                row.setdefault("contract_identity_status", "unavailable")
                row.setdefault("series_basis", "provider_generic_root_unknown")
        _attach_finviz_delayed_root_metadata(out)
        _append_finviz_warning(out, _FINVIZ_FUTURES_DELAYED_WARNING)
    if rows_key in {"pairs", "coins", "futures"}:
        out["performance_format"] = "percent"
        if rank_field is not None:
            out["rank_by"] = rank_by_value
            out["order"] = rank_order_value
            out["selection_order"] = (
                f"{rank_field}_descending"
                if rank_order_value == "desc"
                else f"{rank_field}_ascending"
            )
        else:
            out["selection_order"] = "provider_table_order"
    units = _finviz_screen_units_for_rows(output_rows, rows_key=rows_key)
    if units:
        out["units"] = units
    if rows_key == "stocks" and any(
        isinstance(row, dict) and row.get("change_pct") not in (None, "")
        for row in output_rows
    ):
        out["change_pct_basis"] = "delayed_price_vs_previous_close"
    if rows_key in {"pairs", "coins", "futures"}:
        limitations: Dict[str, Any] = {}
        periods = _finviz_market_performance_periods(output_rows)
        if periods:
            limitations["performance_periods"] = periods
        if rows_key == "pairs" and has_price:
            limitations["price"] = "delayed_web_quote_not_executable"
        if rows_key == "futures":
            limitations["contract_identity"] = (
                "expiry_exchange_and_roll_basis_unavailable"
            )
            if not has_price:
                limitations["price"] = "not_available_from_source"
        if limitations:
            out["data_limitations"] = limitations
    if detail_mode == "full":
        out["meta"] = _build_tool_contract_meta(
            tool=tool,
            request=request,
            stats={
                "available": available,
                "returned": len(limited_rows),
                "filtered_non_forex": max(0, upstream_count - available),
            },
        )
    return out


def _canonicalize_finviz_symbol_key(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    if "symbol" not in out:
        for source_key in ("ticker", "pair"):
            if source_key in out:
                out["symbol"] = out.pop(source_key)
                break
    return out


def _canonicalize_finviz_market_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = _canonicalize_finviz_symbol_key(row)
    if "name" not in out and "label" in out:
        out["name"] = out.pop("label")
    if "p_e" in out and "pe_ratio" not in out:
        out["pe_ratio"] = out.pop("p_e")
    if "perf" in out and not any(key.startswith("perf_") for key in out):
        out["perf_pct"] = out.pop("perf")
    if "change" in out and "change_pct" not in out:
        out["change_pct"] = out.pop("change")
    for source_key, distance_key in (
        ("high_52w", "high_52w_distance_pct"),
        ("low_52w", "low_52w_distance_pct"),
    ):
        if source_key not in out:
            continue
        if distance_key not in out:
            out[distance_key] = out[source_key]
        del out[source_key]
    change_pct = _finviz_percent_value(out.get("change_pct"))
    if change_pct is not None:
        out["change_pct"] = change_pct
    percent_fields = _FINVIZ_SCREEN_PERCENT_FIELDS | {
        field for field in out if field.endswith("_pct")
    }
    for field in percent_fields:
        if field not in out:
            continue
        pct_value = _finviz_percent_value(
            out.get(field),
            fraction_input=field in _FINVIZ_SCREEN_FRACTION_PERCENT_FIELDS,
        )
        if pct_value is not None:
            out[field] = pct_value
    if _is_known_forex_pair_row(out):
        out = _normalize_finviz_forex_symbol(out)
    out = _mark_finviz_delayed_price(out)
    return out


def _coerce_finviz_limit(limit: Optional[int], *, default: int) -> int:
    if limit is None:
        return max(0, int(default))
    return max(0, int(limit))


def _coerce_finviz_offset(offset: Optional[int]) -> int:
    try:
        return max(0, int(offset or 0))
    except Exception:
        return 0


_FINVIZ_FLAT_PAGINATION_FIELDS = (
    "total",
    "total_count",
    "total_lower_bound",
    "total_is_lower_bound",
    "offset",
    "limit",
    "page",
    "pages",
    "has_more",
    "more_available",
    "next_offset",
    "next_page",
    "truncated",
)


def _apply_finviz_pagination_contract(
    out: Dict[str, Any],
    *,
    returned: int,
    limit: int,
    page: int = 1,
    offset: Optional[int] = None,
    total: Any = None,
    total_lower_bound: Any = None,
    has_more: Optional[bool] = None,
) -> Dict[str, Any]:
    """Replace Finviz page aliases with the canonical offset pagination block."""
    limit_value = max(1, int(limit))
    page_value = max(1, int(page))
    offset_value = (
        (page_value - 1) * limit_value
        if offset is None
        else _coerce_finviz_offset(offset)
    )
    returned_value = max(0, int(returned))

    exact_total: Optional[int]
    try:
        exact_total = None if total in (None, "") else max(0, int(total))
    except (TypeError, ValueError):
        exact_total = None
    try:
        lower_bound = (
            None
            if total_lower_bound in (None, "")
            else max(0, int(total_lower_bound))
        )
    except (TypeError, ValueError):
        lower_bound = None

    if exact_total is not None:
        pagination = build_pagination_meta(
            total=exact_total,
            returned=returned_value,
            offset=offset_value,
            limit=limit_value,
        )
    else:
        if returned_value == 0:
            lower_bound_value = lower_bound if lower_bound is not None else 0
            has_more = False
        else:
            minimum_total = offset_value + returned_value + (1 if has_more else 0)
            lower_bound_value = max(lower_bound or 0, minimum_total)
        pagination = {
            "total": None,
            "total_lower_bound": lower_bound_value,
            "returned": returned_value,
            "offset": offset_value,
            "limit": limit_value,
            "has_more": bool(has_more),
            "more_available": None,
        }

    for field in _FINVIZ_FLAT_PAGINATION_FIELDS:
        out.pop(field, None)
    out["pagination"] = pagination
    return out


def _build_tool_contract_meta(
    *,
    tool: str,
    request: Dict[str, Any],
    stats: Optional[Dict[str, Any]] = None,
    pagination: Optional[Dict[str, Any]] = None,
    legends: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"tool": tool}
    if stats:
        out["stats"] = {
            key: value for key, value in stats.items() if value is not None
        }
    if pagination:
        out["pagination"] = {
            key: value for key, value in pagination.items() if value is not None
        }
    if legends:
        out["legends"] = legends
    return out


def _normalize_finviz_screen_numeric_fields(
    row: Dict[str, Any],
) -> Dict[str, Any]:
    out = dict(row)
    for field in _FINVIZ_FUNDAMENTAL_NUMERIC_KEYS:
        if field in out:
            out[field] = _normalize_finviz_fundamental_value(
                field,
                out.get(field),
            )
    return out


_FINVIZ_OUTPUT_KEY_MAP = {
    "#Shares": "shares",
    "#Shares Total": "shares_total",
    "Datetime": "datetime",
    "For": "for_currency",
    "Market Cap": "market_cap",
    "Market Cap.": "market_cap",
    "ReferenceDate": "reference_date",
    "SEC Form 4": "sec_form_4",
    "SEC Form 4 Link": "sec_form_4_link",
    "Insider Trading": "owner",
    "Insider_id": "insider_id",
    "Ticker": "symbol",
    "ticker": "symbol",
    "Value ($)": "value_usd",
    "dateFrom": "start",
    "dateTo": "end",
    "earningsdate": "earnings_date",
    "EarningsDate": "earnings_date",
    "isearningdateestimate": "is_earning_date_estimate",
    "IsEarningDateEstimate": "is_earning_date_estimate",
    "epsestimate": "eps_estimate",
    "EPSEstimate": "eps_estimate",
    "epsactual": "eps_actual",
    "EPSActual": "eps_actual",
    "epssurprise": "eps_surprise",
    "EPSSurprise": "eps_surprise",
    "epsreportedsurprise": "eps_reported_surprise",
    "EPSReportedSurprise": "eps_reported_surprise",
    "salesestimate": "sales_estimate",
    "SalesEstimate": "sales_estimate",
    "salesactual": "sales_actual",
    "SalesActual": "sales_actual",
    "salessurprise": "sales_surprise",
    "SalesSurprise": "sales_surprise",
    "marketcap": "market_cap",
    "MarketCap": "market_cap",
    "P/E": "pe_ratio",
    "Forward P/E": "forward_pe",
    "P/S": "price_to_sales",
    "P/B": "price_to_book",
    "P/C": "price_to_cash",
    "P/FCF": "price_to_free_cash_flow",
    "Price/Cash": "price_to_cash",
    "Price/Free Cash Flow": "price_to_free_cash_flow",
    "EPS past 3/5Y": "eps_past_3_5_y",
    "Sales past 3/5Y": "sales_past_3_5_y",
    "EPS (ttm)": "eps_ttm",
    "EPS this Y": "eps_this_year_growth_pct",
    "EPS next Y": "eps_next_year_growth_pct",
    "EPS next 5 Y": "eps_next_5y_growth_pct",
    "EPS Y/Y TTM": "eps_yoy_ttm_growth_pct",
    "EPS Q/Q": "eps_qoq_growth_pct",
    "Sales Y/Y TTM": "sales_yoy_ttm_growth_pct",
    "Sales Q/Q": "sales_qoq_growth_pct",
    "EPS next Q": "eps_next_q",
    "52W High": "high_52w",
    "52W Low": "low_52w",
    "RSI (14)": "rsi_14",
    "SMA20": "sma20_distance_pct",
    "SMA50": "sma50_distance_pct",
    "SMA200": "sma200_distance_pct",
    "ATR (14)": "atr_14",
    "ROA": "return_on_assets",
    "ROE": "return_on_equity",
    "ROI": "return_on_investment",
    "ROIC": "return_on_invested_capital",
    "Curr R": "current_ratio",
    "Quick R": "quick_ratio",
    "LT Debt/Eq": "long_term_debt_to_equity",
    "LTDebt/Eq": "long_term_debt_to_equity",
    "Debt/Eq": "debt_to_equity",
    "Outer": "firm",
    "outer": "firm",
    "Gross M": "gross_margin",
    "Oper M": "operating_margin",
    "Oper Margin": "operating_margin",
    "Profit M": "profit_margin",
    "Book/sh": "book_value_per_share",
    "Shs Outstand": "shares_outstanding",
    "Shs Float": "shares_float",
    "Perf Week": "performance_week",
    "Perf Month": "performance_month",
    "Perf Quarter": "performance_quarter",
    "Perf Half Y": "performance_half_year",
    "Perf Year": "performance_year",
    "Perf YTD": "performance_ytd",
    "Perf 3Y": "performance_3y",
    "Perf 5Y": "performance_5y",
    "Perf 10Y": "performance_10y",
    "Volatility W": "volatility_w_pct",
    "Volatility M": "volatility_m_pct",
    "Volatility": "volatility",
    "Cash/sh": "cash_per_share",
    "Cash/Sh": "cash_per_share",
    "Employees": "employees",
    "EV/EBITDA": "ev_ebitda",
    "EV/Sales": "ev_sales",
    "Recom": "recom",
    "Target Price": "target_price",
    "Prev Close": "prev_close",
    "Short Interest": "short_interest",
    "Option/Short": "option_short",
    "EPS/Sales Surpr.": "eps_sales_surpr",
    "EPS/Sales Surprise": "eps_sales_surpr",
    "Earnings": "earnings",
    "Dividend %": "dividend_yield",
    "Dividend Est.": "dividend_est",
    "Dividend TTM": "dividend_ttm",
    "Dividend Ex-Date": "dividend_ex_date",
    "Dividend Gr. 3Y": "dividend_growth_3y_cagr_pct",
    "Dividend Gr. 5Y": "dividend_growth_5y_cagr_pct",
    "Dividend Gr. 3/5Y": "dividend_growth_3_5_y",
}
_FINVIZ_OUTPUT_KEY_ALIASES = {
    "dividend_growth_3y": "dividend_growth_3y_cagr_pct",
    "dividend_growth_5y": "dividend_growth_5y_cagr_pct",
    "eps_next_5_y": "eps_next_5y_growth_pct",
    "eps_next_5y": "eps_next_5y_growth_pct",
    "eps_next_y": "eps_next_year_growth_pct",
    "eps_past_5_y": "eps_past_5y_cagr_pct",
    "eps_past_5y": "eps_past_5y_cagr_pct",
    "eps_q_q": "eps_qoq_growth_pct",
    "eps_this_y": "eps_this_year_growth_pct",
    "eps_y_y_ttm": "eps_yoy_ttm_growth_pct",
    "oper_margin": "operating_margin",
    "perf_quart": "performance_quarter",
    "perf_quart_pct": "performance_quarter",
    "perf_half": "performance_half_year",
    "perf_half_pct": "performance_half_year",
    "change_from_open": "change_from_open_pct",
    "gap": "gap_pct",
    "sales_past_5_y": "sales_past_5y_cagr_pct",
    "sales_past_5y": "sales_past_5y_cagr_pct",
    "sales_q_q": "sales_qoq_growth_pct",
    "sales_y_y_ttm": "sales_yoy_ttm_growth_pct",
    "date_from": "start",
    "date_to": "end",
    "cash_sh": "cash_per_share",
    "cash_per_sh": "cash_per_share",
    "eps_sales_surprise": "eps_sales_surpr",
}
_FINVIZ_FUNDAMENTAL_NUMERIC_KEYS = frozenset(
    {
        "market_cap",
        "price",
        "change_pct",
        "change_from_open_pct",
        "gap_pct",
        "change_price",
        "enterprise_value",
        "income",
        "sales",
        "pe_ratio",
        "forward_pe",
        "peg",
        "price_to_sales",
        "price_to_book",
        "price_to_cash",
        "price_to_free_cash_flow",
        "eps_ttm",
        "eps_this_year_growth_pct",
        "eps_next_year_growth_pct",
        "eps_next_q",
        "eps_next_5y_growth_pct",
        "eps_past_5y_cagr_pct",
        "eps_qoq_growth_pct",
        "eps_yoy_ttm_growth_pct",
        "sales_past_5y_cagr_pct",
        "sales_qoq_growth_pct",
        "sales_yoy_ttm_growth_pct",
        "rsi_14",
        "sma20_distance_pct",
        "sma50_distance_pct",
        "sma200_distance_pct",
        "atr_14",
        "beta",
        "volatility_w_pct",
        "volatility_m_pct",
        "volume",
        "avg_volume",
        "rel_volume",
        "return_on_assets",
        "return_on_equity",
        "return_on_investment",
        "return_on_invested_capital",
        "current_ratio",
        "quick_ratio",
        "long_term_debt_to_equity",
        "debt_to_equity",
        "gross_margin",
        "operating_margin",
        "profit_margin",
        "book_value_per_share",
        "shares_outstanding",
        "shares_float",
        "performance_week",
        "performance_month",
        "performance_quarter",
        "performance_half_year",
        "performance_year",
        "performance_ytd",
        "performance_3y",
        "performance_5y",
        "performance_10y",
        "dividend_yield",
        "dividend_est",
        "dividend_ttm",
        "eps_past_3y_cagr_pct",
        "sales_past_3y_cagr_pct",
        "dividend_growth_3y_cagr_pct",
        "dividend_growth_5y_cagr_pct",
        "payout",
        "insider_own",
        "insider_trans",
        "inst_own",
        "inst_trans",
        "short_float",
        "short_ratio",
        "employees",
        "cash_per_share",
        "ev_ebitda",
        "ev_sales",
        "recom",
        "target_price",
        "prev_close",
        "short_interest",
    }
)
_FINVIZ_NUMERIC_SUFFIX_MULTIPLIERS = {
    "K": 1_000.0,
    "M": 1_000_000.0,
    "B": 1_000_000_000.0,
    "T": 1_000_000_000_000.0,
}
_FINVIZ_INTEGER_NUMERIC_KEYS = frozenset(
    {
        "market_cap",
        "enterprise_value",
        "income",
        "sales",
        "volume",
        "avg_volume",
        "shares_outstanding",
        "shares_float",
        "employees",
    }
)


def _normalize_finviz_output_key(key: Any) -> str:
    text = str(key).strip()
    mapped = _FINVIZ_OUTPUT_KEY_MAP.get(text)
    if mapped:
        return mapped
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    text = text.replace("%", " pct ").replace("&", " and ").replace("/", " ")
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    normalized = text or str(key)
    return _FINVIZ_OUTPUT_KEY_ALIASES.get(normalized, normalized)


def _parse_finviz_numeric_value(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "N/A", "n/a", "None", "none", "null"}:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    multiplier = 1.0
    if text and text[-1].upper() in _FINVIZ_NUMERIC_SUFFIX_MULTIPLIERS:
        multiplier = _FINVIZ_NUMERIC_SUFFIX_MULTIPLIERS[text[-1].upper()]
        text = text[:-1].strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return None
    return float(text) * multiplier


def _parse_finviz_numeric_tokens(value: Any) -> list[float]:
    tokens = re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?", str(value or ""))
    parsed: list[float] = []
    for token in tokens:
        number = _parse_finviz_numeric_value(token)
        if number is not None:
            parsed.append(number)
    return parsed


def _normalize_finviz_fundamental_value(key: str, value: Any) -> Any:
    if key not in _FINVIZ_FUNDAMENTAL_NUMERIC_KEYS and not key.endswith("_pct"):
        return value
    parsed = _parse_finviz_numeric_value(value)
    if parsed is None:
        return None
    if key in _FINVIZ_INTEGER_NUMERIC_KEYS:
        rounded = round(float(parsed))
        if abs(float(parsed) - float(rounded)) <= 1e-6 * max(1.0, abs(float(parsed))):
            return int(rounded)
    return parsed


def _format_finviz_large_number(value: Any) -> Optional[str]:
    number = _parse_finviz_numeric_value(value)
    if number is None:
        return None
    abs_number = abs(float(number))
    for threshold, suffix in (
        (1_000_000_000_000.0, "T"),
        (1_000_000_000.0, "B"),
        (1_000_000.0, "M"),
        (1_000.0, "K"),
    ):
        if abs_number >= threshold:
            text = f"{float(number) / threshold:.2f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return f"{float(number):.0f}"


def _finite_finviz_float(value: Any) -> Optional[float]:
    parsed = _parse_finviz_numeric_value(value)
    if parsed is None:
        return None
    try:
        numeric = float(parsed)
    except Exception:
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def _normalize_finviz_output_row(row: Any) -> Any:
    if not isinstance(row, dict):
        return row
    normalized = {
        _normalize_finviz_output_key(key): value for key, value in row.items()
    }
    conflicts = normalized.get("provider_conflicts")
    if isinstance(conflicts, dict):
        normalized["provider_conflicts"] = {
            _normalize_finviz_output_key(key): value
            for key, value in conflicts.items()
        }
    return normalized


def _normalize_finviz_output_rows(rows: Any) -> Any:
    if not isinstance(rows, list):
        return rows
    return [_normalize_finviz_output_row(row) for row in rows]


def _validate_finviz_detail(detail: str, *, operation: str) -> Optional[Dict[str, Any]]:
    normalized = str(detail or "compact").strip().lower()
    if normalized in {"compact", "standard", "summary", "full"}:
        return None
    return _finviz_error_payload(
        _FINVIZ_DETAIL_ERROR,
        code=f"{operation}_invalid_detail",
        operation=operation,
        details={"detail": detail},
    )
