"""Finviz fundamentals and company-description adapters."""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from mtdata.core.finviz.common import (
    _FINVIZ_DELAY_MINUTES_MAX,
    _FINVIZ_DELAY_MINUTES_MIN,
    _FINVIZ_DELAYED_FRESHNESS,
    _FINVIZ_DETAIL_ERROR,
    _FINVIZ_USD_PRICE_CURRENCY,
    _append_finviz_warning,
    _attach_finviz_symbol_identity,
    _finite_finviz_float,
    _finviz_error_payload,
    _format_finviz_large_number,
    _normalize_finviz_fundamental_value,
    _normalize_finviz_output_key,
    _parse_finviz_numeric_tokens,
    _require_equity_symbol,
    _run_logged_tool,
    _validate_finviz_detail,
)
from mtdata.core.output_contract import normalize_output_verbosity_detail
from mtdata.services.finviz import (
    get_stock_description,
    get_stock_fundamentals,
)
from mtdata.shared.schema import DetailLiteral

_FINVIZ_FUNDAMENTAL_CATEGORIES: Dict[str, tuple[str, ...]] = {
    "summary": (
        "Company",
        "Sector",
        "Industry",
        "Market Cap",
        "Price",
        "Change",
        "P/E",
        "Forward P/E",
        "EPS (ttm)",
        "52W High",
        "52W Low",
        "RSI (14)",
    ),
    "valuation": (
        "Market Cap",
        "P/E",
        "Forward P/E",
        "PEG",
        "P/S",
        "P/B",
        "P/C",
        "P/FCF",
        "EPS (ttm)",
        "EPS next Y",
        "EPS next Q",
    ),
    "performance": (
        "Perf Week",
        "Perf Month",
        "Perf Quarter",
        "Perf Half Y",
        "Perf Year",
        "Perf YTD",
        "Perf 3Y",
        "Perf 5Y",
        "Perf 10Y",
        "52W High",
        "52W Low",
    ),
    "technical": (
        "RSI (14)",
        "SMA20",
        "SMA50",
        "SMA200",
        "ATR (14)",
        "Beta",
        "Volatility W",
        "Volatility M",
        "Price",
        "Change",
        "Volume",
        "Avg Volume",
        "Rel Volume",
    ),
    "dividends": (
        "Dividend Est.",
        "Dividend TTM",
        "Dividend Ex-Date",
        "Dividend Gr. 3Y",
        "Dividend Gr. 5Y",
        "Payout",
    ),
    "ownership": (
        "Insider Own",
        "Insider Trans",
        "Inst Own",
        "Inst Trans",
        "Short Float",
        "Short Ratio",
    ),
    "profile": (
        "Company",
        "Sector",
        "Industry",
        "Country",
        "Exchange",
        "Index",
        "Employees",
        "IPO",
    ),
}
_FINVIZ_FUNDAMENTAL_CATEGORY_ALIASES = {
    "overview": "summary",
    "tech": "technical",
    "valuation_metrics": "valuation",
}
_FINVIZ_MARKET_CAP_BUCKETS = {
    "nano",
    "micro",
    "small",
    "mid",
    "large",
    "mega",
}
_FINVIZ_52W_COMPOUND_FIELDS = {
    "high_52w": ("high_52w_price", "high_52w_distance_pct"),
    "low_52w": ("low_52w_price", "low_52w_distance_pct"),
}
_FINVIZ_DUAL_PERIOD_FIELDS = {
    "eps_past_3_5_y": ("eps_past_3y_cagr_pct", "eps_past_5y_cagr_pct"),
    "sales_past_3_5_y": ("sales_past_3y_cagr_pct", "sales_past_5y_cagr_pct"),
    "dividend_gr_3_5_y": ("dividend_growth_3y_cagr_pct", "dividend_growth_5y_cagr_pct"),
    "dividend_growth_3_5_y": ("dividend_growth_3y_cagr_pct", "dividend_growth_5y_cagr_pct"),
}
_FINVIZ_PERCENT_FUNDAMENTAL_KEYS = frozenset(
    {
        "change_pct",
        "return_on_assets",
        "return_on_equity",
        "return_on_investment",
        "return_on_invested_capital",
        "gross_margin",
        "operating_margin",
        "profit_margin",
        "eps_next_5y_growth_pct",
        "eps_next_year_growth_pct",
        "eps_past_5y_cagr_pct",
        "eps_qoq_growth_pct",
        "eps_this_year_growth_pct",
        "eps_yoy_ttm_growth_pct",
        "sales_past_5y_cagr_pct",
        "sales_qoq_growth_pct",
        "sales_yoy_ttm_growth_pct",
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
        "payout",
        "insider_own",
        "insider_trans",
        "inst_own",
        "inst_trans",
        "short_float",
    }
)
_FINVIZ_CURRENCY_PER_SHARE_FUNDAMENTAL_KEYS = frozenset(
    {
        "eps_ttm",
        "eps_next_q",
    }
)
_FINVIZ_LARGE_NUMBER_FORMAT_KEYS = frozenset(
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


def _expand_finviz_compound_fundamental(
    key: str,
    value: Any,
) -> Optional[Dict[str, Any]]:
    values = _parse_finviz_numeric_tokens(value)
    if key in _FINVIZ_52W_COMPOUND_FIELDS and len(values) >= 2:
        price_key, distance_key = _FINVIZ_52W_COMPOUND_FIELDS[key]
        return {
            price_key: values[0],
            distance_key: values[1],
        }
    if key in _FINVIZ_DUAL_PERIOD_FIELDS and len(values) >= 2:
        first_key, second_key = _FINVIZ_DUAL_PERIOD_FIELDS[key]
        return {
            first_key: values[0],
            second_key: values[1],
        }
    return None


def _finviz_compound_output_keys(key: str) -> tuple[str, ...]:
    if key in _FINVIZ_52W_COMPOUND_FIELDS:
        return _FINVIZ_52W_COMPOUND_FIELDS[key]
    if key in _FINVIZ_DUAL_PERIOD_FIELDS:
        return _FINVIZ_DUAL_PERIOD_FIELDS[key]
    return ()


def _add_finviz_large_number_formats(fundamentals: Dict[str, Any]) -> None:
    for key in sorted(_FINVIZ_LARGE_NUMBER_FORMAT_KEYS):
        if key not in fundamentals:
            continue
        formatted_key = f"{key}_formatted"
        if formatted_key in fundamentals:
            continue
        formatted = _format_finviz_large_number(fundamentals.get(key))
        if formatted:
            fundamentals[formatted_key] = formatted


def _finviz_fundamental_units(fundamentals: Dict[str, Any]) -> Dict[str, str]:
    units: Dict[str, str] = {}
    for key in fundamentals:
        if key.endswith("_pct") or key in _FINVIZ_PERCENT_FUNDAMENTAL_KEYS:
            units[key] = "percent (1.0 = 1%)"
        elif key in _FINVIZ_CURRENCY_PER_SHARE_FUNDAMENTAL_KEYS:
            units[key] = "listing_currency_per_share"
    return units


def _compact_finviz_fundamentals(fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in fundamentals.items()
        if not str(key).endswith("_recomputed")
    }


def _add_finviz_52w_quality_flags(
    fundamentals: Dict[str, Any],
    *,
    include_diagnostics: bool = False,
) -> None:
    price = _finite_finviz_float(fundamentals.get("price"))
    if price is None or price <= 0:
        return
    warnings_out: List[str] = []
    high = _finite_finviz_float(fundamentals.get("high_52w_price"))
    if high is not None and high > 0 and price > high:
        fundamentals["new_52w_high"] = False
        fundamentals["new_52w_high_unconfirmed"] = True
        if include_diagnostics:
            fundamentals["high_52w_distance_pct_recomputed"] = round(
                ((price - high) / high) * 100.0,
                2,
            )
        warnings_out.append(
            "Current price is above the reported 52-week high; upstream 52-week data may be delayed."
        )
    low = _finite_finviz_float(fundamentals.get("low_52w_price"))
    if low is not None and low > 0 and price < low:
        fundamentals["new_52w_low"] = False
        fundamentals["new_52w_low_unconfirmed"] = True
        if include_diagnostics:
            fundamentals["low_52w_distance_pct_recomputed"] = round(
                ((price - low) / low) * 100.0,
                2,
            )
        warnings_out.append(
            "Current price is below the reported 52-week low; upstream 52-week data may be delayed."
        )
    if warnings_out:
        existing = fundamentals.get("data_quality_warnings")
        if not isinstance(existing, list):
            existing = []
        for warning in warnings_out:
            if warning not in existing:
                existing.append(warning)
        fundamentals["data_quality_warnings"] = existing


def _parse_finviz_fields(fields: Optional[Union[str, list[str]]]) -> Optional[list[str]]:
    if fields is None:
        return None
    if isinstance(fields, str):
        return [field.strip() for field in fields.split(",") if field.strip()]
    return [str(field).strip() for field in fields if str(field).strip()]


def _finviz_public_fundamental_keys(field: str) -> tuple[str, ...]:
    output_key = _normalize_finviz_output_key(field)
    if output_key == "change":
        output_key = "change_pct"
    keys = [output_key]
    keys.extend(_finviz_compound_output_keys(output_key))
    if output_key == "market_cap":
        keys.append("market_cap_formatted")
    if output_key == "exchange":
        keys.append("market_cap_category")
    return tuple(dict.fromkeys(keys))


def _finviz_fundamental_field_returned(
    field: str,
    filtered: Dict[str, Any],
) -> bool:
    return any(key in filtered for key in _finviz_public_fundamental_keys(field))


def _resolve_finviz_fundamental_fields(
    fundamentals: Dict[str, Any],
    requested_fields: list[str],
) -> tuple[list[str], list[str]]:
    lookup: Dict[str, str] = {}
    for field in fundamentals:
        for candidate in (field, *_finviz_public_fundamental_keys(field)):
            text = str(candidate).strip()
            if text:
                lookup.setdefault(text.lower(), field)

    selected: list[str] = []
    seen: set[str] = set()
    missing: list[str] = []
    missing_seen: set[str] = set()
    for field in requested_fields:
        if field in fundamentals:
            resolved = field
        else:
            resolved = lookup.get(str(field).strip().lower())
        if resolved is None:
            missing_key = str(field).strip().lower()
            if missing_key not in missing_seen:
                missing.append(field)
                missing_seen.add(missing_key)
            continue
        if resolved not in seen:
            selected.append(resolved)
            seen.add(resolved)
    return selected, missing


def _available_finviz_fundamental_fields(
    fundamentals: Dict[str, Any],
) -> list[str]:
    return sorted(
        {
            public_key
            for field in fundamentals
            for public_key in _finviz_public_fundamental_keys(field)
        }
    )


def _filter_finviz_fundamentals_payload(
    result: Dict[str, Any],
    *,
    detail: str,
    category: str,
    fields: Optional[Union[str, list[str]]],
) -> Dict[str, Any]:
    fundamentals = result.get("fundamentals")
    if not isinstance(fundamentals, dict):
        return result

    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    category_input = str(category or "summary").strip().lower()
    category_mode = _FINVIZ_FUNDAMENTAL_CATEGORY_ALIASES.get(
        category_input,
        category_input,
    )
    if str(detail or "compact").strip().lower() not in {"compact", "standard", "summary", "full"}:
        return _finviz_error_payload(
            _FINVIZ_DETAIL_ERROR,
            code="finviz_fundamentals_invalid_detail",
            operation="finviz_fundamentals",
            details={"detail": detail},
        )
    if category_mode != "all" and category_mode not in _FINVIZ_FUNDAMENTAL_CATEGORIES:
        return _finviz_error_payload(
            (
                "category must be one of: all, "
                + ", ".join(sorted(_FINVIZ_FUNDAMENTAL_CATEGORIES))
                + "."
            ),
            code="finviz_fundamentals_invalid_category",
            operation="finviz_fundamentals",
            details={"category": category},
        )

    requested_fields = _parse_finviz_fields(fields)
    missing_fields: list[str] = []
    if requested_fields is not None:
        selected_fields, missing_fields = _resolve_finviz_fundamental_fields(
            fundamentals,
            requested_fields,
        )
        if not requested_fields or not selected_fields:
            valid_fields = _available_finviz_fundamental_fields(fundamentals)
            error = _finviz_error_payload(
                (
                    "fields must contain at least one available Finviz fundamental "
                    "field."
                ),
                code="finviz_fundamentals_fields_invalid",
                operation="finviz_fundamentals",
                details={
                    "requested_fields": requested_fields,
                    "missing_fields": missing_fields,
                },
            )
            error["valid_values"] = {"fields": valid_fields}
            error["remediation"] = (
                "Choose a field from valid_values.fields, or omit fields and use "
                "category='all' with detail='full' to inspect available metrics."
            )
            return error
        category_out = "custom"
    elif category_mode != "all":
        selected_fields = list(_FINVIZ_FUNDAMENTAL_CATEGORIES[category_mode])
        category_out = category_mode
    else:
        selected_fields = list(fundamentals.keys())
        category_out = "all"

    filtered: Dict[str, Any] = {}
    for field in selected_fields:
        if field not in fundamentals:
            continue
        value = fundamentals[field]
        if value in (None, ""):
            continue
        output_key = _normalize_finviz_output_key(field)
        if output_key == "change":
            output_key = "change_pct"
        if output_key == "exchange" and str(value).strip().lower() in _FINVIZ_MARKET_CAP_BUCKETS:
            output_key = "market_cap_category"
        expanded = _expand_finviz_compound_fundamental(output_key, value)
        if expanded is not None:
            filtered.update(
                {
                    expanded_key: expanded_value
                    for expanded_key, expanded_value in expanded.items()
                    if expanded_value not in (None, "")
                }
            )
            continue
        output_value = _normalize_finviz_fundamental_value(output_key, value)
        if output_value in (None, ""):
            continue
        filtered[output_key] = output_value
    _add_finviz_large_number_formats(filtered)
    if detail_mode == "compact":
        filtered = _compact_finviz_fundamentals(filtered)
    out = dict(result)
    out["currency"] = "USD"
    _add_finviz_52w_quality_flags(
        filtered,
        include_diagnostics=detail_mode == "full",
    )
    out["fundamentals"] = filtered
    if filtered.get("data_quality_warnings"):
        out["trust"] = "degraded"
    units = _finviz_fundamental_units(filtered)
    if units:
        out["units"] = units
    out["detail"] = detail_mode
    out["category"] = category_out
    if "price" in filtered:
        out["price_source"] = _FINVIZ_DELAYED_FRESHNESS
        out["price_currency"] = _FINVIZ_USD_PRICE_CURRENCY
        out["freshness"] = _FINVIZ_DELAYED_FRESHNESS
        filtered["price_source"] = _FINVIZ_DELAYED_FRESHNESS
        filtered["data_delayed"] = True
        filtered["nominal_provider_delay_minutes_min"] = _FINVIZ_DELAY_MINUTES_MIN
        filtered["nominal_provider_delay_minutes_max"] = _FINVIZ_DELAY_MINUTES_MAX
    if category_input != category_mode:
        out["category_requested"] = category_input
    if detail_mode == "full":
        out["available_field_count"] = len(fundamentals)
        out["omitted_field_count"] = sum(
            1
            for field in fundamentals
            if not _finviz_fundamental_field_returned(field, filtered)
        )
    if requested_fields is not None:
        if missing_fields:
            out["missing_fields"] = missing_fields
            out["partial_failure"] = True
            _append_finviz_warning(
                out,
                "Some requested fundamental fields were unavailable and were omitted.",
            )
    return out


def finviz_fundamentals(
    symbol: str,
    detail: DetailLiteral = "compact",  # type: ignore
    category: str = "summary",
    fields: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get fundamental data for a US stock symbol.
    
    Returns metrics like P/E, EPS, market cap, sector, industry, dividend yield,
    52-week range, analyst recommendations, and more.
    
    Parameters
    ----------
    symbol : str
        US stock ticker or broker-style equity symbol (for example AAPL or
        AAPL.NAS). The response distinguishes the requested symbol from the
        normalized Finviz ticker.
    
    Returns
    -------
    dict
        Fundamental metrics for the stock. By default this returns a compact
        summary. ``category="all"`` returns every available metric; compact
        omits diagnostics, while ``detail="full"`` keeps them.
    
    Example
    -------
    >>> finviz_fundamentals("AAPL")
    {"success": True, "symbol": "AAPL", "fundamentals": {"pe_ratio": "28.5", ...}}
    """
    def _run() -> Dict[str, Any]:
        symbol_norm, error = _require_equity_symbol(
            symbol,
            tool_name="finviz_fundamentals",
        )
        if error is not None:
            return error
        result = get_stock_fundamentals(symbol_norm)
        result = _filter_finviz_fundamentals_payload(
            result,
            detail=detail,
            category=category,
            fields=fields,
        )
        return _attach_finviz_symbol_identity(
            result,
            requested_symbol=symbol,
            finviz_ticker=symbol_norm,
        )

    return _run_logged_tool(
        "finviz_fundamentals",
        {"symbol": symbol, "detail": detail, "category": category, "fields": fields},
        _run,
    )


_FINVIZ_DESCRIPTION_COMPACT_CHARS = 600


def _apply_finviz_description_detail(
    result: Dict[str, Any], *, detail: str
) -> Dict[str, Any]:
    """Truncate a long company description for compact detail."""
    if not isinstance(result, dict) or result.get("error"):
        return result
    if str(detail or "compact").strip().lower() == "full":
        return result
    description = result.get("description")
    if not isinstance(description, str):
        return result
    full_length = len(description)
    if full_length <= _FINVIZ_DESCRIPTION_COMPACT_CHARS:
        return result
    truncated = description[:_FINVIZ_DESCRIPTION_COMPACT_CHARS].rstrip()
    sentence_cut = truncated.rfind(". ")
    if sentence_cut >= int(_FINVIZ_DESCRIPTION_COMPACT_CHARS * 0.5):
        truncated = truncated[: sentence_cut + 1]
    out = dict(result)
    out["description"] = truncated
    out["description_truncated"] = True
    out["description_full_length"] = full_length
    out["detail_hint"] = "Use --detail full for the complete description."
    return out


def finviz_description(
    symbol: str,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Get company business description for a US stock.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., AAPL, TSLA)
    detail : str
        Output detail: compact (default) truncates a long description for token
        efficiency; full returns the complete text.

    Returns
    -------
    dict
        Company description text
    """
    def _run() -> Dict[str, Any]:
        detail_error = _validate_finviz_detail(detail, operation="finviz_description")
        if detail_error is not None:
            return detail_error
        symbol_norm, error = _require_equity_symbol(
            symbol,
            tool_name="finviz_description",
        )
        if error is not None:
            return error
        return _attach_finviz_symbol_identity(
            _apply_finviz_description_detail(
                get_stock_description(symbol_norm), detail=detail
            ),
            requested_symbol=symbol,
            finviz_ticker=symbol_norm,
        )

    return _run_logged_tool(
        "finviz_description",
        {"symbol": symbol, "detail": detail},
        _run,
    )
