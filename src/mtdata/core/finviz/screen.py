"""Finviz screener adapter and filter parsers."""

import difflib
import json
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from pydantic import Field

from mtdata.core.finviz.common import (
    _coerce_finviz_offset,
    _finviz_error_payload,
    _normalize_finviz_market_payload,
    _run_logged_tool,
)
from mtdata.core.output_contract import (
    build_pagination_meta,
    normalize_output_verbosity_detail,
)
from mtdata.services.finviz import screen_stocks
from mtdata.shared.schema import DetailLiteral


def _invalid_finviz_screen_filters_error(
    filters: Any,
    *,
    reason: Optional[str] = None,
    invalid_tokens: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if isinstance(filters, str):
        raw = filters.strip()
        if raw and not raw.startswith("{"):
            message = (
                "Invalid filters format. Received a string value, but screener "
                "expects a JSON object (dict) with filter names as keys, "
                "key=value or key:value pairs, or Finviz screener shorthand tokens "
                "like "
                "'cap_largeover,exch_nyse'. Examples: "
                "'country=USA,marketcap=mega', "
                "'country:USA,marketcap:mega', "
                "{'Exchange': 'NASDAQ', 'Sector': 'Technology'} or "
                "'{\"Exchange\": \"NASDAQ\", \"Sector\": \"Technology\"}'. "
                f"Got: {filters!r}"
            )
        else:
            message = (
                "Invalid filters format. Provide filters as key=value or key:value "
                "pairs, a JSON object (dict), or JSON string with filter names as keys "
                "and filter values as values. Example: 'country=USA,marketcap=mega', "
                "{'Exchange': 'NASDAQ', 'Sector': 'Technology'} or "
                "'{\"Exchange\": \"NASDAQ\", \"Sector\": \"Technology\"}'. "
                f"Got: {filters}"
            )
    else:
        message = (
            "Invalid filters format. Provide filters as key=value or key:value pairs, "
            "a JSON object (dict), or JSON string with filter names as keys and filter "
            "values as values. Example: 'country=USA,marketcap=mega', "
            "{'Exchange': 'NASDAQ', 'Sector': 'Technology'} or "
            "'{\"Exchange\": \"NASDAQ\", \"Sector\": \"Technology\"}'. "
            f"Got: {filters}"
        )
    if reason:
        message = f"{message} {reason}"
    details: Dict[str, Any] = {"received_type": type(filters).__name__}
    if invalid_tokens:
        details["invalid_tokens"] = list(invalid_tokens)
    examples = _finviz_screen_filter_name_examples()
    if examples:
        details["valid_filter_examples"] = examples
    payload = _finviz_error_payload(
        message,
        code="finviz_screen_filters_invalid",
        operation="finviz_screen",
        details=details,
    )
    payload["related_tools"] = ["screener"]
    payload["remediation"] = (
        "Run screener(list_filters=true, filter_name='<filter>') to inspect accepted "
        "values, or use shorthand tokens such as fa_pe_under_20."
    )
    return payload


def _finviz_filter_dict() -> Dict[str, Any]:
    from finvizfinance.screener.base import filter_dict

    return filter_dict


def _finviz_screen_shorthand_token_map() -> Dict[str, tuple[str, str]]:
    reverse_filters: Dict[str, tuple[str, str]] = {}
    for filter_name, spec in _finviz_filter_dict().items():
        prefix = str(spec.get("prefix") or "").strip()
        for option_name, option_code in (spec.get("option") or {}).items():
            code = str(option_code or "").strip()
            if prefix and code:
                reverse_filters[f"{prefix}_{code}"] = (str(filter_name), str(option_name))
    return reverse_filters


def _finviz_screen_filter_name_examples(limit: int = 12) -> List[str]:
    return [str(name) for name in list(_finviz_filter_dict().keys())[: max(1, int(limit))]]


def _parse_finviz_screen_shorthand(raw: str) -> Optional[Dict[str, Any]]:
    reverse_filters = _finviz_screen_shorthand_token_map()
    filters: Dict[str, Any] = {}
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        match = reverse_filters.get(token)
        if match is None:
            return None
        filters[match[0]] = match[1]
    return filters or None


def _unknown_finviz_screen_shorthand_tokens(raw: str) -> List[str]:
    reverse_filters = _finviz_screen_shorthand_token_map()
    tokens = [part.strip() for part in raw.split(",") if part.strip()]
    return [token for token in tokens if token not in reverse_filters]


def _compact_finviz_filter_token(value: Any, *, keep_sign: bool = False) -> str:
    text = str(value or "").strip().lower()
    return "".join(
        ch for ch in text if ch.isalnum() or (keep_sign and ch in {"+", "-"})
    )


def _resolve_finviz_filter_option(spec: Dict[str, Any], raw_value: str) -> Optional[str]:
    aliases: Dict[str, str] = {}
    for option_name, option_code in (spec.get("option") or {}).items():
        option_name_text = str(option_name)
        option_code_text = str(option_code)
        aliases[_compact_finviz_filter_token(option_name_text, keep_sign=True)] = option_name_text
        aliases[_compact_finviz_filter_token(option_code_text, keep_sign=True)] = option_name_text
        first_word = option_name_text.split(maxsplit=1)[0]
        if first_word.startswith(("+", "-")):
            aliases[_compact_finviz_filter_token(first_word, keep_sign=True)] = option_name_text

    value_key = _compact_finviz_filter_token(raw_value, keep_sign=True)
    if value_key in aliases:
        return aliases[value_key]
    if value_key.startswith(("+", "-")):
        return aliases.get(value_key[1:])
    return None


def _split_finviz_filter_operator_key(raw_key: str) -> tuple[str, Optional[str]]:
    key = str(raw_key or "").strip()
    compact_key = _compact_finviz_filter_token(key)
    for suffix, option_prefix in (
        ("under", "Under"),
        ("below", "Under"),
        ("over", "Over"),
        ("above", "Over"),
    ):
        marker = f"_{suffix}"
        if key.lower().endswith(marker):
            return key[: -len(marker)], option_prefix
        compact_marker = suffix
        if compact_key.endswith(compact_marker) and len(compact_key) > len(compact_marker):
            return compact_key[: -len(compact_marker)], option_prefix
    return key, None


def _parse_finviz_screen_key_value_filters(raw: str) -> Optional[Dict[str, Any]]:
    if "=" not in raw and ":" not in raw:
        return None

    filter_dict = _finviz_filter_dict()
    filter_names = {
        _compact_finviz_filter_token(name): str(name)
        for name in filter_dict
    }
    parsed: Dict[str, Any] = {}
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        if "=" in token:
            key_raw, value_raw = token.split("=", 1)
        elif ":" in token:
            key_raw, value_raw = token.split(":", 1)
        else:
            return None
        filter_name = filter_names.get(_compact_finviz_filter_token(key_raw))
        option_prefix = None
        if filter_name is None:
            base_key, option_prefix = _split_finviz_filter_operator_key(key_raw)
            filter_name = filter_names.get(_compact_finviz_filter_token(base_key))
        if filter_name is None:
            return None
        option_value = (
            f"{option_prefix} {value_raw}".strip()
            if option_prefix
            else value_raw
        )
        option_name = _resolve_finviz_filter_option(
            filter_dict[filter_name],
            option_value,
        )
        if option_name is None:
            return None
        parsed[filter_name] = option_name
    return parsed or None


def _normalize_finviz_screen_filter_dict(
    filters: Dict[str, Any],
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    filter_dict = _finviz_filter_dict()
    filter_names = {
        _compact_finviz_filter_token(name): str(name)
        for name in filter_dict
    }
    normalized: Dict[str, Any] = {}
    invalid_tokens: List[str] = []
    for key, value in filters.items():
        filter_name = filter_names.get(_compact_finviz_filter_token(key))
        if filter_name is None:
            invalid_tokens.append(str(key))
            continue
        option_name = _resolve_finviz_filter_option(filter_dict[filter_name], value)
        if option_name is None:
            invalid_tokens.append(f"{filter_name}={value}")
            continue
        normalized[filter_name] = option_name
    if invalid_tokens:
        return None, _invalid_finviz_screen_filters_error(
            filters,
            reason=(
                "Unknown filter key or value. Use exact Finviz filter names, "
                "supported shorthand, or key=value aliases such as pe_under=15."
            ),
            invalid_tokens=invalid_tokens,
        )
    return normalized, None


def _resolve_finviz_screen_filters(filters: Any) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if filters is None:
        return None, None
    if isinstance(filters, dict):
        return _normalize_finviz_screen_filter_dict(filters)
    if not isinstance(filters, str):
        return None, _invalid_finviz_screen_filters_error(filters)

    raw = filters.strip()
    if not raw:
        return None, None
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None, _invalid_finviz_screen_filters_error(filters)
        if not isinstance(parsed, dict):
            return None, _invalid_finviz_screen_filters_error(filters)
        return parsed, None
    if "=" in raw or ":" in raw:
        parsed = _parse_finviz_screen_key_value_filters(raw)
        if parsed is not None:
            return parsed, None
        invalid_tokens = [part.strip() for part in raw.split(",") if part.strip()]
        return None, _invalid_finviz_screen_filters_error(
            filters,
            reason=(
                "Unsupported Finviz key=value filter or option. Use Finviz "
                "discrete filters such as beta_under=1, pe_under=15, "
                "country=USA, or native shorthand like cap_largeover."
            ),
            invalid_tokens=invalid_tokens,
        )
    if "_" in raw:
        parsed = _parse_finviz_screen_shorthand(raw)
        if parsed is not None:
            return parsed, None
        invalid_tokens = _unknown_finviz_screen_shorthand_tokens(raw)
        if invalid_tokens:
            return None, _invalid_finviz_screen_filters_error(
                filters,
                reason=(
                    "Unrecognized Finviz shorthand token(s): "
                    f"{', '.join(invalid_tokens)}."
                ),
                invalid_tokens=invalid_tokens,
            )
    return None, _invalid_finviz_screen_filters_error(filters)


def _finviz_filter_search_rank(
    *,
    display_name: str,
    prefix: str,
    options: List[Dict[str, Any]],
    query: str,
) -> tuple[Optional[int], List[Dict[str, Any]]]:
    name_l = display_name.strip().lower()
    prefix_l = prefix.strip().lower()
    matched = [
        option
        for option in options
        if query in str(option.get("value") or "").lower()
        or query in str(option.get("token") or "").lower()
    ]
    if name_l == query or (prefix_l and prefix_l == query):
        return 0, matched
    if name_l.startswith(query) or (prefix_l and prefix_l.startswith(query)):
        return 1, matched
    if query in name_l or (prefix_l and query in prefix_l):
        return 2, matched
    if matched:
        return 3, matched
    return None, []


def finviz_filters_list(
    search: Optional[str] = None,
    filter_name: Optional[str] = None,
    limit: Annotated[int, Field(ge=1)] = 20,
    offset: Annotated[int, Field(ge=0)] = 0,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """List valid Finviz screener filters and accepted values."""
    filter_dict = _finviz_filter_dict()
    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    query = str(search or "").strip().lower()
    filter_query = str(filter_name or "").strip().lower()
    known_filters = {
        str(name).strip().lower(): (str(name), str(spec.get("prefix") or "").strip())
        for name, spec in filter_dict.items()
    }
    known_prefixes = {
        prefix.lower(): (name, prefix)
        for name, prefix in known_filters.values()
        if prefix
    }
    if filter_query and filter_query not in known_filters and filter_query not in known_prefixes:
        choices = list(known_filters) + list(known_prefixes)
        matches = difflib.get_close_matches(filter_query, choices, n=5, cutoff=0.45)
        suggestions = []
        for match in matches:
            display_name, prefix = known_filters.get(match) or known_prefixes[match]
            suggestion = {"filter": display_name, "prefix": prefix}
            if suggestion not in suggestions:
                suggestions.append(suggestion)
        return _finviz_error_payload(
            f"Unknown Finviz filter '{filter_name}'.",
            code="finviz_filters_list_filter_not_found",
            operation="finviz_filters_list",
            details={
                "filter_name": filter_name,
                "suggestions": suggestions,
                "hint": "Use search to discover filters or omit filter_name to list them.",
            },
        )
    ranked_rows: List[tuple[int, int, Dict[str, Any]]] = []
    for original_index, (display_name, spec) in enumerate(filter_dict.items()):
        prefix = str(spec.get("prefix") or "").strip()
        options = [
            {"value": str(option_name), "token": f"{prefix}_{option_code}"}
            for option_name, option_code in (spec.get("option") or {}).items()
            if str(option_name).strip()
        ]
        if filter_query and filter_query not in {
            str(display_name).strip().lower(),
            prefix.lower(),
        }:
            continue
        matched_values: List[Dict[str, Any]] = []
        rank: Optional[int] = None
        if query:
            rank, matched_values = _finviz_filter_search_rank(
                display_name=str(display_name),
                prefix=prefix,
                options=options,
                query=query,
            )
            if rank is None:
                continue
        row: Dict[str, Any] = {
            "filter": str(display_name),
            "prefix": prefix,
            "value_count": len(options),
        }
        if detail_mode == "full" or filter_query:
            row["values"] = options
        elif query and matched_values:
            row["matched_values"] = matched_values[:5]
        ranked_rows.append((rank if rank is not None else 0, original_index, row))
    ranked_rows.sort(key=lambda item: (item[0], item[1]))
    rows = [row for _rank, _index, row in ranked_rows]

    try:
        limit_value = max(1, int(limit or 20))
    except Exception:
        return {"error": "limit must be a positive integer."}
    offset_value = _coerce_finviz_offset(offset)
    limited_rows = rows[offset_value: offset_value + limit_value]
    out: Dict[str, Any] = {
        "success": True,
        "items": limited_rows,
        "row_key": "items",
        "count": len(limited_rows),
        "pagination": build_pagination_meta(
            total=len(rows),
            returned=len(limited_rows),
            offset=offset_value,
            limit=limit_value,
        ),
        "detail": detail_mode,
        "hint": (
            "Use screener filters as Filter=Value pairs or shorthand "
            "tokens such as cap_largeover; pass --filter-name or --detail full "
            "for accepted values."
        ),
    }
    if search not in (None, ""):
        out["search"] = search
    if filter_name not in (None, ""):
        out["filter_name"] = filter_name
    return out


def finviz_screen(
    filters: Optional[Union[str, Dict[str, Any]]] = None,
    order: Optional[str] = None,
    limit: Annotated[int, Field(ge=1)] = 20,
    page: Annotated[int, Field(ge=1)] = 1,
    view: Literal["overview", "valuation", "financial", "ownership", "performance", "technical"] = "overview",
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """
    Screen stocks using Finviz screener with filters.
    
    Parameters
    ----------
    filters : str or dict, optional
        Filter criteria as a JSON string, dict, or Finviz URL shorthand string.
        Dict filter names should be keys with filter values as values. Use the
        exact filter names and values shown on finviz.com screener.
        
        Can be provided as:
        - Finviz shorthand: "cap_largeover,exch_nyse"
        - JSON string: '{"Exchange": "NASDAQ", "Sector": "Technology"}'
        - Dict object: {"Exchange": "NASDAQ", "Sector": "Technology"}
        
        Common filter names: Exchange, Index, Sector, Industry, Country, Market Cap.,
        P/E, Forward P/E, PEG, P/S, P/B, Price/Cash, Price/Free Cash Flow,
        EPS growth this year, EPS growth next year, Sales growth past 5 years,
        EPS growth past 5 years, Dividend Yield, Return on Assets, Return on Equity,
        Return on Investment, Current Ratio, Quick Ratio, LT Debt/Equity, Debt/Equity,
        Gross Margin, Operating Margin, Net Profit Margin, Payout Ratio,
        Insider Ownership, Insider Transactions, Institutional Ownership,
        Institutional Transactions, Float Short, Analyst Recom., Option/Short,
        Earnings Date, Performance, Performance 2, Volatility, RSI (14),
        Gap, 20-Day Simple Moving Average, 50-Day Simple Moving Average,
        200-Day Simple Moving Average, Change, Change from Open, 20-Day High/Low,
        50-Day High/Low, 52-Week High/Low, Pattern, Candlestick, Beta,
        Average True Range, Average Volume, Relative Volume, Current Volume,
        Price, Target Price, IPO Date, Shares Outstanding, Float
        
    order : str
        Sort order. Default "-marketcap" (largest first). Use "price" for
        ascending price. Pagination follows this provider order.
    limit : int
        Max results per page (default 20)
    page : int
        Page number for pagination (default 1)
    view : str
        Data view: overview, valuation, financial, ownership, performance, technical
    detail : str
        Output detail: compact (default) or full. Full includes request/meta.
    
    Returns
    -------
    dict
        Matching stock rows under `items` with compact market-tool metadata.
    
    Examples
    --------
    Screen for tech stocks on NASDAQ (using dict):
    >>> finviz_screen(filters={"Exchange": "NASDAQ", "Sector": "Technology"})
    
    Screen for large NYSE stocks (using Finviz shorthand):
    >>> finviz_screen(filters="cap_largeover,exch_nyse")

    Screen for tech stocks on NASDAQ (using JSON string):
    >>> finviz_screen(filters='{"Exchange": "NASDAQ", "Sector": "Technology"}')
    
    Screen for undervalued large caps:
    >>> finviz_screen(filters={"Market Cap.": "Large ($10bln to $200bln)", "P/E": "Under 15"})
    
    Screen for high dividend stocks with specific view:
    >>> finviz_screen(filters={"Dividend Yield": "Over 5%"}, view="valuation")
    
    Notes
    -----
    - Filter names must exactly match those used on the Finviz screener website
    - Filter values must match the available options for each filter
    - Visit finviz.com/screener.ashx to see available filters and their values
    """
    fields = {"limit": limit, "page": page, "view": view, "order": order, "detail": detail}

    def _run() -> Dict[str, Any]:
        filters_dict, filter_error = _resolve_finviz_screen_filters(filters)
        if filter_error is not None:
            return filter_error

        result = screen_stocks(filters=filters_dict, order=order, limit=limit, page=page, view=view)
        if result.get("success") and isinstance(result.get("stocks"), list):
            return _normalize_finviz_market_payload(
                result,
                rows_key="stocks",
                limit=limit,
                detail=detail,
                tool="finviz_screen",
                request={
                    "filters": filters_dict,
                    "order": order,
                    "limit": limit,
                    "page": page,
                    "view": view,
                },
            )
        return result

    return _run_logged_tool("finviz_screen", fields, _run)
