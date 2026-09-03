"""Finviz service implementation."""
import ast
import datetime
import difflib
import importlib
import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

from ..news_text import normalize_news_text
from .client import (
    get_finviz_http_timeout,
    get_finviz_page_limit_max,
    get_finviz_screener_max_rows,
)
from .dates import (
    FINVIZ_CALENDAR_TIMEZONE,
    _finviz_market_date,
    _finviz_market_time,
    align_to_next_monday_if_weekend,
    finviz_earnings_period_window,
    normalize_finviz_dates_in_rows,
    parse_finviz_earnings_date,
    resolve_date_range,
)
from .pagination import _sanitize_row
from .symbols import looks_like_non_equity_symbol, normalize_finviz_equity_symbol
from .utils import (
    apply_finvizfinance_timeout_patch,
    crypto_day_week_identical,
    crypto_price_display,
)

# Private alias keeps existing tests/call sites patchable.
_apply_finvizfinance_timeout_patch = apply_finvizfinance_timeout_patch


def _earnings_event_has_elapsed(
    value: Any,
    *,
    event_date: datetime.date,
    reference_date: datetime.date,
    reference_time: datetime.time,
) -> bool:
    """Return whether a dated Finviz `/b` or `/a` earnings slot has passed."""
    if event_date < reference_date:
        return True
    if event_date > reference_date:
        return False
    token = str(value or "").strip().lower()
    if token.endswith("/b"):
        return reference_time >= datetime.time(9, 30)
    if token.endswith("/a"):
        return reference_time >= datetime.time(16, 0)
    return False

logger = logging.getLogger(__name__)

# Configuration constants
_FINVIZ_HTTP_TIMEOUT = get_finviz_http_timeout()
_FINVIZ_SCREENER_MAX_ROWS = get_finviz_screener_max_rows()
_FINVIZ_PAGE_LIMIT_MAX = get_finviz_page_limit_max()
_FINVIZ_CALENDAR_PROVIDER_PAGE_SIZE = 50

def _sanitize_error_message(exc: Exception, *, symbol: str | None = None) -> str:
    """Sanitize exception messages to hide internal implementation details.
    
    Strips HTTP URLs, internal parameter structures, and replaces with
    user-friendly error messages.
    
    Parameters
    ----------
    exc : Exception
        The exception to sanitize
    symbol : str, optional
        The symbol that was being requested (used for more specific error messages)
    """
    error_str = str(exc)
    error_lower = error_str.lower()

    if isinstance(exc, ValueError) and error_lower.startswith("invalid order"):
        match = re.match(r"Invalid order\s+([^.]*)", error_str, flags=re.IGNORECASE)
        received = match.group(1).strip() if match else "value"
        return f"Invalid Finviz parameter: Invalid order {received}."
    if isinstance(exc, ValueError) and error_lower.startswith("invalid"):
        return f"Invalid Finviz parameter: {error_str}"
    
    # Check for HTTP error patterns and replace with user-friendly message
    if "404" in error_str and "Client Error" in error_str:
        if symbol and looks_like_non_equity_symbol(symbol.upper()):
            return (
                f"{str(symbol).upper()} is not a Finviz-supported symbol. "
                "finviz_news only covers US equities."
            )
        return "Symbol not found. Please check the ticker symbol and try again."
    if "403" in error_str or "Forbidden" in error_str:
        return "Access denied by Finviz. Retry later; the upstream endpoint may be blocking automated access."
    if "429" in error_str or "too many requests" in error_lower or "rate limit" in error_lower:
        return "Finviz rate limit encountered. Retry after 60 seconds."
    if "401" in error_str or "unauthorized" in error_lower or "authentication" in error_lower:
        return "Finviz rejected the request as unauthorized. The upstream endpoint may now require authentication."
    if "500" in error_str or "Server Error" in error_str:
        return "Finviz service error. Retry later; the upstream service returned a server error."
    if "timeout" in error_lower:
        return "Finviz request timed out. Retry later or reduce the requested page size."
    if "connection" in error_lower:
        return "Connection error while contacting Finviz. Check internet connectivity and retry."
    if any(token in error_lower for token in ("parse", "parser", "schema", "column", "html", "json")):
        return "Finviz response could not be parsed. The upstream page or API may have changed."
    if "no " in error_lower and "available" in error_lower:
        return f"{error_str}. Adjust filters or retry later if Finviz data should be available."
    
    # For other errors, return a generic message instead of full exception
    return "Unable to fetch data from Finviz. Please try again later."


def _finviz_error_kind(message: str) -> tuple[str, bool]:
    """Map a sanitized Finviz error message to a (error_code, retryable) pair."""
    low = message.lower()
    if "symbol not found" in low or "ticker symbol" in low and "not found" in low:
        return "finviz_symbol_not_found", False
    if "access denied" in low or "blocking automated access" in low:
        return "finviz_provider_blocked", True
    if "rate limit" in low:
        return "finviz_rate_limited", True
    if "unauthorized" in low:
        return "finviz_unauthorized", False
    if "service error" in low or "server error" in low:
        return "finviz_upstream_error", True
    if "timed out" in low:
        return "finviz_timeout", True
    if "connection error" in low:
        return "finviz_connection_error", True
    if "could not be parsed" in low:
        return "finviz_parse_error", False
    if "invalid finviz parameter" in low:
        return "finviz_invalid_parameter", False
    if "adjust filters" in low or "available" in low:
        return "finviz_no_data", False
    return "finviz_unavailable", True


def _finviz_error_payload(
    exc: Exception,
    *,
    symbol: Optional[str] = None,
    **context: Any,
) -> Dict[str, Any]:
    """Build the common machine-readable envelope for Finviz failures."""
    message = _sanitize_error_message(exc, symbol=symbol)
    error_code, retryable = _finviz_error_kind(message)
    payload: Dict[str, Any] = {
        "success": False,
        "error": message,
        "error_code": error_code,
        "retryable": retryable,
        "provider": "finviz",
        "remediation": (
            "Retry after the provider backoff interval."
            if error_code == "finviz_rate_limited"
            else "Retry after the upstream condition clears."
            if retryable
            else "Check the request parameters and provider compatibility."
        ),
    }
    if error_code == "finviz_rate_limited":
        payload["retry_after_seconds"] = 60
    if error_code == "finviz_invalid_parameter" and "invalid order" in str(exc).lower():
        raw_error = str(exc)
        order_match = re.search(r"Invalid order\s+['\"]?([^'\".]+)", raw_error, re.I)
        choices_match = re.search(r"Possible order:\s*(\[[\s\S]*\])", raw_error, re.I)
        received = order_match.group(1).strip() if order_match else None
        choices: List[str] = []
        if choices_match:
            try:
                parsed = ast.literal_eval(choices_match.group(1))
                if isinstance(parsed, list):
                    choices = [str(item) for item in parsed]
            except (SyntaxError, ValueError):
                choices = []
        payload["parameter"] = "order"
        payload["received"] = received
        payload["valid_values_count"] = len(choices)
        payload["suggestions"] = difflib.get_close_matches(
            str(received or ""), choices, n=3, cutoff=0.2
        )
        payload["remediation"] = (
            "Use one of the suggested order labels or a documented alias such as "
            "-marketcap, price, volume, or change."
        )
    if symbol:
        payload["symbol"] = normalize_finviz_equity_symbol(symbol)
    payload.update({key: value for key, value in context.items() if value is not None})
    return payload


def _sanitize_pagination(limit: int, page: int) -> tuple[int, int]:
    """Clamp pagination inputs to sane bounds."""
    from .pagination import sanitize_pagination

    return sanitize_pagination(limit, page, page_limit_max=_FINVIZ_PAGE_LIMIT_MAX)


def _paginate_finviz_records(
    items: Any,
    *,
    limit: int,
    page: int,
) -> tuple[List[Any], int, int, int, int]:
    from .pagination import paginate_finviz_records

    return paginate_finviz_records(
        items,
        limit=limit,
        page=page,
        page_limit_max=_FINVIZ_PAGE_LIMIT_MAX,
    )


def _screener_pagination_metadata(
    *,
    fetched_count: int,
    fetch_limit: int,
    limit: int,
    page: int,
) -> Dict[str, Any]:
    from .pagination import screener_pagination_metadata

    return screener_pagination_metadata(
        fetched_count=fetched_count,
        fetch_limit=fetch_limit,
        limit=limit,
        page=page,
    )


def _strip_string_fields_in_rows(rows: List[Dict[str, Any]], *keys: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    wanted = set(keys)
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_out = dict(row)
        for key in wanted:
            value = row_out.get(key)
            if isinstance(value, str):
                row_out[key] = normalize_news_text(value)
        out.append(row_out)
    return out


def _run_screener_view(
    screener: Any,
    *,
    order: str = "Ticker",
    ascend: bool = True,
    limit: int = 20,
    page: int = 1,
    fetch_limit_override: Optional[int] = None,
) -> Any:
    """Run screener_view with bounded rows and no inter-page sleep."""
    from .pagination import run_screener_view

    return run_screener_view(
        screener,
        order=order,
        ascend=ascend,
        limit=limit,
        page=page,
        screener_max_rows=_FINVIZ_SCREENER_MAX_ROWS,
        page_limit_max=_FINVIZ_PAGE_LIMIT_MAX,
        fetch_limit_override=fetch_limit_override,
    )


_FINVIZ_SCREEN_ORDER_ALIASES = {
    "ticker": "Ticker",
    "symbol": "Ticker",
    "company": "Company",
    "marketcap": "Market Cap.",
    "market_cap": "Market Cap.",
    "market_capitalization": "Market Cap.",
    "price": "Price",
    "volume": "Volume",
    "change": "Change",
}


def _resolve_finviz_screen_order(order: Any) -> tuple[str, bool, str]:
    text = str(order or "").strip()
    if not text:
        return "Market Cap.", False, "-marketcap"
    ascend = True
    if text.startswith("-"):
        ascend = False
        text = text[1:].strip()
    elif text.startswith("+"):
        text = text[1:].strip()
    key = text.strip().lower().replace(" ", "_").replace(".", "")
    return _FINVIZ_SCREEN_ORDER_ALIASES.get(key, text), ascend, str(order)


def _finviz_http_get(url: str, *, headers: Dict[str, str], params: Dict[str, Any]) -> Any:
    """HTTP GET helper with centralized timeout and pooled connections."""
    from .client import finviz_http_get

    return finviz_http_get(
        url,
        headers=headers,
        params=params,
        timeout=_FINVIZ_HTTP_TIMEOUT,
    )


def _drop_duplicate_day_week_performance(rows: List[Dict[str, Any]]) -> bool:
    if not crypto_day_week_identical(rows):
        return False
    for row in rows:
        if not isinstance(row, dict):
            continue
        row.pop("Perf Week", None)
        row.pop("Perf WTD", None)
    return True


_FINVIZ_SCREENER_VIEWS = {
    "overview": ("finvizfinance.screener.overview", "Overview"),
    "valuation": ("finvizfinance.screener.valuation", "Valuation"),
    "financial": ("finvizfinance.screener.financial", "Financial"),
    "ownership": ("finvizfinance.screener.ownership", "Ownership"),
    "performance": ("finvizfinance.screener.performance", "Performance"),
    "technical": ("finvizfinance.screener.technical", "Technical"),
}


def _load_finviz_attr(module_name: str, attr_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _get_finviz_stock_quote(symbol: str) -> tuple[str, Any]:
    _apply_finvizfinance_timeout_patch()
    finvizfinance = _load_finviz_attr("finvizfinance.quote", "finvizfinance")
    symbol_norm = normalize_finviz_equity_symbol(symbol)
    return symbol_norm, finvizfinance(symbol_norm)


def _build_finviz_screener(view: str) -> Any:
    module_name, class_name = _FINVIZ_SCREENER_VIEWS.get(
        view,
        _FINVIZ_SCREENER_VIEWS["overview"],
    )
    screener_cls = _load_finviz_attr(module_name, class_name)
    return screener_cls()


def _fetch_finviz_market_performance_rows(
    *,
    module_name: str,
    class_name: str,
    empty_error: str,
) -> List[Dict[str, Any]]:
    _apply_finvizfinance_timeout_patch()
    market_cls = _load_finviz_attr(module_name, class_name)
    market_client = market_cls()
    df = market_client.performance()
    if df is None or df.empty:
        raise ValueError(empty_error)
    records = df.to_dict(orient="records")
    return [_sanitize_row(row) for row in records]


def _extract_finviz_futures_performance_rows(html: str) -> List[Dict[str, Any]]:
    marker = "FinvizInitFuturesPerformance("
    start = str(html or "").find(marker)
    if start < 0:
        raise ValueError("Unable to parse Finviz futures performance data")
    payload = str(html)[start + len(marker):].lstrip()
    data, _ = json.JSONDecoder().raw_decode(payload)
    if not isinstance(data, list):
        raise TypeError("Unexpected Finviz futures performance payload shape")
    return [_sanitize_row(row) for row in data if isinstance(row, dict)]


def _fetch_finviz_futures_performance_rows() -> List[Dict[str, Any]]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://finviz.com/futures.ashx",
    }
    resp = _finviz_http_get(
        "https://finviz.com/futures_performance.ashx",
        headers=headers,
        params={},
    )
    try:
        resp.raise_for_status()
        rows = _extract_finviz_futures_performance_rows(resp.text)
    finally:
        resp.close()
    if not rows:
        raise ValueError("No futures performance data available")
    return rows


def _parse_stock_fundamentals_from_quote_page(stock: Any) -> Dict[str, Any]:
    """Parse the current Finviz quote layout when finvizfinance lags it."""
    soup = getattr(stock, "soup", None)
    if soup is None:
        return {}

    fundamentals: Dict[str, Any] = {}
    company = soup.select_one("h2.quote-header_ticker-wrapper_company")
    if company is not None:
        fundamentals["Company"] = company.get_text(" ", strip=True)

    category_keys = {
        "sec_": "Sector",
        "ind_": "Industry",
        "geo_": "Country",
        "exch_": "Exchange",
    }
    for link in soup.select(".quote-header_categories .quote-header_category"):
        href = str(link.get("href") or "").lower()
        value = link.get_text(" ", strip=True)
        if not value:
            continue
        for token, key in category_keys.items():
            if token in href:
                fundamentals[key] = value
                break

    for table in soup.select("table.snapshot-table2"):
        cells = [cell.get_text(" ", strip=True) for cell in table.select("td")]
        for index in range(0, len(cells) - 1, 2):
            key = cells[index].strip()
            if key:
                fundamentals[key] = cells[index + 1].strip()
    return fundamentals


def get_stock_fundamentals(symbol: str) -> Dict[str, Any]:
    """
    Get fundamental data for a stock symbol.
    
    Returns metrics like P/E, EPS, market cap, sector, industry, etc.
    """
    try:
        symbol_norm, stock = _get_finviz_stock_quote(symbol)
        try:
            fundament = stock.ticker_fundament()
        except AttributeError:
            fundament = _parse_stock_fundamentals_from_quote_page(stock)
        if fundament is None:
            return {
                "success": False,
                "error": f"No fundamental data found for {symbol_norm}.",
                "error_code": "finviz_no_data",
                "retryable": False,
                "remediation": "Check the equity ticker and provider compatibility before retrying.",
                "provider": "finviz",
                "endpoint": "fundamentals",
                "stage": "ticker_fundament",
                "symbol": symbol_norm,
            }
        fundament = _sanitize_row(fundament)
        return {
            "success": True,
            "symbol": symbol_norm,
            "fundamentals": fundament,
        }
    except Exception as e:
        logger.exception(f"Error fetching fundamentals for {symbol}")
        message = _sanitize_error_message(e, symbol=symbol)
        error_code, retryable = _finviz_error_kind(message)
        if error_code == "finviz_unavailable":
            error_code = "finviz_endpoint_failed"
            message = (
                f"Finviz fundamentals failed for "
                f"{normalize_finviz_equity_symbol(symbol)}. Other Finviz endpoints "
                "may still be available."
            )
            remediation = (
                "Retry this endpoint or use screener valuation fields as an "
                "alternative fundamentals source."
            )
        else:
            remediation = (
                "Retry this endpoint after the upstream condition clears."
                if retryable
                else "Check the equity ticker and provider compatibility before retrying."
            )
        payload = {
            "success": False,
            "error": message,
            "error_code": error_code,
            "retryable": retryable,
            "remediation": remediation,
            "provider": "finviz",
            "endpoint": "fundamentals",
            "stage": "ticker_fundament",
            "symbol": normalize_finviz_equity_symbol(symbol),
        }
        if error_code == "finviz_rate_limited":
            payload["retry_after_seconds"] = 60
        return payload


def get_stock_description(symbol: str) -> Dict[str, Any]:
    """Get company description for a stock symbol."""
    try:
        symbol_norm, stock = _get_finviz_stock_quote(symbol)
        desc = stock.ticker_description()
        if not desc:
            return {"error": f"No description found for {symbol}"}
        return {
            "success": True,
            "symbol": symbol_norm,
            "description": desc,
        }
    except Exception as e:
        logger.exception(f"Error fetching description for {symbol}")
        return _finviz_error_payload(e, symbol=symbol, endpoint="description")


def get_stock_news(symbol: str, limit: int = 20, page: int = 1) -> Dict[str, Any]:
    """
    Get latest news for a stock symbol.
    
    Returns list of news items with title, link, date, source.
    """
    try:
        symbol_norm, stock = _get_finviz_stock_quote(symbol)
        news_df = stock.ticker_news()
        if news_df is None or news_df.empty:
            return {"error": f"No news found for {symbol}"}
        news_list, total, safe_limit, safe_page, pages = _paginate_finviz_records(
            news_df,
            limit=limit,
            page=page,
        )
        news_list = _strip_string_fields_in_rows(news_list, "Title", "Source", "Date", "Link")
        return {
            "success": True,
            "symbol": symbol_norm,
            "count": len(news_list),
            "total": total,
            "page": safe_page,
            "pages": pages,
            "news": news_list,
        }
    except Exception as e:
        logger.warning("Error fetching news for %s: %s", symbol, str(e))
        return _finviz_error_payload(e, symbol=symbol, endpoint="news")


def get_stock_insider_trades(symbol: str, limit: int = 20, page: int = 1) -> Dict[str, Any]:
    """
    Get insider trading activity for a stock symbol.
    
    Returns list of insider trades with owner, relationship, date, transaction, cost, shares, value.
    """
    try:
        symbol_norm, stock = _get_finviz_stock_quote(symbol)
        insider_df = stock.ticker_inside_trader()
        if insider_df is None or insider_df.empty:
            return {"error": f"No insider trades found for {symbol}"}
        trades_list, total, safe_limit, safe_page, pages = _paginate_finviz_records(
            insider_df,
            limit=limit,
            page=page,
        )
        trades_list = normalize_finviz_dates_in_rows(trades_list, "Date")
        return {
            "success": True,
            "symbol": symbol_norm,
            "count": len(trades_list),
            "total": total,
            "page": safe_page,
            "pages": pages,
            "insider_trades": trades_list,
        }
    except Exception as e:
        logger.exception(f"Error fetching insider trades for {symbol}")
        return _finviz_error_payload(e, symbol=symbol, endpoint="insider_trades")


def get_stock_ratings(symbol: str) -> Dict[str, Any]:
    """
    Get analyst ratings for a stock symbol.
    
    Returns list of ratings with date, status, analyst, rating, price target.
    """
    try:
        symbol_norm, stock = _get_finviz_stock_quote(symbol)
        ratings_df = stock.ticker_outer_ratings()
        if ratings_df is None or ratings_df.empty:
            return {"error": f"No ratings found for {symbol}"}
        ratings_list = ratings_df.to_dict(orient="records")
        ratings_list = [_sanitize_row(row) for row in ratings_list]
        return {
            "success": True,
            "symbol": symbol_norm,
            "count": len(ratings_list),
            "ratings": ratings_list,
        }
    except Exception as e:
        logger.exception(f"Error fetching ratings for {symbol}")
        return _finviz_error_payload(e, symbol=symbol, endpoint="ratings")


def get_stock_peers(symbol: str) -> Dict[str, Any]:
    """Get peer companies for a stock symbol."""
    try:
        symbol_norm, stock = _get_finviz_stock_quote(symbol)
        peers = stock.ticker_peer()
        if not peers:
            return {"error": f"No peers found for {symbol}"}
        return {
            "success": True,
            "symbol": symbol_norm,
            "peers": peers if isinstance(peers, list) else [peers],
        }
    except Exception as e:
        logger.exception(f"Error fetching peers for {symbol}")
        return _finviz_error_payload(e, symbol=symbol, endpoint="peers")


def screen_stocks(
    filters: Optional[Dict[str, str]] = None,
    order: Optional[str] = None,
    limit: int = 20,
    page: int = 1,
    view: str = "overview",
) -> Dict[str, Any]:
    """
    Screen stocks using Finviz screener.
    
    Parameters
    ----------
    filters : dict, optional
        Filter dictionary, e.g. {"Exchange": "NASDAQ", "Sector": "Technology"}
        Available filters: Exchange, Index, Sector, Industry, Country, Market Cap.,
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
    order : str, optional
        Sort order, e.g. "-marketcap" for descending market cap
    limit : int
        Max results per page (default 20)
    page : int
        Page number (default 1)
    view : str
        Screener view type: "overview", "valuation", "financial", "ownership",
        "performance", "technical"
    
    Returns
    -------
    dict
        Screener results with stock list
    """
    try:
        _apply_finvizfinance_timeout_patch()
        view_lower = view.lower().strip()
        safe_limit, safe_page = _sanitize_pagination(limit, page)
        max_rows = int(get_finviz_screener_max_rows())
        start_idx = (safe_page - 1) * safe_limit
        if start_idx >= max_rows:
            last_page = max(1, (max_rows + safe_limit - 1) // safe_limit)
            return {
                "success": False,
                "error": (
                    f"Requested page {safe_page} starts at offset {start_idx}, "
                    f"beyond the screener fetch cap of {max_rows} rows."
                ),
                "error_code": "pagination_limit_reached",
                "max_rows": max_rows,
                "max_offset": max_rows,
                "last_page": last_page,
                "limit": safe_limit,
                "page": safe_page,
                "remediation": (
                    f"Use --page {last_page} or lower, or reduce --limit. "
                    "The Finviz adapter fetches a bounded prefix, not the full universe."
                ),
            }
        screener = _build_finviz_screener(view_lower)
        
        if filters:
            screener.set_filter(filters_dict=filters)
        order_name, order_ascend, order_applied = _resolve_finviz_screen_order(order)

        df, fetch_limit = _run_screener_view(
            screener,
            order=order_name,
            ascend=order_ascend,
            limit=limit,
            page=page,
        )
        if df is None or df.empty:
            safe_limit, safe_page = _sanitize_pagination(limit, page)
            return {
                "success": True,
                "view": view_lower,
                "filters": filters or {},
                "order": order_applied,
                "count": 0,
                "total": 0,
                "limit": safe_limit,
                "page": safe_page,
                "pages": 0,
                "stocks": [],
                "empty_reason": "no_filter_matches",
                "message": "No stocks matched the filter criteria",
            }

        stocks_list, total, safe_limit, safe_page, _pages = _paginate_finviz_records(
            df,
            limit=limit,
            page=page,
        )
        pagination_meta = _screener_pagination_metadata(
            fetched_count=total,
            fetch_limit=fetch_limit,
            limit=safe_limit,
            page=safe_page,
        )
        return {
            "success": True,
            "view": view_lower,
            "filters": filters or {},
            "order": order_applied,
            "count": len(stocks_list),
            "limit": safe_limit,
            "page": safe_page,
            **pagination_meta,
            "stocks": stocks_list,
        }
    except Exception as e:
        logger.warning("Error running stock screener: %s", e)
        return _finviz_error_payload(e, endpoint="screener")


def get_general_news(news_type: str = "news", limit: int = 20, page: int = 1) -> Dict[str, Any]:
    """
    Get general financial news from Finviz.
    
    Parameters
    ----------
    news_type : str
        Type of news: "news" or "blogs"
    limit : int
        Max items per page
    page : int
        Page number (default 1)
    """
    try:
        _apply_finvizfinance_timeout_patch()
        from finvizfinance.news import News

        fnews = News()
        all_news = fnews.get_news()

        if news_type.lower() == "blogs":
            items = all_news.get("blogs", [])
        else:
            items = all_news.get("news", [])

        # Check if items is empty (handle DataFrame or list)
        if hasattr(items, "empty"):
            if items.empty:
                return {"error": f"No {news_type} found"}
            total = len(items)
        elif not items:
            return {"error": f"No {news_type} found"}
        items_list, total, safe_limit, safe_page, pages = _paginate_finviz_records(
            items,
            limit=limit,
            page=page,
        )
        items_list = _strip_string_fields_in_rows(items_list, "Title", "Source", "Date", "Link")

        return {
            "success": True,
            "type": news_type.lower(),
            "count": len(items_list),
            "total": total,
            "page": safe_page,
            "pages": pages,
            "items": items_list,
        }
    except Exception as e:
        logger.exception("Error fetching general news")
        return _finviz_error_payload(e, endpoint="market_news")


def get_insider_activity(option: str = "latest", limit: int = 50, page: int = 1) -> Dict[str, Any]:
    """
    Get general insider trading activity.
    
    Parameters
    ----------
    option : str
        Type: "latest"/"latest buys"/"latest sales", "top week" variants,
        or "top owner" variants supported by finvizfinance.
    limit : int
        Max items per page
    page : int
        Page number (default 1)
    """
    try:
        _apply_finvizfinance_timeout_patch()
        from finvizfinance.insider import Insider

        finsider = Insider(option=option)
        df = finsider.get_insider()

        if df is None or df.empty:
            return {"error": f"No insider activity found for option '{option}'"}

        symbols_by_form_link, ordered_symbols = _extract_insider_activity_symbols(
            getattr(finsider, "soup", None)
        )
        if symbols_by_form_link or len(ordered_symbols) == len(df.index):
            df = df.copy()
            if "SEC Form 4 Link" in df.columns and symbols_by_form_link:
                for index, form_link in df["SEC Form 4 Link"].items():
                    canonical = symbols_by_form_link.get(str(form_link))
                    if canonical:
                        df.at[index, "Ticker"] = canonical
            elif len(ordered_symbols) == len(df.index):
                df["Ticker"] = ordered_symbols

        identity_columns = [
            column
            for column in (
                "Ticker",
                "Insider Trading",
                "Owner",
                "SEC Form 4 Link",
                "SEC Form 4",
                "Date",
                "Transaction",
                "Cost",
                "#Shares",
                "Shares",
                "Value ($)",
                "#Shares Total",
            )
            if column in df.columns
        ]
        rows_before_deduplication = len(df.index)
        if identity_columns:
            df = df.drop_duplicates(subset=identity_columns, keep="first")
        duplicates_removed = rows_before_deduplication - len(df.index)

        items_list, total, safe_limit, safe_page, pages = _paginate_finviz_records(
            df,
            limit=limit,
            page=page,
        )
        items_list = normalize_finviz_dates_in_rows(items_list, "Date")
        return {
            "success": True,
            "option": option,
            "count": len(items_list),
            "total": total,
            "page": safe_page,
            "pages": pages,
            "insider_trades": items_list,
            "duplicates_removed": int(duplicates_removed),
        }
    except Exception as e:
        logger.exception("Error fetching insider activity")
        return _finviz_error_payload(e, endpoint="insider_activity", option=option)


def _extract_insider_activity_symbols(
    soup: Any,
) -> tuple[Dict[str, str], List[str]]:
    """Extract canonical tickers from the market-wide insider table markup."""
    if soup is None or not callable(getattr(soup, "find_all", None)):
        return {}, []

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        headers = [cell.get_text(strip=True) for cell in rows[0].find_all("th")]
        if "Ticker" not in headers:
            continue
        ticker_index = headers.index("Ticker")
        symbols_by_form_link: Dict[str, str] = {}
        ordered_symbols: List[str] = []
        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) < 5 or ticker_index >= len(cells):
                continue
            ticker_cell = cells[ticker_index]
            symbol = _extract_insider_activity_symbol(ticker_cell)
            if not symbol:
                continue
            ordered_symbols.append(symbol)
            form_anchor = cells[-1].find("a", href=True)
            if form_anchor is not None:
                symbols_by_form_link[str(form_anchor.get("href"))] = symbol
        return symbols_by_form_link, ordered_symbols
    return {}, []


def _extract_insider_activity_symbol(ticker_cell: Any) -> Optional[str]:
    value = ticker_cell.get("data-boxover-ticker")
    if not value:
        canonical_node = ticker_cell.find(attrs={"data-boxover-ticker": True})
        value = (
            canonical_node.get("data-boxover-ticker")
            if canonical_node is not None
            else None
        )
    if value:
        return normalize_finviz_equity_symbol(str(value))

    for anchor in ticker_cell.find_all("a", href=True):
        query = parse_qs(urlparse(str(anchor.get("href"))).query)
        ticker_values = query.get("t")
        if ticker_values and ticker_values[0]:
            return normalize_finviz_equity_symbol(ticker_values[0])
    return None


def get_forex_performance() -> Dict[str, Any]:
    """Get forex currency pairs performance data."""
    try:
        items_list = _fetch_finviz_market_performance_rows(
            module_name="finvizfinance.forex",
            class_name="Forex",
            empty_error="No forex performance data available",
        )
        warnings_out: List[str] = []
        if _drop_duplicate_day_week_performance(items_list):
            warnings_out.append(
                "Finviz returned identical 'Perf Day' and 'Perf Week' values; "
                "omitted weekly performance because only day-level performance is reliable."
            )
        out = {
            "success": True,
            "market": "forex",
            "count": len(items_list),
            "pairs": items_list,
        }
        if warnings_out:
            out["warnings"] = warnings_out
        return out
    except Exception as e:
        logger.exception("Error fetching forex performance")
        return _finviz_error_payload(e, endpoint="forex")


def get_crypto_performance() -> Dict[str, Any]:
    """Get cryptocurrency performance data."""
    try:
        items_list = _fetch_finviz_market_performance_rows(
            module_name="finvizfinance.crypto",
            class_name="Crypto",
            empty_error="No crypto performance data available",
        )
        warnings_out: List[str] = []
        rounded_zero_symbols: List[str] = []
        for row in items_list:
            if not isinstance(row, dict) or "Price" not in row:
                continue
            price_display = crypto_price_display(row.get("Price"))
            if price_display is not None and float(price_display) == 0.0:
                row["Price"] = None
                row["Price Status"] = "unavailable_provider_rounded_zero"
                symbol = str(row.get("Ticker") or row.get("Name") or "unknown")
                rounded_zero_symbols.append(symbol)
            elif price_display is not None:
                row["Price"] = price_display
        if rounded_zero_symbols:
            warnings_out.append(
                "Finviz returned zero for crypto prices that its feed cannot "
                "represent at sub-penny precision; omitted those prices for: "
                + ", ".join(rounded_zero_symbols)
                + "."
            )
        if _drop_duplicate_day_week_performance(items_list):
            warnings_out.append(
                "Finviz returned identical 'Perf Day' and 'Perf Week' values; "
                "omitted weekly performance because only day-level performance is reliable."
            )

        out = {
            "success": True,
            "market": "crypto",
            "count": len(items_list),
            "coins": items_list,
        }
        if warnings_out:
            out["warnings"] = warnings_out
        return out
    except Exception as e:
        logger.exception("Error fetching crypto performance")
        return _finviz_error_payload(e, endpoint="crypto")


def get_futures_performance() -> Dict[str, Any]:
    """Get futures market performance data."""
    try:
        items_list = _fetch_finviz_futures_performance_rows()
        return {
            "success": True,
            "market": "futures",
            "count": len(items_list),
            "futures": items_list,
        }
    except Exception as e:
        logger.exception("Error fetching futures performance")
        return _finviz_error_payload(e, endpoint="futures")


def get_earnings_calendar(
    period: str = "This Week",
    limit: int = 50,
    page: int = 1,
    include_elapsed: bool = False,
) -> Dict[str, Any]:
    """Get upcoming earnings calendar from Finviz.

    Notes
    -----
    finvizfinance exposes earnings via ``finvizfinance.earnings.Earnings``.
    Supported periods (per library): "This Week", "Next Week", "Previous Week",
    "This Month".
    """
    try:
        _apply_finvizfinance_timeout_patch()
        from finvizfinance.screener.financial import Financial

        allowed_periods = {"This Week", "Next Week", "Previous Week", "This Month"}
        if period not in allowed_periods:
            raise ValueError(
                "Invalid period '{period}'. Available period: {periods}".format(
                    period=period,
                    periods=sorted(allowed_periods),
                )
            )

        screener = Financial()
        screener.set_filter(filters_dict={"Earnings Date": period})
        period_key = period.lower().replace(" ", "-")
        filter_elapsed = not include_elapsed and period_key in {
            "this-week",
            "this-month",
        }
        reference_date = _finviz_market_date()
        reference_time = _finviz_market_time()
        period_window = finviz_earnings_period_window(
            period_key,
            reference_date,
        )
        reference_at = datetime.datetime.combine(
            reference_date,
            reference_time,
        ).replace(tzinfo=ZoneInfo(FINVIZ_CALENDAR_TIMEZONE))
        requested_end = max(1, int(limit)) * max(1, int(page))
        scan_limit = min(
            _FINVIZ_SCREENER_MAX_ROWS,
            max(50, requested_end + 1),
        )
        df = None
        fetch_limit = 0
        source_count = 0
        source_complete = False
        elapsed_filter_applied = False
        period_filter_applied = False
        provider_returned_rows = False
        period_rows_rejected = 0
        while True:
            df, fetch_limit = _run_screener_view(
                screener,
                order="Earnings Date",
                limit=scan_limit,
                page=1,
                fetch_limit_override=scan_limit,
            )
            if df is None or df.empty:
                break
            provider_returned_rows = True
            source_count = len(df.index)
            source_complete = source_count < fetch_limit
            elapsed_filter_applied = False
            period_filter_applied = False
            period_rows_rejected = 0
            keep_positions = []
            earnings_column = "Earnings" if "Earnings" in df.columns else None
            if earnings_column is not None:
                period_filter_applied = True
                elapsed_filter_applied = filter_elapsed
                for position, value in enumerate(df[earnings_column].tolist()):
                    earnings_date = parse_finviz_earnings_date(
                        value,
                        reference_date=reference_date,
                        period_window=period_window,
                    )
                    if earnings_date is None:
                        period_rows_rejected += 1
                        continue
                    if filter_elapsed and _earnings_event_has_elapsed(
                        value,
                        event_date=earnings_date,
                        reference_date=reference_date,
                        reference_time=reference_time,
                    ):
                        continue
                    keep_positions.append(position)
            else:
                period_rows_rejected = source_count
            df = df.iloc[keep_positions].reset_index(drop=True)
            if (
                len(df.index) > requested_end
                or source_complete
                or fetch_limit >= _FINVIZ_SCREENER_MAX_ROWS
                or earnings_column is None
            ):
                break
            next_scan_limit = min(_FINVIZ_SCREENER_MAX_ROWS, fetch_limit * 2)
            if next_scan_limit <= fetch_limit:
                break
            scan_limit = next_scan_limit

        if not provider_returned_rows:
            return {"error": "No earnings calendar data available"}

        items_list, total, safe_limit, safe_page, _pages = _paginate_finviz_records(
            df,
            limit=limit,
            page=page,
        )
        if not source_complete:
            requested_page_end = safe_limit * safe_page
            pagination_meta = {
                "total": None,
                "pages": None,
                # The provider prefix may be truncated, but every CLI page is
                # drawn from this same filtered prefix. Do not advertise a page
                # that the caller cannot actually retrieve.
                "has_more": requested_page_end < total,
                "total_lower_bound": total,
                "truncated": True,
            }
        else:
            pagination_meta = _screener_pagination_metadata(
                fetched_count=total,
                fetch_limit=fetch_limit,
                limit=safe_limit,
                page=safe_page,
            )
        out = {
            "success": True,
            "period": period,
            "include_elapsed": bool(include_elapsed),
            "period_filter_applied": period_filter_applied,
            "period_start": period_window[0].isoformat(),
            "period_end": period_window[1].isoformat(),
            "period_rows_rejected": int(period_rows_rejected),
            "elapsed_filter_applied": elapsed_filter_applied,
            "calendar_reference_date": reference_date.isoformat(),
            "calendar_reference_at": reference_at.isoformat(timespec="seconds"),
            "calendar_timezone": FINVIZ_CALENDAR_TIMEZONE,
            "count": len(items_list),
            "limit": safe_limit,
            "page": safe_page,
            **pagination_meta,
            "earnings": items_list,
        }
        warnings_out: List[str] = []
        if period_rows_rejected:
            warnings_out.append(
                f"Rejected {period_rows_rejected} provider row(s) whose earnings "
                f"date could not be reconciled with {period_window[0].isoformat()} "
                f"through {period_window[1].isoformat()}."
            )
            out["partial"] = True
        if not source_complete:
            out["source_incomplete"] = True
            warnings_out.append(
                f"The provider earnings scan stopped after {fetch_limit} source rows "
                "before the period was exhausted; results are a bounded prefix. "
                "Use calendar(kind='earnings') for the detailed "
                "date-range feed."
            )
            out["related_tools"] = ["calendar"]
        if warnings_out:
            out["warnings"] = warnings_out
        return out
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("Error fetching earnings calendar")
        return _finviz_error_payload(e, endpoint="earnings_period")


def _calendar_event_identity(event: Dict[str, Any]) -> tuple[Any, ...]:
    calendar_id = event.get("calendarId", event.get("calendar_id"))
    if calendar_id not in (None, ""):
        return ("calendar_id", str(calendar_id).strip())

    fields = (
        "date",
        "event",
        "ticker",
        "symbol",
        "category",
        "reference",
        "country",
        "currency",
    )
    composite = tuple(str(event.get(field) or "").strip().casefold() for field in fields)
    if any(composite):
        return ("composite", *composite)
    return ("raw", json.dumps(event, sort_keys=True, default=str))


def _calendar_value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _calendar_value_is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _merge_calendar_event_variants(
    retained: Dict[str, Any],
    variant: Dict[str, Any],
) -> tuple[Dict[str, Any], bool]:
    merged = dict(retained)
    conflicts = dict(merged.get("providerConflicts") or {})
    for field, value in variant.items():
        if field == "providerConflicts":
            continue
        if field in conflicts:
            values = list(conflicts[field])
            known = {_calendar_value_key(item) for item in values}
            if not _calendar_value_is_empty(value) and _calendar_value_key(value) not in known:
                values.append(value)
            conflicts[field] = sorted(values, key=_calendar_value_key)
            merged[field] = None
            continue
        if field not in merged or _calendar_value_is_empty(merged.get(field)):
            merged[field] = value
            continue
        if _calendar_value_is_empty(value):
            continue
        if _calendar_value_key(merged[field]) == _calendar_value_key(value):
            continue
        conflicts[field] = sorted(
            [merged[field], value],
            key=_calendar_value_key,
        )
        merged[field] = None
    if conflicts:
        merged["providerConflicts"] = conflicts
    return merged, bool(conflicts)


def _deduplicate_calendar_events(
    events: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], int, int]:
    by_identity: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    duplicate_variants = 0
    for event in events:
        identity = _calendar_event_identity(event)
        retained = by_identity.get(identity)
        if retained is None:
            by_identity[identity] = dict(event)
            continue
        duplicate_variants += 1
        by_identity[identity], _has_conflict = _merge_calendar_event_variants(
            retained,
            event,
        )
    deduplicated = list(by_identity.values())
    conflict_events = sum(
        1 for event in deduplicated if event.get("providerConflicts")
    )
    return deduplicated, duplicate_variants, conflict_events


def get_economic_calendar(
    limit: int = 100,
    page: int = 1,
    impact: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Get Finviz economic calendar (macro releases)."""
    safe_limit, safe_page = _sanitize_pagination(limit, page)

    impact_levels: Optional[list[str]] = None
    if impact is not None:
        allowed = {"low", "medium", "high"}
        parts = [
            part.strip().lower()
            for part in str(impact).split(",")
            if part.strip()
        ]
        invalid = [part for part in parts if part not in allowed]
        if not parts or invalid:
            return {
                "error": "Invalid impact '{impact}'. Expected one or more of: low, medium, high".format(
                    impact=impact
                )
            }
        impact_levels = list(dict.fromkeys(parts))
    try:
        # Finviz migrated the calendar UI to client-side rendering; the legacy
        # finvizfinance HTML table parser often returns no rows. Prefer the JSON API.
        default_days = 7
        date_from, date_to = resolve_date_range(
            date_from=date_from,
            date_to=date_to,
            default_days=default_days,
        )

        range_start = datetime.date.fromisoformat(date_from)
        range_end = datetime.date.fromisoformat(date_to)
        events = []
        cursor = range_start
        while cursor <= range_end:
            window_end = min(range_end, cursor + datetime.timedelta(days=6))
            api_date_from = align_to_next_monday_if_weekend(cursor.isoformat())
            if datetime.date.fromisoformat(api_date_from) > window_end:
                cursor = window_end + datetime.timedelta(days=1)
                continue
            for event in _fetch_finviz_economic_calendar_items(
                date_from=api_date_from,
                date_to=window_end.isoformat(),
            ):
                events.append(event)
            cursor = window_end + datetime.timedelta(days=1)
        events = _filter_calendar_events_by_date(events, date_from=date_from, date_to=date_to)
        events, duplicate_variants, conflict_events = _deduplicate_calendar_events(
            events
        )

        if impact_levels is not None:
            impact_values = {
                {"low": 1, "medium": 2, "high": 3}[level] for level in impact_levels
            }
            events = [
                e
                for e in events
                if _calendar_importance_value(e.get("importance")) in impact_values
            ]

        events.sort(
            key=lambda event: (
                str(event.get("date", "")),
                str(event.get("calendarId", event.get("calendar_id", ""))),
                str(event.get("event", "")),
            )
        )

        total = len(events)
        start_idx = (safe_page - 1) * safe_limit
        end_idx = start_idx + safe_limit
        items_list = events[start_idx:end_idx]

        message = None
        impact_norm = ",".join(impact_levels) if impact_levels else None
        if impact_norm and total == 0:
            message = "No economic releases matched impact='{impact}'".format(impact=impact_norm)

        result: Dict[str, Any] = {
            "success": True,
            "source": "finviz_api",
            "impact": impact_norm,
            "dateFrom": date_from,
            "dateTo": date_to,
            "calendarTimezone": FINVIZ_CALENDAR_TIMEZONE,
            "count": len(items_list),
            "total": total,
            "page": safe_page,
            "pages": (total + safe_limit - 1) // safe_limit if total else 0,
            "items": items_list,
            "message": message,
            "duplicateVariantsMerged": duplicate_variants,
            "providerConflictEvents": conflict_events,
            "eventIdentity": (
                "calendarId when present; otherwise date, event, ticker/symbol, "
                "category, reference, country, and currency"
            ),
        }
        if duplicate_variants:
            result["warnings"] = [
                f"Merged {duplicate_variants} duplicate provider event variant(s) "
                "before impact filtering and pagination. Conflicting non-empty fields "
                "are null with alternatives in providerConflicts."
            ]
        return result
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("Error fetching economic calendar")
        return _finviz_error_payload(e, endpoint="calendar_economic")


def get_earnings_calendar_api(
    limit: int = 50,
    page: int = 1,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Get Finviz earnings calendar via the Finviz JSON API."""
    try:
        safe_limit, safe_page = _sanitize_pagination(limit, page)
        default_days = 7 if (date_from is not None and date_to is None) else 30
        date_from, date_to = resolve_date_range(date_from=date_from, date_to=date_to, default_days=default_days)
        payload = _fetch_finviz_calendar_client_page(
            kind="earnings",
            date_from=date_from,
            date_to=date_to,
            page=safe_page,
            limit=safe_limit,
        )
        items = payload.get("items") or []
        total = int(payload.get("totalItemsCount") or len(items))
        pages = (total + safe_limit - 1) // safe_limit if total else 0
        return {
            "success": True,
            "source": "finviz_api",
            "calendar": "earnings",
            "dateFrom": date_from,
            "dateTo": date_to,
            "calendarTimezone": FINVIZ_CALENDAR_TIMEZONE,
            "count": len(items),
            "total": total,
            "page": safe_page,
            "pages": pages,
            "items": items,
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("Error fetching earnings calendar (API)")
        return _finviz_error_payload(e, endpoint="calendar_earnings")


def get_dividends_calendar_api(
    limit: int = 50,
    page: int = 1,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Get Finviz dividends calendar via the Finviz JSON API."""
    try:
        safe_limit, safe_page = _sanitize_pagination(limit, page)
        default_days = 7 if (date_from is not None and date_to is None) else 30
        date_from, date_to = resolve_date_range(date_from=date_from, date_to=date_to, default_days=default_days)
        requested_date_from = date_from
        requested_date_to = date_to
        payload = _fetch_finviz_calendar_client_page(
            kind="dividends",
            date_from=date_from,
            date_to=date_to,
            page=safe_page,
            limit=safe_limit,
        )
        items = payload.get("items") or []
        range_metadata: Dict[str, Any] = {}
        market_date = _finviz_market_date()
        requested_start = datetime.date.fromisoformat(date_from)
        requested_end = datetime.date.fromisoformat(date_to)
        if not items and requested_start < market_date <= requested_end:
            supported_start = market_date.isoformat()
            payload = _fetch_finviz_calendar_client_page(
                kind="dividends",
                date_from=supported_start,
                date_to=date_to,
                page=safe_page,
                limit=safe_limit,
            )
            items = payload.get("items") or []
            date_from = supported_start
            range_metadata = {
                "requested_start": requested_date_from,
                "requested_end": requested_date_to,
                "supported_start": supported_start,
                "range_complete": False,
                "partial": True,
                "range_recovery": "current_forward_retry",
                "warnings": [
                    "Finviz returned no dividend rows for a range beginning before "
                    "the current New York date. Results were retried from the current "
                    "date; the earlier portion is not represented."
                ],
            }
        total = int(payload.get("totalItemsCount") or len(items))
        pages = (total + safe_limit - 1) // safe_limit if total else 0
        return {
            "success": True,
            "source": "finviz_api",
            "calendar": "dividends",
            "dateFrom": date_from,
            "dateTo": date_to,
            "calendarTimezone": FINVIZ_CALENDAR_TIMEZONE,
            "count": len(items),
            "total": total,
            "page": safe_page,
            "pages": pages,
            "items": items,
            **range_metadata,
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("Error fetching dividends calendar (API)")
        return _finviz_error_payload(e, endpoint="calendar_dividends")


def _filter_calendar_events_by_date(
    events: List[Dict[str, Any]],
    *,
    date_from: str,
    date_to: str,
) -> List[Dict[str, Any]]:
    """Filter events to the inclusive [date_from, date_to] date range."""
    df = datetime.date.fromisoformat(date_from)
    dt = datetime.date.fromisoformat(date_to)

    filtered: List[Dict[str, Any]] = []
    for event in events:
        raw = event.get("date")
        if not raw:
            continue
        try:
            if isinstance(raw, str):
                s = raw.strip()
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                if "T" in s:
                    d = datetime.datetime.fromisoformat(s).date()
                else:
                    d = datetime.date.fromisoformat(s)
            elif isinstance(raw, datetime.datetime):
                d = raw.date()
            elif isinstance(raw, datetime.date):
                d = raw
            else:
                continue
        except ValueError:
            continue

        if df <= d <= dt:
            filtered.append(event)

    return filtered


def _calendar_importance_value(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fetch_finviz_economic_calendar_items(date_from: str, date_to: str) -> List[Dict[str, Any]]:
    """Fetch raw economic calendar items from Finviz's JSON API."""
    url = "https://finviz.com/api/calendar/economic"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://finviz.com/calendar.ashx",
    }
    params = {"dateFrom": date_from, "dateTo": date_to}

    resp = _finviz_http_get(url, headers=headers, params=params)
    try:
        resp.raise_for_status()
        data = resp.json()
    finally:
        resp.close()
    if not isinstance(data, list):
        raise TypeError("Unexpected response type from Finviz API: {t}".format(t=type(data).__name__))

    items: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            items.append(item)
    return items


def _fetch_finviz_calendar_paged(
    *,
    kind: Literal["earnings", "dividends"],
    date_from: str,
    date_to: str,
    page: int,
    page_size: int,
) -> Dict[str, Any]:
    """Fetch a paged calendar payload from Finviz's JSON API."""

    url = f"https://finviz.com/api/calendar/{kind}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://finviz.com/calendar.ashx",
    }
    params = {
        "dateFrom": date_from,
        "dateTo": date_to,
        "page": max(1, int(page)),
        "pageSize": max(1, int(page_size)),
    }

    resp = _finviz_http_get(url, headers=headers, params=params)
    try:
        resp.raise_for_status()
        data = resp.json()
    finally:
        resp.close()
    if not isinstance(data, dict):
        raise TypeError("Unexpected response type from Finviz API: {t}".format(t=type(data).__name__))
    if "items" not in data or not isinstance(data.get("items"), list):
        raise TypeError("Unexpected payload shape from Finviz API (missing items list)")
    
    items = data.get("items") or []
    data["items"] = [_clean_calendar_item(item) for item in items]
    return data


def _fetch_finviz_calendar_client_page(
    *,
    kind: Literal["earnings", "dividends"],
    date_from: str,
    date_to: str,
    page: int,
    limit: int,
) -> Dict[str, Any]:
    """Map a client page onto Finviz's fixed-size provider pages."""
    safe_limit, safe_page = _sanitize_pagination(limit, page)
    client_offset = (safe_page - 1) * safe_limit

    first_payload = _fetch_finviz_calendar_paged(
        kind=kind,
        date_from=date_from,
        date_to=date_to,
        page=1,
        page_size=_FINVIZ_CALENDAR_PROVIDER_PAGE_SIZE,
    )
    try:
        provider_page_size = max(
            1,
            int(
                first_payload.get("pageSize")
                or _FINVIZ_CALENDAR_PROVIDER_PAGE_SIZE
            ),
        )
    except (TypeError, ValueError):
        provider_page_size = _FINVIZ_CALENDAR_PROVIDER_PAGE_SIZE
    first_items = list(first_payload.get("items") or [])
    try:
        provider_total = max(
            0,
            int(first_payload.get("totalItemsCount") or len(first_items)),
        )
    except (TypeError, ValueError):
        provider_total = len(first_items)

    first_provider_page = (client_offset // provider_page_size) + 1
    last_client_index = client_offset + safe_limit - 1
    last_provider_page = (last_client_index // provider_page_size) + 1
    provider_payloads: Dict[int, Dict[str, Any]] = {1: first_payload}
    for provider_page in range(first_provider_page, last_provider_page + 1):
        if provider_page in provider_payloads:
            continue
        if provider_total and (provider_page - 1) * provider_page_size >= provider_total:
            break
        payload = _fetch_finviz_calendar_paged(
            kind=kind,
            date_from=date_from,
            date_to=date_to,
            page=provider_page,
            page_size=_FINVIZ_CALENDAR_PROVIDER_PAGE_SIZE,
        )
        provider_payloads[provider_page] = payload
        if not payload.get("items"):
            break

    window_start = (first_provider_page - 1) * provider_page_size
    window_items: List[Any] = []
    for provider_page in range(first_provider_page, last_provider_page + 1):
        payload = provider_payloads.get(provider_page)
        if payload is None:
            break
        window_items.extend(list(payload.get("items") or []))
    local_offset = client_offset - window_start
    items = window_items[local_offset : local_offset + safe_limit]

    out = dict(first_payload)
    out["items"] = items
    out["page"] = safe_page
    out["pageSize"] = safe_limit
    out["totalItemsCount"] = provider_total
    if not items and client_offset < provider_total:
        out["providerTotalItemsCount"] = provider_total
        out["totalItemsCount"] = client_offset
        out["paginationWarning"] = (
            "Finviz returned an empty provider page before its reported total; "
            "pagination was closed at the last reachable offset."
        )
    return out


def _clean_calendar_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Remove redundant/internal fields from calendar items."""
    if not isinstance(item, dict):
        return item
    cleaned = dict(item)
    cleaned.pop("boxoverData", None)
    return cleaned



