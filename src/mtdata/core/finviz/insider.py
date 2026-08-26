"""Finviz insider trades, analyst ratings, and peer adapters."""

import json
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

from mtdata.core.finviz.common import (
    _apply_finviz_pagination_contract,
    _attach_finviz_symbol_identity,
    _coerce_finviz_limit,
    _coerce_finviz_offset,
    _normalize_finviz_output_rows,
    _require_equity_symbol,
    _run_logged_tool,
    _validate_finviz_detail,
)
from mtdata.core.output_contract import (
    build_pagination_meta,
    normalize_output_verbosity_detail,
)
from mtdata.services.finviz import (
    get_insider_activity,
    get_stock_insider_trades,
    get_stock_peers,
    get_stock_ratings,
)
from mtdata.shared.schema import DetailLiteral


def _normalize_finviz_date_value(value: Any) -> Any:
    if value in (None, ""):
        return value
    if hasattr(value, "date") and callable(value.date):
        try:
            return value.date().isoformat()
        except Exception:
            pass
    text = str(value).strip()
    if not text:
        return text
    if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
        return text[:10]
    return text


_FINVIZ_PRICE_TARGET_ARROW_TOKENS = (
    "\u2192",
    "\u00e2\u0086\u0092",
    "\u00e2\u2020\u2019",
    "\u00d4\u00e5\u00c6",
)


def _clean_finviz_price_target_display(value: Any) -> str:
    display = str(value or "").strip()
    for token in _FINVIZ_PRICE_TARGET_ARROW_TOKENS:
        display = display.replace(token, " -> ")
    return re.sub(r"\s+", " ", display).strip()


def _finviz_price_target_fields(value: Any) -> Dict[str, Any]:
    if value in (None, ""):
        return {}
    display = _clean_finviz_price_target_display(value)
    if not display:
        return {}
    prices = [
        float(match.group(1).replace(",", ""))
        for match in re.finditer(
            r"[$]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            display,
        )
    ]
    if not prices:
        return {"price_target_display": display}
    previous = prices[0] if len(prices) > 1 else None
    latest = prices[-1]
    out: Dict[str, Any] = {
        "price_target_display": display,
        "price_target_new": latest,
    }
    if previous is not None:
        out["price_target_previous"] = previous
        if previous > 0:
            out["price_target_change_pct"] = round(
                ((latest - previous) / previous) * 100.0,
                2,
            )
    return out


def _normalize_finviz_rating_rows(rows: Any) -> List[Any]:
    normalized = _normalize_finviz_output_rows(rows)
    if not isinstance(normalized, list):
        return []
    for row in normalized:
        if not isinstance(row, dict):
            continue
        if "date" in row:
            row["date"] = _normalize_finviz_date_value(row.get("date"))
        if row.get("price") not in (None, ""):
            row["price"] = _clean_finviz_price_target_display(row.get("price"))
            row.update(_finviz_price_target_fields(row["price"]))
    return normalized


def _compact_finviz_rating_row(row: Any) -> Any:
    if not isinstance(row, dict):
        return row
    compact = dict(row)
    if compact.get("price_target_new") not in (None, ""):
        compact.pop("price", None)
        compact.pop("price_target_display", None)
    return compact


def _transaction_text(row: Dict[str, Any]) -> str:
    parts = [
        str(value)
        for key, value in row.items()
        if "transaction" in str(key).lower() or "trade" in str(key).lower()
    ]
    return " ".join(parts).lower()


def _insider_transaction_class(row: Dict[str, Any]) -> str:
    text = re.sub(r"\s+", " ", _transaction_text(row)).strip()
    if "proposed sale" in text or "planned sale" in text:
        return "proposed_sale"
    if "purchase" in text or re.search(r"\bbuy\b", text):
        return "purchase"
    if re.search(r"\b(?:sale|sell|sold)\b", text):
        return "executed_sale"
    if "exercise" in text:
        return "option_exercise"
    return "other"


def _insider_transaction_counts(rows: List[Any]) -> Dict[str, int]:
    classes = [
        _insider_transaction_class(row)
        for row in rows
        if isinstance(row, dict)
    ]
    return {
        "buy_transactions": classes.count("purchase"),
        "sell_transactions": classes.count("executed_sale"),
        "proposed_sale_transactions": classes.count("proposed_sale"),
        "option_exercise_transactions": classes.count("option_exercise"),
        "other_transactions": classes.count("other"),
    }


def _coerce_finviz_number(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(str(value).replace("$", "").replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return 0.0


def _summarize_insider_activity_tickers(
    rows: List[Any],
    *,
    transaction_class: str,
) -> List[Dict[str, Any]]:
    by_ticker: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if _insider_transaction_class(row) != transaction_class:
            continue
        symbol = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
        if not symbol:
            continue
        item = by_ticker.setdefault(
            symbol,
            {"symbol": symbol, "transactions": 0, "shares": 0.0, "value_usd": 0.0},
        )
        item["transactions"] += 1
        item["shares"] += abs(_coerce_finviz_number(row.get("shares")))
        item["value_usd"] += abs(_coerce_finviz_number(row.get("value_usd")))
    ranked = sorted(
        by_ticker.values(),
        key=lambda item: (item["value_usd"], item["shares"], item["transactions"]),
        reverse=True,
    )
    return [
        {
            "symbol": item["symbol"],
            "transactions": int(item["transactions"]),
            "shares": round(float(item["shares"]), 2),
            "value_usd": round(float(item["value_usd"]), 2),
        }
        for item in ranked[:5]
    ]


def _compact_finviz_insider_row(row: Dict[str, Any], *, include_symbol: bool) -> Dict[str, Any]:
    normalized = dict(row)
    if "price_per_share" not in normalized and normalized.get("cost") not in (None, ""):
        normalized["price_per_share"] = normalized["cost"]
    fields = (
        ("symbol",)
        if include_symbol
        else ()
    ) + (
        "owner",
        "transaction_date",
        "filed_at",
        "transaction",
        "price_per_share",
        "shares",
        "value_usd",
    )
    out = {field: normalized[field] for field in fields if field in normalized}
    out["transaction_class"] = _insider_transaction_class(normalized)
    return out


def _normalize_finviz_filing_timestamp(
    value: Any,
    *,
    transaction_date: Any = None,
) -> Optional[str]:
    """Normalize Finviz's yearless SEC filing time as US Eastern ISO-8601."""
    text = str(value or "").strip()
    if not text:
        return None
    reference_date = None
    try:
        reference_date = datetime.fromisoformat(str(transaction_date)).date()
    except (TypeError, ValueError):
        pass
    eastern = ZoneInfo("America/New_York")
    if reference_date is None:
        reference_date = datetime.now(timezone.utc).astimezone(eastern).date()
    try:
        parsed = datetime.strptime(
            f"{reference_date.year} {text}",
            "%Y %b %d %I:%M %p",
        )
    except ValueError:
        return None
    year = reference_date.year
    candidate = parsed.replace(tzinfo=eastern)
    if candidate.date() < reference_date - timedelta(days=180):
        candidate = candidate.replace(year=year + 1)
    elif candidate.date() > reference_date + timedelta(days=180):
        candidate = candidate.replace(year=year - 1)
    return candidate.isoformat()


def _normalize_finviz_insider_rows(rows: Any) -> List[Any]:
    normalized = _normalize_finviz_output_rows(rows)
    if not isinstance(normalized, list):
        return []
    for row in normalized:
        if not isinstance(row, dict):
            continue
        if row.get("cost") not in (None, ""):
            row.setdefault("price_per_share", row["cost"])
        if row.get("date") not in (None, ""):
            row.setdefault("transaction_date", row["date"])
        filed_at = _normalize_finviz_filing_timestamp(
            row.get("sec_form_4"),
            transaction_date=row.get("transaction_date"),
        )
        if filed_at is not None:
            row["filed_at"] = filed_at
    return normalized


def _finviz_insider_identity(row: Any) -> str:
    if not isinstance(row, dict):
        return repr(row)
    fields = (
        "symbol",
        "owner",
        "sec_form_4_link",
        "filed_at",
        "transaction_date",
        "transaction",
        "price_per_share",
        "shares",
        "value_usd",
        "shares_total",
    )
    identity = {
        key: row.get(key)
        for key in fields
        if row.get(key) not in (None, "")
    }
    if not identity:
        identity = row
    return json.dumps(identity, sort_keys=True, default=str, separators=(",", ":"))


def _dedupe_finviz_insider_rows(rows: List[Any]) -> tuple[List[Any], int]:
    unique: List[Any] = []
    seen: set[str] = set()
    for row in rows:
        identity = _finviz_insider_identity(row)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(row)
    return unique, len(rows) - len(unique)


def _prepare_finviz_insider_payload(
    result: Dict[str, Any],
    *,
    detail: str,
    operation: str,
    limit: int,
    page: int,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Any], int, bool]:
    """Shared validation, normalization, and full-detail setup for insider payloads."""
    error = _validate_finviz_detail(detail, operation=operation)
    if error is not None or not result.get("success"):
        return error or result, None, [], 0, False
    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    rows = result.get("insider_trades")
    if not isinstance(rows, list):
        return result, None, [], 0, False
    normalized_rows, duplicates_removed = _dedupe_finviz_insider_rows(
        _normalize_finviz_insider_rows(rows)
    )
    out = {key: value for key, value in result.items() if key != "insider_trades"}
    if duplicates_removed:
        out["duplicates_removed"] = int(out.get("duplicates_removed") or 0) + int(
            duplicates_removed
        )
    out["detail"] = detail_mode
    page_value = int(result.get("page") or page or 1)
    pages = result.get("pages")
    has_more = bool(
        result.get("has_more")
        or (pages not in (None, "") and page_value < int(pages))
    )
    if detail_mode == "full":
        out["items"] = normalized_rows
        out["row_key"] = "items"
        out["count"] = len(normalized_rows)
        _apply_finviz_pagination_contract(
            out,
            returned=len(normalized_rows),
            limit=limit,
            page=page_value,
            total=result.get("total"),
            total_lower_bound=result.get("total_lower_bound"),
            has_more=has_more,
        )
        return out, None, [], 0, False
    return None, out, normalized_rows, page_value, has_more


def _compact_finviz_insider_payload(
    result: Dict[str, Any],
    *,
    detail: str,
    limit: int = 20,
    page: int = 1,
) -> Dict[str, Any]:
    early, out, normalized_rows, page_value, has_more = _prepare_finviz_insider_payload(
        result,
        detail=detail,
        operation="finviz_insider",
        limit=limit,
        page=page,
    )
    if early is not None or out is None:
        return early or result
    compact_rows = [
        _compact_finviz_insider_row(row, include_symbol=False)
        for row in normalized_rows
    ]
    out["items"] = compact_rows
    out["row_key"] = "items"
    out["count"] = len(compact_rows)
    out["summary"] = _insider_transaction_counts(normalized_rows)
    out["hint"] = (
        "Single-symbol insider trades; use finviz_insider_activity for "
        "market-wide scans."
    )
    _apply_finviz_pagination_contract(
        out,
        returned=len(compact_rows),
        limit=limit,
        page=page_value,
        total=result.get("total"),
        total_lower_bound=result.get("total_lower_bound"),
        has_more=has_more,
    )
    return out


def _compact_finviz_insider_activity_payload(
    result: Dict[str, Any],
    *,
    detail: str,
    limit: int = 50,
    page: int = 1,
) -> Dict[str, Any]:
    early, out, normalized_rows, page_value, has_more = _prepare_finviz_insider_payload(
        result,
        detail=detail,
        operation="finviz_insider_activity",
        limit=limit,
        page=page,
    )
    if early is not None or out is None:
        return early or result

    compact_rows: List[Any] = []
    for row in normalized_rows:
        if not isinstance(row, dict):
            compact_rows.append(row)
            continue
        compact_rows.append(_compact_finviz_insider_row(row, include_symbol=True))

    out["items"] = compact_rows
    out["row_key"] = "items"
    out["count"] = len(compact_rows)
    out["summary"] = {
        **_insider_transaction_counts(normalized_rows),
        "top_executed_sales": _summarize_insider_activity_tickers(
            normalized_rows,
            transaction_class="executed_sale",
        ),
        "top_proposed_sales": _summarize_insider_activity_tickers(
            normalized_rows,
            transaction_class="proposed_sale",
        ),
        "top_purchases": _summarize_insider_activity_tickers(
            normalized_rows,
            transaction_class="purchase",
        ),
        "aggregation_note": (
            "Executed sales, proposed sales, and purchases are aggregated "
            "separately; filing lifecycle records are not combined."
        ),
    }
    out["hint"] = "Market-wide insider activity; use finviz_insider SYMBOL for one ticker."
    if str(result.get("option") or "").startswith("latest"):
        out["ordering"] = "filed_at_descending"
    _apply_finviz_pagination_contract(
        out,
        returned=len(compact_rows),
        limit=limit,
        page=page_value,
        total=result.get("total"),
        total_lower_bound=result.get("total_lower_bound"),
        has_more=has_more,
    )
    return out


def _compact_finviz_ratings_payload(
    result: Dict[str, Any],
    *,
    detail: str,
    limit: Optional[int],
    offset: Optional[int] = 0,
) -> Dict[str, Any]:
    error = _validate_finviz_detail(detail, operation="finviz_ratings")
    if error is not None or not result.get("success"):
        return error or result
    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    rows = result.get("ratings")
    if not isinstance(rows, list):
        return result
    out = dict(result)
    normalized_rows = _normalize_finviz_rating_rows(rows)
    limit_value = _coerce_finviz_limit(limit, default=len(normalized_rows))
    offset_value = _coerce_finviz_offset(offset)
    limited_rows = normalized_rows[offset_value : offset_value + limit_value]
    omitted = max(
        0,
        len(normalized_rows) - offset_value - len(limited_rows),
    )
    out["ratings"] = limited_rows
    out["row_key"] = "ratings"
    out["count"] = len(limited_rows)
    out["pagination"] = build_pagination_meta(
        total=len(normalized_rows),
        returned=len(limited_rows),
        offset=offset_value,
        limit=max(1, limit_value),
    )
    out["detail"] = detail_mode
    rating_fields = {
        key
        for row in limited_rows
        if isinstance(row, dict)
        for key, value in row.items()
        if value not in (None, "")
    }
    units: Dict[str, str] = {}
    if "price_target_change_pct" in rating_fields:
        units["price_target_change_pct"] = "percent (1.0 = 1%)"
    for field in ("price_target_previous", "price_target_new"):
        if field in rating_fields:
            units[field] = "USD_per_share"
    if units:
        out["units"] = units
    if {"price_target_previous", "price_target_new"} & rating_fields:
        out["currency"] = "USD"
        out["currency_basis"] = "US_equity_listing_currency"
    if detail_mode == "full":
        if omitted:
            out["show_all_hint"] = (
                f"Use --offset {offset_value + len(limited_rows)} for the next ratings page."
            )
        return out
    compact_rows = [_compact_finviz_rating_row(row) for row in limited_rows]
    out["ratings"] = compact_rows
    out["summary"] = {
        "latest": compact_rows[0] if compact_rows else None,
    }
    if omitted:
        out["show_all_hint"] = (
            f"Use --offset {offset_value + len(limited_rows)} for the next ratings page."
        )
    return out


def _compact_finviz_peers_payload(
    result: Dict[str, Any], *, detail: str, limit: Optional[int], offset: Optional[int] = 0
) -> Dict[str, Any]:
    error = _validate_finviz_detail(detail, operation="finviz_peers")
    if error is not None or not result.get("success"):
        return error or result
    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    peers = result.get("peers")
    if not isinstance(peers, list):
        return result
    out = dict(result)
    limit_value = _coerce_finviz_limit(limit, default=len(peers))
    out["row_key"] = "peers"
    offset_value = _coerce_finviz_offset(offset)
    limited_peers = peers[offset_value: offset_value + limit_value]
    omitted = max(0, len(peers) - offset_value - len(limited_peers))
    out["detail"] = detail_mode
    if detail_mode == "full":
        out["peers"] = limited_peers
        out["count"] = len(limited_peers)
        out["pagination"] = build_pagination_meta(
            total=len(peers),
            returned=len(limited_peers),
            offset=offset_value,
            limit=max(1, limit_value),
        )
        return out
    compact_peers = limited_peers
    out["peers"] = compact_peers
    out["count"] = len(compact_peers)
    out["pagination"] = build_pagination_meta(
        total=len(peers),
        returned=len(compact_peers),
        offset=offset_value,
        limit=max(1, limit_value),
    )
    if omitted:
        out["show_all_hint"] = (
            f"{omitted} more peers available; pass --offset {offset_value + len(compact_peers)}."
        )
    return out


def finviz_insider(
    symbol: str,
    limit: Annotated[int, Field(ge=1)] = 20,
    page: Annotated[int, Field(ge=1)] = 1,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Get insider trading activity for a US stock.
    
    Returns recent insider buys/sells with owner name, relationship,
    transaction type, shares, value, and date.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol
    limit : int
        Max trades per page (default 20)
    page : int
        Page number for pagination (default 1)
    detail : {"compact", "full"}
        "compact" normalizes each row in the requested page and adds aggregate
        buy/sell counts. "full" preserves all fields for the returned page.
    
    Returns
    -------
    dict
        List of insider trades
    """
    def _run() -> Dict[str, Any]:
        symbol_norm, error = _require_equity_symbol(
            symbol,
            tool_name="finviz_insider",
        )
        if error is not None:
            return error
        return _attach_finviz_symbol_identity(
            _compact_finviz_insider_payload(
                get_stock_insider_trades(symbol_norm, limit=limit, page=page),
                detail=detail,
                limit=limit,
                page=page,
            ),
            requested_symbol=symbol,
            finviz_ticker=symbol_norm,
        )

    return _run_logged_tool(
        "finviz_insider",
        {"symbol": symbol, "limit": limit, "page": page, "detail": detail},
        _run,
    )


def finviz_ratings(
    symbol: str,
    detail: Literal["compact", "full"] = "compact",
    limit: Annotated[int, Field(ge=1)] = 3,
    offset: Annotated[int, Field(ge=0)] = 0,
) -> Dict[str, Any]:
    """
    Get analyst ratings for a US stock.
    
    Returns ratings history with date, analyst firm, rating action,
    rating, and price target.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol
    detail : {"compact", "full"}
        "full" preserves the requested ratings fields. "compact" returns the
        latest limited rows plus a latest-rating summary.
    limit : int
        Maximum rating rows to return (default 3).
    offset : int
        Zero-based rating-row offset for deterministic pagination.
    Returns
    -------
    dict
        List of analyst ratings
    """
    def _run() -> Dict[str, Any]:
        symbol_norm, error = _require_equity_symbol(
            symbol,
            tool_name="finviz_ratings",
        )
        if error is not None:
            return error
        return _attach_finviz_symbol_identity(
            _compact_finviz_ratings_payload(
                get_stock_ratings(symbol_norm),
                detail=detail,
                limit=int(limit),
                offset=int(offset),
            ),
            requested_symbol=symbol,
            finviz_ticker=symbol_norm,
        )

    return _run_logged_tool(
        "finviz_ratings",
        {
            "symbol": symbol,
            "detail": detail,
            "limit": limit,
            "offset": offset,
        },
        _run,
    )


def finviz_peers(
    symbol: str,
    detail: DetailLiteral = "compact",  # type: ignore
    limit: Annotated[int, Field(ge=1)] = 5,
    offset: Annotated[int, Field(ge=0)] = 0,
) -> Dict[str, Any]:
    """
    Get peer companies for a US stock.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol
    detail : {"compact", "full"}
        "full" preserves the complete peer list. "compact" returns up to
        five peers plus peer counts.
    
    Returns
    -------
    dict
        List of peer ticker symbols
    """
    def _run() -> Dict[str, Any]:
        symbol_norm, error = _require_equity_symbol(
            symbol,
            tool_name="finviz_peers",
        )
        if error is not None:
            return error
        return _attach_finviz_symbol_identity(
            _compact_finviz_peers_payload(
                get_stock_peers(symbol_norm),
                detail=detail,
                limit=limit,
                offset=offset,
            ),
            requested_symbol=symbol,
            finviz_ticker=symbol_norm,
        )

    return _run_logged_tool(
        "finviz_peers",
        {"symbol": symbol, "detail": detail, "limit": limit, "offset": offset},
        _run,
    )


def finviz_insider_activity(
    option: Literal[
        "latest",
        "latest buys",
        "latest sales",
        "top week",
        "top week buys",
        "top week sales",
        "top owner trade",
        "top owner buys",
        "top owner sales",
    ] = "latest",
    limit: Annotated[int, Field(ge=1)] = 50,
    page: Annotated[int, Field(ge=1)] = 1,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Get general insider trading activity across the market.
    
    Parameters
    ----------
    option : str
        Activity type:
        - "latest", "latest buys", "latest sales": newest SEC filings first
        - "top week", "top week buys", "top week sales"
        - "top owner trade", "top owner buys", "top owner sales"
    limit : int
        Max items per page (default 50)
    page : int
        Page number for pagination (default 1)
    detail : {"compact", "full"}
        Response detail level. Compact normalizes every row in the requested
        page and adds separate executed-sale, proposed-sale, and purchase
        summaries; full keeps all fields including SEC links.
    
    Returns
    -------
    dict
        List of insider trades with ticker, owner, transaction details
    """
    def _run() -> Dict[str, Any]:
        detail_error = _validate_finviz_detail(detail, operation="finviz_insider_activity")
        if detail_error is not None:
            return detail_error
        return _compact_finviz_insider_activity_payload(
            get_insider_activity(option=option, limit=limit, page=page),
            detail=detail,
            limit=limit,
            page=page,
        )

    return _run_logged_tool(
        "finviz_insider_activity",
        {"option": option, "limit": limit, "page": page, "detail": detail},
        _run,
    )
