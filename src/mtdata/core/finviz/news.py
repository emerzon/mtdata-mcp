"""Finviz news adapters and news payload normalizers."""

import re
from datetime import datetime, timedelta, timezone
from datetime import time as datetime_time
from typing import (
    Annotated,
    Any,
    Dict,
    Literal,
    Optional,
)
from urllib.parse import (
    urljoin,
    urlparse,
)

from pydantic import Field

from mtdata.core.finviz.common import (
    _FINVIZ_CALENDAR_LOCAL_TZ,
    _apply_finviz_pagination_contract,
    _attach_finviz_symbol_identity,
    _require_equity_symbol,
    _run_logged_tool,
    _validate_positive_finviz_limit,
)
from mtdata.core.output_contract import normalize_output_detail
from mtdata.services.finviz import (
    get_general_news,
    get_stock_news,
)
from mtdata.services.finviz.dates import (
    FINVIZ_CALENDAR_TIMEZONE,
    parse_finviz_publication_date,
)
from mtdata.services.news_text import normalize_news_text
from mtdata.shared.schema import DetailLiteral


def _clean_finviz_text_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_news_text(value)
    return value


def _normalize_finviz_published_at(value: Any, *, now: Optional[datetime] = None) -> Any:
    if isinstance(value, datetime):
        dt = (
            value
            if value.tzinfo is not None
            else value.replace(tzinfo=_FINVIZ_CALENDAR_LOCAL_TZ)
        )
        return dt.astimezone(timezone.utc).isoformat()
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text

    iso_text = text
    try:
        dt = datetime.fromisoformat(iso_text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_FINVIZ_CALENDAR_LOCAL_TZ)
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            return (
                dt.replace(tzinfo=_FINVIZ_CALENDAR_LOCAL_TZ)
                .astimezone(timezone.utc)
                .isoformat()
            )
        except ValueError:
            continue

    for fmt in ("%I:%M%p", "%I:%M %p"):
        try:
            parsed_time = datetime.strptime(text.upper(), fmt).time()
        except ValueError:
            continue
        reference = now or datetime.now(timezone.utc)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        reference = reference.astimezone(timezone.utc)
        reference_local = reference.astimezone(_FINVIZ_CALENDAR_LOCAL_TZ)
        dt = datetime.combine(
            reference_local.date(),
            datetime_time(
                parsed_time.hour,
                parsed_time.minute,
                parsed_time.second,
                tzinfo=_FINVIZ_CALENDAR_LOCAL_TZ,
            ),
        )
        if dt.astimezone(timezone.utc) > reference + timedelta(hours=1):
            dt -= timedelta(days=1)
        return dt.astimezone(timezone.utc).isoformat()

    return text


def _finviz_relative_time_from_text(
    value: Any,
    *,
    now: Optional[datetime] = None,
) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if " ago" in text.lower():
        return text
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    delta = reference.astimezone(timezone.utc) - dt.astimezone(timezone.utc)
    signed_seconds = int(delta.total_seconds())
    if signed_seconds < 0:
        future_seconds = abs(signed_seconds)
        future_minutes = max(1, future_seconds // 60)
        if future_minutes < 90:
            return f"in {future_minutes} minutes"
        future_hours = future_minutes // 60
        return f"in {future_hours} hours"
    seconds = signed_seconds
    if seconds < 90:
        return "just now"
    minutes = seconds // 60
    if minutes < 90:
        return f"{minutes} minutes ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} hours ago"
    days = hours // 24
    if days < 14:
        return f"{days} days ago"
    weeks = days // 7
    return f"{weeks} weeks ago"


def _normalize_finviz_news_item(
    item: Any,
    *,
    kind: str = "headline",
    now: Optional[datetime] = None,
) -> Any:
    if not isinstance(item, dict):
        return item

    out: Dict[str, Any] = {}
    raw_published_at: Any = None
    for source_key, target_key in (
        ("Title", "title"),
        ("Source", "source"),
        ("Date", "published_at"),
        ("Link", "url"),
        ("title", "title"),
        ("source", "source"),
        ("published_at", "published_at"),
        ("relative_time", "relative_time"),
        ("kind", "kind"),
        ("url", "url"),
    ):
        if source_key not in item:
            continue
        value = _clean_finviz_text_value(item.get(source_key))
        if value in (None, ""):
            continue
        if target_key == "published_at":
            raw_published_at = value
            publication_date = parse_finviz_publication_date(value, now=now)
            if publication_date is not None:
                out["publication_date"] = publication_date.isoformat()
                out["timestamp_precision"] = "date"
                out["source_timezone"] = FINVIZ_CALENDAR_TIMEZONE
                continue
            value = _normalize_finviz_published_at(value, now=now)
        if target_key == "url" and isinstance(value, str):
            resolved = urljoin("https://finviz.com/", value)
            parsed_url = urlparse(resolved)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                out["url_status"] = "invalid"
                continue
            value = resolved
        out[target_key] = value
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    raw_was_naive = isinstance(raw_published_at, datetime) and (
        raw_published_at.tzinfo is None
    )
    if isinstance(raw_published_at, str):
        try:
            raw_dt = datetime.fromisoformat(raw_published_at)
            raw_was_naive = raw_dt.tzinfo is None
        except ValueError:
            raw_was_naive = False
    published_text = out.get("published_at")
    try:
        published_dt = datetime.fromisoformat(
            str(published_text)
        )
    except (TypeError, ValueError):
        published_dt = None
    if published_dt is not None:
        if published_dt.tzinfo is None:
            published_dt = published_dt.replace(tzinfo=timezone.utc)
        future_by = published_dt.astimezone(timezone.utc) - reference.astimezone(
            timezone.utc
        )
        if future_by > timedelta(minutes=5):
            corrected = published_dt - timedelta(hours=12)
            if raw_was_naive and corrected <= reference + timedelta(minutes=5):
                out["original_published_at"] = published_text
                out["published_at"] = corrected.astimezone(timezone.utc).isoformat()
                out["timestamp_quality"] = "provider_meridiem_corrected"
                out["timestamp_correction"] = "subtracted_12_hours_from_future_naive_provider_time"
            else:
                out["timestamp_quality"] = "future_provider_time"
                out["timestamp_anomaly"] = True
                out["future_by_seconds"] = int(future_by.total_seconds())
    if "relative_time" not in out:
        relative_time = (
            _finviz_relative_time_from_text(out.get("published_at"), now=reference)
            or _finviz_relative_time_from_text(raw_published_at, now=reference)
        )
        if relative_time:
            out["relative_time"] = relative_time
    out.setdefault("kind", kind)
    out["content_type"] = "blog" if str(kind).lower() == "blog" else "news"
    return out


def _finviz_news_item_has_symbol_evidence(item: Any, symbol: str) -> bool:
    if not isinstance(item, dict):
        return False
    symbol_text = str(symbol or "").strip()
    title = str(item.get("title") or "")
    if not symbol_text or not title:
        return False
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(symbol_text)}(?![A-Za-z0-9])",
            title,
            flags=re.IGNORECASE,
        )
    )


def _normalize_finviz_news_payload(
    result: Dict[str, Any],
    *,
    detail: DetailLiteral = "compact",  # type: ignore
    kind: str = "headline",
    limit: int = 20,
    page: int = 1,
) -> Dict[str, Any]:
    out = dict(result)
    out.pop("tool_scope", None)
    out.pop("preferred_tool", None)
    out.pop("output_shape", None)
    out.pop("timezone", None)
    detail_mode = normalize_output_detail(detail, default="compact")
    out["detail"] = detail_mode
    out["provider"] = "finviz"
    out["delivery"] = "aggregated_web_feed"
    out["is_realtime"] = False
    out["freshness_note"] = (
        "Finviz aggregates third-party headlines and does not guarantee real-time delivery."
    )

    news_rows = result.get("news")
    items_rows = result.get("items")
    if not isinstance(news_rows, list) and not isinstance(items_rows, list):
        return out

    source_rows = news_rows if isinstance(news_rows, list) else items_rows
    normalized_at = datetime.now(timezone.utc)
    normalized_items = [
        _normalize_finviz_news_item(item, kind=kind, now=normalized_at)
        for item in source_rows
    ]
    out.pop("news", None)
    pages = result.get("pages")
    page_value = int(result.get("page") or page or 1)
    has_more = bool(
        result.get("has_more")
        or (pages not in (None, "") and page_value < int(pages))
    )
    _apply_finviz_pagination_contract(
        out,
        returned=len(normalized_items),
        limit=limit,
        page=page_value,
        total=result.get("total"),
        total_lower_bound=result.get("total_lower_bound"),
        has_more=has_more,
    )
    out.pop("omitted_item_count", None)
    if detail_mode == "summary":
        out.pop("items", None)
        out["count"] = int(out.get("count") or len(normalized_items))
        return out
    if detail_mode == "compact":
        compact_fields = {
            "title",
            "source",
            "published_at",
            "publication_date",
            "timestamp_precision",
            "source_timezone",
            "relative_time",
            "url",
            "kind",
            "content_type",
            "timestamp_quality",
            "timestamp_anomaly",
            "timestamp_correction",
            "original_published_at",
            "future_by_seconds",
        }
        out["items"] = [
            {
                key: value
                for key, value in item.items()
                if key in compact_fields and value not in (None, "")
            }
            if isinstance(item, dict)
            else item
            for item in normalized_items
        ]
    else:
        out["items"] = normalized_items
    out["row_key"] = "items"
    return out


def finviz_news(
    symbol: str,
    limit: Annotated[int, Field(ge=1)] = 20,
    page: Annotated[int, Field(ge=1)] = 1,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Raw Finviz per-ticker news provider endpoint.

    Raw Finviz per-ticker news provider endpoint.

    Internal adapter. Public callers use ``news(view='ticker', source='finviz')``.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., NVDA, META).
    limit : int
        Max news items per page (default 20)
    page : int
        Page number for pagination (default 1)
    
    Returns
    -------
    dict
        Stock-specific normalized `items` rows with `title`, `source`,
        `published_at`, and `url` fields.
    """
    fields = {"symbol": symbol, "limit": limit, "page": page, "detail": detail}

    def _run() -> Dict[str, Any]:
        limit_error = _validate_positive_finviz_limit(
            limit,
            operation="finviz_news",
        )
        if limit_error is not None:
            return limit_error
        symbol_norm, error = _require_equity_symbol(
            symbol,
            tool_name="finviz_news",
        )
        if error is not None:
            return error
        payload = _normalize_finviz_news_payload(
            get_stock_news(symbol_norm, limit=limit, page=page),
            detail=detail,
            kind="provider_associated",
            limit=limit,
            page=page,
        )
        payload["provider_context_symbol"] = symbol_norm
        payload["relevance_basis"] = "finviz_ticker_page_association_unverified"
        payload["relevance_note"] = (
            "Rows may cover peers, suppliers, industries, or macro topics. Use news "
            "for symbol-evidence relevance filtering."
        )
        for item in payload.get("items") or []:
            if not isinstance(item, dict):
                continue
            if _finviz_news_item_has_symbol_evidence(item, symbol_norm):
                item["kind"] = "direct_symbol"
                item["relevance_basis"] = "headline_symbol_token"
            else:
                item["relevance_basis"] = (
                    "finviz_ticker_page_association_unverified"
                )
        return _attach_finviz_symbol_identity(
            payload,
            requested_symbol=symbol,
            finviz_ticker=symbol_norm,
        )

    return _run_logged_tool("finviz_news", fields, _run)


def finviz_market_news(
    news_type: Literal["news", "blogs"] = "news",
    limit: Annotated[int, Field(ge=1)] = 20,
    page: Annotated[int, Field(ge=1)] = 1,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """
    Raw Finviz general market news/blog provider endpoint.

    Internal adapter. Public callers use ``news(view='market', source='finviz')``.
    
    Parameters
    ----------
    news_type : str
        Type: "news" for headlines, "blogs" for blog posts
    limit : int
        Max items per page (default 20)
    page : int
        Page number for pagination (default 1)
    
    Returns
    -------
    dict
        List of news/blog items. Top-level metadata marks this as a raw Finviz
        provider endpoint and points traders to `news` as the preferred unified
        tool.
    """
    return _run_logged_tool(
        "finviz_market_news",
        {"news_type": news_type, "limit": limit, "page": page, "detail": detail},
        lambda: _normalize_finviz_news_payload(
            get_general_news(news_type=news_type, limit=limit, page=page),
            detail=detail,
            kind="blog" if str(news_type).lower().strip() == "blogs" else "headline",
            limit=limit,
            page=page,
        ),
    )
