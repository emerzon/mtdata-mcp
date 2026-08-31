"""Compact radar and session-strip routes for the Web UI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._mcp_tools import shape_public_tool_output
from .radar import (
    RADAR_MAX_SYMBOLS,
    MarketRadarRequest,
    parse_radar_symbols,
    run_market_radar,
)
from .tool_calling import call_tool_sync_structured
from .web_api_handlers import _http_error, _raise_tool_error


def get_radar_response(
    *,
    symbols: Optional[str],
    timeframe: str,
    rank_by: str,
    limit: int,
    compose_impl: Any = None,
) -> Dict[str, Any]:
    requested = parse_radar_symbols(symbols)
    if symbols is not None and str(symbols).strip() and not requested:
        raise _http_error(
            400,
            "symbols must contain at least one broker name.",
            code="radar_invalid_symbols",
            operation="get_radar",
        )
    runner = compose_impl or run_market_radar
    try:
        request = MarketRadarRequest(
            symbols=",".join(requested) if requested else None,
            timeframe=timeframe,
            rank_by=rank_by,  # type: ignore[arg-type]
            limit=min(int(limit), RADAR_MAX_SYMBOLS),
            detail="compact",
        )
    except Exception as exc:
        raise _http_error(
            400,
            str(exc),
            code="radar_invalid_request",
            operation="get_radar",
        ) from exc
    result = runner(request)
    if isinstance(result, dict) and result.get("error") and not result.get("rows"):
        _raise_tool_error(
            result,
            operation="get_radar",
            default_code="radar_empty",
            default_status=404,
        )
    return shape_public_tool_output(
        result,
        detail="compact",
        tool_name="market_radar",
    )


def _compact_news_headlines(payload: Any, *, limit: int = 5) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    headlines: List[Dict[str, Any]] = []
    for key in ("general_news", "related_news", "impact_news", "upcoming_events"):
        bucket = payload.get(key)
        if not isinstance(bucket, list):
            continue
        for item in bucket:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            row: Dict[str, Any] = {"title": title, "bucket": key}
            if item.get("time") not in (None, ""):
                row["time"] = item["time"]
            if item.get("source") not in (None, ""):
                row["source"] = item["source"]
            headlines.append(row)
            if len(headlines) >= limit:
                return headlines
    return headlines


def compact_session_strip(
    *,
    account: Any,
    news: Any,
    exposure: Any,
    market_status: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"success": True}
    failed: List[str] = []
    if isinstance(account, dict) and not account.get("error"):
        compact_account = {
            key: account[key]
            for key in ("login", "server", "company", "equity", "balance", "currency", "is_demo")
            if account.get(key) not in (None, "")
        }
        if compact_account:
            payload["account"] = compact_account
    else:
        failed.append("account")
        if isinstance(account, dict) and account.get("error"):
            payload["account_error"] = account.get("error")

    headlines = _compact_news_headlines(news)
    if headlines:
        payload["news"] = headlines
    elif isinstance(news, dict) and news.get("error"):
        failed.append("news")

    count = None
    if isinstance(exposure, dict) and not exposure.get("error"):
        if isinstance(exposure.get("count"), int):
            count = exposure["count"]
        elif isinstance(exposure.get("items"), list):
            count = len(exposure["items"])
    if count is not None:
        payload["exposure_count"] = count
    elif isinstance(exposure, dict) and exposure.get("error"):
        failed.append("exposure")

    if isinstance(market_status, dict) and not market_status.get("error"):
        compact_status = {
            key: market_status[key]
            for key in ("status", "is_tradable", "can_open_new_positions", "reason")
            if market_status.get(key) not in (None, "")
        }
        if compact_status:
            payload["market_status"] = compact_status
    elif isinstance(market_status, dict) and market_status.get("error"):
        failed.append("market_status")

    if failed:
        payload["partial_failure"] = True
        payload["failed_sections"] = failed
    return payload


def get_session_strip_response(
    *,
    symbol: Optional[str] = None,
    account_tool: Any = None,
    news_tool: Any = None,
    open_tool: Any = None,
    status_tool: Any = None,
) -> Dict[str, Any]:
    from .market_status import market_status as default_status
    from .news import news as default_news
    from .trading.account import trade_account_info
    from .trading.positions import trade_get_open
    from .trading.requests import TradeGetOpenRequest

    account = None
    news_payload = None
    exposure = None
    status = None
    try:
        account = call_tool_sync_structured(account_tool or trade_account_info, detail="compact")
    except Exception as exc:
        account = {"error": str(exc)}
    try:
        news_payload = call_tool_sync_structured(
            news_tool or default_news,
            **({"symbol": symbol, "limit": 5, "detail": "compact"} if symbol else {"limit": 5, "detail": "compact"}),
        )
    except Exception as exc:
        news_payload = {"error": str(exc)}
    try:
        exposure = call_tool_sync_structured(
            open_tool or trade_get_open,
            request=TradeGetOpenRequest(detail="compact", limit=50),
        )
    except Exception as exc:
        exposure = {"error": str(exc)}
    if symbol:
        try:
            status = call_tool_sync_structured(
                status_tool or default_status,
                symbol=symbol,
                detail="compact",
            )
        except Exception as exc:
            status = {"error": str(exc)}
    payload = compact_session_strip(
        account=account,
        news=news_payload,
        exposure=exposure,
        market_status=status,
    )
    # This route is an aggregate DTO, not a trade_account_info response. Applying
    # that tool's compact allowlist erases the independently composed sections.
    return payload
