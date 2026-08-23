"""Compact chart-geometry DTOs for dedicated Web API routes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..utils.coercion import coerce_finite_float as _as_float
from .output_contract import apply_output_verbosity
from .tool_calling import call_tool_sync_structured
from .web_api_handlers import _http_error, _raise_tool_error


def compact_confluence_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    levels: List[Dict[str, Any]] = []
    raw_levels = payload.get("levels")
    if isinstance(raw_levels, list):
        for row in raw_levels[:8]:
            if not isinstance(row, dict):
                continue
            price = _as_float(row.get("price") if "price" in row else row.get("value"))
            if price is None:
                continue
            item: Dict[str, Any] = {"price": price}
            level_type = str(row.get("type") or "").strip().lower()
            if level_type:
                item["type"] = level_type
            score = _as_float(row.get("score"))
            if score is not None:
                item["score"] = score
            range_payload = row.get("range")
            if isinstance(range_payload, dict):
                compact_range = {
                    name: _as_float(range_payload.get(name))
                    for name in ("low", "high")
                    if _as_float(range_payload.get(name)) is not None
                }
                if compact_range:
                    item["range"] = compact_range
            levels.append(item)
    return {
        "success": True,
        "symbol": payload.get("symbol"),
        "pivot_timeframe": payload.get("pivot_timeframe"),
        "sr_timeframe": payload.get("sr_timeframe"),
        "levels": levels,
    }


def compact_volume_profile_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    def _level_price(value: Any) -> Optional[float]:
        if isinstance(value, dict):
            return _as_float(value.get("price") or value.get("value"))
        return _as_float(value)

    poc = _level_price(payload.get("poc"))
    vah = _level_price(payload.get("vah"))
    val = _level_price(payload.get("val"))
    if vah is None or val is None:
        value_area = payload.get("value_area")
        if isinstance(value_area, dict):
            vah = vah if vah is not None else _as_float(value_area.get("high"))
            val = val if val is not None else _as_float(value_area.get("low"))
    out: Dict[str, Any] = {
        "success": True,
        "symbol": payload.get("symbol"),
        "timeframe": payload.get("timeframe"),
    }
    if poc is not None:
        out["poc"] = poc
    if vah is not None:
        out["vah"] = vah
    if val is not None:
        out["val"] = val
    return out


def _compact_exposure_rows(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in items[:20]:
        if not isinstance(item, dict):
            continue
        price = _as_float(
            item.get("price_open")
            or item.get("price_current")
            or item.get("price")
        )
        row: Dict[str, Any] = {}
        ticket = item.get("ticket") or item.get("order")
        if ticket not in (None, ""):
            row["ticket"] = ticket
        side = item.get("type") or item.get("side") or item.get("order_type")
        if side not in (None, ""):
            row["type"] = side
        volume = _as_float(item.get("volume"))
        if volume is not None:
            row["volume"] = volume
        if price is not None:
            row["price"] = price
        sl = _as_float(item.get("sl") or item.get("stop_loss"))
        tp = _as_float(item.get("tp") or item.get("take_profit"))
        if sl is not None:
            row["sl"] = sl
        if tp is not None:
            row["tp"] = tp
        if row:
            rows.append(row)
    return rows


def compact_exposure_payload(
    *,
    symbol: str,
    positions: Any,
    pending: Any,
) -> Dict[str, Any]:
    return {
        "success": True,
        "symbol": symbol,
        "positions": _compact_exposure_rows(positions),
        "pending": _compact_exposure_rows(pending),
    }


def get_confluence_response(
    *,
    symbol: str,
    pivot_timeframe: str,
    sr_timeframe: str,
    confluence_tool: Any,
) -> Dict[str, Any]:
    result = call_tool_sync_structured(
        confluence_tool,
        symbol=symbol,
        pivot_timeframe=pivot_timeframe,
        sr_timeframe=sr_timeframe,
        detail="compact",
    )
    _raise_tool_error(
        result,
        operation="get_confluence",
        default_code="confluence_failed",
        invalid_message="Unexpected geometry payload",
    )
    payload = compact_confluence_payload(result)
    if not payload["levels"]:
        raise _http_error(
            404,
            "No confluence levels returned",
            code="confluence_levels_missing",
            operation="get_confluence",
        )
    return apply_output_verbosity(payload, detail="compact", tool_name="confluence_levels")


def get_volume_profile_response(
    *,
    symbol: str,
    timeframe: str,
    volume_profile_tool: Any,
    start: Any = None,
    end: Any = None,
    lookback: Any = None,
    source: str = "auto",
    price_source: str = "mid",
    volume_source: str = "auto",
    bucket_size: Any = None,
    bucket_points: Any = None,
    bucket_count: Any = None,
    max_buckets: int = 120,
    value_area_pct: float = 70.0,
    reference_price: Any = None,
    max_tick_window_days: int = 1,
    max_ticks: int = 50_000,
    max_m1_bars: int = 20_000,
    detail: str = "compact",
) -> Dict[str, Any]:
    result = call_tool_sync_structured(
        volume_profile_tool,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        lookback=lookback,
        source=source,
        price_source=price_source,
        volume_source=volume_source,
        bucket_size=bucket_size,
        bucket_points=bucket_points,
        bucket_count=bucket_count,
        max_buckets=max_buckets,
        value_area_pct=value_area_pct,
        reference_price=reference_price,
        max_tick_window_days=max_tick_window_days,
        max_ticks=max_ticks,
        max_m1_bars=max_m1_bars,
        detail=detail,
    )
    _raise_tool_error(
        result,
        operation="get_volume_profile",
        default_code="volume_profile_failed",
        invalid_message="Unexpected geometry payload",
    )
    payload = compact_volume_profile_payload(result)
    if payload.get("poc") is None and payload.get("vah") is None and payload.get("val") is None:
        raise _http_error(
            404,
            "No volume-profile levels returned",
            code="volume_profile_levels_missing",
            operation="get_volume_profile",
        )
    return apply_output_verbosity(payload, detail="compact", tool_name="volume_profile_levels")


def get_exposure_response(
    *,
    symbol: str,
    open_tool: Any,
    pending_tool: Any,
) -> Dict[str, Any]:
    from .trading.requests import TradeGetOpenRequest, TradeGetPendingRequest

    positions = call_tool_sync_structured(
        open_tool,
        request=TradeGetOpenRequest(symbol=symbol, detail="compact"),
    )
    pending = call_tool_sync_structured(
        pending_tool,
        request=TradeGetPendingRequest(symbol=symbol, detail="compact"),
    )
    if isinstance(positions, dict) and positions.get("error"):
        _raise_tool_error(positions, operation="get_exposure", default_code="exposure_failed")
    if isinstance(pending, dict) and pending.get("error"):
        _raise_tool_error(pending, operation="get_exposure", default_code="exposure_failed")
    payload = compact_exposure_payload(symbol=symbol, positions=positions, pending=pending)
    return apply_output_verbosity(payload, detail="compact", tool_name="trade_get_open")
