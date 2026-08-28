"""Candle-close boundary payloads for wait events."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from mtdata.core.data.requests import WaitEventRequest
from mtdata.core.data.wait_events.compile import _resolved_wait_result_symbol
from mtdata.core.data.wait_events.ticks import (
    _coerce_rows,
    _finite_number,
    _format_utc_iso,
    _normalize_optional_utc_datetime,
    _row_float,
    _row_value,
)
from mtdata.shared.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from mtdata.utils.mt5 import _normalize_times_in_struct, _to_server_query_dt

_BOUNDARY_CANDLE_LOOKBACK_BARS = 3

def _boundary_event_payload(
    boundary: Dict[str, Any],
    *,
    request: Optional[WaitEventRequest] = None,
    watch_for_payload: Optional[List[Dict[str, Any]]] = None,
    gateway: Any = None,
) -> Dict[str, Any]:
    payload = {
        "type": "candle_close",
        "timeframe": boundary["timeframe"],
        "buffer_seconds": boundary["buffer_seconds"],
        "next_candle_close_utc": boundary["preview"]["next_candle_close_utc"],
        "next_candle_close_server": boundary["preview"]["next_candle_close_server"],
        "server_timezone": boundary["preview"]["server_timezone"],
    }
    if request is not None:
        if request.symbols is not None:
            closed_candles, candle_failures = _boundary_closed_candle_collection(
                boundary=boundary,
                symbols=request.symbols,
                gateway=gateway,
            )
            payload["closed_candles"] = closed_candles
            if candle_failures:
                payload["candle_failures"] = candle_failures
        else:
            closed_candle = _boundary_closed_candle_payload(
                boundary=boundary,
                request=request,
                watch_for_payload=watch_for_payload or [],
                gateway=gateway,
            )
            if closed_candle is not None:
                payload["closed_candle"] = closed_candle
    return payload

def _boundary_closed_candle_collection(
    *,
    boundary: Dict[str, Any],
    symbols: List[str],
    gateway: Any,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    closed_candles: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    timeframe = str(boundary.get("timeframe") or "").upper().strip()
    for symbol in symbols:
        closed_candle = _boundary_closed_candle_for_symbol(
            boundary=boundary,
            symbol=symbol,
            gateway=gateway,
        )
        if closed_candle is not None:
            closed_candles.append(closed_candle)
            continue
        failures.append(
            {
                "symbol": str(symbol).upper().strip(),
                "timeframe": timeframe,
                "error_code": "wait_event_closed_candle_unavailable",
                "error": (
                    f"No completed {timeframe} candle was available for "
                    f"{str(symbol).upper().strip()} at the boundary."
                ),
            }
        )
    return closed_candles, failures

def _boundary_closed_candle_payload(
    *,
    boundary: Dict[str, Any],
    request: WaitEventRequest,
    watch_for_payload: List[Dict[str, Any]],
    gateway: Any,
) -> Optional[Dict[str, Any]]:
    symbol = _resolved_wait_result_symbol(
        request,
        watch_for_payload=watch_for_payload,
    )
    if not symbol:
        return None
    return _boundary_closed_candle_for_symbol(
        boundary=boundary,
        symbol=symbol,
        gateway=gateway,
    )

def _boundary_closed_candle_for_symbol(
    *,
    boundary: Dict[str, Any],
    symbol: str,
    gateway: Any,
) -> Optional[Dict[str, Any]]:
    timeframe = str(boundary.get("timeframe") or "").upper().strip()
    if timeframe not in TIMEFRAME_MAP:
        return None
    close_at_utc = _normalize_optional_utc_datetime(boundary.get("boundary_at_utc"))
    if close_at_utc is None:
        close_at_utc = _normalize_optional_utc_datetime(
            boundary.get("preview", {}).get("next_candle_close_utc")
        )
    if close_at_utc is None:
        return None
    seconds_per_bar = float(TIMEFRAME_SECONDS.get(timeframe) or 0.0)
    if seconds_per_bar <= 0.0:
        return None
    rows = _fetch_boundary_candle_rows(
        gateway=gateway,
        symbol=symbol,
        timeframe=timeframe,
        close_at_utc=close_at_utc,
        seconds_per_bar=seconds_per_bar,
    )
    if not rows:
        return None
    row = _select_boundary_closed_candle_row(
        rows,
        close_at_utc=close_at_utc,
        seconds_per_bar=seconds_per_bar,
    )
    if row is None:
        return None
    point = None
    try:
        info = gateway.symbol_info(symbol)
        point_value = float(getattr(info, "point", 0.0) or 0.0)
        if math.isfinite(point_value) and point_value > 0.0:
            point = point_value
    except Exception:
        point = None
    return _format_boundary_closed_candle(
        row,
        symbol=symbol,
        timeframe=timeframe,
        close_at_utc=close_at_utc,
        seconds_per_bar=seconds_per_bar,
        price_point=point,
    )

def _fetch_boundary_candle_rows(
    *,
    gateway: Any,
    symbol: str,
    timeframe: str,
    close_at_utc: datetime,
    seconds_per_bar: float,
) -> List[Any]:
    if gateway is None:
        return []
    mt5_timeframe = TIMEFRAME_MAP.get(timeframe)
    if mt5_timeframe is None:
        return []
    if hasattr(gateway, "symbol_select"):
        try:
            gateway.symbol_select(symbol, True)
        except Exception:
            pass
    query_to_utc = close_at_utc - timedelta(microseconds=1)
    if hasattr(gateway, "copy_rates_from"):
        try:
            rows = gateway.copy_rates_from(
                symbol,
                mt5_timeframe,
                _to_server_query_dt(query_to_utc),
                _BOUNDARY_CANDLE_LOOKBACK_BARS,
            )
            coerced = _coerce_rows(_normalize_times_in_struct(rows))
            if coerced:
                return coerced
        except Exception:
            pass
    if hasattr(gateway, "copy_rates_range"):
        try:
            from_utc = close_at_utc - timedelta(seconds=seconds_per_bar * 2.0)
            rows = gateway.copy_rates_range(
                symbol,
                mt5_timeframe,
                _to_server_query_dt(from_utc),
                _to_server_query_dt(query_to_utc),
            )
            return _coerce_rows(_normalize_times_in_struct(rows))
        except Exception:
            return []
    return []

def _select_boundary_closed_candle_row(
    rows: List[Any],
    *,
    close_at_utc: datetime,
    seconds_per_bar: float,
) -> Any:
    if not rows:
        return None
    expected_open_epoch = close_at_utc.timestamp() - float(seconds_per_bar)
    close_epoch = close_at_utc.timestamp()
    candidates: List[tuple[float, Any]] = []
    for row in rows:
        open_epoch = _rate_open_epoch(row)
        if open_epoch is None:
            continue
        if open_epoch <= close_epoch - 1e-6:
            candidates.append((abs(open_epoch - expected_open_epoch), row))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        tolerance = max(1.0, float(seconds_per_bar) * 0.5)
        if candidates[0][0] <= tolerance:
            return candidates[0][1]
    return None

def _format_boundary_closed_candle(
    row: Any,
    *,
    symbol: str,
    timeframe: str,
    close_at_utc: datetime,
    seconds_per_bar: float,
    price_point: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    open_price = _row_float(row, "open")
    high_price = _row_float(row, "high")
    low_price = _row_float(row, "low")
    close_price = _row_float(row, "close")
    if None in (open_price, high_price, low_price, close_price):
        return None

    open_at_utc = _rate_open_datetime(row)
    if open_at_utc is None:
        open_at_utc = close_at_utc - timedelta(seconds=seconds_per_bar)
    tick_volume = _row_float(row, "tick_volume")
    real_volume = _row_float(row, "real_volume")
    spread = _row_float(row, "spread")

    payload: Dict[str, Any] = {
        "symbol": str(symbol).upper().strip(),
        "timeframe": timeframe,
        "open_time_utc": _format_utc_iso(open_at_utc),
        "close_time_utc": _format_utc_iso(close_at_utc),
        "open": float(open_price),
        "high": float(high_price),
        "low": float(low_price),
        "close": float(close_price),
    }
    if tick_volume is not None:
        payload["tick_volume"] = _volume_payload_value(tick_volume)
    if real_volume is not None:
        payload["real_volume"] = _volume_payload_value(real_volume)
    volume = real_volume if real_volume not in (None, 0.0) else tick_volume
    if volume is not None:
        payload["volume"] = _volume_payload_value(volume)
        payload["volume_source"] = (
            "real_volume" if real_volume not in (None, 0.0) else "tick_volume"
        )
    if spread is not None:
        payload["spread_points"] = _volume_payload_value(spread)
        if price_point is not None:
            payload["spread"] = _round_market_stat(float(spread) * price_point)

    units: Dict[str, str] = {}
    if tick_volume is not None:
        units["tick_volume"] = "broker_tick_count"
    if real_volume is not None:
        units["real_volume"] = "traded_volume"
    if volume is not None:
        units["volume"] = (
            "traded_volume"
            if real_volume not in (None, 0.0)
            else "broker_tick_count"
        )
    if spread is not None:
        units["spread_points"] = "broker_points"
        if price_point is not None:
            units["spread"] = "absolute_price"
    if units:
        payload["units"] = units

    payload.update(
        _boundary_closed_candle_stats(
            open_price=float(open_price),
            high_price=float(high_price),
            low_price=float(low_price),
            close_price=float(close_price),
        )
    )
    return payload

def _boundary_closed_candle_stats(
    *,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
) -> Dict[str, Any]:
    change = close_price - open_price
    price_range = high_price - low_price
    body = abs(change)
    upper_wick = max(0.0, high_price - max(open_price, close_price))
    lower_wick = max(0.0, min(open_price, close_price) - low_price)
    direction = "doji"
    if change > 0.0:
        direction = "bullish"
    elif change < 0.0:
        direction = "bearish"

    stats: Dict[str, Any] = {
        "direction": direction,
        "change": _round_market_stat(change),
        "range": _round_market_stat(price_range),
        "body": _round_market_stat(body),
        "upper_wick": _round_market_stat(upper_wick),
        "lower_wick": _round_market_stat(lower_wick),
        "midpoint": _round_market_stat((high_price + low_price) / 2.0),
        "typical_price": _round_market_stat(
            (high_price + low_price + close_price) / 3.0
        ),
    }
    if open_price != 0.0:
        stats["change_pct"] = _round_market_stat((change / open_price) * 100.0)
        stats["range_pct"] = _round_market_stat((price_range / open_price) * 100.0)
    if price_range > 0.0:
        stats["body_pct_of_range"] = _round_market_stat((body / price_range) * 100.0)
        stats["close_position"] = _round_market_stat(
            (close_price - low_price) / price_range
        )
    else:
        stats["body_pct_of_range"] = 0.0
        stats["close_position"] = None
    return stats

def _rate_open_datetime(row: Any) -> Optional[datetime]:
    epoch = _rate_open_epoch(row)
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc)
    except Exception:
        return None

def _rate_open_epoch(row: Any) -> Optional[float]:
    return _finite_number(_row_value(row, "time"))

def _round_market_stat(value: float) -> float:
    rounded = round(float(value), 10)
    return 0.0 if rounded == -0.0 else rounded

def _volume_payload_value(value: float) -> int | float:
    numeric = float(value)
    rounded = round(numeric)
    if abs(numeric - rounded) <= 1e-9:
        return int(rounded)
    return numeric

def _boundary_cutoff_utc(boundary: Dict[str, Any]) -> datetime:
    return datetime.fromtimestamp(float(boundary["boundary_at_epoch"]), tz=timezone.utc)
