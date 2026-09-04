import math
import time
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any, Dict, List, Optional

from ...core.error_envelope import build_error_payload
from ...shared.schema import TimeframeLiteral
from ...utils.freshness import closed_session_context, is_standard_weekend_closure
from ...utils.mt5 import (
    _mt5_copy_rates_from_pos,
    get_symbol_info_cached,
    mt5,
)
from ...utils.time import _format_time_minimal
from ...utils.utils import _utc_epoch_seconds
from .query import _candle_query_applied, _parse_fetch_datetime_arg


def _format_mt5_last_error() -> str:
    try:
        err = mt5.last_error()
    except Exception as exc:
        return str(exc)
    if isinstance(err, tuple) and len(err) == 2:
        code, message = err
        return f"({code}, {message!r})"
    return str(err)


def _describe_rate_fetch_error(symbol: str, *, info_before: Any = None) -> str:
    if info_before is None:
        try:
            info_before = get_symbol_info_cached(symbol)
        except Exception:
            info_before = None

    error_text = _format_mt5_last_error()
    if info_before is None:
        return (
            f"Symbol '{symbol}' was not found or is not available in MT5. "
            f"Use symbols_list(search_term='{symbol}') to find broker-specific names and suffixes."
        )
    return f"Failed to get rates for {symbol}: {error_text}"


def _bounded_weekend_no_data_context(
    symbol: str,
    start_datetime: Optional[str],
    end_datetime: Optional[str],
    *,
    item: str = "candles",
) -> Dict[str, Any]:
    if not start_datetime or not end_datetime:
        return {}
    try:
        start_utc, _ = _parse_fetch_datetime_arg(start_datetime)
        end_utc, _ = _parse_fetch_datetime_arg(end_datetime, end_bound=True)
        if start_utc is None or end_utc is None:
            return {}
        start_utc = (
            start_utc.replace(tzinfo=dt_timezone.utc)
            if start_utc.tzinfo is None
            else start_utc.astimezone(dt_timezone.utc)
        )
        end_utc = (
            end_utc.replace(tzinfo=dt_timezone.utc)
            if end_utc.tzinfo is None
            else end_utc.astimezone(dt_timezone.utc)
        )
        duration = end_utc - start_utc
        if duration.total_seconds() < 0 or duration > timedelta(days=3):
            return {}
        midpoint = start_utc + duration / 2
        if not (
            is_standard_weekend_closure(start_utc)
            and (
                is_standard_weekend_closure(end_utc)
                or is_standard_weekend_closure(midpoint)
            )
        ):
            return {}
        item_label = str(item or "data").strip() or "data"
        session = closed_session_context(
            symbol,
            now_epoch=midpoint.timestamp(),
            item=item_label,
        )
        if not session or session.get("market_status_reason") != "weekend":
            return {}
    except Exception:
        return {}

    return {
        "no_data_reason": "market_closed_weekend",
        "market_status": "closed",
        "market_status_reason": "weekend",
        "market_status_source": "standard_weekend_hours",
        "note": (
            f"The requested range falls entirely within standard weekend closure "
            f"hours for {symbol}; no {item_label} are expected."
        ),
        "suggestion": "Choose a range containing an open trading session.",
    }


def attach_empty_range_weekend_context(
    payload: Dict[str, Any],
    *,
    symbol: str,
    start: Optional[str],
    end: Optional[str],
    item: str = "ticks",
) -> Dict[str, Any]:
    """Copy asset-aware weekend closure fields onto an empty range payload."""
    context = _bounded_weekend_no_data_context(
        symbol,
        start,
        end,
        item=item,
    )
    if not context:
        return payload
    reason = context.get("no_data_reason")
    if reason:
        payload["empty_reason"] = reason
        payload["no_data_reason"] = reason
    for key in (
        "market_status",
        "market_status_reason",
        "market_status_source",
        "note",
        "suggestion",
    ):
        if context.get(key) is not None:
            payload[key] = context[key]
    return payload


def _build_no_data_error_with_context(
    symbol: str,
    timeframe: TimeframeLiteral,
    mt5_timeframe: int,
    start_datetime: Optional[str],
    end_datetime: Optional[str],
) -> Dict[str, Any]:
    """Build a detailed error payload when no data is available for the requested range."""
    error_msg = "No data available"
    details: Dict[str, Any] = {}

    if start_datetime or end_datetime:
        details["requested_range"] = {
            k: v for k, v in [("start", start_datetime), ("end", end_datetime)]
            if v is not None
        }
    details.update(
        _bounded_weekend_no_data_context(symbol, start_datetime, end_datetime)
    )

    try:
        available_bars = _mt5_copy_rates_from_pos(symbol, mt5_timeframe, 0, 1)

        if available_bars is not None and len(available_bars) > 0:
            times: List[float] = []
            for bar in available_bars:
                try:
                    epoch = float(bar["time"])
                except Exception:
                    continue
                if math.isfinite(epoch):
                    times.append(epoch)
            if not times:
                raise ValueError("available bars have no finite timestamps")
            last_epoch = max(times)
            last_time = datetime.fromtimestamp(last_epoch, tz=dt_timezone.utc)

            details["available_range"] = {
                "latest": _format_time_minimal(last_epoch),
                "earliest": None,
                "earliest_status": "not_scanned",
            }

            if start_datetime:
                try:
                    req_start, _ = _parse_fetch_datetime_arg(
                        start_datetime,
                        timeframe=timeframe,
                    )
                    if req_start is not None and req_start.tzinfo is None:
                        req_start = req_start.replace(tzinfo=dt_timezone.utc)
                    elif req_start is not None:
                        req_start = req_start.astimezone(dt_timezone.utc)
                    if req_start and req_start > last_time:
                        error_msg = f"No data available - requested start date is after latest available data ({_format_time_minimal(last_epoch)})"
                        details["suggestion"] = f"Use start='{_format_time_minimal(last_epoch)}' or earlier"
                except Exception:
                    pass
    except Exception:
        pass

    payload = build_error_payload(
        error_msg,
        code="data_fetch_candles_no_data",
        operation="data_fetch_candles",
        details=details or None,
    )
    if start_datetime or end_datetime:
        payload["query_applied"] = _candle_query_applied(
            timeframe=timeframe,
            start=start_datetime,
            end=end_datetime,
            limit=None,
        )
    return payload


def _future_start_error(
    start_datetime: str, from_date: datetime, seconds_per_bar: int
) -> Optional[str]:
    """Return an error when the requested start is in the future.

    A future ``start`` yields no historical bars; MT5 silently returns recent
    bars that are then trimmed away, producing an opaque empty success. Reject
    it explicitly (like reversed dates) so callers get an actionable signal.
    A one-bar + clock-skew tolerance avoids false positives near the live bar.
    """
    try:
        from_epoch = _utc_epoch_seconds(from_date)
        tolerance = max(int(seconds_per_bar), 300)
        if from_epoch > time.time() + tolerance:
            return (
                f"start datetime {start_datetime} is in the future; "
                "no historical data is available for future dates."
            )
    except Exception:
        return None
    return None
