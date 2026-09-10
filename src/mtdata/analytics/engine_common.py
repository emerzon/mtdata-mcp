"""Statistical engines for the advanced MT5-native analytics tools."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..services.data_service import fetch_history_frame, fetch_tick_frame
from ..utils.time import MAX_TRADING_MINUTES_BACK, format_datetime_utc
from ..utils.utils import (
    _parse_end_datetime,
    _parse_start_datetime,
)


def _mapping(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    converter = getattr(row, "_asdict", None)
    if callable(converter):
        return dict(converter())
    return {
        name: getattr(row, name)
        for name in dir(row)
        if not name.startswith("_")
        and not callable(getattr(row, name, None))
    }


def _parse_time(
    value: Optional[str],
    default: datetime,
    *,
    end_bound: bool = False,
) -> datetime:
    if not value:
        return default
    parser = _parse_end_datetime if end_bound else _parse_start_datetime
    parsed = parser(str(value))
    if parsed is None:
        raise ValueError(f"Could not parse datetime: {value}")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)



def _window(start: Optional[str], end: Optional[str], minutes_back: int) -> Tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    to_dt = _parse_time(end, now, end_bound=True)
    if start:
        from_dt = _parse_time(start, to_dt)
    else:
        minutes = int(minutes_back)
        if minutes > MAX_TRADING_MINUTES_BACK:
            raise ValueError(
                f"minutes_back={minutes} exceeds the maximum supported lookback of "
                f"{MAX_TRADING_MINUTES_BACK} minutes (~20 years)."
            )
        try:
            from_dt = to_dt - timedelta(minutes=minutes)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                f"minutes_back={minutes} exceeds the maximum supported lookback of "
                f"{MAX_TRADING_MINUTES_BACK} minutes (~20 years)."
            ) from exc
    if from_dt >= to_dt:
        raise ValueError("start must be earlier than end")
    return from_dt, to_dt



def _analysis_window_metadata(
    request: Any,
    start: datetime,
    end: datetime,
    *,
    source_override: Optional[str] = None,
) -> Dict[str, Any]:
    explicit_fields = request.model_fields_set
    requested = {
        name: getattr(request, name)
        for name in ("start", "end", "minutes_back")
        if name in explicit_fields
    }
    if source_override is not None:
        source = source_override
    elif request.start is not None and request.end is not None:
        source = "explicit_range"
    elif request.start is not None:
        source = "explicit_start_to_now"
    elif request.end is not None:
        source = "end_anchored_default_lookback"
    elif "minutes_back" in explicit_fields:
        source = "minutes_back"
    else:
        source = "default_lookback"
    out: Dict[str, Any] = {
        "start": format_datetime_utc(start, timespec="auto"),
        "end": format_datetime_utc(end, timespec="auto"),
        "timezone": "UTC",
        "source": source,
        "minutes_back_effective": round(
            (end - start).total_seconds() / 60.0,
            6,
        ),
        "requested": requested,
    }
    if "minutes_back" in explicit_fields:
        out["minutes_back_requested"] = int(request.minutes_back)
    elif request.start is None:
        out["defaulted"] = {"minutes_back": int(request.minutes_back)}
    return out



def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)



def _percentiles(values: Iterable[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return {key: None for key in ("mean", "median", "p90", "p95", "p99", "max")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }



def _circular_block_bootstrap_means(
    values: Sequence[float],
    samples: int,
    seed: int = 42,
    *,
    min_block_size: int = 1,
) -> Optional[np.ndarray]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return None
    rng = np.random.default_rng(seed)
    block = max(int(min_block_size), int(round(math.sqrt(len(arr)))))
    means = []
    for _ in range(int(samples)):
        starts = rng.integers(0, len(arr), size=math.ceil(len(arr) / block))
        draw = np.concatenate([arr[(start + np.arange(block)) % len(arr)] for start in starts])[: len(arr)]
        means.append(float(np.mean(draw)))
    return np.asarray(means, dtype=float)


def _bootstrap_mean_ci(values: Sequence[float], samples: int, seed: int = 42) -> Optional[List[float]]:
    means = _circular_block_bootstrap_means(values, samples, seed)
    if means is None:
        return None
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _log_close_returns(bars: pd.DataFrame, *, name: Optional[str] = None) -> pd.Series:
    """Return close log differences indexed by native bar timestamps."""
    return pd.Series(
        np.log(bars["close"]).diff().to_numpy(),
        index=bars["time"].to_numpy(),
        name=name,
    )



def _round_execution_stat(value: Any, *, significant_digits: int = 6) -> Any:
    """Remove binary float tails from derived execution statistics."""
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric == 0.0:
        return numeric
    decimals = significant_digits - int(math.floor(math.log10(abs(numeric)))) - 1
    return round(numeric, decimals)



def _tick_frame(gateway: Any, symbol: str, start: datetime, end: datetime, max_ticks: int) -> Tuple[pd.DataFrame, bool]:
    return fetch_tick_frame(
        symbol,
        start,
        end,
        max_ticks,
        gateway=gateway,
    )



def _rates(
    gateway: Any,
    symbol: str,
    timeframe: str,
    count: int,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    try:
        return fetch_history_frame(
            symbol,
            timeframe,
            int(count),
            start=start,
            end=end,
            include_incomplete=False,
            gateway=gateway,
        )
    except ValueError as exc:
        if "No data is available" not in str(exc):
            raise
        return pd.DataFrame()
