"""Statistical engines for the advanced MT5-native analytics tools."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..shared.constants import TIMEFRAME_MAP
from ..utils.tick_flags import bid_ask_flags
from ..utils.time import MAX_TRADING_MINUTES_BACK, bar_close_epoch, format_datetime_utc
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
    return {name: getattr(row, name) for name in dir(row) if not name.startswith("_") and not callable(getattr(row, name, None))}



def _frame(rows: Any) -> pd.DataFrame:
    if rows is None:
        return pd.DataFrame()
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    if isinstance(rows, np.ndarray) and rows.dtype.names:
        return pd.DataFrame(rows)
    return pd.DataFrame([_mapping(row) for row in list(rows)])



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
    flags = getattr(gateway, "COPY_TICKS_ALL", 0)
    df = _frame(gateway.copy_ticks_range(symbol, start, end, flags))
    if df.empty:
        return pd.DataFrame(
            {
                column: pd.Series(dtype=float)
                for column in (
                    "epoch",
                    "bid",
                    "ask",
                    "last",
                    "volume",
                    "volume_real",
                    "flags",
                    "spread_valid",
                    "spread_quality",
                    "mid",
                    "spread",
                )
            }
        ), False
    time_msc = _finite(df.get("time_msc", pd.Series(index=df.index, dtype=float)))
    epoch = _finite(df.get("time", pd.Series(index=df.index, dtype=float)))
    df["epoch"] = np.where(time_msc > 0, time_msc / 1000.0, epoch)
    dedupe_columns = [
        column
        for column in ("epoch", "bid", "ask", "last", "volume", "volume_real", "flags")
        if column in df.columns
    ]
    df = (
        df[np.isfinite(df["epoch"])]
        .sort_values("epoch", kind="stable")
        .drop_duplicates(
            subset=dedupe_columns,
            keep="last",
        )
    )
    truncated = len(df) > int(max_ticks)
    if truncated:
        df = df.tail(int(max_ticks)).copy()
    for column in ("bid", "ask", "last", "volume", "volume_real", "flags"):
        if column not in df:
            df[column] = 0.0
        df[column] = _finite(df[column]).fillna(0.0)
    bid_flag, ask_flag = bid_ask_flags(gateway)
    flag_values = df["flags"].astype(np.int64)
    one_sided_update = ((flag_values & bid_flag) != 0) != (
        (flag_values & ask_flag) != 0
    )
    two_sided_quote = (df["bid"] > 0) & (df["ask"] > df["bid"])
    # MqlTick flags identify changed fields; bid and ask remain a complete quote
    # snapshot even when only one side changed in this event.
    spread_sample_eligible = two_sided_quote
    incomplete_one_sided_update = one_sided_update & ~two_sided_quote
    locked_quote = (df["bid"] > 0) & (df["ask"] == df["bid"])
    inverted_quote = (df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] < df["bid"])
    df["spread_quality"] = np.select(
        [incomplete_one_sided_update, locked_quote, inverted_quote],
        ["one_sided_update", "locked", "inverted"],
        default="two_sided",
    )
    df.loc[(df["bid"] <= 0) | (df["ask"] <= 0), "spread_quality"] = "one_sided"
    df["spread_valid"] = two_sided_quote
    df["spread_sample_eligible"] = spread_sample_eligible
    df["mid"] = np.where(
        two_sided_quote,
        (df["bid"] + df["ask"]) / 2.0,
        np.nan,
    )
    df["spread"] = np.where(np.isfinite(df["mid"]), df["ask"] - df["bid"], np.nan)
    return df.reset_index(drop=True), truncated



def _rates(
    gateway: Any,
    symbol: str,
    timeframe: str,
    count: int,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    if start and end:
        from_dt, to_dt = _window(start, end, 1)
        raw = gateway.copy_rates_range(symbol, TIMEFRAME_MAP[timeframe], from_dt, to_dt)
    elif end:
        _, to_dt = _window(None, end, 1)
        raw = gateway.copy_rates_from(symbol, TIMEFRAME_MAP[timeframe], to_dt, int(count) + 2)
    else:
        raw = gateway.copy_rates_from_pos(symbol, TIMEFRAME_MAP[timeframe], 0, int(count) + 2)
    df = _frame(raw)
    if df.empty or "close" not in df or "time" not in df:
        return pd.DataFrame()
    df = df.sort_values("time", kind="stable").drop_duplicates("time", keep="last")
    for column in ("open", "high", "low", "close", "tick_volume", "real_volume", "spread"):
        if column not in df:
            df[column] = 0.0
        df[column] = _finite(df[column])
    now = datetime.now(timezone.utc).timestamp()
    information_cutoff = min(now, to_dt.timestamp()) if end else now
    close_epochs = df["time"].map(lambda value: bar_close_epoch(float(value), timeframe))
    df = df[close_epochs <= information_cutoff]
    if not (start and end):
        df = df.tail(int(count))
    return df.reset_index(drop=True)
