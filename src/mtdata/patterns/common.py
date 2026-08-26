import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ..services.data_service.candles import _is_last_bar_forming
from ..shared.symbols import is_probably_crypto_symbol
from ..utils.time import bar_close_epoch, coerce_time_epoch_seconds
from ..utils.utils import _utc_epoch_seconds, to_float_np


def compute_atr_sma(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """SMA of true range over ``period`` (min_periods = max(2, period//2))."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    n = min(h.size, l.size, c.size)
    if n <= 0:
        return np.asarray([], dtype=float)
    h = h[:n]
    l = l[:n]
    c = c[:n]
    prev_c = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    win = max(2, int(period))
    try:
        return (
            pd.Series(tr)
            .rolling(win, min_periods=max(2, win // 2))
            .mean()
            .to_numpy(dtype=float)
        )
    except (TypeError, ValueError):
        return tr.astype(float)


def fallback_local_extrema(
    src: np.ndarray,
    min_dist: int,
    order: int,
    *,
    prefer_high: bool,
) -> np.ndarray:
    """Find local extrema with a sliding window when primary peak detection undershoots.

    Plateau runs are collapsed to their midpoint so flat tops/bottoms yield a
    single representative index. Candidates closer than ``min_dist`` keep the
    more extreme value.
    """
    values = np.asarray(src, dtype=float)
    n = int(values.size)
    if n < (2 * order + 1):
        return np.asarray([], dtype=int)
    candidates: List[int] = []
    for idx in range(order, n - order):
        center = float(values[idx])
        if not np.isfinite(center):
            continue
        window = values[idx - order : idx + order + 1]
        if not np.all(np.isfinite(window)):
            continue
        plateau_tol = max(1e-12, abs(center) * 1e-12)
        plateau_left = idx
        while (
            plateau_left > 0
            and np.isfinite(values[plateau_left - 1])
            and np.isclose(
                values[plateau_left - 1],
                center,
                rtol=0.0,
                atol=plateau_tol,
            )
        ):
            plateau_left -= 1
        plateau_right = idx
        while (
            plateau_right < (n - 1)
            and np.isfinite(values[plateau_right + 1])
            and np.isclose(
                values[plateau_right + 1],
                center,
                rtol=0.0,
                atol=plateau_tol,
            )
        ):
            plateau_right += 1
        if plateau_left != plateau_right:
            if int((plateau_left + plateau_right) // 2) != int(idx):
                continue
        if prefer_high:
            if center < float(np.max(window)):
                continue
        elif center > float(np.min(window)):
            continue
        candidates.append(int(idx))
    if not candidates:
        return np.asarray([], dtype=int)
    reduced: List[int] = []
    for idx in candidates:
        if not reduced or (idx - reduced[-1]) >= int(min_dist):
            reduced.append(int(idx))
            continue
        prev_idx = int(reduced[-1])
        prev_val = float(values[prev_idx])
        curr_val = float(values[idx])
        better = idx if (curr_val > prev_val if prefer_high else curr_val < prev_val) else prev_idx
        reduced[-1] = int(better)
    return np.asarray(reduced, dtype=int)


def compute_pivot_thresholds(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    cfg: Any,
) -> Tuple[float, int]:
    """ATR-adaptive prominence/distance for pivot extraction.

    ``cfg`` is duck-typed (Classic/Harmonic detector configs share these fields).
    """
    x = np.asarray(close, dtype=float)
    finite = x[np.isfinite(x)]
    base = float(np.median(finite)) if finite.size else 0.0
    if not np.isfinite(base) or abs(base) <= 1e-12:
        base = float(np.mean(finite)) if finite.size else 1.0
    prom_abs = abs(base) * (float(getattr(cfg, "min_prominence_pct", 0.5)) / 100.0)
    min_dist = max(2, int(getattr(cfg, "min_distance", 5)))

    use_prom = bool(getattr(cfg, "pivot_use_atr_adaptive_prominence", False))
    use_dist = bool(getattr(cfg, "pivot_use_atr_adaptive_distance", False))
    if use_prom or use_dist:
        atr = compute_atr_sma(high, low, x, int(getattr(cfg, "pivot_atr_period", 14)))
        finite_atr = atr[np.isfinite(atr) & (atr > 0.0)]
        if finite_atr.size > 0:
            atr_med = float(np.median(finite_atr))
            if use_prom:
                prom_abs = max(
                    prom_abs,
                    float(getattr(cfg, "pivot_atr_prominence_mult", 1.0)) * atr_med,
                )
            if use_dist and abs(base) > 1e-12:
                atr_pct = abs(atr_med / base) * 100.0
                dist_mult = float(getattr(cfg, "pivot_atr_distance_mult", 0.0))
                scale = 1.0 + max(0.0, dist_mult) * atr_pct
                max_scale = float(max(1.0, getattr(cfg, "pivot_max_distance_scale", 3.0)))
                scale = min(max_scale, max(1.0, scale))
                base_dist = float(getattr(cfg, "min_distance", 5))
                min_dist = max(2, int(round(base_dist * scale)))
    return float(max(1e-12, prom_abs)), int(min_dist)


def detect_pivots(
    close: np.ndarray,
    cfg: Any,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return peak and trough indices using close or optional high/low arrays."""
    x = np.asarray(close, dtype=float)
    if x.size < max(5, int(getattr(cfg, "min_distance", 5)) * 3):
        return np.asarray([], dtype=int), np.asarray([], dtype=int)

    hi = np.asarray(high, dtype=float) if high is not None else x
    lo = np.asarray(low, dtype=float) if low is not None else x
    if hi.size != x.size or not np.isfinite(hi).all():
        hi = x
    if lo.size != x.size or not np.isfinite(lo).all():
        lo = x

    prom_abs, min_dist = compute_pivot_thresholds(x, hi, lo, cfg)
    src_hi = hi if bool(getattr(cfg, "pivot_use_hl", True)) else x
    src_lo = lo if bool(getattr(cfg, "pivot_use_hl", True)) else x
    try:
        peaks, _ = find_peaks(src_hi, prominence=prom_abs, distance=min_dist)
        troughs, _ = find_peaks(-src_lo, prominence=prom_abs, distance=min_dist)
    except ValueError:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)

    if bool(getattr(cfg, "pivot_enable_fallback", True)):
        min_peaks = int(max(0, getattr(cfg, "pivot_fallback_min_peaks", 2)))
        min_troughs = int(max(0, getattr(cfg, "pivot_fallback_min_troughs", 2)))
        order = max(1, int(getattr(cfg, "pivot_fallback_order", 2)))
        if int(peaks.size) < min_peaks:
            peaks = fallback_local_extrema(src_hi, min_dist, order, prefer_high=True)
        if int(troughs.size) < min_troughs:
            troughs = fallback_local_extrema(src_lo, min_dist, order, prefer_high=False)
    return peaks.astype(int), troughs.astype(int)


def _coerce_pattern_time_epoch(values: Any, expected_size: int) -> np.ndarray:
    """Convert a time column to UTC epoch seconds."""
    times = np.asarray(coerce_time_epoch_seconds(values), dtype=float)
    if times.size == 0:
        return times
    finite = times[np.isfinite(times)]
    if finite.size == 0:
        return times if times.size == expected_size else np.asarray([], dtype=float)
    return times


def prepare_ohlc_pattern_inputs(
    df: pd.DataFrame,
    *,
    max_bars: int,
    min_input_bars: int,
    log_label: str = "Pattern detection",
    log_extra: str = "",
    time_mode: Literal["empty", "arange"] = "arange",
) -> Optional[Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    """Slice history and extract close/high/low/time arrays for pattern detectors."""
    if not isinstance(df, pd.DataFrame) or "close" not in df.columns:
        return None
    if len(df) > int(max_bars):
        df = df.iloc[-int(max_bars) :].copy()

    close = to_float_np(df["close"])
    used_close_for_high = "high" not in df.columns
    used_close_for_low = "low" not in df.columns
    high = to_float_np(df["high"]) if not used_close_for_high else close
    low = to_float_np(df["low"]) if not used_close_for_low else close
    if high.size != close.size:
        used_close_for_high = True
        high = close
    if low.size != close.size:
        used_close_for_low = True
        low = close
    if used_close_for_high or used_close_for_low:
        logging.getLogger(__name__).warning(
            "%s falling back to close for missing/mismatched "
            "high/low columns (high_fallback=%s, low_fallback=%s)%s",
            log_label,
            used_close_for_high,
            used_close_for_low,
            log_extra,
        )

    n = int(close.size)
    if n < int(min_input_bars):
        return None

    if "time" in df.columns:
        times = _coerce_pattern_time_epoch(df["time"], n)
        if times.size != n or not np.isfinite(times).any():
            if time_mode == "arange":
                times = np.arange(n, dtype=float)
            else:
                times = np.asarray([], dtype=float)
    elif time_mode == "arange":
        times = np.arange(n, dtype=float)
    else:
        times = np.asarray([], dtype=float)

    return df, times, close, high, low, n


@dataclass
class PatternResultBase:
    confidence: float
    start_index: int
    end_index: int
    start_time: Optional[float]
    end_time: Optional[float]

    @staticmethod
    def resolve_time(times: Any, index: int) -> Optional[float]:
        try:
            idx = int(index)
        except (TypeError, ValueError):
            return None
        if idx < 0:
            return None
        try:
            arr = np.asarray(times, dtype=float)
        except (TypeError, ValueError):
            return None
        if arr.ndim == 0 or arr.size <= idx:
            return None
        value = float(arr[idx])
        return value if np.isfinite(value) else None


def interval_overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Return the inclusive overlap ratio between two index intervals."""
    lo = max(int(a_start), int(b_start))
    hi = min(int(a_end), int(b_end))
    inter = max(0, hi - lo + 1)
    union = max(int(a_end), int(b_end)) - min(int(a_start), int(b_start)) + 1
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _crosses_weekend(start_epoch: float, end_epoch: float) -> bool:
    try:
        start_dt = datetime.fromtimestamp(float(start_epoch), tz=timezone.utc)
        end_dt = datetime.fromtimestamp(float(end_epoch), tz=timezone.utc)
    except Exception:
        return False
    if end_dt <= start_dt:
        return False
    current = start_dt
    while current <= end_dt:
        if current.weekday() >= 5:
            return True
        current = (current + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    return end_dt.weekday() >= 5


def data_quality_warnings(  # noqa: C901
    df: Any,
    *,
    symbol: Optional[str] = None,
    timeframe_seconds: Optional[float] = None,
) -> list[str]:
    warnings: list[str] = []
    if not isinstance(df, pd.DataFrame) or len(df) < 3:
        return warnings

    close_col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
    if close_col is not None:
        try:
            close = pd.to_numeric(df[close_col], errors="coerce").to_numpy(dtype=float, copy=False)
        except Exception:
            close = np.asarray([], dtype=float)
        close = close[np.isfinite(close)]
        if close.size >= 3:
            steps = np.abs(np.diff(close))
            if steps.size > 0:
                zero_share = float(np.mean(steps <= 1e-12))
                if zero_share >= 0.6:
                    warnings.append("Data quality warning: repeated close prices dominate the sample.")
                elif float(np.nanmax(steps)) <= 1e-12:
                    warnings.append("Data quality warning: close prices are nearly flat across the sample.")

    if timeframe_seconds is not None and float(timeframe_seconds) > 0:
        time_col = "time" if "time" in df.columns else ("Time" if "Time" in df.columns else None)
        if time_col is not None:
            try:
                times = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float, copy=False)
            except Exception:
                times = np.asarray([], dtype=float)
            times = times[np.isfinite(times)]
            if times.size >= 3:
                gaps = np.diff(times)
                threshold = 1.5 * float(timeframe_seconds)
                if gaps.size > 0 and float(np.nanmax(gaps)) > threshold:
                    expected_weekend_gaps = 0
                    unexpected_gaps = 0
                    is_crypto = is_probably_crypto_symbol(symbol)
                    for idx, gap in enumerate(gaps):
                        if not np.isfinite(gap) or float(gap) <= threshold:
                            continue
                        start_epoch = float(times[idx])
                        end_epoch = float(times[idx + 1])
                        if not is_crypto and _crosses_weekend(
                            start_epoch, end_epoch
                        ):
                            expected_weekend_gaps += 1
                        else:
                            unexpected_gaps += 1
                    if unexpected_gaps > 0:
                        suffix = ""
                        if expected_weekend_gaps:
                            suffix = (
                                f" ({expected_weekend_gaps} expected weekend/session "
                                "gap(s) suppressed)."
                            )
                        warnings.append(
                            "Data quality warning: detected "
                            f"{unexpected_gaps} unexpected time gap(s) larger than "
                            f"1.5 bar intervals.{suffix}"
                        )

    volume_col = None
    volume_series: Optional[np.ndarray] = None
    fallback_volume_col = None
    fallback_volume_series: Optional[np.ndarray] = None
    for candidate in ("real_volume", "volume", "tick_volume", "Volume"):
        if candidate in df.columns:
            try:
                candidate_volume = pd.to_numeric(
                    df[candidate], errors="coerce"
                ).to_numpy(dtype=float, copy=False)
            except Exception:
                candidate_volume = np.asarray([], dtype=float)
            candidate_volume = candidate_volume[np.isfinite(candidate_volume)]
            if candidate_volume.size < 5:
                continue
            if fallback_volume_series is None:
                fallback_volume_col = candidate
                fallback_volume_series = candidate_volume
            if np.any(candidate_volume > 0):
                volume_col = candidate
                volume_series = candidate_volume
                break
    if volume_series is None:
        volume_col = fallback_volume_col
        volume_series = fallback_volume_series
    if volume_col is not None and volume_series is not None:
        if volume_series.size >= 5:
            zero_share = float(np.mean(volume_series <= 0))
            if zero_share >= 0.6:
                if is_probably_crypto_symbol(symbol):
                    warnings.append(
                        "Data quality warning: zero-volume bars dominate the sample "
                        "(common for crypto low-volume periods)."
                    )
                else:
                    warnings.append("Data quality warning: zero-volume bars dominate the sample.")

    return warnings


def should_drop_last_live_bar(
    df: pd.DataFrame,
    timeframe: str,
    *,
    now_utc: Optional[datetime] = None,
    current_time_epoch: Optional[float] = None,
) -> bool:
    """Return True when the last bar is still forming or cannot be validated."""
    epoch = current_time_epoch
    if epoch is None and now_utc is not None:
        epoch = now_utc.timestamp()
    return _is_last_bar_forming(df, timeframe, current_time_epoch=epoch)


def closed_bar_cutoff_epoch(
    end_dt: Optional[datetime],
    now_utc: datetime,
) -> Optional[float]:
    """Return min(parsed end, now) as UTC epoch, or None when no end was given."""
    if end_dt is None:
        return None
    return min(float(_utc_epoch_seconds(end_dt)), float(now_utc.timestamp()))


def keep_bars_closed_at_or_before(
    df: pd.DataFrame,
    timeframe: str,
    cutoff_epoch: float,
) -> pd.DataFrame:
    """Keep bars whose close is knowable at *cutoff_epoch*."""
    if df.empty or "time" not in df.columns:
        return df
    return df.loc[
        df["time"].map(
            lambda value: bar_close_epoch(value, timeframe) <= cutoff_epoch
        )
    ].copy()
