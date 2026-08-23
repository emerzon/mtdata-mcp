from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...utils.utils import to_float_np
from ..common import (
    PatternResultBase,
    detect_pivots,
)
from .config import ClassicDetectorConfig, ClassicPatternResult


def _level_close(a: float, b: float, tol_pct: float) -> bool:
    if a == 0 or b == 0:
        return abs(a - b) <= 1e-12
    return abs((a - b) / ((abs(a) + abs(b)) / 2.0)) * 100.0 <= tol_pct


def _effective_flat_slope(close: np.ndarray, cfg: ClassicDetectorConfig) -> float:
    """Return a price-scaled slope used as 'flat'.

    ``max_flat_slope`` stays an absolute floor (so explicit 0.02 overrides
    still mean 0.02 price units per bar). ``max_flat_slope_pct_per_bar``
    scales with median price so a 1-point wiggle on a 5000 index is not a
    rising wedge.
    """
    floor = abs(float(getattr(cfg, "max_flat_slope", 1e-4)))
    pct = abs(float(getattr(cfg, "max_flat_slope_pct_per_bar", 0.01)))
    finite = np.asarray(close, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return max(floor, 1e-12)
    med = abs(float(np.median(finite)))
    return max(floor, med * (pct / 100.0), 1e-12)


@lru_cache(maxsize=1)
def _get_ransac_regressor_cls():
    from sklearn.linear_model import RANSACRegressor  # type: ignore
    return RANSACRegressor


def _stable_r2(ss_res: float, ss_tot: float) -> float:
    if ss_tot == 0.0:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    # slope, intercept, r2
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0, float(y.mean() if y.size else 0.0), 0.0
    p = np.polyfit(x, y, 1)
    y_hat = p[0] * x + p[1]
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) if y.size else 0.0
    r2 = _stable_r2(ss_res, ss_tot)
    return float(p[0]), float(p[1]), float(r2)


def _ransac_residual_threshold(x: np.ndarray, y: np.ndarray, cfg: ClassicDetectorConfig) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return 1e-9

    slope, intercept, _ = _fit_line(x, y)
    y_hat = slope * x + intercept
    baseline_resid = np.abs(y - y_hat)
    finite_resid = baseline_resid[np.isfinite(baseline_resid)]
    resid_scale = float(np.median(finite_resid)) if finite_resid.size else 0.0
    if not np.isfinite(resid_scale) or resid_scale <= 1e-9:
        finite_y = y[np.isfinite(y)]
        resid_scale = float(np.ptp(finite_y)) if finite_y.size else 0.0
    return max(1e-9, float(cfg.ransac_residual_scale) * max(1e-9, resid_scale))


def _fit_line_robust(x: np.ndarray, y: np.ndarray, cfg: ClassicDetectorConfig) -> Tuple[float, float, float]:
    """Optionally fit a robust line via RANSAC; fallback to ordinary fit."""
    if not cfg.use_robust_fit:
        return _fit_line(x, y)
    X = x.reshape(-1, 1).astype(float)
    yv = y.astype(float)
    if X.shape[0] < max(2, int(cfg.ransac_min_samples)):
        return _fit_line(x, y)
    resid = _ransac_residual_threshold(x, yv, cfg)
    try:
        ransac_cls = _get_ransac_regressor_cls()
    except ImportError:
        return _fit_line(x, y)
    try:
        model = ransac_cls(
            min_samples=max(2, int(cfg.ransac_min_samples)),
            max_trials=max(10, int(cfg.ransac_max_trials)),
            residual_threshold=resid,
            random_state=0,
        )
        model.fit(X, yv)
        est = model.estimator_
        slope = float(getattr(est, 'coef_', [0.0])[0])
        intercept = float(getattr(est, 'intercept_', 0.0))
        inlier_mask = np.asarray(
            getattr(model, "inlier_mask_", np.ones(yv.size, dtype=bool)),
            dtype=bool,
        )
        if inlier_mask.shape != yv.shape or int(inlier_mask.sum()) < 2:
            inlier_mask = np.ones(yv.size, dtype=bool)
        x_inliers = x[inlier_mask]
        y_inliers = yv[inlier_mask]
        y_hat = slope * x_inliers + intercept
        ss_res = float(np.sum((y_inliers - y_hat) ** 2))
        ss_tot = (
            float(np.sum((y_inliers - y_inliers.mean()) ** 2))
            if y_inliers.size
            else 0.0
        )
        r2 = _stable_r2(ss_res, ss_tot)
        return slope, intercept, r2
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return _fit_line(x, y)


def _znorm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a
    mu = float(np.mean(a))
    sd = float(np.std(a))
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(a, dtype=float)
    return (a - mu) / sd


def _paa(a: np.ndarray, m: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    n = a.size
    if n == 0 or m <= 0:
        return np.asarray([], dtype=float)
    if n == m:
        return a.copy()
    idx = np.linspace(0, n, num=m + 1, dtype=float)
    starts = idx[:-1].astype(int)
    ends = idx[1:].astype(int)
    ends = np.maximum(ends, starts + 1)
    ends = np.minimum(ends, n)
    prefix = np.concatenate(([0.0], np.cumsum(a, dtype=float)))
    sums = prefix[ends] - prefix[starts]
    counts = np.maximum(1, ends - starts)
    return sums / counts


def _template_hs(L: int, inverse: bool = False) -> np.ndarray:
    """Simple H&S template over L points (z-norm target)."""
    L = max(20, int(L))
    x = np.linspace(0, 1, L)
    # Shoulders around 0.7, head at 1.0 (or inverted)
    y = 0.0 * x
    y += 0.7 * np.exp(-((x - 0.25) ** 2) / 0.01)
    y += 1.0 * np.exp(-((x - 0.5) ** 2) / 0.008)
    y += 0.7 * np.exp(-((x - 0.75) ** 2) / 0.01)
    if inverse:
        y = -y
    return _znorm(y)


def _template_hs_variants(L: int, inverse: bool = False) -> Tuple[np.ndarray, ...]:
    L = max(20, int(L))
    x = np.linspace(0, 1, L)
    variants = (
        ((0.25, 0.7, 0.01), (0.50, 1.0, 0.008), (0.75, 0.7, 0.01)),
        ((0.23, 0.65, 0.012), (0.50, 1.0, 0.010), (0.77, 0.72, 0.012)),
        ((0.28, 0.72, 0.014), (0.52, 1.0, 0.010), (0.74, 0.66, 0.014)),
    )
    out: List[np.ndarray] = []
    for peaks in variants:
        y = 0.0 * x
        for center, height, variance in peaks:
            y += float(height) * np.exp(-((x - float(center)) ** 2) / max(1e-6, float(variance)))
        if inverse:
            y = -y
        out.append(_znorm(y))
    return tuple(out)


def _detect_pivots_close(
    close: np.ndarray,
    cfg: ClassicDetectorConfig,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of peak and trough indices using close or optional high/low arrays."""
    return detect_pivots(close, cfg, high=high, low=low)


def _last_touch_indexes(bound_y: np.ndarray, idxs: np.ndarray, y: np.ndarray, tol: float) -> List[int]:
    out: List[int] = []
    for i in idxs.tolist():
        if i < 0 or i >= y.size:
            continue
        if abs(bound_y[i] - y[i]) <= tol:
            out.append(int(i))
    return out


def _build_time_array(df: pd.DataFrame) -> np.ndarray:
    t = df.get('time')
    if t is None:
        return np.asarray([], dtype=float)
    try:
        arr = to_float_np(t)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.asarray([], dtype=float)
    return arr


def _tol_abs_from_close(close: np.ndarray, tol_pct: float) -> float:
    """Absolute tolerance based on median close and percent threshold."""
    med = float(np.median(close)) if close.size else 0.0
    return abs(med) * (float(tol_pct) / 100.0)


def _boundary_tol_abs(boundary: float, tol_abs: float, tol_pct: Optional[float] = None) -> float:
    tol_floor = abs(float(tol_abs))
    if tol_pct is None:
        return tol_floor
    pct = abs(float(tol_pct)) / 100.0
    if pct <= 0.0 or not np.isfinite(boundary) or abs(boundary) <= 1e-12:
        return tol_floor
    return abs(boundary) * pct


def _result(name: str, status: str, confidence: float,
            start_index: int, end_index: int,
            t: np.ndarray, details: Dict[str, Any]) -> ClassicPatternResult:
    return ClassicPatternResult(
        name=name,
        status=status,
        confidence=float(confidence),
        start_index=int(start_index),
        end_index=int(end_index),
        start_time=PatternResultBase.resolve_time(t, int(start_index)),
        end_time=PatternResultBase.resolve_time(t, int(end_index)),
        details=details,
    )


def _fit_lines_and_arrays(
    ih: np.ndarray,
    il: np.ndarray,
    c: np.ndarray,
    n: int,
    cfg: ClassicDetectorConfig,
    *,
    upper_source: Optional[np.ndarray] = None,
    lower_source: Optional[np.ndarray] = None,
):
    upper_values = c if upper_source is None else np.asarray(upper_source, dtype=float)
    lower_values = c if lower_source is None else np.asarray(lower_source, dtype=float)
    fit_func = _fit_line_robust if bool(cfg.use_robust_fit) else _fit_line
    if fit_func is _fit_line_robust:
        sh, bh, r2h = fit_func(ih.astype(float), upper_values[ih], cfg)
        sl, bl, r2l = fit_func(il.astype(float), lower_values[il], cfg)
    else:
        sh, bh, r2h = fit_func(ih.astype(float), upper_values[ih])
        sl, bl, r2l = fit_func(il.astype(float), lower_values[il])
    x = np.arange(n, dtype=float)
    upper = sh * x + bh
    lower = sl * x + bl
    return sh, bh, r2h, sl, bl, r2l, upper, lower


def _boundaries_are_ordered(
    upper: np.ndarray,
    lower: np.ndarray,
    *,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    min_gap: float = 0.0,
) -> bool:
    up = np.asarray(upper, dtype=float)
    lo = np.asarray(lower, dtype=float)
    n = min(up.size, lo.size)
    if n <= 0:
        return False
    start = max(0, int(start_idx))
    stop = n if end_idx is None else min(n, int(end_idx) + 1)
    if stop <= start:
        return False
    span = up[start:stop] - lo[start:stop]
    if span.size == 0 or not np.all(np.isfinite(span)):
        return False
    return bool(np.all(span > max(1e-9, float(min_gap))))


def _count_touches(upper: np.ndarray, lower: np.ndarray,
                   peak_idxs: np.ndarray, trough_idxs: np.ndarray,
                   c: np.ndarray, tol_abs: float,
                   *,
                   upper_source: Optional[np.ndarray] = None,
                   lower_source: Optional[np.ndarray] = None) -> int:
    upper_values = c if upper_source is None else np.asarray(upper_source, dtype=float)
    lower_values = c if lower_source is None else np.asarray(lower_source, dtype=float)
    touches = 0
    touches += len(_last_touch_indexes(upper, peak_idxs, upper_values, tol_abs))
    touches += len(_last_touch_indexes(lower, trough_idxs, lower_values, tol_abs))
    return touches


def _count_recent_touches(
    series: np.ndarray,
    close: np.ndarray,
    tol_abs: float,
    lookback_bars: int,
) -> int:
    if series.size == 0 or close.size == 0:
        return 0
    n = min(series.size, close.size)
    lb = max(1, int(lookback_bars))
    start = max(0, n - lb)
    recent_s = series[start:n]
    recent_c = close[start:n]
    if recent_s.size == 0 or recent_c.size == 0:
        return 0
    return int(np.sum(np.abs(recent_s - recent_c) <= tol_abs))


def _is_converging(
    upper: np.ndarray,
    lower: np.ndarray,
    k: int,
    n: int,
    cfg: ClassicDetectorConfig,
) -> bool:
    """Heuristic to detect converging lines over recent window vs past window."""
    span = upper - lower
    last = max(5, int(k))
    recent = float(np.mean(span[-last:])) if span.size >= last else float(np.mean(span))
    past_win = max(20, 2 * int(k))
    prev_win = span[-past_win:-last] if n > past_win else span[:max(1, span.size // 2)]
    past = float(np.mean(prev_win)) if prev_win.size > 0 else recent * float(cfg.convergence_fallback_scale)
    return bool(recent < past)


def _find_recent_breakout(
    close: np.ndarray,
    *,
    upper: Optional[np.ndarray] = None,
    lower: Optional[np.ndarray] = None,
    tol_abs: float = 0.0,
    tol_pct: Optional[float] = None,
    lookback_bars: int = 8,
) -> Tuple[Optional[str], Optional[int]]:
    n = int(close.size)
    if n <= 0:
        return None, None
    lb = max(1, int(lookback_bars))
    start = max(0, n - lb)
    last_dir: Optional[str] = None
    last_idx: Optional[int] = None
    for i in range(start, n):
        px = float(close[i])
        breach_up = False
        breach_down = False
        if upper is not None and i < int(upper.size):
            up = float(upper[i])
            tol = _boundary_tol_abs(up, tol_abs, tol_pct)
            if np.isfinite(up) and px > (up + tol):
                breach_up = True
        if lower is not None and i < int(lower.size):
            lo = float(lower[i])
            tol = _boundary_tol_abs(lo, tol_abs, tol_pct)
            if np.isfinite(lo) and px < (lo - tol):
                breach_down = True
        if breach_up and breach_down:
            continue
        if breach_up:
            last_dir = "up"
            last_idx = int(i)
        elif breach_down:
            last_dir = "down"
            last_idx = int(i)
    return last_dir, last_idx


def _find_forward_level_breakout(
    close: np.ndarray,
    start_idx: int,
    level: float,
    direction: str,
    lookahead: int,
    tol_abs: float,
    tol_pct: Optional[float] = None,
) -> Optional[int]:
    n = int(close.size)
    if n <= 0:
        return None
    lo = max(0, int(start_idx) + 1)
    hi = min(n, lo + max(1, int(lookahead)))
    tol = _boundary_tol_abs(float(level), tol_abs, tol_pct)
    for i in range(lo, hi):
        px = float(close[i])
        if direction == "up" and px > (float(level) + tol):
            return int(i)
        if direction == "down" and px < (float(level) - tol):
            return int(i)
    return None


def _collect_calibration_points(cal_map: Any, name: str) -> List[Tuple[float, float]]:
    points_src: Any = cal_map
    if isinstance(cal_map, dict):
        key = str(name or "").strip().lower()
        if key in cal_map and isinstance(cal_map.get(key), dict):
            points_src = cal_map.get(key)
        elif "default" in cal_map and isinstance(cal_map.get("default"), dict):
            points_src = cal_map.get("default")
    points: List[Tuple[float, float]] = []
    if isinstance(points_src, dict):
        for k, v in points_src.items():
            try:
                x = float(k)
                y = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(x) and np.isfinite(y):
                points.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))
    elif isinstance(points_src, list):
        for item in points_src:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                x = float(item[0])
                y = float(item[1])
            except (TypeError, ValueError):
                continue
            if np.isfinite(x) and np.isfinite(y):
                points.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))
    points.sort(key=lambda p: p[0])
    return points


def _calibrate_confidence(raw: float, name: str, cfg: ClassicDetectorConfig) -> float:
    conf = float(max(0.0, min(1.0, raw)))
    if not bool(getattr(cfg, "calibrate_confidence", False)):
        return conf
    points = _collect_calibration_points(getattr(cfg, "confidence_calibration_map", {}), name)
    if len(points) < 2:
        return conf

    if conf <= points[0][0]:
        cal = points[0][1]
    elif conf >= points[-1][0]:
        cal = points[-1][1]
    else:
        cal = conf
        for i in range(1, len(points)):
            x0, y0 = points[i - 1]
            x1, y1 = points[i]
            if x0 <= conf <= x1:
                frac = 0.0 if abs(x1 - x0) <= 1e-12 else (conf - x0) / (x1 - x0)
                cal = y0 + frac * (y1 - y0)
                break
    blend = float(max(0.0, min(1.0, getattr(cfg, "confidence_calibration_blend", 1.0))))
    out = (1.0 - blend) * conf + blend * float(cal)
    return float(max(0.0, min(1.0, out)))

def _conf(touches: int, r2: float, geom_ok: float, cfg: ClassicDetectorConfig) -> float:
    raw_weights = np.asarray(
        [
            max(0.0, float(getattr(cfg, "touch_weight", 0.35))),
            max(0.0, float(getattr(cfg, "r2_weight", 0.35))),
            max(0.0, float(getattr(cfg, "geometry_weight", 0.30))),
        ],
        dtype=float,
    )
    total = float(np.sum(raw_weights))
    if total <= 1e-12:
        raw_weights = np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    else:
        raw_weights = raw_weights / total
    # Soft-saturate touch quality so meeting min_touches is not already a full score.
    # 1.0 requires roughly 2x min_touches (or more), rewarding extra confirmations.
    min_touches = max(1, int(cfg.min_touches))
    touch_score = min(1.0, float(touches) / float(2 * min_touches))
    scores = np.asarray(
        [
            touch_score,
            max(0.0, min(1.0, float(r2))),
            max(0.0, min(1.0, float(geom_ok))),
        ],
        dtype=float,
    )
    return float(min(1.0, max(0.0, float(np.dot(scores, raw_weights)))))


def _apply_breakout_confidence_bonus(confidence: float, cfg: ClassicDetectorConfig) -> float:
    bonus = max(0.0, float(getattr(cfg, "breakout_confidence_bonus", 0.08)))
    return float(min(1.0, max(0.0, float(confidence)) + bonus))


def _robust_level_center(values: np.ndarray, cfg: ClassicDetectorConfig) -> Optional[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return None
    if arr.size < 3:
        return float(np.median(arr))
    median = float(np.median(arr))
    deviations = np.abs(arr - median)
    mad = float(np.median(deviations))
    cutoff = float(max(1.0, getattr(cfg, "rectangle_outlier_zscore", 3.5)))
    filtered = arr
    if mad > 1e-9:
        robust_z = deviations / max(1e-9, 1.4826 * mad)
        kept = arr[robust_z <= cutoff]
        if kept.size >= 2:
            filtered = kept
    else:
        q1, q3 = np.quantile(arr, [0.25, 0.75])
        iqr = float(q3 - q1)
        if iqr > 1e-9:
            lo = float(q1 - 1.5 * iqr)
            hi = float(q3 + 1.5 * iqr)
            kept = arr[(arr >= lo) & (arr <= hi)]
            if kept.size >= 2:
                filtered = kept
    return float(np.median(filtered))
