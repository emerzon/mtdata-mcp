"""
Simplification helpers extracted for reuse across server tools.

Contains target-point selection utilities and core selection algorithms.
"""
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ruptures as rpt

from ..shared.constants import (
    SIMPLIFY_DEFAULT_MODE,
    SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT,
)

try:
    from tsdownsample import MinMaxLTTBDownsampler
except Exception:
    MinMaxLTTBDownsampler = None


def _normalize_simplify_spec(
    simplify: Optional[Dict[str, Any]],
    *,
    limit: int,
    fallback_rows: int,
) -> Optional[Dict[str, Any]]:
    """Inject the documented 10%-of-request default when no size is given."""
    if simplify is None:
        return None

    simplify_eff = dict(simplify)
    simplify_eff["mode"] = str(simplify_eff.get("mode", SIMPLIFY_DEFAULT_MODE)).lower().strip()
    has_points = any(
        key in simplify_eff and simplify_eff[key] is not None
        for key in ("points", "target_points", "max_points", "ratio")
    )
    if has_points:
        return simplify_eff

    try:
        default_pts = max(3, int(round(int(limit) * SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT)))
    except Exception:
        default_pts = max(
            3, int(round(fallback_rows * SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT))
        )
    simplify_eff["points"] = default_pts
    return simplify_eff


def _choose_simplify_points(total: int, spec: Dict[str, Any]) -> int:
    """Determine target number of points from a simplify spec.

    Supports keys: 'points', 'max_points', 'target_points', or 'ratio' (0..1).
    Enforces bounds [3, total]. Returns total if no explicit size is present.
    """
    try:
        if not spec:
            return total
        n = None
        for k in ("points", "max_points", "target_points"):
            if k in spec and spec[k] is not None:
                try:
                    n = int(spec[k])
                    break
                except Exception:
                    pass
        if n is None and "ratio" in spec and spec["ratio"] is not None:
            try:
                r = float(spec["ratio"])
                if 0 < r < 1:
                    n = int(max(3, round(total * r)))
            except Exception:
                pass
        if n is None:
            return total
        n = max(3, min(int(n), total))
        return n
    except Exception:
        return total


_LTTB_DOWNSAMPLER = MinMaxLTTBDownsampler() if MinMaxLTTBDownsampler is not None else None


def _fallback_lttb_indices(
    x: np.ndarray, y: np.ndarray, n_out: int
) -> List[int]:
    """Pure-Python Largest-Triangle-Three-Buckets implementation."""
    m = int(len(y))
    if n_out >= m:
        return list(range(m))
    if n_out <= 2 or m <= 2:
        return [0, max(0, m - 1)]
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    every = float(m - 2) / float(n_out - 2)
    selected = [0]
    anchor = 0
    for bucket_index in range(n_out - 2):
        avg_start = int(np.floor((bucket_index + 1) * every)) + 1
        avg_end = int(np.floor((bucket_index + 2) * every)) + 1
        avg_end = min(avg_end, m)
        if avg_start >= avg_end:
            avg_x, avg_y = float(xx[-1]), float(yy[-1])
        else:
            avg_x = float(np.mean(xx[avg_start:avg_end]))
            avg_y = float(np.mean(yy[avg_start:avg_end]))

        range_start = int(np.floor(bucket_index * every)) + 1
        range_end = int(np.floor((bucket_index + 1) * every)) + 1
        range_end = min(range_end, m - 1)
        if range_start >= range_end:
            range_end = min(range_start + 1, m - 1)

        ax, ay = float(xx[anchor]), float(yy[anchor])
        candidate = range_start
        max_area = -1.0
        for index in range(range_start, range_end):
            area = abs(
                (ax - avg_x) * (float(yy[index]) - ay)
                - (ax - float(xx[index])) * (avg_y - ay)
            )
            if area > max_area:
                max_area = area
                candidate = index
        selected.append(candidate)
        anchor = candidate
    selected.append(m - 1)
    return _finalize_indices(m, selected)


def _finalize_indices(n: int, idxs: List[int]) -> List[int]:
    """Ensure first/last indices exist and output is unique/increasing."""
    if n <= 0:
        return []
    if not idxs:
        return list(range(n))
    out = sorted(set(int(i) for i in idxs if 0 <= int(i) < n))
    if not out:
        return [0, n - 1] if n > 1 else [0]
    if out[0] != 0:
        out.insert(0, 0)
    if out[-1] != n - 1:
        out.append(n - 1)
    return out


def _segment_endpoints_to_indices(n: int, bkps: List[int]) -> List[int]:
    """Convert ruptures breakpoints (segment end positions) into row indices."""
    idxs = [0]
    for b in bkps:
        bi = int(b)
        if 0 < bi < n:
            idxs.append(bi - 1)
    idxs.append(n - 1)
    return _finalize_indices(n, idxs)


def _n_bkps_from_segments_points(n: int, segments: Optional[int], points: Optional[int]) -> Optional[int]:
    seg = segments
    if seg is None and points is not None:
        try:
            seg = max(1, int(points) - 1)
        except Exception:
            seg = None
    if seg is None:
        return None
    seg_i = max(1, min(int(seg), n - 1))
    return max(0, seg_i - 1)


def _lttb_select_indices(x: List[float], y: List[float], n_out: int) -> List[int]:
    """Downsampling using tsdownsample when available, else a Python fallback."""
    m = len(x)
    if n_out >= m:
        return list(range(m))
    if n_out <= 2 or m <= 2:
        return [0, max(0, m - 1)]
    yy = np.asarray(y, dtype=float)
    if _LTTB_DOWNSAMPLER is None:
        return _fallback_lttb_indices(
            np.asarray(x, dtype=float), yy, int(n_out)
        )
    xx = np.asarray(x, dtype=float)
    idxs = _LTTB_DOWNSAMPLER.downsample(xx, yy, n_out=int(n_out))
    return _finalize_indices(m, np.asarray(idxs, dtype=int).tolist())


def _point_line_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Vertical distance from P to line y(x) through (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    if dx == 0.0:
        return abs(px - x1)
    m = (y2 - y1) / dx
    y_on_line = y1 + m * (px - x1)
    return abs(py - y_on_line)


def _rdp_keep_mask(x: List[float], y: List[float], epsilon: float) -> np.ndarray:
    """Iterative Douglas-Peucker mask to avoid an external rdp dependency."""
    n = len(x)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    keep[0] = True
    keep[-1] = True

    stack: List[Tuple[int, int]] = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue

        x0, y0 = x[i0], y[i0]
        x1, y1 = x[i1], y[i1]
        split_idx = -1
        split_dist = -1.0

        for i in range(i0 + 1, i1):
            dist = _point_line_distance(x[i], y[i], x0, y0, x1, y1)
            if dist > split_dist:
                split_idx = i
                split_dist = dist

        if split_idx >= 0 and split_dist > float(epsilon):
            keep[split_idx] = True
            stack.append((i0, split_idx))
            stack.append((split_idx, i1))

    return keep


def _rdp_select_indices(x: List[float], y: List[float], epsilon: float) -> List[int]:
    """Douglas-Peucker simplification returning kept indices."""
    n = len(x)
    if n <= 2 or epsilon <= 0:
        return list(range(n))
    mask = _rdp_keep_mask(list(map(float, x)), list(map(float, y)), float(epsilon))
    idxs = np.flatnonzero(np.asarray(mask, dtype=bool)).astype(int).tolist()
    return _finalize_indices(n, idxs)


def _max_line_error(x: List[float], y: List[float], i0: int, i1: int) -> float:
    if i1 <= i0 + 1:
        return 0.0
    x0, y0 = x[i0], y[i0]
    x1, y1 = x[i1], y[i1]
    m = 0.0
    for i in range(i0 + 1, i1):
        d = _point_line_distance(x[i], y[i], x0, y0, x1, y1)
        if d > m:
            m = d
    return m


def _pla_select_indices(
    x: List[float],
    y: List[float],
    max_error: Optional[float] = None,
    segments: Optional[int] = None,
    points: Optional[int] = None,
) -> List[int]:
    """Piecewise-linear selection using ruptures."""
    n = len(x)
    if n <= 2:
        return list(range(n))
    if max_error is not None and float(max_error) > 0:
        return _rdp_select_indices(x, y, float(max_error))
    signal = np.column_stack([np.asarray(y, dtype=float), np.asarray(x, dtype=float)])
    algo = rpt.BottomUp(model="linear", min_size=2, jump=1).fit(signal)
    n_bkps = _n_bkps_from_segments_points(n, segments, points)
    if n_bkps is not None:
        bkps = algo.predict(n_bkps=int(n_bkps))
    else:
        return list(range(n))
    return _segment_endpoints_to_indices(n, [int(b) for b in bkps])


def _apca_select_indices(
    y: List[float],
    max_error: Optional[float] = None,
    segments: Optional[int] = None,
    points: Optional[int] = None,
) -> List[int]:
    """Piecewise-constant selection using ruptures."""
    n = len(y)
    if n <= 2:
        return list(range(n))
    if max_error is not None and float(max_error) > 0:
        tolerance = float(max_error)
        values = np.asarray(y, dtype=float)
        starts = [0]
        start = 0
        while start < n - 1:
            end = start
            while end + 1 < n:
                candidate = values[start : end + 2]
                mean = float(np.mean(candidate))
                if float(np.max(np.abs(candidate - mean))) > tolerance:
                    break
                end += 1
            next_start = end + 1
            if next_start >= n:
                break
            starts.append(next_start)
            start = next_start
        return _finalize_indices(n, starts)
    signal = np.asarray(y, dtype=float).reshape(-1, 1)
    algo = rpt.BottomUp(model="l2", min_size=2, jump=1).fit(signal)
    n_bkps = _n_bkps_from_segments_points(n, segments, points)
    if n_bkps is not None:
        bkps = algo.predict(n_bkps=int(n_bkps))
    else:
        return list(range(n))
    return _segment_endpoints_to_indices(n, [int(b) for b in bkps])


def _rdp_autotune_epsilon(x: List[float], y: List[float], target_points: int, max_iter: int = 24) -> Tuple[List[int], float]:
    """Binary-search epsilon for rdp to keep ~target_points."""
    n = len(x)
    target = max(3, min(int(target_points), n))
    if target >= n:
        return list(range(n)), 0.0
    y_arr = np.asarray(y, dtype=float)
    hi = float(np.ptp(y_arr)) if y_arr.size else 1.0
    if not np.isfinite(hi) or hi <= 0:
        hi = 1.0
    lo = 0.0
    best = _rdp_select_indices(x, y, lo)
    best_eps = lo
    best_diff = abs(len(best) - target)
    for _ in range(max(1, int(max_iter))):
        mid = (lo + hi) / 2.0
        idxs = _rdp_select_indices(x, y, mid)
        diff = abs(len(idxs) - target)
        if diff < best_diff:
            best, best_eps, best_diff = idxs, mid, diff
        if len(idxs) > target:
            lo = mid
        elif len(idxs) < target:
            hi = mid
        else:
            best, best_eps = idxs, mid
            break
        if abs(hi - lo) <= 1e-12:
            break
    return best, float(best_eps)


def _observed_pla_max_error(
    x: List[float], y: List[float], idxs: List[int]
) -> float:
    return max(
        (_max_line_error(x, y, left, right) for left, right in zip(idxs, idxs[1:])),
        default=0.0,
    )


def _observed_apca_max_error(y: List[float], idxs: List[int]) -> float:
    values = np.asarray(y, dtype=float)
    errors = []
    for position, start in enumerate(idxs):
        stop = idxs[position + 1] if position + 1 < len(idxs) else len(values)
        segment = values[int(start) : int(stop)]
        if segment.size:
            errors.append(float(np.max(np.abs(segment - float(np.mean(segment))))))
    return max(errors, default=0.0)


def _pla_autotune_max_error(
    x: List[float], y: List[float], target_points: int
) -> Tuple[List[int], float]:
    """Select a fixed segment count and return its observed line error."""
    n = len(x)
    target = max(3, min(int(target_points), n))
    if target >= n:
        return list(range(n)), 0.0
    idxs = _pla_select_indices(
        x, y, max_error=None, segments=max(1, target - 1), points=None
    )
    return idxs, _observed_pla_max_error(x, y, idxs)


def _apca_autotune_max_error(
    y: List[float], target_points: int
) -> Tuple[List[int], float]:
    """Select a fixed segment count and return its observed constant error."""
    n = len(y)
    target = max(3, min(int(target_points), n))
    if target >= n:
        return list(range(n)), 0.0
    idxs = _apca_select_indices(
        y, max_error=None, segments=max(1, target - 1), points=None
    )
    return idxs, _observed_apca_max_error(y, idxs)


def _select_indices_for_timeseries(x: List[float], y: List[float], spec: Optional[Dict[str, Any]]) -> Tuple[List[int], str, Dict[str, Any]]:  # noqa: C901
    """Select representative indices according to simplify spec.

    Returns (indices, method_used, params_meta).
    """
    meta: Dict[str, Any] = {}
    if not spec:
        return list(range(len(x))), "none", meta
    method = str(spec.get("method", "lttb")).lower().strip()
    if method in ("lttb", "default"):
        n_out = _choose_simplify_points(len(x), spec)
        idxs = _lttb_select_indices(x, y, n_out)
        meta.update({"points": n_out})
        if _LTTB_DOWNSAMPLER is None:
            meta["implementation"] = "python-fallback"
        return idxs, "lttb", meta
    if method == "rdp":
        eps = spec.get("epsilon", None) or spec.get("tolerance", None) or spec.get("eps", None)
        try:
            eps_val = float(eps) if eps is not None else None
        except Exception:
            eps_val = None
        if eps_val is not None and eps_val > 0:
            idxs = _rdp_select_indices(x, y, eps_val)
            meta.update({"epsilon": eps_val})
            return idxs, "rdp", meta
        target = spec.get("points") or spec.get("target_points") or spec.get("max_points") or None
        if target is None and spec.get("ratio") is not None:
            try:
                r = float(spec.get("ratio"))
                if 0 < r < 1:
                    target = max(3, int(round(len(x) * r)))
            except Exception:
                pass
        try:
            target_n = int(target) if target is not None else None
        except Exception:
            target_n = None
        if target_n is not None and target_n < len(x):
            idxs, eps_used = _rdp_autotune_epsilon(x, y, target_n)
            meta.update({"epsilon": eps_used, "points": len(idxs), "auto_tuned": True})
            return idxs, "rdp", meta
        n_out = _choose_simplify_points(len(x), spec)
        idxs = _lttb_select_indices(x, y, n_out)
        meta.update({"points": n_out, "fallback": "rdp->lttb"})
        if _LTTB_DOWNSAMPLER is None:
            meta["implementation"] = "python-fallback"
        return idxs, "lttb", meta
    if method == "pla":
        max_error = spec.get("max_error", None)
        try:
            me = float(max_error) if max_error is not None else None
        except Exception:
            me = None
        segments = spec.get("segments", None)
        points = spec.get("points", None) or spec.get("target_points", None) or spec.get("max_points", None)
        if me is not None and me > 0:
            idxs = _pla_select_indices(x, y, me, None, None)
            meta.update(
                {
                    "max_error": me,
                    "observed_max_error": _observed_pla_max_error(x, y, idxs),
                }
            )
            return idxs, "pla", meta
        if segments is not None:
            idxs = _pla_select_indices(x, y, None, segments, None)
            meta.update({"segments": segments})
            return idxs, "pla", meta
        try:
            p = int(points) if points is not None else None
        except Exception:
            p = None
        if p is not None and p < len(x):
            idxs, me_used = _pla_autotune_max_error(x, y, p)
            meta.update(
                {
                    "observed_max_error": me_used,
                    "points": len(idxs),
                    "selection_basis": "fixed_segments_from_target_points",
                }
            )
            return idxs, "pla", meta
        n_out = _choose_simplify_points(len(x), spec)
        idxs = _lttb_select_indices(x, y, n_out)
        meta.update({"points": n_out, "fallback": "pla->lttb"})
        if _LTTB_DOWNSAMPLER is None:
            meta["implementation"] = "python-fallback"
        return idxs, "lttb", meta
    if method == "apca":
        max_error = spec.get("max_error", None)
        try:
            me = float(max_error) if max_error is not None else None
        except Exception:
            me = None
        segments = spec.get("segments", None)
        points = spec.get("points", None) or spec.get("target_points", None) or spec.get("max_points", None)
        if me is not None and me > 0:
            idxs = _apca_select_indices(y, me, None, None)
            meta.update(
                {
                    "max_error": me,
                    "observed_max_error": _observed_apca_max_error(y, idxs),
                }
            )
            return idxs, "apca", meta
        if segments is not None:
            idxs = _apca_select_indices(y, None, segments, None)
            meta.update({"segments": segments})
            return idxs, "apca", meta
        try:
            p = int(points) if points is not None else None
        except Exception:
            p = None
        if p is not None and p < len(x):
            idxs, me_used = _apca_autotune_max_error(y, p)
            meta.update(
                {
                    "observed_max_error": me_used,
                    "points": len(idxs),
                    "selection_basis": "fixed_segments_from_target_points",
                }
            )
            return idxs, "apca", meta
        n_out = _choose_simplify_points(len(x), spec)
        idxs = _lttb_select_indices(x, y, n_out)
        meta.update({"points": n_out, "fallback": "apca->lttb"})
        if _LTTB_DOWNSAMPLER is None:
            meta["implementation"] = "python-fallback"
        return idxs, "lttb", meta
    n_out = _choose_simplify_points(len(x), spec)
    idxs = _lttb_select_indices(x, y, n_out)
    meta.update({"points": n_out, "fallback": f"{method}->lttb"})
    if _LTTB_DOWNSAMPLER is None:
        meta["implementation"] = "python-fallback"
    return idxs, "lttb", meta


def _handle_select_mode(df: pd.DataFrame, headers: List[str], spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    original_count = len(df)
    if original_count <= 2:
        return df, None

    n_out = _choose_simplify_points(original_count, spec)
    if n_out >= original_count:
        return df, None

    series = None
    if 'close' in df.columns:
        series = df['close'].values
    elif len(headers) > 1:
        for h in headers:
            if h != 'time' and h in df.columns:
                try:
                    series = df[h].astype(float).values
                    break
                except Exception:
                    pass
    if series is None:
        return df, None

    epochs = df['__epoch'].values if '__epoch' in df.columns else np.arange(original_count)
    idxs, method, params = _select_indices_for_timeseries(epochs, series, spec)
    simplified_df = df.iloc[idxs].copy()
    meta: Dict[str, Any] = {
        'mode': 'select',
        'method': method,
        'original_rows': int(original_count),
        'returned_rows': int(len(simplified_df)),
        'points': int(len(simplified_df)),
    }
    if params:
        meta.update(params)
    return simplified_df, meta


def _first_non_null_value(series: pd.Series) -> Any:
    for value in series.tolist():
        if pd.notna(value):
            return value
    return series.iloc[0] if len(series) else None


def _last_non_null_value(series: pd.Series) -> Any:
    values = series.tolist()
    for value in reversed(values):
        if pd.notna(value):
            return value
    return series.iloc[-1] if len(series) else None


def _resolve_resample_bucket_seconds(df: pd.DataFrame, spec: Dict[str, Any]) -> Optional[int]:
    bucket_seconds = spec.get("bucket_seconds")
    if bucket_seconds is not None:
        try:
            return max(1, int(bucket_seconds))
        except Exception:
            return None

    # Only infer a relative bucket size when the caller supplied a target shape.
    if "__epoch" not in df.columns:
        return None
    if not any(spec.get(key) is not None for key in ("points", "target_points", "max_points", "ratio")):
        return None

    try:
        n_out = _choose_simplify_points(len(df), spec)
        t0 = float(df["__epoch"].iloc[0])
        t1 = float(df["__epoch"].iloc[-1])
        span = max(1.0, t1 - t0)
        return max(1, int(round(span / max(1, n_out))))
    except Exception:
        return None


def _aggregate_resample_segment(seg: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for col in columns:
        if col not in seg.columns:
            continue
        series = seg[col]
        if col == "time":
            row[col] = _first_non_null_value(series)
            continue
        if col == "__epoch":
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            row[col] = float(numeric.iloc[0]) if not numeric.empty else _first_non_null_value(series)
            continue
        if col == "open":
            row[col] = _first_non_null_value(series)
            continue
        if col in {"high", "low", "tick_volume", "real_volume", "volume"}:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                row[col] = _last_non_null_value(series)
            elif col == "high":
                row[col] = float(numeric.max())
            elif col == "low":
                row[col] = float(numeric.min())
            else:
                row[col] = float(numeric.sum())
            continue
        row[col] = _last_non_null_value(series)
    return row


def _aggregate_approximate_segment(
    seg: pd.DataFrame, columns: List[str]
) -> Dict[str, Any]:
    row = _aggregate_resample_segment(seg, columns)
    specialized = {
        "time",
        "__epoch",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "real_volume",
        "volume",
    }
    for col in columns:
        if col in specialized or col not in seg.columns:
            continue
        numeric = pd.to_numeric(seg[col], errors="coerce").dropna()
        if not numeric.empty:
            row[col] = float(numeric.mean())
    return row


def _handle_approximate_mode(
    df: pd.DataFrame, headers: List[str], spec: Dict[str, Any]
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    original_count = len(df)
    if original_count <= 2:
        return df, None
    series_col = "close" if "close" in df.columns else next(
        (
            name
            for name in headers
            if name not in {"time", "__epoch"}
            and name in df.columns
            and pd.api.types.is_numeric_dtype(df[name])
        ),
        None,
    )
    if series_col is None:
        return df, {"mode": "approximate", "error": "No numeric value column"}
    epochs = (
        pd.to_numeric(df["__epoch"], errors="coerce").to_numpy(dtype=float)
        if "__epoch" in df.columns
        else np.arange(original_count, dtype=float)
    )
    values = pd.to_numeric(df[series_col], errors="coerce").to_numpy(dtype=float)
    idxs, method, params = _select_indices_for_timeseries(
        epochs.tolist(), values.tolist(), spec
    )
    columns = list(df.columns)
    specialized = {
        "time",
        "__epoch",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "real_volume",
        "volume",
    }
    segment_mean_columns = [
        str(column)
        for column in columns
        if column not in specialized
        and pd.to_numeric(df[column], errors="coerce").notna().any()
    ]
    rows = []
    for position, start in enumerate(idxs):
        stop = idxs[position + 1] if position + 1 < len(idxs) else original_count
        if int(stop) <= int(start):
            continue
        rows.append(
            _aggregate_approximate_segment(
                df.iloc[int(start) : int(stop)], columns
            )
        )
    out = pd.DataFrame(rows, columns=columns)
    meta: Dict[str, Any] = {
        "mode": "approximate",
        "method": method,
        "value_col": series_col,
        "original_rows": int(original_count),
        "returned_rows": int(len(out)),
        "breakpoints": int(len(idxs)),
    }
    meta.update(params)
    if segment_mean_columns:
        meta["non_ohlc_numeric_aggregation"] = "segment_mean"
        meta["segment_mean_columns"] = segment_mean_columns
    return out.reset_index(drop=True), meta


def _handle_resample_mode(df: pd.DataFrame, headers: List[str], spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    rule = spec.get("rule") or spec.get("interval")
    bucket_seconds = _resolve_resample_bucket_seconds(df, spec)
    if not rule and bucket_seconds is None:
        return df, {"error": "Missing rule for resample"}
    try:
        if "__epoch" in df.columns:
            time_index = pd.to_datetime(pd.to_numeric(df["__epoch"], errors="coerce"), unit="s", utc=True)
        elif "time" in df.columns:
            time_index = pd.to_datetime(df["time"], errors="coerce")
        else:
            return df, {"error": "Resample requires a time or __epoch column"}

        working = df.copy()
        working["__time_index"] = time_index
        working = working[working["__time_index"].notna()].copy()
        if working.empty:
            return df, {"error": "Resample requires at least one valid timestamp"}

        source_columns = [col for col in df.columns if col != "__time_index"]
        rows: List[Dict[str, Any]] = []

        if rule:
            grouped = working.groupby(pd.Grouper(key="__time_index", freq=str(rule)), sort=True)
        else:
            if "__epoch" in working.columns:
                base_seconds = pd.to_numeric(working["__epoch"], errors="coerce")
            else:
                base_seconds = working["__time_index"].astype("int64") / 1_000_000_000.0
            base_seconds = pd.to_numeric(base_seconds, errors="coerce")
            working = working[base_seconds.notna()].copy()
            if working.empty:
                return df, {"error": "Resample requires at least one valid timestamp"}
            base_seconds = pd.to_numeric(base_seconds[base_seconds.notna()], errors="coerce")
            origin = float(base_seconds.iloc[0])
            bucket_keys = ((base_seconds - origin) // int(bucket_seconds)).astype(int)
            grouped = working.groupby(bucket_keys, sort=True)

        for bucket, seg in grouped:
            if seg.empty or pd.isna(bucket):
                continue
            rows.append(_aggregate_resample_segment(seg, source_columns))

        out_df = pd.DataFrame(rows)
        if rule:
            return out_df.reset_index(drop=True), {"mode": "resample", "rule": str(rule), "rows": int(len(out_df))}
        return out_df.reset_index(drop=True), {
            "mode": "resample",
            "bucket_seconds": int(bucket_seconds),
            "rows": int(len(out_df)),
        }
    except Exception as exc:
        return df, {"error": f"Resample failed: {exc}"}


def _handle_encode_mode(df: pd.DataFrame, headers: List[str], spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    value_col = str(spec.get('value_col') or '').strip()
    if not value_col or value_col not in df.columns:
        if 'close' in df.columns:
            value_col = 'close'
        else:
            value_col = next(
                (
                    h for h in headers
                    if h in df.columns and h != 'time' and pd.api.types.is_numeric_dtype(df[h])
                ),
                '',
            )
    if not value_col:
        return df, {'mode': 'encode', 'error': 'No numeric column available for encoding'}

    vals = pd.to_numeric(df[value_col], errors='coerce').to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 0:
        return df, {'mode': 'encode', 'error': 'No finite values to encode'}

    schema = str(spec.get('schema', 'delta')).lower().strip()
    if schema not in ('delta', 'envelope'):
        schema = 'delta'

    if schema == 'envelope':
        encoded = (
            f"start={float(vals[0]):.6g}|end={float(vals[-1]):.6g}|"
            f"min={float(np.min(vals)):.6g}|max={float(np.max(vals)):.6g}"
        )
    else:
        diffs = np.diff(vals, prepend=vals[0])
        scale = spec.get('scale', 'auto')
        scale_source = 'explicit'
        if scale in (None, '', 'auto'):
            nonzero = np.abs(diffs[np.isfinite(diffs) & (np.abs(diffs) > 1e-12)])
            scale_f = float(np.median(nonzero)) if nonzero.size else 1.0
            scale_source = 'median_nonzero_delta' if nonzero.size else 'constant_series_fallback'
        else:
            try:
                scale_f = float(scale)
            except Exception:
                scale_f = 1.0
                scale_source = 'invalid_scale_fallback'
        scale_f = scale_f if abs(scale_f) > 1e-12 else 1.0
        q = np.round(diffs / scale_f).astype(int)
        if bool(spec.get('as_chars', False)):
            zero_char = str(spec.get('zero_char', '0'))[:1] or '0'
            encoded = ''.join('+' if d > 0 else '-' if d < 0 else zero_char for d in q.tolist())
        else:
            encoded = ','.join(str(int(v)) for v in q.tolist())

    out_df = pd.DataFrame([{'encoding': encoded}])
    meta = {
        'mode': 'encode',
        'schema': schema,
        'value_col': value_col,
        'headers': ['encoding'],
        'original_rows': int(len(df)),
        'returned_rows': 1,
        'points': 1,
    }
    if schema == 'delta':
        meta['scale'] = float(scale_f)
        meta['scale_source'] = scale_source
        meta['degenerate'] = bool(np.all(q == 0))
    return out_df, meta


def _handle_segment_mode(df: pd.DataFrame, headers: List[str], spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    value_col = str(spec.get('value_col') or '').strip()
    if not value_col or value_col not in df.columns:
        value_col = 'close' if 'close' in df.columns else ''
    if not value_col:
        return _handle_select_mode(df, headers, spec)

    vals = pd.to_numeric(df[value_col], errors='coerce').to_numpy(dtype=float)
    if vals.size <= 2:
        return df, {'mode': 'segment', 'algo': 'zigzag', 'points': int(len(df))}

    try:
        threshold_pct = float(spec.get('threshold_pct', 0.005))
    except Exception:
        threshold_pct = 0.005
    threshold_pct = max(0.0, threshold_pct)
    if threshold_pct <= 0.0:
        return _handle_select_mode(df, headers, spec)

    idxs: List[int] = [0]
    anchor_val = float(vals[0])
    trend = 0
    for i in range(1, len(vals)):
        cur = float(vals[i])
        denom = max(abs(anchor_val), 1e-12)
        move = (cur - anchor_val) / denom
        if trend >= 0 and cur >= anchor_val:
            anchor_val = cur
            if idxs:
                idxs[-1] = i
            continue
        if trend <= 0 and cur <= anchor_val:
            anchor_val = cur
            if idxs:
                idxs[-1] = i
            continue
        if abs(move) >= threshold_pct:
            trend = 1 if move > 0 else -1
            idxs.append(i)
            anchor_val = cur

    if idxs[-1] != len(vals) - 1:
        idxs.append(len(vals) - 1)
    idxs = sorted(set(int(i) for i in idxs if 0 <= int(i) < len(df)))
    if len(idxs) < 2:
        idxs = [0, len(df) - 1]

    out_df = df.iloc[idxs].copy()
    meta = {
        'mode': 'segment',
        'algo': 'zigzag',
        'threshold_pct': float(threshold_pct),
        'value_col': value_col,
        'original_rows': int(len(df)),
        'returned_rows': int(len(out_df)),
        'points': int(len(out_df)),
    }
    return out_df, meta


def _handle_symbolic_mode(df: pd.DataFrame, headers: List[str], spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    value_col = str(spec.get('value_col') or '').strip()
    if not value_col or value_col not in df.columns:
        value_col = 'close' if 'close' in df.columns else ''
    if not value_col:
        return df, {'mode': 'symbolic', 'error': 'No numeric column available for symbolic mode'}

    vals = pd.to_numeric(df[value_col], errors='coerce').to_numpy(dtype=float)
    finite_mask = np.isfinite(vals)
    if not bool(np.any(finite_mask)):
        return df, {'mode': 'symbolic', 'error': 'No finite values to symbolize'}

    try:
        paa = int(spec.get('paa', 8))
    except Exception:
        paa = 8
    paa = max(1, min(paa, int(vals.size)))

    alphabet = str(spec.get('alphabet') or 'abcdefghijklmnopqrstuvwxyz')
    alphabet = ''.join(dict.fromkeys(ch for ch in alphabet if ch.strip()))
    if len(alphabet) < 2:
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
    bins_n = min(26, len(alphabet))
    alphabet = alphabet[:bins_n]

    x = vals.copy()
    if bool(spec.get('znorm', True)):
        mu = float(np.mean(x[finite_mask]))
        sigma = float(np.std(x[finite_mask]))
        if sigma > 1e-12:
            x = (x - mu) / sigma
        else:
            x = x - mu

    chunks = np.array_split(x, paa)
    paa_vals = np.array(
        [
            float(np.mean(c[np.isfinite(c)]))
            if bool(np.any(np.isfinite(c)))
            else 0.0
            for c in chunks
        ],
        dtype=float,
    )
    quantiles = np.linspace(0.0, 1.0, bins_n + 1)
    normal = NormalDist()
    edges = np.asarray(
        [
            float('-inf') if q <= 0.0
            else float('inf') if q >= 1.0
            else normal.inv_cdf(float(q))
            for q in quantiles
        ],
        dtype=float,
    )
    ids = np.searchsorted(edges[1:-1], paa_vals, side='right')
    symbols = ''.join(alphabet[int(i)] for i in ids.tolist())

    out_df = pd.DataFrame([{'symbolic': symbols}])
    meta = {
        'mode': 'symbolic',
        'schema': 'sax',
        'breakpoint_basis': 'standard_normal',
        'value_col': value_col,
        'paa': int(paa),
        'alphabet': alphabet,
        'headers': ['symbolic'],
        'original_rows': int(len(df)),
        'returned_rows': 1,
        'points': 1,
    }
    return out_df, meta


def _simplify_dataframe_rows_ext(
    df: pd.DataFrame,
    headers: List[str],
    simplify: Optional[Dict[str, Any]],
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Extended simplify dispatcher shared by core and service adapters."""
    if df.empty:
        return df, None

    spec = dict(simplify) if simplify else {}
    mode = str(spec.get('mode', SIMPLIFY_DEFAULT_MODE)).lower().strip() or SIMPLIFY_DEFAULT_MODE

    if mode == 'resample':
        return _handle_resample_mode(df, headers, spec)
    if mode == 'approximate':
        return _handle_approximate_mode(df, headers, spec)
    if mode == 'encode':
        return _handle_encode_mode(df, headers, spec)
    if mode == 'segment':
        return _handle_segment_mode(df, headers, spec)
    if mode == 'symbolic':
        return _handle_symbolic_mode(df, headers, spec)
    return _handle_select_mode(df, headers, spec)

