from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.spatial import cKDTree

try:
    import hnswlib as _HNSW  # type: ignore
except Exception:
    _HNSW = None  # optional ANN backend

from ..services.data_service import fetch_history_frame
from ..shared.constants import TIMEFRAME_MAP
from .denoise import (
    denoise_series as apply_denoise_series,
)
from .denoise import (
    get_denoise_methods_data as _get_denoise_methods_data,
)
from .denoise import (
    normalize_denoise_spec as _normalize_denoise_spec,
)
from .dimred import DimReducer as _DimReducer
from .dimred import create_reducer as _create_reducer
from .dtw import dtw_distance
from .utils import align_finite


@lru_cache(maxsize=1)
def _get_stumpy():
    try:
        import stumpy  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "matrix-profile search requires 'stumpy'; install the core dependencies"
        ) from exc
    return stumpy


@lru_cache(maxsize=1)
def _get_ts_soft_dtw():
    from tslearn.metrics import soft_dtw as _ts_soft_dtw  # type: ignore

    return _ts_soft_dtw


def _minmax_scale_row(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    rng = float(mx - mn)
    if not np.isfinite(rng) or rng <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - mn) / rng
    return y.astype(np.float32, copy=False)


def _mass_distance_profile(query: np.ndarray, series: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Compute z-normalized sliding distances using stumpy MASS.

    Returns an array of length len(series) - len(query) + 1. Non-finite windows are mapped to inf.
    """
    q = np.asarray(query, dtype=float).ravel()
    s = np.asarray(series, dtype=float).ravel()
    m = q.size
    n = s.size
    if m == 0 or n < m:
        return np.array([], dtype=float)

    if not np.isfinite(q).all() or not np.isfinite(s).all():
        return np.full(max(n - m + 1, 0), np.inf, dtype=float)
    q_std = float(np.std(q))
    if q_std <= eps:
        return np.full(max(n - m + 1, 0), np.inf, dtype=float)
    profile = np.asarray(_get_stumpy().mass(q, s), dtype=float)
    profile[~np.isfinite(profile)] = np.inf
    return profile


@dataclass
class _SeriesStore:
    symbol: str
    time_epoch: np.ndarray  # float64 UTC epoch seconds, ascending
    close: np.ndarray       # float64, ascending


@dataclass
class _SeriesPrepareInfo:
    source: str
    base_col: str
    bars_raw: int
    bars_used: int
    denoise_requested: bool = False
    denoise_applied: bool = False
    denoise_method: Optional[str] = None
    denoise_error: Optional[str] = None
    provided_history: bool = False


class PatternIndex:
    """In-memory sliding-window index for pattern similarity search.

    - Builds per-window min-max normalized vectors of length `window_size` from
      MT5 close prices.
    - Uses cKDTree for fast L2 nearest neighbor search.
    - Keeps mappings to reconstruct symbol, dates, and original values (+future).
    """

    def __init__(
        self,
        timeframe: str,
        window_size: int,
        future_size: int,
        symbols: List[str],
        tree: Any,
        X: np.ndarray,
        start_end_idx: np.ndarray,
        labels: np.ndarray,
        series: List[_SeriesStore],
        scale: str = "minmax",
        metric: str = "euclidean",
        dimred_method: Optional[str] = None,
        dimred_params: Optional[Dict[str, Any]] = None,
        reducer: Optional[_DimReducer] = None,
        engine: str = "ckdtree",
        max_bars_per_symbol: int = 5000,
        build_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.timeframe = timeframe
        self.window_size = int(window_size)
        self.future_size = int(future_size)
        self.symbols = list(symbols)
        self.tree = tree
        self.X = X
        self.start_end_idx = start_end_idx  # shape (N,2)
        self.labels = labels                 # shape (N,)
        self._series = series               # list aligned with label indices
        self.scale = (scale or "minmax").lower()
        self.metric = (metric or "euclidean").lower()
        self.dimred_method = (dimred_method or "none").lower()
        self.dimred_params = dict(dimred_params or {})
        self._reducer = reducer  # type: ignore
        self.engine = (engine or "ckdtree").lower()
        self.max_bars_per_symbol = int(max_bars_per_symbol)
        self.build_metadata = dict(build_metadata or {})

    def search(self, anchor_values: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Query by a raw (unscaled) anchor window. Returns (indices, distances)."""
        v = np.asarray(anchor_values, dtype=float).ravel()
        if v.size != self.window_size:
            raise ValueError(f"anchor_values must be length {self.window_size}")
        if self.engine in ("matrix_profile", "mass"):
            return self._profile_search(v, top_k=top_k)
        q = v.astype(float)
        # Scale
        q = _apply_scale_vector(q, self.scale)
        # Dimensionality reduction
        if self._reducer is not None:
            if not self._reducer.supports_transform():
                raise RuntimeError(f"Reducer '{self.dimred_method}' does not support transforming new samples")
            q = np.asarray(self._reducer.transform(q.reshape(1, -1)), dtype=np.float32).ravel()
        # Metric post-process
        q = _apply_metric_vector(q, self.metric)
        k = min(int(top_k), len(self.X))
        if self.engine == "hnsw":
            # hnswlib with 'l2' space returns squared L2 distances; take sqrt to match cKDTree
            labels, distances = self.tree.knn_query(q.reshape(1, -1).astype(np.float32), k=k)
            idxs = labels[0].astype(int)
            dists = np.sqrt(distances[0].astype(float))
            return idxs, dists
        else:
            dists, idxs = self.tree.query(q, k=k)
            # Ensure 1D arrays
            if np.ndim(idxs) == 0:
                idxs = np.asarray([int(idxs)])
                dists = np.asarray([float(dists)])
            return idxs.astype(int), dists.astype(float)

    def _profile_search(self, anchor_values: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sliding search using matrix profile / MASS style distances."""
        if self.scale not in ("zscore",):
            raise ValueError("matrix_profile/mass engines require scale='zscore'")
        if self.metric not in ("euclidean", "l2"):
            raise ValueError("matrix_profile/mass engines require metric='euclidean'")

        q = np.asarray(anchor_values, dtype=float).ravel()
        m = q.size
        idxs_all: List[int] = []
        dists_all: List[float] = []
        offset = 0

        for ser in self._series:
            n = ser.close.size
            limit = n - (self.window_size + self.future_size) + 1
            if limit <= 0:
                continue
            series_slice = ser.close[: limit + self.window_size + max(self.future_size - 1, 0)]
            if series_slice.size < m:
                offset += max(limit, 0)
                continue

            if self.engine == "matrix_profile":
                # AB-join: distances from every subsequence in series_slice to the query subsequence
                mp = _get_stumpy().stump(
                    series_slice.astype(float),
                    m,
                    T_B=q.astype(float),
                    ignore_trivial=False,
                )
                profile = np.asarray(mp[:, 0], dtype=float)
            else:
                profile = _mass_distance_profile(q, series_slice)

            if profile.size > limit:
                profile = profile[:limit]
            for i, d in enumerate(profile.tolist()):
                idxs_all.append(offset + i)
                dists_all.append(float(d))
            offset += max(limit, 0)

        if not idxs_all:
            return np.array([], dtype=int), np.array([], dtype=float)

        order = np.argsort(np.asarray(dists_all, dtype=float))
        k = min(int(top_k), order.size)
        sel = order[:k]
        return np.asarray([idxs_all[i] for i in sel], dtype=int), np.asarray([dists_all[i] for i in sel], dtype=float)

    def get_match_symbol(self, index: int) -> str:
        lbl = int(self.labels[int(index)])
        return self._series[lbl].symbol

    def get_match_times(self, index: int, include_future: bool = True) -> np.ndarray:
        s, e = self.start_end_idx[int(index)]
        ser = self._series[int(self.labels[int(index)])]
        if include_future and self.future_size > 0:
            end = min(int(e + self.future_size), len(ser.time_epoch) - 1)
            start = max(0, int(s))
        else:
            start, end = int(s), int(e)
        return ser.time_epoch[start : end + 1]

    def get_match_values(self, index: int, include_future: bool = True) -> np.ndarray:
        s, e = self.start_end_idx[int(index)]
        ser = self._series[int(self.labels[int(index)])]
        if include_future and self.future_size > 0:
            end = min(int(e + self.future_size), len(ser.close) - 1)
            start = max(0, int(s))
        else:
            start, end = int(s), int(e)
        return ser.close[start : end + 1]

    def _scaled_window(self, vals: np.ndarray) -> np.ndarray:
        # Apply the same scaling as index vectors for fair comparison
        return _apply_scale_vector(np.asarray(vals, dtype=float), self.scale)

    def _ncc_max(self, a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
        """Compute maximum normalized cross-correlation within +/- max_lag.
        a and b are same-length 1D arrays (window only).
        """
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        n = int(min(a.size, b.size))
        if n <= 2:
            return 0.0
        # Z-normalize for correlation
        def znorm(x: np.ndarray) -> np.ndarray:
            xm = float(np.nanmean(x))
            xs = float(np.nanstd(x))
            if not np.isfinite(xs) or xs <= 1e-12:
                return np.zeros_like(x, dtype=float)
            return (x - xm) / xs
        a = znorm(a)
        b = znorm(b)
        L = int(max(0, max_lag))
        best = -1.0
        numerators = correlate(a, b, mode='full', method='auto')
        prefix_a = np.concatenate(([0.0], np.cumsum(a * a)))
        prefix_b = np.concatenate(([0.0], np.cumsum(b * b)))

        for lag in range(-L, L + 1):
            overlap = n - abs(lag)
            if overlap <= 2:
                continue

            if lag >= 0:
                sumsq_a = float(prefix_a[n] - prefix_a[lag])
                sumsq_b = float(prefix_b[overlap] - prefix_b[0])
            else:
                shift = -lag
                sumsq_a = float(prefix_a[overlap] - prefix_a[0])
                sumsq_b = float(prefix_b[n] - prefix_b[shift])

            den = float(np.sqrt(max(sumsq_a, 0.0) * max(sumsq_b, 0.0)))
            if not np.isfinite(den) or den <= 1e-12:
                corr = 0.0
            else:
                idx = lag + (n - 1)
                corr = float(numerators[idx] / den)
            if corr > best:
                best = corr
        if not np.isfinite(best):
            best = 0.0
        return float(max(min(best, 1.0), -1.0))

    def refine_matches(
        self,
        anchor_values: np.ndarray,
        idxs: np.ndarray,
        dists: np.ndarray,
        top_k: int,
        shape_metric: Optional[str] = None,
        allow_lag: int = 0,
        dtw_band_frac: Optional[float] = None,
        soft_dtw_gamma: Optional[float] = None,
        affine_alpha_min: float = 0.5,
        affine_alpha_max: float = 2.0,
        affine_penalty: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Re-rank candidates using a shape metric (e.g., NCC with lag) and return top_k.

        shape_metric:
          - 'ncc': normalized cross-correlation with +/- allow_lag bar shifts
          - None/'none': no re-ranking
        """
        if shape_metric is None:
            shape_metric = 'none'
        sm = str(shape_metric).lower().strip()
        if sm in ("", "none"):
            # No refinement; just truncate
            k = min(int(top_k), idxs.size)
            return idxs[:k], dists[:k]

        # Prepare scaled anchor window (window part only)
        a = self._scaled_window(np.asarray(anchor_values, dtype=float))
        scores: List[Tuple[float, int]] = []
        max_lag = int(allow_lag) if allow_lag and int(allow_lag) > 0 else 0
        for idx, _d in zip(idxs.tolist(), dists.tolist()):
            w = self.get_match_values(int(idx), include_future=False)
            w = self._scaled_window(w)
            if sm == 'ncc':
                corr = self._ncc_max(a, w, max_lag)
                # Convert to distance-like score: lower is better
                score = 1.0 - float(corr)
            elif sm == 'affine':
                # Fit alpha, beta minimizing ||a - (alpha*w + beta)||_2
                aw = float(np.dot(a - np.mean(a), w - np.mean(w)))
                ww = float(np.dot(w - np.mean(w), w - np.mean(w)))
                alpha = (aw / ww) if (np.isfinite(ww) and ww > 1e-12) else 0.0
                # Constrain alpha
                alpha = max(float(affine_alpha_min), min(float(affine_alpha_max), float(alpha)))
                beta = float(np.mean(a) - alpha * np.mean(w))
                resid = a - (alpha * w + beta)
                rmse = float(np.sqrt(np.mean(resid * resid)))
                # Optional penalty to discourage extreme scaling
                score = rmse + float(affine_penalty) * abs(float(alpha) - 1.0)
            elif sm in ('dtw', 'softdtw'):
                # Compute DTW/Soft-DTW distance with the required tslearn backend.
                n = a.size
                band = None
                if dtw_band_frac is not None and dtw_band_frac > 0:
                    band = max(1, int(round(float(dtw_band_frac) * n)))
                if sm == 'dtw':
                    score = dtw_distance(
                        a,
                        w,
                        sakoe_chiba_radius=int(band) if band else None,
                    )
                else:  # softdtw
                    ts_soft_dtw = _get_ts_soft_dtw()
                    gamma = float(soft_dtw_gamma) if (soft_dtw_gamma is not None and soft_dtw_gamma > 0) else 1.0
                    score = float(ts_soft_dtw(a.reshape(1, -1), w.reshape(1, -1), gamma=gamma))
            else:
                # Fallback to euclidean on scaled windows
                diff = a - w
                score = float(np.sqrt(np.dot(diff, diff)))
            scores.append((score, int(idx)))
        scores.sort(key=lambda x: x[0])
        take = min(int(top_k), len(scores))
        new_idxs = np.array([i for _, i in scores[:take]], dtype=int)
        new_scores = np.array([s for s, _ in scores[:take]], dtype=float)
        return new_idxs, new_scores

    def bars_per_symbol(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for ser in self._series:
            out[ser.symbol] = int(len(ser.close))
        return out

    def windows_per_symbol(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for ser in self._series:
            n = int(len(ser.close))
            w = max(0, n - (self.window_size + self.future_size) + 1)
            out[ser.symbol] = w
        return out

    def get_symbol_series(self, symbol: str) -> Optional[np.ndarray]:
        for ser in self._series:
            if ser.symbol == symbol:
                return ser.close
        return None

    def get_symbol_returns(self, symbol: str, lookback: int = 1000) -> Optional[np.ndarray]:
        arr = self.get_symbol_series(symbol)
        if arr is None or len(arr) < 3:
            return None
        # Simple log returns to stabilize scale
        x = np.asarray(arr, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            ret = np.diff(np.log(x))
        ret = ret[np.isfinite(ret)]
        if ret.size <= 0:
            return None
        if lookback and lookback > 0 and ret.size > lookback:
            ret = ret[-int(lookback):]
        return ret.astype(np.float32, copy=False)


def _fetch_symbol_df(
    symbol: str,
    timeframe: str,
    bars: int,
    *,
    as_of: Optional[Any] = None,
    drop_last_live: bool = True,
) -> pd.DataFrame:
    """Fetch last `bars` candles for symbol/timeframe.

    Returns DataFrame with columns at least ['time','open','high','low','close','tick_volume','real_volume']
    where available. 'time' is UTC epoch seconds as float.
    """
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    as_of_text: Optional[str] = None
    if as_of is not None:
        try:
            timestamp = pd.Timestamp(as_of)
            timestamp = (
                timestamp.tz_localize("UTC")
                if timestamp.tzinfo is None
                else timestamp.tz_convert("UTC")
            )
            as_of_text = timestamp.isoformat()
        except Exception as exc:
            raise ValueError(f"Invalid as_of time: {as_of}") from exc
    return fetch_history_frame(
        symbol,
        timeframe,
        int(bars),
        as_of=as_of_text,
        include_incomplete=not drop_last_live,
    )


def _coerce_time_epoch(values: Any) -> np.ndarray:
    ser = pd.Series(values)
    numeric = pd.to_numeric(ser, errors='coerce')
    if numeric.notna().all():
        return numeric.to_numpy(dtype=float)
    dt = pd.to_datetime(ser, utc=True, errors='coerce')
    if dt.isna().any():
        raise ValueError("history DataFrame must provide numeric or datetime-convertible 'time' values")
    return (dt.astype('int64') / 1_000_000_000.0).to_numpy(dtype=float)


def _denoise_method_available(method: str) -> Tuple[bool, str]:
    method_l = str(method or "none").strip().lower()
    if method_l in {"", "none"}:
        return True, ""
    try:
        methods = _get_denoise_methods_data().get("methods", [])
    except Exception:
        return True, ""
    for entry in methods:
        if str(entry.get("method") or "").strip().lower() == method_l:
            return bool(entry.get("available", True)), str(entry.get("requires") or "")
    return True, ""


def _prepare_series(
    symbol: str,
    timeframe: str,
    max_bars: int,
    denoise: Optional[Any] = None,
    *,
    as_of: Optional[Any] = None,
    drop_last_live: bool = True,
    source_df: Optional[pd.DataFrame] = None,
    source_base_col: str = "close",
) -> Tuple[Optional[_SeriesStore], _SeriesPrepareInfo]:
    """Prepare a symbol series from provided history or MT5 fetch."""
    using_provided_history = source_df is not None
    data = source_df.copy() if source_df is not None else _fetch_symbol_df(
        symbol,
        timeframe,
        max_bars,
        as_of=as_of,
        drop_last_live=drop_last_live,
    )
    info = _SeriesPrepareInfo(
        source="provided_history" if using_provided_history else "mt5_fetch",
        base_col=str(source_base_col or "close"),
        bars_raw=int(len(data)),
        bars_used=0,
        denoise_requested=bool(denoise),
        provided_history=using_provided_history,
    )
    if 'volume' not in data.columns and 'tick_volume' in data.columns:
        data['volume'] = data['tick_volume']

    series_col = str(source_base_col or "close").strip() or "close"
    if using_provided_history:
        if series_col not in data.columns:
            info.denoise_error = f"provided history did not include requested base column '{series_col}'"
            return None, info
        info.denoise_applied = bool(denoise) or series_col.endswith("_dn")
        info.denoise_method = str((denoise or {}).get("method")) if isinstance(denoise, dict) else (str(denoise) if denoise else None)
    else:
        series_col = "close"
        normalized_denoise = _normalize_denoise_spec(denoise, default_when='pre_ti')
        if normalized_denoise:
            method_name = str(normalized_denoise.get("method") or "").strip().lower()
            info.denoise_method = method_name or None
            available, requires = _denoise_method_available(method_name)
            if not available:
                info.denoise_error = f"denoise method '{method_name}' is unavailable; requires {requires}"
                raise RuntimeError(info.denoise_error)
            params = dict(normalized_denoise.get("params") or {})
            causality = str(
                normalized_denoise.get("causality")
                or ("causal" if str(normalized_denoise.get("when") or "pre_ti") == "pre_ti" else "zero_phase")
            )
            try:
                data[series_col] = apply_denoise_series(
                    data[series_col],
                    method=method_name,
                    params=params,
                    causality=causality,
                )
                info.denoise_applied = True
            except Exception as exc:
                info.denoise_error = f"failed to denoise '{series_col}' using '{method_name}': {exc}"
                raise RuntimeError(info.denoise_error) from exc

    try:
        t = _coerce_time_epoch(data['time'])
        c = data[series_col].to_numpy(dtype=float)
        t, c = align_finite(t, c)
    except Exception:
        return None, info
    info.bars_used = int(t.size)
    if t.size < 10:
        return None, info
    return _SeriesStore(symbol=symbol, time_epoch=t, close=c), info


def build_index(
    symbols: List[str],
    timeframe: str,
    window_size: int,
    future_size: int,
    max_bars_per_symbol: int = 5000,
    denoise: Optional[Any] = None,
    scale: str = "minmax",
    metric: str = "euclidean",
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    engine: str = "ckdtree",
    *,
    as_of: Optional[Any] = None,
    drop_last_live: bool = True,
    history_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
    history_base_cols: Optional[Dict[str, str]] = None,
) -> PatternIndex:
    """Build a PatternIndex from MT5 data for the provided symbols.

    Notes:
    - Windows are created over ascending time arrays. Window i corresponds to
      close[i : i+window_size]. Matches can expose window+future values.
    - Per-window min-max normalization is applied for the index vectors.
    """
    if window_size < 5:
        raise ValueError("window_size must be at least 5")
    symbols_ok: List[str] = []
    series: List[_SeriesStore] = []
    prepare_info_by_symbol: Dict[str, Dict[str, Any]] = {}
    history_by_symbol = dict(history_by_symbol or {})
    history_base_cols = dict(history_base_cols or {})
    for sym in symbols:
        ser, prep_info = _prepare_series(
            sym,
            timeframe,
            max_bars=max_bars_per_symbol,
            denoise=denoise,
            as_of=as_of,
            drop_last_live=drop_last_live,
            source_df=history_by_symbol.get(sym),
            source_base_col=str(history_base_cols.get(sym) or "close"),
        )
        prepare_info_by_symbol[str(sym)] = {
            "source": prep_info.source,
            "base_col": prep_info.base_col,
            "bars_raw": prep_info.bars_raw,
            "bars_used": prep_info.bars_used,
            "denoise_requested": bool(prep_info.denoise_requested),
            "denoise_applied": bool(prep_info.denoise_applied),
            "denoise_method": prep_info.denoise_method,
            "denoise_error": prep_info.denoise_error,
            "provided_history": bool(prep_info.provided_history),
        }
        if ser is None:
            continue
        # Require enough bars for at least one window
        if ser.close.size >= (window_size + future_size):
            series.append(ser)
            symbols_ok.append(sym)
    if not series:
        raise RuntimeError("No symbols had sufficient data to build pattern index")

    # Allocate the persistent index arrays once. The source windows are views;
    # only their scaled float32 representation needs to be materialized.
    window_counts = [
        max(0, ser.close.size - (window_size + future_size) + 1)
        for ser in series
    ]
    total_windows = int(sum(window_counts))
    if total_windows <= 0:
        raise RuntimeError("Failed to create any windows for the provided symbols")
    X = np.empty((total_windows, window_size), dtype=np.float32)
    start_end_idx = np.empty((total_windows, 2), dtype=int)
    labels_arr = np.empty(total_windows, dtype=int)
    offset = 0
    for lbl, ser in enumerate(series):
        limit = window_counts[lbl]
        if limit <= 0:
            continue
        starts = np.arange(limit, dtype=int)
        ends = starts + (window_size - 1)
        windows = np.lib.stride_tricks.sliding_window_view(
            ser.close, window_size
        )[:limit]
        target = X[offset : offset + limit]
        # Apply per-row scaling
        sc = (scale or "minmax").lower()
        if sc == "zscore":
            mu = np.nanmean(windows, axis=1, keepdims=True)
            sd = np.nanstd(windows, axis=1, keepdims=True)
            sd[sd <= 1e-12] = 1.0
            np.subtract(windows, mu, out=target, casting="unsafe")
            np.divide(target, sd, out=target)
        elif sc == "none":
            target[:] = windows
        else:  # minmax
            mn = np.nanmin(windows, axis=1, keepdims=True)
            mx = np.nanmax(windows, axis=1, keepdims=True)
            rng = mx - mn
            rng[rng <= 1e-12] = 1.0
            np.subtract(windows, mn, out=target, casting="unsafe")
            np.divide(target, rng, out=target)
        start_end_idx[offset : offset + limit, 0] = starts
        start_end_idx[offset : offset + limit, 1] = ends
        labels_arr[offset : offset + limit] = lbl
        offset += limit
    # Optional dimensionality reduction
    reducer: Optional[_DimReducer] = None
    effective_dimred_method = dimred_method or "none"
    effective_dimred_params: Dict[str, Any] = dict(dimred_params or {})
    if effective_dimred_method and str(effective_dimred_method).lower() not in ("none", "false"):
        reducer, info = _create_reducer(effective_dimred_method, effective_dimred_params)
        # If reducer requires n_components, ensure it does not exceed window length
        try:
            if hasattr(reducer, "n_components"):
                nc = int(reducer.n_components)
                if nc > int(X.shape[1]):
                    # Recreate reducer with clipped components
                    effective_dimred_params["n_components"] = int(X.shape[1])
                    reducer, info = _create_reducer(effective_dimred_method, effective_dimred_params)
        except Exception:
            pass
        X = reducer.fit_transform(X)
        X = X.astype(np.float32, copy=False)

    # Metric transform (for cosine/correlation)
    met = (metric or "euclidean").lower()
    if met == "cosine":
        # L2-normalize rows
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms <= 1e-12] = 1.0
        X = (X / norms).astype(np.float32)
    elif met == "correlation":
        # Re-center rows; then L2 normalize
        X = X - np.nanmean(X, axis=1, keepdims=True)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms <= 1e-12] = 1.0
        X = (X / norms).astype(np.float32)
    eng = (engine or "ckdtree").lower()
    if eng in ("matrix_profile", "mass"):
        tree_obj = None  # search will bypass tree
    elif eng == "hnsw":
        if _HNSW is None:
            raise RuntimeError("hnswlib not available; install hnswlib or use engine='ckdtree'")
        dim = int(X.shape[1])
        index = _HNSW.Index(space='l2', dim=dim)
        # Defaults tuned for good recall/speed tradeoff; can be parameterized later
        index.init_index(max_elements=int(X.shape[0]), ef_construction=200, M=16)
        index.add_items(X.astype(np.float32), np.arange(X.shape[0], dtype=np.int32))
        index.set_ef(64)  # ef_search
        tree_obj = index
    elif eng == "ckdtree":
        tree_obj = cKDTree(X)
    else:
        raise ValueError(f"Unknown engine '{eng}'")

    return PatternIndex(
        timeframe=timeframe,
        window_size=int(window_size),
        future_size=int(future_size),
        symbols=symbols_ok,
        tree=tree_obj,
        X=X,
        start_end_idx=start_end_idx,
        labels=labels_arr,
        series=series,
        scale=(scale or "minmax").lower(),
        metric=(metric or "euclidean").lower(),
        dimred_method=str(effective_dimred_method or 'none'),
        dimred_params=effective_dimred_params,
        reducer=reducer,
        engine=eng,
        max_bars_per_symbol=int(max_bars_per_symbol),
        build_metadata={
            "series_prepare_info": prepare_info_by_symbol,
            "bars_per_symbol": {ser.symbol: int(len(ser.close)) for ser in series},
            "windows_per_symbol": {
                ser.symbol: max(0, int(len(ser.close)) - (int(window_size) + int(future_size)) + 1)
                for ser in series
            },
        },
    )


def _apply_scale_vector(x: np.ndarray, scale: str) -> np.ndarray:
    s = (scale or "minmax").lower()
    x = np.asarray(x, dtype=float)
    if s == "zscore":
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x))
        if not np.isfinite(sd) or sd <= 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - mu) / sd).astype(np.float32)
    if s == "none":
        return x.astype(np.float32)
    # minmax
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    rng = mx - mn
    if not np.isfinite(rng) or rng <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / rng).astype(np.float32)


def _apply_metric_vector(x: np.ndarray, metric: str) -> np.ndarray:
    m = (metric or "euclidean").lower()
    v = np.asarray(x, dtype=np.float32)
    if m == "cosine":
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n <= 1e-12:
            return np.zeros_like(v, dtype=np.float32)
        return (v / n).astype(np.float32)
    if m == "correlation":
        v = v - float(np.nanmean(v))
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n <= 1e-12:
            return np.zeros_like(v, dtype=np.float32)
        return (v / n).astype(np.float32)
    return v


