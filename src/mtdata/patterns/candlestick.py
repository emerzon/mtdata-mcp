import logging
import warnings
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..shared.constants import TIME_DISPLAY_FORMAT, TIMEFRAME_SECONDS
from ..shared.validators import invalid_timeframe_error
from ..utils.freshness import completed_bar_freshness_fields
from ..utils.time import _format_time_minimal_local, _use_client_tz
from ..utils.utils import (
    _parse_end_datetime,
    _parse_start_datetime,
    _table_from_rows,
)
from .common import data_quality_warnings, should_drop_last_live_bar
from .enrichment import (
    _apply_confidence_delta,
    _config_bool,
    _config_float,
    _config_int,
    _infer_market_regime,
    _resolve_volume_series,
    _round_value,
    _volume_window_mean,
    directional_regime_verdict,
    volume_confirmation_verdict,
)

logger = logging.getLogger(__name__)


ta: Any = None
mt5: Any = None
TIMEFRAME_MAP: Optional[Dict[str, Any]] = None
_mt5_copy_rates_from: Any = None
_mt5_copy_rates_range: Any = None
_rates_to_df: Any = None
_symbol_ready_guard: Any = None
_CANDLESTICK_PATTERN_METHOD_CACHE: Optional[Tuple[str, ...]] = None
_CANDLESTICK_PATTERN_METHOD_CACHE_KEY: Optional[str] = None
_CANDLESTICK_PATTERN_METHOD_CACHE_LOCK = Lock()
_ROBUST_CANDLESTICK_WHITELIST = {
    "engulfing",
    "harami",
    "3inside",
    "3outside",
    "eveningstar",
    "morningstar",
    "darkcloudcover",
    "piercing",
    "inside",
    "outside",
    "hikkake",
}
_CANDLESTICK_PATTERN_BAR_SPANS = {
    "2crows": 3,
    "counterattack": 2,
    "darkcloudcover": 2,
    "engulfing": 2,
    "harami": 2,
    "haramicross": 2,
    "hikkake": 3,
    "hikkakemod": 3,
    "inside": 2,
    "outside": 2,
    "piercing": 2,
    "tasukigap": 2,
    "3blackcrows": 3,
    "3inside": 3,
    "3outside": 3,
    "3starsinsouth": 3,
    "3whitesoldiers": 3,
    "advanceblock": 3,
    "deliberation": 3,
    "eveningstar": 3,
    "gapsidesidewhite": 3,
    "identical3crows": 3,
    "morningstar": 3,
    "sticksandwich": 3,
    "tristar": 3,
    "unique3river": 3,
    "xsidegap3methods": 3,
    "breakaway": 5,
    "ladderbottom": 5,
    "mathold": 5,
    "risefall3methods": 5,
}
_DEPRIORITIZED_CANDLESTICK_PATTERNS = {
    "shortline",
    "longline",
    "spinningtop",
    "highwave",
    "marubozu",
    "closingmarubozu",
    "doji",
    "gravestonedoji",
    "longleggeddoji",
    "rickshawman",
}
_CANDLESTICK_REDUNDANCY_SUPPRESSORS = {
    "doji": frozenset(
        {"dragonflydoji", "gravestonedoji", "longleggeddoji", "rickshawman"}
    ),
    "inside": frozenset({"harami", "haramicross"}),
    "outside": frozenset({"engulfing"}),
    "harami": frozenset({"haramicross"}),
    "hikkake": frozenset({"hikkakemod"}),
    "marubozu": frozenset({"closingmarubozu"}),
}


def _normalize_candlestick_name(pattern_name: str) -> str:
    nm = _candlestick_display_name(pattern_name)
    while nm:
        parts = nm.replace("_", " ").replace("-", " ").split()
        if len(parts) > 1 and parts[0].lower() in {"bullish", "bearish", "neutral"}:
            nm = " ".join(parts[1:])
            continue
        if len(parts) > 1 and parts[0].lower() == "cdl":
            nm = " ".join(parts[1:])
            continue
        if nm.lower().startswith("cdl_"):
            nm = nm[len("cdl_") :]
            continue
        break
    return "".join(ch for ch in nm.lower() if ch.isalnum())


def _candlestick_display_name(pattern_name: str) -> str:
    """Return a detector name without pandas-ta's encoded numeric parameters."""
    parts = str(pattern_name).strip().replace("_", " ").replace("-", " ").split()
    while len(parts) > 1:
        try:
            float(parts[-1])
        except (TypeError, ValueError):
            break
        parts.pop()
    return " ".join(parts)


def _candlestick_detector_label(pattern_name: str) -> str:
    return _normalize_candlestick_name(pattern_name).upper()


def _format_candlestick_detector_labels(
    pattern_methods: List[str], *, limit: int = 40
) -> str:
    labels = sorted(
        {
            label
            for method in pattern_methods
            if (label := _candlestick_detector_label(method))
        }
    )
    if not labels:
        return ""
    shown = labels[:limit]
    suffix = f", ... (+{len(labels) - limit})" if len(labels) > limit else ""
    return ", ".join(shown) + suffix


def _parse_min_strength(min_strength: float) -> float:
    try:
        thr = float(min_strength)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_strength must be a float between 0.0 and 1.0.") from exc
    if not (0.0 <= thr <= 1.0):
        raise ValueError("min_strength must be between 0.0 and 1.0.")
    return thr


def _candlestick_base_strength(
    pattern_name: str,
    *,
    robust_set: set[str],
    deprioritize: set[str],
) -> float:
    normalized = _normalize_candlestick_name(pattern_name)
    base = 0.50
    if normalized in robust_set:
        base += 0.20
    if normalized in deprioritize:
        base -= 0.10
    return float(max(0.0, min(1.0, base)))


def _candlestick_strength_score(
    pattern_name: str,
    raw_signal: float,
    *,
    robust_set: set[str],
    deprioritize: set[str],
    geometry_score: float = 0.5,
) -> float:
    raw = abs(float(raw_signal))
    if not np.isfinite(raw) or raw <= 0.0:
        return 0.0
    span_bars = _candlestick_span_bars(pattern_name)
    base = _candlestick_base_strength(
        pattern_name,
        robust_set=robust_set,
        deprioritize=deprioritize,
    )
    span_bonus = min(0.10, 0.05 * max(0, span_bars - 1))
    detector_bonus = 0.05
    geometry = float(np.clip(float(geometry_score), 0.0, 1.0))
    return float(_combine_candlestick_strength(base, span_bonus, geometry, detector_bonus))


def _combine_candlestick_strength(
    base: Any,
    span_bonus: Any,
    geometry_score: Any,
    detector_bonus: float = 0.05,
) -> Any:
    return np.clip(base + span_bonus + detector_bonus + 0.40 * geometry_score, 0.0, 1.0)


def _candlestick_span_bars(pattern_name: str) -> int:
    return int(
        _CANDLESTICK_PATTERN_BAR_SPANS.get(_normalize_candlestick_name(pattern_name), 1)
    )


def _ensure_candlestick_runtime() -> None:
    global \
        ta, \
        mt5, \
        TIMEFRAME_MAP, \
        _mt5_copy_rates_from, \
        _mt5_copy_rates_range, \
        _rates_to_df, \
        _symbol_ready_guard

    if (
        ta is not None
        and mt5 is not None
        and TIMEFRAME_MAP is not None
        and _mt5_copy_rates_from is not None
        and _mt5_copy_rates_range is not None
        and _rates_to_df is not None
        and _symbol_ready_guard is not None
    ):
        return

    with _CANDLESTICK_PATTERN_METHOD_CACHE_LOCK:
        if ta is None:
            try:
                import pandas_ta as ta_mod  # type: ignore
            except ModuleNotFoundError:
                try:
                    import pandas_ta_classic as ta_mod  # type: ignore
                except ModuleNotFoundError as e:
                    raise ModuleNotFoundError(
                        "pandas_ta not found. Install 'pandas-ta-classic' (or 'pandas-ta')."
                    ) from e
            ta = ta_mod
        if mt5 is None:
            try:
                from ..utils.mt5 import mt5 as mt5_mod
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "MetaTrader5 not found. Install 'MetaTrader5' to use candlestick detection."
                ) from e
            mt5 = mt5_mod
        if TIMEFRAME_MAP is None:
            from ..shared.constants import TIMEFRAME_MAP as timeframe_map

            TIMEFRAME_MAP = timeframe_map
        if (
            _mt5_copy_rates_from is None
            or _mt5_copy_rates_range is None
            or _rates_to_df is None
            or _symbol_ready_guard is None
        ):
            from ..utils.mt5 import (
                _mt5_copy_rates_from as copy_rates_from,
            )
            from ..utils.mt5 import (
                _mt5_copy_rates_range as copy_rates_range,
            )
            from ..utils.mt5 import (
                _rates_to_df as rates_to_df,
            )
            from ..utils.mt5 import (
                _symbol_ready_guard as symbol_ready_guard,
            )

            if _mt5_copy_rates_from is None:
                _mt5_copy_rates_from = copy_rates_from
            if _mt5_copy_rates_range is None:
                _mt5_copy_rates_range = copy_rates_range
            if _rates_to_df is None:
                _rates_to_df = rates_to_df
            if _symbol_ready_guard is None:
                _symbol_ready_guard = symbol_ready_guard


def _discover_candlestick_pattern_methods(ta_accessor: Any) -> Tuple[str, ...]:
    methods: List[str] = []
    for attr in dir(ta_accessor):
        if not attr.startswith("cdl_"):
            continue
        func = getattr(ta_accessor, attr, None)
        if callable(func):
            methods.append(attr)
    return tuple(sorted(methods))


def _candlestick_accessor_cache_key(ta_accessor: Any) -> str:
    accessor_type = type(ta_accessor)
    return f"{accessor_type.__module__}.{accessor_type.__qualname__}"


def _get_candlestick_pattern_methods(temp: pd.DataFrame) -> List[str]:
    global _CANDLESTICK_PATTERN_METHOD_CACHE, _CANDLESTICK_PATTERN_METHOD_CACHE_KEY

    cache_key = _candlestick_accessor_cache_key(temp.ta)

    if (
        _CANDLESTICK_PATTERN_METHOD_CACHE is not None
        and _CANDLESTICK_PATTERN_METHOD_CACHE_KEY == cache_key
    ):
        return list(_CANDLESTICK_PATTERN_METHOD_CACHE)

    with _CANDLESTICK_PATTERN_METHOD_CACHE_LOCK:
        if (
            _CANDLESTICK_PATTERN_METHOD_CACHE is not None
            and _CANDLESTICK_PATTERN_METHOD_CACHE_KEY == cache_key
        ):
            return list(_CANDLESTICK_PATTERN_METHOD_CACHE)
        try:
            _CANDLESTICK_PATTERN_METHOD_CACHE = _discover_candlestick_pattern_methods(
                temp.ta
            )
            _CANDLESTICK_PATTERN_METHOD_CACHE_KEY = cache_key
        except Exception:
            logger.warning(
                "Failed to enumerate candlestick pattern detectors from pandas_ta.",
                exc_info=True,
            )
            _CANDLESTICK_PATTERN_METHOD_CACHE = None
            _CANDLESTICK_PATTERN_METHOD_CACHE_KEY = None
            return []
    return list(_CANDLESTICK_PATTERN_METHOD_CACHE)


def _candlestick_dispatch_catalog(ta_module: Any) -> set[str]:
    """Return public pattern names understood by an aggregate dispatcher."""
    for attribute in ("CDL_PATTERN_NAMES", "ALL_PATTERNS"):
        values = getattr(ta_module, attribute, None)
        if not isinstance(values, (list, tuple, set, frozenset)):
            continue
        catalog = {
            normalized
            for value in values
            if (normalized := _normalize_candlestick_name(str(value)))
        }
        if catalog:
            return catalog
    return set()


def _candlestick_direct_method_map(
    pattern_methods: List[str],
) -> Dict[str, str]:
    methods: Dict[str, str] = {}
    for method_name in sorted(pattern_methods):
        normalized = _normalize_candlestick_name(method_name)
        if not normalized or normalized == "pattern":
            continue
        methods.setdefault(normalized, method_name)
    return methods


def _dedupe_redundant_candlestick_hits(
    hit_idx: np.ndarray,
    *,
    values_row: np.ndarray,
    normalized_names: np.ndarray,
    span_values: np.ndarray,
    end_index: int,
) -> np.ndarray:
    if hit_idx.size < 2:
        return hit_idx

    hit_start_idx = np.asarray(
        [max(0, int(end_index - span_values[idx] + 1)) for idx in hit_idx], dtype=int
    )
    hit_direction = np.sign(values_row[hit_idx]).astype(int, copy=False)
    hit_names = normalized_names[hit_idx]
    keep_mask = np.ones(hit_idx.size, dtype=bool)

    for pos, name in enumerate(hit_names.tolist()):
        suppressors = _CANDLESTICK_REDUNDANCY_SUPPRESSORS.get(str(name))
        if not suppressors:
            continue
        same_window = hit_start_idx == hit_start_idx[pos]
        same_direction = hit_direction == hit_direction[pos]
        more_specific = np.asarray(
            [str(other_name) in suppressors for other_name in hit_names.tolist()],
            dtype=bool,
        )
        if bool(np.any(same_window & same_direction & more_specific)):
            keep_mask[pos] = False

    return hit_idx[keep_mask]


def _extract_candlestick_rows(
    df_tail: pd.DataFrame,
    temp_tail: pd.DataFrame,
    pattern_cols: List[str],
    *,
    threshold: float,
    robust_only: bool,
    robust_set: set[str],
    whitelist_set: Optional[set[str]],
    min_gap: int,
    top_k: int,
    deprioritize: set[str],
    include_metrics: bool = False,
    start_index: int = 0,
) -> List[List[Any]]:
    if not pattern_cols:
        return []

    base_names = np.asarray(
        [
            col[len("cdl_") :] if col.lower().startswith("cdl_") else col
            for col in pattern_cols
        ],
        dtype=object,
    )
    normalized_names = np.asarray(
        [_normalize_candlestick_name(name) for name in base_names], dtype=object
    )
    try:
        values = (
            temp_tail.loc[:, pattern_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float, copy=False)
        )
    except Exception:
        values = temp_tail.loc[:, pattern_cols].to_numpy(dtype=float, copy=True)

    strength_values = np.zeros_like(values, dtype=float)
    span_values = np.asarray(
        [_candlestick_span_bars(str(name)) for name in base_names.tolist()], dtype=int
    )
    geometry_available = all(
        column in df_tail.columns for column in ("open", "high", "low", "close")
    )
    if geometry_available:
        open_values = pd.to_numeric(df_tail["open"], errors="coerce").to_numpy(
            dtype=float, copy=False
        )
        high_values = pd.to_numeric(df_tail["high"], errors="coerce").to_numpy(
            dtype=float, copy=False
        )
        low_values = pd.to_numeric(df_tail["low"], errors="coerce").to_numpy(
            dtype=float, copy=False
        )
        close_geometry_values = pd.to_numeric(
            df_tail["close"], errors="coerce"
        ).to_numpy(dtype=float, copy=False)
        candle_range = high_values - low_values
        valid_range = np.isfinite(candle_range) & (candle_range > 0.0)
        body_ratio = np.divide(
            np.abs(close_geometry_values - open_values),
            candle_range,
            out=np.full(len(df_tail), 0.5, dtype=float),
            where=valid_range,
        )
        range_series = pd.Series(candle_range).where(valid_range)
        typical_range = (
            range_series.shift(1).rolling(20, min_periods=1).median().to_numpy()
        )
        range_expansion = np.divide(
            candle_range,
            typical_range,
            out=np.ones(len(df_tail), dtype=float),
            where=np.isfinite(typical_range) & (typical_range > 0.0),
        )
        range_expansion = np.clip(range_expansion / 2.0, 0.0, 1.0)
    else:
        high_values = low_values = close_geometry_values = candle_range = None
        valid_range = body_ratio = range_expansion = None

    for col_idx, name in enumerate(base_names.tolist()):
        base_strength = _candlestick_base_strength(
            str(name),
            robust_set=robust_set,
            deprioritize=deprioritize,
        )
        span_bars = int(span_values[col_idx])
        span_bonus = min(0.10, 0.05 * max(0, span_bars - 1))
        geometry_score = np.full(len(df_tail), 0.5, dtype=float)
        normalized_name = str(normalized_names[col_idx])
        if geometry_available:
            assert high_values is not None
            assert low_values is not None
            assert close_geometry_values is not None
            assert candle_range is not None
            assert valid_range is not None
            assert body_ratio is not None
            assert range_expansion is not None
            bullish_location = np.divide(
                close_geometry_values - low_values,
                candle_range,
                out=np.full(len(df_tail), 0.5, dtype=float),
                where=valid_range,
            )
            bearish_location = np.divide(
                high_values - close_geometry_values,
                candle_range,
                out=np.full(len(df_tail), 0.5, dtype=float),
                where=valid_range,
            )
            directional_location = np.where(
                values[:, col_idx] >= 0.0,
                bullish_location,
                bearish_location,
            )
            if normalized_name in deprioritize:
                geometry_score = (
                    0.70 * (1.0 - np.clip(body_ratio, 0.0, 1.0))
                    + 0.30 * range_expansion
                )
            else:
                geometry_score = (
                    0.50 * np.clip(body_ratio, 0.0, 1.0)
                    + 0.30 * np.clip(directional_location, 0.0, 1.0)
                    + 0.20 * range_expansion
                )
        strength_values[:, col_idx] = _combine_candlestick_strength(
            base_strength,
            span_bonus,
            geometry_score,
        )

    active_mask = (
        np.isfinite(values)
        & (np.abs(values) > 0.0)
        & (strength_values >= float(threshold))
    )
    if not bool(np.any(active_mask)):
        return []

    rows: List[List[Any]] = []
    gap = max(0, int(min_gap))
    k = max(1, int(top_k))
    last_pick_idx = 10**9
    non_dep_mask = np.asarray(
        [str(name) not in deprioritize for name in normalized_names], dtype=bool
    )

    if "time" in df_tail.columns:
        time_vals = df_tail["time"].astype(str).to_numpy(dtype=object, copy=False)
    else:
        time_vals = np.full(len(df_tail), "", dtype=object)
    close_vals: Optional[np.ndarray] = None
    if include_metrics and "close" in df_tail.columns:
        try:
            close_vals = pd.to_numeric(df_tail["close"], errors="coerce").to_numpy(
                dtype=float, copy=False
            )
        except Exception:
            close_vals = None

    start_idx = max(0, int(start_index))
    candidate_rows = np.flatnonzero(np.any(active_mask, axis=1))
    if start_idx > 0:
        candidate_rows = candidate_rows[candidate_rows >= start_idx]
    # Recent-signal consumers care about the newest completed pattern in a
    # collision window. Select newest-first so an older hit cannot suppress a
    # later signal merely because the detector returned chronological rows.
    for i in candidate_rows[::-1].tolist():
        if last_pick_idx - i < gap:
            continue
        hit_idx = np.flatnonzero(active_mask[i])
        if hit_idx.size == 0:
            continue
        hit_idx = _dedupe_redundant_candlestick_hits(
            hit_idx,
            values_row=values[i],
            normalized_names=normalized_names,
            span_values=span_values,
            end_index=i,
        )
        if hit_idx.size == 0:
            continue
        pool_idx = hit_idx[non_dep_mask[hit_idx]]
        if pool_idx.size == 0:
            pool_idx = hit_idx
        order = np.lexsort(
            (
                -np.abs(values[i, pool_idx]),
                -strength_values[i, pool_idx],
            )
        )[:k]
        chosen_idx = pool_idx[order]
        t_val = str(time_vals[i])
        for col_idx in chosen_idx.tolist():
            name = str(base_names[col_idx])
            value = float(values[i, col_idx])
            label_core = _candlestick_display_name(name).strip().upper()
            normalized = str(normalized_names[col_idx])
            if normalized in deprioritize:
                dir_title = "Neutral"
            else:
                dir_title = "Bullish" if value > 0 else "Bearish"
            if include_metrics:
                span_bars = int(span_values[col_idx])
                start_bar_idx = max(0, int(i - span_bars + 1))
                start_time = str(time_vals[start_bar_idx])
                end_time = str(time_vals[i])
                if normalized in deprioritize:
                    direction = "neutral"
                else:
                    direction = "bullish" if value > 0 else "bearish"
                strength = float(strength_values[i, col_idx])
                raw_signal: Any
                if normalized in deprioritize:
                    raw_signal = 0
                elif abs(value - round(value)) <= 1e-9:
                    raw_signal = int(round(value))
                else:
                    raw_signal = float(value)
                price = (
                    _round_value(close_vals[i])
                    if close_vals is not None and np.isfinite(close_vals[i])
                    else None
                )
                rows.append(
                    [
                        end_time,
                        f"{dir_title} {label_core}" if label_core else dir_title,
                        direction,
                        strength,
                        raw_signal,
                        price,
                        start_time,
                        end_time,
                        int(span_bars),
                        int(start_bar_idx),
                        int(i),
                    ]
                )
            else:
                rows.append(
                    [t_val, f"{dir_title} {label_core}" if label_core else dir_title]
                )
        last_pick_idx = i
    return rows


def _row_meets_min_strength(row: Dict[str, Any], threshold: float) -> bool:
    try:
        confidence = float(row.get("confidence"))
    except Exception:
        return False
    return np.isfinite(confidence) and confidence >= float(threshold)


def _filter_rows_by_min_strength(rows: Any, threshold: float) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, dict) and _row_meets_min_strength(row, threshold)
    ]


def _is_candlestick_allowed(
    pattern_name: str,
    *,
    robust_only: bool,
    robust_set: set[str],
    whitelist_set: Optional[set[str]],
) -> bool:
    nm = _normalize_candlestick_name(pattern_name)
    if whitelist_set is not None and nm not in whitelist_set:
        return False
    if robust_only and nm not in robust_set:
        return False
    return True


def detect_candlestick_patterns(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    limit: int,
    min_strength: float,
    min_gap: int,
    robust_only: bool,
    whitelist: Optional[str],
    top_k: int,
    last_n_bars: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    denoise: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        _ensure_candlestick_runtime()
    except ModuleNotFoundError as exc:
        return {"error": str(exc)}
    if timeframe not in TIMEFRAME_MAP:
        return {"error": invalid_timeframe_error(timeframe, TIMEFRAME_MAP or {})}
    try:
        thr = _parse_min_strength(min_strength)
    except ValueError as exc:
        return {"error": str(exc)}

    mt5_timeframe = TIMEFRAME_MAP[timeframe]
    utc_now = datetime.now(timezone.utc)
    start_dt = _parse_start_datetime(start) if start else None
    if start and start_dt is None:
        return {"error": "Invalid start time."}
    end_dt = _parse_end_datetime(end) if end else None
    if end and end_dt is None:
        return {"error": "Invalid end time."}
    if start_dt is not None and end_dt is None:
        end_dt = utc_now.replace(tzinfo=None)
    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        return {"error": "start must be before or equal to end."}

    with _symbol_ready_guard(symbol) as (err, _info):
        if err:
            return {"error": err}
        if start_dt is not None:
            rates = _mt5_copy_rates_range(symbol, mt5_timeframe, start_dt, end_dt)
        elif end_dt is not None:
            rates = _mt5_copy_rates_from(
                symbol, mt5_timeframe, end_dt, int(limit) + 1
            )
        else:
            rates = _mt5_copy_rates_from(
                symbol, mt5_timeframe, utc_now, int(limit) + 1
            )

    if rates is None:
        return {"error": f"Failed to get rates for {symbol}: {mt5.last_error()}"}
    if len(rates) == 0:
        return {"error": "No candle data available"}

    df = _rates_to_df(rates)
    from ..services.data_service.candles import _resolve_live_bar_reference_epoch

    live_bar_reference_epoch = _resolve_live_bar_reference_epoch(symbol, timeframe)
    if should_drop_last_live_bar(
        df,
        timeframe,
        now_utc=utc_now,
        current_time_epoch=live_bar_reference_epoch,
    ):
        df = df.iloc[:-1].copy()
    denoise_warnings: List[str] = []
    if denoise:
        try:
            from ..utils.denoise import apply_denoise as apply_denoise_util
            from ..utils.denoise import (
                normalize_denoise_spec as _normalize_denoise_spec,
            )

            dn = _normalize_denoise_spec(denoise, default_when="pre_ti")
            if dn:
                apply_denoise_util(df, dn, default_when="pre_ti")
                suffix = str(dn.get("suffix") or "_dn")
                for name in ("open", "high", "low", "close", "volume", "tick_volume"):
                    candidate = f"{name}{suffix}"
                    if candidate in df.columns:
                        df[name] = df[candidate]
        except Exception as exc:
            logger.warning(
                "Denoise failed for candlestick detection on %s %s; raw prices were used.",
                symbol,
                timeframe,
                exc_info=True,
            )
            denoise_warnings.append(
                f"Denoise failed for pattern detection on {symbol} {timeframe}; "
                f"raw prices were used. {exc}"
            )
    if len(df) > int(limit):
        df = df.iloc[-int(limit):].copy()
    if len(df) == 0:
        return {"error": "No closed candle data available"}
    warnings_out = list(denoise_warnings)
    warnings_out.extend(
        data_quality_warnings(
            df,
            symbol=symbol,
            timeframe_seconds=float(TIMEFRAME_SECONDS.get(timeframe, 0) or 0),
        )
    )
    epochs = pd.to_numeric(df["time"], errors="coerce").tolist()
    _use_ctz = _use_client_tz()
    if _use_ctz:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df["time"] = df["time"].apply(_format_time_minimal_local)
    else:
        time_fmt = TIME_DISPLAY_FORMAT
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df["time"] = df["time"].apply(
                lambda t: datetime.fromtimestamp(float(t), tz=timezone.utc).strftime(
                    time_fmt
                )
            )

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            return {"error": f"Missing '{col}' data from rates"}

    temp = df.copy()
    temp["__epoch"] = [float(e) for e in epochs]
    try:
        temp.index = pd.to_datetime(temp["__epoch"], unit="s")
    except Exception:
        pass

    pattern_methods = _get_candlestick_pattern_methods(temp)
    if not pattern_methods:
        return {"error": "No candlestick pattern detectors (cdl_*) found in pandas_ta."}

    parsed_whitelist: Optional[set[str]] = None
    whitelist_parts: List[str] = []
    if whitelist and isinstance(whitelist, str):
        try:
            whitelist_parts = [p.strip() for p in whitelist.split(",") if p.strip()]
            if whitelist_parts:
                parsed_whitelist = {
                    _normalize_candlestick_name(p) for p in whitelist_parts
                }
        except Exception:
            pass

    dispatcher_method = next(
        (
            name
            for name in pattern_methods
            if _normalize_candlestick_name(name) == "pattern"
        ),
        None,
    )
    dispatch_catalog = _candlestick_dispatch_catalog(ta)
    direct_methods = _candlestick_direct_method_map(pattern_methods)
    backend_detector_names = set(direct_methods) | dispatch_catalog
    available_detector_names = (
        backend_detector_names & _ROBUST_CANDLESTICK_WHITELIST
        if robust_only
        else backend_detector_names
    )

    requested_order = list(
        dict.fromkeys(
            _normalize_candlestick_name(part)
            for part in whitelist_parts
            if _normalize_candlestick_name(part)
        )
    )
    requested_names = set(requested_order) if parsed_whitelist is not None else None
    robust_filtered_names = sorted(
        (requested_names or set()) - _ROBUST_CANDLESTICK_WHITELIST
    ) if robust_only else []
    selected_names = requested_names
    if robust_only:
        selected_names = (
            (requested_names & _ROBUST_CANDLESTICK_WHITELIST)
            if requested_names is not None
            else set(_ROBUST_CANDLESTICK_WHITELIST)
        )

    dispatcher_catalog_unknown = bool(dispatcher_method and not dispatch_catalog)
    supported_selected = (
        available_detector_names
        if selected_names is None
        else selected_names & available_detector_names
    )
    if (
        selected_names is not None
        and not supported_selected
        and not dispatcher_catalog_unknown
    ):
        available = _format_candlestick_detector_labels(
            sorted(available_detector_names)
        )
        requested = ", ".join(whitelist_parts) if whitelist_parts else str(whitelist)
        return {
            "error": (
                "No candlestick detectors match whitelist "
                f"'{requested}'. Available detectors: {available}"
            ),
            "requested_detectors": requested_order,
            "unsupported_detectors": sorted(
                (requested_names or set()) - backend_detector_names
            ),
            "filtered_by_robust_only": robust_filtered_names,
        }
    if selected_names is None and not available_detector_names and not dispatcher_method:
        return {
            "error": "No candlestick detectors match the requested filters.",
        }

    before_cols = set(temp.columns)
    dispatcher_succeeded = False
    failed_detectors: List[str] = []
    if dispatcher_method:
        dispatch_names: Any
        if selected_names is None:
            dispatch_names = "all"
        elif dispatch_catalog:
            dispatch_names = sorted(selected_names & dispatch_catalog)
        else:
            dispatch_names = sorted(selected_names)
        if dispatch_names:
            try:
                method = getattr(temp.ta, dispatcher_method)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    method(name=dispatch_names, append=True)
                dispatcher_succeeded = True
            except Exception:
                logger.warning(
                    "Aggregate candlestick pattern dispatcher '%s' failed.",
                    dispatcher_method,
                    exc_info=True,
                )
                if isinstance(dispatch_names, list):
                    failed_detectors.extend(str(name) for name in dispatch_names)

    for detector_name, method_name in sorted(direct_methods.items()):
        if not _is_candlestick_allowed(
            detector_name,
            robust_only=robust_only,
            robust_set=_ROBUST_CANDLESTICK_WHITELIST,
            whitelist_set=selected_names,
        ):
            continue
        if dispatcher_succeeded and detector_name in dispatch_catalog:
            continue
        try:
            method = getattr(temp.ta, method_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                method(append=True)
        except Exception:
            failed_detectors.append(detector_name)
            logger.warning(
                "Candlestick pattern detector '%s' failed.", method_name, exc_info=True
            )
            continue

    pattern_cols = [
        c for c in temp.columns if c not in before_cols and c.lower().startswith("cdl_")
    ]
    evaluated_detectors = sorted(
        {
            normalized
            for column in pattern_cols
            if (normalized := _normalize_candlestick_name(str(column)))
        }
    )
    unsupported_detectors = sorted(
        (requested_names or set())
        - set(evaluated_detectors)
        - set(robust_filtered_names)
    )
    unresolved_failures = sorted(set(failed_detectors) - set(evaluated_detectors))
    if not pattern_cols:
        available = _format_candlestick_detector_labels(
            sorted(available_detector_names)
        )
        return {
            "error": (
                "No requested candlestick detectors produced outputs. "
                f"Available detectors: {available}"
            ),
            "requested_detectors": requested_order,
            "unsupported_detectors": unsupported_detectors,
            "failed_detectors": unresolved_failures,
        }

    try:
        gap = max(0, int(min_gap))
    except Exception:
        gap = 3
    try:
        k = max(1, int(top_k))
    except Exception:
        k = 1
    last_n_val: Optional[int] = None
    if last_n_bars is not None:
        try:
            last_n_val = int(last_n_bars)
        except Exception:
            return {"error": "last_n_bars must be a positive integer."}
        if last_n_val <= 0:
            return {"error": "last_n_bars must be >= 1."}
    start_index = 0
    if last_n_val is not None and len(df) > last_n_val:
        start_index = int(len(df) - last_n_val)
    rows = _extract_candlestick_rows(
        df,
        temp,
        pattern_cols,
        threshold=thr,
        robust_only=bool(robust_only),
        robust_set=_ROBUST_CANDLESTICK_WHITELIST,
        whitelist_set=parsed_whitelist,
        min_gap=gap,
        top_k=k,
        deprioritize=_DEPRIORITIZED_CANDLESTICK_PATTERNS,
        include_metrics=True,
        start_index=start_index,
    )

    headers = [
        "time",
        "pattern",
        "direction",
        "confidence",
        "raw_signal",
        "price",
        "start_time",
        "end_time",
        "n_bars",
        "start_index",
        "end_index",
    ]
    payload = _table_from_rows(headers, rows)
    _enrich_candlestick_payload(payload, df, config)
    filtered_rows = _filter_rows_by_min_strength(payload.get("data"), thr)
    payload["data"] = filtered_rows
    payload["count"] = len(filtered_rows)
    payload.update(
        {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": int(len(df)),
            "requested_lookback": int(limit),
            "lookback_satisfied": int(len(df)) >= int(limit),
            "mode": "candlestick",
            "min_strength": float(thr),
            "strength_scale": "ohlc_geometry_and_pattern_reliability_v3",
            "signal_scale": "backend_native_cdl_signal",
            "gap_selection_policy": "newest_first",
            "detectors_evaluated": evaluated_detectors,
        }
    )
    if requested_names is not None:
        payload["requested_detectors"] = requested_order
    if unsupported_detectors:
        payload["unsupported_detectors"] = unsupported_detectors
        warnings_out.append(
            "Unsupported candlestick detectors were not evaluated: "
            + ", ".join(name.upper() for name in unsupported_detectors)
            + "."
        )
    if robust_filtered_names:
        payload["filtered_by_robust_only"] = robust_filtered_names
    if unresolved_failures:
        payload["failed_detectors"] = unresolved_failures
    if warnings_out:
        payload["warnings"] = warnings_out
    if not start and not end and epochs:
        payload.update(
            completed_bar_freshness_fields(
                symbol,
                timeframe,
                epochs[-1],
                now_epoch=utc_now.timestamp(),
                item="bar",
            )
        )
    if last_n_val is not None:
        payload["last_n_bars"] = int(last_n_val)
    if _use_ctz:
        from ..utils.time import _resolve_client_tz

        client_tz = _resolve_client_tz()
        payload["timezone"] = str(getattr(client_tz, "zone", None) or client_tz or "local")
    else:
        payload["timezone"] = "UTC"
    return payload


def _enrich_candlestick_payload(
    payload: Dict[str, Any], df: pd.DataFrame, config: Optional[Dict[str, Any]]
) -> None:
    rows = payload.get("data")
    if not isinstance(rows, list) or not isinstance(df, pd.DataFrame) or len(df) <= 0:
        return
    volume, volume_source = _resolve_volume_series(df)
    regime_cache: Dict[int, Optional[Dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        _attach_candlestick_volume_confirmation(row, volume, volume_source, config)
        try:
            end_index = min(max(0, int(row.get("end_index"))), len(df) - 1)
        except (TypeError, ValueError):
            end_index = len(df) - 1
        if end_index not in regime_cache:
            regime_cache[end_index] = _infer_market_regime(
                df.iloc[: end_index + 1], config
            )
        regime_context = regime_cache[end_index]
        _attach_candlestick_regime_context(row, regime_context, config)


def _attach_candlestick_volume_confirmation(
    row: Dict[str, Any],
    volume: Optional[np.ndarray],
    volume_source: Optional[str],
    config: Optional[Dict[str, Any]],
) -> None:
    payload: Dict[str, Any] = {
        "mode": "signal_window",
        "status": "disabled"
        if not _config_bool(config, "use_volume_confirmation", True)
        else "unavailable",
        "volume_source": volume_source,
    }
    if payload["status"] == "disabled":
        row["volume_confirmation"] = payload
        return
    if volume is None or volume_source is None:
        row["volume_confirmation"] = payload
        return
    try:
        end_index = int(row.get("end_index"))
        start_index = int(row.get("start_index"))
    except Exception:
        row["volume_confirmation"] = payload
        return
    breakout_bars = _config_int(config, "volume_confirm_breakout_bars", 2, minimum=1)
    lookback_bars = _config_int(
        config, "volume_confirm_lookback_bars", 20, minimum=breakout_bars + 1
    )
    min_ratio = _config_float(config, "volume_confirm_min_ratio", 1.10, minimum=1.0)
    bonus = _config_float(config, "volume_confirm_bonus", 0.08, minimum=0.0)
    penalty = _config_float(config, "volume_confirm_penalty", 0.06, minimum=0.0)
    signal_end = max(int(start_index), int(end_index))
    pattern_start = min(int(start_index), int(end_index))
    signal_start = max(0, min(pattern_start, int(signal_end - breakout_bars + 1)))
    baseline_end = int(signal_start - 1)
    baseline_start = max(0, int(baseline_end - lookback_bars + 1))
    signal_avg = _volume_window_mean(volume, signal_start, signal_end)
    baseline_avg = _volume_window_mean(volume, baseline_start, baseline_end)
    ratio = (
        float(signal_avg) / float(baseline_avg)
        if signal_avg is not None and baseline_avg is not None and baseline_avg > 0
        else None
    )
    payload["lookback_bars"] = int(lookback_bars)
    payload["breakout_bars"] = int(breakout_bars)
    if baseline_avg is not None:
        payload["baseline_avg_volume"] = _round_value(baseline_avg)
    if signal_avg is not None:
        payload["signal_avg_volume"] = _round_value(signal_avg)
    if ratio is not None and np.isfinite(ratio):
        payload["signal_to_baseline_ratio"] = _round_value(ratio)
    payload["status"], confidence_delta = volume_confirmation_verdict(
        ratio,
        min_ratio=min_ratio,
        bonus=bonus,
        penalty=penalty,
    )
    if abs(confidence_delta) > 1e-12:
        payload["confidence_delta"] = _round_value(confidence_delta)
        _apply_confidence_delta(row, confidence_delta)
    row["volume_confirmation"] = payload


def _attach_candlestick_regime_context(
    row: Dict[str, Any],
    regime_context: Optional[Dict[str, Any]],
    config: Optional[Dict[str, Any]],
) -> None:
    payload: Dict[str, Any] = {
        "status": "disabled"
        if not _config_bool(config, "use_regime_context", True)
        else "unavailable",
    }
    if payload["status"] == "disabled":
        row["regime_context"] = payload
        return
    if not isinstance(regime_context, dict):
        row["regime_context"] = payload
        return
    payload.update(regime_context)
    bias = str(row.get("direction") or "").strip().lower()
    if bias not in {"bullish", "bearish"}:
        payload["status"] = "not_directional"
        row["regime_context"] = payload
        return
    payload["pattern_bias"] = bias
    bonus = _config_float(config, "regime_alignment_bonus", 0.05, minimum=0.0)
    penalty = _config_float(config, "regime_countertrend_penalty", 0.05, minimum=0.0)
    status, alignment, confidence_delta = directional_regime_verdict(
        bias,
        state=payload.get("state"),
        regime_direction=payload.get("direction"),
        bonus=bonus,
        penalty=penalty,
    )
    payload["status"] = status
    payload["alignment"] = alignment
    if abs(confidence_delta) > 1e-12:
        payload["confidence_delta"] = _round_value(confidence_delta)
        _apply_confidence_delta(row, confidence_delta)
    row["regime_context"] = payload
