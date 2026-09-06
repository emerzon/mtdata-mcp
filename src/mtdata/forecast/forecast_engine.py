"""
Forecast engine core logic and orchestration.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ..bootstrap.settings import mt5_config
from ..shared.constants import (
    CALENDAR_TIMEFRAMES,
    SANITY_BARS_TOLERANCE,
    TIMEFRAME_MAP,
    TIMEFRAME_SECONDS,
)
from ..shared.schema import DenoiseSpec, ForecastMethodLiteral, TimeframeLiteral
from ..shared.symbols import is_probably_crypto_symbol
from ..shared.validators import (
    invalid_timeframe_error,
    unsupported_timeframe_seconds_error,
)
from ..utils.denoise import (
    apply_denoise,
    consume_denoise_warnings,
    effective_denoise_base_col,
)
from ..utils.denoise import (
    normalize_denoise_spec as _normalize_denoise_spec,
)
from ..utils.freshness import completed_bar_freshness_fields, format_age_seconds
from ..utils.mt5 import (
    get_cached_mt5_time_alignment,
    get_symbol_info_cached,
    symbol_candle_price_basis_for,
)
from ..utils.time import (
    _format_time_minimal,
    _format_time_minimal_local,
    _resolve_client_tz,
    _use_client_tz,
    bar_close_epoch,
    display_timezone_label,
)
from ..utils.utils import (
    parse_kv_or_json as _parse_kv_or_json,
)
from . import forecast_preprocessing as _forecast_preprocessing
from .common import (
    _parse_as_of_bound,
    default_seasonality,
    describe_forecast_calendar_treatment,
    is_standard_weekend_closed_epoch,
    next_times_from_last,
    resolve_forecast_symbol,
    uses_exchange_intraday_projection,
    uses_standard_weekend_projection,
)
from .common import (
    fetch_history as _fetch_history,
)
from .exceptions import ModelCompatibilityError, UnknownFeatureColumnError
from .forecast_validation import (
    attach_denoise_causality_disclosure,
    forecast_method_resolution_error,
    format_invalid_method_error,
)
from .interface import ArtifactCompatibilityError, ForecastCallContext
from .model_compatibility import (
    build_model_reuse_metadata,
    fingerprint_mismatches,
)

if TYPE_CHECKING:
    from .interface import ForecastMethod


class _AsyncTrainingStarted(Exception):
    """Raised by ``_run_registered_forecast_method`` when an async training
    task is submitted instead of producing a synchronous forecast."""

    def __init__(self, response: Dict[str, Any]) -> None:
        self.response = response
        super().__init__("async training started")


from .forecast_registry import (
    ForecastRegistry,
    get_forecast_method_availability_snapshot,
)
from .target_builder import (
    build_target_series,
    forecast_interval_recovery,
    resolve_alias_base,
)

logger = logging.getLogger(__name__)

_FEATURE_CAPABILITY_ERROR_CODE = "feature_consumption_unsupported"
_FEATURE_ATTESTATION_ERROR_CODE = "feature_consumption_unverified"
_FORECAST_DIMRED_ERROR_CODE = "forecast_dimred_unsupported"


def _count_weekend_forecast_times(times: List[str]) -> int:
    weekend_count = 0
    for value in times:
        try:
            timestamp = pd.Timestamp(value)
        except Exception:
            continue
        if timestamp.weekday() >= 5:
            weekend_count += 1
    return weekend_count


def _forex_forecast_market_status(epoch: Any) -> str:
    try:
        float(epoch)
    except Exception:
        return "unknown"
    return "closed_weekend" if is_standard_weekend_closed_epoch(epoch) else "open"


def _forecast_calendar_gap_rows(
    future_epochs: List[float],
    tf_secs: int,
    fmt_time: Any,
) -> Tuple[List[Dict[str, Any]], int]:
    try:
        step = float(tf_secs)
    except Exception:
        return [], 0
    if step <= 0:
        return [], 0

    gaps: List[Dict[str, Any]] = []
    total_skipped = 0
    for previous_epoch, current_epoch in zip(future_epochs, future_epochs[1:]):
        delta = float(current_epoch) - float(previous_epoch)
        if delta <= step * 1.5:
            continue
        skipped_bars = max(1, int(round(delta / step)) - 1)
        total_skipped += skipped_bars
        gaps.append(
            {
                "from": fmt_time(float(previous_epoch) + step),
                "to": fmt_time(float(current_epoch) - step),
                "skipped_bars": skipped_bars,
                "reason": "weekend",
            }
        )
    return gaps, total_skipped


@dataclass(frozen=True)
class TrainingExecutionContext:
    method_l: str
    data_scope: str
    target_series: pd.Series
    horizon: int
    seasonality: int
    method_params: Dict[str, Any]
    timeframe: str
    exog_used: Optional[np.ndarray]


# Supported forecast methods - dynamically fetch from registry
def _get_available_methods():
    availability = get_forecast_method_availability_snapshot()
    return tuple(
        method
        for method in ForecastRegistry.get_all_method_names()
        if availability.get(method, False)
    )


def _feature_method_capability_error(
    methods: List[str],
    *,
    features: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Reject feature forecasts unless every adapter has audited exog support."""
    if not isinstance(features, dict) or not features:
        return None

    incompatible: List[Dict[str, Any]] = []
    for method_name in methods:
        try:
            adapter = ForecastRegistry.get(str(method_name))
        except Exception:
            historical = False
            future = False
        else:
            historical = bool(
                getattr(adapter, "supports_historical_exog", False)
            )
            future = bool(getattr(adapter, "supports_future_exog", False))
        if not historical or not future:
            incompatible.append(
                {
                    "method": str(method_name),
                    "supports_historical_exog": historical,
                    "supports_future_exog": future,
                }
            )

    if not incompatible:
        return None
    compatible_methods: List[str] = []
    for method_name in _get_available_methods():
        try:
            adapter = ForecastRegistry.get(str(method_name))
        except Exception:
            continue
        if bool(getattr(adapter, "supports_historical_exog", False)) and bool(
            getattr(adapter, "supports_future_exog", False)
        ):
            compatible_methods.append(str(method_name))
    names = ", ".join(row["method"] for row in incompatible)
    return {
        "error": (
            "Feature-bearing forecasts require audited consumption of both "
            "historical and future exogenous inputs; unsupported methods: "
            f"{names}."
        ),
        "error_code": _FEATURE_CAPABILITY_ERROR_CODE,
        "incompatible_methods": incompatible,
        "compatible_methods": sorted(compatible_methods),
        "remediation": (
            "Select a compatible method reported by forecast_list_methods, or "
            "remove --features for a univariate forecast."
        ),
    }


def _forecast_dimred_method_error(
    dimred_method: Optional[str],
) -> Optional[Dict[str, Any]]:
    if str(dimred_method or "").strip().lower() != "tsne":
        return None
    return {
        "error": (
            "dimred method 'tsne' is not supported for forecasting because "
            "t-SNE cannot transform out-of-sample prediction rows; use pca, "
            "svd, umap, or selectkbest"
        ),
        "error_code": _FORECAST_DIMRED_ERROR_CODE,
        "valid_values": {
            "dimred_method": ["pca", "selectkbest", "svd", "umap"]
        },
    }



def _analog_window_size(params: Optional[Dict[str, Any]]) -> int:
    try:
        return max(1, int((params or {}).get("window_size", 64)))
    except Exception:
        return 64


def _analog_search_depth(params: Optional[Dict[str, Any]]) -> int:
    try:
        return max(1, int((params or {}).get("search_depth", 5000)))
    except Exception:
        return 5000


def _analog_overhead_bars(window_size: int, horizon: int) -> int:
    return (2 * int(window_size)) + int(horizon) - 1


def _cap_analog_params_to_lookback(
    params: Dict[str, Any],
    *,
    lookback: Optional[int],
    horizon: int,
) -> Optional[Dict[str, Any]]:
    """Bound analog search_depth so history never exceeds an explicit lookback."""
    if lookback is None or int(lookback) <= 0:
        return None
    window_size = _analog_window_size(params)
    search_depth = _analog_search_depth(params)
    overhead = _analog_overhead_bars(window_size, horizon)
    min_bars = overhead + 1
    if int(lookback) < min_bars:
        return {
            "error": (
                f"Analog lookback {int(lookback)} cannot fit window_size={window_size} "
                f"and horizon={int(horizon)}; at least {min_bars} bars are required."
            ),
            "error_code": "analog_lookback_too_small",
            "lookback": int(lookback),
            "minimum_lookback_bars": min_bars,
            "window_size": window_size,
            "horizon": int(horizon),
            "remediation": (
                f"Increase --lookback to at least {min_bars}, or reduce window_size."
            ),
        }
    fitted = min(search_depth, int(lookback) - overhead)
    if fitted < search_depth:
        params["search_depth"] = fitted
    return None


def _calculate_lookback_bars(method_l: str, horizon: int, lookback: Optional[int],
                             seasonality: int, timeframe: str,
                             params: Optional[Dict[str, Any]] = None) -> int:
    """Calculate the number of bars needed for forecasting."""
    if method_l == 'analog':
        p = dict(params or {})
        window_size = _analog_window_size(p)
        search_depth = _analog_search_depth(p)
        overhead = _analog_overhead_bars(window_size, horizon)
        analog_history_bars = search_depth + overhead
        if lookback is not None and lookback > 0:
            return int(lookback)
        return max(100, analog_history_bars)

    if lookback is not None and lookback > 0:
        return int(lookback)

    if method_l == 'ensemble':
        p = dict(params or {})
        mode = str(p.get('mode', 'average')).lower().strip()
        if mode in ('rmse_weighted', 'stacking'):
            methods = p.get('methods')
            if isinstance(methods, str):
                method_count = len(
                    [item for item in methods.split(',') if item.strip()]
                )
            elif isinstance(methods, (list, tuple)):
                method_count = len(methods)
            else:
                method_count = 3
            cv_points = int(p.get('cv_points', max(6, method_count * 2)))
            min_train = int(p.get('min_train_size', max(30, int(horizon) * 3)))
            return max(100, min_train + int(horizon) + cv_points + 2)
        return max(100, int(horizon) + 10)
    if method_l == 'seasonal_naive':
        return max(3 * seasonality, int(horizon) + seasonality + 2)
    elif method_l in ('theta', 'fourier_ols'):
        return max(300, int(horizon) + (2 * seasonality if seasonality else 50))
    else:  # naive, drift and others
        return max(100, int(horizon) + 10)


def _resolve_history_context(
    *,
    symbol: str,
    timeframe: TimeframeLiteral,
    need: int,
    as_of: Optional[str],
    start: Optional[str],
    end: Optional[str],
    prefetched_df: Optional[pd.DataFrame],
    prefetched_base_col: Optional[str],
    prefetched_denoise_spec: Optional[Any],
    denoise: Optional[DenoiseSpec],
    cap_explicit_range: bool = False,
) -> Tuple[pd.DataFrame, str, Optional[Any]]:
    """Return the source DataFrame, active base column, and denoise spec used."""
    if prefetched_df is not None:
        df = prefetched_df.copy()
        if cap_explicit_range and (start or end) and len(df) > int(need):
            df = df.iloc[-int(need):].reset_index(drop=True)
        dn_spec_used = None
        if prefetched_denoise_spec:
            dn_spec_used = _normalize_denoise_spec(
                prefetched_denoise_spec,
                default_when='pre_ti',
            )
        elif denoise:
            normalized = _normalize_denoise_spec(denoise, default_when='pre_ti')
            added = apply_denoise(df, normalized) if normalized else []
            dn_spec_used = normalized
            if not prefetched_base_col:
                base_col = effective_denoise_base_col(
                    df,
                    normalized,
                    base_col='close',
                    added_columns=added,
                )
                return df, base_col, dn_spec_used
        if prefetched_base_col:
            base_col = prefetched_base_col
        else:
            base_col = effective_denoise_base_col(
                df, dn_spec_used, base_col='close'
            )
            if base_col == 'close' and 'close_dn' in df.columns:
                base_col = 'close_dn'
        return df, base_col, dn_spec_used

    history_kwargs: Dict[str, Any] = {}
    if start or end:
        history_kwargs.update({"start": start, "end": end})
    df = _fetch_history(symbol, timeframe, int(need), as_of, **history_kwargs)
    if cap_explicit_range and (start or end) and len(df) > int(need):
        df = df.iloc[-int(need):].reset_index(drop=True)
    if len(df) < 3:
        raise ValueError("Not enough closed bars to compute forecast")

    base_col = 'close'
    dn_spec_used = None
    if denoise:
        normalized = _normalize_denoise_spec(denoise, default_when='pre_ti')
        added = apply_denoise(df, normalized) if normalized else []
        dn_spec_used = normalized
        base_col = effective_denoise_base_col(
            df,
            normalized,
            base_col='close',
            added_columns=added,
        )
    return df, base_col, dn_spec_used


def _prepare_target_series_context(
    *,
    df: pd.DataFrame,
    quantity_l: str,
    base_col: str,
    features: Optional[Dict[str, Any]],
    target_spec: Optional[Dict[str, Any]],
) -> Tuple[pd.Series, str, str, Dict[str, Any]]:
    """Prepare the effective base column and target series consumed by forecasters."""
    base_col_initial = base_col
    base_col_prepared = _forecast_preprocessing._prepare_base_data(df, quantity_l, base_col)
    base_col_prepared = _forecast_preprocessing._apply_features_and_target_spec(
        df,
        features,
        target_spec,
        base_col_prepared,
        parse_kv_or_json=_parse_kv_or_json,
    )

    target_series = df[base_col_prepared].dropna()
    target_info: Dict[str, Any] = {}
    if target_spec:
        y_arr, target_info = build_target_series(df, base_col_initial, target_spec, quantity=quantity_l)
        target_series = pd.Series(y_arr, index=df.index)
        base_col_final = target_info.get('base', base_col_initial)
    else:
        base_col_final = base_col_prepared
        if quantity_l == 'return':
            target_info = {'mode': 'return', 'base': base_col_initial, 'transform': 'log_return'}
        else:
            target_series = df[base_col_final]
            target_info = {'mode': quantity_l, 'base': base_col_final, 'transform': 'none'}

    target_series = target_series.dropna()
    return target_series, base_col_initial, base_col_final, target_info


def _reconstruct_prices_from_target(
    forecast_values: np.ndarray,
    price_history: Optional[np.ndarray],
    target_info: Optional[Dict[str, Any]],
) -> Optional[np.ndarray]:
    history = np.asarray(price_history, dtype=float).reshape(-1) if price_history is not None else np.asarray([], dtype=float)
    if history.size == 0:
        return None

    forecast_arr = np.asarray(forecast_values, dtype=float)
    transform = str((target_info or {}).get("transform", "log_return")).strip().lower()
    lag = 1
    if "(k=" in transform:
        try:
            lag = max(1, int(transform.rsplit("(k=", 1)[1].rstrip(") ")))
        except Exception:
            lag = 1

    if transform == "none":
        return forecast_arr.astype(float, copy=True)
    if transform == "log":
        return np.exp(forecast_arr)
    if history.size < lag or not np.all(np.isfinite(history[-lag:])):
        return None

    inverse_fn = _RECONSTRUCTION_MODES.get(
        transform.split("(")[0] if "(" in transform else transform,
    )
    if inverse_fn is None:
        logger.warning("Unknown transform %r – cannot reconstruct prices", transform)
        return None

    reconstructed: List[float] = []
    anchors = history.astype(float).tolist()
    for value in forecast_arr:
        anchor = anchors[-lag]
        fallback_anchor = anchor
        if not np.isfinite(fallback_anchor):
            fallback_anchor = next(
                (candidate for candidate in reversed(anchors) if np.isfinite(candidate)),
                float("nan"),
            )
        if not (np.isfinite(anchor) and np.isfinite(value)):
            price = float("nan")
        else:
            price = inverse_fn(anchor, float(value))
            if not np.isfinite(price):
                price = float("nan")
        anchors.append(price if np.isfinite(price) else fallback_anchor)
        reconstructed.append(price)

    return np.asarray(reconstructed, dtype=float)


def _reconstruct_price_intervals_from_target(
    forecast_values: np.ndarray,
    ci_values: Tuple[np.ndarray, np.ndarray],
    reconstructed_prices: np.ndarray,
    price_history: Optional[np.ndarray],
    target_info: Optional[Dict[str, Any]],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Map marginal target intervals to cumulative price uncertainty.

    Return/difference bounds describe uncertainty at each forecast step. Treating
    every lower (or upper) bound as a path compounds the same confidence tail at
    every horizon and greatly overstates long-horizon width. Independent marginal
    errors instead accumulate in variance space along each transform lag chain.
    """
    point = np.asarray(forecast_values, dtype=float).reshape(-1)
    lower = np.asarray(ci_values[0], dtype=float).reshape(-1)
    upper = np.asarray(ci_values[1], dtype=float).reshape(-1)
    prices = np.asarray(reconstructed_prices, dtype=float).reshape(-1)
    if not (point.size == lower.size == upper.size == prices.size):
        return None

    transform = str((target_info or {}).get("transform", "none")).strip().lower()
    base_transform = transform.split("(", 1)[0]
    if base_transform in {"none", "log"}:
        lower_prices = _reconstruct_prices_from_target(
            lower, price_history, target_info
        )
        upper_prices = _reconstruct_prices_from_target(
            upper, price_history, target_info
        )
        if lower_prices is None or upper_prices is None:
            return None
        return lower_prices, upper_prices

    lag = 1
    if "(k=" in transform:
        try:
            lag = max(1, int(transform.rsplit("(k=", 1)[1].rstrip(") ")))
        except Exception:
            lag = 1

    if base_transform == "log_return":
        lower_step = point - lower
        upper_step = upper - point
        multiplicative = True
    elif base_transform in {"return", "pct_change", "pct"}:
        scale = 100.0 if base_transform == "pct" else 1.0
        point_factor = 1.0 + point / scale
        lower_factor = 1.0 + lower / scale
        upper_factor = 1.0 + upper / scale
        if np.any(point_factor <= 0) or np.any(lower_factor <= 0) or np.any(upper_factor <= 0):
            return None
        lower_step = np.log(point_factor / lower_factor)
        upper_step = np.log(upper_factor / point_factor)
        multiplicative = True
    elif base_transform == "diff":
        lower_step = point - lower
        upper_step = upper - point
        multiplicative = False
    else:
        return None

    lower_variance = np.square(np.maximum(lower_step, 0.0))
    upper_variance = np.square(np.maximum(upper_step, 0.0))
    for index in range(lag, point.size):
        lower_variance[index] += lower_variance[index - lag]
        upper_variance[index] += upper_variance[index - lag]
    lower_width = np.sqrt(lower_variance)
    upper_width = np.sqrt(upper_variance)
    if multiplicative:
        return prices * np.exp(-lower_width), prices * np.exp(upper_width)
    return prices - lower_width, prices + upper_width


def _inverse_log_return(anchor: float, value: float) -> float:
    """log_return: price = anchor * exp(value)"""
    return anchor * float(np.exp(value))


def _inverse_return(anchor: float, value: float) -> float:
    """return / pct_change: price = anchor * (1 + value)"""
    return anchor * (1.0 + value)


def _inverse_pct(anchor: float, value: float) -> float:
    """pct: price = anchor * (1 + value/100)"""
    return anchor * (1.0 + value / 100.0)


def _inverse_diff(anchor: float, value: float) -> float:
    """diff: price = anchor + value"""
    return anchor + value


_RECONSTRUCTION_MODES = {
    "log_return": _inverse_log_return,
    "return": _inverse_return,
    "pct_change": _inverse_return,
    "pct": _inverse_pct,
    "diff": _inverse_diff,
}


def _prepare_feature_context(
    *,
    df: pd.DataFrame,
    features: Optional[Dict[str, Any]],
    exog_used: Optional[np.ndarray],
    exog_future: Optional[np.ndarray],
    tf_secs: int,
    horizon: int,
    target_series: pd.Series,
    dimred_method: Optional[str],
    dimred_params: Optional[Dict[str, Any]],
    timeframe: str = "",
    symbol: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """Prepare training and future exogenous features if requested."""
    X = exog_used
    future_exog = exog_future
    feat_info: Dict[str, Any] = {}
    if X is None and features:
        future_times = next_times_from_last(
            float(df['time'].iloc[-1]),
            int(tf_secs),
            int(horizon),
            skip_weekends=uses_standard_weekend_projection(symbol, int(tf_secs)),
            timeframe=timeframe,
            symbol=symbol,
            observed_times=df.get("time"),
        )
        try:
            X, built_future_exog, feat_info = _forecast_preprocessing.prepare_features(
                df,
                features,
                future_times,
                horizon,
                training_index=target_series.index,
                dimred_method=dimred_method,
                dimred_params=dimred_params,
                parse_kv_or_json=_parse_kv_or_json,
                reducer_factory=_forecast_preprocessing._create_dimred_reducer,
            )
        except UnknownFeatureColumnError as exc:
            X, built_future_exog, feat_info = None, None, {
                "error": str(exc),
                "error_code": exc.error_code,
                **exc.details(),
            }
        except Exception as exc:
            logger.warning("Feature preparation failed; using univariate fallback: %s", exc)
            X, built_future_exog, feat_info = None, None, {'error': f"feature_build_error: {str(exc)}"}
        if future_exog is None:
            future_exog = built_future_exog
    return X, future_exog, feat_info


def build_training_context(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    method: ForecastMethodLiteral = "theta",
    horizon: int = 12,
    lookback: Optional[int] = None,
    as_of: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    quantity: Literal["price", "return", "volatility"] = "price",
    denoise: Optional[DenoiseSpec] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    target_spec: Optional[Dict[str, Any]] = None,
    exog_used: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
    prefetched_df: Optional[pd.DataFrame] = None,
    prefetched_base_col: Optional[str] = None,
    prefetched_denoise_spec: Optional[Any] = None,
) -> TrainingExecutionContext:
    method_l = str(method).lower().strip()
    quantity_l = str(quantity).lower().strip()
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(invalid_timeframe_error(timeframe, TIMEFRAME_MAP))
    tf_secs = TIMEFRAME_SECONDS.get(timeframe)
    if not tf_secs:
        raise ValueError(unsupported_timeframe_seconds_error(timeframe))
    available_methods = _get_available_methods()
    if method_l not in available_methods:
        resolution_error = forecast_method_resolution_error(method)
        if resolution_error is not None:
            raise ValueError(resolution_error["error"])
        raise ValueError(format_invalid_method_error(method, list(available_methods)))
    if quantity_l == "volatility" or method_l.startswith("vol_"):
        raise ValueError("Use forecast_volatility for volatility models")

    dimred_error = _forecast_dimred_method_error(dimred_method)
    if dimred_error is not None:
        raise ValueError(str(dimred_error["error"]))
    feature_capability_error = _feature_method_capability_error(
        [method_l],
        features=features,
    )
    if feature_capability_error is not None:
        raise ValueError(str(feature_capability_error["error"]))

    p = _parse_kv_or_json(params)
    if method_l == "analog":
        analog_error = _cap_analog_params_to_lookback(
            p, lookback=lookback, horizon=int(horizon)
        )
        if analog_error is not None:
            raise ValueError(str(analog_error["error"]))
    seasonality = int(p.get("seasonality")) if p.get("seasonality") is not None else default_seasonality(timeframe)
    need = _calculate_lookback_bars(method_l, int(horizon), lookback, seasonality, timeframe, params=p)
    df, base_col, _ = _resolve_history_context(
        symbol=symbol,
        timeframe=timeframe,
        need=need,
        as_of=as_of,
        start=start,
        end=end,
        prefetched_df=prefetched_df,
        prefetched_base_col=prefetched_base_col,
        prefetched_denoise_spec=prefetched_denoise_spec,
        denoise=denoise,
        cap_explicit_range=lookback is not None,
    )
    if p.get("seasonality") is None and "time" in df.columns:
        seasonality = default_seasonality(timeframe, df["time"])
    target_series, _, base_col, _ = _prepare_target_series_context(
        df=df,
        quantity_l=quantity_l,
        base_col=base_col,
        features=features,
        target_spec=target_spec,
    )
    if len(target_series) < 3:
        raise ValueError(f"Not enough valid data points in column '{base_col}'")
    X, _, feature_info = _prepare_feature_context(
        df=df,
        features=features,
        exog_used=exog_used,
        exog_future=exog_future,
        tf_secs=int(tf_secs),
        horizon=int(horizon),
        target_series=target_series,
        timeframe=timeframe,
        dimred_method=dimred_method,
        dimred_params=dimred_params,
        symbol=symbol,
    )
    if features and feature_info.get("error"):
        raise ValueError(
            "Requested features could not be prepared: "
            f"{feature_info['error']}"
        )
    training_params = dict(p)
    # Seasonality is a shared engine argument passed positionally to every
    # forecaster. Do not also leak it into method-specific parameter mappings.
    training_params.pop("seasonality", None)
    training_params["_training_context"] = _training_context_fingerprint(
        df=df,
        target_series=target_series,
        base_col=base_col,
        quantity=quantity_l,
        denoise=denoise,
        features=features,
        target_spec=target_spec,
        exog=X,
        dimred_method=dimred_method,
        dimred_params=dimred_params,
        training_window_mode=(
            "as_of"
            if as_of is not None
            else "range"
            if start is not None or end is not None
            else "latest"
        ),
    )
    return TrainingExecutionContext(
        method_l=method_l,
        data_scope=f"{symbol}_{timeframe}",
        target_series=target_series,
        horizon=int(horizon),
        seasonality=int(seasonality),
        method_params=training_params,
        timeframe=str(timeframe),
        exog_used=X,
    )


def _build_engine_diagnostics(
    *,
    df: pd.DataFrame,
    need: int,
    lookback: Optional[int],
    seasonality: int,
    quantity_l: str,
    base_col: str,
    target_series: pd.Series,
) -> Dict[str, Any]:
    history_start_epoch: Optional[float]
    history_end_epoch: Optional[float]
    try:
        history_start_epoch = float(df['time'].iloc[0])
    except Exception:
        history_start_epoch = None
    try:
        history_end_epoch = float(df['time'].iloc[-1])
    except Exception:
        history_end_epoch = None

    fmt_time = _format_time_minimal_local if _use_client_tz() else _format_time_minimal
    diagnostics: Dict[str, Any] = {
        "lookback_bars_requested": int(lookback) if lookback is not None else None,
        "minimum_history_bars_requested": int(need),
        "history_bars_received": int(len(df)),
        "history_bars_used": int(len(df)),
        "target_points_used": int(len(target_series)),
        "seasonality_used": int(seasonality),
        "quantity": quantity_l,
        "base_col_used": str(base_col),
    }
    if history_start_epoch is not None:
        diagnostics["history_start_epoch"] = history_start_epoch
        diagnostics["history_start_time"] = fmt_time(history_start_epoch)
    if history_end_epoch is not None:
        diagnostics["history_end_epoch"] = history_end_epoch
        diagnostics["history_end_time"] = fmt_time(history_end_epoch)
    history_quality = df.attrs.get("history_quality")
    if isinstance(history_quality, dict):
        diagnostics["history_quality"] = dict(history_quality)
    return diagnostics


def _forecast_history_sample_quality(
    *,
    method: str,
    horizon: int,
    history_bars: int,
    lookback_requested: Optional[int] = None,
) -> Dict[str, Any]:
    recommended = max(30, 3 * max(1, int(horizon)))
    bars = max(0, int(history_bars))
    recommended_ok = bars >= recommended
    requested = (
        max(1, int(lookback_requested))
        if lookback_requested is not None
        else None
    )
    lookback_satisfied = requested is None or bars >= requested
    sample_ok = recommended_ok and lookback_satisfied
    out: Dict[str, Any] = {
        "history_sample_ok": sample_ok,
        "forecast_reliability": "adequate" if sample_ok else "low",
        "recommended_history_bars": recommended,
    }
    if requested is not None:
        out["lookback_satisfied"] = lookback_satisfied
        out["lookback_shortfall_bars"] = max(0, requested - bars)
    if not recommended_ok:
        out["history_shortfall_bars"] = recommended - bars
    if not sample_ok:
        if not recommended_ok and not lookback_satisfied:
            reason = "below_recommended_history_and_requested_lookback"
        elif not recommended_ok:
            reason = "below_recommended_history"
        else:
            reason = "requested_lookback_shortfall"
        out["forecast_reliability_reason"] = reason
        warning_parts: List[str] = []
        if not recommended_ok:
            warning_parts.append(
                f"Low-history forecast: method '{method}' used {bars} bars; at least "
                f"{recommended} are recommended for horizon {int(horizon)}."
            )
        if requested is not None and not lookback_satisfied:
            warning_parts.append(
                f"Requested lookback was not satisfied: {bars} of {requested} bars "
                f"were available ({requested - bars} bars short)."
            )
        warning_parts.append(
            "Treat the result as exploratory and validate it with "
            "forecast_backtest_run."
        )
        out["warning"] = " ".join(warning_parts)
    return out


def _compute_model_key(
    forecaster: "ForecastMethod",
    method_l: str,
    horizon: int,
    seasonality: int,
    params: Dict[str, Any],
    timeframe: str,
    has_exog: bool,
) -> str:
    """Compute a stable params_hash for the model store lookup."""
    from .interface import ForecastMethod as _FM
    fp = forecaster.training_fingerprint(
        horizon=horizon,
        seasonality=seasonality,
        params=params,
        timeframe=timeframe,
        has_exog=has_exog,
    )
    return _FM.hash_fingerprint(fp)


def _stable_training_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(key): _stable_training_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_training_value(item) for item in value]
    return value


def _training_context_fingerprint(
    *,
    df: pd.DataFrame,
    target_series: pd.Series,
    base_col: str,
    quantity: str,
    denoise: Any,
    features: Any,
    target_spec: Any,
    exog: Optional[np.ndarray],
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    training_window_mode: str = "latest",
) -> Dict[str, Any]:
    if features in (None, {}):
        features = {}
    fingerprint = {
        "target_points": int(len(target_series)),
        "history_start_epoch": float(df["time"].iloc[0]),
        "training_end_epoch": float(df["time"].iloc[-1]),
        "base_col": str(base_col),
        "denoise": _stable_training_value(denoise),
        "features": _stable_training_value(features),
        "target_spec": _stable_training_value(target_spec),
        "dimred": (
            {
                "method": str(dimred_method),
                "params": _stable_training_value(dimred_params or {}),
            }
            if dimred_method
            else None
        ),
        "exog_columns": int(exog.shape[1]) if exog is not None and exog.ndim > 1 else 0,
        "training_window_mode": str(training_window_mode),
    }
    target_transform = (
        str(target_spec.get("transform") or "").strip().lower()
        if isinstance(target_spec, dict)
        else ""
    )
    if str(quantity).strip().lower() == "return" or target_transform in {
        "log",
        "log_return",
        "pct",
        "pct_change",
        "return",
    }:
        fingerprint["invalid_target_value_policy"] = "mask_v1"
    return fingerprint


def _params_hash_from_model_id(
    model_id: str,
    *,
    method: str,
    data_scope: str,
) -> str:
    parts = str(model_id).split("/")
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(
            "model_id must use the canonical method/data_scope/params_hash format "
            "returned by forecast_train or forecast_models_list."
        )
    stored_method, stored_scope, params_hash = parts
    if stored_method != method or stored_scope != data_scope:
        raise ValueError(
            f"model_id '{model_id}' does not match requested method '{method}' "
            f"and data scope '{data_scope}'."
        )
    return params_hash


def _try_predict_with_stored_model(  # noqa: C901
    forecaster: "ForecastMethod",
    method_l: str,
    data_scope: str,
    params_hash: str,
    target_series: pd.Series,
    horizon: int,
    seasonality: int,
    method_params: Dict[str, Any],
    future_exog: Optional[np.ndarray],
    call_kwargs: Dict[str, Any],
    current_anchor_epoch: Optional[float] = None,
    *,
    require_exact_anchor: bool = False,
    timeframe_seconds: Optional[int] = None,
    max_staleness_bars: Optional[int] = None,
    rejection: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]]:
    """Attempt to load a trained model and predict. Returns None if no model found."""
    try:
        from .model_store import describe_store_metadata_compatibility
        from .model_store import model_store as _store
        handle = _store.find(method_l, data_scope, params_hash)
        if handle is None:
            if rejection is not None:
                rejection.update({"reason": "not_found"})
            return None
        trained_anchor: Optional[float] = None
        if current_anchor_epoch is not None:
            training_context = handle.metadata.get("training_context")
            if not isinstance(training_context, dict):
                if require_exact_anchor:
                    if rejection is not None:
                        rejection.update(
                            {
                                "reason": "missing_training_anchor",
                                "model_id": handle.model_id,
                            }
                        )
                    return None
            else:
                try:
                    trained_anchor = float(training_context.get("training_end_epoch"))
                    requested_anchor = float(current_anchor_epoch)
                except (TypeError, ValueError):
                    trained_anchor = None
                if trained_anchor is None:
                    if require_exact_anchor:
                        if rejection is not None:
                            rejection.update(
                                {
                                    "reason": "missing_training_anchor",
                                    "model_id": handle.model_id,
                                }
                            )
                        return None
                elif trained_anchor > requested_anchor + 1e-6:
                    if rejection is not None:
                        rejection.update(
                            {
                                "reason": "trained_after_requested_anchor",
                                "model_id": handle.model_id,
                                "trained_anchor_epoch": trained_anchor,
                                "requested_anchor_epoch": requested_anchor,
                            }
                        )
                    return None
                elif require_exact_anchor and abs(trained_anchor - requested_anchor) >= 1e-6:
                    if rejection is not None:
                        rejection.update(
                            {
                                "reason": "historical_anchor_mismatch",
                                "model_id": handle.model_id,
                                "trained_anchor_epoch": trained_anchor,
                                "requested_anchor_epoch": requested_anchor,
                            }
                        )
                    return None
                elif (
                    not require_exact_anchor
                    and timeframe_seconds is not None
                    and int(timeframe_seconds) > 0
                    and max_staleness_bars is not None
                    and (requested_anchor - trained_anchor) / float(timeframe_seconds)
                    > float(max_staleness_bars)
                ):
                    if rejection is not None:
                        rejection.update(
                            {
                                "reason": "model_staleness_limit_exceeded",
                                "model_id": handle.model_id,
                                "trained_anchor_epoch": trained_anchor,
                                "requested_anchor_epoch": requested_anchor,
                                "max_staleness_bars": int(max_staleness_bars),
                            }
                        )
                    return None
        raw = _store.load_bytes(handle.model_id)
        if raw is None:
            return None
        artifact = forecaster.deserialize_artifact(raw)
        res = forecaster.predict_with_model(
            artifact,
            target_series,
            horizon,
            seasonality,
            method_params,
            exog_future=future_exog,
            **call_kwargs,
        )
        _store.mark_used(handle.model_id)
        metadata = res.metadata or {}
        metadata['params_used'] = res.params_used
        model_info = {
            'model_id': handle.model_id,
            'trained_at': handle.created_at,
            'data_scope': handle.data_scope,
            'source': 'model_store',
            'reuse_policy': (
                'exact_training_anchor' if require_exact_anchor else 'live_latest_artifact'
            ),
        }
        if trained_anchor is not None and current_anchor_epoch is not None:
            staleness_seconds = max(0.0, float(current_anchor_epoch) - trained_anchor)
            model_info["training_end_epoch"] = trained_anchor
            model_info["model_staleness_seconds"] = staleness_seconds
            if timeframe_seconds is not None and int(timeframe_seconds) > 0:
                model_info["model_staleness_bars"] = round(
                    staleness_seconds / float(timeframe_seconds), 4
                )
        try:
            compatibility = describe_store_metadata_compatibility(handle.store_metadata)
        except Exception as compat_exc:
            logger.debug(
                "Model store compatibility inspection failed for %s: %s",
                handle.model_id,
                compat_exc,
            )
        else:
            if compatibility.get("status") == "warning":
                model_info["compatibility"] = compatibility
                warnings = metadata.get("warnings")
                if not isinstance(warnings, list):
                    warnings = []
                for warning_text in compatibility.get("warnings") or []:
                    warning_text = str(warning_text).strip()
                    if warning_text and warning_text not in warnings:
                        warnings.append(warning_text)
                if warnings:
                    metadata["warnings"] = warnings
        metadata['model_info'] = model_info
        return res.forecast, res.ci_values, metadata
    except ArtifactCompatibilityError as exc:
        logger.warning(
            "Stored model rejected for %s/%s: %s",
            method_l,
            data_scope,
            exc,
        )
        if rejection is not None:
            rejection.update(
                {
                    "reason": "artifact_runtime_incompatible",
                    "message": str(exc),
                }
            )
        return None
    except Exception as exc:
        logger.warning("Model store predict failed for %s/%s: %s", method_l, data_scope, exc)
        return None


def _submit_async_training(
    forecaster: "ForecastMethod",
    method_l: str,
    target_series: pd.Series,
    horizon: int,
    seasonality: int,
    params: Dict[str, Any],
    data_scope: str,
    params_hash: str,
    timeframe: str,
    exog: Optional[np.ndarray],
    training_window: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Submit a background training task. Returns an async response dict."""
    from .task_manager import get_task_manager

    tm = get_task_manager()
    task_id, is_new = tm.submit(
        method_name=method_l,
        series=target_series,
        horizon=horizon,
        seasonality=seasonality,
        params=params,
        data_scope=data_scope,
        exog=exog,
        timeframe=timeframe,
        training_window=training_window,
    )

    category = getattr(forecaster, "training_category", "unknown")
    duration_hint = {
        "heavy": "1-10 minutes (GPU training)",
        "moderate": "10-60 seconds",
        "fast": "1-10 seconds",
    }.get(category, "varies")

    return {
        "status": "pending" if is_new else "running",
        "task_id": task_id,
        "method": method_l,
        "data_scope": data_scope,
        "estimated_duration": duration_hint,
        "next_step": (
            f"Poll forecast_task_status(task_id='{task_id}') for progress. "
            f"Once complete, call forecast_generate again — the trained model will be used automatically."
        ),
    }


def _run_registered_forecast_method(
    *,
    method_l: str,
    method: ForecastMethodLiteral,
    df: pd.DataFrame,
    target_series: pd.Series,
    horizon: int,
    seasonality: int,
    params: Dict[str, Any],
    ci_alpha: Optional[float],
    as_of: Optional[str],
    training_window_mode: Optional[str] = None,
    quantity_l: str,
    symbol: str,
    timeframe: TimeframeLiteral,
    base_col: str,
    denoise_spec_used: Optional[Any],
    X: Optional[np.ndarray],
    future_exog: Optional[np.ndarray],
    features: Optional[Dict[str, Any]] = None,
    feature_info: Optional[Dict[str, Any]] = None,
    target_spec: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    training_window: Optional[Dict[str, Any]] = None,
    async_mode: bool = False,
    model_id: Optional[str] = None,
    model_cache: Literal["reuse", "ephemeral", "require_existing"] = "reuse",
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    training_window_mode = str(
        training_window_mode or ("as_of" if as_of is not None else "latest")
    ).strip().lower()
    forecaster = ForecastRegistry.get(method_l)
    method_params = dict(params)
    declared_params = getattr(forecaster, "PARAMS", ())
    accepts_ci_param = any(
        isinstance(spec, dict) and spec.get("name") == "ci_alpha"
        for spec in declared_params
    )
    if ci_alpha is not None and accepts_ci_param and 'ci_alpha' not in method_params:
        method_params['ci_alpha'] = ci_alpha
    requested_model_id = str(model_id or "").strip()
    cache_policy = str(model_cache or "reuse").strip().lower()
    if cache_policy not in {"reuse", "ephemeral", "require_existing"}:
        raise ValueError(
            "model_cache must be one of: reuse, ephemeral, require_existing"
        )
    if cache_policy == "ephemeral" and requested_model_id:
        raise ValueError("model_id cannot be used with model_cache='ephemeral'")
    if cache_policy != "reuse" and async_mode:
        raise ValueError(
            "async_mode requires model_cache='reuse' because background "
            "training persists its artifact"
        )
    supports_training = bool(getattr(forecaster, 'supports_training', False))
    if async_mode and not supports_training:
        raise ValueError(
            f"async_mode is not supported for non-trainable method '{method_l}'. "
            "Omit async_mode or select a trainable method."
        )
    if requested_model_id and not supports_training:
        raise ValueError(
            f"model_id '{requested_model_id}' was provided, but method "
            f"'{method_l}' does not support stored model prediction. "
            "Use forecast_models_list to inspect stored trainable models, or omit model_id."
        )

    call_kwargs: Dict[str, Any] = {
        'ci_alpha': ci_alpha,
        'as_of': as_of,
        'quantity': quantity_l,
        'timeframe': timeframe,
    }
    if X is not None:
        call_kwargs['exog_used'] = X
    call_context = ForecastCallContext(
        method=method_l,
        symbol=symbol,
        timeframe=str(timeframe),
        quantity=quantity_l,
        horizon=int(horizon),
        seasonality=int(seasonality),
        base_col=str(base_col),
        ci_alpha=ci_alpha,
        as_of=as_of,
        denoise_spec_used=denoise_spec_used,
        history_df=df,
        target_series=target_series,
        exog_used=X,
        future_exog=future_exog,
        features=dict(features) if isinstance(features, dict) else None,
        feature_info=dict(feature_info) if isinstance(feature_info, dict) else None,
    )
    prepare_call = getattr(forecaster, "prepare_forecast_call", None)
    if callable(prepare_call):
        method_params, call_kwargs = prepare_call(
            method_params,
            call_kwargs,
            call_context,
        )

    # --- Model store fast path ---
    if supports_training:
        data_scope = f"{symbol}_{timeframe}"
        has_exog = X is not None
        training_params = dict(method_params)
        training_params["quantity"] = quantity_l
        training_params["_training_context"] = _training_context_fingerprint(
            df=df,
            target_series=target_series,
            base_col=base_col,
            quantity=quantity_l,
            denoise=denoise_spec_used,
            features=features,
            target_spec=target_spec,
            exog=X,
            dimred_method=dimred_method,
            dimred_params=dimred_params,
            training_window_mode=training_window_mode,
        )
        compatibility_fingerprint = forecaster.training_fingerprint(
            horizon=horizon,
            seasonality=seasonality,
            params=training_params,
            timeframe=str(timeframe),
            has_exog=has_exog,
        )
        expected_params_hash = _compute_model_key(
            forecaster, method_l, horizon, seasonality,
            training_params, str(timeframe), has_exog,
        )
        if requested_model_id:
            supplied_params_hash = _params_hash_from_model_id(
                requested_model_id,
                method=method_l,
                data_scope=data_scope,
            )
            if supplied_params_hash != expected_params_hash:
                from .model_store import model_store as _store

                supplied_handle = _store.find(
                    method_l,
                    data_scope,
                    supplied_params_hash,
                )
                if supplied_handle is None:
                    raise ValueError(
                        f"Model with ID '{requested_model_id}' was not found in the "
                        "model store. Use forecast_models_list to see available models."
                    )
                stored_fingerprint = supplied_handle.metadata.get(
                    "compatibility_fingerprint"
                )
                mismatches = fingerprint_mismatches(
                    stored_fingerprint,
                    compatibility_fingerprint,
                )
                raise ModelCompatibilityError(
                    f"model_id '{requested_model_id}' is incompatible with the "
                    "requested forecast identity.",
                    model_id=requested_model_id,
                    stored_fingerprint=stored_fingerprint,
                    requested_fingerprint=compatibility_fingerprint,
                    mismatches=mismatches,
                )
            params_hash = supplied_params_hash
        else:
            params_hash = expected_params_hash

        model_rejection: Dict[str, Any] = {}
        live_model_update = (
            training_window_mode == "latest"
            and bool(getattr(forecaster, "supports_live_model_update", False))
        )
        if cache_policy != "ephemeral":
            stored_result = _try_predict_with_stored_model(
                forecaster, method_l, data_scope, params_hash,
                target_series, horizon, seasonality,
                method_params, future_exog, call_kwargs,
                float(df["time"].iloc[-1]),
                require_exact_anchor=not live_model_update,
                timeframe_seconds=TIMEFRAME_SECONDS.get(str(timeframe)),
                max_staleness_bars=max(1, int(seasonality)),
                rejection=model_rejection,
            )
            if stored_result is not None:
                return stored_result
        if requested_model_id:
            reason = str(model_rejection.get("reason") or "unloadable")
            if reason != "not_found":
                anchor_detail = ""
                if model_rejection.get("trained_anchor_epoch") is not None:
                    anchor_detail = (
                        f" Trained anchor={model_rejection['trained_anchor_epoch']}; "
                        f"requested anchor={model_rejection.get('requested_anchor_epoch')}."
                    )
                anchor_policy = (
                    "Historical forecasts and methods without live history refresh "
                    "require an exact training anchor; live forecasts reject artifacts "
                    "trained after the request."
                )
                raise ValueError(
                    f"Model with ID '{requested_model_id}' exists but was rejected: "
                    f"{reason}.{anchor_detail} {anchor_policy}"
                )
            raise ValueError(
                f"Model with ID '{requested_model_id}' was not found in the model store. "
                "Use forecast_models_list to see available models."
            )
        if cache_policy == "require_existing":
            reason = str(model_rejection.get("reason") or "not_found")
            raise ValueError(
                "model_cache='require_existing' found no compatible stored model "
                f"for method '{method_l}' and data scope '{data_scope}' "
                f"(reason: {reason})."
            )

        # No stored model — async route for any trainable method when requested
        if async_mode:
            # Pass training_params (includes quantity) so TaskManager's recomputed
            # hash matches the model-store key used above.
            async_resp = _submit_async_training(
                forecaster, method_l, target_series,
                horizon, seasonality, training_params,
                data_scope, params_hash, str(timeframe), X,
                training_window,
            )
            raise _AsyncTrainingStarted(async_resp)

    # --- Default synchronous path ---
    if supports_training:
        training_context = training_params.pop("_training_context", None)
        trained = forecaster.train(
            target_series,
            horizon,
            seasonality,
            training_params,
            exog=X,
            timeframe=str(timeframe),
        )
        handle = None
        if cache_policy != "ephemeral":
            from .model_store import model_store as _store

            handle = _store.save(
                method=method_l,
                data_scope=data_scope,
                params_hash=params_hash,
                artifact_bytes=trained.artifact_bytes,
                metadata={
                    **(trained.metadata or {}),
                    "params_used": trained.params_used,
                    "source_task_id": None,
                    "training_context": training_context,
                    **(
                        {"training_window": dict(training_window)}
                        if training_window
                        else {}
                    ),
                    **build_model_reuse_metadata(
                        compatibility_fingerprint,
                        data_scope,
                        training_window,
                    ),
                },
            )
        artifact = forecaster.deserialize_artifact(trained.artifact_bytes)
        res = forecaster.predict_with_model(
            artifact,
            target_series,
            horizon,
            seasonality,
            method_params,
            exog_future=future_exog,
            **call_kwargs,
        )
        metadata = {**(trained.metadata or {}), **(res.metadata or {})}
        metadata["params_used"] = res.params_used
        if cache_policy == "ephemeral":
            metadata["model_info"] = {
                "model_id": None,
                "data_scope": data_scope,
                "source": "ephemeral_training",
                "reuse_policy": "not_persisted",
                "training_end_epoch": float(df["time"].iloc[-1]),
                "model_staleness_seconds": 0.0,
                "model_staleness_bars": 0.0,
            }
        else:
            assert handle is not None
            metadata["model_info"] = {
                "model_id": handle.model_id,
                "trained_at": handle.created_at,
                "data_scope": handle.data_scope,
                "source": "synchronous_training",
                "reuse_policy": "live_latest_artifact",
                "training_end_epoch": float(df["time"].iloc[-1]),
                "model_staleness_seconds": 0.0,
                "model_staleness_bars": 0.0,
            }
        return res.forecast, res.ci_values, metadata

    res = forecaster.forecast(
        target_series,
        horizon,
        seasonality,
        method_params,
        exog_future=future_exog,
        **call_kwargs,
    )
    metadata = res.metadata or {}
    metadata['params_used'] = res.params_used
    return res.forecast, res.ci_values, metadata


def _merge_engine_diagnostics(metadata: Dict[str, Any], diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        metadata = {}
    existing_diagnostics = metadata.get("diagnostics")
    if not isinstance(existing_diagnostics, dict):
        existing_diagnostics = {}
    merged_diagnostics = dict(existing_diagnostics)
    for key, value in diagnostics.items():
        if key not in merged_diagnostics:
            merged_diagnostics[key] = value
    metadata["diagnostics"] = merged_diagnostics
    return metadata


def _feature_consumption_attestation_error(
    metadata: Dict[str, Any],
    *,
    horizon: int,
) -> Optional[str]:
    """Cross-check prepared exogenous inputs with adapter runtime evidence."""
    diagnostics = metadata.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return "forecast diagnostics are missing"
    prepared = diagnostics.get("feature_preparation")
    consumed = diagnostics.get("feature_consumption")
    if not isinstance(prepared, dict):
        return "feature preparation diagnostics are missing"
    if not isinstance(consumed, dict):
        return "runtime feature-consumption attestation is missing"

    selected_columns = prepared.get("selected_columns")
    try:
        prepared_count = int(prepared.get("n_features"))
    except (TypeError, ValueError):
        return "prepared feature count is missing or invalid"
    if (
        prepared_count <= 0
        or not isinstance(selected_columns, list)
        or len(selected_columns) != prepared_count
    ):
        return "selected feature columns do not match prepared feature count"

    if consumed.get("status") != "consumed":
        return "runtime feature-consumption status is not consumed"
    if consumed.get("historical_consumed") is not True:
        return "historical exogenous inputs were not attested as consumed"
    if consumed.get("future_consumed") is not True:
        return "future exogenous inputs were not attested as consumed"
    try:
        consumed_count = int(consumed.get("n_features"))
        historical_rows = int(consumed.get("historical_rows"))
        future_rows = int(consumed.get("future_rows"))
        target_points = int(diagnostics.get("target_points_used"))
    except (TypeError, ValueError):
        return "runtime feature-consumption row or column counts are invalid"
    if consumed_count != prepared_count:
        return "adapter feature count differs from prepared feature count"
    adapter_columns = consumed.get("adapter_columns")
    if adapter_columns != [f"x{index}" for index in range(prepared_count)]:
        return "adapter columns do not match generic prepared feature identity"
    if historical_rows != target_points:
        return "historical exogenous row count differs from target row count"
    if future_rows != int(horizon):
        return "future exogenous row count differs from forecast horizon"
    return None


def _last_price_freshness_fields(
    *,
    last_epoch: float,
    tf_secs: int,
    now_epoch: Optional[float] = None,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    timeframe = next(
        (
            name
            for name, seconds in TIMEFRAME_SECONDS.items()
            if int(seconds) == int(tf_secs)
        ),
        None,
    )
    if timeframe is None:
        return {}
    freshness = completed_bar_freshness_fields(
        symbol,
        timeframe,
        last_epoch,
        now_epoch=now_epoch,
        item="forecast anchor",
        tolerance_bars=SANITY_BARS_TOLERANCE,
    )
    if not freshness:
        return {}
    rounded_age = int(freshness["data_age_seconds"])
    out: Dict[str, Any] = {
        "last_price_age_seconds": rounded_age,
        "last_price_stale": bool(freshness["data_stale"]),
        "freshness_basis": freshness["freshness_basis"],
        "freshness_age_metric": freshness["freshness_age_metric"],
        "last_observation_close_epoch": freshness["data_as_of_epoch"],
        "stale_after_seconds": freshness["stale_after_seconds"],
    }
    age_text = format_age_seconds(rounded_age)
    if age_text:
        out["last_price_age"] = age_text
    for key in (
        "market_status",
        "market_status_reason",
        "market_status_source",
        "note",
        "freshness_policy_relaxed",
        "assumed_closure_start",
        "assumed_closure_end",
        "assumed_closure_seconds",
        "history_policy_ok",
    ):
        if key in freshness:
            out[key] = freshness[key]
    if out["last_price_stale"]:
        out["stale_warning"] = (
            "Last forecast anchor is older than the bar freshness policy; "
            "market may be closed or broker data may be stale."
        )
    return out


def _forecast_direction_threshold_from_history(
    price_anchors: Any,
    horizon: int,
) -> Tuple[float, str]:
    minimum_pct = 0.05
    anchor_values = np.asarray(price_anchors, dtype=float)
    if anchor_values.size <= 1:
        return minimum_pct, "minimum_effect_size_0.05_pct"
    previous_values = anchor_values[:-1]
    valid_returns = np.isfinite(previous_values) & (previous_values != 0.0)
    if not np.any(valid_returns):
        return minimum_pct, "minimum_effect_size_0.05_pct"
    absolute_returns_pct = np.abs(
        (anchor_values[1:][valid_returns] - previous_values[valid_returns])
        / previous_values[valid_returns]
        * 100.0
    )
    median_bar_move_pct = float(np.median(absolute_returns_pct))
    horizon_noise_pct = median_bar_move_pct * float(
        np.sqrt(max(1, int(horizon)))
    )
    return (
        max(minimum_pct, horizon_noise_pct),
        "max(0.05_pct,median_abs_bar_return_pct*sqrt(horizon))",
    )


def _forecast_target_bar_states(
    future_epochs: List[float],
    tf_secs: int,
    *,
    now_epoch: Optional[float] = None,
) -> List[str]:
    current_epoch = (
        datetime.now(timezone.utc).timestamp()
        if now_epoch is None
        else float(now_epoch)
    )
    step_seconds = max(1, int(tf_secs))
    states: List[str] = []
    for value in future_epochs:
        bar_open = float(value)
        if current_epoch < bar_open:
            states.append("future")
        elif current_epoch < bar_open + step_seconds:
            states.append("forming")
        else:
            states.append("closed")
    return states


def _forecast_bar_state_reference_epoch(
    as_of: Optional[str],
    *,
    timeframe: Optional[str] = None,
    historical_cutoff_epoch: Optional[float] = None,
) -> float:
    if as_of:
        parsed = _parse_as_of_bound(as_of, timeframe=timeframe)
        if parsed is not None:
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            return float(parsed.timestamp())
    if historical_cutoff_epoch is not None:
        return float(historical_cutoff_epoch)
    return float(datetime.now(timezone.utc).timestamp())


def _forecast_session_projection_metadata(
    enabled: bool,
    *,
    horizon: int,
    tf_secs: int,
) -> Dict[str, Any]:
    if not enabled:
        return {}
    return {
        "forecast_nominal_step_seconds": int(tf_secs),
        "horizon_note": (
            f"{horizon} exchange-session bars forecast; overnight, weekend, "
            "holiday, and shortened-session closures do not count toward the horizon."
        ),
    }


def _attach_reconstructed_price_interval(
    result: Dict[str, Any],
    reconstructed_price_ci: Optional[Tuple[np.ndarray, np.ndarray]],
) -> None:
    if reconstructed_price_ci is None:
        return
    result["lower_price"] = [float(v) for v in reconstructed_price_ci[0]]
    result["upper_price"] = [float(v) for v in reconstructed_price_ci[1]]


def _format_forecast_output(
    forecast_values: np.ndarray,
    last_epoch: float,
    tf_secs: int,
    horizon: int,
    base_col: str,
    df: pd.DataFrame,
    ci_alpha: Optional[float],
    ci_values: Optional[np.ndarray],
    method: str,
    quantity: str,
    denoise_used: bool,
    metadata: Optional[Dict[str, Any]] = None,
    digits: Optional[int] = None,
    forecast_return_values: Optional[np.ndarray] = None,
    reconstructed_prices: Optional[np.ndarray] = None,
    reconstructed_price_ci: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    target_info: Optional[Dict[str, Any]] = None,
    last_target_value: Optional[float] = None,
    now_epoch: Optional[float] = None,
) -> Dict[str, Any]:
    """Format forecast output with proper structure."""
    # Generate future time indices
    observed_times = df.get("time")
    session_projection = uses_exchange_intraday_projection(
        symbol,
        tf_secs,
        observed_times=observed_times,
    )
    weekend_projection = uses_standard_weekend_projection(symbol, tf_secs)
    future_epochs = next_times_from_last(
        last_epoch,
        tf_secs,
        horizon,
        skip_weekends=weekend_projection,
        timeframe=timeframe,
        symbol=symbol,
        observed_times=observed_times,
    )
    use_client_tz = _use_client_tz()
    client_tz = _resolve_client_tz() if use_client_tz else None
    fmt_time = _format_time_minimal_local if use_client_tz else _format_time_minimal
    timezone_label = display_timezone_label(
        use_client_tz=use_client_tz,
        client_tz=client_tz,
        fallback="local",
    )
    forecast_times = [fmt_time(float(epoch)) for epoch in future_epochs]
    forecast_bar_states = _forecast_target_bar_states(
        future_epochs,
        tf_secs,
        now_epoch=now_epoch,
    )
    last_bar_open_time = fmt_time(float(last_epoch))
    calendar_timeframe = str(timeframe or "").upper() in CALENDAR_TIMEFRAMES
    if calendar_timeframe or session_projection:
        calendar_gaps, skipped_bars = [], 0
    else:
        calendar_gaps, skipped_bars = _forecast_calendar_gap_rows(
            [float(last_epoch), *future_epochs],
            tf_secs,
            fmt_time,
        )
    price_anchor_series = df["close"] if "close" in df.columns else df[base_col]
    price_anchor_numeric = pd.to_numeric(price_anchor_series, errors="coerce")
    finite_price_anchors = price_anchor_numeric[np.isfinite(price_anchor_numeric)]
    last_price = (
        float(finite_price_anchors.iloc[-1])
        if len(finite_price_anchors) > 0
        else None
    )
    direction_threshold_pct, direction_threshold_basis = (
        _forecast_direction_threshold_from_history(finite_price_anchors, horizon)
    )

    # Build base result
    forecast_start_epoch = float(future_epochs[0]) if future_epochs else None
    forecast_start_gap_bars = (
        1.0
        if (
            calendar_timeframe or session_projection or weekend_projection
        ) and forecast_start_epoch is not None
        else float(forecast_start_epoch - float(last_epoch)) / float(tf_secs)
        if forecast_start_epoch is not None and tf_secs
        else None
    )
    current_epoch = (
        datetime.now(timezone.utc).timestamp()
        if now_epoch is None
        else float(now_epoch)
    )
    observation_close_epoch = (
        bar_close_epoch(float(last_epoch), timeframe)
        if timeframe
        else float(last_epoch) + float(tf_secs)
    )
    last_observation_time = fmt_time(float(observation_close_epoch))
    result: Dict[str, Any] = {
        "success": True,
        "method": method,
        "horizon": horizon,
        "base_col": base_col,
        "last_observation_epoch": float(observation_close_epoch),
        "last_bar_open_epoch": float(last_epoch),
        "last_bar_open": last_bar_open_time,
        "last_observation_time": last_observation_time,
        "data_as_of": last_observation_time,
        "bar_timestamp_basis": "mt5_bar_open",
        "last_bar_complete": observation_close_epoch <= current_epoch,
        "timezone": timezone_label,
        "forecast_from": {
            "time": last_observation_time,
            "anchor": "last_completed_bar_close",
        },
        "forecast_start_epoch": forecast_start_epoch,
        "forecast_start_time": forecast_times[0] if forecast_times else None,
        "forecast_start_gap_bars": forecast_start_gap_bars,
        "forecast_start_gap_note": (
            "Bars from the last closed observation to the first forecast; "
            "1.0 means the next timeframe bar."
        ),
        "forecast_anchor": "next_timeframe_bar_after_last_observation",
        "forecast_step_seconds": (
            None if calendar_timeframe or session_projection else int(tf_secs)
        ),
        "forecast_epoch": future_epochs,
        "forecast_time": forecast_times,
        "forecast_bar_states": forecast_bar_states,
        "horizon_includes_forming_bar": "forming" in forecast_bar_states,
        "forecast_time_semantics": "target_bar_open_time",
        "forecast_value_semantics": (
            "target_bar_log_return_and_reconstructed_close"
            if quantity == "return"
            else "target_bar_close"
            if quantity == "price"
            else "target_bar_value"
        ),
        "last_price": last_price,
        "last_price_source": "candle_close" if last_price is not None else None,
        "price_basis": (
            symbol_candle_price_basis_for(symbol) if last_price is not None else None
        ),
        "direction_threshold_pct": float(round(direction_threshold_pct, 6)),
        "direction_threshold_basis": direction_threshold_basis,
        "calendar_treatment": describe_forecast_calendar_treatment(
            symbol,
            tf_secs,
            calendar_timeframe=calendar_timeframe,
            observed_times=observed_times,
        ),
    }
    result.update(
        _forecast_session_projection_metadata(
            session_projection,
            horizon=horizon,
            tf_secs=tf_secs,
        )
    )
    if calendar_gaps:
        result["forecast_calendar_gaps"] = calendar_gaps
        result["horizon_note"] = (
            f"{horizon} trading bars forecast; {skipped_bars} "
            f"{str(timeframe or '').upper() or 'timeframe'} bars skipped (weekend)."
        )
    elif (
        not weekend_projection
        and not is_probably_crypto_symbol(symbol)
    ):
        weekend_bars = _count_weekend_forecast_times(forecast_times)
        if weekend_bars:
            result.setdefault("warnings", []).append(
                f"{weekend_bars} of {horizon} forecast timestamps fall on a weekend; "
                "the broker session schedule is unavailable, so calendar-timeframe "
                "targets are estimates and may not correspond to tradable bars."
            )

    # Choose which arrays to expose. Custom targets retain their own semantic
    # identity instead of being mislabeled as close prices or returns.
    custom_target = str((target_info or {}).get("mode") or "") == "custom"
    if custom_target:
        quantity = "custom"
        result.pop("last_price", None)
        result.pop("last_price_source", None)
        result.pop("price_basis", None)
        result.pop("direction_threshold_pct", None)
        result.pop("direction_threshold_basis", None)
        result["forecast_target"] = [float(v) for v in forecast_values]
        result["target"] = dict(target_info or {})
        result["target_quantity"] = (target_info or {}).get("quantity")
        result["target_units"] = (target_info or {}).get("units")
        result["last_target"] = last_target_value
        result["forecast_value_semantics"] = "target_bar_custom_transformed_value"
    elif quantity == 'return':
        if forecast_return_values is None:
            forecast_return_values = forecast_values
        result["forecast_return"] = [float(v) for v in forecast_return_values]
        if reconstructed_prices is not None:
            result["forecast_price"] = [float(v) for v in reconstructed_prices]
    elif reconstructed_prices is not None:
        result["forecast_price"] = [float(v) for v in reconstructed_prices]
    else:
        result["forecast_price"] = [float(v) for v in forecast_values]
    
    if digits is not None:
        result["digits"] = int(digits)

    # Add confidence intervals if available. If they are requested but missing,
    # surface an explicit warning to avoid misleading point-only interpretation.
    interval_remediation = forecast_interval_recovery(target_info).get(
        "remediation", "Use forecast_conformal_intervals for residual-quantile uncertainty bands."
    )
    if ci_alpha is not None:
        ci_alpha_value: Optional[float] = None
        try:
            ci_alpha_value = float(ci_alpha)
        except Exception:
            ci_alpha_value = None
        if ci_alpha_value is not None:
            result["ci_alpha"] = ci_alpha_value

        if ci_values is not None and len(ci_values) == 2:  # [lower, upper]
            result["ci_status"] = "available"
            result["ci_available"] = True
            lower_vals = [float(v) for v in ci_values[0]]
            upper_vals = [float(v) for v in ci_values[1]]
            if custom_target:
                result["lower_target"] = lower_vals
                result["upper_target"] = upper_vals
                result["lower"] = lower_vals
                result["upper"] = upper_vals
            elif quantity == 'return':
                result["lower_return"] = lower_vals
                result["upper_return"] = upper_vals
                # Keep generic keys for lightweight renderers expecting non-price intervals.
                result["lower"] = lower_vals
                result["upper"] = upper_vals
                _attach_reconstructed_price_interval(result, reconstructed_price_ci)
            else:
                if reconstructed_price_ci is not None:
                    result["lower_price"] = [
                        float(v) for v in reconstructed_price_ci[0]
                    ]
                    result["upper_price"] = [
                        float(v) for v in reconstructed_price_ci[1]
                    ]
                else:
                    result["lower_price"] = lower_vals
                    result["upper_price"] = upper_vals
        else:
            if ci_alpha_value is not None:
                warning_text = (
                    f"ci_alpha={ci_alpha_value:g} was requested but confidence intervals "
                    f"are unavailable for method '{method}'; returning a point forecast only. "
                    f"{interval_remediation}"
                )
            else:
                warning_text = (
                    f"Point forecast only for method '{method}'; confidence intervals are unavailable. "
                    f"{interval_remediation}"
                )
            warning_text = str((metadata or {}).get("ci_unavailable_reason") or warning_text)
            warnings = result.get("warnings")
            if not isinstance(warnings, list):
                warnings = []
            warnings.append(warning_text)
            result["warnings"] = warnings
            result["ci_status"] = "unavailable"
            result["ci_available"] = False

    # Add metadata
    result.update({
        "quantity": quantity,
        "denoise_applied": denoise_used,
    })
    
    if metadata:
        result.update(metadata)

    if (
        uses_standard_weekend_projection(symbol, tf_secs)
        and forecast_times
    ):
        market_status = [_forex_forecast_market_status(epoch) for epoch in future_epochs]
        weekend_count = sum(1 for status in market_status if status == "closed_weekend")
        if weekend_count:
            result["forecast_market_status"] = market_status
            result["open_market_forecast_bars"] = int(len(forecast_times) - weekend_count)
            result["closed_market_forecast_bars"] = weekend_count
            note = (
                f"{weekend_count} of {len(forecast_times)} forecast bars fall on "
                "Saturday/Sunday for a forex symbol; treat those timestamps as "
                "closed-market placeholders."
            )
            warnings = result.get("warnings")
            if not isinstance(warnings, list):
                warnings = [] if warnings in (None, "", [], {}) else [warnings]
            if note not in warnings:
                warnings.append(note)
            result["warnings"] = warnings
            result["market_hours_note"] = note

    return result


def forecast_engine(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    method: ForecastMethodLiteral = "theta",
    horizon: int = 12,
    lookback: Optional[int] = None,
    as_of: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    ci_alpha: Optional[float] = 0.05,
    quantity: Literal['price','return','volatility'] = 'price',
    denoise: Optional[DenoiseSpec] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    target_spec: Optional[Dict[str, Any]] = None,
    exog_used: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
    prefetched_df: Optional[pd.DataFrame] = None,
    prefetched_base_col: Optional[str] = None,
    prefetched_denoise_spec: Optional[Any] = None,
    async_mode: bool = False,
    model_id: Optional[str] = None,
    model_cache: Literal["reuse", "ephemeral", "require_existing"] = "reuse",
) -> Dict[str, Any]:
    """Core forecast engine implementation.

    This is the main orchestration function that coordinates all forecasting operations.
    """
    try:
        ci_values = None
        # Coerce CLI string inputs to proper types
        try:
            horizon = int(horizon) if horizon is not None else 12
        except (ValueError, TypeError):
            horizon = 12
            
        try:
            lookback = int(lookback) if lookback is not None else None
        except (ValueError, TypeError):
            lookback = None
        
        # Validation
        if timeframe not in TIMEFRAME_MAP:
            return {"error": invalid_timeframe_error(timeframe, TIMEFRAME_MAP)}
        tf_secs = TIMEFRAME_SECONDS.get(timeframe)
        if not tf_secs:
            return {"error": unsupported_timeframe_seconds_error(timeframe)}

        symbol, symbol_requested = resolve_forecast_symbol(symbol)

        method_l = str(method).lower().strip()
        quantity_l = str(quantity).lower().strip()
        
        # Refresh available methods
        available_methods = _get_available_methods()
        if method_l not in available_methods:
            resolution_error = forecast_method_resolution_error(method)
            if resolution_error is not None:
                return resolution_error
            return {
                "error": format_invalid_method_error(method, list(available_methods)),
                "error_code": "invalid_method",
            }

        # Volatility models have a dedicated endpoint
        if quantity_l == 'volatility' or method_l.startswith('vol_'):
            return {"error": "Use forecast_volatility for volatility models"}

        dimred_error = _forecast_dimred_method_error(dimred_method)
        if dimred_error is not None:
            return dimred_error
        feature_capability_error = _feature_method_capability_error(
            [method_l],
            features=features,
        )
        if feature_capability_error is not None:
            return feature_capability_error

        # Parse method params
        p = _parse_kv_or_json(params)
        if method_l == "analog":
            from .methods.analog import validate_analog_similarity_settings

            try:
                validate_analog_similarity_settings(p)
            except ValueError as ex:
                return {
                    "error": str(ex),
                    "error_code": "invalid_analog_similarity_settings",
                }
            analog_error = _cap_analog_params_to_lookback(
                p, lookback=lookback, horizon=int(horizon)
            )
            if analog_error is not None:
                return analog_error
        seasonality = int(p.get('seasonality')) if p.get('seasonality') is not None else default_seasonality(timeframe)

        if method_l == 'seasonal_naive' and (not seasonality or seasonality <= 0):
            return {"error": "seasonal_naive requires a positive 'seasonality' in params or auto period"}

        # Calculate lookback bars
        need = _calculate_lookback_bars(method_l, horizon, lookback, seasonality, timeframe, params=p)

        # Fetch data (or reuse prefetched) and optional denoise
        try:
            df, base_col, dn_spec_used = _resolve_history_context(
                symbol=symbol,
                timeframe=timeframe,
                need=need,
                as_of=as_of,
                start=start,
                end=end,
                prefetched_df=prefetched_df,
                prefetched_base_col=prefetched_base_col,
                prefetched_denoise_spec=prefetched_denoise_spec,
                denoise=denoise,
                cap_explicit_range=lookback is not None,
            )
        except ValueError as ex:
            return {"error": str(ex)}
        except Exception as ex:
            return {"error": str(ex)}
        if p.get("seasonality") is None and "time" in df.columns:
            seasonality = default_seasonality(timeframe, df["time"])
        history_warnings = df.attrs.get("warnings")
        if not isinstance(history_warnings, list):
            history_warnings = []
        denoise_warnings = consume_denoise_warnings(df)

        # Prepare target series, honoring target_spec if provided
        try:
            target_series, base_col_initial, base_col, target_info = _prepare_target_series_context(
                df=df,
                quantity_l=quantity_l,
                base_col=base_col,
                features=features,
                target_spec=target_spec,
            )
        except Exception as ex:
            return {"error": f"Invalid target_spec: {ex}"}

        if len(target_series) < 3:
            return {"error": f"Not enough valid data points in column '{base_col}'"}

        price_anchor_base = str((target_info or {}).get("base") or base_col)
        if quantity_l == "return" and str(base_col).startswith("__"):
            price_anchor_base = base_col_initial
        if price_anchor_base in df.columns:
            price_anchor_history = df[price_anchor_base].astype(float).to_numpy()
        else:
            alias_inputs = {
                name: df[name].to_numpy()
                for name in ("open", "high", "low", "close")
                if name in df.columns
            }
            price_anchor_history = resolve_alias_base(alias_inputs, price_anchor_base)

        # Prepare feature matrices if applicable (only if exog_used not provided).
        X, future_exog, feature_info = _prepare_feature_context(
            df=df,
            features=features,
            exog_used=exog_used,
            exog_future=exog_future,
            tf_secs=tf_secs,
            horizon=horizon,
            target_series=target_series,
            timeframe=timeframe,
            dimred_method=dimred_method,
            dimred_params=dimred_params,
            symbol=symbol,
        )
        if features and feature_info.get("error"):
            error_code = str(feature_info.get("error_code") or "feature_build_error")
            error_text = str(feature_info.get("error") or "")
            if error_code == "unknown_feature_column":
                payload = {
                    "error": error_text,
                    "error_code": error_code,
                }
            else:
                payload = {
                    "error": (
                        "Requested features could not be prepared: "
                        f"{error_text}"
                    ),
                    "error_code": error_code,
                }
            for key in ("unknown_columns", "available_columns"):
                value = feature_info.get(key)
                if value not in (None, "", [], {}):
                    payload[key] = value
            return payload

        # Get last timestamp and values
        last_epoch = float(df['time'].iloc[-1])

        # Core run diagnostics to make model context explicit for users.
        engine_diagnostics = _build_engine_diagnostics(
            df=df,
            need=need,
            lookback=lookback,
            seasonality=seasonality,
            quantity_l=quantity_l,
            base_col=base_col,
            target_series=target_series,
        )
        sample_quality = _forecast_history_sample_quality(
            method=method_l,
            horizon=horizon,
            history_bars=len(df),
            lookback_requested=lookback,
        )
        engine_diagnostics.update(
            {
                key: value
                for key, value in sample_quality.items()
                if key != "warning"
            }
        )
        if feature_info:
            engine_diagnostics["feature_preparation"] = feature_info
        broker_time_check_result: Optional[Dict[str, Any]] = None
        broker_time_check_enabled = bool(getattr(mt5_config, "broker_time_check_enabled", False))
        broker_time_check_ttl_seconds = int(getattr(mt5_config, "broker_time_check_ttl_seconds", 60))
        if (
            broker_time_check_enabled
            and prefetched_df is None
            and as_of is None
            and start is None
            and end is None
        ):
            try:
                broker_time_check_result = get_cached_mt5_time_alignment(
                    symbol=symbol,
                    probe_timeframe='M1',
                    ttl_seconds=broker_time_check_ttl_seconds,
                )
            except Exception as exc:
                broker_time_check_result = {
                    "symbol": str(symbol),
                    "probe_timeframe": "M1",
                    "status": "unavailable",
                    "reason": "inspection_failed",
                    "error": str(exc),
                }
            engine_diagnostics["broker_time_check"] = broker_time_check_result

        # Get symbol info for digits
        digits = None
        try:
            s_info = get_symbol_info_cached(symbol)
            if s_info:
                digits = s_info.digits
        except Exception:
            pass

        # Call engine
        metadata: Dict[str, Any] = {}
        try:
            forecast_values, ci_values, metadata = _run_registered_forecast_method(
                method_l=method_l,
                method=method,
                df=df,
                target_series=target_series,
                horizon=horizon,
                seasonality=seasonality,
                params=p,
                ci_alpha=ci_alpha,
                as_of=as_of,
                training_window_mode=(
                    "as_of"
                    if as_of is not None
                    else "range"
                    if start is not None or end is not None
                    else "latest"
                ),
                quantity_l=quantity_l,
                symbol=symbol,
                timeframe=timeframe,
                base_col=base_col,
                denoise_spec_used=dn_spec_used,
                X=X,
                future_exog=future_exog,
                features=features,
                feature_info=feature_info,
                target_spec=target_spec,
                dimred_method=dimred_method,
                dimred_params=dimred_params,
                training_window={
                    "mode": (
                        "as_of"
                        if as_of is not None
                        else "range"
                        if start is not None or end is not None
                        else "latest"
                    ),
                    **({"lookback": int(lookback)} if lookback is not None else {}),
                    **({"as_of": as_of} if as_of is not None else {}),
                    **({"start": start} if start is not None else {}),
                    **({"end": end} if end is not None else {}),
                },
                async_mode=async_mode,
                model_id=model_id,
                model_cache=model_cache,
            )
        except _AsyncTrainingStarted as at:
            return at.response
        except ModelCompatibilityError:
            raise
        except ValueError as e:
            if method_l == 'ensemble':
                return {"error": str(e)}
            return {"error": f"Forecast method '{method}' failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Forecast method '{method}' failed: {str(e)}"}

        if forecast_values is None:
            return {"error": f"Method '{method}' returned no forecast values"}

        metadata = _merge_engine_diagnostics(metadata, engine_diagnostics)
        if isinstance(features, dict) and features:
            attestation_error = _feature_consumption_attestation_error(
                metadata,
                horizon=horizon,
            )
            if attestation_error is not None:
                return {
                    "error": (
                        "Feature consumption could not be verified: "
                        f"{attestation_error}."
                    ),
                    "error_code": _FEATURE_ATTESTATION_ERROR_CODE,
                    "method": method_l,
                    "remediation": (
                        "Use a feature-capable method whose runtime adapter emits "
                        "complete feature-consumption diagnostics."
                    ),
                }

        # Prepare output arrays
        forecast_return_vals = None
        reconstructed_prices = None
        reconstructed_price_ci = None
        target_transform = str(target_info.get("transform") or "none").strip().lower()
        custom_target = str(target_info.get("mode") or "") == "custom"
        needs_price_reconstruction = (
            not custom_target
            and (quantity_l == "return" or target_transform != "none")
        )
        if quantity_l == 'return':
            forecast_return_vals = np.asarray(forecast_values, dtype=float)
        if needs_price_reconstruction:
            reconstructed_prices = _reconstruct_prices_from_target(
                np.asarray(forecast_values, dtype=float),
                price_anchor_history,
                target_info,
            )
            if reconstructed_prices is None:
                return {
                    "error": (
                        "Unable to reconstruct price-scale forecast from target "
                        f"transform '{target_transform}'."
                    )
                }
            if ci_values is not None and len(ci_values) == 2:
                reconstructed_price_ci = _reconstruct_price_intervals_from_target(
                    np.asarray(forecast_values, dtype=float),
                    (
                        np.asarray(ci_values[0], dtype=float),
                        np.asarray(ci_values[1], dtype=float),
                    ),
                    reconstructed_prices,
                    price_anchor_history,
                    target_info,
                )
                if target_transform.split("(", 1)[0] in {
                    "log_return", "return", "pct_change", "pct", "diff"
                }:
                    metadata["price_interval_reconstruction"] = (
                        "independent_step_variance_from_marginal_target_intervals"
                    )

        # Format and return output
        denoise_used = dn_spec_used is not None
        historical_cutoff_epoch = (
            bar_close_epoch(last_epoch, timeframe)
            if start is not None or end is not None
            else None
        )
        bar_state_reference_epoch = _forecast_bar_state_reference_epoch(
            as_of,
            timeframe=timeframe,
            historical_cutoff_epoch=historical_cutoff_epoch,
        )
        result = _format_forecast_output(
            forecast_values,
            last_epoch,
            tf_secs,
            horizon,
            base_col,
            df,
            ci_alpha,
            ci_values,
            method,
            quantity_l,
            denoise_used,
            metadata,
            digits=digits,
            forecast_return_values=forecast_return_vals,
            reconstructed_prices=reconstructed_prices,
            reconstructed_price_ci=reconstructed_price_ci,
            symbol=symbol,
            timeframe=timeframe,
            target_info=target_info,
            last_target_value=(
                float(target_series.iloc[-1]) if len(target_series) else None
            ),
            now_epoch=bar_state_reference_epoch,
        )
        result["symbol"] = symbol
        if symbol_requested:
            result["symbol_requested"] = symbol_requested
        if as_of is not None:
            result["bar_state_reference"] = "as_of"
            result["bar_state_reference_time"] = str(as_of)
        elif historical_cutoff_epoch is not None:
            result["bar_state_reference"] = "historical_cutoff"
            result["bar_state_reference_time"] = _format_time_minimal(
                bar_state_reference_epoch
            )
        else:
            result["bar_state_reference"] = "retrieval_time"
            result["bar_state_reference_time"] = _format_time_minimal(
                bar_state_reference_epoch
            )
        result.update(
            {
                key: value
                for key, value in sample_quality.items()
                if key != "warning"
            }
        )
        sample_warning = sample_quality.get("warning")
        if sample_warning:
            warnings = result.get("warnings")
            if not isinstance(warnings, list):
                warnings = []
            if sample_warning not in warnings:
                warnings.append(str(sample_warning))
            result["warnings"] = warnings
        if broker_time_check_result and broker_time_check_result.get("status") == "misaligned":
            warning_text = str(broker_time_check_result.get("warning") or "").strip()
            if warning_text:
                warnings = result.get("warnings")
                if not isinstance(warnings, list):
                    warnings = []
                if warning_text not in warnings:
                    warnings.append(warning_text)
                if warnings:
                    result["warnings"] = warnings
        data_warnings = [*history_warnings, *denoise_warnings]
        if data_warnings:
            warnings = result.get("warnings")
            if not isinstance(warnings, list):
                warnings = []
            for warning_text in data_warnings:
                warning_value = str(warning_text)
                if warning_value not in warnings:
                    warnings.append(warning_value)
            if warnings:
                result["warnings"] = warnings
        if as_of is None and start is None and end is None:
            result.update(
                _last_price_freshness_fields(
                    last_epoch=last_epoch,
                    tf_secs=int(tf_secs),
                    symbol=symbol,
                )
            )
        attach_denoise_causality_disclosure(result, dn_spec_used)
        if method_l == 'ensemble' and metadata:
            generic_metadata_keys = {"params_used", "diagnostics", "model_info", "warnings"}
            ensemble_metadata = {
                key: value
                for key, value in metadata.items()
                if key not in generic_metadata_keys
            }
            for key in ensemble_metadata:
                result.pop(key, None)
            if ensemble_metadata:
                result["ensemble"] = ensemble_metadata
        return result

    except ModelCompatibilityError:
        raise
    except Exception as e:
        return {"error": f"Forecast engine failed: {str(e)}"}

