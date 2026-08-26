import difflib
import math
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Literal, Optional

import numpy as np
import pandas as pd

from ..services.data_service.candles import fetch_history_frame
from ..shared.constants import CALENDAR_TIMEFRAMES, TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..shared.schema import DenoiseSpec, DetailLiteral, TimeframeLiteral
from ..shared.validators import (
    invalid_timeframe_error,
    unknown_mapping_keys_error,
    unsupported_timeframe_seconds_error,
)
from ..utils.denoise import apply_denoise, effective_denoise_base_col
from ..utils.denoise import normalize_denoise_spec as _normalize_denoise_spec
from ..utils.freshness import (
    completed_bar_freshness_fields,
)
from ..utils.mt5 import mt5  # noqa: F401 - retained for test patch compatibility
from ..utils.time import _format_time_minimal, bar_close_epoch
from ..utils.utils import _parse_end_datetime, _parse_start_datetime, parse_kv_or_json
from .common import (
    annualization_context as _annualization_context,
)
from .common import (
    default_seasonality as _default_seasonality_period,
)
from .common import (
    describe_forecast_calendar_treatment,
    next_times_from_last,
    uses_exchange_intraday_projection,
    uses_standard_weekend_projection,
)
from .common import (
    log_returns_from_prices as _log_returns_from_prices,
)
from .forecast_registry import (
    ForecastRegistry,
    get_forecast_method_availability_snapshot,
)
from .requests import VOLATILITY_PROXY_VALUES

HAR_RV_MIN_ALIGNED_ROWS = 20
HAR_RV_MIN_DAILY_RV = 30
VOLATILITY_PROXY_METHODS = ("arima", "sarima", "ets", "theta")

_VOLATILITY_METHOD_HINTS = (
    "ewma",
    "garch",
    "har_rv",
    "arima",
    "sarima",
    "ets",
    "theta",
    "ensemble",
)

_VOLATILITY_METHOD_CONCEPT_HINTS = {
    "close_to_close": "rolling_std",
    "historical": "rolling_std",
    "historical_volatility": "rolling_std",
    "realized": "realized_kernel",
    "realized_volatility": "realized_kernel",
    "standard_deviation": "rolling_std",
    "stddev": "rolling_std",
}

_RATES_CACHE: ContextVar[
    Optional[Dict[tuple[Any, ...], tuple[int, Any]]]
] = ContextVar("mtdata_volatility_rates_cache", default=None)


@contextmanager
def volatility_rates_cache() -> Iterator[None]:
    """Reuse candle supersets within one explicit volatility workload."""
    if _RATES_CACHE.get() is not None:
        yield
        return
    token = _RATES_CACHE.set({})
    try:
        yield
    finally:
        _RATES_CACHE.reset(token)


try:
    from arch import arch_model as _arch_model  # type: ignore
    _ARCH_AVAILABLE = True
except Exception:
    _ARCH_AVAILABLE = False


# Use shared helpers


def get_volatility_methods_data() -> Dict[str, Any]:
    """Return metadata about available volatility forecasting methods and their parameters."""
    methods: List[Dict[str, Any]] = []
    forecast_availability = get_forecast_method_availability_snapshot()

    methods.append({
        "method": "ewma",
        "available": True,
        "requires": [],
        "description": "Exponentially weighted moving variance (RiskMetrics-style).",
        "params": [
            {"name": "lookback", "type": "int", "default": 1500, "description": "Number of past returns used in the EWMA."},
            {"name": "lambda_", "type": "float", "default": 0.94, "description": "Decay factor for the EWMA weights."},
            {"name": "halflife", "type": "int", "default": None, "description": "Optional half-life (in bars) used to derive lambda; overrides lambda_ when provided."},
        ],
    })

    for name, desc in (
        ("parkinson", "Parkinson high-low range estimator."),
        ("gk", "Garman-Klass OHLC estimator."),
        ("rs", "Rogers-Satchell OHLC estimator."),
        ("yang_zhang", "Yang-Zhang estimator combining overnight jumps and ranges."),
        ("rolling_std", "Rolling standard deviation of simple returns."),
    ):
        methods.append({
            "method": name,
            "available": True,
            "requires": [],
            "description": desc,
            "params": [
                {"name": "window", "type": "int", "default": 20, "description": "Number of bars in the rolling window."},
            ],
        })

    methods.append({
        "method": "realized_kernel",
        "available": True,
        "requires": [],
        "description": "Realized kernel variance with configurable kernel and bandwidth.",
        "params": [
            {"name": "window", "type": "int", "default": 50, "description": "Number of bars of returns fed to the kernel."},
            {"name": "kernel", "type": "str", "default": "tukey_hanning", "description": "Kernel name (tukey_hanning, bartlett, parzen, triangular)."},
            {"name": "bandwidth", "type": "int", "default": None, "description": "Optional kernel bandwidth; auto-selected when omitted."},
        ],
    })

    methods.append({
        "method": "har_rv",
        "available": True,
        "requires": [],
        "description": "HAR-RV model on realized variance aggregated from intraday bars.",
        "params": [
            {"name": "rv_timeframe", "type": "str", "default": "M5", "description": "Timeframe used to build intraday realized variance."},
            {
                "name": "days",
                "type": "int",
                "default": 120,
                "description": (
                    "Number of calendar days fetched for the HAR fit. Need enough "
                    "history for max(30, window_m+5) daily RV observations and 20 "
                    "aligned regression rows after the monthly lag; default 120."
                ),
            },
            {"name": "window_w", "type": "int", "default": 5, "description": "Weekly window size for HAR lags."},
            {"name": "window_m", "type": "int", "default": 22, "description": "Monthly window size for HAR lags."},
        ],
        "sample_gates": {
            "daily_rv_required": "max(30, window_m + 5)",
            "aligned_rows_required": HAR_RV_MIN_ALIGNED_ROWS,
            "days_default": 120,
        },
    })

    def _garch_entry(name: str, base_desc: str, dist_default: str = "normal") -> Dict[str, Any]:
        params = [
            {"name": "fit_bars", "type": "int", "default": 2000, "description": "Number of recent returns used to fit the ARCH model."},
            {"name": "p", "type": "int", "default": 1, "description": "ARCH order (p)."},
            {"name": "q", "type": "int", "default": 1, "description": "GARCH order (q)."},
            {"name": "mean", "type": "str", "default": "Zero", "description": "Mean model ('Zero' or 'Constant')."},
            {"name": "dist", "type": "str", "default": dist_default, "description": "Innovation distribution (normal, studentst, skewt, etc.)."},
        ]
        if "gjr" in name:
            params.append({"name": "o", "type": "int", "default": 1, "description": "Asymmetry (leverage) order for GJR-GARCH."})
        return {
            "method": name,
            "available": _ARCH_AVAILABLE,
            "requires": [] if _ARCH_AVAILABLE else ["arch"],
            "description": base_desc,
            "params": params,
        }

    methods.extend([
        _garch_entry("garch", "GARCH volatility model (ARCH package)."),
        _garch_entry("egarch", "Exponential GARCH volatility model."),
        _garch_entry("gjr_garch", "GJR-GARCH with leverage effects."),
        _garch_entry("garch_t", "GARCH with Student-t innovations.", dist_default="studentst"),
        _garch_entry("egarch_t", "EGARCH with Student-t innovations.", dist_default="studentst"),
        _garch_entry("gjr_garch_t", "GJR-GARCH with Student-t innovations.", dist_default="studentst"),
        _garch_entry("figarch", "Fractionally integrated FIGARCH volatility model."),
    ])

    _proxy_method_descriptions = {
        "arima": "ARIMA model fitted to the volatility proxy series.",
        "sarima": "Seasonal ARIMA on the volatility proxy with the canonical seasonal period.",
        "ets": "Canonical exponential smoothing on the volatility proxy.",
        "theta": "Canonical Theta method applied to the volatility proxy.",
    }
    for method_name in VOLATILITY_PROXY_METHODS:
        description = _proxy_method_descriptions[method_name]
        method = ForecastRegistry.get(method_name)
        available = bool(forecast_availability.get(method_name, False))
        methods.append(
            {
                "method": method_name,
                "available": available,
                "requires": [] if available else list(method.required_packages),
                "description": description,
                "params": list(getattr(method, "PARAMS", []) or []),
                "requires_proxy": True,
                "valid_proxies": list(VOLATILITY_PROXY_VALUES),
            }
        )

    methods.append({
        "method": "ensemble",
        "available": True,
        "requires": [],
        "description": "Blend of multiple direct/general volatility methods.",
        "params": [
            {"name": "methods", "type": "list[str]", "default": [], "description": "Volatility methods to blend (leave blank for defaults)."},
            {"name": "aggregator", "type": "str", "default": "mean", "description": "Aggregation strategy: mean, median, weighted."},
            {"name": "weights", "type": "list[float]", "default": [], "description": "Optional weights for the weighted aggregator."},
            {"name": "expose_components", "type": "bool", "default": True, "description": "Expose individual component forecasts in the response."},
            {"name": "method_params", "type": "dict", "default": {}, "description": "Optional per-method params merged into the shared params payload."},
        ],
    })

    return {"methods": methods}


def _volatility_allowed_param_keys(method: str) -> set[str]:
    method_l = str(method or "").strip().lower()
    keys = {"lookback"}
    for item in get_volatility_methods_data().get("methods") or []:
        if str(item.get("method") or "").strip().lower() != method_l:
            continue
        keys.update(
            str(param.get("name"))
            for param in (item.get("params") or [])
            if str(param.get("name") or "").strip()
        )
        return keys
    return keys


def _forecast_method_supports(method: str) -> Dict[str, bool]:
    try:
        from .forecast_methods import get_method_supports

        supports = get_method_supports(method)
    except Exception:
        return {}
    if not isinstance(supports, dict) or not any(bool(v) for v in supports.values()):
        return {}
    return {
        str(key): bool(value)
        for key, value in supports.items()
        if key in {"price", "return", "volatility", "ci"}
    }


def _invalid_volatility_method_error(
    method: Any,
    *,
    valid_methods: set[str],
) -> Dict[str, Any]:
    method_text = str(method).strip()
    method_l = method_text.lower()
    valid_method_list = sorted(valid_methods)
    normalized_method = method_l.replace("-", "_").replace(" ", "_")
    suggested_method = _VOLATILITY_METHOD_CONCEPT_HINTS.get(normalized_method)
    if suggested_method not in valid_methods:
        matches = difflib.get_close_matches(normalized_method, valid_method_list, n=1, cutoff=0.55)
        suggested_method = matches[0] if matches else None
    suggestion_text = f" Did you mean: {suggested_method}?" if suggested_method else ""
    supports = _forecast_method_supports(method_l)
    supported_quantities = [
        quantity
        for quantity in ("price", "return", "volatility")
        if supports.get(quantity)
    ]

    if supports and not supports.get("volatility"):
        supported_text = ", ".join(supported_quantities) or "none"
        hints = ", ".join(_VOLATILITY_METHOD_HINTS)
        return {
            "error": (
                f"Method '{method_text}' does not support quantity='volatility'. "
                f"Supported quantities: {supported_text}. Use "
                f"forecast_volatility_estimate with a volatility method such as {hints}."
            ),
            "error_code": "unsupported_quantity_method",
            "method": method_l,
            "quantity": "volatility",
            "supported_quantities": supported_quantities,
            "valid_volatility_methods": valid_method_list,
        }

    if supports:
        return {
            "error": (
                f"Method '{method_text}' is registered for forecast_generate but is "
                "not a forecast_volatility_estimate method. Use one of: "
                f"{', '.join(valid_method_list)}.{suggestion_text}"
            ),
            "error_code": "unsupported_volatility_method",
            "method": method_l,
            "quantity": "volatility",
            "supported_quantities": supported_quantities,
            "valid_volatility_methods": valid_method_list,
            **({"suggested_method": suggested_method} if suggested_method else {}),
        }

    return {
        "error": (
            f"Invalid volatility method: {method_text}. Use one of: "
            f"{', '.join(valid_method_list)}.{suggestion_text}"
        ),
        "error_code": "invalid_volatility_method",
        "valid_volatility_methods": valid_method_list,
        **({"suggested_method": suggested_method} if suggested_method else {}),
    }


def _har_rv_daily_rv_required(window_m: int) -> int:
    return max(HAR_RV_MIN_DAILY_RV, int(window_m) + 5)


def _har_rv_recommended_days(*, window_m: int, days_requested: int) -> int:
    daily_required = _har_rv_daily_rv_required(window_m)
    min_trading_days = max(
        daily_required,
        HAR_RV_MIN_ALIGNED_ROWS + int(window_m) + 1,
    )
    calendar_days = int(math.ceil(min_trading_days * 7 / 5) + 2)
    recommended = max(calendar_days, 120)
    requested = int(days_requested)
    if recommended <= requested:
        recommended = requested + max(int(window_m), 20)
    return recommended


def _har_rv_sample_error(
    *,
    error: str,
    error_code: str,
    daily_rv_observed: int,
    daily_rv_required: int,
    aligned_rows_observed: Optional[int],
    aligned_rows_required: int,
    window_m: int,
    window_w: int,
    days_requested: int,
) -> Dict[str, Any]:
    recommended_days = _har_rv_recommended_days(
        window_m=window_m,
        days_requested=days_requested,
    )
    payload: Dict[str, Any] = {
        "success": False,
        "error": error,
        "error_code": error_code,
        "daily_rv_observed": int(daily_rv_observed),
        "daily_rv_required": int(daily_rv_required),
        "aligned_rows_required": int(aligned_rows_required),
        "window_m": int(window_m),
        "window_w": int(window_w),
        "days_requested": int(days_requested),
        "days_recommended": int(recommended_days),
        "remediation": (
            f"Retry forecast_volatility_estimate with --params days={recommended_days} "
            "(or the default days=120). HAR-RV needs at least "
            f"{daily_rv_required} daily RV observations and "
            f"{aligned_rows_required} aligned regression rows after the "
            f"window_m={window_m} monthly lag."
        ),
    }
    if aligned_rows_observed is not None:
        payload["aligned_rows_observed"] = int(aligned_rows_observed)
    return payload


def _volatility_proxy_required_error(method: str) -> Dict[str, Any]:
    proxies = "|".join(VOLATILITY_PROXY_VALUES)
    return {
        "success": False,
        "error": (
            f"General methods require --proxy ({proxies})."
        ),
        "error_code": "volatility_proxy_required",
        "method": str(method).strip().lower(),
        "valid_proxies": list(VOLATILITY_PROXY_VALUES),
        "remediation": (
            "Retry forecast_volatility_estimate with --proxy squared_return "
            f"(or {', '.join(VOLATILITY_PROXY_VALUES[1:])}). Direct estimators "
            "such as ewma, har_rv, and garch do not need --proxy."
        ),
    }


# --- Range-based variance helpers -------------------------------------------------

def _parkinson_sigma_sq(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    eps = 1e-12
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.log(np.maximum(h, eps)) - np.log(np.maximum(l, eps))
    const = 1.0 / (4.0 * math.log(2.0))
    v = const * (x * x)
    v[~np.isfinite(v)] = np.nan
    return np.maximum(v, 0.0)


def _garman_klass_sigma_sq(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    eps = 1e-12
    o = np.asarray(open_, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        hl = np.log(np.maximum(h, eps)) - np.log(np.maximum(l, eps))
        co = np.log(np.maximum(c, eps)) - np.log(np.maximum(o, eps))
    v = 0.5 * (hl * hl) - (2.0 * math.log(2.0) - 1.0) * (co * co)
    v[~np.isfinite(v)] = np.nan
    return np.maximum(v, 0.0)


def _rogers_satchell_sigma_sq(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    eps = 1e-12
    o = np.asarray(open_, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = (np.log(np.maximum(h, eps)) - np.log(np.maximum(c, eps))) * (np.log(np.maximum(h, eps)) - np.log(np.maximum(o, eps)))
        term2 = (np.log(np.maximum(l, eps)) - np.log(np.maximum(c, eps))) * (np.log(np.maximum(l, eps)) - np.log(np.maximum(o, eps)))
    rs = term1 + term2
    rs[~np.isfinite(rs)] = np.nan
    return np.maximum(rs, 0.0)

def _kernel_weight(kind: str, h: int, bandwidth: int) -> float:
    if bandwidth <= 0:
        return 0.0
    x = float(h) / float(bandwidth + 1)
    x = max(0.0, min(1.0, x))
    k = kind.lower()
    if k in {"bartlett", "triangular"}:
        return float(1.0 - x)
    if k in {"parzen", "parzen_bartlett"}:
        if x <= 0.5:
            return float(1.0 - 6.0 * x * x + 6.0 * x * x * x)
        if x <= 1.0:
            return float(2.0 * (1.0 - x) ** 3)
        return 0.0
    # Tukey-Hanning (default)
    return float(0.5 * (1.0 + math.cos(math.pi * x))) if x <= 1.0 else 0.0


def _realized_kernel_variance(
    returns: np.ndarray,
    bandwidth: Optional[int] = None,
    kernel: str = "tukey_hanning",
) -> float:
    """Compute realized kernel variance estimate for a return series."""

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 3:
        return float('nan')
    if bandwidth is None:
        bandwidth = max(1, int(np.floor(np.sqrt(n))))
    bandwidth = int(max(1, min(bandwidth, n - 1)))
    r_centered = r - float(np.mean(r))
    gamma0 = float(np.dot(r_centered, r_centered))
    rk = gamma0
    for h in range(1, bandwidth + 1):
        cov = float(np.dot(r_centered[h:], r_centered[:-h]))
        weight = _kernel_weight(kernel, h, bandwidth)
        rk += 2.0 * weight * cov
    rk = max(rk, 0.0)
    return float(rk / max(1, n))


def _ewma_param_explanations(lambda_source: str) -> Dict[str, str]:
    """Human-readable explanations for EWMA parameters in API output."""
    out = {
        "lambda_": (
            "EWMA decay factor for volatility weights (0-1). "
            "Higher values retain older bars longer; lower values react faster to recent moves."
        ),
    }
    if lambda_source == "halflife":
        out["halflife"] = (
            "Half-life in bars used to derive lambda_ "
            "(lambda_ = exp(-ln(2) / halflife); for halflife=22, "
            "lambda_ is approximately 0.969)."
        )
    return out


def _annualize_horizon_sigma(
    horizon_volatility: float,
    bars_per_year: float,
    horizon: int,
) -> float:
    """Express the horizon-scaled sigma on the annualized return scale."""
    horizon_bars = max(1, int(horizon))
    return float(horizon_volatility * math.sqrt(bars_per_year / horizon_bars))


def _bars_per_session_from_annualization(
    bars_per_year_value: float,
    annualization_basis: str,
) -> float:
    """Recover the per-session bar count from the reported year convention."""
    basis = str(annualization_basis or "")
    sessions_per_year = 365.0 if basis.startswith("365_") else 260.0 if basis.startswith("260_") else 252.0
    return float(bars_per_year_value) / sessions_per_year


_OHLC_VOLATILITY_METHODS = frozenset({"parkinson", "gk", "rs", "yang_zhang", "har_rv"})


def _volatility_denoise_spec(
    spec: Optional[Dict[str, Any]],
    *,
    method: str,
    user_columns: Any,
) -> Optional[Dict[str, Any]]:
    if not spec:
        return spec
    if method in _OHLC_VOLATILITY_METHODS and user_columns is None:
        widened = dict(spec)
        widened["columns"] = "ohlc"
        return widened
    return spec


def _volatility_price_column(
    frame: pd.DataFrame,
    spec: Optional[Dict[str, Any]],
    name: str,
) -> str:
    return effective_denoise_base_col(frame, spec, base_col=name)


def _realized_variance_rows(
    frame: pd.DataFrame,
    *,
    close_col: str = "close",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = pd.DataFrame(
        {
            "time": pd.to_numeric(frame.get("time"), errors="coerce"),
            "close": pd.to_numeric(frame.get(close_col), errors="coerce"),
        }
    ).dropna()
    values = values.sort_values("time", kind="stable")
    values["day"] = pd.to_datetime(values["time"], unit="s", utc=True).dt.floor("D")
    values["return"] = values.groupby("day", sort=True)["close"].transform(
        lambda series: np.log(series.where(series > 0)).diff()
    )
    finite = values[np.isfinite(values["return"])].copy()
    finite["r2"] = np.square(finite["return"].astype(float))
    return values, finite


def _har_daily_realized_variance(
    frame: pd.DataFrame,
    *,
    close_col: str = "close",
    minimum_coverage_fraction: float = 0.9,
) -> tuple[pd.Series, int, Dict[str, Any]]:
    """Exclude an incomplete trailing UTC day from comparable HAR-RV lags."""
    values, finite = _realized_variance_rows(frame, close_col=close_col)
    daily_rv = finite.groupby("day", sort=True)["r2"].sum().astype(float)
    bar_counts = values.groupby("day", sort=True)["time"].count().astype(int)
    if bar_counts.empty:
        return daily_rv, int(len(finite)), {}

    final_day = bar_counts.index[-1]
    prior_counts = bar_counts.iloc[1:-1].tail(20)
    if prior_counts.empty:
        prior_counts = bar_counts.iloc[:-1].tail(20)
    expected_bars = (
        max(1, int(round(float(prior_counts.median()))))
        if not prior_counts.empty
        else int(bar_counts.iloc[-1])
    )
    observed_bars = int(bar_counts.iloc[-1])
    coverage = float(observed_bars) / float(max(1, expected_bars))
    complete = coverage >= float(minimum_coverage_fraction)
    final_values = values[values["day"] == final_day]
    final_finite = finite[finite["day"] == final_day]
    if not complete:
        daily_rv = daily_rv.drop(final_day, errors="ignore")

    first_epoch = float(final_values["time"].iloc[0])
    last_epoch = float(final_values["time"].iloc[-1])
    metadata: Dict[str, Any] = {
        "utc_day": final_day.strftime("%Y-%m-%d"),
        "start": _format_time_minimal(first_epoch),
        "end": _format_time_minimal(last_epoch),
        "observed_bars": observed_bars,
        "expected_bars": expected_bars,
        "coverage_fraction": round(coverage, 4),
        "minimum_coverage_fraction": float(minimum_coverage_fraction),
        "complete": bool(complete),
        "included_in_har": bool(complete),
        "policy": "exclude_final_utc_day_below_recent_median_coverage",
        "expected_bars_basis": "median_of_up_to_20_prior_utc_days",
    }
    returns_used = int(len(finite) - (0 if complete else len(final_finite)))
    return daily_rv, returns_used, metadata


def _volatility_input_context(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    observed_timeframe: Optional[str] = None,
    returns_used: int,
    live_window: bool,
    horizon: int = 1,
    now_epoch: Optional[float] = None,
    forecast_grid_anchor_epoch: Optional[float] = None,
) -> Dict[str, Any]:
    if "time" not in df.columns or len(df) == 0:
        return {}
    try:
        first_epoch = float(df["time"].iloc[0])
        last_epoch = float(df["time"].iloc[-1])
    except (TypeError, ValueError):
        return {}

    observation_timeframe = str(observed_timeframe or timeframe)
    last_bar_open = _format_time_minimal(last_epoch)
    try:
        last_close_epoch = bar_close_epoch(last_epoch, observation_timeframe)
        last_bar_close = _format_time_minimal(last_close_epoch)
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        last_close_epoch = None
        last_bar_close = last_bar_open
    out: Dict[str, Any] = {
        "data_as_of": last_bar_close,
        "last_bar_open": last_bar_open,
        "last_observation_close_time": last_bar_close,
    }
    if last_close_epoch is not None:
        out["data_as_of_epoch"] = last_close_epoch
    out.update(
        {
            "data_window": {
                "start": _format_time_minimal(first_epoch),
                "end": last_bar_open,
                "bars_used": int(len(df)),
                "returns_used": int(returns_used),
                "input_bar_policy": "closed_bars_only",
            },
        }
    )
    if observation_timeframe != str(timeframe):
        out["data_window"]["observed_timeframe"] = observation_timeframe
    tf_secs = int(TIMEFRAME_SECONDS.get(timeframe, 0) or 0)
    grid_anchor_epoch = (
        float(forecast_grid_anchor_epoch)
        if forecast_grid_anchor_epoch is not None
        else last_epoch
    )
    grid_last_epoch = last_epoch
    if (
        observation_timeframe != str(timeframe)
        and forecast_grid_anchor_epoch is not None
    ):
        grid_last_epoch = float(forecast_grid_anchor_epoch)
    observed_times = (
        df.get("time") if observation_timeframe == str(timeframe) else None
    )
    session_projection = uses_exchange_intraday_projection(
        symbol,
        tf_secs,
        observed_times=observed_times,
    )
    forecast_epochs = (
        _next_volatility_times_on_grid(
            grid_last_epoch,
            grid_anchor_epoch,
            tf_secs,
            max(1, int(horizon)),
            symbol=symbol,
            timeframe=timeframe,
            observed_times=observed_times,
        )
        if tf_secs > 0
        else []
    )
    if forecast_epochs:
        start_epoch = float(forecast_epochs[0])
        end_epoch = float(forecast_epochs[-1])
        calendar_timeframe = str(timeframe).upper() in CALENDAR_TIMEFRAMES
        out["forecast_window"] = {
            "anchor": _format_time_minimal(grid_anchor_epoch),
            "start": _format_time_minimal(start_epoch),
            "end": _format_time_minimal(end_epoch),
            "bars": int(len(forecast_epochs)),
            "step_seconds": (
                None if calendar_timeframe or session_projection else tf_secs
            ),
            "nominal_step_seconds": tf_secs if session_projection else None,
            "forecast_start_gap_bars": round(
                1.0
                if (
                    calendar_timeframe
                    or session_projection
                    or uses_standard_weekend_projection(symbol, tf_secs)
                )
                else (start_epoch - last_epoch) / float(tf_secs),
                4,
            ),
            "calendar_treatment": describe_forecast_calendar_treatment(
                symbol,
                tf_secs,
                calendar_timeframe=calendar_timeframe,
                observed_times=observed_times,
            ),
        }
        if not session_projection:
            out["forecast_window"].pop("nominal_step_seconds", None)
        if observation_timeframe != str(timeframe):
            out["forecast_window"]["timeframe"] = str(timeframe)
            out["forecast_window"]["input_data_as_of"] = _format_time_minimal(
                last_epoch
            )
            out["forecast_window"]["alignment_basis"] = (
                "mt5_requested_timeframe_candle_grid"
            )
    if not live_window:
        return out

    if now_epoch is None:
        now_epoch = datetime.now(timezone.utc).timestamp()
    freshness = completed_bar_freshness_fields(
        symbol,
        observation_timeframe,
        last_epoch,
        now_epoch=now_epoch,
        item="data",
    )
    out.update(freshness)
    out["last_observation_close_time"] = freshness.get("data_as_of")
    if observation_timeframe != str(timeframe):
        out["freshness_timeframe"] = observation_timeframe
    return out


def _next_volatility_times_on_grid(
    last_observation_epoch: float,
    grid_anchor_epoch: float,
    tf_secs: int,
    horizon: int,
    *,
    symbol: str,
    timeframe: str,
    observed_times: Any = None,
) -> List[float]:
    """Project targets from a requested-timeframe grid, not the input cadence."""
    skip_weekends = uses_standard_weekend_projection(symbol, tf_secs)
    normalized_timeframe = str(timeframe).upper()
    if uses_exchange_intraday_projection(
        symbol,
        tf_secs,
        observed_times=observed_times,
    ):
        return next_times_from_last(
            last_observation_epoch,
            tf_secs,
            horizon,
            skip_weekends=skip_weekends,
            timeframe=timeframe,
            symbol=symbol,
            observed_times=observed_times,
        )
    if normalized_timeframe not in CALENDAR_TIMEFRAMES:
        steps_after_anchor = math.floor(
            (float(last_observation_epoch) - float(grid_anchor_epoch))
            / float(tf_secs)
        ) + 1
        aligned_predecessor = (
            float(grid_anchor_epoch)
            + float(steps_after_anchor - 1) * float(tf_secs)
        )
        return next_times_from_last(
            aligned_predecessor,
            tf_secs,
            horizon,
            skip_weekends=skip_weekends,
            timeframe=timeframe,
            symbol=symbol,
            observed_times=observed_times,
        )

    candidate_count = max(int(horizon) + 8, 16)
    future: List[float] = []
    for _ in range(4):
        candidates = next_times_from_last(
            grid_anchor_epoch,
            tf_secs,
            candidate_count,
            skip_weekends=skip_weekends,
            timeframe=timeframe,
            symbol=symbol,
            observed_times=observed_times,
        )
        future = [
            value for value in candidates
            if float(value) > float(last_observation_epoch)
        ]
        if len(future) >= int(horizon):
            return future[: int(horizon)]
        candidate_count *= 2
    return future[: int(horizon)]


def _finalize_volatility_with_context(
    payload: Dict[str, Any],
    *,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    returns_used: int,
    live_window: bool,
    detail: str,
    data_timeframe: Optional[str] = None,
    forecast_grid_anchor_epoch: Optional[float] = None,
) -> Dict[str, Any]:
    annualization_bars, annualization_basis = _annualization_context(
        timeframe,
        symbol,
        observed_times=df.get("time"),
        observed_timeframe=data_timeframe or timeframe,
    )
    if math.isfinite(annualization_bars) and annualization_bars > 0:
        payload.setdefault("bars_per_year", round(annualization_bars, 4))
        payload.setdefault("annualization_basis", annualization_basis)
    payload.update(
        _volatility_input_context(
            df,
            symbol=symbol,
            timeframe=timeframe,
            observed_timeframe=data_timeframe,
            returns_used=returns_used,
            live_window=live_window,
            horizon=int(payload.get("horizon", 1) or 1),
            forecast_grid_anchor_epoch=forecast_grid_anchor_epoch,
        )
    )
    return _finalize_volatility_output(payload, detail=detail)


def _finalize_volatility_output(
    payload: Dict[str, Any],
    *,
    detail: str = "full",
) -> Dict[str, Any]:
    """Add explanatory metadata to canonical volatility output."""
    if not isinstance(payload, dict) or not payload.get("success"):
        return payload

    out = dict(payload)
    detail_mode = str(detail or "compact").strip().lower()
    out.setdefault("volatility_unit", "return_fraction")
    out.setdefault("volatility_measure", "standard_deviation_of_returns")
    out.setdefault(
        "volatility_unit_note",
        "Volatility values are decimal return fractions; *_pct aliases are percentages.",
    )
    for source_key, pct_key in (
        ("volatility_per_bar", "volatility_per_bar_pct"),
        ("volatility_annualized", "volatility_annualized_pct"),
        ("volatility_horizon", "volatility_horizon_pct"),
        ("volatility_horizon_annualized", "volatility_horizon_annualized_pct"),
    ):
        value = out.get(source_key)
        if value is None:
            continue
        try:
            out.setdefault(pct_key, round(float(value) * 100.0, 6))
        except Exception:
            pass

    if detail_mode != "full":
        for key in (
            "params_explained",
            "params_used",
            "volatility_interpretation",
        ):
            out.pop(key, None)
        horizon = out.get("horizon")
        if isinstance(horizon, (int, float)) and int(horizon) == 1:
            out.setdefault(
                "horizon_note",
                "horizon=1, so volatility_horizon equals volatility_per_bar.",
            )
        for key in (
            "volatility_per_bar",
            "volatility_annualized",
            "volatility_horizon",
            "volatility_horizon_annualized",
        ):
            try:
                out[key] = round(float(out[key]), 6)
            except Exception:
                pass
        for key in (
            "volatility_per_bar_pct",
            "volatility_annualized_pct",
            "volatility_horizon_pct",
            "volatility_horizon_annualized_pct",
        ):
            try:
                out[key] = round(float(out[key]), 4)
            except Exception:
                pass
        try:
            if math.isclose(
                float(out.get("volatility_horizon_annualized")),
                float(out.get("volatility_annualized")),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                out.pop("volatility_horizon_annualized", None)
                out.pop("volatility_horizon_annualized_pct", None)
                out.setdefault(
                    "volatility_annualized_note",
                    "volatility_horizon_annualized equals volatility_annualized under sqrt-time scaling; "
                    "volatility_horizon remains scaled to the requested horizon.",
                )
        except Exception:
            pass
        if detail_mode == "compact":
            for key in (
                "volatility_per_bar_pct",
                "volatility_annualized_pct",
                "volatility_horizon_pct",
                "volatility_horizon_annualized_pct",
                "volatility_unit_note",
                "volatility_annualized_note",
                "horizon_note",
            ):
                out.pop(key, None)
        return out

    horizon = out.get("horizon")
    interpretation = {
        "volatility_per_bar": "Estimated one-bar return volatility for the selected timeframe.",
        "volatility_annualized": "volatility_per_bar annualized using the timeframe's bars-per-year convention.",
        "volatility_horizon": "Return volatility scaled to the requested horizon in bars.",
        "volatility_horizon_annualized": (
            "volatility_horizon expressed on the same annualized return scale. "
            "With sqrt-time scaling this can equal volatility_annualized for horizon > 1."
        ),
        "volatility_unit": "All volatility values are decimal return fractions; 0.0525 means 5.25%.",
    }
    if isinstance(horizon, (int, float)) and int(horizon) == 1:
        interpretation["horizon_note"] = (
            "horizon=1, so volatility_horizon equals volatility_per_bar."
        )
    out.setdefault("volatility_interpretation", interpretation)

    return out


def _fetch_mt5_rates_guarded(
    symbol: str,
    mt5_timeframe: Any,
    count: int,
    *,
    as_of: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> tuple[Optional[Any], Optional[str]]:
    if as_of and (start or end):
        return None, "as_of cannot be combined with start/end."
    _start_dt, _end_dt, range_error = _volatility_range_bounds(start, end)
    if range_error:
        return None, range_error
    requested_count = int(count)
    cache = _RATES_CACHE.get()
    cache_key = (
        str(symbol),
        str(mt5_timeframe),
        str(as_of or ""),
        str(start or ""),
        str(end or ""),
        str(timeframe or ""),
    )
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None and cached[0] >= requested_count:
            cached_rates = cached[1]
            if len(cached_rates) > requested_count:
                cached_rates = cached_rates[-requested_count:]
            return cached_rates, None

    def _remember(rates: Any) -> Any:
        if cache is not None and rates is not None:
            cache[cache_key] = (requested_count, rates)
        return rates
    try:
        rates = fetch_history_frame(
            symbol,
            str(timeframe),
            requested_count,
            as_of,
            start=start,
            end=end,
            include_incomplete=False,
        )
        return _remember(rates), None
    except (RuntimeError, ValueError) as exc:
        return None, str(exc)


def _volatility_range_bounds(
    start: Optional[str],
    end: Optional[str],
) -> tuple[Optional[datetime], Optional[datetime], Optional[str]]:
    """Resolve a historical range without touching the MT5 provider."""
    start_dt = _parse_start_datetime(start) if start else None
    if start and start_dt is None:
        return None, None, "Invalid start time."
    end_dt = _parse_end_datetime(end) if end else None
    if end and end_dt is None:
        return None, None, "Invalid end time."
    now_utc = datetime.now(timezone.utc)
    now_naive = now_utc.replace(tzinfo=None)
    if start_dt is not None and start_dt > now_naive:
        return None, None, "start must not be in the future."
    if end:
        end_lower_bound = _parse_start_datetime(end)
        if end_lower_bound is not None and end_lower_bound > now_naive:
            return None, None, "end must not be in the future."
    if start_dt is not None and end_dt is None:
        end_dt = now_naive
    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        return None, None, "start must be before or equal to end."
    return start_dt, end_dt, None


def _requested_timeframe_grid_anchor(
    symbol: str,
    mt5_timeframe: Any,
    *,
    timeframe: str,
    observed_last_epoch: float,
    as_of: Optional[str] = None,
    end: Optional[str] = None,
) -> tuple[Optional[float], Optional[str]]:
    """Read an actual requested-timeframe candle open for grid alignment."""
    rates, fetch_error = _fetch_mt5_rates_guarded(
        symbol,
        mt5_timeframe,
        3,
        as_of=as_of,
        end=end,
        timeframe=timeframe,
    )
    if fetch_error:
        return None, fetch_error
    if rates is None or len(rates) == 0:
        return None, f"No {timeframe} candles were available to resolve the forecast grid."
    try:
        frame = pd.DataFrame(rates)
        epochs = pd.to_numeric(frame["time"], errors="coerce")
        epochs = epochs[np.isfinite(epochs)]
    except (KeyError, TypeError, ValueError):
        return None, f"Returned {timeframe} candles did not contain usable timestamps."
    if epochs.empty:
        return None, f"Returned {timeframe} candles did not contain usable timestamps."
    cutoff_epoch = float(observed_last_epoch)
    close_epochs = epochs.map(
        lambda epoch: bar_close_epoch(float(epoch), timeframe)
    )
    completed = epochs[close_epochs <= cutoff_epoch]
    if completed.empty:
        return None, (
            f"No completed {timeframe} candles were available at the "
            "observation cutoff."
        )
    return float(completed.iloc[-1]), None


def _volatility_fetch_error_payload(
    message: str,
    *,
    start: Optional[str],
    end: Optional[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"error": str(message)}
    if "must not be in the future" in str(message).lower():
        out.update(
            {
                "success": False,
                "error_code": "forecast_range_in_future",
                "requested_range": {"start": start, "end": end},
                "remediation": (
                    "Correct the requested time range/timezone so "
                    "both bounds are at or before the current UTC time."
                ),
            }
        )
    return out


def _volatility_no_rates_payload(
    symbol: str,
    *,
    start: Optional[str],
    end: Optional[str],
    observed_bars: int,
    minimum_bars: int,
    data_timeframe: str,
) -> Dict[str, Any]:
    ranged = bool(start or end)
    return {
        "success": False,
        "error": (
            f"No sufficient closed MT5 rates were available for {symbol} in the "
            "requested range."
            if ranged
            else f"Not enough closed MT5 rates were available for {symbol}."
        ),
        "error_code": "no_data_for_range" if ranged else "insufficient_history",
        "requested_range": {"start": start, "end": end},
        "data_timeframe": data_timeframe,
        "observed_bars": int(observed_bars),
        "minimum_bars": int(minimum_bars),
        "remediation": (
            "Correct the requested time range/timezone or choose a range with "
            "available closed market data."
            if ranged
            else "Increase available history or choose a shorter-lookback method."
        ),
    }


def forecast_volatility(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 1,
    method: Literal['ewma','parkinson','gk','rs','yang_zhang','rolling_std','realized_kernel','har_rv','garch','egarch','gjr_garch','garch_t','egarch_t','gjr_garch_t','figarch','arima','sarima','ets','theta','ensemble'] = 'ewma',  # type: ignore
    proxy: Optional[Literal['squared_return','abs_return','log_r2']] = None,  # type: ignore
    params: Optional[Dict[str, Any]] = None,
    lookback: Optional[int] = None,
    as_of: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    denoise: Optional[DenoiseSpec] = None,
    detail: DetailLiteral = "full",
) -> Dict[str, Any]:
    """Forecast volatility over `horizon` bars with direct estimators/GARCH or general forecasters on a proxy.

    Direct: ewma, parkinson, gk, rs, yang_zhang, rolling_std, realized_kernel, har_rv, garch(+variants).
    General: arima, sarima, ets, theta (require `proxy`: squared_return|abs_return|log_r2).
    Meta: ensemble aggregates multiple successful component volatility forecasts.
    """
    try:
        user_denoise_columns = (
            denoise.get("columns") if isinstance(denoise, dict) and "columns" in denoise else None
        )
        try:
            denoise = _normalize_denoise_spec(
                denoise,
                default_when="pre_ti",
            )
        except Exception as ex:
            return {"error": f"Invalid denoise specification: {ex}"}
        if timeframe not in TIMEFRAME_MAP:
            return {"error": invalid_timeframe_error(timeframe, TIMEFRAME_MAP)}
        mt5_tf = TIMEFRAME_MAP[timeframe]
        tf_secs = TIMEFRAME_SECONDS.get(timeframe)
        if not tf_secs:
            return {"error": unsupported_timeframe_seconds_error(timeframe)}
        annualization_bars_per_year, annualization_basis = (
            _annualization_context(timeframe, symbol)
        )
        method_l = str(method).lower().strip()
        garch_family = {'garch','egarch','gjr_garch','garch_t','egarch_t','gjr_garch_t','figarch'}
        valid_direct = {'ewma','parkinson','gk','rs','yang_zhang','rolling_std','realized_kernel','har_rv'} | garch_family
        valid_general = set(VOLATILITY_PROXY_METHODS)
        valid_meta = {'ensemble'}
        valid_methods = valid_direct.union(valid_general).union(valid_meta)
        if method_l not in valid_methods:
            return _invalid_volatility_method_error(method, valid_methods=valid_methods)
        if method_l in valid_direct and proxy is not None:
            return {
                "error": (
                    f"Direct volatility method '{method_l}' does not accept proxy. "
                    "Omit proxy or use arima, sarima, ets, or theta."
                )
            }
        if method_l in garch_family and not _ARCH_AVAILABLE:
            return {"error": f"{method_l} requires 'arch' package."}

        # Parse method params: accept dict, JSON string, or k=v pairs
        __stage = 'parse_params'
        p = parse_kv_or_json(params)
        if lookback is not None:
            nested_lookback = p.get("lookback")
            if nested_lookback is not None:
                try:
                    lookbacks_match = int(nested_lookback) == int(lookback)
                except (TypeError, ValueError):
                    lookbacks_match = False
                if not lookbacks_match:
                    return {
                        "error": (
                            "Conflicting volatility lookbacks: top-level lookback="
                            f"{lookback} and params.lookback={nested_lookback}. Use one "
                            "value or make them equal."
                        )
                    }
            p["lookback"] = int(lookback)
        param_error = unknown_mapping_keys_error(
            p,
            _volatility_allowed_param_keys(method_l),
            subject=f"{method_l} params",
        )
        if param_error is not None:
            return param_error

        if method_l == "ewma":
            try:
                lookback_value = int(p.get("lookback", 1500))
            except (TypeError, ValueError):
                return {"error": "EWMA lookback must be a positive integer."}
            if lookback_value < 2:
                return {"error": "EWMA lookback must be at least 2."}
            if p.get("halflife") is not None:
                try:
                    halflife_value = float(p["halflife"])
                except (TypeError, ValueError):
                    return {"error": "EWMA halflife must be finite and greater than 0."}
                if not math.isfinite(halflife_value) or halflife_value <= 0.0:
                    return {"error": "EWMA halflife must be finite and greater than 0."}
            if p.get("lambda_") is not None:
                try:
                    lambda_value = float(p["lambda_"])
                except (TypeError, ValueError):
                    return {"error": "EWMA lambda_ must be finite and between 0 and 1."}
                if not math.isfinite(lambda_value) or not 0.0 < lambda_value < 1.0:
                    return {"error": "EWMA lambda_ must be finite and strictly between 0 and 1."}

        if method_l == 'ensemble':
            default_methods = ['ewma', 'parkinson', 'rolling_std']
            base_methods_in = p.get('methods')
            if isinstance(base_methods_in, str):
                base_methods = [tok.strip().lower() for tok in base_methods_in.split(',') if tok.strip()]
            elif isinstance(base_methods_in, (list, tuple)):
                base_methods = [str(item).strip().lower() for item in base_methods_in if str(item).strip()]
            else:
                base_methods = list(default_methods)
            invalid_components = sorted(
                set(base_methods) - valid_direct.union(valid_general)
            )
            if invalid_components or "ensemble" in base_methods:
                return {
                    "error": (
                        "Unknown ensemble component method(s): "
                        + ", ".join([*invalid_components, *(["ensemble"] if "ensemble" in base_methods else [])])
                    )
                }
            seen_methods: set[str] = set()
            base_methods = [m for m in base_methods if not (m in seen_methods or seen_methods.add(m))]
            if not base_methods:
                return {"error": "Ensemble requires at least one valid component method."}

            aggregator = str(p.get('aggregator', 'mean')).lower().strip()
            if aggregator not in {'mean', 'median', 'weighted'}:
                return {
                    "error": (
                        f"Unknown ensemble aggregator '{aggregator}'. "
                        "Use mean, median, or weighted."
                    )
                }

            expose_components = bool(p.get('expose_components', True))
            method_params = p.get('method_params') if isinstance(p.get('method_params'), dict) else {}
            shared_params = dict(p)
            for key in ('methods', 'aggregator', 'weights', 'expose_components', 'method_params'):
                shared_params.pop(key, None)

            raw_weights = p.get('weights')
            weight_map: dict[str, float] = {}
            if isinstance(raw_weights, (list, tuple)) and len(raw_weights) == len(base_methods):
                parsed_weights: list[float] = []
                for item in raw_weights:
                    try:
                        weight = float(item)
                    except Exception:
                        parsed_weights = []
                        break
                    if not np.isfinite(weight) or weight <= 0.0:
                        parsed_weights = []
                        break
                    parsed_weights.append(weight)
                if parsed_weights:
                    total_weight = float(sum(parsed_weights))
                    if total_weight > 0.0:
                        weight_map = {
                            method_name: float(weight / total_weight)
                            for method_name, weight in zip(base_methods, parsed_weights)
                        }
            if aggregator == "weighted" and not weight_map:
                return {
                    "error": (
                        "Weighted ensemble requires one finite positive weight "
                        "for each component method."
                    )
                }

            component_results: list[dict[str, Any]] = []
            component_errors: list[dict[str, Any]] = []
            first_component_context: Optional[Dict[str, Any]] = None
            for base_method in base_methods:
                call_params = dict(shared_params)
                per_method_params = method_params.get(base_method)
                if isinstance(per_method_params, dict):
                    call_params.update(per_method_params)
                result = forecast_volatility(
                    symbol=symbol,
                    timeframe=timeframe,
                    horizon=horizon,
                    method=base_method,  # type: ignore[arg-type]
                    proxy=proxy if base_method in valid_general else None,
                    params=call_params or None,
                    as_of=as_of,
                    start=start,
                    end=end,
                    denoise=denoise,
                    detail="full",
                )
                if not isinstance(result, dict) or not result.get('success'):
                    err = result.get('error') if isinstance(result, dict) else None
                    component_errors.append({"method": base_method, "error": str(err or "Component forecast failed")})
                    continue
                try:
                    sigma_bar = float(result['volatility_per_bar'])
                    horizon_sigma = float(result['volatility_horizon'])
                except Exception:
                    component_errors.append({"method": base_method, "error": "Component output missing volatility metrics"})
                    continue
                if not (np.isfinite(sigma_bar) and np.isfinite(horizon_sigma)):
                    component_errors.append({"method": base_method, "error": "Component output contains non-finite volatility metrics"})
                    continue
                component_row: dict[str, Any] = {
                    "method": base_method,
                    "volatility_per_bar": sigma_bar,
                    "volatility_horizon": horizon_sigma,
                    "volatility_annualized": float(
                        result.get('volatility_annualized', float('nan'))
                    ),
                    "volatility_horizon_annualized": float(
                        result.get(
                            'volatility_horizon_annualized',
                            result.get('volatility_annualized', float('nan')),
                        )
                    ),
                    "params_used": result.get('params_used'),
                }
                if result.get('proxy') is not None:
                    component_row['proxy'] = result.get('proxy')
                component_results.append(component_row)
                if first_component_context is None:
                    first_component_context = {
                        key: result[key]
                        for key in (
                            "data_as_of",
                            "data_window",
                            "data_age_seconds",
                            "data_stale",
                            "stale_after_seconds",
                            "freshness_basis",
                            "freshness_age_metric",
                            "last_observation_close_time",
                            "freshness",
                            "market_status",
                            "market_status_reason",
                            "market_status_source",
                            "note",
                            "bars_per_year",
                            "annualization_basis",
                        )
                        if result.get(key) is not None
                    }

            if not component_results:
                return {"error": "Ensemble failed: no successful component methods", "component_errors": component_errors}

            def _aggregate_metric(metric_name: str) -> float:
                values = np.asarray([float(row[metric_name]) for row in component_results], dtype=float)
                if aggregator == 'median':
                    return float(np.median(values))
                if aggregator == 'weighted' and weight_map:
                    weights = np.asarray([float(weight_map.get(str(row['method']), 0.0)) for row in component_results], dtype=float)
                    total = float(np.sum(weights))
                    if total > 0.0:
                        return float(np.sum(values * weights) / total)
                return float(np.mean(values))

            try:
                bpy = float((first_component_context or {}).get("bars_per_year"))
            except (TypeError, ValueError):
                bpy = annualization_bars_per_year
            component_annualization_basis = str(
                (first_component_context or {}).get("annualization_basis")
                or annualization_basis
            )
            volatility_per_bar = _aggregate_metric('volatility_per_bar')
            volatility_horizon = _aggregate_metric('volatility_horizon')
            out: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": "ensemble",
                "horizon": int(horizon),
                "volatility_per_bar": volatility_per_bar,
                "volatility_annualized": float(volatility_per_bar * math.sqrt(bpy)),
                "volatility_horizon": volatility_horizon,
                "volatility_horizon_annualized": _annualize_horizon_sigma(
                    volatility_horizon,
                    bpy,
                    int(horizon),
                ),
                "bars_per_year": round(bpy, 4),
                "annualization_basis": component_annualization_basis,
                "params_used": {
                    "methods": base_methods,
                    "aggregator": aggregator,
                    "weights": [weight_map.get(method_name) for method_name in base_methods] if weight_map else None,
                },
            }
            if proxy is not None:
                out["proxy"] = str(proxy).lower().strip()
            if expose_components:
                out["components"] = component_results
            if component_errors:
                out["component_errors"] = component_errors
                out["warning"] = f"{len(component_errors)} ensemble component(s) failed."
            if first_component_context:
                out.update(first_component_context)
            return _finalize_volatility_output(out, detail=detail)

        # If using general forecasters on proxy, compute proxy series and return using internal logic
        if method_l in valid_general:
            # Fetch recent closes and build returns
            # Reuse unified forecast branch for fetching by delegating to data_fetch_candles/forecast_generate where possible is heavy; implement lightweight here
            # Determine lookback bars
            requested_lookback = None
            if lookback is not None:
                try:
                    requested_lookback = int(lookback)
                except (TypeError, ValueError):
                    requested_lookback = None
            if requested_lookback is None and isinstance(p, dict) and p.get("lookback") is not None:
                try:
                    requested_lookback = int(p.get("lookback"))
                except (TypeError, ValueError):
                    requested_lookback = None
            need = max(300, int(horizon) + 50)
            if requested_lookback is not None:
                need = max(int(requested_lookback) + 1, int(horizon) + 2, 5)
            rates, fetch_error = _fetch_mt5_rates_guarded(
                symbol,
                mt5_tf,
                need,
                as_of=as_of,
                start=start,
                end=end,
                timeframe=timeframe,
            )
            if fetch_error:
                return _volatility_fetch_error_payload(
                    fetch_error,
                    start=start,
                    end=end,
                )
            if rates is None or len(rates) < 5:
                return _volatility_no_rates_payload(
                    symbol,
                    start=start,
                    end=end,
                    observed_bars=0 if rates is None else len(rates),
                    minimum_bars=5,
                    data_timeframe=timeframe,
                )
            df = pd.DataFrame(rates)
            if requested_lookback is not None and len(df) > requested_lookback:
                df = df.iloc[-int(requested_lookback):].copy()
            if len(df) < 5:
                return {"error": "Not enough closed bars"}
            bpy, _ = _annualization_context(
                timeframe,
                symbol,
                observed_times=df.get("time"),
            )
            if denoise:
                apply_denoise(df, denoise)
            close_col = _volatility_price_column(df, denoise, "close")
            r = _log_returns_from_prices(df[close_col].astype(float).to_numpy())
            r = r[np.isfinite(r)]
            if r.size < 10:
                return {"error": "Insufficient returns to estimate volatility proxy"}
            # Build proxy
            if not proxy:
                return _volatility_proxy_required_error(method_l)
            proxy_l = str(proxy).lower().strip()
            eps = 1e-12
            if proxy_l == 'squared_return':
                y = r * r; back = 'sqrt'
            elif proxy_l == 'abs_return':
                y = np.abs(r); back = 'abs'
            elif proxy_l == 'log_r2':
                y = np.log(r * r + eps); back = 'exp_sqrt'
            else:
                return {
                    "success": False,
                    "error": (
                        f"Unsupported proxy: {proxy}. Use --proxy "
                        f"{'|'.join(VOLATILITY_PROXY_VALUES)}."
                    ),
                    "error_code": "invalid_volatility_proxy",
                    "method": method_l,
                    "valid_proxies": list(VOLATILITY_PROXY_VALUES),
                    "remediation": (
                        "Retry forecast_volatility_estimate with --proxy "
                        "squared_return, abs_return, or log_r2."
                    ),
                }
            y = y[np.isfinite(y)]
            fh = int(horizon)
            try:
                forecast_result = ForecastRegistry.get(method_l).forecast(
                    pd.Series(y.astype(float)),
                    horizon=fh,
                    seasonality=_default_seasonality_period(timeframe),
                    params=dict(p),
                    timeframe=timeframe,
                )
                yhat = np.asarray(forecast_result.forecast, dtype=float)
                model_params_used = dict(forecast_result.params_used or {})
            except Exception as exc:
                return {"error": f"{method_l.upper()} proxy forecast error: {exc}"}
            # Back-transform to per-step sigma and aggregate horizon
            clipped_forecast_steps = 0
            if back == 'sqrt':
                clipped_forecast_steps = int(np.sum(np.asarray(yhat, dtype=float) < 0.0))
                sig = np.sqrt(np.clip(yhat, 0.0, None))
            elif back == 'abs':
                sig = np.maximum(0.0, yhat) * math.sqrt(math.pi/2.0)
            else:
                # Duan smearing corrects the Jensen gap when converting a
                # forecast of E[log(r²)] back to the arithmetic r² scale.
                centered_log_r2 = y - float(np.mean(y))
                log_r2_smearing_factor = float(np.mean(np.exp(centered_log_r2)))
                if not math.isfinite(log_r2_smearing_factor) or log_r2_smearing_factor <= 0:
                    log_r2_smearing_factor = 1.0
                sig = np.sqrt(np.exp(yhat) * log_r2_smearing_factor)
            hsig = float(math.sqrt(np.sum(sig[:fh]**2)))
            # Root-mean-square forecast sigma per modeled horizon step.
            sbar = float(hsig / math.sqrt(max(1, int(fh))))
            zero_path = not math.isfinite(hsig) or hsig <= 0.0 or not np.any(sig[:fh] > 0.0)
            proxy_payload = {
                "success": True, "symbol": symbol, "timeframe": timeframe, "method": method_l, "proxy": proxy_l,
                 "horizon": int(horizon), "volatility_per_bar": sbar, "volatility_annualized": float(sbar*math.sqrt(bpy)),
                 "volatility_horizon": hsig, "volatility_horizon_annualized": _annualize_horizon_sigma(hsig, bpy, int(horizon)),
                 "params_used": {
                     **model_params_used,
                     "per_bar_volatility_basis": "forecast_horizon_rms",
                     "lookback": int(len(df)),
                     **(
                         {"log_r2_smearing_factor": log_r2_smearing_factor}
                         if back == "exp_sqrt"
                         else {}
                     ),
                 },
            }
            if clipped_forecast_steps:
                proxy_payload["clipped_forecast_steps"] = int(clipped_forecast_steps)
                proxy_payload["warnings"] = [
                    f"{clipped_forecast_steps} proxy forecast step(s) were negative and clipped to zero before converting to volatility."
                ]
            if zero_path:
                proxy_payload["trust_level"] = "unusable"
                proxy_payload["history_policy_ok"] = False
                proxy_payload.setdefault("warnings", []).append(
                    "Volatility proxy forecast collapsed to zero; do not use this estimate for sizing."
                )
            return _finalize_volatility_with_context(
                proxy_payload,
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                returns_used=int(r.size),
                live_window=as_of is None and end is None,
                detail=detail,
            )

        if method_l == 'har_rv':
            dn_spec_used = None
            if denoise is not None:
                try:
                    dn_spec_used = _normalize_denoise_spec(denoise, default_when='pre_ti')
                except Exception:
                    dn_spec_used = None
                dn_spec_used = _volatility_denoise_spec(
                    dn_spec_used,
                    method=method_l,
                    user_columns=user_denoise_columns,
                )

            try:
                rv_tf = str(p.get('rv_timeframe', 'M5')).upper()
                rv_mt5_tf = TIMEFRAME_MAP.get(rv_tf)
                if rv_mt5_tf is None:
                    return {"error": f"Invalid rv_timeframe: {rv_tf}"}
                days = int(p.get('days', 120))
                w = int(p.get('window_w', 5))
                m = int(p.get('window_m', 22))
                rv_tf_secs = TIMEFRAME_SECONDS.get(rv_tf, 300)
                bars_needed = int(days * max(1, (86400 // max(1, rv_tf_secs))) + 50)
                rates_rv, fetch_error = _fetch_mt5_rates_guarded(
                    symbol,
                    rv_mt5_tf,
                    bars_needed,
                    as_of=as_of,
                    start=start,
                    end=end,
                    timeframe=rv_tf,
                )
                if fetch_error:
                    return _volatility_fetch_error_payload(
                        fetch_error,
                        start=start,
                        end=end,
                    )
                if rates_rv is None or len(rates_rv) < 50:
                    return _volatility_no_rates_payload(
                        symbol,
                        start=start,
                        end=end,
                        observed_bars=0 if rates_rv is None else len(rates_rv),
                        minimum_bars=50,
                        data_timeframe=rv_tf,
                    )
                dfrv = pd.DataFrame(rates_rv)
                if dn_spec_used:
                    try:
                        apply_denoise(dfrv, dn_spec_used)
                    except Exception:
                        pass
                bpy, annualization_basis = _annualization_context(
                    timeframe,
                    symbol,
                    observed_times=dfrv.get("time"),
                    observed_timeframe=rv_tf,
                )
                if len(dfrv) < 10:
                    return {"error": "Insufficient intraday bars for RV"}
                observed_last_epoch = float(dfrv["time"].iloc[-1])
                if rv_tf == timeframe:
                    forecast_grid_anchor_epoch = observed_last_epoch
                else:
                    (
                        forecast_grid_anchor_epoch,
                        grid_error,
                    ) = _requested_timeframe_grid_anchor(
                        symbol,
                        mt5_tf,
                        timeframe=timeframe,
                        observed_last_epoch=observed_last_epoch,
                        as_of=as_of,
                        end=end,
                    )
                    if grid_error or forecast_grid_anchor_epoch is None:
                        return {
                            "success": False,
                            "error": (
                                "Unable to resolve the requested-timeframe candle "
                                f"grid for HAR-RV: {grid_error or 'no usable anchor'}"
                            ),
                            "error_code": "forecast_grid_unavailable",
                            "requested_timeframe": timeframe,
                            "observed_timeframe": rv_tf,
                            "remediation": (
                                "Verify that MT5 provides candles for the requested "
                                "timeframe and retry the same historical cutoff."
                            ),
                        }
                daily_rv, realized_returns, final_daily_aggregate = (
                    _har_daily_realized_variance(
                    dfrv,
                    close_col=_volatility_price_column(dfrv, dn_spec_used, "close"),
                    )
                )
                daily_rv_required = _har_rv_daily_rv_required(m)
                if len(daily_rv) < daily_rv_required:
                    return _har_rv_sample_error(
                        error=(
                            "Not enough daily RV observations for HAR-RV "
                            f"({len(daily_rv)} observed, {daily_rv_required} required)."
                        ),
                        error_code="har_rv_insufficient_daily_rv",
                        daily_rv_observed=len(daily_rv),
                        daily_rv_required=daily_rv_required,
                        aligned_rows_observed=None,
                        aligned_rows_required=HAR_RV_MIN_ALIGNED_ROWS,
                        window_m=m,
                        window_w=w,
                        days_requested=days,
                    )
                RV = daily_rv.to_numpy(dtype=float)
                Dlag = RV[:-1]

                def rmean(arr, k):
                    s = pd.Series(arr)
                    return s.rolling(window=k, min_periods=k).mean().to_numpy()

                Wlag_full = rmean(RV, w)
                Mlag_full = rmean(RV, m)
                y = RV[1:]
                Wlag = Wlag_full[:-1]
                Mlag = Mlag_full[:-1]
                Xd = Dlag
                mask = np.isfinite(Xd) & np.isfinite(Wlag) & np.isfinite(Mlag) & np.isfinite(y)
                X = np.vstack([np.ones_like(Xd[mask]), Xd[mask], Wlag[mask], Mlag[mask]]).T
                yv = y[mask]
                if X.shape[0] < HAR_RV_MIN_ALIGNED_ROWS:
                    return _har_rv_sample_error(
                        error=(
                            "Insufficient samples after alignment for HAR-RV "
                            f"({int(X.shape[0])} aligned rows, "
                            f"{HAR_RV_MIN_ALIGNED_ROWS} required)."
                        ),
                        error_code="har_rv_insufficient_aligned_samples",
                        daily_rv_observed=len(daily_rv),
                        daily_rv_required=daily_rv_required,
                        aligned_rows_observed=int(X.shape[0]),
                        aligned_rows_required=HAR_RV_MIN_ALIGNED_ROWS,
                        window_m=m,
                        window_w=w,
                        days_requested=days,
                    )
                beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
                D_last = RV[-1]
                W_last = float(pd.Series(RV).tail(w).mean())
                M_last = float(pd.Series(RV).tail(m).mean())
                rv_next = float(beta[0] + beta[1]*D_last + beta[2]*W_last + beta[3]*M_last)
                rv_next = max(0.0, rv_next)
                tf_secs = TIMEFRAME_SECONDS.get(timeframe)
                if not tf_secs:
                    return {"error": unsupported_timeframe_seconds_error(timeframe)}
                bars_per_session = _bars_per_session_from_annualization(
                    bpy, annualization_basis
                )
                if not math.isfinite(bars_per_session) or bars_per_session <= 0:
                    return {"error": "Unable to resolve HAR-RV bars per trading session"}
                sbar = float(math.sqrt(rv_next / bars_per_session))
                h_days = float(int(horizon)) / bars_per_session
                hsig = float(math.sqrt(rv_next * max(h_days, 0.0)))
                return _finalize_volatility_with_context(
                    {"success": True, "symbol": symbol, "timeframe": timeframe, "method": method_l, "horizon": int(horizon),
                     "volatility_per_bar": sbar, "volatility_annualized": float(sbar*math.sqrt(bpy)),
                     "volatility_horizon": hsig, "volatility_horizon_annualized": _annualize_horizon_sigma(hsig, bpy, int(horizon)),
                     "params_used": {"rv_timeframe": rv_tf, "window_w": w, "window_m": m,
                                      "beta": [float(b) for b in beta.tolist()],
                                      "days": days,
                                      "bars_per_session": float(bars_per_session),
                                      "daily_rv_gap_policy": "within_utc_day_returns_only",
                                      "partial_day_policy": final_daily_aggregate.get("policy"),
                                      "minimum_daily_coverage_fraction": final_daily_aggregate.get(
                                          "minimum_coverage_fraction"
                                      )},
                     "final_daily_aggregate": final_daily_aggregate,
                     **(
                         {"warnings": [
                             "Excluded the final incomplete UTC-day realized-variance aggregate from HAR lags."
                         ]}
                         if not final_daily_aggregate.get("included_in_har", True)
                         else {}
                     ),
                     "denoise_used": dn_spec_used},
                    df=dfrv,
                    symbol=symbol,
                    timeframe=timeframe,
                    returns_used=int(realized_returns),
                    live_window=as_of is None and end is None,
                    detail=detail,
                    data_timeframe=rv_tf,
                    forecast_grid_anchor_epoch=forecast_grid_anchor_epoch,
                )
            except Exception as ex:
                return {"error": f"HAR-RV error: {ex}"}

        # Direct volatility methods
        # Fetch history sized by method
        def _need_bars_direct() -> int:
            requested = int(lookback) if lookback is not None else 0
            if method_l == 'ewma':
                lb = int(p.get('lookback', 1500)); return max(lb + 5, requested, int(horizon) + 5)
            if method_l in {'parkinson','gk','rs','yang_zhang','rolling_std','realized_kernel'}:
                w = int(p.get('window', 20)); return max(w + int(horizon) + 10, requested, 60)
            if method_l in garch_family:
                fb = int(p.get('fit_bars', 2000)); return max(fb + 10, requested, int(horizon) + 10)
            return max(300, requested, int(horizon) + 50)

        need = _need_bars_direct()
        rates, fetch_error = _fetch_mt5_rates_guarded(
            symbol,
            mt5_tf,
            need,
            as_of=as_of,
            start=start,
            end=end,
            timeframe=timeframe,
        )
        if fetch_error:
            return _volatility_fetch_error_payload(
                fetch_error,
                start=start,
                end=end,
            )
        if rates is None or len(rates) < 3:
            return _volatility_no_rates_payload(
                symbol,
                start=start,
                end=end,
                observed_bars=0 if rates is None else len(rates),
                minimum_bars=3,
                data_timeframe=timeframe,
            )

        df = pd.DataFrame(rates)
        requested_history_lookback = (
            lookback if lookback is not None else p.get("lookback")
        )
        if requested_history_lookback is not None:
            history_bars = int(requested_history_lookback)
            if len(df) > history_bars:
                df = df.iloc[-history_bars:].copy()
        if len(df) < 3:
            return {"error": "Not enough closed bars"}
        bpy, _ = _annualization_context(
            timeframe,
            symbol,
            observed_times=df.get("time"),
        )
        # Normalize and apply denoise spec (uniform behavior)
        dn_spec_used = None
        if denoise is not None:
            try:
                dn_spec_used = _normalize_denoise_spec(denoise, default_when='pre_ti')
            except Exception:
                dn_spec_used = None
            dn_spec_used = _volatility_denoise_spec(
                dn_spec_used,
                method=method_l,
                user_columns=user_denoise_columns,
            )
            if dn_spec_used:
                apply_denoise(df, dn_spec_used)

        # Compute returns and helpers
        close_col = _volatility_price_column(df, dn_spec_used, "close")
        r = _log_returns_from_prices(df[close_col].astype(float).to_numpy())
        r = r[np.isfinite(r)]
        if r.size < 5:
            return {"error": "Insufficient returns to estimate volatility"}
        if method_l == 'ewma':
            lb = int(p.get('lookback', 1500))
            halflife = p.get('halflife')
            lam = p.get('lambda_', 0.94)
            lambda_source = "lambda_"
            halflife_used = None
            tail = r[-lb:] if r.size >= lb else r
            if halflife is not None:
                try:
                    halflife_used = float(halflife)
                    lam = math.exp(-math.log(2.0) / halflife_used)
                    lambda_source = "halflife"
                except Exception:
                    lam = 0.94
            lam = float(lam)
            if not math.isfinite(lam) or not 0.0 < lam < 1.0:
                return {"error": "EWMA decay must be finite and strictly between 0 and 1."}
            w = np.power(lam, np.arange(len(tail)-1, -1, -1, dtype=float))
            weight_sum = float(np.sum(w))
            if not np.all(np.isfinite(w)) or weight_sum <= 0.0:
                return {"error": "EWMA produced invalid decay weights."}
            w /= weight_sum
            sigma2 = float(np.sum(w * (tail * tail)))
            if not math.isfinite(sigma2) or sigma2 < 0.0:
                return {"error": "EWMA produced a non-finite variance estimate."}
            sbar = math.sqrt(max(0.0, sigma2))
            hsig = float(sbar * math.sqrt(max(1, int(horizon))))
            params_used = {"lookback": lb, "lambda_": lam, "lambda_source": lambda_source}
            if halflife_used is not None:
                params_used["halflife"] = halflife_used
            return _finalize_volatility_with_context(
                {"success": True, "symbol": symbol, "timeframe": timeframe, "method": method_l, "horizon": int(horizon),
                 "volatility_per_bar": sbar, "volatility_annualized": float(sbar*math.sqrt(bpy)),
                 "volatility_horizon": hsig, "volatility_horizon_annualized": _annualize_horizon_sigma(hsig, bpy, int(horizon)),
                 "params_used": params_used,
                 "params_explained": _ewma_param_explanations(lambda_source),
                 "denoise_used": dn_spec_used},
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                returns_used=int(r.size),
                live_window=as_of is None and end is None,
                detail=detail,
            )

        if method_l in {'parkinson','gk','rs','yang_zhang','rolling_std'}:
            window = int(p.get('window', 20))
            if window < 1:
                return {"error": "window must be at least 1 bar."}
            o = df[_volatility_price_column(df, dn_spec_used, "open")].astype(float).to_numpy()
            h = df[_volatility_price_column(df, dn_spec_used, "high")].astype(float).to_numpy()
            l = df[_volatility_price_column(df, dn_spec_used, "low")].astype(float).to_numpy()
            c = df[_volatility_price_column(df, dn_spec_used, "close")].astype(float).to_numpy()
            if method_l == 'parkinson':
                v = _parkinson_sigma_sq(h, l)
            elif method_l == 'gk':
                v = _garman_klass_sigma_sq(o, h, l, c)
            elif method_l == 'rs':
                v = _rogers_satchell_sigma_sq(o, h, l, c)
            elif method_l == 'yang_zhang':
                with np.errstate(divide='ignore', invalid='ignore'):
                    oc = np.log(np.maximum(o[1:], 1e-12)) - np.log(np.maximum(c[:-1], 1e-12))
                    co = np.log(np.maximum(c[1:], 1e-12)) - np.log(np.maximum(o[1:], 1e-12))
                    rs = (
                        (np.log(np.maximum(h[1:], 1e-12)) - np.log(np.maximum(c[1:], 1e-12)))
                        * (np.log(np.maximum(h[1:], 1e-12)) - np.log(np.maximum(o[1:], 1e-12)))
                        + (np.log(np.maximum(l[1:], 1e-12)) - np.log(np.maximum(c[1:], 1e-12)))
                        * (np.log(np.maximum(l[1:], 1e-12)) - np.log(np.maximum(o[1:], 1e-12)))
                    )
                k = 0.34/(1.34 + (window+1)/(window-1)) if window>1 else 0.34
                co_var = pd.Series(co).rolling(window=window, min_periods=window).var(ddof=0).to_numpy()
                oc_var = pd.Series(oc).rolling(window=window, min_periods=window).var(ddof=0).to_numpy()
                rs_mean = pd.Series(rs).rolling(window=window, min_periods=window).mean().to_numpy()
                v = (oc_var + k*co_var + (1-k)*rs_mean)
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    simple_returns = np.diff(c) / c[:-1]
                v = (
                    pd.Series(simple_returns)
                    .rolling(window=window, min_periods=window)
                    .var(ddof=0)
                    .to_numpy()
                )
            if method_l in {'parkinson', 'gk', 'rs'}:
                range_tail = np.asarray(v[-window:], dtype=float)
                finite_tail = range_tail[np.isfinite(range_tail)]
                if finite_tail.size < window:
                    return {
                        "error": (
                            f"{method_l} requires {window} finite range observations; "
                            f"only {finite_tail.size} are available."
                        )
                    }
                sigma2 = float(np.mean(finite_tail))
            else:
                finite_tail = np.asarray(v[-window:], dtype=float)
                finite_tail = finite_tail[np.isfinite(finite_tail)]
                if finite_tail.size == 0:
                    return {
                        "error": (
                            f"{method_l} requires at least {window} applicable "
                            "observations; no finite rolling estimate is available."
                        )
                    }
                sigma2 = float(finite_tail[-1])
            if not math.isfinite(sigma2):
                return {"error": f"{method_l} produced a non-finite variance estimate."}
            sbar = math.sqrt(max(0.0, sigma2))
            hsig = float(sbar * math.sqrt(max(1, int(horizon))))
            return _finalize_volatility_with_context(
                {"success": True, "symbol": symbol, "timeframe": timeframe, "method": method_l, "horizon": int(horizon),
                 "volatility_per_bar": sbar, "volatility_annualized": float(sbar*math.sqrt(bpy)),
                 "volatility_horizon": hsig, "volatility_horizon_annualized": _annualize_horizon_sigma(hsig, bpy, int(horizon)),
                 "params_used": {"window": int(window)},
                 "denoise_used": dn_spec_used},
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                returns_used=int(r.size),
                live_window=as_of is None and end is None,
                detail=detail,
            )

        if method_l == 'realized_kernel':
            window = int(p.get('window', 50))
            kernel = str(p.get('kernel', 'tukey_hanning') or 'tukey_hanning')
            bandwidth = p.get('bandwidth')
            try:
                bandwidth_val = int(bandwidth) if bandwidth is not None else None
            except Exception:
                bandwidth_val = None
            tail = r[-window:] if r.size >= window else r
            rk_var = _realized_kernel_variance(tail, bandwidth=bandwidth_val, kernel=kernel)
            if not math.isfinite(rk_var) or rk_var < 0:
                return {"error": "Failed to compute realized kernel variance"}
            sigma_bar = math.sqrt(rk_var)
            sigma_h = math.sqrt(max(1, int(horizon)) * rk_var)
            return _finalize_volatility_with_context(
                {
                    "success": True,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "method": method_l,
                    "horizon": int(horizon),
                    "volatility_per_bar": float(sigma_bar),
                    "volatility_annualized": float(sigma_bar * math.sqrt(bpy)),
                    "volatility_horizon": float(sigma_h),
                    "volatility_horizon_annualized": _annualize_horizon_sigma(float(sigma_h), bpy, int(horizon)),
                    "params_used": {"window": int(window), "kernel": kernel, "bandwidth": bandwidth_val},
                    "denoise_used": dn_spec_used,
                },
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                returns_used=int(r.size),
                live_window=as_of is None and end is None,
                detail=detail,
            )

        if method_l in garch_family:
            fit_bars = int(p.get('fit_bars', 2000))
            mean_model = {
                'zero': 'Zero',
                'constant': 'Constant',
            }.get(str(p.get('mean', 'Zero')).strip().lower(), 'Zero')
            dist = str(p.get('dist','normal'))
            r_pct = 100.0 * r
            r_fit = r_pct[-fit_bars:] if r_pct.size > fit_bars else r_pct
            try:
                base_method = method_l.replace('_t', '')
                if method_l.endswith('_t'):
                    dist = 'studentst'
                p_order = int(p.get('p', 1))
                q_order = int(p.get('q', 1))
                if base_method == 'egarch':
                    am = _arch_model(r_fit, mean=mean_model, vol='EGARCH', p=p_order, q=q_order, dist=dist, rescale=False)
                elif base_method == 'gjr_garch':
                    o_order = int(p.get('o', 1))
                    am = _arch_model(r_fit, mean=mean_model, vol='GARCH', p=p_order, o=o_order, q=q_order, dist=dist, rescale=False)
                elif base_method == 'figarch':
                    am = _arch_model(r_fit, mean=mean_model, vol='FIGARCH', p=p_order, q=q_order, dist=dist, rescale=False)
                else:
                    am = _arch_model(r_fit, mean=mean_model, vol='GARCH', p=p_order, q=q_order, dist=dist, rescale=False)
                res = am.fit(disp='off')
                fc = res.forecast(horizon=max(1, int(horizon)), reindex=False)
                variances = fc.variance.values[-1]
                sbar = float(math.sqrt(max(0.0, float(variances[0])))) / 100.0
                hsig = float(math.sqrt(max(0.0, float(np.sum(variances))))) / 100.0
                params_used = {k: p[k] for k in p}
                params_used.update({
                    "dist": dist,
                    "mean": mean_model,
                    "p": p_order,
                    "q": q_order,
                })
                if base_method == 'gjr_garch':
                    params_used['o'] = int(p.get('o', 1))
                return _finalize_volatility_with_context(
                    {"success": True, "symbol": symbol, "timeframe": timeframe, "method": method_l, "horizon": int(horizon),
                     "volatility_per_bar": sbar, "volatility_annualized": float(sbar*math.sqrt(bpy)),
                     "volatility_horizon": hsig, "volatility_horizon_annualized": _annualize_horizon_sigma(hsig, bpy, int(horizon)),
                     "params_used": params_used,
                     "denoise_used": dn_spec_used},
                    df=df,
                    symbol=symbol,
                    timeframe=timeframe,
                    returns_used=int(r.size),
                    live_window=as_of is None and end is None,
                    detail=detail,
                )
            except Exception as ex:
                return {"error": f"{method_l} error: {ex}"}

        return {"error": f"Unsupported direct volatility method: {method_l}"}
    except Exception as e:
        return {"error": f"Error computing volatility forecast: {str(e)}"}

