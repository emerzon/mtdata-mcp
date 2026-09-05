import difflib
import math
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence

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
from ..utils.time import _format_time_minimal, bar_close_epoch
from ..utils.utils import _parse_end_datetime, _parse_start_datetime, parse_kv_or_json
from .common import (
    _parse_as_of_bound,
    describe_forecast_calendar_treatment,
    future_as_of_error,
    next_times_from_last,
    uses_exchange_intraday_projection,
    uses_standard_weekend_projection,
)
from .common import (
    annualization_context as _annualization_context,
)
from .common import (
    default_seasonality as _default_seasonality_period,
)
from .common import (
    log_returns_from_prices as _log_returns_from_prices,
)
from .forecast_registry import (
    ForecastRegistry,
    get_forecast_method_availability_snapshot,
)
from .requests import VOLATILITY_PROXY_VALUES
from .volatility_evidence import (
    VOLATILITY_DIGEST_ALGORITHM,
    VOLATILITY_DIGEST_ENCODING,
    VOLATILITY_INPUT_EVIDENCE_SCHEMA_VERSION,
    build_array_evidence,
    build_volatility_input_evidence,
    source_positions_for_returns,
)

HAR_RV_MIN_ALIGNED_ROWS = 20
HAR_RV_MIN_DAILY_RV = 30
HAR_RV_MIN_DAILY_COVERAGE_FRACTION = 0.9
HAR_RV_MAX_MISSING_BARS_PER_GAP = 12
HAR_RV_MIN_SAME_WEEKDAY_BASELINE_OBSERVATIONS = 3
VOLATILITY_PROXY_METHODS = ("arima", "sarima", "ets", "theta")
_DEFAULT_VOLATILITY_ENSEMBLE_METHODS = (
    "ewma",
    "parkinson",
    "rolling_std",
)

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
            {
                "name": "rv_timeframe",
                "type": "str",
                "default": "M5",
                "description": (
                    "Intraday timeframe used to build realized variance; "
                    "calendar/session timeframes D1, W1, and MN1 are rejected."
                ),
            },
            {
                "name": "days",
                "type": "int",
                "default": 120,
                "description": (
                    "Maximum trailing calendar-day span used for the HAR fit, ending "
                    "at the requested cutoff and intersected with an explicit start. "
                    "This is independent of requested-timeframe lookback. Need enough "
                    "history for max(30, window_m+5) daily RV observations and 20 "
                    "aligned regression rows after the monthly lag; default 120."
                ),
            },
            {"name": "window_w", "type": "int", "default": 5, "description": "Weekly window size for HAR lags."},
            {"name": "window_m", "type": "int", "default": 22, "description": "Monthly window size for HAR lags."},
            {
                "name": "minimum_daily_coverage_fraction",
                "type": "float",
                "default": HAR_RV_MIN_DAILY_COVERAGE_FRACTION,
                "description": (
                    "Minimum observed-to-expected intraday bar and exact-return "
                    "coverage for each UTC-day RV aggregate. Expected coverage "
                    "is established causally from prior data."
                ),
            },
            {
                "name": "maximum_missing_bars_per_gap",
                "type": "int",
                "default": HAR_RV_MAX_MISSING_BARS_PER_GAP,
                "description": (
                    "Largest internal gap, measured in missing rv_timeframe bars, "
                    "allowed in an included UTC-day RV aggregate. Returns never "
                    "bridge even an allowed gap."
                ),
            },
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


def _har_rv_lookback_error() -> Dict[str, Any]:
    return {
        "success": False,
        "error": (
            "HAR-RV does not accept lookback because its fit history is built "
            "from intraday bars over a calendar-day window. Omit lookback and "
            "set params.days (and optionally params.rv_timeframe) instead."
        ),
        "error_code": "har_rv_lookback_unsupported",
        "remediation": (
            "Remove lookback and use params.days to set the maximum trailing "
            "calendar-day fit window."
        ),
    }


def _volatility_ensemble_methods(params: Dict[str, Any]) -> list[str]:
    """Return normalized ensemble components using the execution defaults."""
    methods_value = params.get("methods")
    if isinstance(methods_value, str):
        methods = [
            token.strip().lower()
            for token in methods_value.split(",")
            if token.strip()
        ]
    elif isinstance(methods_value, (list, tuple)):
        methods = [
            str(item).strip().lower()
            for item in methods_value
            if str(item).strip()
        ]
    else:
        methods = list(_DEFAULT_VOLATILITY_ENSEMBLE_METHODS)
    seen: set[str] = set()
    return [
        method
        for method in methods
        if not (method in seen or seen.add(method))
    ]


def _har_rv_lookback_requested(
    method: str,
    params: Optional[Dict[str, Any]],
    *,
    lookback_supplied: bool = False,
) -> bool:
    """Detect an unsupported HAR-RV lookback before fetching any history."""
    method_l = str(method or "").strip().lower()
    effective_params = params if isinstance(params, dict) else {}
    if method_l == "har_rv":
        return lookback_supplied or "lookback" in effective_params
    if method_l != "ensemble":
        return False
    if "har_rv" not in _volatility_ensemble_methods(effective_params):
        return False
    if lookback_supplied or "lookback" in effective_params:
        return True
    method_params = effective_params.get("method_params")
    har_params = (
        method_params.get("har_rv")
        if isinstance(method_params, dict)
        else None
    )
    return isinstance(har_params, dict) and "lookback" in har_params


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
    daily_rv_quality: Optional[Dict[str, Any]] = None,
    daily_rv: Optional[List[Dict[str, Any]]] = None,
    input_evidence: Optional[Dict[str, Any]] = None,
    remediation: Optional[str] = None,
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
        "remediation": remediation or (
            f"Retry forecast_volatility_estimate with --params days={recommended_days} "
            "(or the default days=120). HAR-RV needs at least "
            f"{daily_rv_required} daily RV observations and "
            f"{aligned_rows_required} aligned regression rows after the "
            f"window_m={window_m} monthly lag."
        ),
    }
    if aligned_rows_observed is not None:
        payload["aligned_rows_observed"] = int(aligned_rows_observed)
    if daily_rv_quality is not None:
        payload["daily_rv_quality"] = daily_rv_quality
    if daily_rv is not None:
        payload["daily_rv"] = daily_rv
    if input_evidence is not None:
        payload["input_evidence"] = input_evidence
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


def _finite_log_return_inputs(
    frame: pd.DataFrame,
    *,
    close_col: str,
    expected_interval_seconds: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return finite, cadence-valid log returns and their paired timestamps."""
    prices = pd.to_numeric(frame[close_col], errors="coerce").to_numpy(dtype=float)
    raw_returns = _log_returns_from_prices(prices)
    all_timestamps = pd.to_numeric(
        frame["time"],
        errors="coerce",
    ).to_numpy(dtype=float)
    timestamp_deltas = np.diff(all_timestamps)
    finite_mask = (
        np.isfinite(raw_returns)
        & np.isfinite(all_timestamps[:-1])
        & np.isfinite(all_timestamps[1:])
        & (timestamp_deltas > 0.0)
    )
    price_finite_count = int(np.count_nonzero(finite_mask))
    if expected_interval_seconds is not None:
        expected = float(expected_interval_seconds)
        finite_mask &= np.isclose(
            timestamp_deltas,
            expected,
            rtol=0.0,
            atol=max(1e-6, expected * 1e-9),
        )
    excluded_interval_returns = price_finite_count - int(np.count_nonzero(finite_mask))
    positions = np.flatnonzero(finite_mask).astype(np.int64)
    return (
        raw_returns[finite_mask],
        positions,
        all_timestamps[positions],
        all_timestamps[positions + 1],
        excluded_interval_returns,
    )


def _requested_timeframe_return_policy(
    timeframe: str,
) -> tuple[Optional[float], str, str]:
    """Describe the interval accepted by requested-timeframe return models."""
    if str(timeframe).upper() in CALENDAR_TIMEFRAMES:
        return (
            None,
            "adjacent_completed_session_bars_log_return",
            "adjacent_completed_session_bars",
        )
    expected = float(TIMEFRAME_SECONDS.get(timeframe, 0) or 0)
    if expected <= 0.0:
        return None, "adjacent_observed_rows_log_return", "adjacent_observed_rows"
    return (
        expected,
        "adjacent_rows_log_return_exactly_one_requested_timeframe_apart",
        "exact_requested_timeframe_interval_only",
    )


def _return_interval_filter_metadata(
    *,
    expected_interval_seconds: Optional[float],
    excluded_returns: int,
    timestamp_policy: str,
) -> Dict[str, Any]:
    return {
        "policy": timestamp_policy,
        "expected_interval_seconds": expected_interval_seconds,
        "excluded_gap_returns": int(excluded_returns),
    }


def _snapshot_volatility_raw_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> Dict[str, str]:
    """Keep private pre-denoise values available for full-detail digests."""
    snapshots: Dict[str, str] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        snapshot = f"_mtdata_volatility_raw_{column}"
        frame[snapshot] = pd.to_numeric(frame[column], errors="coerce")
        snapshots[str(column)] = snapshot
    return snapshots


def _volatility_denoise_application(
    frame: pd.DataFrame,
    spec: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(spec, dict) or not spec:
        return None
    application = frame.attrs.get("denoise_last_application")
    if not isinstance(application, dict):
        return {
            "status": "not_attested",
            "added_columns": [],
            "overwrote_columns": [],
            "warnings": [],
        }
    added = [str(value) for value in application.get("added_columns") or []]
    overwritten = [str(value) for value in application.get("overwrote_columns") or []]
    warnings_out = [str(value) for value in frame.attrs.get("denoise_warnings") or []]
    return {
        "status": "applied" if added or overwritten else "not_applied",
        "added_columns": added,
        "overwrote_columns": overwritten,
        "ohlc_geometry_repaired_rows": int(
            application.get("ohlc_geometry_repaired") or 0
        ),
        "warnings": warnings_out,
    }


def _finite_int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _finite_float_or_none(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _garch_fit_diagnostics(
    result: Any,
    *,
    method: str,
    timeframe: str,
    fit_returns: np.ndarray,
    forecast_variances_percent_sq: np.ndarray,
    expected_horizon: int,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Extract strict JSON-native ARCH fit evidence and readiness."""
    convergence_flag = _finite_int_or_none(getattr(result, "convergence_flag", None))
    optimizer_result = getattr(result, "optimization_result", None)
    raw_optimizer_success = getattr(optimizer_result, "success", None)
    optimizer_success = (
        bool(raw_optimizer_success)
        if isinstance(raw_optimizer_success, (bool, np.bool_))
        else None
    )
    optimizer_status = _finite_int_or_none(getattr(optimizer_result, "status", None))
    optimizer_iterations = _finite_int_or_none(getattr(optimizer_result, "nit", None))
    optimizer_objective = _finite_float_or_none(getattr(optimizer_result, "fun", None))
    raw_optimizer_message = getattr(optimizer_result, "message", None)
    optimizer_message = (
        str(raw_optimizer_message)[:500]
        if isinstance(raw_optimizer_message, (str, bytes))
        else None
    )
    if isinstance(raw_optimizer_message, bytes):
        optimizer_message = raw_optimizer_message.decode(
            "utf-8",
            errors="replace",
        )[:500]

    diagnostics: Dict[str, Any] = {
        "converged": False,
        "convergence_flag": convergence_flag,
        "optimizer": {
            "success": optimizer_success,
            "status": optimizer_status,
            "message": optimizer_message,
            "iterations": optimizer_iterations,
            "objective": optimizer_objective,
        },
        "fit_return_count": int(len(fit_returns)),
        "model_input_scale": "percent_log_return",
        "coefficient_parameterization": (
            "arch_native_units_for_percent_log_return_input"
        ),
        "expected_horizon": int(expected_horizon),
    }

    coefficient_error: Optional[str] = None
    coefficient_error_message: Optional[str] = None
    params = getattr(result, "params", None)
    coefficient_rows: List[Dict[str, Any]] = []
    if isinstance(params, pd.Series):
        coefficient_items = list(params.items())
    elif isinstance(params, dict):
        coefficient_items = list(params.items())
    else:
        try:
            values = np.asarray(params, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            values = np.asarray([], dtype=float)
        names = getattr(params, "index", None)
        if names is None or len(names) != len(values):
            names = [f"parameter_{index}" for index in range(len(values))]
        coefficient_items = list(zip(names, values))
    for name, value in coefficient_items:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            coefficient_error = "non_numeric_coefficient"
            coefficient_error_message = (
                "GARCH fitted coefficients are unavailable or non-numeric."
            )
            break
        if not math.isfinite(numeric):
            coefficient_error = "non_finite_coefficient"
            coefficient_error_message = (
                "GARCH fitted coefficients contain a non-finite value."
            )
            break
        coefficient_rows.append({"name": str(name), "value": numeric})
    if coefficient_error is None and not coefficient_rows:
        coefficient_error = "coefficients_unavailable"
        coefficient_error_message = "GARCH fitted coefficients are unavailable."
    coefficient_names = [row["name"] for row in coefficient_rows]
    if coefficient_error is None and (
        any(not name.strip() for name in coefficient_names)
        or len(coefficient_names) != len(set(coefficient_names))
    ):
        coefficient_error = "coefficient_names_invalid"
        coefficient_error_message = "GARCH fitted coefficient names are invalid."
    if coefficient_error is not None:
        diagnostics.update(
            {
                "coefficients_finite": False,
                "coefficient_error": coefficient_error,
            }
        )
    else:
        coefficient_values = np.asarray(
            [[row["value"] for row in coefficient_rows]],
            dtype=float,
        )
        diagnostics.update(
            {
                "coefficients_finite": True,
                "coefficients": coefficient_rows,
                "coefficient_evidence": build_array_evidence(
                    coefficient_values,
                    domain="volatility_garch_fitted_coefficients",
                    operation="ordered_arch_native_fitted_coefficients",
                    fields=coefficient_names,
                    context={"method": method, "timeframe": timeframe},
                ),
            }
        )

    try:
        variance_percent_sq = np.asarray(
            forecast_variances_percent_sq,
            dtype=float,
        ).reshape(-1)
        variance_conversion_error = None
    except (TypeError, ValueError) as exc:
        variance_percent_sq = np.asarray([], dtype=float)
        variance_conversion_error = type(exc).__name__
    variance_finite = bool(np.all(np.isfinite(variance_percent_sq)))
    variance_positive = bool(np.all(variance_percent_sq > 0.0))
    variance_count_matches = bool(variance_percent_sq.size == int(expected_horizon))
    converged = bool(convergence_flag == 0 and optimizer_success is not False)
    diagnostics.update(
        {
            "converged": converged,
            "forecast_variance_path_count": int(variance_percent_sq.size),
            "forecast_variance_path_count_matches_horizon": (variance_count_matches),
            "forecast_variance_path_finite": variance_finite,
            "forecast_variance_path_positive": variance_positive,
            "forecast_variance_unit": "decimal_return_squared",
            **(
                {"forecast_variance_conversion_error": variance_conversion_error}
                if variance_conversion_error is not None
                else {}
            ),
        }
    )
    if variance_finite:
        variance_decimal_sq = variance_percent_sq / 10_000.0
        diagnostics.update(
            {
                "forecast_variance_path": [
                    float(value) for value in variance_decimal_sq.tolist()
                ],
                "forecast_variance_path_evidence": build_array_evidence(
                    variance_decimal_sq,
                    domain="volatility_garch_forecast_variance_path",
                    operation=(
                        "arch_forecast_variance_percent_squared_divided_by_10000"
                    ),
                    fields=["forecast_variance_decimal_return_squared"],
                    context={"method": method, "timeframe": timeframe},
                ),
            }
        )
    fit_ready = bool(
        coefficient_error is None
        and variance_count_matches
        and variance_finite
        and variance_positive
        and converged
    )
    diagnostics["fit_ready"] = fit_ready
    if coefficient_error_message is not None:
        return diagnostics, coefficient_error_message
    if not (variance_count_matches and variance_finite and variance_positive):
        return diagnostics, (
            "GARCH forecast variance path must match the requested horizon and "
            "contain only finite positive values."
        )
    if not converged:
        if convergence_flag is None:
            return diagnostics, "GARCH convergence_flag is unavailable."
        if convergence_flag != 0:
            return diagnostics, (
                f"GARCH optimizer did not converge (convergence_flag={convergence_flag})."
            )
        return diagnostics, "GARCH optimizer reported success=false."
    return diagnostics, None


def _realized_variance_rows(
    frame: pd.DataFrame,
    *,
    close_col: str = "close",
    raw_close_col: Optional[str] = None,
    expected_bar_seconds: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_column = raw_close_col or close_col
    values = pd.DataFrame(
        {
            "time": pd.to_numeric(frame.get("time"), errors="coerce"),
            "close": pd.to_numeric(frame.get(close_col), errors="coerce"),
            "raw_close": pd.to_numeric(
                frame.get(raw_column),
                errors="coerce",
            ),
        }
    ).dropna(subset=["time", "close"])
    values = values.sort_values("time", kind="stable")
    values["day"] = pd.to_datetime(values["time"], unit="s", utc=True).dt.floor("D")
    values["time_delta_seconds"] = values.groupby("day", sort=True)["time"].diff()
    values["previous_time"] = values.groupby("day", sort=True)["time"].shift(1)
    values["previous_close"] = values.groupby("day", sort=True)["close"].shift(1)
    values["previous_raw_close"] = values.groupby("day", sort=True)["raw_close"].shift(
        1
    )
    step_seconds = int(expected_bar_seconds or 0)
    if step_seconds <= 0:
        positive_deltas = values.loc[
            values["time_delta_seconds"] > 0,
            "time_delta_seconds",
        ]
        if not positive_deltas.empty:
            rounded_deltas = positive_deltas.round().astype(int)
            modes = rounded_deltas.mode()
            if not modes.empty:
                step_seconds = int(modes.iloc[0])
    raw_returns = values.groupby("day", sort=True)["close"].transform(
        lambda series: np.log(series.where(series > 0)).diff()
    )
    if step_seconds > 0:
        exact_interval = np.isclose(
            values["time_delta_seconds"].to_numpy(dtype=float),
            float(step_seconds),
            rtol=0.0,
            atol=1e-6,
        )
        values["return"] = raw_returns.where(exact_interval)
    else:
        values["return"] = np.nan
    values.attrs["expected_bar_seconds"] = step_seconds or None
    finite = values[np.isfinite(values["return"])].copy()
    finite["r2"] = np.square(finite["return"].astype(float))
    return values, finite


def _har_timestamp_phase_context(
    times: np.ndarray,
    *,
    step_seconds: int,
    prior_same_weekday_phases: List[int],
    prior_same_weekday_complete_grids: int,
) -> Dict[str, Any]:
    phase_offsets = np.remainder(times, float(step_seconds))
    on_utc_phase_zero = np.isclose(
        phase_offsets,
        0.0,
        rtol=0.0,
        atol=1e-6,
    ) | np.isclose(
        phase_offsets,
        float(step_seconds),
        rtol=0.0,
        atol=1e-6,
    )
    absolute_grid_enforced = step_seconds < 3600
    utc_phase_zero_mismatches = int(np.count_nonzero(~on_utc_phase_zero))
    first_phase = float(phase_offsets[0])
    phase_consistent = bool(
        np.all(
            np.isclose(
                phase_offsets,
                first_phase,
                rtol=0.0,
                atol=1e-6,
            )
        )
    )
    phase_seconds = (
        int(round(first_phase)) % step_seconds
        if phase_consistent
        else None
    )
    if absolute_grid_enforced:
        expected_phase: Optional[int] = 0
        phase_basis = "absolute_utc_phase_zero"
    elif (
        prior_same_weekday_complete_grids
        >= HAR_RV_MIN_SAME_WEEKDAY_BASELINE_OBSERVATIONS
    ):
        expected_phase = 0
        phase_basis = "prior_same_weekday_complete_24h_utc_grid"
    elif (
        len(prior_same_weekday_phases)
        >= HAR_RV_MIN_SAME_WEEKDAY_BASELINE_OBSERVATIONS
    ):
        expected_phase = min(
            set(prior_same_weekday_phases),
            key=lambda value: (
                -prior_same_weekday_phases.count(value),
                value,
            ),
        )
        phase_basis = "mode_of_retained_same_weekday_profiles"
    else:
        expected_phase = None
        phase_basis = "same_weekday_phase_bootstrap_unready"
    phase_drift = bool(
        phase_consistent
        and expected_phase is not None
        and phase_seconds != expected_phase
    )
    return {
        "absolute_grid_validation_enforced": absolute_grid_enforced,
        "utc_phase_zero_mismatch_count": utc_phase_zero_mismatches,
        "off_grid_timestamp_count": (
            utc_phase_zero_mismatches if absolute_grid_enforced else 0
        ),
        "timestamp_phase_seconds": phase_seconds,
        "timestamp_phase_consistent": phase_consistent,
        "expected_timestamp_phase_seconds": expected_phase,
        "timestamp_phase_basis": phase_basis,
        "timestamp_phase_drift": phase_drift,
    }


def _har_expected_daily_profile(
    *,
    exact_full_utc_grid: bool,
    full_day_slots: Optional[int],
    prior_same_weekday_complete_grids: int,
    baseline_counts: List[int],
    baseline_returns: List[int],
) -> tuple[Optional[int], Optional[int], str, int]:
    if exact_full_utc_grid and full_day_slots is not None:
        return (
            int(full_day_slots),
            max(0, int(full_day_slots) - 1),
            "complete_24h_utc_grid",
            0,
        )
    if (
        prior_same_weekday_complete_grids
        >= HAR_RV_MIN_SAME_WEEKDAY_BASELINE_OBSERVATIONS
        and full_day_slots is not None
    ):
        return (
            int(full_day_slots),
            max(0, int(full_day_slots) - 1),
            "prior_same_weekday_complete_24h_utc_grid",
            int(prior_same_weekday_complete_grids),
        )
    if len(baseline_counts) >= HAR_RV_MIN_SAME_WEEKDAY_BASELINE_OBSERVATIONS:
        return (
            max(1, int(max(baseline_counts))),
            max(0, int(max(baseline_returns))),
            "high_water_of_retained_same_weekday_profiles",
            int(len(baseline_counts)),
        )
    return (
        None,
        None,
        "insufficient_prior_same_weekday_history",
        int(len(baseline_counts)),
    )


def _har_baseline_update_decision(
    *,
    role: str,
    included: bool,
    exact_full_utc_grid: bool,
    prior_same_weekday_complete_grids: int,
    baseline_established: bool,
    structurally_valid: bool,
) -> tuple[bool, str, bool, bool]:
    if role in {"leading", "leading_final"}:
        return False, "leading_boundary_not_used", True, False
    if exact_full_utc_grid and (
        prior_same_weekday_complete_grids + 1
        < HAR_RV_MIN_SAME_WEEKDAY_BASELINE_OBSERVATIONS
    ):
        return (
            False,
            "exact_full_grid_awaiting_same_weekday_corroboration",
            False,
            True,
        )
    if exact_full_utc_grid and included:
        return True, "corroborated_exact_full_grid_observation", False, False
    if exact_full_utc_grid:
        return (
            False,
            "corroborated_schedule_but_ineligible_rv_profile",
            True,
            False,
        )
    if included:
        return True, "eligible_observation", False, False
    if not baseline_established and structurally_valid:
        return True, "bootstrap_observation", False, False
    if not baseline_established:
        return (
            False,
            "structurally_invalid_bootstrap_observation_not_used",
            True,
            False,
        )
    return False, "rejected_observation_not_used", True, False


def _har_daily_realized_variance(
    frame: pd.DataFrame,
    *,
    close_col: str = "close",
    raw_close_col: Optional[str] = None,
    expected_bar_seconds: Optional[int] = None,
    minimum_coverage_fraction: float = HAR_RV_MIN_DAILY_COVERAGE_FRACTION,
    maximum_missing_bars_per_gap: int = HAR_RV_MAX_MISSING_BARS_PER_GAP,
    history_start_epoch: Optional[float] = None,
    history_cutoff_epoch: Optional[float] = None,
) -> tuple[pd.Series, int, Dict[str, Any]]:
    """Build causal, gap-aware UTC-day RV aggregates for HAR-RV."""
    values, finite = _realized_variance_rows(
        frame,
        close_col=close_col,
        raw_close_col=raw_close_col,
        expected_bar_seconds=expected_bar_seconds,
    )
    step_seconds = int(values.attrs.get("expected_bar_seconds") or 0)
    daily_rv_raw = finite.groupby("day", sort=True)["r2"].sum().astype(float)
    bar_counts = values.groupby("day", sort=True)["time"].count().astype(int)
    if bar_counts.empty:
        return daily_rv_raw, 0, {}
    if step_seconds <= 0:
        raise ValueError("Unable to resolve a positive HAR-RV bar interval")

    minimum_coverage = float(minimum_coverage_fraction)
    maximum_missing = int(maximum_missing_bars_per_gap)
    full_day_slots = (
        86400 // step_seconds
        if step_seconds > 0 and 86400 % step_seconds == 0
        else None
    )
    day_index = bar_counts.index
    candidate_start = day_index[0]
    candidate_end = day_index[-1]
    calendar_candidate_scope = "first_to_last_observed_utc_day"
    if history_start_epoch is not None and math.isfinite(float(history_start_epoch)):
        candidate_start = pd.to_datetime(
            float(history_start_epoch),
            unit="s",
            utc=True,
        ).floor("D")
        calendar_candidate_scope = "requested_start_to_last_observed_utc_day"
    if history_cutoff_epoch is not None and math.isfinite(float(history_cutoff_epoch)):
        last_closable_bar_open = float(history_cutoff_epoch) - float(step_seconds)
        candidate_end = pd.to_datetime(
            last_closable_bar_open,
            unit="s",
            utc=True,
        ).floor("D")
        calendar_candidate_scope = (
            "requested_history_bounds_through_last_closable_bar"
            if history_start_epoch is not None
            else "first_observed_to_requested_last_closable_bar"
        )
    if candidate_end < candidate_start:
        calendar_day_candidates = pd.DatetimeIndex([], tz="UTC")
    else:
        calendar_day_candidates = pd.date_range(
            start=candidate_start,
            end=candidate_end,
            freq="D",
        )
    observed_day_set = set(day_index.tolist())
    absent_observed_days = [
        day for day in calendar_day_candidates if day not in observed_day_set
    ]
    boundary_candidates = list(calendar_day_candidates[:1])
    if len(calendar_day_candidates) > 1:
        boundary_candidates.extend(calendar_day_candidates[-1:])
    absent_requested_boundary_days = [
        day for day in boundary_candidates if day not in observed_day_set
    ]
    daily_rv = daily_rv_raw.reindex(day_index).astype(float)
    values_by_day = {
        day: group for day, group in values.groupby("day", sort=True)
    }
    exact_returns_by_day = finite.groupby("day", sort=True)["return"].count()
    included_days: List[pd.Timestamp] = []
    exclusions: List[Dict[str, Any]] = []
    aggregates: Dict[pd.Timestamp, Dict[str, Any]] = {}
    prior_counts_by_weekday: Dict[int, List[int]] = {
        weekday: [] for weekday in range(7)
    }
    prior_returns_by_weekday: Dict[int, List[int]] = {
        weekday: [] for weekday in range(7)
    }
    prior_phases_by_weekday: Dict[int, List[int]] = {
        weekday: [] for weekday in range(7)
    }
    prior_complete_utc_grids_by_weekday: Dict[int, int] = {
        weekday: 0 for weekday in range(7)
    }
    rejected_baseline_updates = 0
    withheld_full_grid_updates = 0

    for position, day in enumerate(day_index):
        day_values = values_by_day[day]
        weekday = int(day.weekday())
        weekday_counts = prior_counts_by_weekday[weekday]
        weekday_returns = prior_returns_by_weekday[weekday]
        weekday_phases = prior_phases_by_weekday[weekday]
        prior_same_weekday_complete_grids = (
            prior_complete_utc_grids_by_weekday[weekday]
        )
        observed_bars = int(bar_counts.loc[day])
        times = day_values["time"].to_numpy(dtype=float)
        day_start_epoch = float(day.timestamp())
        phase_context = _har_timestamp_phase_context(
            times,
            step_seconds=step_seconds,
            prior_same_weekday_phases=weekday_phases,
            prior_same_weekday_complete_grids=(
                prior_same_weekday_complete_grids
            ),
        )
        absolute_grid_validation_enforced = bool(
            phase_context["absolute_grid_validation_enforced"]
        )
        utc_phase_zero_mismatches = int(
            phase_context["utc_phase_zero_mismatch_count"]
        )
        off_grid_timestamps = int(
            phase_context["off_grid_timestamp_count"]
        )
        day_phase_consistent = bool(
            phase_context["timestamp_phase_consistent"]
        )
        day_phase_seconds = phase_context["timestamp_phase_seconds"]
        expected_phase_seconds = phase_context[
            "expected_timestamp_phase_seconds"
        ]
        phase_basis = str(phase_context["timestamp_phase_basis"])
        timestamp_phase_drift = bool(
            phase_context["timestamp_phase_drift"]
        )
        deltas = np.diff(times)
        exact_intervals = np.isclose(
            deltas,
            float(step_seconds),
            rtol=0.0,
            atol=1e-6,
        )
        positive_grid_intervals = (
            (deltas > 0)
            & np.isclose(
                deltas / float(step_seconds),
                np.round(deltas / float(step_seconds)),
                rtol=0.0,
                atol=1e-6,
            )
        )
        irregular_intervals = int(np.count_nonzero(~positive_grid_intervals))
        gap_deltas = deltas[deltas > float(step_seconds) + 1e-6]
        missing_per_gap = (
            np.maximum(
                0,
                np.ceil(gap_deltas / float(step_seconds)).astype(int) - 1,
            )
            if gap_deltas.size
            else np.array([], dtype=int)
        )
        max_missing_observed = (
            int(missing_per_gap.max()) if missing_per_gap.size else 0
        )
        max_gap_seconds = int(round(float(gap_deltas.max()))) if gap_deltas.size else 0

        exact_full_utc_grid = bool(
            full_day_slots is not None
            and observed_bars == full_day_slots
            and times.size == full_day_slots
            and off_grid_timestamps == 0
            and math.isclose(times[0], day_start_epoch, rel_tol=0.0, abs_tol=1e-6)
            and math.isclose(
                times[-1],
                day_start_epoch + 86400 - step_seconds,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and bool(np.all(exact_intervals))
        )

        baseline_counts = list(weekday_counts)
        baseline_returns = list(weekday_returns)
        (
            expected_bars,
            expected_returns,
            expected_basis,
            baseline_observations,
        ) = _har_expected_daily_profile(
            exact_full_utc_grid=exact_full_utc_grid,
            full_day_slots=full_day_slots,
            prior_same_weekday_complete_grids=(
                prior_same_weekday_complete_grids
            ),
            baseline_counts=baseline_counts,
            baseline_returns=baseline_returns,
        )

        coverage = (
            float(observed_bars) / float(expected_bars)
            if expected_bars is not None
            else None
        )
        exact_returns_observed = int(exact_returns_by_day.get(day, 0))
        return_coverage = (
            float(exact_returns_observed) / float(expected_returns)
            if expected_returns is not None and expected_returns > 0
            else 1.0
            if expected_returns == 0 and exact_returns_observed == 0
            else None
        )
        role = (
            "leading_final"
            if len(day_index) == 1
            else "leading"
            if position == 0
            else "final"
            if position == len(day_index) - 1
            else "internal"
        )
        open_final_utc_boundary = bool(
            role in {"final", "leading_final"}
            and history_cutoff_epoch is not None
            and math.isfinite(float(history_cutoff_epoch))
            and day_start_epoch <= float(history_cutoff_epoch)
            < day_start_epoch + 86400.0
        )
        reasons: List[str] = []
        if position == 0:
            reasons.append("leading_request_boundary_non_comparable")
        if open_final_utc_boundary:
            reasons.append("open_final_utc_boundary")
        if expected_bars is None:
            reasons.append("causal_coverage_baseline_unready")
        elif coverage is not None and coverage < minimum_coverage:
            reasons.append("coverage_below_minimum")
        if (
            expected_returns is not None
            and return_coverage is not None
            and return_coverage < minimum_coverage
        ):
            reasons.append("exact_return_coverage_below_minimum")
        if irregular_intervals:
            reasons.append("irregular_timestamp_spacing")
        if not day_phase_consistent:
            reasons.append("inconsistent_timestamp_phase")
        if off_grid_timestamps:
            reasons.append("off_grid_timestamps")
        elif timestamp_phase_drift:
            reasons.append("timestamp_phase_drift")
        if max_missing_observed > maximum_missing:
            reasons.append("internal_gap_above_maximum")
        if exact_returns_observed == 0:
            reasons.append("no_exact_interval_returns")

        included = not reasons
        if included:
            included_days.append(day)
        else:
            daily_rv.loc[day] = np.nan

        structurally_valid_for_baseline = bool(
            irregular_intervals == 0
            and day_phase_consistent
            and off_grid_timestamps == 0
            and not timestamp_phase_drift
            and max_missing_observed <= maximum_missing
            and exact_returns_observed > 0
        )
        baseline_established_before_day = bool(
            len(baseline_counts)
            >= HAR_RV_MIN_SAME_WEEKDAY_BASELINE_OBSERVATIONS
            or prior_same_weekday_complete_grids
            >= HAR_RV_MIN_SAME_WEEKDAY_BASELINE_OBSERVATIONS
        )
        (
            baseline_updated,
            baseline_update_reason,
            rejected_update,
            withheld_full_grid_update,
        ) = _har_baseline_update_decision(
            role=role,
            included=included,
            exact_full_utc_grid=exact_full_utc_grid,
            prior_same_weekday_complete_grids=(
                prior_same_weekday_complete_grids
            ),
            baseline_established=baseline_established_before_day,
            structurally_valid=structurally_valid_for_baseline,
        )
        rejected_baseline_updates += int(rejected_update)
        withheld_full_grid_updates += int(withheld_full_grid_update)
        if baseline_updated:
            prior_counts_by_weekday[weekday].append(observed_bars)
            prior_returns_by_weekday[weekday].append(
                exact_returns_observed
            )
            if day_phase_seconds is not None:
                prior_phases_by_weekday[weekday].append(day_phase_seconds)
        if exact_full_utc_grid and role not in {"leading", "leading_final"}:
            prior_complete_utc_grids_by_weekday[weekday] += 1
        baseline_established_after_day = bool(
            len(prior_counts_by_weekday[weekday])
            >= HAR_RV_MIN_SAME_WEEKDAY_BASELINE_OBSERVATIONS
            or prior_complete_utc_grids_by_weekday[weekday]
            >= HAR_RV_MIN_SAME_WEEKDAY_BASELINE_OBSERVATIONS
        )

        aggregate: Dict[str, Any] = {
            "utc_day": day.strftime("%Y-%m-%d"),
            "role": role,
            "start": _format_time_minimal(float(times[0])),
            "end": _format_time_minimal(float(times[-1])),
            "observed_bars": observed_bars,
            "expected_bars": expected_bars,
            "coverage_fraction": (
                round(float(coverage), 4) if coverage is not None else None
            ),
            "observed_exact_interval_returns": exact_returns_observed,
            "expected_exact_interval_returns": expected_returns,
            "exact_return_coverage_fraction": (
                round(float(return_coverage), 4)
                if return_coverage is not None
                else None
            ),
            "minimum_coverage_fraction": minimum_coverage,
            "baseline_observations": baseline_observations,
            "structurally_valid_for_baseline": (
                structurally_valid_for_baseline
            ),
            "baseline_established_before_day": (
                baseline_established_before_day
            ),
            "baseline_updated": baseline_updated,
            "baseline_update_reason": baseline_update_reason,
            "baseline_established_after_day": (
                baseline_established_after_day
            ),
            "complete": bool(included),
            "included_in_har": bool(included),
            "open_final_utc_boundary": open_final_utc_boundary,
            "policy": "causal_utc_day_coverage_and_gap_quality",
            "expected_bars_basis": expected_basis,
            "returns_used": exact_returns_observed if included else 0,
            "maximum_gap_seconds": max_gap_seconds,
            "maximum_missing_bars_per_gap_observed": max_missing_observed,
            "irregular_interval_count": irregular_intervals,
            "timestamp_phase_seconds": day_phase_seconds,
            "timestamp_phase_consistent": day_phase_consistent,
            "expected_timestamp_phase_seconds": expected_phase_seconds,
            "timestamp_phase_basis": phase_basis,
            "timestamp_phase_drift": timestamp_phase_drift,
            "absolute_grid_validation_enforced": (
                absolute_grid_validation_enforced
            ),
            "utc_phase_zero_mismatch_count": utc_phase_zero_mismatches,
            "off_grid_timestamp_count": (
                off_grid_timestamps
                if absolute_grid_validation_enforced
                else None
            ),
            "exclusion_reasons": reasons,
        }
        aggregates[day] = aggregate
        if not included:
            exclusions.append(dict(aggregate))

    included_day_set = set(included_days)
    returns_used = int(finite["day"].isin(included_day_set).sum())
    candidate_return_intervals = int(
        sum(max(0, int(count) - 1) for count in bar_counts.to_list())
    )
    final_day = day_index[-1]
    metadata: Dict[str, Any] = {
        "policy": "causal_observed_utc_day_quality_v1",
        "rv_timeframe_seconds": step_seconds,
        "minimum_daily_coverage_fraction": minimum_coverage,
        "maximum_missing_bars_per_gap": maximum_missing,
        "minimum_same_weekday_baseline_observations": (
            HAR_RV_MIN_SAME_WEEKDAY_BASELINE_OBSERVATIONS
        ),
        "coverage_baseline_bootstrap_policy": (
            "first_3_nonleading_structurally_valid_same_weekday_days"
        ),
        "coverage_baseline_bootstrap_limitation": (
            "identically_truncated_contiguous_profiles_are_not_detectable_"
            "without_a_historical_session_calendar"
        ),
        "coverage_baseline_update_policy": (
            "high_water_never_declines_only_eligible_higher_profiles_raise_it"
        ),
        "complete_24h_grid_evidence_scope": "same_weekday_only",
        "complete_24h_grid_evidence_policy": (
            "timestamp_schedule_evidence_is_separate_from_eligible_rv_baseline_updates"
        ),
        "minimum_complete_24h_grid_observations": (
            HAR_RV_MIN_SAME_WEEKDAY_BASELINE_OBSERVATIONS
        ),
        "rejected_baseline_updates": int(rejected_baseline_updates),
        "withheld_full_grid_updates": int(withheld_full_grid_updates),
        "coverage_baseline_state_by_weekday": {
            str(weekday): {
                "retained_observations": int(len(prior_counts_by_weekday[weekday])),
                "established": bool(
                    len(prior_counts_by_weekday[weekday])
                    >= HAR_RV_MIN_SAME_WEEKDAY_BASELINE_OBSERVATIONS
                    or prior_complete_utc_grids_by_weekday[weekday]
                    >= HAR_RV_MIN_SAME_WEEKDAY_BASELINE_OBSERVATIONS
                ),
                "complete_24h_grid_observations": int(
                    prior_complete_utc_grids_by_weekday[weekday]
                ),
                "bar_count_high_water": (
                    int(max(prior_counts_by_weekday[weekday]))
                    if prior_counts_by_weekday[weekday]
                    else None
                ),
                "exact_return_high_water": (
                    int(max(prior_returns_by_weekday[weekday]))
                    if prior_returns_by_weekday[weekday]
                    else None
                ),
                "retained_timestamp_phases_seconds": sorted(
                    set(prior_phases_by_weekday[weekday])
                ),
            }
            for weekday in range(7)
        },
        "coverage_baseline_weekday_numbering": "Monday=0_through_Sunday=6",
        "return_interval_policy": ("use_only_returns_exactly_one_rv_timeframe_apart"),
        "timestamp_grid_policy": (
            "absolute_utc_phase_zero_for_subhour_timeframes"
            if absolute_grid_validation_enforced
            else "causal_same_weekday_stable_phase_for_hourly_timeframes"
        ),
        "day_position_policy": (
            "preserve_observed_utc_day_positions_with_nan_for_exclusions"
        ),
        "whole_missing_day_detection": (
            "calendar_absence_listed_session_eligibility_unknown"
        ),
        "calendar_candidate_scope": calendar_candidate_scope,
        "calendar_candidate_start": (
            calendar_day_candidates[0].strftime("%Y-%m-%d")
            if len(calendar_day_candidates)
            else None
        ),
        "calendar_candidate_end": (
            calendar_day_candidates[-1].strftime("%Y-%m-%d")
            if len(calendar_day_candidates)
            else None
        ),
        "calendar_day_candidates": [
            {
                "utc_day": day.strftime("%Y-%m-%d"),
                "classification": (
                    "observed"
                    if day in observed_day_set
                    else "unknown_without_session_calendar"
                ),
            }
            for day in calendar_day_candidates
        ],
        "absent_observed_utc_days": [
            day.strftime("%Y-%m-%d") for day in absent_observed_days
        ],
        "absent_observed_utc_days_count": int(len(absent_observed_days)),
        "absent_requested_boundary_utc_days": [
            day.strftime("%Y-%m-%d") for day in absent_requested_boundary_days
        ],
        "observed_utc_days": int(len(day_index)),
        "included_utc_days": int(len(included_days)),
        "excluded_utc_days": int(len(exclusions)),
        "candidate_return_intervals": candidate_return_intervals,
        "exact_interval_returns": int(len(finite)),
        "returns_used": returns_used,
        "return_intervals_rejected": int(candidate_return_intervals - len(finite)),
        "daily_aggregates": [dict(aggregates[day]) for day in day_index],
        "excluded_days": exclusions,
        "final_daily_aggregate": aggregates[final_day],
    }
    return daily_rv, returns_used, metadata


def _har_final_boundary_authorization(
    final_aggregate: Dict[str, Any],
    *,
    history_cutoff_epoch: float,
    expected_bar_seconds: int,
) -> Dict[str, Any]:
    """Authorize skipping only a proven, exact 24-hour final-day prefix."""
    step_seconds = int(expected_bar_seconds)
    utc_day = str(final_aggregate.get("utc_day") or "")
    try:
        day_start = datetime.fromisoformat(utc_day).replace(tzinfo=timezone.utc)
        day_start_epoch = float(day_start.timestamp())
    except (OverflowError, TypeError, ValueError):
        day_start_epoch = float("nan")

    day_boundary_open = bool(
        math.isfinite(day_start_epoch)
        and step_seconds > 0
        and day_start_epoch <= float(history_cutoff_epoch)
        < day_start_epoch + 86400.0
    )
    elapsed_seconds = (
        max(0.0, float(history_cutoff_epoch) - day_start_epoch)
        if day_boundary_open
        else 0.0
    )
    allowed_prefix_bars = (
        min(
            86400 // step_seconds,
            max(
                0,
                int(
                    math.floor(
                        elapsed_seconds / float(step_seconds) + 1e-9
                    )
                ),
            ),
        )
        if day_boundary_open and 86400 % step_seconds == 0
        else 0
    )
    observed_bars = int(final_aggregate.get("observed_bars") or 0)
    observed_returns = int(
        final_aggregate.get("observed_exact_interval_returns") or 0
    )
    try:
        observed_start = datetime.fromisoformat(
            str(final_aggregate.get("start") or "").replace("Z", "+00:00")
        ).timestamp()
        observed_end = datetime.fromisoformat(
            str(final_aggregate.get("end") or "").replace("Z", "+00:00")
        ).timestamp()
    except (OverflowError, TypeError, ValueError):
        observed_start = float("nan")
        observed_end = float("nan")

    expected_last_open = (
        day_start_epoch + (allowed_prefix_bars - 1) * step_seconds
        if allowed_prefix_bars > 0 and math.isfinite(day_start_epoch)
        else float("nan")
    )
    exclusion_reasons = set(final_aggregate.get("exclusion_reasons") or [])
    boundary_compatible_reasons = {
        "coverage_below_minimum",
        "exact_return_coverage_below_minimum",
        "open_final_utc_boundary",
    }
    prior_24h_grid_contract = bool(
        final_aggregate.get("expected_bars_basis")
        == "prior_same_weekday_complete_24h_utc_grid"
        and int(final_aggregate.get("baseline_observations") or 0)
        >= HAR_RV_MIN_SAME_WEEKDAY_BASELINE_OBSERVATIONS
    )
    exact_completed_prefix = bool(
        day_boundary_open
        and allowed_prefix_bars > 0
        and observed_bars == allowed_prefix_bars
        and observed_returns == max(0, observed_bars - 1)
        and int(final_aggregate.get("irregular_interval_count") or 0) == 0
        and int(final_aggregate.get("off_grid_timestamp_count") or 0) == 0
        and int(
            final_aggregate.get("maximum_missing_bars_per_gap_observed") or 0
        )
        == 0
        and math.isclose(
            observed_start,
            day_start_epoch,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and math.isclose(
            observed_end,
            expected_last_open,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    )
    exclusion_is_boundary_compatible = bool(
        exclusion_reasons
        and exclusion_reasons.issubset(boundary_compatible_reasons)
    )
    authorized = bool(
        not final_aggregate.get("included_in_har", True)
        and prior_24h_grid_contract
        and exact_completed_prefix
        and exclusion_is_boundary_compatible
    )
    if authorized:
        reason = "authorized_exact_open_24h_prefix"
    elif final_aggregate.get("included_in_har", True):
        reason = "final_day_already_eligible"
    elif not day_boundary_open:
        reason = "final_utc_day_closed_at_cutoff"
    elif not prior_24h_grid_contract:
        reason = "prior_24h_grid_contract_unavailable"
    elif not exact_completed_prefix:
        reason = "final_day_not_exact_completed_prefix"
    else:
        reason = "final_exclusion_has_non_boundary_quality_failure"

    return {
        "policy": (
            "require_prior_24h_grid_and_exact_gap_free_prefix_from_utc_midnight"
        ),
        "authorized": authorized,
        "reason": reason,
        "utc_day_open_at_cutoff": day_boundary_open,
        "prior_24h_grid_contract": prior_24h_grid_contract,
        "exact_completed_prefix": exact_completed_prefix,
        "exclusion_is_boundary_compatible": (
            exclusion_is_boundary_compatible
        ),
        "allowed_prefix_bars_at_cutoff": int(allowed_prefix_bars),
        "observed_prefix_bars": observed_bars,
        "observed_exact_interval_returns": observed_returns,
    }


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
    denoise_application = _volatility_denoise_application(
        df,
        payload.get("denoise_used"),
    )
    if denoise_application is not None:
        payload.setdefault("denoise_application", denoise_application)
        input_evidence = payload.get("input_evidence")
        if isinstance(input_evidence, dict):
            input_evidence.setdefault(
                "denoise_application",
                deepcopy(denoise_application),
            )
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
    if not isinstance(payload, dict):
        return payload
    detail_mode = str(detail or "compact").strip().lower()
    out = dict(payload)
    if not payload.get("success"):
        if detail_mode != "full":
            for key in (
                "daily_rv",
                "daily_rv_quality",
                "denoise_used",
                "denoise_application",
                "fit_diagnostics",
                "input_evidence",
                "params_used",
            ):
                out.pop(key, None)
        return out

    out.setdefault("volatility_unit", "return_fraction")
    out.setdefault("volatility_measure", "standard_deviation_of_returns")
    out.setdefault(
        "volatility_unit_note",
        "Volatility values are decimal return fractions; *_pct aliases are percentages.",
    )
    if out.get("bars_per_year") not in (None, ""):
        out.setdefault(
            "annualization_formula",
            "volatility_per_bar * sqrt(bars_per_year)",
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
            "daily_rv_quality",
            "daily_rv",
            "denoise_used",
            "denoise_application",
            "fit_diagnostics",
            "input_evidence",
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


def _utc_epoch(value: datetime) -> float:
    """Return a UTC epoch, treating repository-naive datetimes as UTC."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return float(value.timestamp())


def _utc_now_epoch() -> float:
    return float(datetime.now(timezone.utc).timestamp())


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
        if _har_rv_lookback_requested(
            method_l,
            p,
            lookback_supplied=lookback is not None,
        ):
            return _har_rv_lookback_error()
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
            base_methods = _volatility_ensemble_methods(p)
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
            component_denoise = deepcopy(denoise)
            if user_denoise_columns is None and isinstance(component_denoise, dict):
                component_denoise.pop("columns", None)
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
                    denoise=component_denoise,
                    detail="full",
                )
                component_unusable = bool(
                    isinstance(result, dict)
                    and (
                        result.get("trust_level") == "unusable"
                        or result.get("history_policy_ok") is False
                    )
                )
                if (
                    not isinstance(result, dict)
                    or not result.get("success")
                    or component_unusable
                ):
                    err = result.get('error') if isinstance(result, dict) else None
                    component_error: Dict[str, Any] = {
                        "method": base_method,
                        "error": str(
                            err
                            or (
                                "Component forecast was marked unusable"
                                if component_unusable
                                else "Component forecast failed"
                            )
                        ),
                    }
                    if isinstance(result, dict) and result.get("error_code"):
                        component_error["error_code"] = str(result["error_code"])
                    elif component_unusable:
                        component_error["error_code"] = "volatility_component_unusable"
                    if (
                        isinstance(result, dict)
                        and str(detail or "compact").strip().lower() == "full"
                    ):
                        for evidence_key in (
                            "remediation",
                            "params_used",
                            "denoise_used",
                            "denoise_application",
                            "proxy",
                            "trust_level",
                            "history_policy_ok",
                            "clipped_forecast_steps",
                            "daily_rv",
                            "daily_rv_quality",
                            "final_daily_aggregate",
                            "fit_diagnostics",
                            "input_evidence",
                            "warnings",
                            "data_window",
                        ):
                            if result.get(evidence_key) is not None:
                                component_error[evidence_key] = deepcopy(
                                    result[evidence_key]
                                )
                    component_errors.append(component_error)
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
                        result.get("volatility_annualized", float("nan"))
                    ),
                    "volatility_horizon_annualized": float(
                        result.get(
                            "volatility_horizon_annualized",
                            result.get("volatility_annualized", float("nan")),
                        )
                    ),
                }
                if result.get('proxy') is not None:
                    component_row['proxy'] = result.get('proxy')
                if str(detail or "compact").strip().lower() == "full":
                    if result.get("params_used") is not None:
                        component_row["params_used"] = deepcopy(result["params_used"])
                    for evidence_key in (
                        "input_evidence",
                        "fit_diagnostics",
                        "denoise_used",
                        "denoise_application",
                        "daily_rv",
                        "daily_rv_quality",
                        "final_daily_aggregate",
                        "warnings",
                        "data_window",
                        "trust_level",
                        "history_policy_ok",
                        "clipped_forecast_steps",
                    ):
                        if result.get(evidence_key) is not None:
                            component_row[evidence_key] = deepcopy(result[evidence_key])
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
                return _finalize_volatility_output(
                    {
                        "success": False,
                        "error": "Ensemble failed: no successful component methods",
                        "error_code": "volatility_ensemble_all_components_failed",
                        "component_errors": component_errors,
                    },
                    detail=detail,
                )

            effective_weight_map: Dict[str, float] = {}
            if aggregator == "weighted":
                surviving_weight_total = float(
                    sum(
                        weight_map.get(str(row["method"]), 0.0)
                        for row in component_results
                    )
                )
                if surviving_weight_total <= 0.0:
                    return _finalize_volatility_output(
                        {
                            "success": False,
                            "error": (
                                "Ensemble failed: successful components have "
                                "no positive configured weight"
                            ),
                            "error_code": (
                                "volatility_ensemble_survivor_weights_invalid"
                            ),
                            "component_errors": component_errors,
                        },
                        detail=detail,
                    )
                effective_weight_map = {
                    str(row["method"]): float(
                        weight_map.get(str(row["method"]), 0.0) / surviving_weight_total
                    )
                    for row in component_results
                }

            def _aggregate_metric(metric_name: str) -> float:
                values = np.asarray([float(row[metric_name]) for row in component_results], dtype=float)
                if aggregator == 'median':
                    return float(np.median(values))
                if aggregator == "weighted" and effective_weight_map:
                    weights = np.asarray(
                        [
                            float(effective_weight_map[str(row["method"])])
                            for row in component_results
                        ],
                        dtype=float,
                    )
                    return float(np.sum(values * weights))
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
                    "weights": [
                        weight_map.get(method_name) for method_name in base_methods
                    ]
                    if weight_map
                    else None,
                    "effective_component_weights": (
                        [
                            effective_weight_map.get(str(row["method"]))
                            for row in component_results
                        ]
                        if effective_weight_map
                        else None
                    ),
                },
            }
            if proxy is not None:
                out["proxy"] = str(proxy).lower().strip()
            if expose_components:
                out["components"] = component_results
            if str(detail or "compact").strip().lower() == "full":
                aggregation_rows = np.asarray(
                    [
                        [
                            float(row["volatility_per_bar"]),
                            float(row["volatility_horizon"]),
                            float(
                                effective_weight_map.get(
                                    str(row["method"]),
                                    1.0,
                                )
                            ),
                        ]
                        for row in component_results
                    ],
                    dtype=float,
                )
                out["input_evidence"] = {
                    "schema_version": VOLATILITY_INPUT_EVIDENCE_SCHEMA_VERSION,
                    "digest_algorithm": VOLATILITY_DIGEST_ALGORITHM,
                    "digest_encoding": VOLATILITY_DIGEST_ENCODING,
                    "method": "ensemble",
                    "timeframe": timeframe,
                    "operation": f"{aggregator}_of_component_volatility_outputs",
                    "component_count": int(len(component_results)),
                    "component_methods": [
                        str(row["method"]) for row in component_results
                    ],
                    "transformed_input": build_array_evidence(
                        aggregation_rows,
                        domain="volatility_ensemble_aggregation_input",
                        operation=f"{aggregator}_component_aggregation",
                        fields=[
                            "volatility_per_bar",
                            "volatility_horizon",
                            "survivor_normalized_weight_or_one",
                        ],
                        context={"method": "ensemble", "timeframe": timeframe},
                    ),
                    "components": [
                        {
                            key: deepcopy(row[key])
                            for key in (
                                "method",
                                "proxy",
                                "params_used",
                                "input_evidence",
                                "fit_diagnostics",
                                "denoise_used",
                                "denoise_application",
                                "daily_rv",
                                "daily_rv_quality",
                                "final_daily_aggregate",
                                "trust_level",
                                "history_policy_ok",
                                "clipped_forecast_steps",
                            )
                            if row.get(key) is not None
                        }
                        for row in component_results
                    ],
                }
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
            proxy_raw_columns = _snapshot_volatility_raw_columns(
                df,
                ["close"],
            )
            bpy, _ = _annualization_context(
                timeframe,
                symbol,
                observed_times=df.get("time"),
            )
            if denoise:
                apply_denoise(df, denoise)
            close_col = _volatility_price_column(df, denoise, "close")
            (
                expected_return_interval,
                return_operation,
                return_timestamp_policy,
            ) = _requested_timeframe_return_policy(timeframe)
            (
                r,
                r_positions,
                r_start_timestamps,
                r_timestamps,
                excluded_gap_returns,
            ) = _finite_log_return_inputs(
                df,
                close_col=close_col,
                expected_interval_seconds=expected_return_interval,
            )
            if r.size < 10:
                return {
                    "error": (
                        "Insufficient returns: too few cadence-valid pairs to "
                        "estimate volatility proxy"
                    ),
                    "return_interval_filter": _return_interval_filter_metadata(
                        expected_interval_seconds=expected_return_interval,
                        excluded_returns=excluded_gap_returns,
                        timestamp_policy=return_timestamp_policy,
                    ),
                }
            # Build proxy
            if not proxy:
                return _volatility_proxy_required_error(method_l)
            proxy_l = str(proxy).lower().strip()
            eps = 1e-12
            if proxy_l == 'squared_return':
                raw_y = r * r
                back = "sqrt"
            elif proxy_l == 'abs_return':
                raw_y = np.abs(r)
                back = "abs"
            elif proxy_l == 'log_r2':
                raw_y = np.log(r * r + eps)
                back = "exp_sqrt"
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
            proxy_finite = np.isfinite(raw_y)
            y = raw_y[proxy_finite]
            proxy_returns = r[proxy_finite]
            proxy_return_positions = r_positions[proxy_finite]
            proxy_return_start_timestamps = r_start_timestamps[proxy_finite]
            proxy_return_timestamps = r_timestamps[proxy_finite]
            fh = int(horizon)
            model_params = dict(p)
            model_params.pop("lookback", None)
            input_evidence = build_volatility_input_evidence(
                df,
                method=method_l,
                timeframe=timeframe,
                operation=f"forecast_{method_l}_on_{proxy_l}_volatility_proxy",
                value_columns=[close_col],
                raw_value_columns=["close"],
                raw_source_columns=[proxy_raw_columns["close"]],
                source_positions=source_positions_for_returns(proxy_return_positions),
                returns=proxy_returns,
                return_start_timestamps=proxy_return_start_timestamps,
                return_timestamps=proxy_return_timestamps,
                return_operation=return_operation,
                return_timestamp_policy=return_timestamp_policy,
                transformed_input=y,
                transformed_fields=[proxy_l],
                transformed_operation=(
                    "square_log_return"
                    if proxy_l == "squared_return"
                    else "absolute_log_return"
                    if proxy_l == "abs_return"
                    else "log_of_squared_log_return_plus_1e-12"
                ),
            )

            def _proxy_failure(
                error: str,
                *,
                error_code: str,
                diagnostics: Dict[str, Any],
                params_used: Optional[Dict[str, Any]] = None,
            ) -> Dict[str, Any]:
                return _finalize_volatility_with_context(
                    {
                        "success": False,
                        "error": error,
                        "error_code": error_code,
                        "method": method_l,
                        "proxy": proxy_l,
                        "horizon": fh,
                        "input_evidence": input_evidence,
                        "return_interval_filter": _return_interval_filter_metadata(
                            expected_interval_seconds=expected_return_interval,
                            excluded_returns=excluded_gap_returns,
                            timestamp_policy=return_timestamp_policy,
                        ),
                        "fit_diagnostics": diagnostics,
                        "params_used": params_used
                        or {
                            **model_params,
                            "lookback": int(len(df)),
                        },
                        "denoise_used": denoise,
                    },
                    df=df,
                    symbol=symbol,
                    timeframe=timeframe,
                    returns_used=int(r.size),
                    live_window=as_of is None and end is None,
                    detail=detail,
                )

            try:
                forecast_result = ForecastRegistry.get(method_l).forecast(
                    pd.Series(y.astype(float)),
                    horizon=fh,
                    seasonality=_default_seasonality_period(timeframe),
                    params=model_params,
                    timeframe=timeframe,
                )
                model_params_used = dict(forecast_result.params_used or {})
            except Exception as exc:
                return _proxy_failure(
                    (f"{method_l.upper()} proxy forecast error: {str(exc)[:500]}"),
                    error_code="volatility_proxy_forecast_error",
                    diagnostics={
                        "forecast_ready": False,
                        "error_stage": "proxy_forecaster",
                        "exception_type": type(exc).__name__,
                        "requested_horizon": fh,
                    },
                )
            try:
                yhat = np.asarray(forecast_result.forecast, dtype=float)
            except Exception as exc:
                output_shape = getattr(forecast_result.forecast, "shape", None)
                bounded_shape = (
                    [int(size) for size in output_shape]
                    if isinstance(output_shape, tuple)
                    and len(output_shape) <= 4
                    and all(
                        isinstance(size, (int, np.integer)) and int(size) >= 0
                        for size in output_shape
                    )
                    else None
                )
                return _proxy_failure(
                    "Volatility proxy forecast output must be numeric.",
                    error_code="volatility_proxy_forecast_not_ready",
                    diagnostics={
                        "forecast_ready": False,
                        "error_stage": "proxy_forecast_output_validation",
                        "exception_type": type(exc).__name__,
                        "requested_horizon": fh,
                        **(
                            {
                                "output_shape": bounded_shape,
                                "output_ndim": int(len(bounded_shape)),
                            }
                            if bounded_shape is not None
                            else {}
                        ),
                    },
                    params_used={
                        **model_params_used,
                        "lookback": int(len(df)),
                    },
                )
            proxy_output_diagnostics = {
                "forecast_ready": bool(
                    yhat.ndim == 1 and yhat.size == fh and np.all(np.isfinite(yhat))
                ),
                "error_stage": "proxy_forecast_output_validation",
                "requested_horizon": fh,
                "output_shape": [int(size) for size in yhat.shape],
                "output_ndim": int(yhat.ndim),
                "output_count": int(yhat.size),
                "output_finite": bool(np.all(np.isfinite(yhat))),
            }
            if not proxy_output_diagnostics["forecast_ready"]:
                return _proxy_failure(
                    (
                        "Volatility proxy forecast output must be a finite "
                        f"one-dimensional path of exactly {fh} step(s)."
                    ),
                    error_code="volatility_proxy_forecast_not_ready",
                    diagnostics=proxy_output_diagnostics,
                    params_used={
                        **model_params_used,
                        "lookback": int(len(df)),
                    },
                )
            # Back-transform to per-step sigma and aggregate horizon
            clipped_forecast_steps = 0
            if back == 'sqrt':
                back_transform_policy = (
                    "clip_negative_variance_proxy_to_zero_then_square_root"
                )
                clipped_forecast_steps = int(np.sum(yhat < 0.0))
                sig = np.sqrt(np.clip(yhat, 0.0, None))
            elif back == 'abs':
                back_transform_policy = (
                    "clip_negative_absolute_return_proxy_to_zero_then_"
                    "multiply_by_sqrt_pi_over_2"
                )
                clipped_forecast_steps = int(np.sum(yhat < 0.0))
                sig = np.maximum(0.0, yhat) * math.sqrt(math.pi/2.0)
            else:
                back_transform_policy = (
                    "duan_smearing_then_exponentiate_and_square_root"
                )
                # Duan smearing corrects the Jensen gap when converting a
                # forecast of E[log(r²)] back to the arithmetic r² scale.
                centered_log_r2 = y - float(np.mean(y))
                log_r2_smearing_factor = float(np.mean(np.exp(centered_log_r2)))
                if not math.isfinite(log_r2_smearing_factor) or log_r2_smearing_factor <= 0:
                    log_r2_smearing_factor = 1.0
                with np.errstate(over="ignore", invalid="ignore"):
                    sig = np.sqrt(np.exp(yhat) * log_r2_smearing_factor)
            if not bool(np.all(np.isfinite(sig))) or not bool(np.all(sig >= 0.0)):
                return _proxy_failure(
                    "Volatility proxy back-transform produced a non-finite path.",
                    error_code="volatility_proxy_forecast_not_ready",
                    diagnostics={
                        **proxy_output_diagnostics,
                        "forecast_ready": False,
                        "error_stage": "proxy_forecast_back_transform",
                        "back_transform_policy": back_transform_policy,
                        "sigma_path_finite": bool(np.all(np.isfinite(sig))),
                        "sigma_path_nonnegative": bool(np.all(sig >= 0.0)),
                    },
                    params_used={
                        **model_params_used,
                        "lookback": int(len(df)),
                    },
                )
            hsig = float(math.sqrt(np.sum(sig**2)))
            # Root-mean-square forecast sigma per modeled horizon step.
            sbar = float(hsig / math.sqrt(max(1, int(fh))))
            if not math.isfinite(hsig) or not math.isfinite(sbar):
                return _proxy_failure(
                    "Volatility proxy aggregation produced a non-finite result.",
                    error_code="volatility_proxy_forecast_not_ready",
                    diagnostics={
                        **proxy_output_diagnostics,
                        "forecast_ready": False,
                        "error_stage": "proxy_forecast_aggregation",
                        "back_transform_policy": back_transform_policy,
                    },
                    params_used={
                        **model_params_used,
                        "lookback": int(len(df)),
                    },
                )
            zero_path = hsig <= 0.0 or not np.any(sig > 0.0)
            input_evidence["forecast_output"] = {
                "back_transform_policy": back_transform_policy,
                "clipped_forecast_steps": int(clipped_forecast_steps),
                "raw_proxy_forecast": build_array_evidence(
                    yhat,
                    domain="volatility_proxy_raw_forecast_path",
                    operation=f"{method_l}_{proxy_l}_raw_forecast_path",
                    fields=["raw_proxy_forecast"],
                    context={"method": method_l, "timeframe": timeframe},
                ),
                "per_step_sigma": build_array_evidence(
                    sig,
                    domain="volatility_proxy_sigma_path",
                    operation=back_transform_policy,
                    fields=["per_step_sigma_decimal_return"],
                    context={"method": method_l, "timeframe": timeframe},
                ),
                "horizon_aggregation": build_array_evidence(
                    np.asarray([[hsig, sbar]], dtype=float),
                    domain="volatility_proxy_horizon_aggregation",
                    operation="root_sum_squares_and_horizon_root_mean_square",
                    fields=["horizon_sigma", "per_bar_rms_sigma"],
                    context={"method": method_l, "timeframe": timeframe},
                ),
            }
            proxy_output_diagnostics.update(
                {
                    "forecast_ready": True,
                    "error_stage": None,
                    "back_transform_policy": back_transform_policy,
                    "clipped_forecast_steps": int(clipped_forecast_steps),
                }
            )
            proxy_payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method_l,
                "proxy": proxy_l,
                "horizon": int(horizon),
                "volatility_per_bar": sbar,
                "volatility_annualized": float(sbar * math.sqrt(bpy)),
                "volatility_horizon": hsig,
                "volatility_horizon_annualized": _annualize_horizon_sigma(
                    hsig, bpy, int(horizon)
                ),
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
                "input_evidence": input_evidence,
                "return_interval_filter": _return_interval_filter_metadata(
                    expected_interval_seconds=expected_return_interval,
                    excluded_returns=excluded_gap_returns,
                    timestamp_policy=return_timestamp_policy,
                ),
                "fit_diagnostics": proxy_output_diagnostics,
                "denoise_used": denoise,
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
                if rv_tf in CALENDAR_TIMEFRAMES:
                    return {
                        "success": False,
                        "error": (
                            "HAR-RV rv_timeframe must be intraday; "
                            f"{rv_tf} is a calendar/session timeframe."
                        ),
                        "error_code": "har_rv_intraday_timeframe_required",
                        "remediation": (
                            "Use an intraday rv_timeframe such as M5, M15, or H1."
                        ),
                    }
                days_value = p.get('days', 120)
                try:
                    days = int(days_value)
                    days_numeric = float(days_value)
                except (OverflowError, TypeError, ValueError):
                    return {"error": "HAR-RV days must be a positive integer."}
                if (
                    isinstance(days_value, bool)
                    or not math.isfinite(days_numeric)
                    or days_numeric != float(days)
                    or days <= 0
                ):
                    return {"error": "HAR-RV days must be a positive integer."}
                w = int(p.get('window_w', 5))
                m = int(p.get('window_m', 22))
                minimum_coverage_value = p.get(
                    "minimum_daily_coverage_fraction",
                    HAR_RV_MIN_DAILY_COVERAGE_FRACTION,
                )
                try:
                    minimum_daily_coverage = float(minimum_coverage_value)
                except (OverflowError, TypeError, ValueError):
                    return {
                        "error": (
                            "HAR-RV minimum_daily_coverage_fraction must be "
                            "greater than 0 and at most 1."
                        )
                    }
                if (
                    isinstance(minimum_coverage_value, bool)
                    or not math.isfinite(minimum_daily_coverage)
                    or not 0 < minimum_daily_coverage <= 1
                ):
                    return {
                        "error": (
                            "HAR-RV minimum_daily_coverage_fraction must be "
                            "greater than 0 and at most 1."
                        )
                    }
                maximum_missing_value = p.get(
                    "maximum_missing_bars_per_gap",
                    HAR_RV_MAX_MISSING_BARS_PER_GAP,
                )
                try:
                    maximum_missing_bars_per_gap = int(maximum_missing_value)
                    maximum_missing_numeric = float(maximum_missing_value)
                except (OverflowError, TypeError, ValueError):
                    return {
                        "error": (
                            "HAR-RV maximum_missing_bars_per_gap must be a "
                            "non-negative integer."
                        )
                    }
                if (
                    isinstance(maximum_missing_value, bool)
                    or not math.isfinite(maximum_missing_numeric)
                    or maximum_missing_numeric
                    != float(maximum_missing_bars_per_gap)
                    or maximum_missing_bars_per_gap < 0
                ):
                    return {
                        "error": (
                            "HAR-RV maximum_missing_bars_per_gap must be a "
                            "non-negative integer."
                        )
                    }
                rv_tf_secs = TIMEFRAME_SECONDS.get(rv_tf, 300)
                bars_needed = int(days * max(1, (86400 // max(1, rv_tf_secs))) + 50)
                if as_of and (start or end):
                    return _volatility_fetch_error_payload(
                        "as_of cannot be combined with start/end.",
                        start=start,
                        end=end,
                    )
                range_start_dt, range_end_dt, range_error = (
                    _volatility_range_bounds(start, end)
                )
                if range_error:
                    return _volatility_fetch_error_payload(
                        range_error,
                        start=start,
                        end=end,
                    )
                now_epoch = _utc_now_epoch()
                if as_of:
                    as_of_error = future_as_of_error(
                        as_of,
                        now_epoch=now_epoch,
                    )
                    if as_of_error:
                        return _volatility_fetch_error_payload(
                            as_of_error,
                            start=start,
                            end=end,
                        )
                    as_of_dt = _parse_as_of_bound(
                        as_of,
                        timeframe=rv_tf,
                    )
                    if as_of_dt is None:
                        return _volatility_fetch_error_payload(
                            "Invalid as_of time.",
                            start=start,
                            end=end,
                        )
                    requested_cutoff_epoch = _utc_epoch(as_of_dt)
                    history_cutoff_source = "as_of"
                elif range_end_dt is not None:
                    requested_cutoff_epoch = _utc_epoch(range_end_dt)
                    history_cutoff_source = "end" if end else "current_utc_time"
                else:
                    requested_cutoff_epoch = now_epoch
                    history_cutoff_source = "current_utc_time"
                history_cutoff_epoch = min(requested_cutoff_epoch, now_epoch)
                history_start_epoch = history_cutoff_epoch - (
                    float(days) * 86400.0
                )
                if range_start_dt is not None:
                    history_start_epoch = max(
                        history_start_epoch,
                        _utc_epoch(range_start_dt),
                    )

                effective_start = start
                if as_of is None and (start or end):
                    effective_start = _format_time_minimal(history_start_epoch)
                rates_rv, fetch_error = _fetch_mt5_rates_guarded(
                    symbol,
                    rv_mt5_tf,
                    bars_needed,
                    as_of=as_of,
                    start=effective_start,
                    end=end,
                    timeframe=rv_tf,
                )
                if fetch_error:
                    return _volatility_fetch_error_payload(
                        fetch_error,
                        start=start,
                        end=end,
                    )
                if rates_rv is None:
                    return _volatility_no_rates_payload(
                        symbol,
                        start=start,
                        end=end,
                        observed_bars=0,
                        minimum_bars=50,
                        data_timeframe=rv_tf,
                    )
                dfrv = pd.DataFrame(rates_rv)
                observed_times = pd.to_numeric(dfrv.get("time"), errors="coerce")
                finite_times = observed_times[np.isfinite(observed_times)]
                if finite_times.empty:
                    return {"error": "Insufficient intraday bars for RV"}
                observed_close_times = observed_times.map(
                    lambda value: (
                        bar_close_epoch(float(value), rv_tf)
                        if math.isfinite(float(value))
                        else float("nan")
                    )
                )
                history_mask = (
                    np.isfinite(observed_times)
                    & (observed_times >= history_start_epoch)
                    & (observed_close_times <= history_cutoff_epoch)
                )
                dfrv = dfrv.loc[history_mask].copy()
                if len(dfrv) < 50:
                    return _volatility_no_rates_payload(
                        symbol,
                        start=start,
                        end=end,
                        observed_bars=len(dfrv),
                        minimum_bars=50,
                        data_timeframe=rv_tf,
                    )
                har_raw_columns = _snapshot_volatility_raw_columns(
                    dfrv,
                    ["close"],
                )
                if dn_spec_used:
                    try:
                        apply_denoise(dfrv, dn_spec_used)
                    except Exception as exc:
                        denoise_application = _volatility_denoise_application(
                            dfrv,
                            dn_spec_used,
                        )
                        return _finalize_volatility_output(
                            {
                                "success": False,
                                "error": f"HAR-RV denoise failed: {exc}",
                                "error_code": "har_rv_denoise_failed",
                                "denoise_used": dn_spec_used,
                                **(
                                    {"denoise_application": (denoise_application)}
                                    if denoise_application is not None
                                    else {}
                                ),
                                "remediation": (
                                    "Correct or remove the denoise configuration; "
                                    "HAR-RV will not report an unattested fallback."
                                ),
                            },
                            detail=detail,
                        )
                har_denoise_application = _volatility_denoise_application(
                    dfrv,
                    dn_spec_used,
                )
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
                har_close_col = _volatility_price_column(
                    dfrv,
                    dn_spec_used,
                    "close",
                )
                daily_rv, realized_returns, daily_rv_quality = (
                    _har_daily_realized_variance(
                        dfrv,
                        close_col=har_close_col,
                        raw_close_col=har_raw_columns["close"],
                        expected_bar_seconds=int(rv_tf_secs),
                        minimum_coverage_fraction=minimum_daily_coverage,
                        maximum_missing_bars_per_gap=(maximum_missing_bars_per_gap),
                        history_start_epoch=history_start_epoch,
                        history_cutoff_epoch=history_cutoff_epoch,
                    )
                )
                final_daily_aggregate = daily_rv_quality.get(
                    "final_daily_aggregate",
                    {},
                )
                daily_rv_required = _har_rv_daily_rv_required(m)
                RV = daily_rv.to_numpy(dtype=float)
                daily_rv_vector = [
                    {
                        "utc_day": day.strftime("%Y-%m-%d"),
                        "realized_variance": (
                            float(value) if math.isfinite(float(value)) else None
                        ),
                    }
                    for day, value in zip(daily_rv.index, RV)
                ]
                daily_rv_numeric = np.column_stack(
                    (
                        np.asarray(
                            [float(day.timestamp()) for day in daily_rv.index],
                            dtype=float,
                        ),
                        RV,
                    )
                )
                daily_rv_quality["daily_rv_vector_evidence"] = {
                    **build_array_evidence(
                        daily_rv_numeric,
                        domain="volatility_har_daily_rv_vector",
                        operation=(
                            "ordered_observed_utc_day_positions_with_nan_for_"
                            "excluded_aggregates"
                        ),
                        fields=["utc_day_epoch", "daily_realized_variance"],
                        context={"method": method_l, "timeframe": rv_tf},
                    ),
                    "finite_count": int(np.count_nonzero(np.isfinite(RV))),
                    "null_count": int(np.count_nonzero(~np.isfinite(RV))),
                    "null_positions": [
                        int(position)
                        for position in np.flatnonzero(~np.isfinite(RV)).tolist()
                    ],
                }
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
                daily_rv_observed = int(np.count_nonzero(np.isfinite(RV)))
                final_day_excluded = not final_daily_aggregate.get(
                    "included_in_har",
                    True,
                )
                final_boundary_authorization = (
                    _har_final_boundary_authorization(
                        final_daily_aggregate,
                        history_cutoff_epoch=history_cutoff_epoch,
                        expected_bar_seconds=int(rv_tf_secs),
                    )
                )
                daily_rv_quality["final_boundary_authorization"] = (
                    final_boundary_authorization
                )
                final_boundary_excluded_for_lags = bool(
                    final_boundary_authorization["authorized"]
                )
                prediction_rv = (
                    RV[:-1] if final_boundary_excluded_for_lags else RV
                )
                forecast_lag_days_required = max(int(w), int(m), 1)
                forecast_lags_ready = bool(
                    len(prediction_rv) >= forecast_lag_days_required
                    and np.all(
                        np.isfinite(
                            prediction_rv[-forecast_lag_days_required:]
                        )
                    )
                )
                consecutive_recent_days = 0
                for value in prediction_rv[::-1]:
                    if not np.isfinite(value):
                        break
                    consecutive_recent_days += 1
                convergence = {
                    "daily_rv_observed": daily_rv_observed,
                    "daily_rv_required": int(daily_rv_required),
                    "aligned_rows_observed": int(X.shape[0]),
                    "aligned_rows_required": HAR_RV_MIN_ALIGNED_ROWS,
                    "forecast_lag_days_required": forecast_lag_days_required,
                    "consecutive_recent_eligible_days": int(
                        consecutive_recent_days
                    ),
                    "forecast_lags_ready": forecast_lags_ready,
                    "final_day_excluded": bool(final_day_excluded),
                    "final_day_boundary_open_at_cutoff": bool(
                        final_boundary_authorization["utc_day_open_at_cutoff"]
                    ),
                    "final_boundary_excluded_before_forecast_lags": bool(
                        final_boundary_excluded_for_lags
                    ),
                    "model_fit_ready": bool(
                        daily_rv_observed >= daily_rv_required
                        and X.shape[0] >= HAR_RV_MIN_ALIGNED_ROWS
                    ),
                    "forecast_ready": bool(
                        daily_rv_observed >= daily_rv_required
                        and X.shape[0] >= HAR_RV_MIN_ALIGNED_ROWS
                        and forecast_lags_ready
                    ),
                }
                daily_rv_quality["convergence"] = convergence

                rv_values, rv_finite = _realized_variance_rows(
                    dfrv,
                    close_col=har_close_col,
                    raw_close_col=har_raw_columns["close"],
                    expected_bar_seconds=int(rv_tf_secs),
                )
                eligible_days = set(daily_rv.index[np.isfinite(RV)].tolist())
                effective_rv_returns = rv_finite.loc[
                    rv_finite["day"].isin(eligible_days)
                ].copy()
                har_fit_input = np.column_stack((X, yv))
                input_evidence = build_volatility_input_evidence(
                    rv_values,
                    method=method_l,
                    timeframe=rv_tf,
                    operation=(
                        "gap_aware_intraday_log_returns_to_daily_rv_then_har_"
                        "ols_and_latest_lag_forecast"
                    ),
                    value_columns=["close"],
                    raw_value_columns=["close"],
                    raw_source_columns=["raw_close"],
                    source_positions=np.arange(
                        len(rv_values),
                        dtype=np.int64,
                    ),
                    returns=rv_finite["return"].to_numpy(dtype=float),
                    return_start_timestamps=rv_finite["previous_time"].to_numpy(
                        dtype=float
                    ),
                    return_timestamps=rv_finite["time"].to_numpy(dtype=float),
                    return_operation=(
                        "same_utc_day_log_return_exactly_one_rv_timeframe_apart_"
                        "consumed_by_daily_quality_and_rv_aggregation"
                    ),
                    return_timestamp_policy=(
                        "same_utc_day_exact_rv_timeframe_interval_only"
                    ),
                    transformed_input=har_fit_input,
                    transformed_fields=[
                        "intercept",
                        "daily_rv_lag",
                        "weekly_mean_rv_lag",
                        "monthly_mean_rv_lag",
                        "target_daily_rv",
                    ],
                    transformed_operation=(
                        "finite_aligned_har_ols_design_and_target_rows"
                    ),
                )
                input_evidence["source"]["upstream_effective_value_columns"] = [
                    har_close_col
                ]
                input_evidence["daily_rv_vector"] = deepcopy(
                    daily_rv_quality["daily_rv_vector_evidence"]
                )
                full_return_ledger = rv_finite[
                    [
                        "previous_time",
                        "time",
                        "previous_close",
                        "close",
                        "return",
                    ]
                ].to_numpy(dtype=float)
                eligible_return_ledger = effective_rv_returns[
                    [
                        "previous_time",
                        "time",
                        "previous_close",
                        "close",
                        "return",
                    ]
                ].to_numpy(dtype=float)
                input_evidence["exact_return_ledger"] = build_array_evidence(
                    full_return_ledger,
                    domain="volatility_har_exact_return_ledger",
                    operation=(
                        "all_gap_aware_exact_intraday_returns_consumed_by_daily_quality"
                    ),
                    fields=[
                        "previous_time",
                        "current_time",
                        "previous_effective_close",
                        "current_effective_close",
                        "log_return",
                    ],
                    context={"method": method_l, "timeframe": rv_tf},
                )
                input_evidence["eligible_return_ledger"] = build_array_evidence(
                    eligible_return_ledger,
                    domain="volatility_har_eligible_return_ledger",
                    operation="exact_intraday_returns_contributing_to_finite_daily_rv",
                    fields=[
                        "previous_time",
                        "current_time",
                        "previous_effective_close",
                        "current_effective_close",
                        "log_return",
                    ],
                    context={"method": method_l, "timeframe": rv_tf},
                )
                if har_denoise_application is not None:
                    input_evidence["denoise_application"] = deepcopy(
                        har_denoise_application
                    )

                def _har_error(**kwargs: Any) -> Dict[str, Any]:
                    kwargs.setdefault("daily_rv", daily_rv_vector)
                    kwargs.setdefault("input_evidence", input_evidence)
                    error_payload = _har_rv_sample_error(**kwargs)
                    if dn_spec_used:
                        error_payload["denoise_used"] = dn_spec_used
                    if har_denoise_application is not None:
                        error_payload["denoise_application"] = har_denoise_application
                    return _finalize_volatility_output(
                        error_payload,
                        detail=detail,
                    )

                if daily_rv_observed < daily_rv_required:
                    return _har_error(
                        error=(
                            "Not enough eligible daily RV observations for HAR-RV "
                            f"({daily_rv_observed} observed, "
                            f"{daily_rv_required} required)."
                        ),
                        error_code="har_rv_insufficient_daily_rv",
                        daily_rv_observed=daily_rv_observed,
                        daily_rv_required=daily_rv_required,
                        aligned_rows_observed=int(X.shape[0]),
                        aligned_rows_required=HAR_RV_MIN_ALIGNED_ROWS,
                        window_m=m,
                        window_w=w,
                        days_requested=days,
                        daily_rv_quality=daily_rv_quality,
                    )
                if X.shape[0] < HAR_RV_MIN_ALIGNED_ROWS:
                    return _har_error(
                        error=(
                            "Insufficient samples after alignment for HAR-RV "
                            f"({int(X.shape[0])} aligned rows, "
                            f"{HAR_RV_MIN_ALIGNED_ROWS} required)."
                        ),
                        error_code="har_rv_insufficient_aligned_samples",
                        daily_rv_observed=daily_rv_observed,
                        daily_rv_required=daily_rv_required,
                        aligned_rows_observed=int(X.shape[0]),
                        aligned_rows_required=HAR_RV_MIN_ALIGNED_ROWS,
                        window_m=m,
                        window_w=w,
                        days_requested=days,
                        daily_rv_quality=daily_rv_quality,
                    )
                if not forecast_lags_ready:
                    return _har_error(
                        error=(
                            "Recent excluded UTC-day RV positions prevent a "
                            f"contiguous {forecast_lag_days_required}-day HAR "
                            "forecast lag window."
                        ),
                        error_code="har_rv_recent_daily_quality_gap",
                        daily_rv_observed=daily_rv_observed,
                        daily_rv_required=daily_rv_required,
                        aligned_rows_observed=int(X.shape[0]),
                        aligned_rows_required=HAR_RV_MIN_ALIGNED_ROWS,
                        window_m=m,
                        window_w=w,
                        days_requested=days,
                        daily_rv_quality=daily_rv_quality,
                        remediation=(
                            "Wait for enough new eligible UTC-day RV aggregates "
                            "to move the excluded day outside both HAR lag windows, "
                            "or correct the underlying MT5 history gap. Do not fill "
                            "or shift missing bars."
                        ),
                    )
                try:
                    beta, residuals, fit_rank, singular_values = np.linalg.lstsq(
                        X,
                        yv,
                        rcond=None,
                    )
                except Exception as exc:
                    return _har_error(
                        error=f"HAR-RV least-squares fit failed: {exc}",
                        error_code="har_rv_fit_error",
                        daily_rv_observed=daily_rv_observed,
                        daily_rv_required=daily_rv_required,
                        aligned_rows_observed=int(X.shape[0]),
                        aligned_rows_required=HAR_RV_MIN_ALIGNED_ROWS,
                        window_m=m,
                        window_w=w,
                        days_requested=days,
                        daily_rv_quality=daily_rv_quality,
                        remediation=(
                            "Inspect the full-detail HAR input and quality "
                            "evidence before retrying."
                        ),
                    )
                if (
                    not bool(np.all(np.isfinite(beta)))
                    or not bool(np.all(np.isfinite(singular_values)))
                    or not bool(np.all(np.isfinite(residuals)))
                ):
                    return _har_error(
                        error="HAR-RV least-squares fit produced non-finite diagnostics.",
                        error_code="har_rv_nonfinite_fit",
                        daily_rv_observed=daily_rv_observed,
                        daily_rv_required=daily_rv_required,
                        aligned_rows_observed=int(X.shape[0]),
                        aligned_rows_required=HAR_RV_MIN_ALIGNED_ROWS,
                        window_m=m,
                        window_w=w,
                        days_requested=days,
                        daily_rv_quality=daily_rv_quality,
                        remediation=(
                            "Inspect the full-detail daily RV quality ledger and "
                            "correct non-finite source values before retrying."
                        ),
                    )
                input_evidence["ols_fit"] = {
                    "rank": int(fit_rank),
                    "columns": int(X.shape[1]),
                    "full_rank": bool(int(fit_rank) == int(X.shape[1])),
                    "coefficients_finite": True,
                    "residual_sum_squares": (
                        float(residuals[0]) if len(residuals) else None
                    ),
                    "singular_values": [
                        float(value) for value in singular_values.tolist()
                    ],
                    "singular_values_evidence": build_array_evidence(
                        singular_values,
                        domain="volatility_har_ols_singular_values",
                        operation="numpy_lstsq_singular_values",
                        fields=["singular_value"],
                        context={"method": method_l, "timeframe": rv_tf},
                    ),
                }
                D_last = prediction_rv[-1]
                W_last = float(pd.Series(prediction_rv).tail(w).mean())
                M_last = float(pd.Series(prediction_rv).tail(m).mean())
                input_evidence["forecast_lag_input"] = build_array_evidence(
                    np.asarray([[1.0, D_last, W_last, M_last]], dtype=float),
                    domain="volatility_har_forecast_lag_input",
                    operation="intercept_plus_latest_daily_weekly_monthly_rv_lags",
                    fields=[
                        "intercept",
                        "daily_rv_lag",
                        "weekly_mean_rv_lag",
                        "monthly_mean_rv_lag",
                    ],
                    context={"method": method_l, "timeframe": rv_tf},
                )
                rv_next_raw = float(
                    beta[0] + beta[1] * D_last + beta[2] * W_last + beta[3] * M_last
                )
                if not math.isfinite(rv_next_raw):
                    return _har_error(
                        error="HAR-RV produced a non-finite daily RV forecast.",
                        error_code="har_rv_nonfinite_forecast",
                        daily_rv_observed=daily_rv_observed,
                        daily_rv_required=daily_rv_required,
                        aligned_rows_observed=int(X.shape[0]),
                        aligned_rows_required=HAR_RV_MIN_ALIGNED_ROWS,
                        window_m=m,
                        window_w=w,
                        days_requested=days,
                        daily_rv_quality=daily_rv_quality,
                        remediation=(
                            "Inspect the full-detail HAR fit and input evidence "
                            "before retrying."
                        ),
                    )
                rv_next_clipped_to_zero = bool(rv_next_raw < 0.0)
                rv_next = max(0.0, rv_next_raw)
                input_evidence["forecast_output"] = build_array_evidence(
                    np.asarray(
                        [
                            [
                                *beta.tolist(),
                                rv_next_raw,
                                rv_next,
                                float(rv_next_clipped_to_zero),
                            ]
                        ],
                        dtype=float,
                    ),
                    domain="volatility_har_fit_coefficients_and_forecast",
                    operation="ols_beta_and_nonnegative_next_daily_rv_forecast",
                    fields=[
                        "intercept_beta",
                        "daily_beta",
                        "weekly_beta",
                        "monthly_beta",
                        "raw_next_daily_realized_variance",
                        "next_daily_realized_variance_after_nonnegative_clip",
                        "clipped_to_zero_flag",
                    ],
                    context={"method": method_l, "timeframe": rv_tf},
                )
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
                har_warnings: List[str] = []
                nonfinal_exclusions = [
                    item
                    for item in daily_rv_quality.get("excluded_days", [])
                    if item.get("role") not in {"final", "leading_final"}
                ]
                if nonfinal_exclusions:
                    har_warnings.append(
                        "Excluded "
                        f"{len(nonfinal_exclusions)} leading/internal UTC-day "
                        "realized-variance aggregate(s) that failed causal "
                        "coverage or gap quality; their observed-day positions "
                        "remain NaN in HAR lags."
                    )
                if final_boundary_excluded_for_lags:
                    har_warnings.append(
                        "Excluded the final incomplete UTC-day "
                        "realized-variance aggregate from HAR lags."
                    )
                elif final_day_excluded:
                    har_warnings.append(
                        "The final completed UTC-day realized-variance "
                        "aggregate failed coverage or gap quality and remains "
                        "NaN in HAR lags."
                    )
                rejected_intervals = int(
                    daily_rv_quality.get("return_intervals_rejected", 0) or 0
                )
                if rejected_intervals:
                    har_warnings.append(
                        f"Omitted {rejected_intervals} intraday return "
                        "interval(s) because its timestamps were not exactly "
                        f"one {rv_tf} step apart or its price inputs did not "
                        "produce a finite log return; HAR-RV never bridges "
                        "candle gaps."
                    )
                return _finalize_volatility_with_context(
                    {
                        "success": True,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "method": method_l,
                        "horizon": int(horizon),
                        "volatility_per_bar": sbar,
                        "volatility_annualized": float(sbar * math.sqrt(bpy)),
                        "volatility_horizon": hsig,
                        "volatility_horizon_annualized": _annualize_horizon_sigma(
                            hsig, bpy, int(horizon)
                        ),
                        "params_used": {
                            "rv_timeframe": rv_tf,
                            "window_w": w,
                            "window_m": m,
                            "beta": [float(b) for b in beta.tolist()],
                            "days": days,
                            "days_semantics": "maximum_trailing_calendar_days",
                            "history_window_policy": "trailing_calendar_days_intersect_requested_start",
                            "history_cutoff": _format_time_minimal(
                                history_cutoff_epoch
                            ),
                            "history_cutoff_epoch": float(history_cutoff_epoch),
                            "history_cutoff_source": history_cutoff_source,
                            "history_start_bound": _format_time_minimal(
                                history_start_epoch
                            ),
                            "history_start_bound_epoch": float(history_start_epoch),
                            "bars_per_session": float(bars_per_session),
                            "daily_rv_gap_policy": (
                                "exact_rv_timeframe_returns_and_causal_utc_day_quality"
                            ),
                            "daily_rv_day_position_policy": daily_rv_quality.get(
                                "day_position_policy"
                            ),
                            "partial_day_policy": daily_rv_quality.get("policy"),
                            "minimum_daily_coverage_fraction": minimum_daily_coverage,
                            "maximum_missing_bars_per_gap": (
                                maximum_missing_bars_per_gap
                            ),
                        },
                        "final_daily_aggregate": final_daily_aggregate,
                        "daily_rv": daily_rv_vector,
                        "daily_rv_quality": daily_rv_quality,
                        "input_evidence": input_evidence,
                        **({"warnings": har_warnings} if har_warnings else {}),
                        "denoise_used": dn_spec_used,
                    },
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
        direct_raw_columns = _snapshot_volatility_raw_columns(
            df,
            ["open", "high", "low", "close"],
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
        (
            expected_return_interval,
            return_operation,
            return_timestamp_policy,
        ) = _requested_timeframe_return_policy(timeframe)
        (
            r,
            r_positions,
            r_start_timestamps,
            r_timestamps,
            excluded_gap_returns,
        ) = _finite_log_return_inputs(
            df,
            close_col=close_col,
            expected_interval_seconds=expected_return_interval,
        )
        if r.size < 5:
            return {
                "error": (
                    "Insufficient returns: too few cadence-valid pairs to "
                    "estimate volatility"
                ),
                "return_interval_filter": _return_interval_filter_metadata(
                    expected_interval_seconds=expected_return_interval,
                    excluded_returns=excluded_gap_returns,
                    timestamp_policy=return_timestamp_policy,
                ),
            }
        if method_l in garch_family:
            min_garch_returns = 100
            if int(r.size) < min_garch_returns:
                return {
                    "success": False,
                    "error": (
                        f"{method_l} requires at least {min_garch_returns} returns; "
                        f"got {int(r.size)}. Increase --lookback to at least "
                        f"{min_garch_returns + 1} or use --method ewma."
                    ),
                    "error_code": "insufficient_history",
                    "details": {
                        "parameter": "lookback",
                        "method": method_l,
                        "returns_used": int(r.size),
                        "required_minimum_returns": min_garch_returns,
                    },
                }
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
            tail_count = int(len(tail))
            tail_positions = r_positions[-tail_count:]
            input_evidence = build_volatility_input_evidence(
                df,
                method=method_l,
                timeframe=timeframe,
                operation="ewma_weighted_mean_of_squared_log_returns",
                value_columns=[close_col],
                raw_value_columns=["close"],
                raw_source_columns=[direct_raw_columns["close"]],
                source_positions=source_positions_for_returns(tail_positions),
                returns=tail,
                return_start_timestamps=r_start_timestamps[-tail_count:],
                return_timestamps=r_timestamps[-tail_count:],
                return_operation=return_operation,
                return_timestamp_policy=return_timestamp_policy,
                transformed_input=np.column_stack((tail, w)),
                transformed_fields=["log_return", "normalized_ewma_weight"],
                transformed_operation=(
                    "pair_log_returns_with_normalized_exponential_weights"
                ),
            )
            return _finalize_volatility_with_context(
                {
                    "success": True,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "method": method_l,
                    "horizon": int(horizon),
                    "volatility_per_bar": sbar,
                    "volatility_annualized": float(sbar * math.sqrt(bpy)),
                    "volatility_horizon": hsig,
                    "volatility_horizon_annualized": _annualize_horizon_sigma(
                        hsig, bpy, int(horizon)
                    ),
                    "params_used": params_used,
                    "params_explained": _ewma_param_explanations(lambda_source),
                    "input_evidence": input_evidence,
                    "return_interval_filter": _return_interval_filter_metadata(
                        expected_interval_seconds=expected_return_interval,
                        excluded_returns=excluded_gap_returns,
                        timestamp_policy=return_timestamp_policy,
                    ),
                    "denoise_used": dn_spec_used,
                },
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
            open_col = _volatility_price_column(df, dn_spec_used, "open")
            high_col = _volatility_price_column(df, dn_spec_used, "high")
            low_col = _volatility_price_column(df, dn_spec_used, "low")
            range_close_col = _volatility_price_column(
                df,
                dn_spec_used,
                "close",
            )
            o = df[open_col].astype(float).to_numpy()
            h = df[high_col].astype(float).to_numpy()
            l = df[low_col].astype(float).to_numpy()
            c = df[range_close_col].astype(float).to_numpy()
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
                cadence_valid = np.zeros(simple_returns.size, dtype=bool)
                cadence_valid[r_positions] = True
                simple_returns = np.where(
                    cadence_valid,
                    simple_returns,
                    np.nan,
                )
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
                range_columns = {
                    "parkinson": ([high_col, low_col], ["high", "low"]),
                    "gk": (
                        [open_col, high_col, low_col, range_close_col],
                        ["open", "high", "low", "close"],
                    ),
                    "rs": (
                        [open_col, high_col, low_col, range_close_col],
                        ["open", "high", "low", "close"],
                    ),
                }
                effective_columns, raw_columns = range_columns[method_l]
                input_evidence = build_volatility_input_evidence(
                    df,
                    method=method_l,
                    timeframe=timeframe,
                    operation=(f"mean_of_last_{window}_{method_l}_range_variances"),
                    value_columns=effective_columns,
                    raw_value_columns=raw_columns,
                    raw_source_columns=[
                        direct_raw_columns[column] for column in raw_columns
                    ],
                    source_positions=np.arange(
                        len(df) - window,
                        len(df),
                        dtype=np.int64,
                    ),
                    returns=np.asarray([], dtype=float),
                    return_start_timestamps=np.asarray([], dtype=float),
                    return_timestamps=np.asarray([], dtype=float),
                    return_operation="not_consumed_by_range_estimator",
                    return_timestamp_policy="not_applicable_no_return_vector",
                    transformed_input=range_tail,
                    transformed_fields=["per_bar_range_variance"],
                    transformed_operation=(
                        f"{method_l}_ohlc_variance_then_arithmetic_mean"
                    ),
                )
            else:
                tail_start = max(0, len(v) - window)
                tail_positions = np.arange(tail_start, len(v), dtype=np.int64)
                finite_mask = np.isfinite(np.asarray(v[-window:], dtype=float))
                finite_tail = np.asarray(v[-window:], dtype=float)[finite_mask]
                if finite_tail.size == 0:
                    return {
                        "error": (
                            f"{method_l} requires at least {window} applicable "
                            "observations; no finite rolling estimate is available."
                        )
                    }
                sigma2 = float(finite_tail[-1])
                selected_output_position = int(tail_positions[finite_mask][-1])
                input_start = selected_output_position - window + 1
                if input_start < 0:
                    return {
                        "error": (
                            f"{method_l} requires {window} applicable observations."
                        )
                    }
                effective_return_positions = np.arange(
                    input_start,
                    selected_output_position + 1,
                    dtype=np.int64,
                )
                source_positions = np.arange(
                    input_start,
                    selected_output_position + 2,
                    dtype=np.int64,
                )
                if method_l == "yang_zhang":
                    transformed_input = np.column_stack(
                        (
                            oc[input_start : selected_output_position + 1],
                            co[input_start : selected_output_position + 1],
                            rs[input_start : selected_output_position + 1],
                        )
                    )
                    input_evidence = build_volatility_input_evidence(
                        df,
                        method=method_l,
                        timeframe=timeframe,
                        operation=("yang_zhang_last_finite_rolling_ohlc_variance"),
                        value_columns=[
                            open_col,
                            high_col,
                            low_col,
                            range_close_col,
                        ],
                        raw_value_columns=["open", "high", "low", "close"],
                        raw_source_columns=[
                            direct_raw_columns[column]
                            for column in ("open", "high", "low", "close")
                        ],
                        source_positions=source_positions,
                        returns=np.asarray([], dtype=float),
                        return_start_timestamps=np.asarray([], dtype=float),
                        return_timestamps=np.asarray([], dtype=float),
                        return_operation=(
                            "no_standalone_return_vector_yang_zhang_overnight_"
                            "component_uses_previous_observed_close"
                        ),
                        return_timestamp_policy=(
                            "yang_zhang_overnight_component_uses_adjacent_"
                            "observed_rows_no_time_gap_filter"
                        ),
                        transformed_input=transformed_input,
                        transformed_fields=[
                            "overnight_log_return",
                            "open_to_close_log_return",
                            "rogers_satchell_variance",
                        ],
                        transformed_operation=(
                            "population_variance_of_overnight_and_open_close_"
                            "returns_plus_mean_rogers_satchell"
                        ),
                    )
                else:
                    simple_tail = simple_returns[effective_return_positions]
                    range_times = pd.to_numeric(
                        df["time"],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    simple_return_start_times = range_times[effective_return_positions]
                    simple_return_times = range_times[effective_return_positions + 1]
                    centered_tail = simple_tail - float(np.mean(simple_tail))
                    input_evidence = build_volatility_input_evidence(
                        df,
                        method=method_l,
                        timeframe=timeframe,
                        operation=("population_variance_of_last_window_simple_returns"),
                        value_columns=[range_close_col],
                        raw_value_columns=["close"],
                        raw_source_columns=[direct_raw_columns["close"]],
                        source_positions=source_positions,
                        returns=simple_tail,
                        return_start_timestamps=simple_return_start_times,
                        return_timestamps=simple_return_times,
                        return_operation=return_operation.replace(
                            "log_return",
                            "simple_return",
                        ),
                        return_timestamp_policy=return_timestamp_policy,
                        transformed_input=centered_tail,
                        transformed_fields=["mean_centered_simple_return"],
                        transformed_operation="subtract_window_mean",
                    )
            if not math.isfinite(sigma2):
                return {"error": f"{method_l} produced a non-finite variance estimate."}
            sbar = math.sqrt(max(0.0, sigma2))
            hsig = float(sbar * math.sqrt(max(1, int(horizon))))
            return _finalize_volatility_with_context(
                {
                    "success": True,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "method": method_l,
                    "horizon": int(horizon),
                    "volatility_per_bar": sbar,
                    "volatility_annualized": float(sbar * math.sqrt(bpy)),
                    "volatility_horizon": hsig,
                    "volatility_horizon_annualized": _annualize_horizon_sigma(
                        hsig, bpy, int(horizon)
                    ),
                    "params_used": {"window": int(window)},
                    "input_evidence": input_evidence,
                    **(
                        {
                            "return_interval_filter": _return_interval_filter_metadata(
                                expected_interval_seconds=expected_return_interval,
                                excluded_returns=excluded_gap_returns,
                                timestamp_policy=return_timestamp_policy,
                            )
                        }
                        if method_l == "rolling_std"
                        else {}
                    ),
                    "denoise_used": dn_spec_used,
                },
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
            effective_bandwidth = (
                max(1, int(np.floor(np.sqrt(len(tail)))))
                if bandwidth_val is None
                else int(bandwidth_val)
            )
            effective_bandwidth = int(max(1, min(effective_bandwidth, len(tail) - 1)))
            rk_var = _realized_kernel_variance(tail, bandwidth=bandwidth_val, kernel=kernel)
            if not math.isfinite(rk_var) or rk_var < 0:
                return {"error": "Failed to compute realized kernel variance"}
            sigma_bar = math.sqrt(rk_var)
            sigma_h = math.sqrt(max(1, int(horizon)) * rk_var)
            tail_count = int(len(tail))
            tail_positions = r_positions[-tail_count:]
            centered_tail = tail - float(np.mean(tail))
            input_evidence = build_volatility_input_evidence(
                df,
                method=method_l,
                timeframe=timeframe,
                operation=(f"realized_kernel_{kernel}_bandwidth_{effective_bandwidth}"),
                value_columns=[close_col],
                raw_value_columns=["close"],
                raw_source_columns=[direct_raw_columns["close"]],
                source_positions=source_positions_for_returns(tail_positions),
                returns=tail,
                return_start_timestamps=r_start_timestamps[-tail_count:],
                return_timestamps=r_timestamps[-tail_count:],
                return_operation=return_operation,
                return_timestamp_policy=return_timestamp_policy,
                transformed_input=centered_tail,
                transformed_fields=["mean_centered_log_return"],
                transformed_operation="subtract_effective_window_mean",
            )
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
                    "volatility_horizon_annualized": _annualize_horizon_sigma(
                        float(sigma_h), bpy, int(horizon)
                    ),
                    "params_used": {
                        "window": int(window),
                        "kernel": kernel,
                        "bandwidth": bandwidth_val,
                        "effective_bandwidth": effective_bandwidth,
                    },
                    "input_evidence": input_evidence,
                    "return_interval_filter": _return_interval_filter_metadata(
                        expected_interval_seconds=expected_return_interval,
                        excluded_returns=excluded_gap_returns,
                        timestamp_policy=return_timestamp_policy,
                    ),
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
            base_method = method_l.replace("_t", "")
            if method_l.endswith("_t"):
                dist = "studentst"
            p_order = int(p.get("p", 1))
            q_order = int(p.get("q", 1))
            o_order = int(p.get("o", 1)) if base_method == "gjr_garch" else None
            params_used = {k: p[k] for k in p}
            params_used.update(
                {
                    "fit_bars": fit_bars,
                    "dist": dist,
                    "mean": mean_model,
                    "p": p_order,
                    "q": q_order,
                }
            )
            if o_order is not None:
                params_used["o"] = o_order
            r_pct = 100.0 * r
            r_fit = r_pct[-fit_bars:] if r_pct.size > fit_bars else r_pct
            fit_count = int(len(r_fit))
            fit_positions = r_positions[-fit_count:]
            input_evidence = build_volatility_input_evidence(
                df,
                method=method_l,
                timeframe=timeframe,
                operation="arch_model_fit_and_multi_step_variance_forecast",
                value_columns=[close_col],
                raw_value_columns=["close"],
                raw_source_columns=[direct_raw_columns["close"]],
                source_positions=source_positions_for_returns(fit_positions),
                returns=r[-fit_count:],
                return_start_timestamps=r_start_timestamps[-fit_count:],
                return_timestamps=r_timestamps[-fit_count:],
                return_operation=return_operation,
                return_timestamp_policy=return_timestamp_policy,
                transformed_input=r_fit,
                transformed_fields=["percent_log_return"],
                transformed_operation="multiply_log_return_by_100",
            )
            garch_denoise_application = _volatility_denoise_application(
                df,
                dn_spec_used,
            )
            if garch_denoise_application is not None:
                input_evidence["denoise_application"] = deepcopy(
                    garch_denoise_application
                )

            def _garch_failure(
                error: str,
                *,
                error_code: str,
                fit_diagnostics: Optional[Dict[str, Any]] = None,
            ) -> Dict[str, Any]:
                return _finalize_volatility_output(
                    {
                        "success": False,
                        "error": error,
                        "error_code": error_code,
                        **(
                            {"fit_diagnostics": fit_diagnostics}
                            if fit_diagnostics is not None
                            else {}
                        ),
                        "input_evidence": input_evidence,
                        "return_interval_filter": _return_interval_filter_metadata(
                            expected_interval_seconds=expected_return_interval,
                            excluded_returns=excluded_gap_returns,
                            timestamp_policy=return_timestamp_policy,
                        ),
                        "params_used": params_used,
                        "denoise_used": dn_spec_used,
                        **(
                            {"denoise_application": (garch_denoise_application)}
                            if garch_denoise_application is not None
                            else {}
                        ),
                    },
                    detail=detail,
                )

            try:
                if base_method == 'egarch':
                    am = _arch_model(r_fit, mean=mean_model, vol='EGARCH', p=p_order, q=q_order, dist=dist, rescale=False)
                elif base_method == 'gjr_garch':
                    am = _arch_model(r_fit, mean=mean_model, vol='GARCH', p=p_order, o=o_order, q=q_order, dist=dist, rescale=False)
                elif base_method == 'figarch':
                    am = _arch_model(r_fit, mean=mean_model, vol='FIGARCH', p=p_order, q=q_order, dist=dist, rescale=False)
                else:
                    am = _arch_model(r_fit, mean=mean_model, vol='GARCH', p=p_order, q=q_order, dist=dist, rescale=False)
                res = am.fit(disp='off')
            except Exception as ex:
                return _garch_failure(
                    f"{method_l} fit error: {str(ex)[:500]}",
                    error_code="garch_fit_error",
                    fit_diagnostics={
                        "converged": False,
                        "fit_ready": False,
                        "error_stage": "arch_fit",
                        "exception_type": type(ex).__name__,
                    },
                )
            try:
                fc = res.forecast(horizon=max(1, int(horizon)), reindex=False)
                variances = np.asarray(fc.variance.values[-1], dtype=float)
            except Exception as ex:
                fit_diagnostics, _fit_error = _garch_fit_diagnostics(
                    res,
                    method=method_l,
                    timeframe=timeframe,
                    fit_returns=r_fit,
                    forecast_variances_percent_sq=np.asarray([], dtype=float),
                    expected_horizon=max(1, int(horizon)),
                )
                if fit_diagnostics is not None:
                    fit_diagnostics.update(
                        {
                            "fit_ready": False,
                            "error_stage": "arch_forecast",
                            "exception_type": type(ex).__name__,
                        }
                    )
                return _garch_failure(
                    f"{method_l} forecast error: {str(ex)[:500]}",
                    error_code="garch_fit_error",
                    fit_diagnostics=fit_diagnostics,
                )
            fit_diagnostics, fit_error = _garch_fit_diagnostics(
                res,
                method=method_l,
                timeframe=timeframe,
                fit_returns=r_fit,
                forecast_variances_percent_sq=variances,
                expected_horizon=max(1, int(horizon)),
            )
            if fit_error is not None:
                return _garch_failure(
                    fit_error,
                    error_code="garch_fit_not_ready",
                    fit_diagnostics=fit_diagnostics,
                )
            if fit_diagnostics is None:
                return _garch_failure(
                    "GARCH fit diagnostics are unavailable.",
                    error_code="garch_fit_not_ready",
                )
            variance_decimal_sq = np.asarray(
                fit_diagnostics["forecast_variance_path"],
                dtype=float,
            )
            sbar = float(math.sqrt(float(variance_decimal_sq[0])))
            hsig = float(math.sqrt(float(np.sum(variance_decimal_sq))))
            return _finalize_volatility_with_context(
                {
                    "success": True,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "method": method_l,
                    "horizon": int(horizon),
                    "volatility_per_bar": sbar,
                    "volatility_annualized": float(sbar * math.sqrt(bpy)),
                    "volatility_horizon": hsig,
                    "volatility_horizon_annualized": _annualize_horizon_sigma(
                        hsig, bpy, int(horizon)
                    ),
                    "params_used": params_used,
                    "input_evidence": input_evidence,
                    "return_interval_filter": _return_interval_filter_metadata(
                        expected_interval_seconds=expected_return_interval,
                        excluded_returns=excluded_gap_returns,
                        timestamp_policy=return_timestamp_policy,
                    ),
                    "fit_diagnostics": fit_diagnostics,
                    "denoise_used": dn_spec_used,
                },
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                returns_used=int(r.size),
                live_window=as_of is None and end is None,
                detail=detail,
            )

        return {"error": f"Unsupported direct volatility method: {method_l}"}
    except Exception as e:
        return {"error": f"Error computing volatility forecast: {str(e)}"}
