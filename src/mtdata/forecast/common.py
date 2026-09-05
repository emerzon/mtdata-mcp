from __future__ import annotations

import math
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..services.data_service.candles import (
    _parse_candle_calendar_bound,
    fetch_history_frame,
)


def _calendar_bound_or_raise(
    value: Optional[str],
    *,
    timeframe: Optional[str],
    end_bound: bool,
) -> Optional[datetime]:
    try:
        return _parse_candle_calendar_bound(
            value,
            timeframe=timeframe,
            end_bound=end_bound,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
from ..shared.constants import CALENDAR_TIMEFRAMES, TIMEFRAME_SECONDS
from ..shared.market_sessions import (
    exchange_holidays,
    is_early_close_session,
    market_for_exchange_calendar,
)
from ..shared.symbols import (
    FOREX_CURRENCY_CODES,
    is_probably_crypto_symbol,
    is_probably_forex_symbol,
)
from ..utils.freshness import is_standard_weekend_closure, standard_weekend_window
from ..utils.time import bar_close_epoch
from ..utils.utils import _parse_end_datetime, _parse_start_datetime

_FORECAST_RESERVED_COLUMNS = {"unique_id", "ds", "y"}
_FORECAST_PREFERRED_COLUMNS = ("y_hat", "mean", "median", "pred", "forecast")
_FORECAST_AUXILIARY_COLUMN_RE = re.compile(
    r"(?:^|[-_])(lo|low|lower|hi|high|upper|interval|quantile|fitted|residual|cutoff)(?:[-_].*)?$",
    re.IGNORECASE,
)
_NF_ENV_LOCK = threading.RLock()


def edge_pad_to_length(values: np.ndarray, length: int) -> np.ndarray:
    """Validate that a 1D forecast array exactly matches `length`."""
    target = max(0, int(length))
    vals = np.asarray(values, dtype=float).ravel()
    if vals.size != target:
        raise ValueError(
            f"Forecast output length mismatch: requested {target}, received {vals.size}"
        )
    return vals.astype(float, copy=False)


def build_ci_diagnostics(
    *,
    provider: str,
    requested: bool,
    available: bool,
    status: str,
    alpha: Optional[float] = None,
    coverage: Optional[float] = None,
    level: Optional[float] = None,
    warning: Optional[str] = None,
    error: Optional[str] = None,
    error_type: Optional[str] = None,
    interval_columns: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Build standardized CI diagnostics metadata for forecast method results."""
    ci_diag: Dict[str, Any] = {
        "provider": str(provider),
        "requested": bool(requested),
        "available": bool(available),
        "status": str(status),
    }
    if alpha is not None:
        ci_diag["alpha"] = float(alpha)
    if coverage is not None:
        ci_diag["coverage"] = float(coverage)
    if level is not None:
        ci_diag["level"] = float(level)
    if warning:
        ci_diag["warning"] = str(warning)
    if error:
        ci_diag["error"] = str(error)
    if error_type:
        ci_diag["error_type"] = str(error_type)
    if interval_columns is not None:
        ci_diag["interval_columns"] = [str(col) for col in interval_columns]
    return {"diagnostics": {"ci": ci_diag}}


def log_returns_from_prices(prices: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute consecutive log-returns from a price array."""
    arr = np.asarray(prices, dtype=float).ravel()
    if arr.size < 2:
        return np.array([], dtype=float)
    try:
        eps_value = float(eps)
    except Exception as exc:
        raise ValueError("eps must be a positive finite number") from exc
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError("eps must be a positive finite number")
    finite = arr[np.isfinite(arr)]
    if finite.size and np.any(finite <= 0.0):
        raise ValueError("prices must contain only positive values")
    with np.errstate(divide='ignore', invalid='ignore'):
        rets = np.diff(np.log(np.clip(arr, eps_value, None)))
    return np.asarray(rets, dtype=float)


def _normalize_weights(weights: Any, size: int) -> Optional[np.ndarray]:
    if weights is None:
        return None
    vals: List[float] = []
    if isinstance(weights, (list, tuple)):
        vals = [float(v) for v in weights]
    elif isinstance(weights, str):
        parts = [p.strip() for p in weights.split(",") if p.strip()]
        vals = [float(p) for p in parts]
    else:
        return None
    if len(vals) != size:
        return None
    arr = np.asarray(vals, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    arr = np.clip(arr, a_min=0.0, a_max=None)
    total = float(np.sum(arr))
    if total <= 0:
        return None
    return arr / total


def _extract_forecast_values(
    Yf: Any,
    fh: int,
    method_name: str = "forecast",
    *,
    allow_actual_fallback: bool = False,
) -> "np.ndarray":
    """Extract forecast values from prediction DataFrame.
    
    Common logic for finding prediction columns and extracting values.
    """
    pred_col = None
    column_detection_error: Optional[Exception] = None
    try:
        columns = list(Yf.columns)
        pred_candidates = [c for c in columns if c not in _FORECAST_RESERVED_COLUMNS]
        numeric_candidates = []
        for candidate in pred_candidates:
            series = pd.to_numeric(Yf[candidate], errors="coerce")
            if bool(series.notna().any()):
                numeric_candidates.append(candidate)

        preferred_map = {str(candidate).lower(): candidate for candidate in numeric_candidates}
        for preferred_name in _FORECAST_PREFERRED_COLUMNS:
            pred_col = preferred_map.get(preferred_name)
            if pred_col is not None:
                break

        def _method_named_candidates(candidates: List[Any]) -> List[Any]:
            method_tokens = [
                token
                for token in re.split(r"[^a-z0-9]+", str(method_name).lower())
                if len(token) >= 3 and token not in {"forecast", "statsforecast", "neuralforecast"}
            ]
            if not method_tokens:
                return []
            matches = [
                candidate
                for candidate in candidates
                if any(token in str(candidate).lower() for token in method_tokens)
            ]
            return matches

        if pred_col is None:
            filtered_candidates = [
                candidate
                for candidate in numeric_candidates
                if _FORECAST_AUXILIARY_COLUMN_RE.search(str(candidate)) is None
            ]

            named_matches = _method_named_candidates(filtered_candidates)
            if len(named_matches) == 1:
                pred_col = named_matches[0]
            elif len(filtered_candidates) == 1:
                pred_col = filtered_candidates[0]
            elif len(numeric_candidates) == 1:
                pred_col = numeric_candidates[0]
            elif allow_actual_fallback and not numeric_candidates and "y" in columns:
                pred_col = "y"
    except Exception as exc:
        column_detection_error = exc
    
    if pred_col is None:
        columns = []
        try:
            columns = list(Yf.columns)
        except Exception:
            columns = []
        if not allow_actual_fallback and "y" in columns:
            error = RuntimeError(
                f"{method_name} prediction columns not found; refusing to use actuals column 'y'. "
                f"Available columns: {columns}"
            )
            if column_detection_error is not None:
                raise error from column_detection_error
            raise error
        error = RuntimeError(
            f"{method_name} prediction columns not found. Available columns: {columns}"
        )
        if column_detection_error is not None:
            raise error from column_detection_error
        raise error
    
    vals = np.asarray(Yf[pred_col].to_numpy(), dtype=float)
    return edge_pad_to_length(vals, int(fh))


def _as_2d_exog_array(
    value: Optional[np.ndarray],
    *,
    name: str,
) -> Optional[np.ndarray]:
    if value is None or not isinstance(value, np.ndarray) or not value.size:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"{name} must be a 1D or 2D numpy array")


def _resolve_trained_forecast_frames(
    series: Any,
    horizon: int,
    params: Optional[Dict[str, Any]] = None,
    *,
    exog_future: Any = None,
    **kwargs: Any,
) -> Tuple[Any, Optional[Any], Optional[Any]]:
    """Resolve live history and exogenous frames for a fitted forecast adapter."""
    params_used = dict(params or {})
    exog_future_arr = kwargs.get("exog_future")
    if exog_future_arr is None:
        exog_future_arr = exog_future if exog_future is not None else params_used.get("exog_future")
    exog_used = kwargs.get("exog_used")
    if exog_used is None:
        exog_used = params_used.get("exog_used")
    return _create_training_dataframes(
        getattr(series, "values", series),
        horizon,
        exog_used,
        exog_future_arr,
    )


def _create_training_dataframes(series: np.ndarray, fh: int, exog_used: Optional[np.ndarray] = None, exog_future: Optional[np.ndarray] = None) -> Tuple[Any, Optional[Any], Optional[Any]]:
    """Create standardized training DataFrames for forecast methods.
    
    Returns (Y_df, X_df, Xf_df) where:
    - Y_df: training series DataFrame
    - X_df: training exogenous features DataFrame (if provided)
    - Xf_df: future exogenous features DataFrame (if provided)
    """
    import pandas as _pd
    
    train_len = int(len(series))
    train_index = _pd.RangeIndex(start=0, stop=train_len)
    base_train = _pd.DataFrame(
        {
            'unique_id': ['ts'] * train_len,
            'ds': train_index,
        }
    )
    Y_df = base_train.copy()
    Y_df['y'] = np.asarray(series, dtype=float)
    
    X_df = None
    Xf_df = None
    exog_used_2d = _as_2d_exog_array(exog_used, name="exog_used")
    exog_future_2d = _as_2d_exog_array(exog_future, name="exog_future")
    if exog_used_2d is not None:
        cols = [f'x{i}' for i in range(exog_used_2d.shape[1])]
        X_df = base_train.copy()
        for j, cname in enumerate(cols):
            X_df[cname] = exog_used_2d[:, j]
        if exog_future_2d is not None:
            future_len = int(fh)
            future_index = _pd.RangeIndex(start=train_len, stop=train_len + future_len)
            Xf_df = _pd.DataFrame(
                {
                    'unique_id': ['ts'] * future_len,
                    'ds': future_index,
                }
            )
            for j, cname in enumerate(cols):
                Xf_df[cname] = exog_future_2d[:, j]
    
    return Y_df, X_df, Xf_df


def default_seasonality(timeframe: str, observed_times: Any = None) -> int:
    try:
        sec = TIMEFRAME_SECONDS.get(timeframe)
        if not sec or sec <= 0:
            return 0
        if sec < 86400:
            if observed_times is not None:
                try:
                    observed = pd.Series(observed_times)
                    times = pd.to_datetime(
                        observed,
                        unit="s" if pd.api.types.is_numeric_dtype(observed) else None,
                        utc=True,
                        errors="coerce",
                    )
                    valid = pd.Series(times).dropna()
                    daily_counts = valid.groupby(valid.dt.date).size()
                    if len(daily_counts) >= 2:
                        empirical = int(round(float(daily_counts.median())))
                        if empirical >= 2:
                            return empirical
                except Exception:
                    pass
            return int(round(86400.0 / float(sec)))
        if timeframe == 'D1':
            return 5
        if timeframe == 'W1':
            return 52
        if timeframe == 'MN1':
            return 12
        return 0
    except Exception:
        return 0


def observed_bars_per_session(observed_times: Any) -> Optional[float]:
    """Estimate median complete UTC-session bar count from observed timestamps."""
    try:
        observed = pd.Series(observed_times)
        times = pd.to_datetime(
            observed,
            unit="s" if pd.api.types.is_numeric_dtype(observed) else None,
            utc=True,
            errors="coerce",
        )
        valid = pd.Series(times).dropna().sort_values()
        daily_counts = valid.groupby(valid.dt.date).size()
        if len(daily_counts) < 3:
            return None
        # Fetch windows commonly start and end mid-session. Excluding those
        # edges prevents partial days from lowering the session estimate.
        complete_counts = daily_counts.iloc[1:-1]
        if complete_counts.empty:
            return None
        median = float(complete_counts.median())
        return median if math.isfinite(median) and median >= 1.0 else None
    except Exception:
        return None


def bars_per_year(
    timeframe: str,
    symbol: Optional[str] = None,
    observed_times: Any = None,
) -> float:
    """Approximate bars per year using calendar and observed session density."""
    try:
        tf = str(timeframe).upper().strip()
        secs = TIMEFRAME_SECONDS.get(tf)
        if not secs or secs <= 0:
            return float("nan")
        if tf == "MN1":
            return 12.0
        if tf == "W1":
            return 52.0
        days = (
            365.0
            if is_probably_crypto_symbol(symbol)
            else 260.0
            if is_probably_forex_symbol(symbol)
            else 252.0
        )
        if float(secs) >= 86400.0:
            return float((days * 86400.0) / float(secs))
        observed_per_session = observed_bars_per_session(observed_times)
        if observed_per_session is not None:
            return float(days * observed_per_session)
        return float((days * 24.0 * 3600.0) / float(secs))
    except Exception:
        return float("nan")


def annualization_context(
    timeframe: str,
    symbol: Optional[str] = None,
    *,
    observed_times: Any = None,
    observed_timeframe: Optional[str] = None,
) -> Tuple[float, str]:
    """Return bars per year and an explicit annualization basis."""
    tf = str(timeframe or "").strip().upper()
    if is_probably_crypto_symbol(symbol):
        return bars_per_year(tf, symbol), "365_calendar_days_24h_crypto"
    if is_probably_forex_symbol(symbol):
        return bars_per_year(tf, symbol), "260_fx_weekdays_24h"

    target_seconds = TIMEFRAME_SECONDS.get(tf)
    if target_seconds and float(target_seconds) >= 86400.0:
        return bars_per_year(tf, symbol), "252_trading_days_calendar"

    observed_per_session = observed_bars_per_session(observed_times)
    if observed_per_session is not None:
        source_tf = str(observed_timeframe or tf).strip().upper()
        source_seconds = TIMEFRAME_SECONDS.get(source_tf)
        if source_seconds and target_seconds:
            target_bars_per_session = (
                observed_per_session * float(source_seconds) / float(target_seconds)
            )
            if math.isfinite(target_bars_per_session) and target_bars_per_session > 0:
                return (
                    float(252.0 * target_bars_per_session),
                    "252_trading_days_observed_session",
                )

    return bars_per_year(tf, symbol), "252_trading_days_assumed_24h"


def quantity_to_target(quantity: str) -> str:
    """Map a forecast quantity to the corresponding price/return target mode."""
    return "return" if str(quantity).strip().lower() == "return" else "price"


def is_standard_weekend_closed_epoch(epoch: Any) -> bool:
    try:
        dt_utc = datetime.fromtimestamp(float(epoch), tz=timezone.utc)
    except Exception:
        return False
    return is_standard_weekend_closure(dt_utc)


def uses_standard_weekend_projection(symbol: Optional[str], tf_secs: int) -> bool:
    """Return whether forecasts should skip the universal non-crypto weekend."""
    return not is_probably_crypto_symbol(symbol)


def _forecast_exchange_calendar(symbol: Optional[str]) -> Optional[str]:
    """Resolve an exchange calendar only when the broker suffix is unambiguous."""
    normalized = str(symbol or "").strip().upper()
    match = re.search(r"[._-]([A-Z0-9]+)(?:[._-].*)?$", normalized)
    if match is None:
        return None
    if match.group(1) in {
        "AMEX",
        "ARCA",
        "ASE",
        "BATS",
        "NAS",
        "NASDAQ",
        "NQ",
        "NY",
        "NYSE",
        "NYS",
        "NYQ",
        "O",
        "US",
    }:
        return "XNYS"
    return None


def _is_exchange_session_holiday(calendar: str, session_date: Any) -> bool:
    return session_date in exchange_holidays(calendar, int(session_date.year))


def _exchange_holiday(
    _country: str,
    session_date: Any,
    exchange: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    if not exchange:
        return False, None
    calendar = exchange_holidays(exchange, int(session_date.year))
    holiday_name = calendar.get(session_date)
    if holiday_name is None:
        return False, None
    return True, str(holiday_name)


def _observed_intraday_session_slots(
    observed_times: Any,
    *,
    calendar: str,
) -> Optional[List[int]]:
    """Infer recurring exchange-local bar-open slots from broker observations."""
    if observed_times is None:
        return None
    try:
        values = list(observed_times)
    except TypeError:
        return None
    if not values:
        return None

    market = market_for_exchange_calendar(calendar)
    if market is None:
        return None
    exchange_tz = ZoneInfo(str(market["timezone"]))
    slots_by_date: Dict[Any, set[int]] = {}
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            epoch = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(epoch):
            continue
        local = datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone(exchange_tz)
        session_date = local.date()
        if session_date.weekday() >= 5:
            continue
        if _is_exchange_session_holiday(calendar, session_date):
            continue
        slot = local.hour * 3600 + local.minute * 60 + local.second
        slots_by_date.setdefault(session_date, set()).add(slot)

    if len(slots_by_date) < 2:
        return None
    counts: Dict[int, int] = {}
    for slots in slots_by_date.values():
        for slot in slots:
            counts[slot] = counts.get(slot, 0) + 1
    if not counts:
        return None
    max_count = max(counts.values())
    threshold = max(2, int(math.ceil(max_count * 0.8)))
    recurring = sorted(slot for slot, count in counts.items() if count >= threshold)
    return recurring or None


def _regular_exchange_intraday_slots(tf_secs: int, *, calendar: str) -> List[int]:
    market = market_for_exchange_calendar(calendar)
    if market is None or int(tf_secs) <= 0:
        return []
    open_hour, open_minute = market["open"]
    close_hour, close_minute = market["close"]
    session_open = int(open_hour) * 3600 + int(open_minute) * 60
    session_close = int(close_hour) * 3600 + int(close_minute) * 60
    return list(range(session_open, session_close, int(tf_secs)))


def _exchange_intraday_schedule(
    symbol: Optional[str],
    tf_secs: int,
    observed_times: Any = None,
) -> Optional[Tuple[List[int], str, str]]:
    calendar = _forecast_exchange_calendar(symbol)
    if calendar is None or int(tf_secs) <= 0 or int(tf_secs) >= 86400:
        return None
    observed_slots = _observed_intraday_session_slots(
        observed_times,
        calendar=calendar,
    )
    if observed_slots:
        return observed_slots, "observed_broker_slots", calendar
    regular_slots = _regular_exchange_intraday_slots(tf_secs, calendar=calendar)
    if regular_slots:
        return regular_slots, "exchange_regular_session_fallback", calendar
    return None


def uses_exchange_intraday_projection(
    symbol: Optional[str],
    tf_secs: int,
    *,
    observed_times: Any = None,
) -> bool:
    return _exchange_intraday_schedule(
        symbol,
        tf_secs,
        observed_times,
    ) is not None


def _next_exchange_intraday_times(
    last_epoch: float,
    horizon: int,
    *,
    slots: List[int],
    calendar: str,
) -> List[float]:
    if int(horizon) <= 0:
        return []
    market = market_for_exchange_calendar(calendar)
    if market is None:
        return []
    exchange_tz = ZoneInfo(str(market["timezone"]))
    base = float(last_epoch)
    local_base = datetime.fromtimestamp(base, tz=timezone.utc).astimezone(exchange_tz)
    session_date = local_base.date()
    out: List[float] = []
    guard = 0
    max_days = max(370, int(horizon) * 4)
    while len(out) < int(horizon) and guard < max_days:
        if (
            session_date.weekday() < 5
            and not _is_exchange_session_holiday(calendar, session_date)
        ):
            early_close = market.get("early_close")
            early_session = bool(
                early_close
                and is_early_close_session(
                    market,
                    str(market["country"]),
                    session_date,
                    holiday_resolver=_exchange_holiday,
                )
            )
            close_slot = None
            if early_session:
                close_hour, close_minute = early_close
                close_slot = int(close_hour) * 3600 + int(close_minute) * 60
            for slot in slots:
                if close_slot is not None and slot >= close_slot:
                    continue
                hour, remainder = divmod(int(slot), 3600)
                minute, second = divmod(remainder, 60)
                candidate_local = datetime(
                    session_date.year,
                    session_date.month,
                    session_date.day,
                    hour,
                    minute,
                    second,
                    tzinfo=exchange_tz,
                )
                candidate = float(candidate_local.astimezone(timezone.utc).timestamp())
                if candidate <= base + 1e-6:
                    continue
                out.append(candidate)
                if len(out) >= int(horizon):
                    return out
        session_date += timedelta(days=1)
        guard += 1
    return out


def _is_exchange_holiday_epoch(symbol: Optional[str], epoch: float) -> bool:
    calendar = _forecast_exchange_calendar(symbol)
    if calendar is None:
        return False
    # MT5 daily equity bars commonly open at the prior evening's broker
    # boundary. The following New York date is therefore the trading session.
    local_open = datetime.fromtimestamp(float(epoch), tz=timezone.utc).astimezone(
        ZoneInfo("America/New_York")
    )
    session_date = (local_open + timedelta(days=1)).date()
    return _is_exchange_session_holiday(calendar, session_date)


def describe_forecast_calendar_treatment(
    symbol: Optional[str],
    tf_secs: int,
    *,
    calendar_timeframe: bool,
    observed_times: Any = None,
) -> str:
    """Return the forecast calendar-treatment label for a symbol and step."""
    exchange_calendar = _forecast_exchange_calendar(symbol)
    if calendar_timeframe and exchange_calendar is not None:
        return (
            "broker_calendar_boundaries_and_"
            f"{exchange_calendar.lower()}_holidays_skipped"
        )
    if calendar_timeframe and uses_standard_weekend_projection(symbol, tf_secs):
        if is_probably_forex_symbol(
            symbol,
            currency_codes=FOREX_CURRENCY_CODES,
        ):
            return "broker_calendar_boundaries_and_forex_weekend_skipped"
        return "broker_calendar_boundaries_and_weekend_skipped_holidays_unknown"
    if calendar_timeframe and is_probably_crypto_symbol(symbol):
        return "broker_calendar_boundaries_continuous_crypto"
    if calendar_timeframe:
        return "calendar_estimate_session_schedule_unknown"
    exchange_schedule = _exchange_intraday_schedule(
        symbol,
        tf_secs,
        observed_times,
    )
    if exchange_schedule is not None:
        _slots, schedule_source, calendar = exchange_schedule
        return (
            f"{calendar.lower()}_{schedule_source}_holidays_and_"
            "early_closes_applied"
        )
    if uses_standard_weekend_projection(symbol, tf_secs):
        if is_probably_forex_symbol(
            symbol,
            currency_codes=FOREX_CURRENCY_CODES,
        ):
            return "forex_weekend_skipped"
        return "standard_weekend_skipped_session_hours_unknown"
    return "continuous_no_weekend_skip"


def _next_standard_weekend_open_epoch(epoch: float) -> float:
    dt_utc = datetime.fromtimestamp(float(epoch), tz=timezone.utc)
    window = standard_weekend_window(dt_utc)
    if window is None:
        return float(epoch)
    return float(window[1].timestamp())


def next_times_from_last(
    last_epoch: float,
    tf_secs: int,
    horizon: int,
    *,
    skip_weekends: bool = False,
    timeframe: Optional[str] = None,
    symbol: Optional[str] = None,
    observed_times: Any = None,
) -> List[float]:
    base = float(last_epoch)
    step = float(tf_secs)
    normalized_timeframe = str(timeframe or "").upper()
    if normalized_timeframe in CALENDAR_TIMEFRAMES:
        out: List[float] = []
        current = base
        for _ in range(int(horizon)):
            current = bar_close_epoch(current, normalized_timeframe)
            guard = 0
            while guard < 10:
                if skip_weekends and is_standard_weekend_closed_epoch(current):
                    current = _next_standard_weekend_open_epoch(current)
                    guard += 1
                    continue
                if (
                    normalized_timeframe == "D1"
                    and _is_exchange_holiday_epoch(symbol, current)
                ):
                    current = bar_close_epoch(current, normalized_timeframe)
                    guard += 1
                    continue
                break
            out.append(current)
        return out
    exchange_schedule = _exchange_intraday_schedule(
        symbol,
        tf_secs,
        observed_times,
    )
    if exchange_schedule is not None:
        slots, _schedule_source, calendar = exchange_schedule
        return _next_exchange_intraday_times(
            base,
            horizon,
            slots=slots,
            calendar=calendar,
        )
    if not skip_weekends:
        return [base + step * (i + 1) for i in range(int(horizon))]
    out: List[float] = []
    current = base
    for _ in range(int(horizon)):
        current += step
        guard = 0
        while is_standard_weekend_closed_epoch(current) and guard < 8:
            next_open = _next_standard_weekend_open_epoch(current)
            if next_open <= current:
                break
            current = next_open
            guard += 1
        out.append(current)
    return out


def pd_freq_from_timeframe(tf: str) -> str:
    t = str(tf).upper()
    mapping = {
        'M1': '1min', 'M2': '2min', 'M3': '3min', 'M4': '4min', 'M5': '5min',
        'M10': '10min', 'M12': '12min', 'M15': '15min', 'M20': '20min', 'M30': '30min',
        'H1': '1h', 'H2': '2h', 'H3': '3h', 'H4': '4h', 'H6': '6h', 'H8': '8h', 'H12': '12h',
        'D1': '1d', 'W1': '1w', 'MN1': 'MS'
    }
    return mapping.get(t, 'D')


# ------------------------------------------------------------------
# Composable NeuralForecast building blocks (used by train/predict)
# ------------------------------------------------------------------

def _nf_resolve_accelerator() -> str:
    """Return 'cpu' or 'gpu' based on torch availability and env."""
    accel = 'cpu'
    try:
        import torch as _torch
        accel_env = os.environ.get('MTDATA_NF_ACCEL')
        if isinstance(accel_env, str):
            accel = 'gpu' if accel_env.strip().lower() == 'gpu' else 'cpu'
        else:
            accel = 'gpu' if hasattr(_torch, 'cuda') and _torch.cuda.is_available() else 'cpu'
        try:
            if accel == 'gpu' and hasattr(_torch, 'set_float32_matmul_precision'):
                _torch.set_float32_matmul_precision('high')
        except Exception:
            pass
    except Exception:
        accel = 'cpu'
    return accel


def nf_build_model_kwargs(
    *,
    model_class,
    fh: int,
    input_size: int,
    batch_size: int,
    steps: int,
    learning_rate: Optional[float] = None,
    accel: Optional[str] = None,
    enable_progress_bar: bool = False,
    early_stop_patience_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Build keyword arguments for a NeuralForecast model constructor.

    Does NOT instantiate the model — returns the kwargs dict so that
    callers can inspect or modify them before construction.
    """
    import inspect as _inspect

    if accel is None:
        accel = _nf_resolve_accelerator()

    try:
        ctor_params = _inspect.signature(model_class.__init__).parameters
    except Exception:
        ctor_params = {}

    model_kwargs: Dict[str, Any] = {
        'h': int(fh),
        'input_size': int(input_size),
        'batch_size': int(batch_size),
    }
    if 'max_steps' in ctor_params:
        model_kwargs['max_steps'] = int(steps)
    elif 'max_epochs' in ctor_params:
        model_kwargs['max_epochs'] = int(steps)
    else:
        model_kwargs['max_steps'] = int(steps)
    if learning_rate is not None:
        try:
            model_kwargs['learning_rate'] = float(learning_rate)
        except Exception:
            pass
    if early_stop_patience_steps is not None and "early_stop_patience_steps" in ctor_params:
        model_kwargs["early_stop_patience_steps"] = int(early_stop_patience_steps)

    base_trainer: Dict[str, Any] = {
        'accelerator': accel,
        'devices': 1,
        'num_nodes': 1,
    }
    quiet_opts: Dict[str, Any] = {
        'logger': False,
        'enable_progress_bar': enable_progress_bar,
        'enable_checkpointing': False,
        'enable_model_summary': False,
        'log_every_n_steps': 0,
    }
    for _opt, _val in quiet_opts.items():
        base_trainer.setdefault(_opt, _val)
    for _opt, _val in base_trainer.items():
        model_kwargs.setdefault(_opt, _val)

    return model_kwargs


_NF_ENV_VARS_TO_CLEAR = (
    'KUBERNETES_SERVICE_HOST', 'KUBERNETES_SERVICE_PORT',
    'GROUP_RANK', 'NODE_RANK', 'LOCAL_RANK', 'RANK', 'WORLD_SIZE',
    'GLOBAL_RANK', 'MASTER_ADDR', 'MASTER_PORT',
    'LT_CLOUD_PROVIDER', 'LT_CLUSTER', 'TORCHELASTIC_RUN_ID',
    'ETCD_HOST', 'ETCD_PORT',
)
_NF_MANAGED_ENV_VARS = _NF_ENV_VARS_TO_CLEAR + (
    'PL_TORCH_DISTRIBUTED_BACKEND',
    'LT_DISABLE_DISTRIBUTED',
    'CUDA_VISIBLE_DEVICES',
)


class _NfEnvGuard:
    """Context manager that sanitizes env vars for single-device NF training."""

    def __init__(self, accel: str = 'cpu') -> None:
        self._accel = accel
        self._snapshot: Dict[str, Optional[str]] = {}
        self._missing: set[str] = set()

    def __enter__(self) -> '_NfEnvGuard':
        for key in _NF_MANAGED_ENV_VARS:
            if key in os.environ:
                self._snapshot[key] = os.environ.get(key)
            else:
                self._missing.add(key)
        for _var in _NF_ENV_VARS_TO_CLEAR:
            os.environ.pop(_var, None)
        os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
        os.environ['LT_DISABLE_DISTRIBUTED'] = '1'
        if self._accel == 'gpu':
            try:
                cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                if ',' in cvd:
                    os.environ['CUDA_VISIBLE_DEVICES'] = cvd.split(',')[0].strip()
                elif cvd.strip() == '':
                    import torch as _torch
                    if _torch.cuda.device_count() > 1:
                        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            except Exception:
                pass
        return self

    def __exit__(self, *exc_info: Any) -> None:
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                _dist.destroy_process_group()
        except Exception:
            pass
        try:
            import torch as _torch
            if hasattr(_torch, 'cuda') and _torch.cuda.is_available():
                _torch.cuda.synchronize()
                _torch.cuda.empty_cache()
        except Exception:
            pass
        for key in _NF_MANAGED_ENV_VARS:
            if key in self._missing:
                os.environ.pop(key, None)
                continue
            restored = self._snapshot.get(key)
            if restored is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = restored


def _nf_build_trainer_kwargs(accel: str) -> Dict[str, Any]:
    """Build trainer kwargs for the NeuralForecast constructor."""
    import inspect as _inspect

    base_trainer: Dict[str, Any] = {
        'accelerator': accel,
        'devices': 1,
        'num_nodes': 1,
    }
    cand_opts: Dict[str, Any] = {
        'logger': False,
        'enable_progress_bar': False,
        'enable_checkpointing': False,
        'log_every_n_steps': 0,
    }
    try:
        try:
            import lightning.pytorch as _L
            _Trainer = _L.Trainer
        except Exception:
            import pytorch_lightning as _pl
            _Trainer = _pl.Trainer
        _tparams = _inspect.signature(_Trainer.__init__).parameters
        nf_trainer = dict(base_trainer)
        for k, v in cand_opts.items():
            if k in _tparams and k not in nf_trainer:
                nf_trainer[k] = v
        return nf_trainer
    except Exception:
        return {**base_trainer, **cand_opts}


def nf_create_and_fit(
    *,
    model_class,
    model_kwargs: Dict[str, Any],
    timeframe: str,
    Y_df: pd.DataFrame,
    exog_used: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
    future_times: Optional[List[float]] = None,
    val_size: int = 0,
) -> Any:
    """Instantiate a NeuralForecast wrapper, fit it, and return the fitted NF object.

    Must be called inside ``_NF_ENV_LOCK`` and ``_NfEnvGuard``.
    """
    import inspect as _inspect
    import warnings

    try:
        from neuralforecast import NeuralForecast as _NeuralForecast
    except Exception as ex:
        raise RuntimeError(f"Failed to import neuralforecast: {ex}")

    accel = str(model_kwargs.get('accelerator', 'cpu'))
    nf_kwargs: Dict[str, Any] = {
        'models': [model_class(**model_kwargs)],
        'freq': pd_freq_from_timeframe(timeframe),
    }
    try:
        _nf_init_params = _inspect.signature(_NeuralForecast.__init__).parameters
    except Exception:
        _nf_init_params = {}
    if 'trainer_kwargs' in _nf_init_params:
        nf_trainer = _nf_build_trainer_kwargs(accel)
        try:
            try:
                import lightning.pytorch as _L
                _Trainer = _L.Trainer
            except Exception:
                import pytorch_lightning as _pl
                _Trainer = _pl.Trainer
            trainer_obj = _Trainer(**nf_trainer)
            nf_kwargs['trainer'] = trainer_obj
        except Exception:
            nf_kwargs['trainer_kwargs'] = nf_trainer
    if 'num_workers_loader' in _nf_init_params:
        nf_kwargs['num_workers_loader'] = 0

    nf = _NeuralForecast(**nf_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _fit_params = _inspect.signature(nf.fit).parameters
        except Exception:
            _fit_params = {}
        supports_x = 'X_df' in _fit_params
        fit_kwargs: Dict[str, Any] = {"df": Y_df, "verbose": False}
        if "val_size" in _fit_params and int(val_size) > 0:
            fit_kwargs["val_size"] = int(val_size)

        exog_used_2d = _as_2d_exog_array(exog_used, name="exog_used")
        if exog_used_2d is not None and supports_x:
            X_df = pd.DataFrame({'unique_id': ['ts'] * len(Y_df), 'ds': Y_df['ds'].values})
            cols = [f'x{i}' for i in range(exog_used_2d.shape[1])]
            for j, cname in enumerate(cols):
                X_df[cname] = exog_used_2d[:, j]
            nf.fit(X_df=X_df, **fit_kwargs)
        else:
            nf.fit(**fit_kwargs)

    return nf


def nf_predict_from_fitted(
    nf: Any,
    *,
    fh: int,
    exog_future: Optional[np.ndarray] = None,
    future_times: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Run predictions on an already-fitted NeuralForecast object.

    Must be called inside ``_NF_ENV_LOCK`` and ``_NfEnvGuard``.
    """
    import inspect as _inspect
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _pred_params = _inspect.signature(nf.predict).parameters
        except Exception:
            _pred_params = {}
        try:
            _fit_params = _inspect.signature(nf.fit).parameters
        except Exception:
            _fit_params = {}
        supports_x_predict = 'X_df' in _pred_params

        exog_future_2d = _as_2d_exog_array(exog_future, name="exog_future")
        if exog_future_2d is not None and future_times is not None and supports_x_predict:
            ds_f = pd.to_datetime(pd.Series(future_times), unit='s', utc=True)
            cols = [f'x{i}' for i in range(exog_future_2d.shape[1])]
            Xf_df = pd.DataFrame({'unique_id': ['ts'] * len(ds_f), 'ds': pd.Index(ds_f).to_pydatetime()})
            for j, cname in enumerate(cols):
                Xf_df[cname] = exog_future_2d[:, j]
            if 'h' in _pred_params:
                return nf.predict(h=int(fh), X_df=Xf_df)
            else:
                return nf.predict(X_df=Xf_df)
        else:
            if 'h' in _pred_params:
                return nf.predict(h=int(fh))
            else:
                return nf.predict()


def resolve_forecast_symbol(symbol: str) -> Tuple[str, Optional[str]]:
    """Map accepted FX aliases such as EUR/USD to the canonical broker symbol.

    History fetch already resolves aliases internally. Metadata lookups
    (symbol_info, ticks, digits, currency, price_basis) do not, so callers must
    switch to the canonical name before those reads. Returns
    ``(canonical, requested_alias_or_none)``.
    """
    from ..utils.mt5 import resolve_public_symbol

    return resolve_public_symbol(symbol)


def fetch_history(
    symbol: str,
    timeframe: str,
    need: int,
    as_of: Optional[str] = None,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    drop_last_live: bool = True,
) -> pd.DataFrame:
    """Fetch analysis history through the canonical market-data gateway."""
    return fetch_history_frame(
        symbol,
        timeframe,
        need,
        as_of,
        start=start,
        end=end,
        include_incomplete=not drop_last_live,
    )


def _parse_as_of_bound(
    value: Optional[str],
    *,
    timeframe: Optional[str] = None,
) -> Optional[datetime]:
    """Parse an as-of cutoff, treating a date label as inclusive through that day."""
    if not value:
        return None
    if timeframe:
        calendar_bound = _calendar_bound_or_raise(
            value,
            timeframe=timeframe,
            end_bound=True,
        )
        if calendar_bound is not None:
            return calendar_bound
    return _parse_end_datetime(value)


def future_as_of_error(
    as_of: Optional[str],
    *,
    now_epoch: Optional[float] = None,
) -> Optional[str]:
    """Validate that a point-in-time cutoff is not in the future."""
    if not as_of:
        return None
    parsed = _parse_start_datetime(as_of)
    if parsed is None:
        return "Invalid as_of time."
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    current_epoch = (
        datetime.now(timezone.utc).timestamp()
        if now_epoch is None
        else float(now_epoch)
    )
    if parsed.timestamp() > current_epoch + 1.0:
        return "as_of must not be in the future."
    return None
