"""Time-series diagnostic MCP tools."""

from __future__ import annotations

import logging
import math
import warnings
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import Field
from scipy.signal import find_peaks, periodogram

from ..forecast.common import annualization_context
from ..shared.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..shared.schema import DetailLiteral, TimeframeLiteral
from ..shared.symbols import is_probably_crypto_symbol, is_probably_forex_symbol
from ..utils.mt5 import (
    _ensure_symbol_ready,
    _mt5_copy_rates_from,
    ensure_mt5_connection_or_raise,
    mt5,
    resolve_public_symbol,
    symbol_price_digits_optional,
)
from ..utils.time import bar_close_epoch, format_datetime_utc
from ..utils.utils import _parse_end_datetime
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .mt5_gateway import create_mt5_gateway
from .output_contract import normalize_output_verbosity_detail
from .runtime_metadata import run_mt5_logged_operation

logger = logging.getLogger(__name__)
_CALENDAR_ALIAS_RELATIVE_TOLERANCE = 0.01


def _fetch_diagnostic_bars(
    symbol: str,
    timeframe: str,
    lookback: int,
    *,
    include_incomplete: bool = False,
    as_of: Optional[str] = None,
    operation: str = "diagnostic_analysis",
) -> tuple[pd.DataFrame, str | Dict[str, Any] | None]:
    tf = TIMEFRAME_MAP.get(str(timeframe or "").strip().upper())
    if tf is None:
        return pd.DataFrame(), f"Invalid timeframe '{timeframe}'."
    anchor = _parse_end_datetime(as_of) if as_of else datetime.now(timezone.utc)
    if anchor is None:
        return pd.DataFrame(), "as_of must be a valid date or ISO 8601 timestamp."
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    else:
        anchor = anchor.astimezone(timezone.utc)
    supported_boundary = datetime(1970, 1, 1, tzinfo=timezone.utc)
    if anchor < supported_boundary:
        return pd.DataFrame(), build_error_payload(
            (
                f"as_of {as_of!r} is before MT5's supported history boundary "
                "(1970-01-01T00:00:00Z)."
            ),
            code="diagnostic_unsupported_date_range",
            operation=operation,
            remediation=(
                "Use an as_of date or timestamp on or after "
                "1970-01-01T00:00:00Z."
            ),
            details={
                "as_of": as_of,
                "supported_start": "1970-01-01T00:00:00Z",
            },
        )
    if anchor > datetime.now(timezone.utc):
        return pd.DataFrame(), "as_of cannot be in the future."
    symbol, symbol_input = resolve_public_symbol(symbol)
    symbol_error = _ensure_symbol_ready(symbol)
    if symbol_error:
        return pd.DataFrame(), symbol_error
    rates = _mt5_copy_rates_from(
        symbol,
        tf,
        anchor,
        max(2, int(lookback)) + (0 if include_incomplete else 1),
    )
    if rates is None or len(rates) == 0:
        return pd.DataFrame(), f"Failed to fetch data for {symbol}."
    try:
        frame = pd.DataFrame(rates)
    except Exception:
        frame = pd.DataFrame(list(rates))
    required = {"time", "close"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(), "Fetched bars do not contain time and close fields."
    frame = frame.sort_values("time").drop_duplicates("time", keep="last")
    open_epochs = pd.to_numeric(frame["time"], errors="coerce")
    close_epochs = open_epochs.map(
        lambda value: (
            bar_close_epoch(float(value), str(timeframe))
            if pd.notna(value)
            else float("nan")
        )
    )
    opened = open_epochs <= anchor.timestamp()
    forming = bool((opened & (close_epochs > anchor.timestamp())).any())
    forming_status = "none_detected"
    if forming:
        forming_status = "included" if include_incomplete else "excluded"
        if include_incomplete and as_of and bool(
            (opened & (close_epochs > anchor.timestamp())
             & (close_epochs <= datetime.now(timezone.utc).timestamp())).any()
        ):
            return pd.DataFrame(), build_error_payload(
                "Historical partial-candle values cannot be recovered from completed bars.",
                code="historical_partial_candle_unavailable",
                operation=operation,
                remediation="Use include_incomplete=false for historical analysis.",
            )
    frame = frame[opened if include_incomplete else close_epochs <= anchor.timestamp()]
    frame = frame.tail(max(2, int(lookback))).reset_index(drop=True)
    frame.attrs["history_policy"] = (
        "includes_current_forming_bar" if include_incomplete else "completed_bars_only"
    )
    frame.attrs["forming_candle_status"] = forming_status
    frame.attrs["requested_as_of"] = as_of
    frame.attrs["resolved_as_of"] = format_datetime_utc(anchor)
    frame.attrs["symbol"] = symbol
    if symbol_input is not None:
        frame.attrs["symbol_input"] = symbol_input
    return frame, None


def _diagnostic_history_metadata(
    frame: pd.DataFrame,
    *,
    include_incomplete: bool,
) -> Dict[str, Any]:
    times = (
        pd.to_numeric(frame["time"], errors="coerce").dropna()
        if "time" in frame.columns
        else pd.Series(dtype=float)
    )
    period_start = (
        format_datetime_utc(datetime.fromtimestamp(float(times.iloc[0]), tz=timezone.utc))
        if len(times)
        else None
    )
    period_end = (
        format_datetime_utc(datetime.fromtimestamp(float(times.iloc[-1]), tz=timezone.utc))
        if len(times)
        else None
    )
    return {
        **{key: frame.attrs[key] for key in ("symbol", "symbol_input") if key in frame.attrs},
        "history_policy": frame.attrs.get(
            "history_policy",
            "includes_current_forming_bar"
            if include_incomplete
            else "completed_bars_only",
        ),
        "forming_candle_status": frame.attrs.get(
            "forming_candle_status", "not_reported"
        ),
        "analysis_window": {
            "requested_as_of": frame.attrs.get("requested_as_of"),
            "resolved_as_of": frame.attrs.get("resolved_as_of"),
            "period_start": period_start,
            "period_end": period_end,
            "timezone": "UTC",
            "bar_timestamp_basis": "open_time",
            "bars_used": int(len(frame)),
        },
    }


def _diagnostic_series(frame: pd.DataFrame, target: str) -> pd.Series:
    target_value = str(target or "close").strip().lower()
    close = pd.to_numeric(frame["close"], errors="coerce")
    if target_value == "close":
        values = close
    elif target_value == "log_price":
        values = np.log(close.where(close > 0))
    elif target_value == "return":
        values = close.pct_change(fill_method=None)
    elif target_value == "log_return":
        values = np.log(close.where(close > 0)).diff()
    elif target_value == "diff":
        values = close.diff()
    else:
        raise ValueError("target must be one of: close, log_price, return, log_return, diff.")
    return pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()


_SEASONALITY_QUALITY_THRESHOLDS = {
    "very_weak": "score < 0.05",
    "weak": "0.05 <= score < 0.10",
    "moderate": "0.10 <= score < 0.25",
    "strong": "score >= 0.25",
}
_SEASONALITY_SMALL_SAMPLE = 100


def _seasonality_normalized_score(
    acf: float,
    spectral_strength: float,
    *,
    samples: int,
    positive_frequency_bins: int,
) -> float:
    """Composite seasonality score with sample-size-aware components."""
    n = max(1, int(samples))
    acf_se = 1.0 / math.sqrt(n)
    acf_excess = max(0.0, float(acf) - acf_se)
    acf_strength = min(1.0, acf_excess / max(1e-12, 1.0 - acf_se))
    bins = max(1, int(positive_frequency_bins))
    expected_share = 1.0 / float(bins)
    spectral_excess = max(0.0, float(spectral_strength) - expected_share)
    spectral_strength_adj = min(
        1.0,
        spectral_excess / max(1e-12, 1.0 - expected_share),
    )
    return 0.55 * acf_strength + 0.45 * spectral_strength_adj


def _seasonality_signal_quality(
    score: float,
    acf: float = 0.0,
    spectral_strength: float = 0.0,
    *,
    samples: int = 0,
) -> str:
    del spectral_strength
    signal = float(score)
    if signal < 0.05:
        label = "very_weak"
    elif signal < 0.10:
        label = "weak"
    elif signal < 0.25:
        label = "moderate"
    else:
        label = "strong"
    n = int(samples or 0)
    if n > 0 and n < _SEASONALITY_SMALL_SAMPLE and label in {"moderate", "strong"}:
        acf_critical = 1.96 / math.sqrt(n)
        if float(acf) <= acf_critical:
            return "weak"
    return label


def _format_period_duration(duration_seconds: float) -> str:
    if duration_seconds % 86_400 == 0:
        return f"{duration_seconds / 86_400:g} days"
    if duration_seconds >= 86_400:
        return f"{duration_seconds / 86_400:.2f} days"
    if duration_seconds % 3_600 == 0:
        return f"{duration_seconds / 3_600:g} hours"
    if duration_seconds >= 3_600:
        return f"{duration_seconds / 3_600:.2f} hours"
    return f"{duration_seconds / 60:g} minutes"


def _seasonality_period_context(
    period_bars: int,
    timeframe: str,
    *,
    observed_times: Any = None,
) -> Dict[str, Any]:
    seconds_per_bar = int(TIMEFRAME_SECONDS.get(str(timeframe).upper(), 0) or 0)
    nominal_seconds = max(0, int(period_bars) * seconds_per_bar)
    if nominal_seconds <= 0:
        return {}
    observed_durations = np.asarray([], dtype=float)
    try:
        epochs = pd.to_numeric(pd.Series(observed_times), errors="coerce").to_numpy(
            dtype=float
        )
        epochs = epochs[np.isfinite(epochs)]
        if len(epochs) > int(period_bars):
            observed_durations = epochs[int(period_bars) :] - epochs[: -int(period_bars)]
            observed_durations = observed_durations[
                np.isfinite(observed_durations) & (observed_durations > 0)
            ]
    except Exception:
        observed_durations = np.asarray([], dtype=float)

    if observed_durations.size:
        duration_seconds = float(np.median(observed_durations))
        basis = "median_observed_timestamp_lag"
    else:
        duration_seconds = float(nominal_seconds)
        basis = "nominal_timeframe_seconds"
    duration_seconds_out: int | float = (
        int(round(duration_seconds))
        if math.isclose(duration_seconds, round(duration_seconds), abs_tol=1e-9)
        else round(duration_seconds, 3)
    )
    aliases = {
        86_400: "calendar_day",
        604_800: "calendar_week",
        2_592_000: "30_day_month",
    }
    nearest = min(aliases, key=lambda value: abs(value - duration_seconds))
    out: Dict[str, Any] = {
        "period_duration": _format_period_duration(duration_seconds),
        "period_duration_seconds": duration_seconds_out,
        "period_duration_basis": basis,
        "nominal_period_duration": _format_period_duration(float(nominal_seconds)),
        "nominal_period_duration_seconds": nominal_seconds,
    }
    if observed_durations.size:
        minimum = float(np.min(observed_durations))
        maximum = float(np.max(observed_durations))
        out["period_duration_observed_range"] = {
            "min_seconds": int(round(minimum)),
            "max_seconds": int(round(maximum)),
            "min": _format_period_duration(minimum),
            "max": _format_period_duration(maximum),
            "pairs": int(observed_durations.size),
        }
    if (
        abs(nearest - duration_seconds) / nearest
        <= _CALENDAR_ALIAS_RELATIVE_TOLERANCE
    ):
        out["calendar_alias"] = aliases[nearest]
    return out


def _critical_values(values: Any) -> Dict[str, float]:
    if not isinstance(values, dict):
        return {}
    return {
        str(key): round(float(value), 6)
        for key, value in values.items()
        if value is not None and math.isfinite(float(value))
    }


def _clean_stationarity_warning(text: Any) -> str:
    """Translate raw statsmodels/scipy stationarity warnings into plain guidance."""
    raw = str(getattr(text, "message", text)).strip()
    low = raw.lower()
    if "p-value" in low and ("look-up table" in low or "lookup table" in low or "outside of the range" in low):
        if "smaller" in low:
            direction = "smaller than the reported value"
        elif "greater" in low or "larger" in low:
            direction = "greater than the reported value"
        else:
            direction = "outside the reported range"
        return (
            "KPSS p-value is approximate: the test statistic falls outside the "
            f"lookup table, so the actual p-value is {direction}."
        )
    return raw


def _kpss_critical_key(alpha: float) -> Optional[str]:
    """Map a significance level to the Statsmodels KPSS critical-value label."""
    mapping = ((0.01, "1%"), (0.025, "2.5%"), (0.05, "5%"), (0.1, "10%"))
    for level, label in mapping:
        if abs(float(alpha) - level) < 1e-12:
            return label
    return None


def _kpss_is_stationary(
    *,
    p_value: float,
    statistic: float,
    critical_values: Any,
    alpha: float,
    bound_warning: Optional[str] = None,
) -> bool:
    """Decide KPSS stationarity. H0 is stationarity; reject at p <= alpha.

    Statsmodels censors lookup-table p-values at 0.01/0.10, so a displayed
    p-value equal to alpha can hide a stronger rejection. Prefer the matching
    critical value, then honor bound-direction warnings, then use p > alpha.
    """
    crit_key = _kpss_critical_key(alpha)
    if crit_key and isinstance(critical_values, dict) and crit_key in critical_values:
        try:
            return float(statistic) <= float(critical_values[crit_key])
        except (TypeError, ValueError):
            pass
    warning_low = str(bound_warning or "").lower()
    if "smaller than the reported" in warning_low and float(p_value) <= float(alpha):
        return False
    if "greater than the reported" in warning_low and float(p_value) >= float(alpha):
        return True
    return float(p_value) > float(alpha)


@mcp.tool()
def stationarity_test(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    lookback: Annotated[int, Field(ge=20)] = 500,
    target: Literal["close", "log_price", "return", "log_return", "diff"] = "log_return",
    tests: str = "adf,kpss,pp",
    trend: Literal["c", "ct"] = "c",
    significance: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05,
    include_incomplete: bool = False,
    as_of: Optional[str] = None,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Test an MT5 time series for stationarity using ADF, KPSS, and optional PP."""

    def _run() -> Dict[str, Any]:
        minimum_lookback = (
            21 if target in {"return", "log_return", "diff"} else 20
        )
        if int(lookback) < minimum_lookback:
            return {
                "error": (
                    f"lookback must be at least {minimum_lookback} for "
                    f"target={target}; this leaves 20 observations after preprocessing."
                )
            }
        if not 0.0 < float(significance) < 1.0:
            return build_error_payload(
                "significance must be strictly between 0 and 1.",
                code="invalid_parameter",
                operation="stationarity_test",
                details={"parameter": "significance", "received": significance},
                remediation="Set significance to a decimal such as 0.05.",
                valid_values={"significance": "0 < value < 1"},
                example="--significance 0.05",
            )
        requested = [part.strip().lower() for part in str(tests or "").split(",") if part.strip()]
        requested = list(dict.fromkeys(requested))
        invalid = [name for name in requested if name not in {"adf", "kpss", "pp"}]
        if not requested or invalid:
            return {"error": "tests must contain one or more of: adf, kpss, pp."}
        detail_mode = normalize_output_verbosity_detail(detail, default="compact")
        gateway = create_mt5_gateway(adapter=mt5, ensure_connection_impl=ensure_mt5_connection_or_raise)
        gateway.ensure_connection()
        frame, fetch_error = _fetch_diagnostic_bars(
            symbol,
            timeframe,
            int(lookback),
            include_incomplete=include_incomplete,
            as_of=as_of,
            operation="stationarity_test",
        )
        if fetch_error:
            return fetch_error if isinstance(fetch_error, dict) else {"error": fetch_error}
        try:
            series = _diagnostic_series(frame, target)
        except ValueError as exc:
            return {"error": str(exc)}
        if len(series) < 20 or float(series.std(ddof=0)) <= 1e-15:
            return {"error": "At least 20 non-constant finite observations are required."}

        rows: List[Dict[str, Any]] = []
        warnings_out: List[str] = []
        alpha = float(significance)
        if "adf" in requested:
            from statsmodels.tsa.stattools import adfuller

            regression = "ct" if trend == "ct" else "c"
            result = adfuller(series.to_numpy(), regression=regression, autolag="AIC")
            adf_samples = int(result[3])
            adf_sufficient = adf_samples >= 20
            rows.append(
                {
                    "test": "adf",
                    "statistic": round(float(result[0]), 6),
                    "p_value": float(result[1]),
                    "lags": int(result[2]),
                    "samples": adf_samples,
                    "stationary": (
                        bool(float(result[1]) < alpha) if adf_sufficient else None
                    ),
                    "status": "ok" if adf_sufficient else "insufficient_sample",
                    "null_hypothesis": "unit_root",
                    **({"critical_values": _critical_values(result[4])} if detail_mode == "full" else {}),
                }
            )
            if not adf_sufficient:
                warnings_out.append(
                    "ADF was excluded from the combined conclusion because lag "
                    f"selection left only {adf_samples} effective observations; "
                    "at least 20 are required."
                )
        if "kpss" in requested:
            from statsmodels.tsa.stattools import kpss

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = kpss(series.to_numpy(), regression=trend, nlags="auto")
            kpss_warnings = [_clean_stationarity_warning(item) for item in caught]
            critical_values = _critical_values(result[3])
            rows.append(
                {
                    "test": "kpss",
                    "statistic": round(float(result[0]), 6),
                    "p_value": float(result[1]),
                    "lags": int(result[2]),
                    "samples": int(len(series)),
                    "stationary": _kpss_is_stationary(
                        p_value=float(result[1]),
                        statistic=float(result[0]),
                        critical_values=critical_values,
                        alpha=alpha,
                        bound_warning=" ".join(kpss_warnings),
                    ),
                    "status": "ok",
                    "null_hypothesis": "stationary",
                    **({"critical_values": critical_values} if detail_mode == "full" else {}),
                }
            )
            warnings_out.extend(kpss_warnings)
        if "pp" in requested:
            try:
                from arch.unitroot import PhillipsPerron
            except ImportError:
                warnings_out.append("Phillips-Perron skipped because optional package 'arch' is not installed.")
            else:
                result = PhillipsPerron(series.to_numpy(), trend=trend)
                pp_samples = int(result.nobs)
                pp_sufficient = pp_samples >= 20
                rows.append(
                    {
                        "test": "pp",
                        "statistic": round(float(result.stat), 6),
                        "p_value": float(result.pvalue),
                        "lags": int(result.lags),
                        "samples": pp_samples,
                        "stationary": (
                            bool(float(result.pvalue) < alpha) if pp_sufficient else None
                        ),
                        "status": "ok" if pp_sufficient else "insufficient_sample",
                        "null_hypothesis": "unit_root",
                        **({"critical_values": _critical_values(result.critical_values)} if detail_mode == "full" else {}),
                    }
                )
                if not pp_sufficient:
                    warnings_out.append(
                        "Phillips-Perron was excluded from the combined conclusion because "
                        f"only {pp_samples} effective observations were available; "
                        "at least 20 are required."
                    )

        votes = [
            bool(row["stationary"])
            for row in rows
            if row.get("stationary") is not None
        ]
        stationary_votes = int(sum(votes))
        conclusion = (
            "inconclusive"
            if not votes
            else "stationary"
            if stationary_votes == len(votes)
            else "non_stationary"
            if stationary_votes == 0
            else "mixed"
        )
        out: Dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "target": target,
            "significance": alpha,
            "conclusion": conclusion,
            "stationary_votes": stationary_votes,
            "tests_completed": len(rows),
            "items": rows,
            "samples": int(len(series)),
            **_diagnostic_history_metadata(
                frame, include_incomplete=include_incomplete
            ),
        }
        if warnings_out:
            out["warnings"] = list(dict.fromkeys(warnings_out))
        if detail_mode == "full":
            out["interpretation"] = (
                "ADF and PP reject a unit-root null when stationary; KPSS fails to reject a stationarity null when stationary."
            )
        return out

    return run_mt5_logged_operation(
        logger,
        operation="stationarity_test",
        symbol=symbol,
        timeframe=timeframe,
        lookback=lookback,
        target=target,
        func=_run,
    )


@mcp.tool()
def seasonality_detect(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    lookback: Annotated[int, Field(ge=31)] = 1000,
    target: Literal["close", "log_price", "return", "log_return", "diff"] = "log_return",
    min_period: Annotated[int, Field(ge=2)] = 2,
    max_period: Annotated[Optional[int], Field(ge=2)] = None,
    min_cycles: Annotated[int, Field(ge=2)] = 3,
    top_n: Annotated[int, Field(ge=1)] = 5,
    include_incomplete: bool = False,
    as_of: Optional[str] = None,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Detect dominant seasonal periods using autocorrelation and spectral power."""

    def _run() -> Dict[str, Any]:
        if int(lookback) < 31:
            return {
                "error": (
                    "lookback must be at least 31; seasonality preprocessing "
                    "requires 30 analyzed observations."
                )
            }
        if int(min_period) < 2 or int(min_cycles) < 2 or int(top_n) < 1:
            return {"error": "min_period >= 2, min_cycles >= 2, and top_n >= 1 are required."}
        gateway = create_mt5_gateway(adapter=mt5, ensure_connection_impl=ensure_mt5_connection_or_raise)
        gateway.ensure_connection()
        frame, fetch_error = _fetch_diagnostic_bars(
            symbol,
            timeframe,
            int(lookback),
            include_incomplete=include_incomplete,
            as_of=as_of,
            operation="seasonality_detect",
        )
        if fetch_error:
            return fetch_error if isinstance(fetch_error, dict) else {"error": fetch_error}
        try:
            series = _diagnostic_series(frame, target)
        except ValueError as exc:
            return {"error": str(exc)}
        preprocessing = "none"
        analyzed_target = target
        if target in {"close", "log_price"}:
            series = series.diff().dropna()
            preprocessing = "first_difference_for_stationarity"
            analyzed_target = "diff" if target == "close" else "log_return"
        n = int(len(series))
        upper = min(int(max_period) if max_period is not None else max(2, n // int(min_cycles)), n // int(min_cycles))
        if upper < int(min_period) or n < 30:
            return {"error": "Insufficient samples for the requested period and cycle constraints."}
        values = series.to_numpy(dtype=float)
        centered = values - float(np.mean(values))
        variance = float(np.dot(centered, centered))
        if variance <= 1e-15:
            return {"error": "Cannot detect seasonality in a constant series."}
        periods = np.arange(int(min_period), upper + 1, dtype=int)
        acf_scores = np.asarray(
            [float(np.dot(centered[lag:], centered[:-lag]) / variance) for lag in periods],
            dtype=float,
        )
        frequencies, powers = periodogram(centered, detrend="linear", scaling="spectrum")
        spectral_by_period: Dict[int, float] = {}
        positive = frequencies > 0
        for frequency, power in zip(
            frequencies[positive],
            powers[positive],
            strict=False,
        ):
            period = int(round(1.0 / float(frequency)))
            if int(min_period) <= period <= upper:
                spectral_by_period[period] = max(spectral_by_period.get(period, 0.0), float(power))
        total_spectral_power = float(np.sum(powers[positive]))
        positive_bin_count = int(np.count_nonzero(positive))
        positive_acf = np.maximum(acf_scores, 0.0)
        peak_idx, _ = find_peaks(positive_acf)
        candidates = set(int(periods[index]) for index in peak_idx)
        candidates.update(sorted(spectral_by_period, key=spectral_by_period.get, reverse=True)[: max(int(top_n) * 3, 5)])
        if not candidates:
            candidates.update(int(value) for value in periods[np.argsort(positive_acf)[-max(int(top_n), 1) :]])
        rows: List[Dict[str, Any]] = []
        for period in candidates:
            acf_value = float(acf_scores[period - int(min_period)])
            spectral_strength = (
                float(spectral_by_period.get(period, 0.0) / total_spectral_power)
                if total_spectral_power > 0
                else 0.0
            )
            score = _seasonality_normalized_score(
                acf_value,
                spectral_strength,
                samples=n,
                positive_frequency_bins=positive_bin_count,
            )
            score_rounded = round(score, 6)
            acf_rounded = round(acf_value, 6)
            spectral_rounded = round(spectral_strength, 6)
            row: Dict[str, Any] = {
                "period_bars": int(period),
                **_seasonality_period_context(
                    int(period),
                    timeframe,
                    observed_times=frame.get("time"),
                ),
                "score": score_rounded,
                "acf": acf_rounded,
                "spectral_strength": spectral_rounded,
                "quality_statistic": score_rounded,
                "signal_quality": _seasonality_signal_quality(
                    score_rounded,
                    acf_rounded,
                    spectral_rounded,
                    samples=n,
                ),
                "cycles_observed": round(n / float(period), 2),
            }
            if spectral_rounded == 0.0:
                row["spectral_strength_note"] = (
                    "no_periodogram_power_at_this_rounded_period"
                )
            rows.append(
                row
            )
        rows.sort(key=lambda row: (-float(row["score"]), int(row["period_bars"])))
        rows = rows[: int(top_n)]
        out: Dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "target": target,
            "analyzed_target": analyzed_target,
            "preprocessing": preprocessing,
            "samples": n,
            "search_range_bars": {"min": int(min_period), "max": upper},
            "items": rows,
            "count": len(rows),
            "dominant_period_bars": rows[0]["period_bars"] if rows else None,
            "score_formula": (
                "0.55*max(0, acf - 1/sqrt(n))/(1 - 1/sqrt(n)) + "
                "0.45*max(0, spectral_share - 1/bins)/(1 - 1/bins); "
                "range 0-1, higher = stronger seasonality"
            ),
            "quality_statistic": "score",
            "quality_formula": (
                "0.55*max(0, acf - 1/sqrt(n))/(1 - 1/sqrt(n)) + "
                "0.45*max(0, spectral_share - 1/bins)/(1 - 1/bins); "
                "range 0-1, higher = stronger seasonality"
            ),
            "quality_thresholds": dict(_SEASONALITY_QUALITY_THRESHOLDS),
            **_diagnostic_history_metadata(
                frame, include_incomplete=include_incomplete
            ),
        }
        if rows:
            qualities = [str(row.get("signal_quality") or "") for row in rows]
            out["signal_quality"] = rows[0].get("signal_quality")
            if all(quality in {"very_weak", "weak"} for quality in qualities):
                out["detection_status"] = "not_detected"
                out["dominant_period_bars"] = None
                out["quality_note"] = (
                    "Returned periods are weak statistical candidates; treat as exploratory, not confirmed seasonality."
                )
            elif rows[0].get("signal_quality") == "strong":
                out["detection_status"] = "detected"
            else:
                out["detection_status"] = "candidate"
            if all(float(row.get("spectral_strength") or 0.0) == 0.0 for row in rows):
                out["spectral_strength_note"] = (
                    "All returned periods have zero rounded spectral strength; ranking is driven by autocorrelation only."
                )
            if n < _SEASONALITY_SMALL_SAMPLE:
                out["small_sample"] = True
                out["small_sample_note"] = (
                    f"Only {n} observations; labels are capped unless "
                    "autocorrelation exceeds the 95% sampling bound."
                )
        else:
            out["detection_status"] = "not_detected"
            out["dominant_period_bars"] = None
        if normalize_output_verbosity_detail(detail, default="compact") == "full":
            out["method"] = {
                "acf_weight": 0.55,
                "periodogram_weight": 0.45,
                "spectral_component": (
                    "excess of candidate_power / total_positive_frequency_power "
                    "over the equal-bin share 1/bins"
                ),
                "acf_component": "max(0, acf - 1/sqrt(n)) scaled to [0, 1]",
                "minimum_cycles": int(min_cycles),
                "signal_quality": (
                    "Labels use the sample-size-normalized composite score. "
                    "Moderate/strong require a significant positive "
                    "autocorrelation when samples < 100."
                ),
            }
        return out

    return run_mt5_logged_operation(
        logger,
        operation="seasonality_detect",
        symbol=symbol,
        timeframe=timeframe,
        lookback=lookback,
        target=target,
        func=_run,
    )


def _robust_scores(values: pd.Series, method: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)

    def _zero_scale_scores(center: float) -> pd.Series:
        deviations = (numeric - center).abs()
        return deviations.where(deviations == 0.0, np.inf)

    method_value = str(method or "mad").strip().lower()
    if method_value == "zscore":
        scale = float(numeric.std(ddof=0))
        center = float(numeric.mean())
        return (numeric - center).abs() / scale if scale > 0 else _zero_scale_scores(center)
    if method_value == "iqr":
        q1, q3 = numeric.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        center = float(numeric.median())
        scale = iqr / 1.3489795003921634
        return (numeric - center).abs() / scale if scale > 0 else _zero_scale_scores(center)
    if method_value != "mad":
        raise ValueError("method must be one of: mad, iqr, zscore.")
    center = float(numeric.median())
    mad = float((numeric - center).abs().median())
    return 0.67448975 * (numeric - center).abs() / mad if mad > 0 else _zero_scale_scores(center)


@mcp.tool()
def outliers_detect(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    lookback: Annotated[int, Field(ge=20)] = 500,
    score_fields: str = "return,volume,range",
    method: Literal["mad", "iqr", "zscore"] = "mad",
    threshold: Annotated[float, Field(gt=0.0)] = 3.5,
    limit: Annotated[int, Field(ge=1)] = 10,
    include_incomplete: bool = False,
    as_of: Optional[str] = None,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Detect anomalous MT5 bars using MAD, IQR, or mean/std z-score scores."""

    def _run() -> Dict[str, Any]:
        if int(lookback) < 20 or int(limit) < 1 or float(threshold) <= 0:
            return {"error": "lookback >= 20, limit >= 1, and threshold > 0 are required."}
        requested = [
            part.strip().lower()
            for part in str(score_fields or "").split(",")
            if part.strip()
        ]
        requested = list(dict.fromkeys(requested))
        if not requested or any(field not in {"return", "volume", "range"} for field in requested):
            return {
                "error": (
                    "score_fields must contain one or more of: "
                    "return, volume, range."
                )
            }
        gateway = create_mt5_gateway(adapter=mt5, ensure_connection_impl=ensure_mt5_connection_or_raise)
        gateway.ensure_connection()
        try:
            price_precision = symbol_price_digits_optional(
                gateway.symbol_info(symbol)
            )
        except Exception:
            price_precision = None
        frame, fetch_error = _fetch_diagnostic_bars(
            symbol,
            timeframe,
            int(lookback),
            include_incomplete=include_incomplete,
            as_of=as_of,
            operation="outliers_detect",
        )
        if fetch_error:
            return fetch_error if isinstance(fetch_error, dict) else {"error": fetch_error}
        close = pd.to_numeric(frame["close"], errors="coerce")
        data: Dict[str, pd.Series] = {"return": close.pct_change(fill_method=None).abs()}
        epochs = pd.to_numeric(frame["time"], errors="coerce") if "time" in frame.columns else None
        expected_bar_seconds = TIMEFRAME_SECONDS.get(str(timeframe).strip().upper())
        session_gap_mask = pd.Series(False, index=frame.index)
        if epochs is not None and expected_bar_seconds:
            session_gap_mask = epochs.diff() > (float(expected_bar_seconds) * 1.5)
        volume_col: Optional[str] = None
        if "volume" in requested:
            volume_col = "real_volume" if "real_volume" in frame and float(pd.to_numeric(frame["real_volume"], errors="coerce").fillna(0).sum()) > 0 else "tick_volume"
            if volume_col not in frame:
                return {"error": "Fetched bars do not contain volume data."}
            data["volume"] = pd.to_numeric(frame[volume_col], errors="coerce")
        if "range" in requested:
            if not {"high", "low"}.issubset(frame.columns):
                return {"error": "Fetched bars do not contain high and low fields."}
            data["range"] = (pd.to_numeric(frame["high"], errors="coerce") - pd.to_numeric(frame["low"], errors="coerce")).abs()
        score_frame = pd.DataFrame(index=frame.index)
        for field in requested:
            try:
                score_frame[field] = _robust_scores(data[field], method)
            except ValueError as exc:
                return {"error": str(exc)}
        max_scores = score_frame.max(axis=1, skipna=True)
        flagged = frame.loc[max_scores >= float(threshold)].copy()
        flagged["_score"] = max_scores.loc[flagged.index]
        flagged = flagged.sort_values("_score", ascending=False).head(int(limit))
        rows: List[Dict[str, Any]] = []
        full = normalize_output_verbosity_detail(detail, default="compact") == "full"

        def _price(value: Any) -> float:
            number = float(value)
            return (
                round(number, price_precision)
                if price_precision is not None
                else number
            )

        for index, bar in flagged.iterrows():
            raw_field_scores = {
                field: float(score_frame.at[index, field])
                for field in requested
            }
            field_scores = {
                field: round(score if math.isfinite(score) else float(threshold), 4)
                for field, score in raw_field_scores.items()
                if not math.isnan(score)
            }
            raw_score = float(bar["_score"])
            item: Dict[str, Any] = {
                "time": format_datetime_utc(
                    datetime.fromtimestamp(float(bar["time"]), tz=timezone.utc),
                    timespec="auto",
                ),
                "score": round(raw_score if math.isfinite(raw_score) else float(threshold), 4),
                "fields": [
                    field
                    for field, score in raw_field_scores.items()
                    if not math.isnan(score) and score >= float(threshold)
                ],
            }
            if bool(session_gap_mask.get(index, False)):
                item["session_gap"] = True
            if full:
                item.update(
                    {
                        "field_scores": field_scores,
                        "open": _price(bar.get("open")),
                        "high": _price(bar.get("high")),
                        "low": _price(bar.get("low")),
                        "close": _price(bar.get("close")),
                    }
                )
                if volume_col is not None:
                    item["volume"] = float(bar.get(volume_col))
            rows.append(item)
        method_value = str(method or "mad").strip().lower()
        if method_value == "zscore":
            score_meaning = (
                f"mean/std z-score magnitude per bar; score >= threshold "
                f"({float(threshold)}) flags an outlier"
            )
            score_units = "mean_std_zscore"
        elif method_value == "iqr":
            score_meaning = (
                f"IQR robust-z magnitude per bar (Gaussian-consistent IQR scale); "
                f"score >= threshold ({float(threshold)}) flags an outlier"
            )
            score_units = "robust_iqr_zscore"
        else:
            score_meaning = (
                f"robust MAD deviation magnitude per bar; score >= threshold "
                f"({float(threshold)}) flags an outlier"
            )
            score_units = "robust_mad_deviation"
        result: Dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "method": method,
            "threshold": float(threshold),
            "score_meaning": score_meaning,
            "fields_analyzed": requested,
            "samples": int(len(frame)),
            "outliers_total": int((max_scores >= float(threshold)).sum()),
            "items": rows,
            "count": len(rows),
            "truncated": bool(int((max_scores >= float(threshold)).sum()) > len(rows)),
            **_diagnostic_history_metadata(
                frame, include_incomplete=include_incomplete
            ),
        }
        result["units"] = {"score": score_units, "field_scores": score_units}
        if price_precision is not None:
            result["price_precision"] = price_precision
        if volume_col is not None:
            result["volume_source"] = volume_col
            result["volume_type"] = (
                "traded_volume" if volume_col == "real_volume" else "tick_count"
            )
            result["units"]["volume"] = (
                "broker_reported_real_volume"
                if volume_col == "real_volume"
                else "bid_update_count"
            )
            result["units"]["field_scores.volume"] = score_units
        gap_flagged = sum(1 for row in rows if row.get("session_gap") is True)
        if gap_flagged:
            result["session_gap_outliers"] = int(gap_flagged)
            warnings_out = list(result.get("warnings") or [])
            warnings_out.append(
                f"{gap_flagged} flagged bar(s) span a session gap (weekend or "
                "holiday open) and are calendar artifacts, not information events."
            )
            result["warnings"] = warnings_out
        return result

    return run_mt5_logged_operation(
        logger,
        operation="outliers_detect",
        symbol=symbol,
        timeframe=timeframe,
        lookback=lookback,
        method=method,
        func=_run,
    )


@mcp.tool()
def volatility_term_structure(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    lookback: Annotated[int, Field(ge=1)] = 1000,
    horizons: str = "5,10,20,60",
    percentiles: str = "10,25,50,75,90",
    annualize: bool = True,
    include_incomplete: bool = False,
    as_of: Optional[str] = None,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Compute current realized volatility and historical cones at multiple horizons."""

    def _run() -> Dict[str, Any]:
        try:
            horizon_values = sorted(
                set(int(part.strip()) for part in str(horizons).split(",") if part.strip())
            )
            percentile_values = sorted(
                set(float(part.strip()) for part in str(percentiles).split(",") if part.strip())
            )
        except Exception:
            return {"error": "horizons and percentiles must be comma-separated numbers."}
        if not horizon_values or any(value < 1 for value in horizon_values):
            return {"error": "horizons must contain positive integers."}
        if not percentile_values:
            return build_error_payload(
                "percentiles must contain at least one value strictly between 0 and 100.",
                code="invalid_parameter",
                operation="volatility_term_structure",
                details={"parameter": "percentiles", "received": percentiles},
                remediation="Pass a comma-separated list such as 10,25,50,75,90.",
                valid_values={"percentiles": "0 < p < 100"},
                example="--percentiles 10,25,50,75,90",
            )
        if any(value <= 0.0 or value >= 100.0 for value in percentile_values):
            return build_error_payload(
                "percentiles must be strictly between 0 and 100.",
                code="invalid_parameter",
                operation="volatility_term_structure",
                details={"parameter": "percentiles", "received": percentiles},
                remediation="Pass values such as 10,25,50,75,90.",
                valid_values={"percentiles": "0 < p < 100"},
                example="--percentiles 10,25,50,75,90",
            )
        maximum_horizon = max(horizon_values)
        minimum_lookback = max(30, maximum_horizon + 1)
        if int(lookback) < minimum_lookback:
            return build_error_payload(
                (
                    f"lookback must be at least {minimum_lookback} for the requested "
                    f"horizons; the largest horizon ({maximum_horizon}) must be "
                    "smaller than lookback."
                ),
                code="incompatible_parameters",
                operation="volatility_term_structure",
                details={
                    "parameter": "lookback",
                    "received": int(lookback),
                    "required_minimum": minimum_lookback,
                    "largest_horizon": maximum_horizon,
                },
                remediation=(
                    f"Increase lookback to at least {minimum_lookback}, or reduce "
                    "the largest requested horizon."
                ),
                example=f"--lookback {minimum_lookback} --horizons {horizons}",
            )
        gateway = create_mt5_gateway(adapter=mt5, ensure_connection_impl=ensure_mt5_connection_or_raise)
        gateway.ensure_connection()
        frame, fetch_error = _fetch_diagnostic_bars(
            symbol,
            timeframe,
            int(lookback),
            include_incomplete=include_incomplete,
            as_of=as_of,
            operation="volatility_term_structure",
        )
        if fetch_error:
            return fetch_error if isinstance(fetch_error, dict) else {"error": fetch_error}
        close = pd.to_numeric(frame["close"], errors="coerce")
        returns = np.log(close.where(close > 0)).diff().replace([np.inf, -np.inf], np.nan)
        observed_times = frame["time"] if "time" in frame else None
        timeframe_seconds = TIMEFRAME_SECONDS.get(str(timeframe).strip().upper())
        intraday = bool(timeframe_seconds and float(timeframe_seconds) < 86400.0)
        bpy, annualization_basis = (
            annualization_context(timeframe, symbol, observed_times=observed_times)
            if annualize
            else (float("nan"), "not_annualized")
        )
        factor = math.sqrt(bpy) if annualize else 1.0
        if not math.isfinite(factor) or factor <= 0.0:
            factor = 1.0
        rows: List[Dict[str, Any]] = []
        low_sample_horizons: List[int] = []
        for horizon in horizon_values:
            realized = returns.pow(2).rolling(window=int(horizon), min_periods=int(horizon)).mean().pow(0.5) * factor
            distribution = realized.dropna()
            if distribution.empty:
                continue
            current = float(distribution.iloc[-1])
            samples = int(len(distribution))
            bars_used = int(len(returns.dropna())) + 1
            effective_samples = int(bars_used // int(horizon)) if int(horizon) > 0 else 0
            sufficient = effective_samples >= 20
            if sufficient:
                percentile_rank: Optional[float] = float(
                    (distribution <= current).mean() * 100.0
                )
                cone: Optional[Dict[str, float]] = {
                    f"p{int(value) if float(value).is_integer() else value:g}": round(
                        float(np.percentile(distribution.to_numpy(dtype=float), value)),
                        8,
                    )
                    for value in percentile_values
                }
            else:
                percentile_rank = None
                cone = None
                low_sample_horizons.append(int(horizon))
            rows.append(
                {
                    "horizon_bars": int(horizon),
                    "current_volatility": round(current, 8),
                    "current_volatility_basis": (
                        "latest_bar_abs_log_return"
                        if int(horizon) == 1
                        else f"rolling_rms_{int(horizon)}_bar"
                    ),
                    "per_bar_volatility": round(current / factor, 8),
                    "stability": (
                        "insufficient_sample"
                        if not sufficient
                        else
                        "very_low"
                        if horizon < 5
                        else "low"
                        if horizon < 10
                        else "moderate"
                    ),
                    "percentile_rank": (
                        round(percentile_rank, 2)
                        if percentile_rank is not None
                        else None
                    ),
                    "cone": cone,
                    "samples": samples,
                    "effective_samples": effective_samples,
                    "sample_sufficiency": (
                        "sufficient" if sufficient else "insufficient"
                    ),
                    "minimum_samples_for_percentiles": 20,
                }
            )
        if not rows:
            return {"error": "Insufficient finite returns for the requested horizons."}
        out: Dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "annualized": bool(annualize),
            "analysis_kind": "historical_realized_volatility_cones",
            "comparable_to_options_iv": False,
            "unit": "annualized_decimal_volatility" if annualize else "per_bar_decimal_volatility",
            "unit_note": (
                "Volatility values are decimal return fractions; 0.01 means 1%. "
                "Multiply by 100 for percent-point tools such as "
                "labels_triple_barrier and forecast_barrier_prob "
                "(0.0465 decimal = 4.65 percent-points)."
            ),
            "units": {
                "current_volatility": "decimal_return_fraction",
                "per_bar_volatility": "per_bar_decimal_return_fraction",
                "cone": "decimal_return_fraction",
                "percentile_rank": "percentile_rank (0=lowest, 100=highest)",
            },
            "cone_methodology": (
                "percentiles of overlapping rolling realized-volatility windows; "
                "sufficiency uses effective independent samples (bars/horizon), "
                "not the overlapping window count; overlapping windows understate "
                "dispersion; this is not an options implied-volatility term structure"
            ),
            "items": rows,
            **_diagnostic_history_metadata(
                frame, include_incomplete=include_incomplete
            ),
            "count": len(rows),
        }
        if low_sample_horizons:
            out["warnings"] = [
                "Percentile ranks and cones were suppressed for horizons with "
                "fewer than 20 effective independent samples (bars/horizon)."
            ]
            out["low_sample_horizons"] = low_sample_horizons
        if annualize:
            out["bars_per_year"] = round(float(bpy), 4) if math.isfinite(bpy) else None
            sessions_per_year = (
                365 if is_probably_crypto_symbol(symbol)
                else 260 if is_probably_forex_symbol(symbol)
                else 252
            )
            if intraday and math.isfinite(bpy):
                out["bars_per_session"] = round(float(bpy) / sessions_per_year, 4)
                out["sessions_per_year"] = sessions_per_year
            out["annualization_basis"] = annualization_basis
        if normalize_output_verbosity_detail(detail, default="compact") == "full":
            out["method"] = "rolling_root_mean_square_log_return"
            out["lookback"] = int(lookback)
        return out

    return run_mt5_logged_operation(
        logger,
        operation="volatility_term_structure",
        symbol=symbol,
        timeframe=timeframe,
        lookback=lookback,
        func=_run,
    )
