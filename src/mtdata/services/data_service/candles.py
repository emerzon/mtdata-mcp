import errno
import json
import logging
import math
import re
import time
import warnings
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ...bootstrap.settings import mt5_config
from ...core.error_envelope import build_error_payload
from ...shared.constants import (
    CALENDAR_TIMEFRAMES,
    DEFAULT_ROW_LIMIT,
    FETCH_RETRY_ATTEMPTS,
    FETCH_RETRY_DELAY,
    SANITY_BARS_TOLERANCE,
    TI_NAN_WARMUP_FACTOR,
    TI_NAN_WARMUP_MIN_ADD,
    TIMEFRAME_MAP,
    TIMEFRAME_SECONDS,
)
from ...shared.schema import DenoiseSpec, IndicatorSpec, SimplifySpec, TimeframeLiteral
from ...shared.symbols import is_probably_crypto_symbol
from ...shared.validators import invalid_timeframe_error
from ...utils.coercion import coerce_finite_float, coerce_scalar, round_finite
from ...utils.denoise import (
    DenoiseCausalityError,
    DenoiseColumnError,
    consume_denoise_warnings,
)
from ...utils.denoise import (
    apply_denoise as apply_denoise_util,
)
from ...utils.denoise import (
    normalize_denoise_spec as _normalize_denoise_spec,
)
from ...utils.denoise.base import DenoiseExecutionError, DenoiseParameterError
from ...utils.denoise.filters.moving_average import ema_alpha
from ...utils.freshness import (
    closed_session_context,
    freshness_hole_explained_by_weekend,
)
from ...utils.indicators import (
    _apply_ta_indicators,
    _estimate_warmup_bars,
    _find_unknown_ta_indicators,
    _parse_ti_specs,
    indicator_engine_provenance,
)
from ...utils.market_metadata import (
    FRESHNESS_ANCHOR_QUERY_EXPECTED_END,
    FRESHNESS_ANCHOR_WALL_CLOCK,
    FRESHNESS_METRIC_LAST_COMPLETED_BAR_AGE,
    FRESHNESS_METRIC_REQUESTED_RANGE_END_GAP,
    TICK_VOLUME_COMPARISON_NOTE,
    TICK_VOLUME_EVENT_BASIS,
    TICK_VOLUME_TAPE_EQUIVALENT,
    TICK_VOLUME_UNIT,
)
from ...utils.mt5 import (
    MT5TimestampNormalizationError,
    _mt5_copy_rates_from,
    _mt5_copy_rates_range,
    _rates_to_df,
    _symbol_ready_guard,
    describe_mt5_time_normalization,
    get_cached_mt5_time_alignment,
    get_symbol_info_cached,
    resolve_broker_symbol_name,
)
from ...utils.mt5 import (
    symbol_candle_price_basis as _symbol_candle_price_basis,
)
from ...utils.mt5 import (
    symbol_price_currency as _symbol_price_currency,
)
from ...utils.mt5 import (
    symbol_price_digits as _symbol_price_digits,
)
from ...utils.mt5 import (
    symbol_price_point as _symbol_price_point,
)
from ...utils.ohlcv import validate_and_clean_ohlcv_frame
from ...utils.simplify import _normalize_simplify_spec, _simplify_dataframe_rows_ext
from ...utils.time import (
    _format_datetime_minute_explicit,
    _format_time_minimal,
    _format_time_minimal_local,
    _resolve_client_tz,
    bar_close_epoch,
    display_timezone_label,
    format_epoch_utc,
)
from ...utils.utils import (
    _format_numeric_rows_from_df,
    _normalize_ohlcv_arg,
    _parse_end_datetime,
    _parse_start_datetime,
    _table_from_rows,
    _utc_epoch_seconds,
)
from .errors import (
    _build_no_data_error_with_context,
    _describe_rate_fetch_error,
    _future_start_error,
)
from .query import (
    _broker_calendar_timezone,
    _candle_query_applied,
    _is_calendar_query_bound,
    _parse_candle_calendar_bound,
    _parse_fetch_datetime_arg,
)

logger = logging.getLogger(__name__)

_MT5_HISTORY_QUERY_MIN = datetime(1970, 1, 1, tzinfo=dt_timezone.utc)
_MT5_RATE_FIELD_ORDER = (
    "time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
)


class _RateDataShapeError(ValueError):
    """Provider rates are missing the fields required to filter or display them."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        super().__init__(str(payload.get("error") or "data_shape_invalid"))
        self.payload = payload


def _provider_rates_missing_time_error() -> Dict[str, Any]:
    return build_error_payload(
        "Provider candle rows are missing the required 'time' field.",
        code="data_shape_invalid",
        operation="data_fetch_candles",
        remediation=(
            "Retry the request. If it persists, the broker history payload is "
            "not in the expected MT5 rate shape."
        ),
        details={"missing_fields": ["time"], "required_fields": ["time"]},
    )


def _rate_row_as_mapping(row: Any) -> Optional[Dict[str, Any]]:
    """Normalize one copy_rates_* row to a mapping with named MT5 fields."""
    if isinstance(row, dict):
        return dict(row)
    names = getattr(getattr(row, "dtype", None), "names", None)
    if names:
        try:
            return {str(name): row[name] for name in names}
        except Exception:
            return None
    if isinstance(row, (tuple, list)):
        if not row:
            return None
        return {
            name: row[index]
            for index, name in enumerate(_MT5_RATE_FIELD_ORDER)
            if index < len(row)
        }
    try:
        return {"time": row["time"]}
    except Exception:
        return None


def _rate_row_epoch(row: Any) -> Optional[float]:
    mapped = row if isinstance(row, dict) else _rate_row_as_mapping(row)
    if not mapped or "time" not in mapped:
        return None
    try:
        epoch = float(mapped["time"])
    except (TypeError, ValueError):
        return None
    return epoch if math.isfinite(epoch) else None


def _normalize_provider_rate_rows(
    rates: Any,
) -> tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """Convert provider rates to validated mappings before time filtering."""
    if rates is None:
        return None, None
    try:
        if len(rates) == 0:
            return [], None
    except TypeError:
        return None, _provider_rates_missing_time_error()

    names = getattr(getattr(rates, "dtype", None), "names", None)
    if names:
        if "time" not in names:
            return None, _provider_rates_missing_time_error()
        try:
            frame = _rates_to_df(rates)
        except Exception:
            return None, _provider_rates_missing_time_error()
        if "time" not in frame.columns:
            return None, _provider_rates_missing_time_error()
        return frame.to_dict("records"), None

    rows: List[Dict[str, Any]] = []
    for row in rates:
        mapped = _rate_row_as_mapping(row)
        if mapped is None or "time" not in mapped:
            return None, _provider_rates_missing_time_error()
        rows.append(mapped)
    return rows, None


_MT5_INVALID_DATE_RANGE_ERROR = (
    "MT5 rejected the requested candle date range because one or more bounds "
    "are outside its supported history window."
)


_CANDLE_PRICE_COLUMNS = frozenset({"open", "high", "low", "close", "spread"})


def _round_price_value(value: Any, digits: int) -> Any:
    if digits <= 0:
        return value
    return round_finite(value, digits, on_invalid="passthrough")


def _round_row_price_columns(
    rows: List[List[Any]],
    headers: List[str],
    *,
    digits: int,
    price_columns: frozenset[str],
) -> List[List[Any]]:
    if digits <= 0:
        return rows
    price_indexes = [
        idx for idx, header in enumerate(headers) if str(header) in price_columns
    ]
    if not price_indexes:
        return rows
    rounded_rows: List[List[Any]] = []
    for row in rows:
        rounded = list(row)
        for idx in price_indexes:
            if idx < len(rounded):
                rounded[idx] = _round_price_value(rounded[idx], digits)
        rounded_rows.append(rounded)
    return rounded_rows


_PRICE_INDICATOR_PREFIXES = (
    "ALMA_",
    "BBL_",
    "BBM_",
    "BBU_",
    "DEMA_",
    "EMA_",
    "HMA_",
    "KAMA_",
    "SMA_",
    "TEMA_",
    "VWAP",
    "VWMA_",
    "WMA_",
)


def _price_indicator_columns(columns: List[str]) -> List[str]:
    out: List[str] = []
    for column in columns:
        name = str(column or "").strip().upper()
        if name.startswith(_PRICE_INDICATOR_PREFIXES):
            out.append(str(column))
    return out


def _indicator_param_syntax_error(ti_spec: Optional[str]) -> Optional[str]:
    if not ti_spec:
        return None
    for name, _args, _kwargs in _parse_ti_specs(ti_spec):
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", str(name or "").strip()):
            return "Indicator params must use parentheses, e.g. sma(20), not sma,20."
    return None


def _indicator_validation_error(
    message: str,
    *,
    received: Any = None,
    parameter: str = "indicators",
) -> Dict[str, Any]:
    details: Dict[str, Any] = {"parameter": parameter}
    if received is not None:
        details["received"] = received
    return build_error_payload(
        message,
        code="invalid_indicator_parameter",
        operation="data_fetch_candles",
        details=details,
        remediation=(
            "Use name(params) syntax such as rsi(14) or rsi(talib=false). "
            "Use true/false for boolean settings. "
            "Keep rolling-window periods greater than 0 and within the "
            "available bar count."
        ),
        related_tools=["indicators_list"],
        valid_values={"catalog": "indicators_list"},
        example="rsi(14)",
    )


def _resolve_live_bar_reference_epoch(symbol: Optional[str], timeframe: str) -> float:
    """Use wall-clock time when classifying whether the latest bar is live."""
    del symbol, timeframe
    system_epoch = _utc_epoch_seconds(datetime.now(dt_timezone.utc))
    return float(system_epoch)


def _is_last_bar_forming(
    rates_or_df: Any,
    timeframe: str,
    *,
    current_time_epoch: Optional[float] = None,
) -> bool:
    """Return True if the last bar in *rates_or_df* is still forming."""
    try:
        current_time = (
            float(current_time_epoch)
            if current_time_epoch is not None and math.isfinite(float(current_time_epoch))
            else float(_utc_epoch_seconds(datetime.now(dt_timezone.utc)))
        )
        if isinstance(rates_or_df, pd.DataFrame):
            if len(rates_or_df) == 0:
                return False
            epoch_column = '__epoch' if '__epoch' in rates_or_df.columns else 'time'
            if epoch_column not in rates_or_df.columns:
                return True
            last_epoch = float(rates_or_df[epoch_column].iloc[-1])
        else:
            if rates_or_df is None or len(rates_or_df) == 0:
                return False
            last_epoch = float(rates_or_df[-1]["time"])
        return current_time < bar_close_epoch(last_epoch, timeframe)
    except Exception:
        # A non-empty tail whose timestamp cannot be classified must not be
        # silently presented as a completed candle.
        try:
            return rates_or_df is not None and len(rates_or_df) > 0
        except Exception:
            return False


def _drop_incomplete_tail(
    rates: Any,
    timeframe: str,
    *,
    current_time_epoch: Optional[float] = None,
) -> Any:
    """Remove every unfinished tail bar from chronologically ordered rates."""
    while (
        rates is not None
        and len(rates) > 0
        and _is_last_bar_forming(rates, timeframe, current_time_epoch=current_time_epoch)
    ):
        rates = rates[:-1]
    return rates


def _drop_incomplete_tail_df(
    df: pd.DataFrame,
    timeframe: str,
    *,
    current_time_epoch: Optional[float] = None,
) -> Tuple[pd.DataFrame, bool]:
    """Remove every unfinished tail row; return (frame, trimmed)."""
    original_count = len(df)
    while len(df) > 0 and _is_last_bar_forming(df, timeframe, current_time_epoch=current_time_epoch):
        df = df.iloc[:-1]
    return df, len(df) != original_count


def _build_candle_freshness_diagnostics(
    *,
    last_bar_epoch: Any,
    expected_end_epoch: Any,
    freshness_cutoff_epoch: Any,
    data_freshness_reference_epoch: Any = None,
) -> Dict[str, Any]:
    def _coerce_epoch(value: Any) -> Optional[float]:
        return coerce_finite_float(value)

    last_epoch = _coerce_epoch(last_bar_epoch)
    expected_epoch = _coerce_epoch(expected_end_epoch)
    reference_epoch = _coerce_epoch(data_freshness_reference_epoch)
    if reference_epoch is None:
        reference_epoch = expected_epoch
    cutoff_epoch = _coerce_epoch(freshness_cutoff_epoch)
    data_freshness_seconds: Optional[float] = None
    last_bar_within_policy_window: Optional[bool] = None

    if last_epoch is not None and reference_epoch is not None:
        data_freshness_seconds = round(
            max(0.0, float(reference_epoch - last_epoch)),
            3,
        )
    if last_epoch is not None and cutoff_epoch is not None:
        last_bar_within_policy_window = bool(last_epoch >= cutoff_epoch)

    diagnostics = {
        "last_bar_epoch": last_epoch,
        "expected_end_epoch": expected_epoch,
        "freshness_cutoff_epoch": cutoff_epoch,
        "data_freshness_seconds": data_freshness_seconds,
        "last_bar_within_policy_window": last_bar_within_policy_window,
    }
    if reference_epoch != expected_epoch:
        diagnostics["data_freshness_reference_epoch"] = reference_epoch
    return diagnostics


def _latest_candle_freshness_cutoff(
    *,
    reference_epoch: float,
    last_bar_open_epoch: Any,
    seconds_per_bar: int,
) -> float:
    """Align the latest-query freshness window to the provider's bar grid."""
    last_open = float(last_bar_open_epoch)
    bar_seconds = float(seconds_per_bar)
    elapsed = max(0.0, float(reference_epoch) - last_open)
    current_bar_open = last_open + math.floor(elapsed / bar_seconds) * bar_seconds
    return current_bar_open - bar_seconds


def _relax_live_completed_bar_freshness(
    *,
    symbol: str,
    rates: Any,
    timeframe: TimeframeLiteral,
    expected_end_ts: float,
    start_datetime: Optional[str],
    end_datetime: Optional[str],
    freshness_meta: Dict[str, Any],
) -> bool:
    if start_datetime or end_datetime:
        return False
    if _is_last_bar_forming(
        rates,
        timeframe,
        current_time_epoch=float(expected_end_ts),
    ):
        return False
    closed_session = closed_session_context(
        symbol,
        now_epoch=expected_end_ts,
        item="bar",
        data_age_seconds=freshness_meta.get("data_freshness_seconds"),
    )
    if not closed_session or not bool(
        closed_session.get("freshness_policy_relaxed")
    ):
        return False
    freshness_meta["freshness_policy_relaxed"] = True
    freshness_meta["market_session_status"] = closed_session.get(
        "market_status"
    )
    freshness_meta["market_session_reason"] = closed_session.get(
        "market_status_reason"
    )
    freshness_meta["market_session_source"] = closed_session.get(
        "market_status_source"
    )
    freshness_meta["freshness_note"] = closed_session.get("note")
    return True


def _session_break_explains_latest_n_freshness(
    *,
    symbol: str,
    timeframe: TimeframeLiteral,
    last_completed_epoch: Optional[float],
    last_completed_open: Optional[float],
    freshness_cutoff: Optional[float],
    next_bar_open_epoch: Optional[float],
    start_datetime: Optional[str],
    end_datetime: Optional[str],
    freshness_meta: Dict[str, Any],
) -> bool:
    """True when an unbounded latest-N hole is a weekend/session break, not a feed gap."""
    if start_datetime or end_datetime:
        return False
    if is_probably_crypto_symbol(symbol):
        return False
    if last_completed_epoch is None or freshness_cutoff is None:
        return False
    try:
        last_epoch = float(last_completed_epoch)
        cutoff_epoch = float(freshness_cutoff)
    except (TypeError, ValueError):
        return False
    if last_epoch >= cutoff_epoch:
        return False
    bar_seconds = float(TIMEFRAME_SECONDS.get(timeframe, 0) or 0)
    if bar_seconds <= 0:
        return False
    if not freshness_hole_explained_by_weekend(
        last_completed_epoch=last_epoch,
        cutoff_epoch=cutoff_epoch,
        bar_seconds=bar_seconds,
    ):
        return False
    previous_open = (
        float(last_completed_open)
        if last_completed_open is not None
        else last_epoch
    )
    next_open = (
        float(next_bar_open_epoch)
        if next_bar_open_epoch is not None
        else cutoff_epoch
    )
    gap = _describe_session_gap(
        previous_open,
        next_open,
        expected_bar_seconds=bar_seconds,
        use_client_tz=False,
    )
    if gap is None:
        return False
    freshness_meta["session_gap_explains_freshness"] = True
    if gap.get("context"):
        freshness_meta["session_gap_context"] = gap.get("context")
    return True


def _fetch_rates_with_warmup(  # noqa: C901
    symbol: str,
    mt5_timeframe: int,
    timeframe: TimeframeLiteral,
    candles: int,
    warmup_bars: int,
    start_datetime: Optional[str],
    end_datetime: Optional[str],
    *,
    include_incomplete: bool = False,
    retry: bool = True,
    sanity_check: bool = True,
    diagnostics: Optional[Dict[str, Any]] = None,
    range_selection: Optional[str] = None,
):
    """Fetch MT5 rates with optional warmup, retry, and end-bar sanity checks."""
    extra_bars = 0 if include_incomplete else 1
    if diagnostics is not None:
        diagnostics.pop("freshness", None)
    if start_datetime and end_datetime:
        seconds_per_bar, timeframe_error = _resolve_fetch_timeframe_seconds(timeframe)
        if timeframe_error:
            return None, timeframe_error
        from_date, from_date_error = _parse_fetch_datetime_arg(
            start_datetime,
            timeframe=timeframe,
        )
        to_date, to_date_error = _parse_fetch_datetime_arg(
            end_datetime,
            end_bound=True,
            timeframe=timeframe,
        )
        if from_date_error or to_date_error:
            return None, from_date_error or to_date_error
        if from_date > to_date:
            return None, "start must be before or equal to end."
        if from_date < _MT5_HISTORY_QUERY_MIN:
            return None, (
                f"start datetime {start_datetime} is before MT5's supported history "
                "boundary (1970-01-01T00:00:00Z)."
            )
        if to_date < _MT5_HISTORY_QUERY_MIN:
            return None, (
                f"end datetime {end_datetime} is before MT5's supported history "
                "boundary (1970-01-01T00:00:00Z)."
            )
        future_error = _future_start_error(start_datetime, from_date, seconds_per_bar)
        if future_error:
            return None, future_error
        overlap_periods = int(
            timeframe in {"W1", "MN1"}
            and _is_calendar_query_bound(start_datetime)
        )
        from_date_internal = _mt5_history_start_with_warmup(
            from_date,
            seconds_per_bar * (warmup_bars + extra_bars + overlap_periods),
        )
        expected_end_ts = _utc_epoch_seconds(to_date)
        requested_rows = max(1, candles + warmup_bars + extra_bars)
        available_span_seconds = max(
            0.0,
            float((to_date - from_date_internal).total_seconds()),
        )
        initial_span_seconds = min(
            available_span_seconds,
            seconds_per_bar * max(requested_rows * 2, requested_rows + 7),
        )
        if diagnostics is not None:
            diagnostics["range_fetch"] = {
                "provider_bounded": False,
                "provider_start": _format_time_minimal(
                    _utc_epoch_seconds(from_date_internal)
                ),
                "requested_start": _format_time_minimal(
                    _utc_epoch_seconds(from_date)
                ),
                "provider_row_budget": requested_rows,
            }

        def _fetch():
            if str(range_selection or "").strip().lower() == "last_n":
                trailing = _mt5_copy_rates_from(
                    symbol,
                    mt5_timeframe,
                    to_date,
                    requested_rows,
                )
                if trailing is None:
                    return None
                normalized, shape_error = _normalize_provider_rate_rows(trailing)
                if shape_error:
                    raise _RateDataShapeError(shape_error)
                start_epoch = _utc_epoch_seconds(from_date_internal)
                filtered = [
                    row
                    for row in (normalized or [])
                    if (epoch := _rate_row_epoch(row)) is not None
                    and epoch >= start_epoch
                ]
                if diagnostics is not None:
                    diagnostics["range_fetch"].update(
                        {
                            "provider_bounded": len(filtered) >= requested_rows,
                            "provider_end": _format_time_minimal(expected_end_ts),
                            "provider_end_bounded": True,
                            "selection_anchor": "end",
                        }
                    )
                return filtered
            candidate_end = min(
                to_date,
                from_date_internal + timedelta(seconds=initial_span_seconds),
            )
            result = None
            for _ in range(20):
                result = _mt5_copy_rates_range(
                    symbol,
                    mt5_timeframe,
                    from_date_internal,
                    candidate_end,
                )
                if result is None:
                    return None
                qualifying = sum(
                    1
                    for row in result
                    if (epoch := _rate_row_epoch(row)) is not None
                    and epoch >= _utc_epoch_seconds(from_date)
                )
                if qualifying >= candles + extra_bars or candidate_end >= to_date:
                    if diagnostics is not None:
                        diagnostics["range_fetch"]["provider_end"] = (
                            _format_time_minimal(_utc_epoch_seconds(candidate_end))
                        )
                        diagnostics["range_fetch"]["provider_end_bounded"] = (
                            candidate_end < to_date
                        )
                    return result
                elapsed = max(
                    seconds_per_bar,
                    (candidate_end - from_date_internal).total_seconds(),
                )
                candidate_end = min(
                    to_date,
                    from_date_internal + timedelta(seconds=elapsed * 2),
                )
            return result

    elif start_datetime:
        from_date, from_date_error = _parse_fetch_datetime_arg(
            start_datetime,
            timeframe=timeframe,
        )
        if from_date_error:
            return None, from_date_error
        seconds_per_bar, timeframe_error = _resolve_fetch_timeframe_seconds(timeframe)
        if timeframe_error:
            return None, timeframe_error
        if from_date < _MT5_HISTORY_QUERY_MIN:
            return None, (
                f"start datetime {start_datetime} is before MT5's supported history "
                "boundary (1970-01-01T00:00:00Z)."
            )
        future_error = _future_start_error(start_datetime, from_date, seconds_per_bar)
        if future_error:
            return None, future_error
        now_utc = datetime.now(dt_timezone.utc)
        scan_start = max(from_date, _MT5_HISTORY_QUERY_MIN)
        overlap_periods = int(
            timeframe in {"W1", "MN1"}
            and _is_calendar_query_bound(start_datetime)
        )
        from_date_internal = _mt5_history_start_with_warmup(
            scan_start,
            seconds_per_bar * (warmup_bars + extra_bars + overlap_periods),
        )
        requested_rows = candles + extra_bars
        last_n_requested = str(range_selection or "").strip().lower() == "last_n"
        # A start-only query defaults to "the first N bars from start".
        # `--selection last_n` instead keeps the latest N bars since start.
        span_seconds = seconds_per_bar * max(requested_rows * 3, requested_rows + 7)
        to_date = (
            now_utc
            if last_n_requested
            else min(now_utc, scan_start + timedelta(seconds=span_seconds))
        )
        expected_end_ts = _utc_epoch_seconds(now_utc)
        if diagnostics is not None:
            diagnostics["range_fetch"] = {
                "provider_bounded": False,
                "provider_start": _format_time_minimal(
                    _utc_epoch_seconds(from_date_internal)
                ),
                "requested_start": _format_time_minimal(
                    _utc_epoch_seconds(from_date)
                ),
                "requested_end": _format_time_minimal(expected_end_ts),
                "requested_end_source": "wall_clock_now",
                "provider_row_budget": requested_rows,
            }

        def _fetch():
            if last_n_requested:
                trailing = _mt5_copy_rates_from(
                    symbol,
                    mt5_timeframe,
                    now_utc,
                    max(1, candles + warmup_bars + extra_bars),
                )
                if trailing is None:
                    return None
                normalized, shape_error = _normalize_provider_rate_rows(trailing)
                if shape_error:
                    raise _RateDataShapeError(shape_error)
                start_epoch = _utc_epoch_seconds(from_date)
                filtered = [
                    row
                    for row in (normalized or [])
                    if (epoch := _rate_row_epoch(row)) is not None
                    and epoch >= start_epoch
                ]
                if diagnostics is not None:
                    diagnostics["range_fetch"].update(
                        {
                            "provider_bounded": len(filtered)
                            >= candles + extra_bars,
                            "provider_end": _format_time_minimal(expected_end_ts),
                            "provider_end_bounded": True,
                            "selection_anchor": "end",
                        }
                    )
                return filtered
            # Closed sessions consume calendar time without producing rows.
            # Expand only when the bounded response cannot satisfy the first-N
            # contract, stopping at the present rather than guessing a fixed
            # weekend/holiday allowance.
            candidate_end = to_date
            result = None
            while True:
                result = _mt5_copy_rates_range(
                    symbol, mt5_timeframe, from_date_internal, candidate_end
                )
                if result is None:
                    return None
                qualifying = sum(
                    1
                    for row in result
                    if (epoch := _rate_row_epoch(row)) is not None
                    and epoch >= _utc_epoch_seconds(from_date)
                )
                if qualifying >= candles or candidate_end >= now_utc:
                    if diagnostics is not None:
                        provider_end_bounded = candidate_end < now_utc
                        diagnostics["range_fetch"].update(
                            {
                                "provider_bounded": bool(
                                    provider_end_bounded and qualifying >= candles
                                ),
                                "provider_end": _format_time_minimal(
                                    _utc_epoch_seconds(candidate_end)
                                ),
                                "provider_end_bounded": provider_end_bounded,
                            }
                        )
                    return result
                elapsed = max(
                    seconds_per_bar,
                    (candidate_end - scan_start).total_seconds(),
                )
                remaining = max(
                    0.0,
                    (now_utc - scan_start).total_seconds(),
                )
                next_elapsed = min(remaining, elapsed * 2)
                next_end = scan_start + timedelta(seconds=next_elapsed)
                if next_end <= candidate_end:
                    return result
                candidate_end = next_end

    elif end_datetime:
        to_date, to_date_error = _parse_fetch_datetime_arg(
            end_datetime,
            end_bound=True,
            timeframe=timeframe,
        )
        if to_date_error:
            return None, to_date_error
        if to_date < _MT5_HISTORY_QUERY_MIN:
            return None, (
                f"end datetime {end_datetime} is before MT5's supported history "
                "boundary (1970-01-01T00:00:00Z)."
            )
        seconds_per_bar, timeframe_error = _resolve_fetch_timeframe_seconds(timeframe)
        if timeframe_error:
            return None, timeframe_error
        expected_end_ts = _utc_epoch_seconds(to_date)

        def _fetch():
            return _mt5_copy_rates_from(symbol, mt5_timeframe, to_date, candles + warmup_bars + extra_bars)

    else:
        utc_now = datetime.now(dt_timezone.utc)
        seconds_per_bar, timeframe_error = _resolve_fetch_timeframe_seconds(timeframe)
        if timeframe_error:
            return None, timeframe_error
        expected_end_ts = _utc_epoch_seconds(utc_now)

        def _fetch():
            return _mt5_copy_rates_from(symbol, mt5_timeframe, utc_now, candles + warmup_bars + extra_bars)

    wall_clock_ts = _utc_epoch_seconds(datetime.now(dt_timezone.utc))
    range_query = bool(start_datetime or end_datetime)
    explicit_now_end = str(end_datetime or "").strip().casefold() == "now"
    live_range = bool(
        range_query
        and (
            expected_end_ts >= wall_clock_ts
            or explicit_now_end
            or (start_datetime and not end_datetime)
        )
    )
    freshness_reference_ts = wall_clock_ts if live_range else expected_end_ts

    attempts = FETCH_RETRY_ATTEMPTS if retry else 1
    rates = None
    stale_last_t: Optional[float] = None
    stale_forming_t: Optional[float] = None
    freshness_cutoff: Optional[float] = None
    for idx in range(attempts):
        try:
            rates = _fetch()
        except _RateDataShapeError as exc:
            return None, exc.payload
        except OSError as exc:
            if exc.errno == errno.EINVAL:
                return None, _MT5_INVALID_DATE_RANGE_ERROR
            raise
        if rates is not None and len(rates) > 0:
            last_t = rates[-1]["time"]
            freshness_policy_bars = (
                SANITY_BARS_TOLERANCE + extra_bars if range_query else 1
            )
            if range_query:
                freshness_cutoff = (
                    freshness_reference_ts
                    - seconds_per_bar * freshness_policy_bars
                )
            else:
                freshness_cutoff = _latest_candle_freshness_cutoff(
                    reference_epoch=freshness_reference_ts,
                    last_bar_open_epoch=last_t,
                    seconds_per_bar=seconds_per_bar,
                )
            tail_is_forming = _is_last_bar_forming(
                rates, timeframe, current_time_epoch=wall_clock_ts
            )
            if tail_is_forming and include_incomplete:
                # The forming bar itself proves the feed reached its open
                # time; completed-bar freshness is attached after it is
                # excluded, while live output gets last-tick freshness.
                last_completed_epoch = float(last_t)
                last_completed_open = float(last_t)
            elif tail_is_forming and len(rates) >= 2:
                last_completed_epoch = bar_close_epoch(
                    rates[-2]["time"], timeframe
                )
                last_completed_open = float(rates[-2]["time"])
            elif tail_is_forming:
                last_completed_epoch = None
                last_completed_open = None
            else:
                last_completed_epoch = bar_close_epoch(last_t, timeframe)
                last_completed_open = float(last_t)
            freshness_meta = _build_candle_freshness_diagnostics(
                last_bar_epoch=last_completed_epoch,
                expected_end_epoch=expected_end_ts,
                freshness_cutoff_epoch=freshness_cutoff,
                data_freshness_reference_epoch=freshness_reference_ts,
            )
            freshness_meta["last_bar_open_epoch"] = last_completed_open
            if live_range:
                if last_completed_epoch is not None:
                    freshness_meta["query_end_gap_seconds"] = round(
                        max(0.0, expected_end_ts - last_completed_epoch),
                        3,
                    )
                freshness_meta["query_end_gap_anchor"] = (
                    FRESHNESS_ANCHOR_QUERY_EXPECTED_END
                )
                freshness_meta["query_end_gap_metric"] = (
                    FRESHNESS_METRIC_REQUESTED_RANGE_END_GAP
                )
            if live_range:
                freshness_meta["data_freshness_anchor"] = FRESHNESS_ANCHOR_WALL_CLOCK
                freshness_meta["data_freshness_metric"] = (
                    FRESHNESS_METRIC_LAST_COMPLETED_BAR_AGE
                )
            elif range_query:
                freshness_meta["data_freshness_anchor"] = (
                    FRESHNESS_ANCHOR_QUERY_EXPECTED_END
                )
                freshness_meta["data_freshness_metric"] = (
                    FRESHNESS_METRIC_REQUESTED_RANGE_END_GAP
                )
            else:
                freshness_meta["data_freshness_anchor"] = FRESHNESS_ANCHOR_WALL_CLOCK
                freshness_meta["data_freshness_metric"] = (
                    FRESHNESS_METRIC_LAST_COMPLETED_BAR_AGE
                )
            if diagnostics is not None:
                diagnostics["freshness"] = freshness_meta
            if not sanity_check:
                break
            if bool(freshness_meta.get("last_bar_within_policy_window")):
                stale_last_t = None
                stale_forming_t = None
                break
            if _relax_live_completed_bar_freshness(
                symbol=symbol,
                rates=rates,
                timeframe=timeframe,
                expected_end_ts=expected_end_ts,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                freshness_meta=freshness_meta,
            ):
                stale_last_t = None
                stale_forming_t = None
                break
            if _session_break_explains_latest_n_freshness(
                symbol=symbol,
                timeframe=timeframe,
                last_completed_epoch=last_completed_epoch,
                last_completed_open=last_completed_open,
                freshness_cutoff=freshness_cutoff,
                next_bar_open_epoch=float(last_t) if tail_is_forming else None,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                freshness_meta=freshness_meta,
            ):
                stale_last_t = None
                stale_forming_t = None
                break
            stale_last_t = last_completed_open
            stale_forming_t = float(last_t) if tail_is_forming else None
        if retry and idx < (attempts - 1):
            time.sleep(FETCH_RETRY_DELAY)
    if (
        sanity_check
        and stale_last_t is not None
        and freshness_cutoff is not None
        and rates is not None
        and len(rates) > 0
    ):
        message = (
            f"Data appears stale for {symbol} {timeframe}: latest completed bar is "
            f"from {_format_time_minimal(stale_last_t)}. Market may be closed; "
            "set allow_stale=true to retrieve the latest "
            "available completed historical bars."
        )
        if stale_forming_t is not None:
            message += (
                f" A forming bar at {_format_time_minimal(stale_forming_t)} was "
                "observed and skipped; pass include_incomplete=true to include it."
            )
        return None, message
    return rates, None


def _mt5_history_start_with_warmup(
    start: datetime,
    warmup_seconds: int,
) -> datetime:
    """Subtract warmup without crossing MT5's Unix-time query boundary."""
    if start <= _MT5_HISTORY_QUERY_MIN:
        return _MT5_HISTORY_QUERY_MIN
    available_seconds = (start - _MT5_HISTORY_QUERY_MIN).total_seconds()
    bounded_warmup = min(max(0.0, float(warmup_seconds)), available_seconds)
    return start - timedelta(seconds=bounded_warmup)


def _resolve_fetch_timeframe_seconds(timeframe: TimeframeLiteral) -> tuple[Optional[int], Optional[str]]:
    seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe)
    if not seconds_per_bar:
        return None, f"Unable to determine timeframe seconds for {timeframe}"
    return int(seconds_per_bar), None


def _collect_candle_time_alignment(
    symbol: str,
    *,
    timeframe: TimeframeLiteral,
    start_datetime: Optional[str],
    end_datetime: Optional[str],
) -> Optional[Dict[str, Any]]:
    broker_time_check_enabled = bool(
        getattr(mt5_config, "broker_time_check_enabled", False)
    )
    if not broker_time_check_enabled or start_datetime or end_datetime:
        return None
    broker_time_check_ttl_seconds = int(
        getattr(mt5_config, "broker_time_check_ttl_seconds", 60) or 60
    )
    probe_timeframe = "M1" if timeframe != "M1" else timeframe
    try:
        return get_cached_mt5_time_alignment(
            symbol=symbol,
            probe_timeframe=probe_timeframe,
            ttl_seconds=broker_time_check_ttl_seconds,
        )
    except Exception as exc:
        return {
            "symbol": str(symbol),
            "probe_timeframe": probe_timeframe,
            "status": "unavailable",
            "reason": "inspection_failed",
            "error": str(exc),
        }


def _format_rate_times(epoch_series: pd.Series, *, use_client_tz: bool) -> pd.Series:
    epochs = pd.to_numeric(epoch_series, errors="coerce")
    dt_series = pd.to_datetime(epochs, unit="s", utc=True, errors="coerce")

    if use_client_tz:
        try:
            target_tz = mt5_config.get_client_tz()
            if target_tz is None:
                target_tz = datetime.now().astimezone().tzinfo
            if target_tz is not None:
                dt_series = dt_series.dt.tz_convert(target_tz)
        except Exception:
            pass

    formatted = dt_series.map(
        lambda value: (
            _format_datetime_minute_explicit(value.to_pydatetime())
            if pd.notna(value)
            else None
        )
    )
    if bool(formatted.isna().any()):
        formatter = _format_time_minimal_local if use_client_tz else _format_time_minimal
        fallback = epochs.map(lambda value: formatter(float(value)) if pd.notna(value) else None)
        formatted = formatted.where(~formatted.isna(), fallback)
    return formatted


def _build_rates_df(rates: Any, use_client_tz: bool) -> pd.DataFrame:
    """Normalize raw MT5 rates into a DataFrame with epoch and display time columns."""
    df = _rates_to_df(rates)
    df['__epoch'] = df['time']
    df["time"] = _format_rate_times(df["time"], use_client_tz=use_client_tz)
    if 'volume' not in df.columns and 'tick_volume' in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df['volume'] = df['tick_volume']
    return df


def _trim_df_to_target(
    df: pd.DataFrame,
    start_datetime: Optional[str],
    end_datetime: Optional[str],
    candles: int,
    *,
    copy_rows: bool = True,
    timeframe: Optional[str] = None,
    range_selection: Optional[str] = None,
) -> pd.DataFrame:
    keep_latest = str(range_selection or "").strip().lower() == "last_n"
    if timeframe in CALENDAR_TIMEFRAMES and (
        _is_calendar_query_bound(start_datetime)
        or _is_calendar_query_bound(end_datetime)
    ):
        out = _trim_calendar_bars_to_session_dates(
            df,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            timeframe=str(timeframe),
        )
        if start_datetime and not end_datetime and len(out) > candles:
            out = out.iloc[-candles:] if keep_latest else out.iloc[:candles]
        elif end_datetime and not start_datetime and len(out) > candles:
            out = out.iloc[-candles:]
        return out.copy() if copy_rows else out
    if start_datetime and end_datetime:
        from_dt = _parse_start_datetime(start_datetime)
        to_dt = _parse_end_datetime(end_datetime)
        if not from_dt or not to_dt:
            out = df.iloc[0:0]
            return out.copy() if copy_rows else out
        target_from = _utc_epoch_seconds(from_dt)
        target_to = _utc_epoch_seconds(to_dt)
        end_epochs = df["__epoch"]
        if timeframe and not _is_calendar_query_bound(end_datetime):
            end_epochs = df["__epoch"].map(
                lambda value: bar_close_epoch(float(value), str(timeframe))
            )
        out = df.loc[
            (df["__epoch"] >= target_from) & (end_epochs <= target_to)
        ]
    elif start_datetime:
        from_dt = _parse_start_datetime(start_datetime)
        if not from_dt:
            out = df.iloc[0:0]
            return out.copy() if copy_rows else out
        target_from = _utc_epoch_seconds(from_dt)
        out = df.loc[df['__epoch'] >= target_from]
        if len(out) > candles:
            out = out.iloc[-candles:] if keep_latest else out.iloc[:candles]
    elif end_datetime:
        to_dt = _parse_end_datetime(end_datetime)
        if not to_dt:
            out = df.iloc[0:0]
            return out.copy() if copy_rows else out
        target_to = _utc_epoch_seconds(to_dt)
        end_epochs = df["__epoch"]
        if timeframe and not _is_calendar_query_bound(end_datetime):
            end_epochs = df["__epoch"].map(
                lambda value: bar_close_epoch(float(value), str(timeframe))
            )
        out = df.loc[end_epochs <= target_to]
        if len(out) > candles:
            out = out.iloc[-candles:]
    else:
        out = df.iloc[-candles:] if len(df) > candles else df
    return out.copy() if copy_rows else out


def _next_calendar_period_date(value: Any, timeframe: str):
    if timeframe == "D1":
        return value + timedelta(days=1)
    if timeframe == "W1":
        return value + timedelta(days=7)
    if value.month == 12:
        return value.replace(year=value.year + 1, month=1, day=1)
    return value.replace(month=value.month + 1, day=1)


def _trim_calendar_bars_to_session_dates(
    df: pd.DataFrame,
    *,
    start_datetime: Optional[str],
    end_datetime: Optional[str],
    timeframe: str,
) -> pd.DataFrame:
    """Match date-only D1/W1/MN1 bounds to broker calendar periods.

    MT5 stamps these bars at the broker session open, which commonly falls on
    the previous UTC date. Instant-bearing bounds retain their UTC semantics;
    ISO date-only and natural calendar-period bounds use this overlap rule.
    """
    broker_tz = _broker_calendar_timezone()
    if broker_tz is None:
        return df.iloc[0:0]

    mask = pd.Series(True, index=df.index, dtype=bool)
    epoch_column = "__epoch" if "__epoch" in df.columns else "time"
    session_dates = df[epoch_column].map(
        lambda epoch: datetime.fromtimestamp(
            float(epoch), tz=dt_timezone.utc
        ).astimezone(broker_tz).date()
    )
    period_end_dates = session_dates.map(
        lambda value: _next_calendar_period_date(value, timeframe)
    )

    if start_datetime:
        if _is_calendar_query_bound(start_datetime):
            requested_start_bound = _parse_candle_calendar_bound(
                start_datetime,
                timeframe=timeframe,
                end_bound=False,
            )
            if requested_start_bound is None:
                return df.iloc[0:0]
            requested_start = requested_start_bound.astimezone(broker_tz).date()
            mask &= period_end_dates > requested_start
        else:
            parsed_start = _parse_start_datetime(start_datetime)
            if parsed_start is None:
                return df.iloc[0:0]
            mask &= df[epoch_column] >= _utc_epoch_seconds(parsed_start)
    if end_datetime:
        if _is_calendar_query_bound(end_datetime):
            requested_end_bound = _parse_candle_calendar_bound(
                end_datetime,
                timeframe=timeframe,
                end_bound=True,
            )
            if requested_end_bound is None:
                return df.iloc[0:0]
            requested_end = requested_end_bound.astimezone(broker_tz).date()
            mask &= session_dates <= requested_end
        else:
            parsed_end = _parse_end_datetime(end_datetime)
            if parsed_end is None:
                return df.iloc[0:0]
            target_end = _utc_epoch_seconds(parsed_end)
            close_epochs = df[epoch_column].map(
                lambda value: bar_close_epoch(float(value), timeframe)
            )
            mask &= close_epochs <= target_end
    return df.loc[mask]


def _normalize_indicator_spec(indicators: Optional[List[IndicatorSpec]]) -> Optional[str]:
    """Normalize indicator input into the compact internal string format."""
    if indicators is None:
        return None

    source: Any = indicators
    if isinstance(source, str):
        payload = source.strip()
        if payload.startswith('[') or payload.startswith('{'):
            try:
                source = json.loads(payload)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid indicator JSON: {exc}") from exc

    if isinstance(source, (list, tuple)):
        parts: List[str] = []
        for item in source:
            if isinstance(item, dict) and 'name' in item:
                name = str(item.get('name'))
                params = item.get('params') or []
                if isinstance(params, (list, tuple)) and len(params) > 0:
                    args_str = ",".join(str(coerce_scalar(str(param))) for param in params)
                    parts.append(f"{name}({args_str})")
                elif isinstance(params, dict) and len(params) > 0:
                    args_str = ",".join(
                        f"{str(key).strip()}={coerce_scalar(str(param))}"
                        for key, param in params.items()
                        if str(key).strip()
                    )
                    parts.append(f"{name}({args_str})" if args_str else name)
                else:
                    parts.append(name)
            else:
                parts.append(str(item))
        return ",".join(parts)

    return str(source)


def _normalize_indicator_spec_for_display(ti_spec: Optional[str]) -> str:
    text = str(ti_spec or "").strip()
    if not text:
        return ""
    return re.sub(r"(?<![\d.])([+-]?\d+)\.0(?!\d)", r"\1", text)


def _display_indicator_column_name(column: str) -> str:
    text = str(column or "")
    if text.startswith("ATRr_"):
        text = "ATR_" + text[len("ATRr_") :]
    text = re.sub(r"^MACD([a-z])_", r"MACD_\1_", text)
    text = _normalize_indicator_spec_for_display(text)
    return text.replace(".", "_").lower()


def _normalize_indicator_columns_for_display(
    df: pd.DataFrame,
    columns: List[str],
) -> List[str]:
    if not columns:
        return []

    rename_map: Dict[str, str] = {}
    normalized: List[str] = []
    for column in columns:
        old_name = str(column)
        new_name = _display_indicator_column_name(old_name)
        normalized_name = old_name
        if new_name != old_name:
            if old_name in df.columns and new_name not in df.columns and new_name not in rename_map.values():
                rename_map[old_name] = new_name
                normalized_name = new_name
        normalized.append(normalized_name)

    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    return normalized


def _extend_unique_headers(headers: List[str], columns: List[str]) -> None:
    for column in columns:
        if column not in headers:
            headers.append(column)


def _build_candle_headers(
    rates: Any,
    ohlcv: Optional[str],
    *,
    include_spread: bool = False,
) -> List[str]:
    """Build the initial candle header set before transforms add derived columns."""
    def _volume_values(field: str) -> List[int]:
        values: List[int] = []
        for rate in rates:
            try:
                value = rate[field]
            except (IndexError, KeyError, TypeError, ValueError):
                value = 0
            try:
                values.append(int(value))
            except (TypeError, ValueError, OverflowError):
                values.append(0)
        return values

    tick_volumes = _volume_values("tick_volume")
    real_volumes = _volume_values("real_volume")

    has_tick_volume = len(set(tick_volumes)) > 1 or any(value != 0 for value in tick_volumes)
    has_real_volume = len(set(real_volumes)) > 1 or any(value != 0 for value in real_volumes)
    requested = _normalize_ohlcv_arg(ohlcv)

    headers = ["time"]
    if requested is not None:
        if "O" in requested:
            headers.append("open")
        if "H" in requested:
            headers.append("high")
        if "L" in requested:
            headers.append("low")
        if "C" in requested:
            headers.append("close")
        if "V" in requested:
            headers.append("tick_volume")
        if include_spread:
            headers.append("spread_points")
        return headers

    headers.extend(["open", "high", "low", "close"])
    if has_tick_volume:
        headers.append("tick_volume")
    if has_real_volume:
        headers.append("real_volume")
    if include_spread:
        headers.append("spread_points")
    return headers


def _candle_volume_metadata(headers: List[str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    units: Dict[str, str] = {}
    if "tick_volume" in headers:
        meta["volume_type"] = TICK_VOLUME_UNIT
        meta["volume_note"] = (
            "MT5 tick_volume counts bid-price updates for the bar, not every "
            "quote update and not exchange traded volume."
        )
        meta["tick_volume_event_basis"] = TICK_VOLUME_EVENT_BASIS
        meta["tick_volume_tape_equivalent"] = TICK_VOLUME_TAPE_EQUIVALENT
        meta["tick_volume_comparison_note"] = TICK_VOLUME_COMPARISON_NOTE
        units["tick_volume"] = TICK_VOLUME_UNIT
    if "real_volume" in headers:
        meta["real_volume_type"] = "traded_volume"
        units["real_volume"] = "traded_volume"
    if "spread" in headers:
        units["spread"] = "absolute_price"
    if "spread_points" in headers:
        units["spread_points"] = "broker_points"
    if units:
        meta["units"] = units
    return meta


def _normalize_candle_spread_columns(
    df: pd.DataFrame,
    headers: List[str],
    *,
    price_point: Optional[float],
) -> None:
    """Expose MT5 candle spread with explicit price and point units."""
    if "spread_points" not in df.columns and "spread" in df.columns:
        df.rename(columns={"spread": "spread_points"}, inplace=True)
    if "spread_points" not in df.columns:
        return
    spread_points = pd.to_numeric(df["spread_points"], errors="coerce")
    spread_available = spread_points.notna() & spread_points.gt(0.0)
    df["spread_available"] = spread_available
    df["spread_points"] = spread_points.astype(object).where(
        spread_available,
        None,
    )
    if "spread_points" not in headers:
        headers.append("spread_points")
    if "spread_available" not in headers:
        headers.append("spread_available")
    if price_point is None or price_point <= 0.0:
        return
    df["spread"] = (spread_points * float(price_point)).astype(object).where(
        spread_available,
        None,
    )
    if "spread" not in headers:
        headers.insert(headers.index("spread_points"), "spread")


def _candle_time_convention_metadata(timeframe: str) -> Dict[str, str]:
    tf = str(timeframe or "").strip().upper()
    if tf in CALENDAR_TIMEFRAMES:
        return {
            "bar_time_convention": "bar_open_time",
            "bar_time_note": (
                "MT5 daily, weekly, and monthly candle time is the broker/server "
                "bar open time; it may not be UTC midnight."
            ),
        }
    return {"bar_time_convention": "bar_open_time"}


def _validate_ohlcv_selection(ohlcv: Optional[str]) -> Optional[str]:
    if ohlcv is None or str(ohlcv).strip() == "":
        return None
    try:
        if _normalize_ohlcv_arg(ohlcv) is not None:
            return None
    except ValueError as exc:
        return str(exc)
    return (
        "Invalid ohlcv value. Use all, ohlcv, ohlc, close/price, compact "
        "letters from o/h/l/c/v, or comma-separated names such as "
        "open,high,low,close,volume."
    )


def _append_denoise_application(
    denoise_apps: List[Dict[str, Any]],
    source_spec: Any,
    *,
    default_when: str,
    default_causality: str,
    default_keep_original: bool,
    added_columns: List[str],
    overwritten_columns: List[str],
    ohlc_geometry_repaired: int = 0,
) -> None:
    if not added_columns and not overwritten_columns:
        return
    try:
        denoise_meta = dict(source_spec or {})
        columns = denoise_meta.get('columns', 'close')
        keep_original = bool(denoise_meta.get('keep_original', default_keep_original))
        application = {
            'method': str(denoise_meta.get('method', 'none')).lower(),
            'when': str(denoise_meta.get('when', default_when)).lower(),
            'causality': str(denoise_meta.get('causality', default_causality)),
            'keep_original': keep_original,
            'columns': columns,
            'params': denoise_meta.get('params') or {},
            'added_columns': added_columns,
            'overwrote_columns': overwritten_columns,
        }
        if ohlc_geometry_repaired > 0:
            application['ohlc_geometry_repaired'] = int(ohlc_geometry_repaired)
        denoise_apps.append(application)
    except Exception:
        pass


def _denoise_history_context(
    denoise: Optional[DenoiseSpec],
) -> Optional[Dict[str, Any]]:
    """Return method-specific history needed for stable denoise output."""
    normalized = _normalize_denoise_spec(denoise, default_when="pre_ti")
    if not normalized:
        return None
    method = str(normalized.get("method") or "none").strip().lower()
    params = dict(normalized.get("params") or {})
    causality = str(normalized.get("causality") or "causal").strip().lower()
    context: Dict[str, Any] = {
        "method": method,
        "causality": causality,
        "warmup_bars": 0,
    }
    if method == "ema":
        alpha = ema_alpha(params)
        tolerance = 1e-8
        settling_bars = math.ceil(math.log(tolerance) / math.log1p(-alpha)) if alpha < 1 else 0
        context.update({
            "alpha": alpha,
            "seed_weight_tolerance": tolerance,
            "initialization": "first_fetched_value",
            "warmup_bars": min(settling_bars, 100_000),
            "recommended_bars": settling_bars + 1,
        })
    elif method == "butterworth" and causality == "zero_phase":
        order = max(1, int(params.get("order", 4)))
        btype = str(params.get("btype") or "low").strip().lower()
        coefficient_count = (
            2 * order + 1
            if btype in {"band", "bandpass", "bandstop"}
            else order + 1
        )
        configured_padlen = params.get("padlen")
        padlen = (
            max(0, int(configured_padlen))
            if configured_padlen is not None
            else 3 * coefficient_count
        )
        context["minimum_bars"] = padlen + 1
        context["warmup_bars"] = padlen + 1
    elif method in {"kalman", "kalman_robust"}:
        context["recommended_bars"] = 20
        context["warmup_bars"] = 20
    elif method in {"savgol", "kama", "preaverage"}:
        window = max(2, int(params.get("window", 10)))
        context["warmup_bars"] = window
        context["recommended_bars"] = window
    elif method == "supersmoother":
        period = max(2, int(params.get("period", 10)))
        context["warmup_bars"] = period
        context["recommended_bars"] = period
    return context


def _latest_indicator_values_missing(df: pd.DataFrame, columns: List[str]) -> bool:
    required_columns = _indicator_columns_required_for_completeness(columns)
    if not required_columns or len(df) <= 0:
        return False
    for column in required_columns:
        if column not in df.columns:
            return True
        value = df[column].iloc[-1]
        try:
            if pd.isna(value):
                return True
        except Exception:
            if value is None:
                return True
    return False


def _apply_stage_denoise(
    df: pd.DataFrame,
    headers: List[str],
    denoise: Optional[DenoiseSpec],
    denoise_apps: List[Dict[str, Any]],
    *,
    when: str,
    require_explicit_when: bool = False,
) -> None:
    if not denoise:
        return
    if require_explicit_when and not (
        isinstance(denoise, dict) and denoise.get("when") not in (None, "")
    ):
        return

    normalized = _normalize_denoise_spec(denoise, default_when=when)
    if not normalized or str(normalized.get("when", when)).lower() != when:
        return
    added_columns = apply_denoise_util(df, normalized)
    last_application = df.attrs.get("denoise_last_application")
    overwritten_columns = (
        list(last_application.get("overwrote_columns") or [])
        if isinstance(last_application, dict)
        else []
    )
    ohlc_geometry_repaired = (
        int(last_application.get("ohlc_geometry_repaired") or 0)
        if isinstance(last_application, dict)
        else 0
    )
    _extend_unique_headers(headers, added_columns)
    _append_denoise_application(
        denoise_apps,
        normalized,
        default_when=when,
        default_causality="causal",
        default_keep_original=True,
        added_columns=added_columns,
        overwritten_columns=overwritten_columns,
        ohlc_geometry_repaired=ohlc_geometry_repaired,
    )


def _apply_indicator_stage(
    df: pd.DataFrame,
    headers: List[str],
    ti_spec: Optional[str],
    denoise: Optional[DenoiseSpec],
) -> List[str]:
    ti_cols: List[str] = []
    if not ti_spec:
        return ti_cols

    # A pre-TI denoise now preserves canonical broker OHLC by default. Feed
    # indicators the suffixed denoised series without allowing the indicator
    # implementation to mistake the raw columns for its intended inputs.
    suffix = str((denoise or {}).get("suffix") or "_dn")
    denoised_sources = {
        column: df[f"{column}{suffix}"].copy()
        for column in ("open", "high", "low", "close", "volume", "tick_volume")
        if column in df.columns and f"{column}{suffix}" in df.columns
    }
    original_sources = {
        column: df[column].copy() for column in denoised_sources
    }
    try:
        for column, values in denoised_sources.items():
            df[column] = values
        columns_before = {str(column) for column in df.columns}
        reported_columns = [
            str(column) for column in _apply_ta_indicators(df, ti_spec)
        ]
    finally:
        for column, values in original_sources.items():
            df[column] = values
    created_columns = [
        str(column) for column in df.columns if str(column) not in columns_before
    ]
    ti_cols = list(dict.fromkeys([*reported_columns, *created_columns]))
    ti_cols = _normalize_indicator_columns_for_display(df, ti_cols)
    price_inputs_denoised = any(
        column in denoised_sources for column in ("open", "high", "low", "close")
    )
    if price_inputs_denoised:
        rename_map: Dict[str, str] = {}
        renamed_columns: List[str] = []
        for column in ti_cols:
            denoised_name = (
                column if column.endswith(suffix) else f"{column}{suffix}"
            )
            if (
                denoised_name != column
                and column in df.columns
                and denoised_name not in df.columns
                and denoised_name not in rename_map.values()
            ):
                rename_map[column] = denoised_name
                renamed_columns.append(denoised_name)
            else:
                renamed_columns.append(column)
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
        ti_cols = renamed_columns
    _extend_unique_headers(headers, ti_cols)

    if denoise and ti_cols:
        dn_base = _normalize_denoise_spec(denoise, default_when='post_ti')
        if dn_base and bool(dn_base.get('apply_to_ti') or dn_base.get('ti')):
            dn_ti = dict(dn_base)
            dn_ti['columns'] = list(ti_cols)
            dn_ti.setdefault('when', 'post_ti')
            dn_ti.setdefault('keep_original', False)
            apply_denoise_util(df, dn_ti)

    return ti_cols


def _indicator_columns_required_for_completeness(columns: List[str]) -> List[str]:
    """Return indicator columns that must be populated on every output row.

    pandas-ta-classic's Supertrend long and short bands are regime-specific:
    ``SUPERTl`` is null in short regimes and ``SUPERTs`` is null in long regimes.
    Those nulls express which band is inactive, rather than a warmup failure.
    """
    required: List[str] = []
    for column in columns:
        family = str(column or "").split("_", 1)[0].lower()
        if family in {"supertl", "superts"}:
            continue
        required.append(column)
    return required


def _indicator_columns_with_missing_values(
    df: pd.DataFrame,
    ti_cols: List[str],
) -> List[str]:
    missing_cols: List[str] = []
    for col in _indicator_columns_required_for_completeness(ti_cols):
        if col not in df.columns:
            continue
        try:
            if bool(df[col].isna().any()):
                missing_cols.append(str(col))
        except Exception:
            continue
    return missing_cols


def _drop_incomplete_indicator_rows(
    df: pd.DataFrame,
    ti_cols: List[str],
) -> Tuple[pd.DataFrame, int, List[str]]:
    required_cols = _indicator_columns_required_for_completeness(ti_cols)
    existing_cols = [col for col in required_cols if col in df.columns]
    if not existing_cols or len(df) == 0:
        return df, 0, []

    missing_cols = _indicator_columns_with_missing_values(df, ti_cols)
    if not missing_cols:
        return df, 0, []

    missing_mask = df[existing_cols].isna().any(axis=1)
    dropped_rows = int(missing_mask.sum())
    if dropped_rows <= 0:
        return df, 0, []

    return df.loc[~missing_mask].copy(), dropped_rows, missing_cols


def _rebuild_candle_indicator_window(
    rates: Any,
    *,
    use_client_tz: bool,
    denoise: Optional[DenoiseSpec],
    ti_spec: Optional[str],
    headers: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Rebuild the warmup window and re-run the pre-indicator stages."""
    df = _build_rates_df(rates, use_client_tz)
    if denoise:
        normalized = _normalize_denoise_spec(denoise, default_when='pre_ti')
        if normalized and str(normalized.get('when', 'pre_ti')).lower() == 'pre_ti':
            apply_denoise_util(df, normalized)
    ti_cols = _apply_indicator_stage(df, headers, ti_spec, denoise)
    return df, ti_cols


def _describe_session_gap(
    previous_epoch: float,
    current_epoch: float,
    *,
    expected_bar_seconds: float,
    use_client_tz: bool,
) -> Optional[Dict[str, Any]]:
    """Describe a discontinuity between two observed broker bars."""
    prev_t = float(previous_epoch)
    curr_t = float(current_epoch)
    if not (math.isfinite(prev_t) and math.isfinite(curr_t)):
        return None
    gap_seconds = float(curr_t - prev_t)
    if expected_bar_seconds <= 0 or gap_seconds <= expected_bar_seconds * 1.5:
        return None

    if use_client_tz:
        from_disp = _format_time_minimal_local(prev_t)
        to_disp = _format_time_minimal_local(curr_t)
    else:
        from_disp = _format_time_minimal(prev_t)
        to_disp = _format_time_minimal(curr_t)

    missing_bars_est = max(1, int(round(gap_seconds / expected_bar_seconds)) - 1)
    prev_dt = datetime.fromtimestamp(prev_t, tz=dt_timezone.utc)
    curr_dt = datetime.fromtimestamp(curr_t, tz=dt_timezone.utc)
    crosses_weekend = (
        prev_dt.weekday() >= 5
        or curr_dt.weekday() >= 5
        or gap_seconds >= (36.0 * 3600.0)
    )
    return {
        "from": from_disp,
        "to": to_disp,
        "gap_seconds": gap_seconds,
        "expected_bar_seconds": expected_bar_seconds,
        "missing_bars_est": int(missing_bars_est),
        "context": "weekend/session break" if crosses_weekend else "session break",
    }


def _collect_session_gaps(
    df: pd.DataFrame,
    *,
    timeframe: TimeframeLiteral,
    use_client_tz: bool,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    session_gaps: List[Dict[str, Any]] = []
    expected_bar_seconds = float(TIMEFRAME_SECONDS.get(timeframe, 0) or 0)
    if expected_bar_seconds <= 0 or '__epoch' not in df.columns or len(df) <= 1:
        return session_gaps, None

    try:
        epochs = pd.to_numeric(df['__epoch'], errors='coerce').to_numpy(dtype=float)
        for index in range(1, len(epochs)):
            gap = _describe_session_gap(
                float(epochs[index - 1]),
                float(epochs[index]),
                expected_bar_seconds=expected_bar_seconds,
                use_client_tz=use_client_tz,
            )
            if gap is not None:
                session_gaps.append(gap)
    except Exception as exc:
        logger.warning("Session gap diagnostics unavailable: %s", exc)
        return session_gaps, "Session gap diagnostics unavailable."

    return session_gaps, None


def _spacing_interval_measurements(
    epochs: pd.Series,
    *,
    expected: float,
) -> Optional[tuple[pd.Series, float, float]]:
    """Return positive interval diffs plus median and matching-interval percent."""
    diffs = epochs.diff().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return None
    median_seconds = float(diffs.median())
    matching_pct = round(
        float(diffs.between(expected * 0.75, expected * 1.5).mean()) * 100.0,
        2,
    )
    return diffs, median_seconds, matching_pct


def _candle_spacing_quality(
    df: pd.DataFrame,
    *,
    timeframe: TimeframeLiteral,
) -> Optional[Dict[str, Any]]:
    """Describe whether the dominant spacing resembles the requested timeframe."""
    expected = float(TIMEFRAME_SECONDS.get(timeframe, 0) or 0)
    if expected <= 0 or "__epoch" not in df.columns or df.empty:
        return None
    measured = _spacing_interval_measurements(
        pd.to_numeric(df["__epoch"], errors="coerce"),
        expected=expected,
    )
    if measured is None:
        return {
            "requested_bar_seconds": expected,
            "observed_median_bar_seconds": None,
            "intervals_checked": 0,
            "matching_interval_pct": None,
            "spacing_matches_timeframe": None,
            "status": "insufficient_sample",
        }
    diffs, median_seconds, matching_pct = measured
    spacing_matches = not (
        median_seconds > expected * 1.5 and matching_pct < 20.0
    )
    return {
        "requested_bar_seconds": expected,
        "observed_median_bar_seconds": round(median_seconds, 3),
        "intervals_checked": int(len(diffs)),
        "matching_interval_pct": matching_pct,
        "spacing_matches_timeframe": spacing_matches,
        "status": "ok" if spacing_matches else "timeframe_mismatch",
    }


def _annotate_candle_gap_rows(
    payload: Dict[str, Any],
    session_gaps: List[Dict[str, Any]],
) -> None:
    rows = payload.get("data")
    if not isinstance(rows, list) or not session_gaps:
        return
    gaps_by_to = {
        str(gap.get("to")): {
            "gap_seconds": gap.get("gap_seconds"),
            "missing_bars_est": gap.get("missing_bars_est"),
            "context": gap.get("context"),
        }
        for gap in session_gaps
        if isinstance(gap, dict) and gap.get("to") not in (None, "")
    }
    if not gaps_by_to:
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        gap = gaps_by_to.get(str(row.get("time")))
        if gap:
            row["gap_before"] = {
                key: value for key, value in gap.items() if value not in (None, "")
            }


def _format_candle_times(
    df: pd.DataFrame,
    headers: List[str],
    *,
    time_as_epoch: bool,
    use_client_tz: bool,
    client_tz: Any,
) -> None:
    if 'time' not in headers or len(df) <= 0:
        return

    epochs = pd.to_numeric(df['__epoch'], errors='coerce').astype(float)
    if time_as_epoch:
        df['time'] = epochs
        df.attrs['_tz_used_name'] = 'UTC'
        return

    tz_used_name = 'UTC'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        time_values = pd.to_datetime(epochs, unit='s', utc=True)
        if use_client_tz:
            tz_used_name = getattr(client_tz, 'zone', None) or str(client_tz)
            time_values = time_values.dt.tz_convert(client_tz)
        df['time'] = time_values.map(lambda value: _format_datetime_minute_explicit(value.to_pydatetime()))
    df.attrs['_tz_used_name'] = tz_used_name


def _public_simplify_meta(meta: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(meta, dict):
        return None
    out: Dict[str, Any] = {}
    for key in (
        "method",
        "mode",
        "points",
        "ratio",
        "non_ohlc_numeric_aggregation",
        "segment_mean_columns",
        "original_rows",
        "returned_rows",
    ):
        value = meta.get(key)
        if value is not None:
            out[key] = value
    return out or None


def _history_spacing_quality(
    df: pd.DataFrame,
    *,
    timeframe: str,
) -> Optional[Dict[str, Any]]:
    """Detect a materially coarser provider cadence before analysis."""
    expected = float(TIMEFRAME_SECONDS.get(str(timeframe).upper(), 0) or 0)
    if expected <= 0 or "time" not in df.columns or len(df) < 6:
        return None
    measured = _spacing_interval_measurements(
        pd.to_numeric(df["time"], errors="coerce"),
        expected=expected,
    )
    if measured is None or len(measured[0]) < 5:
        return None
    diffs, median_seconds, matching_pct = measured
    return {
        "requested_bar_seconds": int(expected),
        "observed_median_bar_seconds": round(median_seconds, 3),
        "matching_interval_pct": matching_pct,
        "intervals_checked": int(len(diffs)),
        "spacing_matches_timeframe": not (
            median_seconds > expected * 1.5 and matching_pct < 20.0
        ),
    }


def fetch_history_frame(
    symbol: str,
    timeframe: str,
    count: int,
    as_of: Optional[str] = None,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    include_incomplete: bool = False,
    retry: bool = True,
) -> pd.DataFrame:
    """Return analysis-ready MT5 candles with native UTC epoch timestamps.

    History-quality diagnostics and any cleanup warnings are attached to the
    returned frame's ``attrs`` without changing the DataFrame return contract.
    """
    if timeframe not in TIMEFRAME_MAP:
        raise RuntimeError(f"Invalid timeframe: {timeframe}")
    if as_of and (start or end):
        raise RuntimeError("as_of cannot be combined with start/end.")
    requested_count = max(1, int(count))
    if as_of:
        parsed_as_of = _parse_start_datetime(as_of)
        if parsed_as_of is None:
            raise RuntimeError("Invalid as_of time.")
        parsed_as_of = (
            parsed_as_of.replace(tzinfo=dt_timezone.utc)
            if parsed_as_of.tzinfo is None
            else parsed_as_of.astimezone(dt_timezone.utc)
        )
        if parsed_as_of.timestamp() > time.time() + 1.0:
            raise RuntimeError("as_of must not be in the future.")

    resolved_symbol = resolve_broker_symbol_name(symbol)
    resolved_end = as_of or end
    parsed_start = parsed_end = None
    for value, end_bound in ((start, False), (resolved_end, True)):
        if not value:
            continue
        parsed, bound_error = _parse_fetch_datetime_arg(value, timeframe=timeframe, end_bound=end_bound)
        if bound_error or parsed is None:
            raise RuntimeError(bound_error or "Invalid history time bound.")
        if end_bound:
            parsed_end = parsed
            resolved_end = parsed.isoformat().replace("+00:00", "Z")
        else:
            parsed_start = parsed
            start = parsed.isoformat().replace("+00:00", "Z")
    fetch_count = requested_count
    if parsed_start is not None and parsed_end is not None:
        span_seconds = max(0.0, (parsed_end - parsed_start).total_seconds())
        seconds_per_bar = int(TIMEFRAME_SECONDS[timeframe])
        fetch_count = max(
            requested_count,
            int(math.ceil(span_seconds / max(1, seconds_per_bar))) + 2,
        )

    info_before = get_symbol_info_cached(resolved_symbol)
    with _symbol_ready_guard(resolved_symbol, info_before=info_before) as (error, _info):
        if error:
            raise RuntimeError(error)
        rates, rates_error = _fetch_rates_with_warmup(
            resolved_symbol,
            TIMEFRAME_MAP[timeframe],
            timeframe,
            fetch_count,
            0,
            start,
            resolved_end,
            include_incomplete=include_incomplete,
            retry=retry,
            sanity_check=False,
        )
    if rates_error:
        if isinstance(rates_error, dict):
            raise RuntimeError(str(rates_error.get("error") or "data_shape_invalid"))
        raise RuntimeError(rates_error)
    if rates is None:
        raise RuntimeError(
            _describe_rate_fetch_error(resolved_symbol, info_before=info_before)
        )
    if len(rates) < 1:
        raise ValueError(
            f"No data is available for {resolved_symbol} {timeframe} in the "
            "requested range. Widen or correct the historical range."
        )

    df = _rates_to_df(rates)
    if "volume" not in df.columns and "tick_volume" in df.columns:
        df["volume"] = df["tick_volume"]
    rows_before_quality = int(len(df))
    df, history_warnings = validate_and_clean_ohlcv_frame(df, epoch_col="time")
    rows_after_quality = int(len(df))

    if parsed_start is not None:
        df = df.loc[df["time"] >= parsed_start.timestamp()]
    if parsed_end is not None:
        cutoff = min(parsed_end.timestamp(), time.time())
        if include_incomplete:
            df = df.loc[df["time"] <= cutoff]
        else:
            df = df.loc[df["time"].map(lambda value: bar_close_epoch(value, timeframe) <= cutoff)]

    if not include_incomplete and not resolved_end:
        df, _trimmed = _drop_incomplete_tail_df(df, timeframe)

    if start and not resolved_end and len(df) > requested_count:
        df = df.iloc[:requested_count]
    elif not start and len(df) > requested_count:
        df = df.iloc[-requested_count:]
    df = df.reset_index(drop=True)

    spacing = _history_spacing_quality(df, timeframe=timeframe)
    if spacing is not None and not spacing["spacing_matches_timeframe"]:
        raise RuntimeError(
            "Observed candle cadence does not match the requested timeframe: "
            f"requested_bar_seconds={spacing['requested_bar_seconds']}, "
            f"observed_median_bar_seconds={spacing['observed_median_bar_seconds']}, "
            f"matching_interval_pct={spacing['matching_interval_pct']}. "
            "The broker returned materially coarser history; retry with the "
            "observed timeframe or verify the symbol/timeframe feed."
        )
    history_quality = {
        "raw_bars_fetched": rows_before_quality,
        "bars_after_quality": rows_after_quality,
        "quality_rows_removed": max(0, rows_before_quality - rows_after_quality),
        "returned_bars": int(len(df)),
        "warnings": list(history_warnings),
    }
    df.attrs["history_quality"] = history_quality
    df.attrs["warnings"] = list(history_warnings)
    return df


def fetch_candles(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    limit: int = DEFAULT_ROW_LIMIT,
    start: Optional[str] = None,
    end: Optional[str] = None,
    ohlcv: Optional[str] = None,
    indicators: Optional[List[IndicatorSpec]] = None,
    denoise: Optional[DenoiseSpec] = None,
    simplify: Optional[SimplifySpec] = None,
    time_as_epoch: bool = False,
    force_utc: bool = False,
    *,
    include_spread: bool = False,
    include_incomplete: bool = False,
    allow_stale: bool = False,
    range_selection: Optional[str] = None,
) -> Dict[str, Any]:
    """Return historical candles as tabular data."""
    warmup_bars = 0
    ti_spec = None
    rate_fetch_diagnostics: Dict[str, Any] = {}
    try:
        if denoise:
            try:
                _normalize_denoise_spec(denoise, default_when="pre_ti")
            except DenoiseCausalityError as exc:
                return {
                    "success": False,
                    "error_code": "denoise_non_causal_requires_opt_in",
                    "error": (
                        f"Denoise method '{exc.method}' requires explicit zero-phase "
                        "opt-in because it uses future bars."
                    ),
                    "operation": "data_fetch_candles",
                    "remediation": (
                        "Set denoise causality to zero_phase only for retrospective "
                        "analysis (CLI: --denoise-params causality=zero_phase). "
                        "Do not use zero-phase output for live signals or forward tests."
                    ),
                    "details": {
                        "method": exc.method,
                        "required_causality": "zero_phase",
                        "uses_future_bars": True,
                    },
                }
        symbol = resolve_broker_symbol_name(symbol)
        query_started_at = time.perf_counter()
        # Backward/compat mappings to internal variable names used in implementation
        candles = int(limit)
        if candles <= 0:
            return {"error": "limit must be greater than 0."}
        start_datetime = start
        end_datetime = end
        ti = indicators
        # Validate timeframe using the shared map
        if timeframe not in TIMEFRAME_MAP:
            return {"error": invalid_timeframe_error(timeframe, TIMEFRAME_MAP)}
        mt5_timeframe = TIMEFRAME_MAP[timeframe]
        ohlcv_error = _validate_ohlcv_selection(ohlcv)
        if ohlcv_error is not None:
            return {"error": ohlcv_error}
        
        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = get_symbol_info_cached(symbol)
        with _symbol_ready_guard(symbol, info_before=_info_before) as (err, _info):
            if err:
                return {"error": err}
            price_digits = _symbol_price_digits(_info, _info_before)
            price_currency = _symbol_price_currency(_info, _info_before)
            price_point = _symbol_price_point(_info, _info_before)
            price_basis = _symbol_candle_price_basis(_info, _info_before)

            try:
                ti_spec = _normalize_indicator_spec(ti)
            except ValueError as exc:
                return _indicator_validation_error(str(exc), received=ti)
            indicator_syntax_error = _indicator_param_syntax_error(ti_spec)
            if indicator_syntax_error:
                return _indicator_validation_error(
                    indicator_syntax_error,
                    received=ti,
                )
            # Determine warmup bars if technical indicators requested
            unknown_indicators = _find_unknown_ta_indicators(ti_spec or "")
            if unknown_indicators:
                return build_error_payload(
                    (
                        "Unknown indicator(s): "
                        + ", ".join(unknown_indicators)
                        + ". Parameters use name(params) syntax, e.g. rsi(14) or "
                        "macd(12,26,9); use indicators_list to view valid indicator names."
                    ),
                    code="indicator_not_found",
                    operation="data_fetch_candles",
                    remediation=(
                        "Use indicators_list to inspect canonical names, then retry "
                        "--indicators with name(params) syntax."
                    ),
                    related_tools=["indicators_list"],
                    valid_values={"catalog": "indicators_list"},
                    example="rsi(14)",
                    details={"unknown_indicators": unknown_indicators},
                )
            indicator_warmup_bars = _estimate_warmup_bars(ti_spec)
            denoise_history_context = _denoise_history_context(denoise)
            denoise_warmup_bars = int(
                (denoise_history_context or {}).get("warmup_bars") or 0
            )
            warmup_bars = max(indicator_warmup_bars, denoise_warmup_bars)
            rate_fetch_diagnostics: Dict[str, Any] = {}
            freshness_diagnostics: Optional[Dict[str, Any]] = None
            historical_bounds_requested = bool(start_datetime or end_datetime)

            rates, rates_error = _fetch_rates_with_warmup(
                symbol,
                mt5_timeframe,
                timeframe,
                candles,
                warmup_bars,
                start_datetime,
                end_datetime,
                include_incomplete=include_incomplete,
                retry=True,
                sanity_check=not bool(allow_stale) and not historical_bounds_requested,
                diagnostics=rate_fetch_diagnostics,
                range_selection=range_selection,
            )
            freshness_diagnostics = rate_fetch_diagnostics.get("freshness")
            time_normalization = describe_mt5_time_normalization(symbol=symbol)
            if rates_error:
                if isinstance(rates_error, dict):
                    return rates_error
                error_payload: Dict[str, Any] = {"error": rates_error}
                if isinstance(freshness_diagnostics, dict):
                    error_payload["details"] = {
                        "diagnostics": {
                            "freshness": dict(freshness_diagnostics),
                        },
                    }
                return error_payload
        # visibility handled by _symbol_ready_guard
        
        if rates is None:
            return {"error": _describe_rate_fetch_error(symbol, info_before=_info_before)}

        # Generate tabular format with dynamic column filtering
        if len(rates) == 0:
            return _build_no_data_error_with_context(
                symbol, timeframe, mt5_timeframe, start_datetime, end_datetime
            )
        raw_bars_fetched = int(len(rates))
        live_bar_reference_epoch = _resolve_live_bar_reference_epoch(symbol, timeframe)
        # Requested bounds only clip the returned window. Bar completion is a
        # live fact and must never be advanced by a future range end.
        completion_reference_epoch = live_bar_reference_epoch
        initial_incomplete_trimmed = 0
        trailing_gap_epochs: Optional[Tuple[float, float]] = None
        if not include_incomplete:
            rates_before_trim = int(len(rates))
            if not historical_bounds_requested and rates_before_trim >= 2:
                try:
                    raw_previous_epoch = float(rates[-2]["time"])
                    raw_tail_epoch = float(rates[-1]["time"])
                    raw_gap = _describe_session_gap(
                        raw_previous_epoch,
                        raw_tail_epoch,
                        expected_bar_seconds=float(
                            TIMEFRAME_SECONDS.get(timeframe, 0) or 0
                        ),
                        use_client_tz=False,
                    )
                    if raw_gap is not None:
                        trailing_gap_epochs = (raw_previous_epoch, raw_tail_epoch)
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    IndexError,
                    OverflowError,
                    OSError,
                ):
                    trailing_gap_epochs = None
            rates = _drop_incomplete_tail(
                rates,
                timeframe,
                current_time_epoch=completion_reference_epoch,
            )
            initial_incomplete_trimmed = rates_before_trim - int(len(rates))
            if not initial_incomplete_trimmed:
                trailing_gap_epochs = None
        if len(rates) == 0:
            return _build_no_data_error_with_context(
                symbol, timeframe, mt5_timeframe, start_datetime, end_datetime
            )
        headers = _build_candle_headers(
            rates,
            ohlcv,
            include_spread=include_spread,
        )
        
        # Construct DataFrame to support indicators and consistent output
        client_tz = None if force_utc else _resolve_client_tz()
        _use_ctz = client_tz is not None
        df = _build_rates_df(rates, _use_ctz)
        if include_spread:
            _normalize_candle_spread_columns(
                df,
                headers,
                price_point=price_point,
            )
        quality_rows_removed = 0
        ohlcv_warnings: List[str] = []
        try:
            rows_before_quality = int(len(df))
            df, new_ohlcv_warnings = validate_and_clean_ohlcv_frame(df, epoch_col="__epoch")
        except ValueError as exc:
            return {"error": str(exc)}
        quality_rows_removed += max(0, rows_before_quality - int(len(df)))
        ohlcv_warnings.extend(new_ohlcv_warnings)
        if len(df) == 0:
            return {"error": f"No valid candle data available for {symbol}"}

        # Track denoise metadata if applied
        denoise_apps: List[Dict[str, Any]] = []
        denoise_warnings: List[str] = []
        ti_warnings: List[str] = []
        minimum_denoise_bars = int(
            (denoise_history_context or {}).get("minimum_bars") or 0
        )
        if minimum_denoise_bars and len(df) < minimum_denoise_bars:
            method = str((denoise_history_context or {}).get("method") or "denoise")
            return {
                "success": False,
                "error_code": "denoise_insufficient_history",
                "error": (
                    f"Denoise method '{method}' requires at least "
                    f"{minimum_denoise_bars} bars, but only {len(df)} were fetched."
                ),
                "operation": "data_fetch_candles",
                "details": {
                    "method": method,
                    "required_bars": minimum_denoise_bars,
                    "fetched_bars": int(len(df)),
                },
                "remediation": (
                    f"Increase --limit or the requested history window so at least "
                    f"{minimum_denoise_bars} bars are available, or omit --denoise."
                ),
            }
        recommended_denoise_bars = int(
            (denoise_history_context or {}).get("recommended_bars") or 0
        )
        if recommended_denoise_bars and len(df) < recommended_denoise_bars:
            denoise_warnings.append(
                "denoise_warmup_insufficient: "
                f"{(denoise_history_context or {}).get('method')} received {len(df)} "
                f"bars; request at least {recommended_denoise_bars} bars to reduce "
                "initial-state effects."
            )
        _apply_stage_denoise(df, headers, denoise, denoise_apps, when="pre_ti")
        denoise_warnings.extend(consume_denoise_warnings(df))
        try:
            ti_cols = _apply_indicator_stage(df, headers, ti_spec, denoise)
        except ValueError as exc:
            return _indicator_validation_error(str(exc), received=ti_spec)
        denoise_warnings.extend(consume_denoise_warnings(df))

        # Post-indicator filters need the fetched warmup rows just as the
        # indicators do.  Apply them before selecting the requested target
        # window so their initial state is not anchored at the first returned
        # candle.
        _apply_stage_denoise(
            df,
            headers,
            denoise,
            denoise_apps,
            when="post_ti",
            require_explicit_when=True,
        )
        denoise_warnings.extend(consume_denoise_warnings(df))

        # Filter out warmup region to return the intended target window only
        df = _trim_df_to_target(
            df,
            start_datetime,
            end_datetime,
            candles,
            copy_rows=True,
            timeframe=timeframe,
            range_selection=range_selection,
        )
        rows_after_target_trim = int(len(df))
        warmup_retry_meta: Dict[str, Any] = {
            "applied": False,
            "warmup_bars": int(warmup_bars),
        }
        indicator_window_rebuilt = False
        indicator_rows_dropped = 0

        # If TI requested, check for NaNs and retry once with increased warmup
        if ti_spec and ti_cols:
            try:
                if _indicator_columns_with_missing_values(df, ti_cols):
                    # Increase warmup and refetch once
                    warmup_bars_retry = max(int(warmup_bars * TI_NAN_WARMUP_FACTOR), warmup_bars + TI_NAN_WARMUP_MIN_ADD)
                    rates_retry, rates_retry_error = _fetch_rates_with_warmup(
                        symbol,
                        mt5_timeframe,
                        timeframe,
                        candles,
                        warmup_bars_retry,
                        start_datetime,
                        end_datetime,
                        include_incomplete=include_incomplete,
                        retry=True,
                        sanity_check=not bool(allow_stale) and not historical_bounds_requested,
                        range_selection=range_selection,
                    )
                    retry_applied = rates_retry is not None and len(rates_retry) > 0
                    warmup_retry_meta = {
                        "applied": bool(retry_applied),
                        "warmup_bars": int(warmup_bars_retry),
                        "raw_bars_fetched": int(len(rates_retry)) if rates_retry is not None else 0,
                    }
                    if rates_retry_error:
                        if isinstance(rates_retry_error, dict):
                            return rates_retry_error
                        warmup_retry_meta["error"] = str(rates_retry_error)
                        ti_warnings.append(
                            "Indicator warmup retry failed: "
                            f"{rates_retry_error}. Indicator values may be incomplete."
                        )
                    # Rebuild df and indicators with the larger window
                    if retry_applied:
                        indicator_window_rebuilt = True
                        df, ti_cols = _rebuild_candle_indicator_window(
                            rates_retry,
                            use_client_tz=_use_ctz,
                            denoise=denoise,
                            ti_spec=ti_spec,
                            headers=headers,
                        )
                        if include_spread:
                            _normalize_candle_spread_columns(
                                df,
                                headers,
                                price_point=price_point,
                            )
                        denoise_warnings.extend(consume_denoise_warnings(df))
                        try:
                            rows_before_quality = int(len(df))
                            df, retry_ohlcv_warnings = validate_and_clean_ohlcv_frame(df, epoch_col="__epoch")
                        except ValueError as exc:
                            return {"error": str(exc)}
                        quality_rows_removed += max(0, rows_before_quality - int(len(df)))
                        for warning_text in retry_ohlcv_warnings:
                            if warning_text not in ohlcv_warnings:
                                ohlcv_warnings.append(warning_text)
                        if len(df) == 0:
                            return {"error": f"No valid candle data available for {symbol}"}
                        _apply_stage_denoise(
                            df,
                            headers,
                            denoise,
                            [],
                            when="post_ti",
                            require_explicit_when=True,
                        )
                        denoise_warnings.extend(consume_denoise_warnings(df))
                        # Re-trim to target window
                        df = _trim_df_to_target(
                            df,
                            start_datetime,
                            end_datetime,
                            candles,
                            copy_rows=False,
                            timeframe=timeframe,
                            range_selection=range_selection,
                        )
                        rows_after_target_trim = int(len(df))
            except Exception as exc:
                warmup_retry_meta["error"] = str(exc)
                logger.warning("Indicator warmup retry failed", exc_info=True)
                ti_warnings.append(
                    f"Indicator warmup retry failed: {exc}. Indicator values may be incomplete."
                )

        if ti_spec and ti_cols:
            df, dropped_rows, missing_indicator_cols = _drop_incomplete_indicator_rows(df, ti_cols)
            if dropped_rows:
                indicator_rows_dropped += int(dropped_rows)
                warmup_retry_meta["incomplete_rows_dropped"] = int(dropped_rows)
                warmup_retry_meta["incomplete_indicator_columns"] = list(missing_indicator_cols)
                if len(df) == 0:
                    warning_text = (
                        f"Dropped {dropped_rows} candle rows with incomplete indicator values; "
                        "no complete indicator rows remain."
                    )
                    if warning_text not in ti_warnings:
                        ti_warnings.append(warning_text)
                    return {
                        "success": False,
                        "error_code": "data_fetch_candles_incomplete_indicators",
                        "error": (
                            f"No complete indicator rows available for {symbol} {timeframe}; "
                            "increase limit, reduce indicator lookback, or allow a larger "
                            "historical warmup window."
                        ),
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "indicator_columns": list(missing_indicator_cols),
                        "warnings": list(ti_warnings),
                        "meta": {
                            "diagnostics": {
                                "query": {
                                    "warmup_retry": warmup_retry_meta,
                                },
                            },
                        },
                    }
                ti_warnings.append(
                    f"Dropped {dropped_rows} candle rows with incomplete indicator values after warmup."
                )
                rows_after_target_trim = int(len(df))

        # Authoritative incomplete-tail trim for paths not already trimmed.
        # The ordinary fetch was classified above; do not classify its new
        # tail a second time when a mocked/feed clock is behind the bar open.
        _trimmed_incomplete = 0
        if not include_incomplete and (
            not initial_incomplete_trimmed or indicator_window_rebuilt
        ):
            rows_before_completion_trim = len(df)
            df, _ = _drop_incomplete_tail_df(
                df,
                timeframe,
                current_time_epoch=completion_reference_epoch,
            )
            _trimmed_incomplete = rows_before_completion_trim - len(df)

        # Ensure headers are unique and exist in df
        headers = [h for h in headers if h in df.columns]

        # Detect large time discontinuities (e.g., closed session windows) and
        # surface them explicitly so users can interpret forecast/analysis gaps.
        session_gaps, session_gap_warning = _collect_session_gaps(
            df,
            timeframe=timeframe,
            use_client_tz=_use_ctz,
        )
        spacing_quality = _candle_spacing_quality(df, timeframe=timeframe)
        expected_bar_seconds = float(TIMEFRAME_SECONDS.get(timeframe, 0) or 0)
        gap_after_last_bar: Optional[Dict[str, Any]] = None
        if trailing_gap_epochs is not None and "__epoch" in df.columns and len(df):
            try:
                returned_tail_epoch = float(df["__epoch"].iloc[-1])
                previous_epoch, forming_epoch = trailing_gap_epochs
                if math.isclose(returned_tail_epoch, previous_epoch, abs_tol=0.001):
                    gap_after_last_bar = _describe_session_gap(
                        previous_epoch,
                        forming_epoch,
                        expected_bar_seconds=expected_bar_seconds,
                        use_client_tz=_use_ctz,
                    )
            except (TypeError, ValueError, OverflowError, OSError):
                gap_after_last_bar = None
        if gap_after_last_bar is not None:
            gap_after_last_bar.update(
                {
                    "position": "after_last_closed_bar",
                    "next_bar_state": "forming_excluded",
                }
            )
            session_gaps.append(gap_after_last_bar)
        if spacing_quality is not None:
            spacing_quality["spacing_complete"] = not bool(session_gaps)
            spacing_quality["session_gap_count"] = len(session_gaps)
            if session_gaps and spacing_quality.get("status") == "ok":
                spacing_quality["status"] = "session_gaps_detected"

        # Reformat time consistently across rows for display, unless caller
        # explicitly requests numeric UTC epoch seconds.
        _format_candle_times(
            df,
            headers,
            time_as_epoch=time_as_epoch,
            use_client_tz=_use_ctz,
            client_tz=client_tz,
        )

        # Optionally reduce number of rows for readability/output size
        original_rows = len(df)
        simplify_eff = _normalize_simplify_spec(simplify, limit=limit, fallback_rows=original_rows)
        df, simplify_meta = _simplify_dataframe_rows_ext(df, headers, simplify_eff if simplify_eff is not None else simplify)
        if simplify_meta is not None and len(df) < original_rows:
            source_spacing_quality = spacing_quality
            spacing_quality = _candle_spacing_quality(df, timeframe=timeframe)
            if spacing_quality is not None:
                spacing_quality["spacing_complete"] = bool(
                    spacing_quality.get("spacing_matches_timeframe") is True
                )
                spacing_quality["session_gap_count"] = len(session_gaps)
                if spacing_quality.get("spacing_matches_timeframe") is not True:
                    spacing_quality["status"] = "simplified_irregular"
        # If simplify changed representation, respect returned headers
        if simplify_meta is not None and 'headers' in simplify_meta and isinstance(simplify_meta['headers'], list):
            headers = [h for h in simplify_meta['headers'] if isinstance(h, str)]

        # Assemble rows from (possibly reduced) DataFrame for selected headers
        tail_is_forming = _is_last_bar_forming(
            df,
            timeframe,
            current_time_epoch=completion_reference_epoch,
        )
        ti_added_cols = [str(c) for c in ti_cols if isinstance(c, str)]
        price_indicator_cols = _price_indicator_columns(ti_added_cols)
        rows = _format_numeric_rows_from_df(df, headers, stringify=False)
        rows = _round_row_price_columns(
            rows,
            headers,
            digits=price_digits,
            price_columns=frozenset([*_CANDLE_PRICE_COLUMNS, *price_indicator_cols]),
        )
        as_of_epoch = time.time()
        query_latency_ms = round((time.perf_counter() - query_started_at) * 1000.0, 3)
        query_mode = "range" if (start_datetime or end_datetime) else "latest"
        broker_time_check_result = _collect_candle_time_alignment(
            symbol,
            timeframe=timeframe,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )
        # Build tabular payload
        payload = _table_from_rows(headers, rows)
        # `candles` is the domain-specific row count for this tool; avoid
        # duplicating the generic `count` field in public output.
        payload.pop("count", None)
        if time_as_epoch:
            for row in payload.get("data", []) or []:
                if isinstance(row, dict) and "time" in row:
                    try:
                        row["time"] = float(row["time"])
                    except Exception:
                        pass
        
        candles_returned = int(len(df))
        source_rows_returned = int(original_rows)
        candles_requested = int(candles)
        candles_excluded = max(0, candles_requested - candles_returned)
        simplification_excluded = max(0, source_rows_returned - candles_returned)
        incomplete_candles_skipped = initial_incomplete_trimmed + _trimmed_incomplete
        has_forming_candle = bool(initial_incomplete_trimmed or _trimmed_incomplete or tail_is_forming)
        forming_candle_included = bool(include_incomplete and tail_is_forming)
        forming_candle_skipped = bool(incomplete_candles_skipped and not include_incomplete)
        latest_indicator_missing = _latest_indicator_values_missing(df, ti_added_cols)
        if forming_candle_included:
            forming_candle_status = "included"
        elif forming_candle_skipped:
            forming_candle_status = "skipped"
        elif has_forming_candle:
            forming_candle_status = "detected"
        else:
            forming_candle_status = "none"
        data_rows = payload.get("data")
        if isinstance(data_rows, list):
            for index, row in enumerate(data_rows):
                if not isinstance(row, dict):
                    continue
                is_forming_row = bool(
                    forming_candle_included and index == len(data_rows) - 1
                )
                row["bar_state"] = "forming" if is_forming_row else "closed"
            if forming_candle_included and ti_added_cols:
                payload["indicator_uses_incomplete_bar"] = True

            if str(timeframe).upper() in CALENDAR_TIMEFRAMES:
                broker_tz = mt5_config.get_server_tz()
                if broker_tz is not None and "__epoch" in df.columns:
                    for row, epoch_value in zip(
                        data_rows,
                        df["__epoch"].tolist(),
                        strict=False,
                    ):
                        if not isinstance(row, dict):
                            continue
                        try:
                            session_date = datetime.fromtimestamp(
                                float(epoch_value),
                                tz=broker_tz,
                            ).date().isoformat()
                        except Exception:
                            continue
                        row["broker_session_date"] = session_date
                        if str(timeframe).upper() == "D1":
                            row["broker_trading_day"] = session_date
                    payload["broker_timezone"] = (
                        getattr(broker_tz, "key", None)
                        or getattr(broker_tz, "zone", None)
                        or str(broker_tz)
                    )
        source_rows_excluded = max(0, candles_requested - source_rows_returned)
        remaining_after_forming = max(
            0,
            source_rows_excluded - incomplete_candles_skipped,
        )
        indicator_excluded = min(int(indicator_rows_dropped), remaining_after_forming)
        remaining_after_indicator = max(0, remaining_after_forming - indicator_excluded)
        quality_excluded = min(int(quality_rows_removed), remaining_after_indicator)
        remaining_excluded = max(0, remaining_after_indicator - quality_excluded)
        window_shortfall = remaining_excluded if (start_datetime or end_datetime) else 0
        source_shortfall = max(0, remaining_excluded - window_shortfall)
        candle_excluded_total = (
            incomplete_candles_skipped
            + indicator_excluded
            + quality_excluded
            + window_shortfall
            + source_shortfall
            + simplification_excluded
        )
        candle_counts = {
            "requested": candles_requested,
            "returned": candles_returned,
            "source_rows_returned": source_rows_returned,
            "excluded": {
                "forming_bar": incomplete_candles_skipped,
                "indicator_warmup": indicator_excluded,
                "quality_filtered": quality_excluded,
                "simplification": simplification_excluded,
                "window_or_source_shortfall": window_shortfall + source_shortfall,
                "total": candle_excluded_total,
            },
        }
        volume_metadata = _candle_volume_metadata(headers)
        latest_bar_epoch = None
        first_bar_time = None
        latest_bar_time = None
        data_rows = payload.get("data")
        if isinstance(data_rows, list) and data_rows:
            first_row = data_rows[0]
            latest_row = data_rows[-1]
            if isinstance(first_row, dict):
                first_bar_time = first_row.get("time")
            if isinstance(latest_row, dict):
                latest_bar_time = latest_row.get("time")
        try:
            if len(df) > 0 and "__epoch" in df.columns:
                latest_bar_epoch = float(df["__epoch"].iloc[-1])
        except Exception:
            latest_bar_epoch = None

        payload.update({
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": candles_returned,
            "requested_limit": candles_requested,
            "returned_count": candles_returned,
            "as_of": format_epoch_utc(as_of_epoch),
            "price_precision": int(price_digits),
            "price_point": price_point,
            **time_normalization,
            **volume_metadata,
            **_candle_time_convention_metadata(timeframe),
            "candles_requested": candles_requested,
            "candles_excluded": candles_excluded,
            "candle_counts": candle_counts,
            "incomplete_candles_skipped": incomplete_candles_skipped,
            "has_forming_candle": has_forming_candle,
            "forming_candle_status": forming_candle_status,
            "forming_candle_included": forming_candle_included,
            "forming_candle_skipped": forming_candle_skipped,
            "meta": {
                "diagnostics": {
                    "query": {
                        "mode": query_mode,
                        "include_spread": bool(include_spread),
                        "include_incomplete": bool(include_incomplete),
                        "latency_ms": query_latency_ms,
                        "requested_bars": candles_requested,
                        "warmup_bars": int(warmup_bars),
                        "indicator_warmup_bars": int(indicator_warmup_bars),
                        "denoise_warmup_bars": int(denoise_warmup_bars),
                        "raw_bars_fetched": raw_bars_fetched,
                        "rows_after_target_trim": rows_after_target_trim,
                        "indicator_rows_dropped": int(indicator_rows_dropped),
                        "quality_rows_removed": int(quality_rows_removed),
                        "cache_status": "unknown",
                        "warmup_retry": warmup_retry_meta,
                        **(
                            dict(rate_fetch_diagnostics.get("range_fetch"))
                            if isinstance(
                                rate_fetch_diagnostics.get("range_fetch"), dict
                            )
                            else {}
                        ),
                    },
                    "indicators": {
                        "requested": bool(ti_spec),
                        "spec": _normalize_indicator_spec_for_display(ti_spec),
                        "added_columns": ti_added_cols,
                        "volume_source": df.attrs.get("indicator_volume_source"),
                        "index_time_basis": "utc_epoch" if ti_spec else None,
                    },
                    "session_gaps": {
                        "expected_bar_seconds": float(expected_bar_seconds) if expected_bar_seconds > 0 else None,
                    },
                    "time_normalization": dict(time_normalization),
                },
            },
        })
        if spacing_quality is not None:
            payload["bar_spacing"] = spacing_quality
            simplified_rows_reduced = simplify_meta is not None and len(df) < original_rows
            if simplified_rows_reduced:
                payload["source_bar_spacing"] = source_spacing_quality
            if spacing_quality.get("spacing_matches_timeframe") is False:
                if simplified_rows_reduced:
                    payload["returned_spacing_irregular"] = True
                else:
                    payload["timeframe_spacing_mismatch"] = True
                    payload.setdefault("warnings", []).append(
                        "Observed candle spacing does not match the requested "
                        f"{timeframe} timeframe; treat this history as partial or "
                        "coarser broker data."
                    )
        if broker_time_check_result is not None:
            payload["meta"]["diagnostics"]["mt5_time_alignment"] = (
                dict(broker_time_check_result)
            )
        if ti_added_cols:
            payload["indicator_columns"] = list(ti_added_cols)
            spec_text = _normalize_indicator_spec_for_display(ti_spec)
            if spec_text:
                payload["indicators_spec"] = spec_text
        if ti_spec:
            payload["indicator_engine"] = indicator_engine_provenance()
        if price_indicator_cols and price_digits > 0:
            rounding_meta = {
                "price_columns": price_indicator_cols,
                "price_precision": int(price_digits),
                "policy": "symbol_price_precision",
            }
            payload["indicator_rounding"] = rounding_meta
            payload["meta"]["diagnostics"]["indicators"]["rounding"] = rounding_meta
        data_window = {
            "start": first_bar_time,
            "end": latest_bar_time,
            "requested_limit": candles_requested,
            "returned_count": candles_returned,
            "latest_bar_complete": not tail_is_forming,
        }
        if latest_bar_epoch is not None and query_mode != "range":
            latest_bar_age_epoch = float(latest_bar_epoch)
            latest_bar_age_metric = "latest_bar_open_age_seconds"
            if not forming_candle_included and expected_bar_seconds > 0:
                latest_bar_age_epoch = bar_close_epoch(latest_bar_epoch, timeframe)
                latest_bar_age_metric = "latest_completed_bar_close_age_seconds"
            data_window["latest_bar_age_seconds"] = round(
                max(0.0, float(as_of_epoch) - latest_bar_age_epoch),
                3,
            )
            data_window["latest_bar_age_metric"] = latest_bar_age_metric
        payload["data_window"] = {
            key: value
            for key, value in data_window.items()
            if value is not None
        }
        if ohlcv not in (None, ""):
            payload["ohlcv_filter_applied"] = True
            payload["ohlcv_filter"] = str(ohlcv).strip()
        if forming_candle_included:
            data_rows = payload.get("data")
            if isinstance(data_rows, list) and data_rows:
                payload["forming_candle_index"] = len(data_rows) - 1
        if query_mode == "range":
            payload["query_applied"] = _candle_query_applied(
                timeframe=timeframe,
                start=start_datetime,
                end=end_datetime,
                limit=candles_requested,
            )
        if price_currency:
            payload["price_currency"] = price_currency
        payload["price_basis"] = price_basis
        if incomplete_candles_skipped and not include_incomplete:
            if ti_spec and latest_indicator_missing:
                payload["hint"] = (
                    "Latest forming candle was skipped. Set include_incomplete=true only if you need "
                    "that bar; increase limit if requested indicators need more warmup context."
                )
            else:
                payload["hint"] = "Set include_incomplete=true to include the latest forming candle."
        if isinstance(freshness_diagnostics, dict):
            payload["meta"]["diagnostics"]["freshness"] = dict(freshness_diagnostics)
        if include_spread:
            payload["spread_historical_source"] = "mt5_candle"
            payload["spread_historical_available"] = True
            payload["spread_mode"] = "per_bar"
        if session_gap_warning:
            payload["meta"]["diagnostics"]["session_gaps"]["warning"] = session_gap_warning
        payload["timezone"] = display_timezone_label(
            use_client_tz=_use_ctz,
            client_tz=client_tz,
            fallback="local",
        )
        if simplify_meta is not None:
            payload["simplified"] = True
            public_simplify = _public_simplify_meta(simplify_meta) or {"applied": True}
            public_simplify["original_rows"] = int(original_rows)
            public_simplify["returned_rows"] = int(len(df))
            payload["simplify"] = public_simplify
            simplify_method = str(simplify_meta.get("method") or "").strip().lower()
            simplify_mode = str(simplify_meta.get("mode") or "").strip().lower()
            segment_mean_columns = list(simplify_meta.get("segment_mean_columns") or [])
            simplify_reduced_rows = int(original_rows) > int(len(df))
            visualization_only = simplify_method in {"lttb", "rdp", "pla", "apca"}
            approximate_derived_values = bool(
                simplify_mode == "approximate" and segment_mean_columns
            )
            if simplify_reduced_rows and (visualization_only or approximate_derived_values):
                payload["series_type"] = "downsampled_visualization"
                payload["equal_interval"] = False
                payload["analysis_compatible"] = False
                if approximate_derived_values:
                    payload.setdefault("warnings", []).append(
                        "Approximate simplification reports non-OHLC numeric columns as "
                        "segment means, not recomputed analytical values; do not use them "
                        "for indicator thresholds or forecasts."
                    )
                else:
                    payload.setdefault("warnings", []).append(
                        "Simplified candle rows are visualization samples with irregular time gaps; "
                        "do not use them as equal-interval OHLC input for indicators or forecasts."
                    )
        # Attach denoise applications metadata if any
        if denoise_apps:
            payload['denoise'] = {'applications': denoise_apps}
            if denoise_history_context:
                payload['denoise']['history_context'] = {
                    **denoise_history_context,
                    "history_bars_fetched": raw_bars_fetched,
                }
            payload['denoise_status'] = 'applied'
        elif denoise:
            payload['denoise_status'] = 'skipped'
            payload['denoise_applied'] = False
            if denoise_warnings:
                payload['denoise_status_reason'] = denoise_warnings[0]
        projection_requested = bool(str(ohlcv or "").strip())
        if denoise_apps or ti_spec or projection_requested:
            denoise_stages = {
                str(app.get("when") or "").lower()
                for app in denoise_apps
                if isinstance(app, dict)
            }
            pipeline = ["fetch_ohlcv"]
            if "pre_ti" in denoise_stages:
                pipeline.append("denoise_pre_ti")
            if ti_spec:
                pipeline.append("indicators")
                payload["indicator_input"] = (
                    "pre_ti_denoised_ohlcv"
                    if "pre_ti" in denoise_stages
                    else "raw_ohlcv"
                )
                suffix = str((denoise or {}).get("suffix") or "_dn")
                if any(str(column).endswith(suffix) for column in ti_cols):
                    payload["indicator_column_suffix"] = suffix
            if "post_ti" in denoise_stages:
                pipeline.append("denoise_post_ti")
            if projection_requested:
                pipeline.append("project_returned_ohlcv")
            payload["processing_pipeline"] = pipeline
        if denoise_warnings:
            warns = payload.get('warnings')
            if not isinstance(warns, list):
                warns = []
            for warning_text in denoise_warnings:
                if warning_text not in warns:
                    warns.append(warning_text)
            payload['warnings'] = warns
        if ohlcv_warnings:
            warns = payload.get('warnings')
            if not isinstance(warns, list):
                warns = []
            for warning_text in ohlcv_warnings:
                if warning_text not in warns:
                    warns.append(warning_text)
            payload['warnings'] = warns
        if ti_warnings:
            warns = payload.get('warnings')
            if not isinstance(warns, list):
                warns = []
            for warning_text in ti_warnings:
                if warning_text not in warns:
                    warns.append(warning_text)
            payload['warnings'] = warns
        if session_gaps:
            payload['session_gaps'] = session_gaps
            if gap_after_last_bar is not None:
                payload['gap_after_last_bar'] = gap_after_last_bar
            _annotate_candle_gap_rows(payload, session_gaps)
            warns = payload.get('warnings')
            if not isinstance(warns, list):
                warns = []
            warns.append(
                "Detected session gaps larger than expected bar spacing ({secs:.0f}s).".format(
                    secs=expected_bar_seconds,
                )
            )
            try:
                first_gap = session_gaps[0]
                warns.append(
                    "Example gap: {from_} -> {to} ({missing} missing bars, likely {context}).".format(
                        from_=str(first_gap.get("from")),
                        to=str(first_gap.get("to")),
                        missing=int(first_gap.get("missing_bars_est") or 0),
                        context=str(first_gap.get("context") or "session break"),
                    )
                )
            except Exception:
                pass
            payload['warnings'] = warns
        elif session_gap_warning:
            warns = payload.get('warnings')
            if not isinstance(warns, list):
                warns = []
            warns.append(session_gap_warning)
            payload['warnings'] = warns

        # If include_spread requested but spread data is missing or all zero, try fallback estimate from recent ticks.
        if include_spread:
            data_rows = payload.get("data", []) or []
            spread_row_count = len(data_rows)
            spread_available_count = 0
            for row in data_rows:
                if isinstance(row, dict):
                    if row.get("spread_available") is True:
                        spread_available_count += 1

            spread_missing_count = max(0, spread_row_count - spread_available_count)
            coverage_pct = (
                (spread_available_count / float(spread_row_count)) * 100.0
                if spread_row_count
                else 0.0
            )
            payload["spread_historical_available"] = bool(
                spread_row_count and spread_missing_count == 0
            )
            payload["spread_historical_coverage_pct"] = round(coverage_pct, 2)
            payload["spread_missing_count"] = spread_missing_count
            payload["spread_source"] = (
                "mt5_candle"
                if spread_missing_count == 0 and spread_row_count
                else "mt5_candle_partial"
                if spread_available_count
                else "unavailable"
            )
            if spread_missing_count:
                payload["spread_mode"] = (
                    "partial_per_bar" if spread_available_count else "unavailable"
                )
                payload.setdefault("warnings", []).append(
                    "Historical candle spread is unavailable for "
                    f"{spread_missing_count} of {spread_row_count} row(s); those "
                    "spread values are null and must not be treated as zero cost."
                )
            if spread_available_count == 0:
                try:
                    live_spread, reference_freshness = _live_tick_spread_reference(
                        symbol
                    )
                    if live_spread is not None:
                        estimate = float(live_spread)
                        payload.setdefault("warnings", []).append(
                            "include_spread requested but historical per-bar spread is "
                            "unavailable; a single current live ticker reference "
                            f"({estimate:g}) is returned "
                            "at payload level and is not per-bar historical spread."
                        )
                        payload["spread_reference"] = {
                            "value": estimate,
                            "unit": "price",
                            "source": "live_ticker",
                            "basis": "single_reference_not_per_bar_historical",
                            **reference_freshness,
                        }
                        payload["spread_mode"] = "single_reference"
                        payload["spread_source"] = "live_ticker"
                        payload.setdefault("meta", {}).setdefault("diagnostics", {}).setdefault("spread_estimate", {})["estimated_mean"] = estimate
                        payload["meta"]["diagnostics"]["spread_estimate"]["source"] = "live_ticker"
                        payload["meta"]["diagnostics"]["spread_estimate"]["unit"] = "price"
                except Exception:
                    payload.setdefault("warnings", []).append("include_spread requested but spread unavailable; no fallback available.")
                if payload.get("spread_mode") == "unavailable" and not any(
                    "include_spread requested" in str(item)
                    for item in payload.get("warnings", [])
                ):
                    payload.setdefault("warnings", []).append(
                        "include_spread requested but historical per-bar spread and a "
                        "live/tick reference are unavailable."
                    )

        return payload
    except MT5TimestampNormalizationError as exc:
        return build_error_payload(
            str(exc), code="timestamp_normalization_failed", operation="data_fetch_candles",
            details={
                **exc.details, "symbol": symbol, "timeframe": timeframe,
                "requested_start": start, "requested_end": end, "requested_limit": limit,
                "warmup_bars": warmup_bars, "indicators_spec": ti_spec,
                "history_fetch": rate_fetch_diagnostics,
            },
            remediation=exc.remediation,
        )
    except DenoiseParameterError as exc:
        return build_error_payload(
            str(exc), code="invalid_denoise_parameter", operation="data_fetch_candles",
            details=exc.details,
            remediation="Correct the denoise parameter using the stated allowed range; see denoise_describe.",
        )
    except DenoiseExecutionError as exc:
        return build_error_payload(
            str(exc), code="denoise_failed", operation="data_fetch_candles",
            remediation="Inspect the denoise parameters and available history; use denoise_describe for supported settings.",
        )
    except DenoiseColumnError as exc:
        return {
            "success": False,
            "error_code": "denoise_column_not_found",
            "error": str(exc),
            "operation": "data_fetch_candles",
            "remediation": (
                "Use lowercase indicator column names from the candle "
                "response or indicators_describe, for example rsi_14."
            ),
            "details": {
                "missing_columns": list(exc.columns),
                "available_columns": list(exc.available),
            },
        }
    except ValueError as e:
        message = str(e)
        lowered = message.lower()
        if indicators and ("indicator" in lowered or "vwap" in lowered):
            return _indicator_validation_error(message, received=indicators)
        return {
            "error": f"Error getting rates: {type(e).__name__}: {e}",
            "error_detail": {
                "operation": "fetch_candles",
                "symbol": symbol,
                "timeframe": timeframe,
                "start": str(start) if start else None,
                "end": str(end) if end else None,
            },
        }
    except Exception as e:
        return {
            "error": f"Error getting rates: {type(e).__name__}: {e}",
            "error_detail": {
                "operation": "fetch_candles",
                "symbol": symbol,
                "timeframe": timeframe,
                "start": str(start) if start else None,
                "end": str(end) if end else None,
            },
        }


def _live_tick_spread_reference(symbol: str):
    from .ticks import _live_tick_spread_reference as resolve_live_tick_spread

    return resolve_live_tick_spread(symbol)
