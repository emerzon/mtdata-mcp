"""Shared causal helpers for symbol expansion, history fetch, transforms, and alignment."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mtdata.core.mt5_gateway import create_mt5_gateway, mt5_connection_error
from mtdata.core.output_contract import build_pagination_meta
from mtdata.services.data_service.candles import (
    _drop_incomplete_tail_df,
    _is_last_bar_forming,
    _parse_candle_calendar_bound,
)
from mtdata.shared.constants import (
    TIMEFRAME_MAP,
)
from mtdata.utils.mt5 import (
    _ensure_symbol_ready,
    _mt5_copy_rates_from,
    _mt5_copy_rates_range,
    ensure_mt5_connection_or_raise,
    mt5,
)
from mtdata.utils.symbol import (
    _extract_group_path as _extract_group_path_util,
)
from mtdata.utils.symbol import (
    _normalize_group_path_query,
)
from mtdata.utils.time import bar_close_epoch, format_datetime_utc, format_epoch_utc
from mtdata.utils.utils import (
    _parse_end_datetime,
    _parse_start_datetime,
    validate_historical_range,
)

_CORRELATION_METHOD_ALIASES: Dict[str, str] = {
    "pearson": "pearson",
    "linear": "pearson",
    "spearman": "spearman",
    "rank": "spearman",
    "rank_corr": "spearman",
    "rank_correlation": "spearman",
}

_TRANSFORM_ALIASES: Dict[str, str] = {
    "log_return": "log_return",
    "logret": "log_return",
    "log-returns": "log_return",
    "pct": "pct",
    "return": "pct",
    "pct_change": "pct",
    "diff": "diff",
    "difference": "diff",
    "first_diff": "diff",
    "level": "level",
    "none": "level",
    "raw": "level",
    "price": "level",
    "log": "log_level",
    "log_level": "log_level",
    "log-price": "log_level",
    "log_price": "log_level",
}

_COINTEGRATION_TRANSFORM_ALIASES: Dict[str, str] = {
    "level": "level",
    "raw": "level",
    "price": "level",
    "log": "log_level",
    "log_level": "log_level",
    "log-price": "log_level",
    "log_price": "log_level",
}

_COINTEGRATION_TREND_ALIASES: Dict[str, str] = {
    "c": "c",
    "const": "c",
    "constant": "c",
    "ct": "ct",
    "trend": "ct",
    "ctt": "ctt",
    "quadratic": "ctt",
    "n": "n",
    "none": "n",
    "no_const": "n",
}

_MIN_PAIR_ALIGNMENT_FRACTION = 0.90

_ALIGNMENT_WARNING_THRESHOLD_PCT = 5.0

# Human-readable legends for output interpretation
_TRANSFORM_LEGEND: Dict[str, Dict[str, str]] = {
    "log_return": {
        "description": "Logarithmic returns (continuously compounded)",
        "formula": "ln(close_t / close_t-1)",
        "use_case": "Stationary, time-additive returns; preferred for multi-horizon analysis",
    },
    "pct": {
        "description": "Simple percentage change (unit fraction)",
        "formula": "(close_t - close_t-1) / close_t-1",
        "use_case": "Intuitive simple returns; 0.01 corresponds to a 1% gain",
    },
    "diff": {
        "description": "First difference (absolute change)",
        "formula": "close_t - close_t-1",
        "use_case": "Removes trends for stationary analysis; preserves scale",
    },
    "level": {
        "description": "Raw price levels (no transformation)",
        "formula": "close_t",
        "use_case": "Direct price analysis; required for cointegration tests",
    },
    "log_level": {
        "description": "Natural log of price levels",
        "formula": "ln(close_t)",
        "use_case": "Price-level analysis with reduced scale effects",
    },
}

_COINTEGRATION_TREND_LEGEND: Dict[str, Dict[str, str]] = {
    "c": {
        "description": "Constant only",
        "interpretation": "Tests for cointegration with non-zero mean but no trend",
    },
    "ct": {
        "description": "Constant and linear trend",
        "interpretation": "Tests for cointegration allowing for deterministic linear trend",
    },
    "ctt": {
        "description": "Constant and quadratic trend",
        "interpretation": "Tests for cointegration allowing for curved deterministic trends",
    },
    "n": {
        "description": "No deterministic terms",
        "interpretation": "Tests for cointegration around zero (rarely appropriate for prices)",
    },
}


def _min_overlap_exceeds_window_message(*, min_overlap: int, window_bars: int) -> str:
    return (
        f"min_overlap ({int(min_overlap)}) cannot exceed window_bars ({int(window_bars)}). "
        "Reduce min_overlap or increase window_bars."
    )


def _causal_connection_error() -> Dict[str, Any] | None:
    return mt5_connection_error(
        create_mt5_gateway(
            adapter=mt5,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )
    )


def _parse_symbol_request(value: Optional[str]) -> tuple[List[str], int]:
    items: List[str] = []
    for chunk in str(value or "").replace(";", ",").split(","):
        name = chunk.strip()
        if name:
            items.append(name)
    return list(dict.fromkeys(items)), len(items)


def _duplicate_only_symbol_error(
    symbols: List[str], entry_count: int
) -> Optional[str]:
    if entry_count <= 1 or len(symbols) >= 2:
        return None
    duplicate = symbols[0] if symbols else "the same symbol"
    return (
        "symbols must contain at least two distinct symbols when multiple entries "
        f"are supplied; {duplicate} was provided more than once."
    )


def _visible_group_members(
    all_symbols: Any,
    group_path: str,
) -> List[str]:
    members: List[str] = []
    for sym in all_symbols or []:
        if not getattr(sym, "visible", True):
            continue
        if _extract_group_path_util(sym) == group_path:
            members.append(sym.name)
    return list(dict.fromkeys(members))


def _expand_symbols_for_group(
    anchor: str, gateway: Any = None
) -> tuple[List[str], str | None, str | None]:
    """Return visible group members for anchor along with the group path."""
    mt5_gateway = gateway or create_mt5_gateway(
        adapter=mt5,
        ensure_connection_impl=ensure_mt5_connection_or_raise,
    )
    info = mt5_gateway.symbol_info(anchor)
    if info is None:
        return [], f"Symbol {anchor} not found", None
    group_path = _extract_group_path_util(info)
    all_symbols = mt5_gateway.symbols_get()
    if all_symbols is None:
        return [], f"Failed to load symbol list: {mt5_gateway.last_error()}", group_path
    members = _visible_group_members(all_symbols, group_path)
    if anchor not in members:
        members.insert(0, anchor)
    deduped = list(dict.fromkeys(members))
    if len(deduped) < 2:
        return (
            deduped,
            f"Symbol group {group_path} has fewer than two visible instruments",
            group_path,
        )
    return deduped, None, group_path


def _expand_symbols_for_group_path(
    query: str, gateway: Any = None
) -> tuple[List[str], str | None, str | None]:
    """Return visible group members for an explicit MT5 group path query."""
    mt5_gateway = gateway or create_mt5_gateway(
        adapter=mt5,
        ensure_connection_impl=ensure_mt5_connection_or_raise,
    )
    group_query = str(query or "").strip()
    if not group_query:
        return [], "Group path must not be empty.", None

    all_symbols = mt5_gateway.symbols_get()
    if all_symbols is None:
        return [], f"Failed to load symbol list: {mt5_gateway.last_error()}", None

    groups: Dict[str, List[str]] = {}
    for sym in all_symbols:
        group_path = _extract_group_path_util(sym)
        if not group_path:
            continue
        if not getattr(sym, "visible", True):
            continue
        groups.setdefault(group_path, []).append(sym.name)

    if not groups:
        return [], "No visible MT5 symbol groups are available.", None

    query_lower = _normalize_group_path_query(group_query).lower()
    exact_matches = [
        group_path
        for group_path in groups
        if _normalize_group_path_query(group_path).lower() == query_lower
    ]
    matched_paths = exact_matches or [
        group_path
        for group_path in groups
        if query_lower in _normalize_group_path_query(group_path).lower()
    ]
    matched_paths = list(dict.fromkeys(matched_paths))
    if not matched_paths:
        return (
            [],
            f"Group '{group_query}' was not found among visible MT5 symbol groups.",
            None,
        )
    if len(matched_paths) > 1:
        preview = ", ".join(sorted(matched_paths)[:5])
        suffix = ", ..." if len(matched_paths) > 5 else ""
        return (
            [],
            (
                f"Group '{group_query}' matched multiple visible MT5 symbol groups: "
                f"{preview}{suffix}"
            ),
            None,
        )

    group_path = matched_paths[0]
    members = _visible_group_members(all_symbols, group_path)
    if len(members) < 2:
        return (
            members,
            f"Symbol group {group_path} has fewer than two visible instruments",
            group_path,
        )
    return members, None, group_path


def _resolve_history_window(
    start: Optional[str],
    end: Optional[str],
    *,
    timeframe: Optional[str] = None,
) -> Tuple[Optional[datetime], Optional[datetime], Optional[str]]:
    range_error = validate_historical_range(start, end)
    if range_error is not None:
        code = str(range_error.get("error_code") or "invalid_date_range")
        return None, None, f"{code}: {range_error['error']}"
    try:
        start_dt = (
            _parse_candle_calendar_bound(
                start,
                timeframe=timeframe,
                end_bound=False,
            )
            or _parse_start_datetime(start)
            if start
            else None
        )
        end_dt = (
            _parse_candle_calendar_bound(
                end,
                timeframe=timeframe,
                end_bound=True,
            )
            or _parse_end_datetime(end)
            if end
            else None
        )
    except ValueError as exc:
        return None, None, str(exc)
    if start and start_dt is None:
        return None, None, "Invalid start time."
    if end and end_dt is None:
        return None, None, "Invalid end time."
    if start_dt is not None and start_dt.tzinfo is not None:
        start_dt = start_dt.astimezone(timezone.utc).replace(tzinfo=None)
    if end_dt is not None and end_dt.tzinfo is not None:
        end_dt = end_dt.astimezone(timezone.utc).replace(tzinfo=None)
    if start_dt is not None and end_dt is None:
        end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        return None, None, "start must be before or equal to end."
    return start_dt, end_dt, None


def _history_fetch_error_code(errors: List[str]) -> str:
    if _symbol_not_found_error(errors):
        return "symbol_not_found"
    return (
        "future_date_range"
        if any(str(error).startswith("future_date_range:") for error in errors)
        else "data_fetch_failed"
    )


def _granger_maximum_lag_for_samples(samples: int) -> int:
    try:
        count = int(samples)
    except (TypeError, ValueError):
        return 0
    return max(0, int((count - 1) / 3) - 1)


def _granger_minimum_samples_for_lag(max_lag: int) -> int:
    try:
        lag = int(max_lag)
    except (TypeError, ValueError):
        lag = 0
    return max(1, 3 * lag + 4)


def _symbol_not_found_error(errors: List[str]) -> Optional[str]:
    for error in errors:
        text = str(error)
        if "was not found" in text or "unknown symbol" in text.lower():
            return text
    return None


def _insufficient_symbol_payload(
    *,
    message: str,
    errors: List[str],
    meta: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    missing = _symbol_not_found_error(errors)
    if missing is not None:
        return _causal_error(
            missing,
            code="symbol_not_found",
            meta=meta,
            warnings=warnings,
            details=list(errors),
        )
    return _causal_error(
        message,
        code="insufficient_symbols",
        meta=meta,
        warnings=warnings,
    )


def _symbol_fetch_data_quality(
    *,
    requested_symbols: List[str],
    analysis_universe: List[str],
    series_map: Dict[str, pd.Series],
    errors: List[str],
    allow_partial: bool,
    analysis_family_kind: str,
) -> Dict[str, Any]:
    resolved = [symbol for symbol in analysis_universe if symbol in series_map]
    omitted = [symbol for symbol in analysis_universe if symbol not in series_map]
    if analysis_family_kind == "multivariate_symbol_set":
        requested_tests = int(len(analysis_universe) >= 2)
        resolved_tests = int(len(resolved) >= 2)
    else:
        directed = analysis_family_kind == "directed_symbol_pairs"
        divisor = 1 if directed else 2
        requested_tests = (
            len(analysis_universe) * (len(analysis_universe) - 1) // divisor
        )
        resolved_tests = len(resolved) * (len(resolved) - 1) // divisor
    analysis_family = {
        "kind": analysis_family_kind,
        "tests_requested": int(requested_tests),
        "tests_available": int(resolved_tests),
        "tests_removed": int(requested_tests - resolved_tests),
    }
    if analysis_family_kind == "multivariate_symbol_set":
        analysis_family.update(
            {
                "dimensions_requested": len(analysis_universe),
                "dimensions_available": len(resolved),
                "dimensions_removed": len(omitted),
                "universe_changed": bool(omitted),
            }
        )
    return {
        "status": "partial" if omitted else "complete",
        "allow_partial": bool(allow_partial),
        "requested_symbols": list(requested_symbols),
        "analysis_universe_symbols": list(analysis_universe),
        "resolved_symbols": resolved,
        "omitted_symbols": omitted,
        "omissions": [
            {
                "symbol": symbol,
                "reason": errors[index] if index < len(errors) else "history_unavailable",
            }
            for index, symbol in enumerate(omitted)
        ],
        "analysis_family": analysis_family,
    }


def _partial_symbol_fetch_error(
    *,
    meta: Dict[str, Any],
    errors: List[str],
    data_quality: Dict[str, Any],
) -> Dict[str, Any]:
    out = _causal_error(
        "One or more requested symbols could not be fetched; subset analysis is "
        "disabled by default.",
        code="symbol_set_incomplete",
        meta=meta,
        details=list(errors),
    )
    out["data_quality"] = data_quality
    out["remediation"] = (
        "Correct the omitted symbols or retry with allow_partial=true to analyze "
        "the disclosed subset."
    )
    return out


def _fetch_series(
    symbol: str,
    timeframe,
    count: int,
    retries: int = 3,
    pause: float = 0.25,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    timeframe_key: Optional[str] = None,
    include_incomplete: bool = False,
) -> Tuple[pd.Series, str | None]:
    """Fetch close prices, excluding the current forming bar by default."""
    err = _ensure_symbol_ready(symbol)
    if err:
        return pd.Series(dtype=float), err
    start_dt, end_dt, window_error = _resolve_history_window(
        start,
        end,
        timeframe=timeframe_key,
    )
    if window_error:
        return pd.Series(dtype=float), window_error
    anchor = end_dt or datetime.now(timezone.utc)
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)

    for _attempt in range(retries):
        if start_dt is not None:
            data = _mt5_copy_rates_range(symbol, timeframe, start_dt, end_dt)
        elif end_dt is not None:
            data = _mt5_copy_rates_from(
                symbol, timeframe, end_dt, count + (0 if include_incomplete else 1)
            )
        else:
            data = _mt5_copy_rates_from(
                symbol, timeframe, anchor, count + (0 if include_incomplete else 1)
            )
        if data is None or len(data) == 0:
            time.sleep(pause)
            continue
        try:
            df = pd.DataFrame(data)
        except Exception:
            df = pd.DataFrame(list(data))
        if df.empty or "time" not in df or "close" not in df:
            time.sleep(pause)
            continue
        df = df.sort_values("time")
        df = df[pd.to_numeric(df["time"], errors="coerce") <= anchor.timestamp()]
        timeframe_name = str(timeframe_key or "").strip().upper()
        if not timeframe_name:
            timeframe_name = next(
                (name for name, value in TIMEFRAME_MAP.items() if value == timeframe),
                "",
            )
        forming_trimmed = False
        last_is_forming = False
        if timeframe_name and not df.empty:
            last_is_forming = _is_last_bar_forming(
                df, timeframe_name, current_time_epoch=anchor.timestamp()
            )
            if include_incomplete and end_dt and last_is_forming and not _is_last_bar_forming(df, timeframe_name):
                return pd.Series(dtype=float), (
                    "Historical partial-candle values cannot be recovered from completed bars; "
                    "use include_incomplete=false."
                )
            if not include_incomplete and last_is_forming:
                df, forming_trimmed = _drop_incomplete_tail_df(
                    df, timeframe_name, current_time_epoch=anchor.timestamp()
                )
                last_is_forming = False
        if df.empty:
            time.sleep(pause)
            continue
        if start_dt is None and len(df) > count:
            df = df.tail(count)
        series = pd.Series(
            df["close"].to_numpy(dtype=float),
            index=pd.to_datetime(df["time"], unit="s"),
        )
        series = series[~series.index.duplicated(keep="last")]
        series.attrs["include_incomplete"] = bool(include_incomplete)
        series.attrs["forming_candle_skipped"] = bool(forming_trimmed)
        series.attrs["forming_candle_included"] = bool(last_is_forming)
        series.attrs["latest_bar_complete"] = not bool(last_is_forming)
        series.attrs["resolved_as_of"] = format_datetime_utc(anchor)
        if end_dt is not None:
            series.attrs["requested_as_of"] = format_datetime_utc(end_dt)
        return series, None
    return pd.Series(dtype=float), f"Failed to fetch data for {symbol}" + (
        f" after {retries} retries" if retries > 1 else ""
    )


def _fetch_series_for_window(
    symbol: str,
    timeframe,
    count: int,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    timeframe_key: Optional[str] = None,
    include_incomplete: bool = False,
) -> Tuple[pd.Series, str | None]:
    if start or end:
        return _fetch_series(
            symbol,
            timeframe,
            count,
            start=start,
            end=end,
            timeframe_key=timeframe_key,
            include_incomplete=include_incomplete,
        )
    return _fetch_series(
        symbol,
        timeframe,
        count,
        timeframe_key=timeframe_key,
        include_incomplete=include_incomplete,
    )


def _bar_completion_context(
    series_map: Dict[str, pd.Series], *, include_incomplete: bool
) -> Dict[str, Any]:
    forming_included = any(
        bool(series.attrs.get("forming_candle_included"))
        for series in series_map.values()
    )
    forming_skipped = any(
        bool(series.attrs.get("forming_candle_skipped"))
        for series in series_map.values()
    )
    if forming_included:
        status = "included"
    elif forming_skipped:
        status = "skipped"
    else:
        status = "none"
    return {
        "include_incomplete": bool(include_incomplete),
        "latest_bar_complete": not forming_included,
        "forming_candle_status": status,
    }


def _transform_frame(frame: pd.DataFrame, transform: str) -> pd.DataFrame:
    transform = transform.strip().lower()
    if transform not in {
        "log_return", "logret", "log-returns", "log_level", "log",
        "log-price", "log_price", "pct", "return", "pct_change",
        "diff", "difference", "first_diff",
    }:
        return frame

    # Transform each symbol on its own observed index. Applying diff/pct_change
    # after an outer join lets another symbol's timestamp interrupt the true
    # predecessor relationship and creates artificial missing returns.
    transformed: Dict[str, pd.Series] = {}
    for column in frame.columns:
        series = frame[column].dropna().astype(float)
        if transform in ("log_return", "logret", "log-returns"):
            clean = series.where(series > 0)
            values = np.log(clean).replace([np.inf, -np.inf], np.nan).diff()
        elif transform in ("log_level", "log", "log-price", "log_price"):
            clean = series.where(series > 0)
            values = np.log(clean).replace([np.inf, -np.inf], np.nan)
        elif transform in ("pct", "return", "pct_change"):
            values = series.pct_change(fill_method=None)
        else:
            values = series.diff()
        transformed[str(column)] = values
    frame = pd.concat(transformed, axis=1, join="outer").sort_index()
    # Keep pairwise-complete rows for each tested symbol pair later.
    return frame.dropna(how="all")


def _normalize_correlation_method(value: str) -> str | None:
    text = str(value).strip().lower()
    if not text:
        return None
    return _CORRELATION_METHOD_ALIASES.get(text)


def _normalize_transform_name(value: str) -> str | None:
    text = str(value).strip().lower()
    if not text:
        return None
    return _TRANSFORM_ALIASES.get(text)


def _normalize_cointegration_transform(value: str) -> str | None:
    text = str(value).strip().lower()
    if not text:
        return None
    return _COINTEGRATION_TRANSFORM_ALIASES.get(text)


def _normalize_cointegration_trend(value: str) -> str | None:
    text = str(value).strip().lower()
    if not text:
        return None
    return _COINTEGRATION_TREND_ALIASES.get(text)


def _causal_transform_reason(tool: str, transform: str) -> str:
    transform_value = str(transform or "").strip().lower()
    if tool == "cointegration_test":
        if transform_value == "log_level":
            return "Cointegration tests price-level relationships; log_level preserves levels while reducing scale effects."
        return "Cointegration tests price-level relationships, so level-style transforms are used."
    if transform_value == "log_return":
        return "Return transforms compare co-movement and predictive links without shared price-scale effects."
    if transform_value == "pct":
        return "Percentage returns compare relative movement across different price scales."
    if transform_value == "diff":
        return "First differences remove level drift before pairwise relationship tests."
    return "Level transform keeps raw price levels; use for level relationships, not return co-movement."


def _pair_transform_comparability(tool: str, transform: str) -> Dict[str, List[str]]:
    """Describe which pair analytics defaults answer the same transformed question."""
    transform_value = str(transform or "").strip().lower()
    if transform_value in {"log_return", "pct", "diff"}:
        comparable = [
            name
            for name in (
                "correlation_matrix(default=log_return)",
                "causal_discover_signals(default=log_return)",
                "trade_var_cvar_calculate(default=log_return)",
            )
            if not name.startswith(str(tool or ""))
        ]
        return {
            "comparable_with": comparable,
            "not_comparable_with": ["cointegration_test(default=log_level)"],
        }
    comparable = ["cointegration_test(default=log_level)"]
    if str(tool or "") == "cointegration_test":
        comparable = []
    return {
        "comparable_with": comparable,
        "not_comparable_with": [
            "correlation_matrix(default=log_return)",
            "causal_discover_signals(default=log_return)",
            "trade_var_cvar_calculate(default=log_return)",
        ],
    }


def _pair_transform_guidance(
    tool: str,
    transform: str,
    *,
    detail: str,
) -> Dict[str, Any]:
    if detail in {"compact", "summary"}:
        return {}
    return {
        "transform_reason": _causal_transform_reason(tool, transform),
        **_pair_transform_comparability(tool, transform),
    }


def _standardize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    cols = list(frame.columns)
    numeric = frame.astype(float)
    means = numeric.mean(axis=0, skipna=True)
    stds = numeric.std(axis=0, ddof=0, skipna=True)
    standardized = (numeric - means) / stds.replace(0.0, np.nan)
    standardized = standardized.reindex(columns=cols)

    # Preserve prior semantics for constant columns.
    for col in cols:
        series = frame[col]
        std = float(series.std(ddof=0))
        if not math.isfinite(std) or std == 0.0:
            standardized[col] = series
    return standardized


def _transform_cointegration_frame(frame: pd.DataFrame, transform: str) -> pd.DataFrame:
    mode = _normalize_cointegration_transform(transform)
    numeric = frame.astype(float)
    if mode == "log_level":
        clean = numeric.where(numeric > 0)
        clean = clean.mask(clean <= 0)
        logged = np.log(clean)
        logged = logged.replace([np.inf, -np.inf], np.nan)
        return logged.dropna(how="all")
    return numeric


def _build_pairwise_frame(
    series_map: Dict[str, pd.Series],
    symbols: List[str],
) -> pd.DataFrame:
    aligned_map = {
        symbol: series_map[symbol]
        for symbol in symbols
        if isinstance(series_map.get(symbol), pd.Series)
    }
    if len(aligned_map) < 2:
        return pd.DataFrame()
    return pd.concat(aligned_map, axis=1, join="outer").sort_index()


def _transform_aligned_pair(
    frame: pd.DataFrame,
    left: str,
    right: str,
    transform: str,
) -> pd.DataFrame:
    """Align a raw price pair before deriving pairwise transformed values."""
    raw_pair = frame[[left, right]].dropna(how="any")
    if raw_pair.empty:
        return raw_pair
    return _transform_frame(raw_pair, transform).dropna(how="any")


def _format_sample_time(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC")
    return format_datetime_utc(timestamp.to_pydatetime(), timespec="minutes")


def _series_timestamp_utc(value: Any) -> Optional[pd.Timestamp]:
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _last_completed_bar_close(
    series_map: Dict[str, pd.Series],
    timeframe: Any,
) -> Optional[str]:
    latest_open: pd.Timestamp | None = None
    for series in series_map.values():
        if series is None or series.empty:
            continue
        forming = bool(series.attrs.get("forming_candle_included"))
        index = series.index[:-1] if forming and len(series.index) > 1 else series.index
        if len(index) == 0:
            continue
        timestamp = _series_timestamp_utc(index[-1])
        if timestamp is None:
            continue
        if latest_open is None or timestamp > latest_open:
            latest_open = timestamp
    if latest_open is None:
        return None
    close_epoch = bar_close_epoch(float(latest_open.timestamp()), str(timeframe or ""))
    return format_epoch_utc(close_epoch)


def _analysis_time_contract(
    *,
    timeframe: Any,
    series_map: Optional[Dict[str, pd.Series]] = None,
    as_of: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "timezone": "UTC",
        "bar_timestamp_basis": "open_time",
        "resolved_as_of": format_datetime_utc(datetime.now(timezone.utc)),
    }
    requested = as_of or end
    if requested not in (None, ""):
        context["requested_as_of"] = requested
    mapping = series_map or {}
    resolved_values = [
        str(series.attrs.get("resolved_as_of"))
        for series in mapping.values()
        if series is not None and series.attrs.get("resolved_as_of") not in (None, "")
    ]
    if resolved_values:
        context["resolved_as_of"] = max(resolved_values)
    requested_values = [
        str(series.attrs.get("requested_as_of"))
        for series in mapping.values()
        if series is not None and series.attrs.get("requested_as_of") not in (None, "")
    ]
    if requested_values and "requested_as_of" not in context:
        context["requested_as_of"] = max(requested_values)
    data_as_of = _last_completed_bar_close(mapping, timeframe)
    if data_as_of:
        context["data_as_of"] = data_as_of
    return context


def _pairwise_analysis_context(
    rows: List[Dict[str, Any]],
    *,
    timeframe: Any,
    series_map: Optional[Dict[str, pd.Series]] = None,
    as_of: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "timeframe": str(timeframe),
        **_analysis_time_contract(
            timeframe=timeframe,
            series_map=series_map,
            as_of=as_of,
            end=end,
        ),
    }
    starts = [
        str(row.get("period_start"))
        for row in rows
        if row.get("period_start") not in (None, "")
    ]
    ends = [
        str(row.get("period_end"))
        for row in rows
        if row.get("period_end") not in (None, "")
    ]
    samples = [
        int(row["samples"])
        for row in rows
        if row.get("samples") is not None
    ]
    if starts:
        context["period_start"] = min(starts)
    if ends:
        context["period_end"] = max(ends)
    if samples:
        samples_min = min(samples)
        samples_max = max(samples)
        if samples_min == samples_max:
            context["samples"] = samples_min
        else:
            context["samples_min"] = samples_min
            context["samples_max"] = samples_max
    return context


def _normalize_output_limit(limit: Optional[int]) -> tuple[int | None, str | None]:
    if limit is None:
        return None, None
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return None, "limit must be a positive integer."
    if value < 1:
        return None, "limit must be a positive integer."
    return value, None


def _normalize_output_offset(offset: int) -> tuple[int, str | None]:
    try:
        value = int(offset)
    except (TypeError, ValueError):
        return 0, "offset must be a non-negative integer."
    if value < 0:
        return 0, "offset must be a non-negative integer."
    return value, None


def _limit_pair_rows(
    rows: List[Dict[str, Any]],
    limit: int | None,
    offset: int = 0,
) -> tuple[List[Dict[str, Any]], bool, Dict[str, Any]]:
    total = int(len(rows))
    start = min(max(0, int(offset)), total)
    if limit is None:
        page = rows[start:]
    else:
        page = rows[start : start + int(limit)]
    has_more = bool(start + len(page) < total)
    truncated = bool(start > 0 or has_more)
    pagination = build_pagination_meta(
        total=total,
        returned=len(page),
        offset=start,
        limit=limit,
    )
    return page, truncated, {"pagination": pagination}


def _public_pair_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if key == "left":
            out["symbol1"] = value
        elif key == "right":
            out["symbol2"] = value
        else:
            out[key] = value
    return out


def _pair_highlight_ref(
    row: Dict[str, Any],
    *,
    metrics: tuple[str, ...],
) -> Dict[str, Any]:
    left = str(row.get("left") or "")
    right = str(row.get("right") or "")
    out: Dict[str, Any] = {
        "pair": f"{left}-{right}",
        "symbol1": left,
        "symbol2": right,
    }
    for key in metrics:
        if key in row:
            out[key] = row.get(key)
    if "samples" in row:
        out["samples"] = row.get("samples")
    return out


def _format_pair_overlap_details(
    pair_overlaps: Dict[str, int],
    minimum_required: int,
) -> List[str]:
    return [
        f"{pair_key}: {int(rows)} rows (minimum {int(minimum_required)} required)"
        for pair_key, rows in sorted(
            pair_overlaps.items(), key=lambda kv: (kv[1], kv[0])
        )
    ]


def _causal_error(
    message: str,
    *,
    code: str,
    meta: Dict[str, Any],
    warnings: List[str] | None = None,
    details: List[str] | None = None,
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "success": False,
        "error": str(message),
        "error_code": str(code),
        "meta": _causal_contract_meta(meta),
    }
    if warnings:
        out["warnings"] = warnings
    if details:
        out["details"] = details
    if context:
        out["context"] = context
    return out


def _causal_history_range_error(
    start: Optional[str],
    end: Optional[str],
    *,
    meta: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    range_error = validate_historical_range(start, end)
    if range_error is None:
        return None
    out = _causal_error(
        str(range_error["error"]),
        code=str(range_error["error_code"]),
        meta=meta,
    )
    if range_error.get("details") is not None:
        out["details"] = range_error["details"]
    if range_error.get("remediation"):
        out["remediation"] = range_error["remediation"]
    return out


def _causal_contract_meta(
    meta: Dict[str, Any],
    *,
    legends: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta_in = dict(meta or {})
    tool_name = str(meta_in.pop("_tool", "") or "").strip()
    request_keys_raw = meta_in.pop("_request_keys", ())
    request_keys = {
        str(key)
        for key in (
            request_keys_raw
            if isinstance(request_keys_raw, (set, frozenset, list, tuple))
            else ()
        )
    }

    request: Dict[str, Any] = {}
    stats: Dict[str, Any] = {}
    for key, value in meta_in.items():
        if value is None:
            continue
        if key in request_keys:
            request[key] = value
        else:
            stats[key] = value

    out: Dict[str, Any] = {
        "tool": tool_name,
        "request": request,
        "runtime": {},
    }
    if stats:
        out["stats"] = stats
    if legends:
        out["legends"] = legends
    return out


def _format_overlap_details(
    symbol_rows: Dict[str, int],
    aligned_rows: int,
    minimum_required: int,
) -> str:
    parts: List[str] = []
    for symbol, count in symbol_rows.items():
        parts.append(f"{symbol}: {int(count)} rows")
    parts.append(
        f"aligned: {int(aligned_rows)} (minimum {int(minimum_required)} required)"
    )
    return ", ".join(parts)


def _pair_overlap_counts(
    series_map: Dict[str, pd.Series], symbols: List[str]
) -> Dict[str, int]:
    overlaps: Dict[str, int] = {}
    for i, left in enumerate(symbols):
        left_series = series_map.get(left)
        if not isinstance(left_series, pd.Series):
            continue
        left_idx = left_series.dropna().index
        for right in symbols[i + 1 :]:
            right_series = series_map.get(right)
            if not isinstance(right_series, pd.Series):
                continue
            right_idx = right_series.dropna().index
            key = f"{left}-{right}"
            overlaps[key] = int(len(left_idx.intersection(right_idx)))
    return overlaps


def _build_overlap_frame(
    series_map: Dict[str, pd.Series],
    symbols: List[str],
    limit: int,
) -> pd.DataFrame:
    aligned_map = {
        symbol: series_map[symbol]
        for symbol in symbols
        if isinstance(series_map.get(symbol), pd.Series)
    }
    if len(aligned_map) < 2:
        return pd.DataFrame()
    return pd.concat(aligned_map, axis=1, join="inner").tail(limit)


def _pair_overlap_symbols(
    pair_key: str, symbols: List[str] | None = None
) -> tuple[str, str]:
    text = str(pair_key)
    if symbols:
        ordered = sorted(
            {str(symbol) for symbol in symbols if symbol}, key=len, reverse=True
        )
        for left in ordered:
            prefix = f"{left}-"
            if not text.startswith(prefix):
                continue
            right = text[len(prefix) :]
            if right in ordered:
                return left, right
    left, _, right = text.partition("-")
    return left, right


def _build_alignment_detail(
    symbol_rows: Dict[str, int],
    pair_overlaps: Dict[str, int],
    aligned_rows: int,
    minimum_required: int,
) -> Optional[Dict[str, Any]]:
    if not symbol_rows or not pair_overlaps:
        return None
    min_symbol_rows = int(min(symbol_rows.values()))
    if min_symbol_rows <= 0:
        return None
    shrinkage_ratio = float(aligned_rows) / float(min_symbol_rows)
    if (
        aligned_rows >= minimum_required
        and shrinkage_ratio >= _MIN_PAIR_ALIGNMENT_FRACTION
    ):
        return None
    bottleneck_pair = min(pair_overlaps.items(), key=lambda kv: kv[1])
    return {
        "pair_overlaps": pair_overlaps,
        "bottleneck_pair": str(bottleneck_pair[0]),
        "bottleneck_rows": int(bottleneck_pair[1]),
        "aligned_rows": int(aligned_rows),
        "min_symbol_rows": int(min_symbol_rows),
        "shrinkage_ratio": float(shrinkage_ratio),
    }


def _pair_alignment_diagnostics(
    symbol_rows: Dict[str, int],
    pair_overlaps: Dict[str, int],
    symbols: List[str],
) -> Dict[str, Any]:
    pairs: Dict[str, Dict[str, Any]] = {}
    for pair, aligned_samples in pair_overlaps.items():
        left, right = _pair_overlap_symbols(pair, symbols)
        left_samples = int(symbol_rows.get(left, 0))
        right_samples = int(symbol_rows.get(right, 0))
        reference_samples = max(left_samples, right_samples)
        alignment_loss_pct = (
            max(
                0.0,
                (1.0 - (float(aligned_samples) / float(reference_samples)))
                * 100.0,
            )
            if reference_samples > 0
            else 0.0
        )
        pairs[str(pair)] = {
            "series_a": left,
            "series_b": right,
            "raw_samples_series_a": left_samples,
            "raw_samples_series_b": right_samples,
            "aligned_samples": int(aligned_samples),
            "alignment_loss_pct": round(alignment_loss_pct, 2),
        }
    return {
        "warning_threshold_pct": _ALIGNMENT_WARNING_THRESHOLD_PCT,
        "loss_reference": "larger_transformed_input_series",
        "pairs": pairs,
    }


def _public_alignment_diagnostics(
    diagnostics: Dict[str, Any],
    *,
    detail: str,
) -> Dict[str, Any]:
    """Return global alignment summary; include per-pair rows only in full detail."""
    pairs = diagnostics.get("pairs") if isinstance(diagnostics, dict) else {}
    if not isinstance(pairs, dict):
        pairs = {}
    losses = [
        float(row.get("alignment_loss_pct") or 0.0)
        for row in pairs.values()
        if isinstance(row, dict)
    ]
    threshold = float(
        (diagnostics or {}).get("warning_threshold_pct")
        or _ALIGNMENT_WARNING_THRESHOLD_PCT
    )
    summary: Dict[str, Any] = {
        "warning_threshold_pct": (diagnostics or {}).get("warning_threshold_pct"),
        "loss_reference": (diagnostics or {}).get("loss_reference"),
        "pairs_checked": int(len(pairs)),
        "max_alignment_loss_pct": round(max(losses), 2) if losses else 0.0,
        "pairs_above_warning": int(sum(1 for loss in losses if loss > threshold)),
    }
    if str(detail or "").strip().lower() == "full":
        summary["pairs"] = pairs
    return summary


def _pair_alignment_warning(diagnostics: Dict[str, Any]) -> Optional[str]:
    pairs = diagnostics.get("pairs")
    if not isinstance(pairs, dict):
        return None
    offenders = [
        (str(pair), row)
        for pair, row in pairs.items()
        if isinstance(row, dict)
        and float(row.get("alignment_loss_pct") or 0.0)
        > _ALIGNMENT_WARNING_THRESHOLD_PCT
    ]
    if not offenders:
        return None
    examples = ", ".join(
        f"{pair} {float(row['alignment_loss_pct']):.2f}% "
        f"({int(row['aligned_samples'])}/"
        f"{max(int(row['raw_samples_series_a']), int(row['raw_samples_series_b']))} retained)"
        for pair, row in offenders[:5]
    )
    suffix = "" if len(offenders) <= 5 else f"; plus {len(offenders) - 5} more pair(s)"
    return (
        "Timestamp alignment discarded more than 5% of at least one pair's "
        f"larger transformed input series: {examples}{suffix}. Session-calendar "
        "or holiday gaps can bias these results."
    )


def _format_alignment_detail_summary(detail: Dict[str, Any]) -> str:
    pair_overlaps = detail.get("pair_overlaps")
    if not isinstance(pair_overlaps, dict):
        return ""
    pair_str = ", ".join(f"{k}: {int(v)}" for k, v in pair_overlaps.items())
    bottleneck_pair = str(detail.get("bottleneck_pair") or "")
    bottleneck_rows = detail.get("bottleneck_rows")
    suffix = ""
    if bottleneck_pair and bottleneck_rows is not None:
        suffix = f"; bottleneck={bottleneck_pair} ({int(bottleneck_rows)} rows)"
    return f"pair_overlaps: {pair_str}{suffix}"
