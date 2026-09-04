import logging
from calendar import monthrange
from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pydantic import Field

from ..services.data_service.candles import (
    _drop_incomplete_tail_df,
    _parse_candle_calendar_bound,
)
from ..shared.constants import CALENDAR_TIMEFRAMES, TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..shared.schema import DetailLiteral, TimeframeLiteral
from ..shared.symbols import (
    is_probably_crypto_symbol,
    is_probably_fx_session_symbol,
)
from ..shared.validators import (
    invalid_timeframe_error,
    unsupported_timeframe_seconds_error,
)
from ..utils.coercion import safe_float as _safe_float
from ..utils.freshness import (
    COMPLETED_BAR_FRESHNESS_KEYS,
    completed_bar_freshness_fields,
)
from ..utils.mt5 import (
    MT5ConnectionError,
    _mt5_copy_rates_from,
    _mt5_copy_rates_range,
    _symbol_ready_guard,
    ensure_mt5_connection_or_raise,
    get_symbol_info_cached,
    mt5,
    resolve_public_symbol,
)
from ..utils.sessions import (
    market_session_label as _market_session_label,
)
from ..utils.sessions import (
    session_definition_for_clock as _session_definition_for_clock,
)
from ..utils.time import (
    _broker_calendar_timezone,
    _format_datetime_minute_explicit,
    _resolve_client_tz,
    format_epoch_utc,
    timezone_label,
)
from ..utils.utils import (
    _parse_end_datetime,
    _parse_start_datetime,
    validate_historical_range,
)
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .mt5_gateway import create_mt5_gateway
from .output_contract import build_pagination_meta, normalize_output_detail
from .runtime_metadata import run_mt5_logged_operation

logger = logging.getLogger(__name__)


_TEMPORAL_RELIABLE_GROUP_BARS = 30
_TEMPORAL_MIN_MONTH_INSTANCES = 2
_TEMPORAL_SAMPLE_WARNING_LIMIT = 10
_PERCENT_UNIT = "percent (1.0 = 1%)"
_TEMPORAL_DEFAULT_LOOKBACK_DAYS = {
    "dow": 210,
    "hour": 60,
    "session": 60,
    "month": 730,
    "all": 365,
}
_TEMPORAL_DEFAULT_LOOKBACK_FLOOR = 200
_TEMPORAL_DEFAULT_LOOKBACK_CAP = 20_000


_DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
_PAGINATION_FIELDS = (
    "total",
    "returned",
    "offset",
    "limit",
    "has_more",
    "more_available",
)
_SESSION_ORDER = {
    "asia": 0,
    "london": 1,
    "london_ny_overlap": 2,
    "ny": 3,
    "off_hours": 4,
    "off_session": 4,
}
_SESSION_CALENDAR_VALUES = frozenset({"auto", "fx", "equity", "continuous_24_7"})
_INTRADAY_GROUP_BY = frozenset({"hour", "session"})
_ALL_GROUP_DIMENSIONS = ("dow", "hour", "month", "session")
_CALENDAR_TF_GROUP_DIMENSIONS = ("dow", "month")


def _fx_trading_weekday(value: Any) -> int:
    """Label the 17:00 New York FX session as the following trading day."""
    local = value.tz_convert(ZoneInfo("America/New_York"))
    weekday = int(local.weekday())
    return (weekday + (1 if int(local.hour) >= 17 else 0)) % 7


_STAGE_ERROR_CODES = {
    "validate": "temporal_invalid_input",
    "symbol": "temporal_symbol_failed",
    "fetch": "temporal_fetch_failed",
    "process": "temporal_process_failed",
    "trim": "temporal_insufficient_data",
    "filter": "temporal_insufficient_data",
    "internal": "temporal_internal_error",
}


def _error_response(
    message: str,
    stage: str,
    *,
    context: Optional[Dict[str, Any]] = None,
    details: Optional[Any] = None,
    bars: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
    code: Optional[str] = None,
) -> Dict[str, Any]:
    error_code = str(code or _STAGE_ERROR_CODES.get(stage) or f"temporal_{stage}")
    payload = build_error_payload(
        message,
        code=error_code,
        operation="temporal_analyze",
    )
    payload["stage"] = stage
    if context:
        payload["context"] = context
    if details is not None:
        payload["details"] = details
    if bars is not None:
        payload["bars"] = int(bars)
    if filters is not None:
        payload["filters"] = filters
    return payload


def _normalize_group_by(value: Optional[str]) -> str:
    if value is None:
        return "dow"
    v = str(value).strip().lower()
    if v in ("day_of_week", "weekday", "week_day", "day", "daily", "dow", "wday"):
        return "dow"
    if v in ("hour", "hours", "hr", "time", "time_of_day"):
        return "hour"
    if v in ("month", "months", "mo"):
        return "month"
    if v in ("session", "sessions", "market_session", "trading_session"):
        return "session"
    if v in ("all", "none", "overall"):
        return "all"
    return v


def _resolve_session_calendar(
    session_calendar: str,
    symbol: str,
    *,
    path: Any = None,
) -> str:
    value = str(session_calendar or "auto").strip().lower()
    if value != "auto":
        return value
    if is_probably_crypto_symbol(symbol):
        return "continuous_24_7"
    if is_probably_fx_session_symbol(symbol, path=path):
        return "fx"
    return "equity"


def _default_temporal_lookback(timeframe: str, group_by: str) -> int:
    seconds_per_bar = TIMEFRAME_SECONDS.get(str(timeframe or "").strip().upper())
    if not seconds_per_bar:
        return _TEMPORAL_DEFAULT_LOOKBACK_FLOOR
    days = _TEMPORAL_DEFAULT_LOOKBACK_DAYS.get(
        str(group_by or "").strip().lower(),
        _TEMPORAL_DEFAULT_LOOKBACK_DAYS["dow"],
    )
    bars = int((int(days) * 86_400 + int(seconds_per_bar) - 1) // int(seconds_per_bar))
    return max(
        _TEMPORAL_DEFAULT_LOOKBACK_FLOOR,
        min(_TEMPORAL_DEFAULT_LOOKBACK_CAP, bars),
    )


def _temporal_edge_status(rows: Any, best: Dict[str, Any]) -> str:
    means = [
        float(row.get("avg_return_pct") or 0.0)
        for row in rows
        if isinstance(row, dict) and row.get("avg_return_pct") is not None
    ]
    if len(means) < 2:
        return "single_group"
    ranked = sorted(means, reverse=True)
    gap = ranked[0] - ranked[1]
    scale = max(abs(value) for value in means)
    if scale < 0.01 or gap < max(1e-4, 0.25 * scale):
        return "not_distinguished"
    return "highest_sample_mean"


def _temporal_row_is_reliable(row: Dict[str, Any]) -> bool:
    if int(row.get("bars", 0) or 0) < _TEMPORAL_RELIABLE_GROUP_BARS:
        return False
    minimum_instances = int(row.get("minimum_period_instances", 0) or 0)
    if minimum_instances <= 0:
        return True
    return int(row.get("distinct_period_instances", 0) or 0) >= minimum_instances


def _reliable_best(rows: Any) -> Optional[Dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if (
            isinstance(row, dict)
            and row.get("avg_return_pct") is not None
        )
    ]
    materialized = [row for row in candidates if _temporal_row_is_reliable(row)]
    best = max(
        materialized,
        key=lambda row: float(row.get("avg_return_pct") or 0.0),
        default=None,
    )
    if best is None:
        return None
    annotated = dict(best)
    annotated["rank_basis"] = "highest_sample_mean_among_reliable_groups"
    annotated["ranking_filter"] = {
        "minimum_bars": _TEMPORAL_RELIABLE_GROUP_BARS,
        "eligible_groups": [
            row.get("group_label", row.get("group")) for row in materialized
        ],
        "excluded_groups": [
            row.get("group_label", row.get("group"))
            for row in candidates
            if row not in materialized
        ],
    }
    annotated["edge_status"] = _temporal_edge_status(materialized, annotated)
    return annotated


def _format_epoch_in_timezone(epoch_seconds: float, analysis_tz: Any) -> str:
    value = datetime.fromtimestamp(epoch_seconds, tz=dt_timezone.utc).astimezone(
        analysis_tz
    )
    return _format_datetime_minute_explicit(value)


def _broker_session_date(epoch_seconds: Any) -> date:
    opened_at = datetime.fromtimestamp(float(epoch_seconds), tz=dt_timezone.utc)
    return opened_at.astimezone(_broker_calendar_timezone(opened_at)).date()


def _parse_mapped_value(
    value: Optional[str],
    *,
    minimum: int,
    maximum: int,
    mapping: Dict[str, int],
    numeric_aliases: Optional[Dict[int, int]] = None,
) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text.isdigit():
        num = int(text)
        if numeric_aliases and num in numeric_aliases:
            return numeric_aliases[num]
        if minimum <= num <= maximum:
            return num
        return None
    return mapping.get(text)


def _parse_weekday(value: Optional[str]) -> Optional[int]:
    mapping = {
        "mon": 0, "monday": 0,
        "tue": 1, "tues": 1, "tuesday": 1,
        "wed": 2, "wednesday": 2,
        "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
        "fri": 4, "friday": 4,
        "sat": 5, "saturday": 5,
        "sun": 6, "sunday": 6,
    }
    return _parse_mapped_value(
        value,
        minimum=0,
        maximum=6,
        mapping=mapping,
    )


def _parse_month(value: Optional[str]) -> Optional[int]:
    mapping = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    return _parse_mapped_value(
        value,
        minimum=1,
        maximum=12,
        mapping=mapping,
    )


def _parse_time_token(token: str) -> Optional[int]:
    text = token.strip()
    if not text:
        return None
    parts = text.split(":")
    if not parts:
        return None
    try:
        hour = int(parts[0])
    except Exception:
        return None
    minute = 0
    if len(parts) > 1:
        try:
            minute = int(parts[1])
        except Exception:
            return None
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return hour * 60 + minute


def _parse_time_range(value: Optional[str]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    if value is None:
        return None, None, None
    text = str(value).strip().lower()
    if not text:
        return None, None, None
    if "to" in text:
        parts = [p.strip() for p in text.split("to") if p.strip()]
    elif "-" in text:
        parts = [p.strip() for p in text.split("-") if p.strip()]
    else:
        return None, None, "Invalid time_range. Use 'HH:MM-HH:MM'."
    if len(parts) != 2:
        return None, None, "Invalid time_range. Use 'HH:MM-HH:MM'."
    start_min = _parse_time_token(parts[0])
    end_min = _parse_time_token(parts[1])
    if start_min is None or end_min is None:
        return None, None, "Invalid time_range. Use 'HH:MM-HH:MM'."
    if start_min == end_min:
        return None, None, "time_range start and end must differ."
    return start_min, end_min, None


def _time_label(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _rounded_temporal_float(value: Any, *, digits: int = 6) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    return round(float(number), digits)


_WEEKEND_GAP_SECONDS = 24 * 3600


def _session_gap_mask(
    epochs: pd.Series,
    timeframe: str,
    *,
    min_gap_seconds: Optional[float] = None,
) -> pd.Series:
    expected = TIMEFRAME_SECONDS.get(str(timeframe or "").strip().upper())
    if not expected:
        return pd.Series(False, index=epochs.index)
    delta = pd.to_numeric(epochs, errors="coerce").diff()
    threshold = float(expected) * 1.5
    if min_gap_seconds is not None:
        threshold = max(threshold, float(min_gap_seconds))
    return delta > threshold


def _stats_for_group(df: pd.DataFrame, volume_col: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "bars": int(len(df)),
    }
    ret = pd.to_numeric(df.get("__return"), errors="coerce")
    gap = (
        df["__session_gap"].fillna(False).astype(bool)
        if "__session_gap" in df.columns
        else pd.Series(False, index=df.index)
    )
    gap_count = int((gap & ret.notna()).sum())
    out["session_gap_observations"] = gap_count
    ret = ret[ret.notna() & ~gap]
    n = int(ret.shape[0])
    out["return_observations"] = n
    if n > 0:
        out["avg_return_pct"] = _rounded_temporal_float(ret.mean())
        out["median_return_pct"] = _rounded_temporal_float(ret.median())
        out["volatility_pct"] = _rounded_temporal_float(ret.std(ddof=0))
        out["avg_abs_return_pct"] = _rounded_temporal_float(ret.abs().mean())
        win_rate = _safe_float(round((ret > 0).sum() / float(n), 4))
        out["win_rate"] = win_rate
        out["win_rate_pct"] = (
            _safe_float(round(float(win_rate) * 100.0, 2))
            if win_rate is not None
            else None
        )
    else:
        out["avg_return_pct"] = None
        out["median_return_pct"] = None
        out["volatility_pct"] = None
        out["avg_abs_return_pct"] = None
        out["win_rate"] = None
        out["win_rate_pct"] = None

    if "__range" in df.columns:
        rng = pd.to_numeric(df["__range"], errors="coerce")
        rng = rng[pd.notna(rng)]
        out["avg_range"] = (
            _rounded_temporal_float(rng.mean(), digits=8) if len(rng) else None
        )
    if "__range_pct" in df.columns:
        rngp = pd.to_numeric(df["__range_pct"], errors="coerce")
        rngp = rngp[pd.notna(rngp)]
        out["avg_range_pct"] = _rounded_temporal_float(rngp.mean()) if len(rngp) else None
    if volume_col and volume_col in df.columns:
        vol = pd.to_numeric(df[volume_col], errors="coerce")
        vol = vol[pd.notna(vol)]
        out["avg_volume"] = (
            _rounded_temporal_float(vol.mean(), digits=2) if len(vol) else None
        )
    return out


def _month_period_instance_metadata(
    group: pd.DataFrame,
    window: pd.DataFrame,
) -> Dict[str, Any]:
    def _dates(frame: pd.DataFrame) -> List[date]:
        source = (
            frame["__session_date"]
            if "__session_date" in frame.columns
            else frame["__dt"]
        )
        dates: List[date] = []
        for value in source:
            if pd.isna(value):
                continue
            if isinstance(value, datetime):
                dates.append(value.date())
            elif isinstance(value, date):
                dates.append(value)
            else:
                dates.append(pd.Timestamp(value).date())
        return dates

    group_dates = _dates(group)
    window_dates = _dates(window)
    instances = {(value.year, value.month) for value in group_dates}
    partial_instances: set[Tuple[int, int]] = set()
    if window_dates:
        first = min(window_dates)
        last = max(window_dates)
        if first.day != 1:
            partial_instances.add((first.year, first.month))
        if last.day != monthrange(last.year, last.month)[1]:
            partial_instances.add((last.year, last.month))
    partial_count = len(instances & partial_instances)
    distinct_count = len(instances)
    return {
        "distinct_period_instances": distinct_count,
        "complete_period_instances": max(0, distinct_count - partial_count),
        "partial_period_instances": partial_count,
        "partial_bucket": partial_count > 0,
        "minimum_period_instances": _TEMPORAL_MIN_MONTH_INSTANCES,
    }


def _compact_temporal_stats(
    row: Dict[str, Any],
    *,
    include_group: bool = True,
) -> Dict[str, Any]:
    keys = (
        "group_label",
        "bars",
        "return_observations",
        "avg_return_pct",
        "median_return_pct",
        "win_rate_pct",
        "volatility_pct",
        "distinct_period_instances",
        "complete_period_instances",
        "partial_period_instances",
        "partial_bucket",
        "minimum_period_instances",
        "session_gap_observations",
    )
    if include_group and row.get("group") is not None:
        keys = ("group", *keys)
    return {key: row.get(key) for key in keys if row.get(key) is not None}


def _standard_temporal_stats(row: Dict[str, Any]) -> Dict[str, Any]:
    out = _compact_temporal_stats(row, include_group=True)
    for key in ("avg_abs_return_pct", "avg_range_pct", "avg_volume"):
        value = row.get(key)
        if value is not None:
            out[key] = value
    return out


def _paginate_temporal_rows(
    rows: List[Dict[str, Any]],
    *,
    limit: Optional[int],
    offset: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    total_count = len(rows)
    paged = rows[offset:]
    if limit is not None:
        paged = paged[:limit]

    if limit is None and offset == 0:
        return paged, {}

    return paged, {
        "pagination": build_pagination_meta(
            total=total_count,
            returned=len(paged),
            offset=offset,
            limit=limit,
        )
    }


def _copy_pagination_meta(source: Dict[str, Any], target: Dict[str, Any]) -> None:
    pagination = source.get("pagination")
    if not isinstance(pagination, dict):
        return
    for key in _PAGINATION_FIELDS:
        if key in pagination:
            target[key] = pagination[key]


def _flatten_temporal_dimension_groups(
    groups: List[Any],
    *,
    formatter: Any,
    best_keys: Tuple[str, ...],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
]:
    flat_groups: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []
    pagination: Dict[str, Dict[str, Any]] = {}
    for item in groups:
        if not isinstance(item, dict):
            continue
        dimension = str(item.get("dimension") or "")
        breakdown = item.get("breakdown")
        if not dimension or not isinstance(breakdown, list):
            continue
        formatted_rows = [
            formatter(row)
            for row in breakdown
            if isinstance(row, dict)
        ]
        for row in formatted_rows:
            flat_row = {"dimension": dimension}
            flat_row.update(row)
            flat_groups.append(flat_row)
        page_meta: Dict[str, Any] = {}
        _copy_pagination_meta(item, page_meta)
        if page_meta:
            pagination[dimension] = page_meta
        best = _reliable_best(formatted_rows)
        if best:
            best_row = {"dimension": dimension}
            best_row.update(
                {
                    key: best[key]
                    for key in best_keys
                    if key in best
                }
            )
            best_rows.append(best_row)
    return flat_groups, best_rows, pagination


def _temporal_sample_warnings(groups: Any) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    def _add_row(row: Dict[str, Any], *, dimension: Optional[str] = None) -> None:
        bars = int(row.get("bars", 0) or 0)
        minimum_instances = int(row.get("minimum_period_instances", 0) or 0)
        distinct_instances = int(row.get("distinct_period_instances", 0) or 0)
        bars_too_small = bars < _TEMPORAL_RELIABLE_GROUP_BARS
        cycles_too_small = (
            minimum_instances > 0 and distinct_instances < minimum_instances
        )
        if not bars_too_small and not cycles_too_small:
            return
        warning = {
            "group_label": row.get("group_label"),
            "bars": bars,
        }
        if bars_too_small:
            warning["recommended_min_bars"] = _TEMPORAL_RELIABLE_GROUP_BARS
        if cycles_too_small:
            warning["distinct_period_instances"] = distinct_instances
            warning["recommended_min_period_instances"] = minimum_instances
            warning["partial_bucket"] = bool(row.get("partial_bucket"))
        if dimension:
            warning["dimension"] = dimension
        if row.get("group") is not None:
            warning["group"] = row.get("group")
        rows.append(warning)

    if isinstance(groups, list):
        for item in groups:
            if not isinstance(item, dict):
                continue
            breakdown = item.get("breakdown")
            if isinstance(breakdown, list):
                dimension = str(item.get("dimension") or "")
                for row in breakdown:
                    if isinstance(row, dict):
                        _add_row(row, dimension=dimension or None)
            else:
                _add_row(item)

    if not rows:
        return {}
    shown = rows[:_TEMPORAL_SAMPLE_WARNING_LIMIT]
    omitted = max(0, len(rows) - len(shown))
    out = {
        "sample_warnings": shown,
        "sample_warning_count": len(rows),
        "sample_notice": (
            "Some temporal groups lack enough bars or repeated calendar instances "
            "for reliable ranking; increase lookback or set min_bars for stricter "
            "filtering."
        ),
    }
    if omitted:
        out["sample_warnings_omitted"] = omitted
    return out


def _temporal_group_count(groups: Any) -> int:
    if not isinstance(groups, list):
        return 0
    total = 0
    for item in groups:
        if not isinstance(item, dict):
            continue
        breakdown = item.get("breakdown")
        if isinstance(breakdown, list):
            total += len([row for row in breakdown if isinstance(row, dict)])
        else:
            total += 1
    return total


def _temporal_best_summary(groups: Any) -> Any:
    if not isinstance(groups, list):
        return None
    best_rows: List[Dict[str, Any]] = []
    for item in groups:
        if not isinstance(item, dict):
            continue
        breakdown = item.get("breakdown")
        if not isinstance(breakdown, list):
            continue
        best = _reliable_best(breakdown)
        if best is not None:
            best_rows.append({"dimension": item.get("dimension"), **best})
    if best_rows:
        return best_rows
    return _reliable_best(groups)


def _base_temporal_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "success",
            "symbol",
            "timeframe",
            "group_by",
            "return_mode",
            "return_basis",
            "return_definition",
            "session_gap_policy",
            "units",
            "timezone",
            "timezone_source",
            "timing_convention",
            "hour_span",
            "session_calendar",
            "session_calendar_source",
            "session_definition",
            "weekday_calendar",
            "weekday_definition",
            "month_calendar",
            "month_definition",
            "lookback",
            "lookback_source",
            "lookback_note",
            "bars",
            "start",
            "end",
            "groups_analyzed",
            "groups_excluded",
            "min_bars_applied",
            "analysis_window",
            "overall_basis",
            "analysis_status",
            "message",
            "pagination",
            "as_of",
            "data_as_of",
            "data_age_seconds",
            "data_stale",
            "stale_after_seconds",
            "freshness",
            "freshness_basis",
            "market_status",
            "market_status_reason",
            "history_policy_ok",
            "stale_warning",
            "note",
        )
        if key in payload
    }


def _drop_compact_temporal_duplicate_units(out: Dict[str, Any]) -> None:
    units = out.get("units")
    if not isinstance(units, dict):
        return
    compact_units = {
        key: value
        for key, value in units.items()
        if key != "win_rate"
    }
    out["units"] = compact_units


def _compact_temporal_payload(
    payload: Dict[str, Any], *, summary_groups: Any = None
) -> Dict[str, Any]:
    out: Dict[str, Any] = _base_temporal_payload(payload)
    _drop_compact_temporal_duplicate_units(out)
    groups = payload.get("groups")
    if isinstance(groups, list) and not groups:
        out["groups"] = []
    if isinstance(groups, list) and groups:
        if all(isinstance(row, dict) and "dimension" in row for row in groups):
            compact_groups, _page_best_rows, pagination = _flatten_temporal_dimension_groups(
                groups,
                formatter=_compact_temporal_stats,
                best_keys=(
                    "group",
                    "group_label",
                    "avg_return_pct",
                    "win_rate",
                    "win_rate_pct",
                    "rank_basis",
                    "edge_status",
                    "distinct_period_instances",
                    "complete_period_instances",
                    "partial_bucket",
                    "minimum_period_instances",
                    "ranking_filter",
                ),
            )
            out["groups"] = compact_groups
            _all_groups, best_rows, _all_pagination = _flatten_temporal_dimension_groups(
                summary_groups if isinstance(summary_groups, list) else groups,
                formatter=_compact_temporal_stats,
                best_keys=(
                    "group",
                    "group_label",
                    "avg_return_pct",
                    "win_rate",
                    "win_rate_pct",
                    "rank_basis",
                    "edge_status",
                    "distinct_period_instances",
                    "complete_period_instances",
                    "partial_bucket",
                    "minimum_period_instances",
                    "ranking_filter",
                ),
            )
            if best_rows:
                out["best"] = best_rows
            if pagination:
                out["dimension_pagination"] = pagination
        else:
            compact_groups = [
                _compact_temporal_stats(row)
                for row in groups
                if isinstance(row, dict)
            ]
            out["groups"] = compact_groups
            best_source = summary_groups if isinstance(summary_groups, list) else groups
            best = _reliable_best(
                _compact_temporal_stats(row)
                for row in best_source
                if isinstance(row, dict)
            )
            if best:
                out["best"] = {
                    key: best[key]
                    for key in (
                        "group",
                        "group_label",
                        "avg_return_pct",
                        "win_rate",
                        "win_rate_pct",
                        "rank_basis",
                        "edge_status",
                        "distinct_period_instances",
                        "complete_period_instances",
                        "partial_bucket",
                        "minimum_period_instances",
                        "ranking_filter",
                    )
                    if key in best
                }
    elif isinstance(payload.get("overall"), dict):
        out["overall"] = _compact_temporal_stats(payload["overall"])
    for key in (
        "warnings",
        "excluded_groups",
        "sample_warnings",
        "sample_warning_count",
        "sample_warnings_omitted",
        "sample_notice",
    ):
        value = payload.get(key)
        if value:
            out[key] = value
    return out


def _summary_temporal_payload(
    payload: Dict[str, Any], *, summary_groups: Any = None
) -> Dict[str, Any]:
    compact = _compact_temporal_payload(payload, summary_groups=summary_groups)
    out = {
        key: compact[key]
        for key in (
            "success",
            "symbol",
            "timeframe",
            "group_by",
            "return_mode",
            "return_basis",
            "return_definition",
            "session_gap_policy",
            "units",
            "timezone",
            "timezone_source",
            "timing_convention",
            "hour_span",
            "lookback",
            "lookback_source",
            "lookback_note",
            "bars",
            "start",
            "end",
            "groups_analyzed",
            "groups_excluded",
            "min_bars_applied",
            "as_of",
            "data_as_of",
            "data_age_seconds",
            "data_stale",
            "freshness",
            "market_status",
            "market_status_reason",
            "stale_warning",
            "note",
        )
        if key in compact
    }
    groups = compact.get("groups")
    if isinstance(groups, list):
        if all(isinstance(row, dict) and "dimension" in row for row in groups):
            pagination = compact.get("dimension_pagination")
            if isinstance(pagination, dict):
                out["group_counts"] = {
                    str(dimension): int(meta.get("total") or 0)
                    for dimension, meta in pagination.items()
                    if isinstance(meta, dict)
                }
            else:
                counts: Dict[str, int] = {}
                for row in groups:
                    dimension = str(row.get("dimension") or "")
                    counts[dimension] = counts.get(dimension, 0) + 1
                out["group_counts"] = counts
        else:
            out["group_count"] = len(groups)
    if compact.get("best") not in (None, "", [], {}):
        out["best"] = compact["best"]
    overall = payload.get("overall")
    if isinstance(overall, dict):
        out["overall"] = {
            key: overall.get(key)
            for key in ("bars", "avg_return_pct", "win_rate_pct", "volatility_pct")
            if overall.get(key) is not None
        }
    for key in (
        "warnings",
        "excluded_groups",
        "sample_warnings",
        "sample_warning_count",
        "sample_warnings_omitted",
        "sample_notice",
    ):
        value = payload.get(key)
        if value:
            out[key] = value
    return out


def _standard_temporal_payload(
    payload: Dict[str, Any], *, summary_groups: Any = None
) -> Dict[str, Any]:
    out: Dict[str, Any] = _base_temporal_payload(payload)
    groups = payload.get("groups")
    if isinstance(groups, list) and groups:
        best = None
        if all(isinstance(row, dict) and "dimension" in row for row in groups):
            standard_groups, _page_best_rows, pagination = _flatten_temporal_dimension_groups(
                groups,
                formatter=_standard_temporal_stats,
                best_keys=(
                    "group_label",
                    "avg_return_pct",
                    "win_rate",
                    "win_rate_pct",
                    "rank_basis",
                    "edge_status",
                    "distinct_period_instances",
                    "complete_period_instances",
                    "partial_bucket",
                    "minimum_period_instances",
                    "ranking_filter",
                ),
            )
            out["groups"] = standard_groups
            _all_groups, best_rows, _all_pagination = _flatten_temporal_dimension_groups(
                summary_groups if isinstance(summary_groups, list) else groups,
                formatter=_standard_temporal_stats,
                best_keys=(
                    "group_label",
                    "avg_return_pct",
                    "win_rate",
                    "win_rate_pct",
                    "rank_basis",
                    "edge_status",
                    "distinct_period_instances",
                    "complete_period_instances",
                    "partial_bucket",
                    "minimum_period_instances",
                    "ranking_filter",
                ),
            )
            if best_rows:
                out["best"] = best_rows
            if pagination:
                out["dimension_pagination"] = pagination
        else:
            standard_groups = [
                _standard_temporal_stats(row)
                for row in groups
                if isinstance(row, dict)
            ]
            out["groups"] = standard_groups
            best_source = summary_groups if isinstance(summary_groups, list) else groups
            best = _reliable_best(
                _standard_temporal_stats(row)
                for row in best_source
                if isinstance(row, dict)
            )
        if best:
            out["best"] = {
                key: best[key]
                for key in (
                    "group_label",
                    "avg_return_pct",
                    "win_rate",
                    "win_rate_pct",
                    "distinct_period_instances",
                    "complete_period_instances",
                    "partial_bucket",
                    "minimum_period_instances",
                    "ranking_filter",
                )
                if key in best
            }
    overall = payload.get("overall")
    if isinstance(overall, dict):
        out["overall"] = _standard_temporal_stats(overall)
    for key in (
        "volume_source",
        "filters",
        "warnings",
        "excluded_groups",
        "sample_warnings",
        "sample_warning_count",
        "sample_notice",
    ):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            out[key] = value
    return out


def _fetch_rates(
    symbol: str,
    timeframe: str,
    limit: int,
    start: Optional[str],
    end: Optional[str],
    gateway: Any = None,
) -> Tuple[Optional[Any], Optional[str]]:
    mt5_gateway = gateway or create_mt5_gateway(
        adapter=mt5,
        ensure_connection_impl=ensure_mt5_connection_or_raise,
    )
    if timeframe not in TIMEFRAME_MAP:
        return None, invalid_timeframe_error(timeframe, TIMEFRAME_MAP)
    mt5_tf = TIMEFRAME_MAP[timeframe]

    if start and end:
        try:
            start_dt = _parse_candle_calendar_bound(
                start,
                timeframe=timeframe,
                end_bound=False,
            ) or _parse_start_datetime(start)
            end_dt = _parse_candle_calendar_bound(
                end,
                timeframe=timeframe,
                end_bound=True,
            ) or _parse_end_datetime(end)
        except ValueError as exc:
            return None, str(exc)
        if not start_dt or not end_dt:
            return None, "Invalid start/end date format."
        if start_dt > end_dt:
            return None, "start must be before end."
        rates = _mt5_copy_rates_range(symbol, mt5_tf, start_dt, end_dt)
        return rates, None

    if start:
        try:
            start_dt = _parse_candle_calendar_bound(
                start,
                timeframe=timeframe,
                end_bound=False,
            ) or _parse_start_datetime(start)
        except ValueError as exc:
            return None, str(exc)
        if not start_dt:
            return None, "Invalid start date format."
        seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe)
        if not seconds_per_bar:
            return None, unsupported_timeframe_seconds_error(timeframe)
        end_dt = start_dt + timedelta(seconds=seconds_per_bar * max(int(limit), 1))
        rates = _mt5_copy_rates_range(symbol, mt5_tf, start_dt, end_dt)
        return rates, None

    if end:
        try:
            end_dt = _parse_candle_calendar_bound(
                end,
                timeframe=timeframe,
                end_bound=True,
            ) or _parse_end_datetime(end)
        except ValueError as exc:
            return None, str(exc)
        if not end_dt:
            return None, "Invalid end date format."
        rates = _mt5_copy_rates_from(symbol, mt5_tf, end_dt, int(limit))
        return rates, None

    tick = mt5_gateway.symbol_info_tick(symbol)
    if tick is not None and getattr(tick, "time", None):
        t_utc = float(tick.time)
        to_dt = datetime.fromtimestamp(t_utc, tz=dt_timezone.utc).replace(tzinfo=None)
    else:
        to_dt = datetime.now(dt_timezone.utc).replace(tzinfo=None)
    rates = _mt5_copy_rates_from(symbol, mt5_tf, to_dt, int(limit))
    return rates, None


@mcp.tool()
def temporal_analyze(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    lookback: Annotated[Optional[int], Field(ge=1)] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    group_by: Literal["dow", "hour", "month", "session", "all"] = "dow",
    session_calendar: Literal["auto", "fx", "equity", "continuous_24_7"] = "auto",
    day_of_week: Optional[str] = None,
    month: Optional[str] = None,
    time_range: Optional[str] = None,
    timezone: Optional[str] = None,
    return_mode: Literal["pct", "log"] = "pct",  # type: ignore
    return_basis: Literal["previous_close", "bar_open"] = "previous_close",
    min_bars: Optional[int] = None,
    limit: Annotated[Optional[int], Field(ge=1)] = None,
    offset: Annotated[int, Field(ge=0)] = 0,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Temporal analysis by day-of-week, hour, market session, or month.

    `lookback` controls the number of historical bars used when start/end are
    omitted. When omitted, the tool chooses a timeframe-aware seasonal window.

    Filters:
    - day_of_week: 0-6 or names like Mon, Tuesday
    - month: 1-12 or names like Jan, September
    - time_range: 'HH:MM-HH:MM' (start inclusive, end exclusive; uses
      `timezone`, then CLIENT_TZ, then UTC; wraps midnight like 22:00-02:00)
    - timezone: IANA clock for hour/session grouping and time_range, such as
      Europe/London or America/New_York
    - return_basis: previous_close includes gaps in the destination bar;
      bar_open measures only each candle's open-to-close move
    - min_bars: exclude grouped rows below this sample size. When omitted for
      day-of-week analysis, sparse weekend groups are auto-filtered.
    - session_calendar: with auto/fx, day-of-week uses the FX trading day that
      rolls at 17:00 America/New_York; equity and continuous_24_7 use civil
      analysis-timezone days (or broker session dates on D1/W1/MN1). auto
      selects fx, equity, or continuous_24_7 from the symbol.
    - limit/offset: for a single group_by, page that row list. For group_by=all,
      they page each of the four breakdowns independently. Compact groups is
      the concatenation; dimension_pagination is the per-dimension cursor;
      groups_analyzed is the unpaged total. Does not change the analysis
      window or overall statistics.
    - volume: uses real_volume when available and non-zero, else tick_volume

    Returns grouped averages for returns and volatility plus simple extras.
    Group keys are numeric for dow/hour/month; session uses DST-aware market
    session labels (asia, london, london_ny_overlap, ny, off_session, or
    off_hours on continuous_24_7) evaluated in the same analysis timezone
    used for hour and time_range.
    Example: temporal_analyze(symbol="EURUSD", group_by="dow")
    Use group_by='all' to return day-of-week, hour, month, and session
    breakdowns in one call. The overall sample statistics are included when
    detail is 'standard' or 'full'. Hour and session require intraday
    candles; D1/W1/MN1 with group_by=all keep dow/month and omit hour/session.
    """
    def _run() -> Dict[str, Any]:  # noqa: C901
        context: Dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "group_by": group_by,
            "session_calendar": session_calendar,
            "return_mode": return_mode,
            "return_basis": return_basis,
            "timezone": timezone,
            "lookback": lookback,
            "start": start,
            "end": end,
            "detail": detail,
            "offset": offset,
        }
        if min_bars is not None:
            context["min_bars"] = min_bars
        if limit is not None:
            context["limit"] = limit
        range_error = validate_historical_range(start, end)
        if range_error is not None:
            return _error_response(
                str(range_error["error"]),
                stage="validate",
                details=range_error.get("details"),
                code=range_error["error_code"],
            )
        try:
            mt5_gateway = create_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            resolved_symbol, symbol_input = resolve_public_symbol(
                symbol,
                gateway=mt5_gateway,
            )
            context["symbol"] = resolved_symbol
            if symbol_input is not None:
                context["symbol_input"] = symbol_input
            group_norm = _normalize_group_by(group_by)
            if group_norm not in ("dow", "hour", "month", "session", "all"):
                return _error_response(
                    "Invalid group_by. Use: dow, hour, month, session, all.",
                    stage="validate",
                    context=context,
                )
            context["group_by"] = group_norm
            calendar_timeframe = (
                str(timeframe or "").strip().upper() in CALENDAR_TIMEFRAMES
            )
            if calendar_timeframe and group_norm in _INTRADAY_GROUP_BY:
                return _error_response(
                    "Hour and session grouping need intraday candles. Use H1 or "
                    "M15 for hour/session; D1, W1, and MN1 are for dow/month.",
                    stage="validate",
                    context=context,
                )
            requested_timezone = str(timezone or "").strip()
            if requested_timezone:
                try:
                    analysis_tz = ZoneInfo(requested_timezone)
                except Exception:
                    return _error_response(
                        f"Invalid timezone '{requested_timezone}'. Use an IANA timezone "
                        "such as Europe/London, America/New_York, or UTC.",
                        stage="validate",
                        context=context,
                        details={"timezone": requested_timezone},
                    )
                timezone_source = "request"
            else:
                client_tz = _resolve_client_tz()
                analysis_tz = client_tz or dt_timezone.utc
                timezone_source = (
                    "client_config" if client_tz is not None else "default_utc"
                )
            tz_name = timezone_label(analysis_tz)
            context["timezone"] = tz_name
            context["timezone_source"] = timezone_source
            return_mode_value = str(return_mode or "").strip().lower()
            if return_mode_value not in {"pct", "log"}:
                return _error_response(
                    "Invalid return_mode. Use: pct or log.",
                    stage="validate",
                    context=context,
                )
            return_basis_value = str(return_basis or "").strip().lower()
            if return_basis_value not in {"previous_close", "bar_open"}:
                return _error_response(
                    "Invalid return_basis. Use: previous_close or bar_open.",
                    stage="validate",
                    context=context,
                )
            context["return_basis"] = return_basis_value
            info_before = get_symbol_info_cached(resolved_symbol)
            session_calendar_value = str(session_calendar or "auto").strip().lower()
            if session_calendar_value not in _SESSION_CALENDAR_VALUES:
                return _error_response(
                    "Invalid session_calendar. Use: auto, fx, equity, continuous_24_7.",
                    stage="validate",
                    context=context,
                )
            resolved_session_calendar = _resolve_session_calendar(
                session_calendar_value,
                resolved_symbol,
                path=getattr(info_before, "path", None),
            )
            lookback_defaulted = lookback is None
            try:
                effective_lookback = (
                    _default_temporal_lookback(timeframe, group_norm)
                    if lookback_defaulted
                    else int(lookback)
                )
            except Exception:
                return _error_response(
                    "lookback must be an integer.",
                    stage="validate",
                    context=context,
                )
            context["lookback"] = effective_lookback
            if lookback_defaulted:
                context["lookback_source"] = "auto"
            if effective_lookback <= 1 and not (start and end):
                return _error_response(
                    "lookback must be >= 2 for return calculations.",
                    stage="validate",
                    context=context,
                )
            requested_detail = str(detail or "compact").strip().lower()
            detail_mode = normalize_output_detail(detail, default="compact")
            if requested_detail not in {"compact", "standard", "summary", "full"}:
                return _error_response(
                    "detail must be one of: compact, standard, summary, full.",
                    stage="validate",
                    context=context,
                )

            limit_value: Optional[int] = None
            if limit is not None:
                try:
                    limit_value = int(float(limit))
                except Exception:
                    return _error_response(
                        "limit must be a positive integer.",
                        stage="validate",
                        context=context,
                    )
                if limit_value <= 0:
                    return _error_response(
                        "limit must be a positive integer.",
                        stage="validate",
                        context=context,
                    )
            try:
                offset_value = int(float(offset or 0))
            except Exception:
                return _error_response(
                    "offset must be a non-negative integer.",
                    stage="validate",
                    context=context,
                )
            if offset_value < 0:
                return _error_response(
                    "offset must be a non-negative integer.",
                    stage="validate",
                    context=context,
                )
            context["offset"] = offset_value
            if limit_value is not None:
                context["limit"] = limit_value

            dow_val = _parse_weekday(day_of_week)
            if day_of_week is not None and dow_val is None:
                return _error_response(
                    "Invalid day_of_week. Use 0-6 or day name (e.g., Mon).",
                    stage="validate",
                    context=context,
                    details={"day_of_week": day_of_week},
                )

            month_val = _parse_month(month)
            if month is not None and month_val is None:
                return _error_response(
                    "Invalid month. Use 1-12 or month name (e.g., Jan).",
                    stage="validate",
                    context=context,
                    details={"month": month},
                )

            tr_start, tr_end, tr_err = _parse_time_range(time_range)
            if tr_err:
                return _error_response(
                    tr_err,
                    stage="validate",
                    context=context,
                    details={"time_range": time_range},
                )

            min_bars_value: Optional[int] = None
            if min_bars is not None:
                try:
                    min_bars_value = int(min_bars)
                except Exception:
                    return _error_response(
                        "min_bars must be a non-negative integer.",
                        stage="validate",
                        context=context,
                    )
                if min_bars_value < 0:
                    return _error_response(
                        "min_bars must be a non-negative integer.",
                        stage="validate",
                        context=context,
                    )

            filters: Dict[str, Any] = {}
            if dow_val is not None:
                filters["day_of_week"] = {"value": dow_val, "label": _DOW_LABELS[dow_val]}
            if month_val is not None:
                filters["month"] = {"value": month_val, "label": _MONTH_LABELS[month_val - 1]}
            if tr_start is not None and tr_end is not None:
                filters["time_range"] = {
                    "start": _time_label(tr_start),
                    "end": _time_label(tr_end),
                    "end_exclusive": True,
                    "wraps_midnight": bool(tr_start > tr_end),
                    "timezone": tz_name,
                }

            with _symbol_ready_guard(resolved_symbol, info_before=info_before) as (err, _info):
                if err:
                    return _error_response(err, stage="symbol", context=context)

                rates, fetch_err = _fetch_rates(
                    resolved_symbol,
                    timeframe,
                    effective_lookback,
                    start,
                    end,
                    gateway=mt5_gateway,
                )
                if fetch_err:
                    return _error_response(fetch_err, stage="fetch", context=context)

            if rates is None or len(rates) < 2:
                return _error_response(
                    f"Failed to get rates for {resolved_symbol}.",
                    stage="fetch",
                    context=context,
                    details={"mt5_error": mt5.last_error()},
                )

            df = pd.DataFrame(rates)
            if df.empty:
                return _error_response("No data available.", stage="fetch", context=context, bars=0)

            try:
                df["__epoch"] = df["time"].astype(float)
            except Exception:
                return _error_response("Failed to normalize bar times.", stage="process", context=context)

            forming_trimmed = False
            if timeframe in TIMEFRAME_SECONDS:
                df, forming_trimmed = _drop_incomplete_tail_df(df, timeframe)

            if len(df) < 2:
                return _error_response(
                    "Insufficient data after trimming live bars.",
                    stage="trim",
                    context=context,
                    bars=len(df),
                )

            dt_utc = pd.to_datetime(df["__epoch"], unit="s", utc=True)
            dt = dt_utc.dt.tz_convert(analysis_tz)
            df["__dt"] = dt
            uses_broker_session_date = timeframe in CALENDAR_TIMEFRAMES
            if uses_broker_session_date:
                df["__session_date"] = df["__epoch"].map(_broker_session_date)
                df["__month"] = df["__session_date"].map(lambda value: value.month)
                month_calendar = "broker_session_date"
                month_definition = (
                    "calendar month of the broker trading-session date, independent "
                    "of the display timezone"
                )
            else:
                df["__month"] = dt.dt.month
                month_calendar = "civil_analysis_timezone"
                month_definition = f"civil month in {tz_name}"
            if resolved_session_calendar == "fx":
                df["__dow"] = dt_utc.map(_fx_trading_weekday)
                weekday_calendar = "fx_trading_day"
                weekday_definition = (
                    "17:00 America/New_York session rollover; the Sunday open "
                    "belongs to Monday"
                )
            else:
                if uses_broker_session_date:
                    df["__dow"] = df["__session_date"].map(lambda value: value.weekday())
                    weekday_calendar = "broker_session_date"
                    weekday_definition = (
                        "weekday of the broker trading-session date, independent of "
                        "the display timezone"
                    )
                else:
                    df["__dow"] = dt.dt.weekday
                    weekday_calendar = "civil_analysis_timezone"
                    weekday_definition = f"civil weekday in {tz_name}"
            session_boundary_cache: Dict[date, Dict[str, datetime]] = {}
            df["__session"] = dt.map(
                lambda value: _market_session_label(
                    value,
                    analysis_tz=analysis_tz,
                    boundary_cache=session_boundary_cache,
                    session_calendar=resolved_session_calendar,
                )
            )

            if "close" not in df.columns:
                return _error_response(
                    "Rates data missing close prices.",
                    stage="process",
                    context=context,
                )

            close = pd.to_numeric(df["close"], errors="coerce").astype(float)
            if return_basis_value == "bar_open":
                if "open" not in df.columns:
                    return _error_response(
                        "Rates data missing open prices required by return_basis=bar_open.",
                        stage="process",
                        context=context,
                    )
                denominator = pd.to_numeric(
                    df["open"], errors="coerce"
                ).astype(float)
                return_definition = "same-bar close divided by same-bar open"
                session_gap_policy = "excluded_from_same_bar_open_to_close_returns"
            else:
                denominator = close.shift(1)
                return_definition = "current close divided by previous available close"
                session_gap_policy = "included_in_the_destination_bar_return"
            if return_mode_value == "log":
                numerator_safe = close.where(close > 0)
                denominator_safe = denominator.where(denominator > 0)
                ret = np.log(numerator_safe / denominator_safe) * 100.0
            else:
                ret = ((close / denominator) - 1.0) * 100.0
            df["__return"] = ret
            df["__session_gap"] = False
            if return_basis_value != "bar_open":
                df["__session_gap"] = _session_gap_mask(
                    df["__epoch"],
                    timeframe,
                    min_gap_seconds=_WEEKEND_GAP_SECONDS,
                )
                if bool(df["__session_gap"].any()):
                    session_gap_policy = "excluded_from_group_return_statistics"

            if "high" in df.columns and "low" in df.columns:
                high = pd.to_numeric(df["high"], errors="coerce").astype(float)
                low = pd.to_numeric(df["low"], errors="coerce").astype(float)
                df["__range"] = high - low
                close_safe = close.where(close > 0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    df["__range_pct"] = (df["__range"] / close_safe) * 100.0

            volume_col = None
            if "real_volume" in df.columns:
                rv = pd.to_numeric(df["real_volume"], errors="coerce")
                if rv.notna().any() and float(rv.fillna(0.0).sum()) > 0.0:
                    volume_col = "real_volume"
            if volume_col is None and "tick_volume" in df.columns:
                volume_col = "tick_volume"

            if dow_val is not None:
                df = df[df["__dow"] == dow_val]
            if month_val is not None:
                df = df[df["__month"] == month_val]
            if tr_start is not None and tr_end is not None:
                mins = df["__dt"].dt.hour * 60 + df["__dt"].dt.minute
                if tr_start < tr_end:
                    mask = (mins >= tr_start) & (mins < tr_end)
                else:
                    mask = (mins >= tr_start) | (mins < tr_end)
                df = df.loc[mask]

            if len(df) < 2:
                return _error_response(
                    "Insufficient data after applying filters.",
                    stage="filter",
                    context=context,
                    bars=len(df),
                    filters=filters,
                )

            def _groups_for_dimension(dimension: str) -> List[Dict[str, Any]]:
                grouped = df.copy()
                if dimension == "dow":
                    grouped["__group"] = grouped["__dow"]
                elif dimension == "month":
                    grouped["__group"] = grouped["__month"]
                elif dimension == "hour":
                    grouped["__group"] = grouped["__dt"].dt.hour
                elif dimension == "session":
                    grouped["__group"] = grouped["__session"]
                else:
                    return []

                out_rows: List[Dict[str, Any]] = []
                sort_groups = dimension != "session"
                for key, grp in grouped.groupby("__group", sort=sort_groups):
                    row = _stats_for_group(grp, volume_col)
                    if dimension == "dow":
                        key_int = int(key)
                        row["group"] = key_int
                        row["group_label"] = (
                            _DOW_LABELS[key_int] if 0 <= key_int <= 6 else str(key)
                        )
                    elif dimension == "month":
                        row.update(_month_period_instance_metadata(grp, grouped))
                        key_int = int(key)
                        row["group"] = key_int
                        row["group_label"] = (
                            _MONTH_LABELS[key_int - 1]
                            if 1 <= key_int <= 12
                            else str(key)
                        )
                    elif dimension == "hour":
                        key_int = int(key)
                        row["group"] = key_int
                        row["group_label"] = f"{key_int:02d}:00"
                    else:
                        key_text = str(key)
                        row["group"] = key_text
                        row["group_label"] = key_text.replace("_", " ")
                        key_int = _SESSION_ORDER.get(key_text, 99)
                    row["_group_key"] = key_int
                    out_rows.append(row)
                if dimension == "session":
                    out_rows.sort(key=lambda row: int(row.get("_group_key", 99)))
                return out_rows

            groups_out: List[Dict[str, Any]] = []
            grouped_dimensions: List[Dict[str, Any]] = []
            payload_warnings: List[str] = []
            omit_intraday_dimensions = bool(
                group_norm == "all" and calendar_timeframe
            )
            if omit_intraday_dimensions:
                payload_warnings.append(
                    f"Hour and session breakdowns omitted for {timeframe}; "
                    "use H1 or M15 for hour/session. Day-of-week and month remain."
                )

            if group_norm in {"dow", "month", "hour", "session"}:
                groups_out = _groups_for_dimension(group_norm)
                if groups_out:
                    if group_norm == "dow":
                        df["__group"] = df["__dow"]
                    elif group_norm == "month":
                        df["__group"] = df["__month"]
                    elif group_norm == "hour":
                        df["__group"] = df["__dt"].dt.hour
                    else:
                        df["__group"] = df["__session"].map(
                            lambda value: _SESSION_ORDER.get(str(value), 99)
                        )
            elif group_norm == "all":
                dimensions = (
                    _CALENDAR_TF_GROUP_DIMENSIONS
                    if omit_intraday_dimensions
                    else _ALL_GROUP_DIMENSIONS
                )
                for dimension in dimensions:
                    breakdown = _groups_for_dimension(dimension)
                    if breakdown:
                        grouped_dimensions.append(
                            {
                                "dimension": dimension,
                                "breakdown": breakdown,
                            }
                        )

            excluded_groups: List[Dict[str, Any]] = []
            auto_min_bars = False
            if (
                min_bars_value is None
                and group_norm in {"dow", "all"}
                and dow_val is None
            ):
                dow_groups = groups_out
                if group_norm == "all":
                    dow_groups = next(
                        (
                            item.get("breakdown", [])
                            for item in grouped_dimensions
                            if item.get("dimension") == "dow"
                        ),
                        [],
                    )
                bar_counts = [int(row.get("bars", 0) or 0) for row in dow_groups]
                median_bars = float(np.median(bar_counts)) if bar_counts else 0.0
                min_bars_value = max(2, int(median_bars * 0.25))
                auto_min_bars = True

            analysis_df = df
            if min_bars_value is not None and min_bars_value > 0 and groups_out:
                excluded = [
                    row
                    for row in groups_out
                    if int(row.get("bars", 0) or 0) < min_bars_value
                ]
                included = [
                    row
                    for row in groups_out
                    if int(row.get("bars", 0) or 0) >= min_bars_value
                ]
                if excluded:
                    excluded_keys = {row.get("_group_key") for row in excluded}
                    excluded_groups = [
                        {
                            "group": row.get("group"),
                            "group_label": row.get("group_label"),
                            "bars": int(row.get("bars", 0) or 0),
                            "min_bars": int(min_bars_value),
                            "auto": bool(auto_min_bars),
                        }
                        for row in excluded
                    ]
                    groups_out = included
                    analysis_df = df[~df["__group"].isin(excluded_keys)]

            if min_bars_value is not None and min_bars_value > 0 and grouped_dimensions:
                for item in grouped_dimensions:
                    dimension = str(item.get("dimension") or "")
                    if auto_min_bars and dimension != "dow":
                        continue
                    breakdown = item.get("breakdown")
                    if not isinstance(breakdown, list):
                        continue
                    excluded = [
                        row
                        for row in breakdown
                        if int(row.get("bars", 0) or 0) < min_bars_value
                    ]
                    included = [
                        row
                        for row in breakdown
                        if int(row.get("bars", 0) or 0) >= min_bars_value
                    ]
                    if excluded:
                        item["breakdown"] = included
                        excluded_groups.extend(
                            {
                                "dimension": dimension,
                                "group": row.get("group"),
                                "group_label": row.get("group_label"),
                                "bars": int(row.get("bars", 0) or 0),
                                "min_bars": int(min_bars_value),
                                "auto": bool(auto_min_bars),
                            }
                            for row in excluded
                        )

            groups_out = [
                {key: value for key, value in row.items() if key != "_group_key"}
                for row in groups_out
            ]
            for item in grouped_dimensions:
                breakdown = item.get("breakdown")
                if isinstance(breakdown, list):
                    item["breakdown"] = [
                        {
                            key: value
                            for key, value in row.items()
                            if key != "_group_key"
                        }
                        for row in breakdown
                    ]
            summary_groups = (
                [
                    {
                        "dimension": item.get("dimension"),
                        "breakdown": list(item.get("breakdown") or []),
                    }
                    for item in grouped_dimensions
                ]
                if grouped_dimensions
                else list(groups_out)
            )
            pagination_meta: Dict[str, Any] = {}
            if grouped_dimensions and (limit_value is not None or offset_value):
                paged_dimensions = []
                for item in grouped_dimensions:
                    breakdown = item.get("breakdown")
                    if not isinstance(breakdown, list):
                        paged_dimensions.append(item)
                        continue
                    paged_breakdown, item_meta = _paginate_temporal_rows(
                        breakdown,
                        limit=limit_value,
                        offset=offset_value,
                    )
                    paged_item = {
                        "dimension": item.get("dimension"),
                        "breakdown": paged_breakdown,
                    }
                    paged_item.update(item_meta)
                    paged_dimensions.append(paged_item)
                grouped_dimensions = paged_dimensions
            elif groups_out:
                groups_out, pagination_meta = _paginate_temporal_rows(
                    groups_out,
                    limit=limit_value,
                    offset=offset_value,
                )
            no_qualifying_groups = analysis_df.empty
            overall = (
                {}
                if no_qualifying_groups
                else _stats_for_group(analysis_df, volume_col)
            )

            window_df = df if no_qualifying_groups else analysis_df
            source_start_epoch = float(df["__epoch"].iloc[0])
            source_end_epoch = float(df["__epoch"].iloc[-1])
            start_epoch = float(window_df["__epoch"].iloc[0])
            end_epoch = float(window_df["__epoch"].iloc[-1])
            start_str = _format_epoch_in_timezone(start_epoch, analysis_tz)
            end_str = _format_epoch_in_timezone(end_epoch, analysis_tz)

            payload: Dict[str, Any] = {
                "success": True,
                "symbol": resolved_symbol,
                "timeframe": timeframe,
                "group_by": group_norm,
                "return_mode": return_mode_value,
                "return_basis": return_basis_value,
                "return_definition": return_definition,
                "session_gap_policy": session_gap_policy,
                "units": {
                    "avg_return_pct": _PERCENT_UNIT,
                    "median_return_pct": _PERCENT_UNIT,
                    "avg_abs_return_pct": _PERCENT_UNIT,
                    "win_rate": "fraction",
                    "win_rate_pct": _PERCENT_UNIT,
                    "avg_range_pct": _PERCENT_UNIT,
                    "volatility_pct": _PERCENT_UNIT,
                },
                "timezone": tz_name,
                "timezone_source": timezone_source,
                "bars": int(len(analysis_df)),
                "start": start_str,
                "end": end_str,
                "history_policy": "completed_bars_only",
                "bar_timestamp_basis": "open_time",
                "latest_bar_complete": True,
                "forming_candle_status": (
                    "skipped" if forming_trimmed else "none_detected"
                ),
                "filters": filters,
                "overall": overall,
                "overall_basis": (
                    "full_filtered_sample_before_per_dimension_group_exclusions"
                    if group_norm == "all"
                    else "sample_after_group_exclusions"
                ),
                "volume_source": volume_col,
            }
            if start_epoch != source_start_epoch or end_epoch != source_end_epoch:
                payload["analysis_window"] = {
                    "start": start_str,
                    "end": end_str,
                    "bars": int(len(analysis_df)),
                    "basis": "sample_after_group_exclusions",
                }
            if symbol_input is not None:
                payload["symbol_input"] = symbol_input
            if start and end:
                payload["query_applied"] = {
                    "mode": "range",
                    "requested_start": start,
                    "requested_end": end,
                    "returned_start": start_str,
                    "returned_end": end_str,
                }
            else:
                payload["lookback"] = effective_lookback
                payload["lookback_source"] = (
                    "auto" if lookback_defaulted else "request"
                )
            if group_norm in {"dow", "all"}:
                payload["weekday_calendar"] = weekday_calendar
                payload["weekday_definition"] = weekday_definition
            if group_norm in {"month", "all"}:
                payload["month_calendar"] = month_calendar
                payload["month_definition"] = month_definition
            if no_qualifying_groups:
                payload["analysis_status"] = "insufficient_group_samples"
                payload["message"] = (
                    "No temporal group met the requested minimum bar count."
                )
            include_hour_metadata = group_norm == "hour" or (
                group_norm == "all" and not omit_intraday_dimensions
            )
            include_session_metadata = group_norm == "session" or (
                group_norm == "all" and not omit_intraday_dimensions
            )
            if include_hour_metadata:
                payload["timing_convention"] = (
                    "candle_open_utc"
                    if tz_name == "UTC"
                    else "candle_open_analysis_timezone"
                )
                payload["hour_span"] = (
                    "Each HH:00 bucket covers candle opens from HH:00 inclusive "
                    f"to the next hour exclusive in {tz_name}."
                )
            if lookback_defaulted and not (start and end):
                payload["lookback_note"] = (
                    f"Auto lookback selected {int(effective_lookback)} bars for "
                    f"{group_norm} analysis on {timeframe}; pass --lookback for a "
                    "smaller, faster sample."
                )
            group_payload_source = summary_groups
            payload["groups_analyzed"] = _temporal_group_count(group_payload_source)
            global_best = _temporal_best_summary(summary_groups)
            if global_best is not None:
                payload["best"] = global_best
            payload["groups_excluded"] = len(excluded_groups)
            if min_bars_value is not None:
                minimum_key = (
                    "day_of_week_min_bars_applied"
                    if auto_min_bars
                    else "min_bars_applied"
                )
                payload[minimum_key] = int(min_bars_value or 0)
            if include_session_metadata:
                payload["session_calendar"] = resolved_session_calendar
                payload["session_calendar_source"] = (
                    "symbol_and_broker_metadata"
                    if session_calendar_value == "auto"
                    else "request"
                )
                payload["session_definition"] = _session_definition_for_clock(
                    tz_name,
                    resolved_session_calendar,
                )
            if min_bars_value is not None:
                filters["min_bars"] = {
                    "value": int(min_bars_value or 0),
                    "auto": bool(auto_min_bars),
                    "source": "auto" if auto_min_bars else "request",
                    "scope": "day_of_week" if auto_min_bars else "all_groupings",
                    "purpose": "exclude grouped rows below this sample size",
                }
            payload.update(pagination_meta)
            sample_context = _temporal_sample_warnings(
                summary_groups
            )
            if sample_context:
                payload.update(sample_context)
            if excluded_groups:
                payload["excluded_groups"] = excluded_groups
                if group_norm == "all":
                    payload_warnings.append(
                        "Sparse temporal groups below min_bars were excluded from "
                        "their grouped breakdowns only; overall statistics use the "
                        "full filtered sample."
                    )
                else:
                    payload_warnings.append(
                        "Sparse temporal groups below min_bars were excluded from "
                        "grouped results and overall statistics."
                    )
            now_epoch = datetime.now(dt_timezone.utc).timestamp()
            as_of = format_epoch_utc(now_epoch)
            if as_of:
                payload["as_of"] = as_of
            freshness = completed_bar_freshness_fields(
                resolved_symbol,
                timeframe,
                source_end_epoch,
                now_epoch=now_epoch,
                item="bar",
            )
            for key in COMPLETED_BAR_FRESHNESS_KEYS:
                if key in freshness:
                    payload[key] = freshness[key]
            stale_warning = freshness.get("stale_warning")
            if stale_warning and stale_warning not in payload_warnings:
                payload_warnings.append(stale_warning)
            if payload_warnings:
                payload["warnings"] = payload_warnings
            if grouped_dimensions:
                payload["groups"] = grouped_dimensions
            else:
                payload["groups"] = groups_out
            if detail_mode == "summary":
                return _summary_temporal_payload(
                    payload, summary_groups=summary_groups
                )
            if detail_mode == "compact":
                return _compact_temporal_payload(
                    payload, summary_groups=summary_groups
                )
            if detail_mode == "standard":
                return _standard_temporal_payload(
                    payload, summary_groups=summary_groups
                )
            return payload
        except MT5ConnectionError as exc:
            return _error_response(
                str(exc),
                stage="fetch",
                context=context,
                code="mt5_connection_error",
            )
        except Exception as exc:
            return _error_response(
                f"Error computing temporal analysis: {str(exc)}",
                stage="internal",
                context=context,
            )

    return run_mt5_logged_operation(
        logger,
        operation="temporal_analyze",
        symbol=symbol,
        timeframe=timeframe,
        group_by=group_by,
        lookback=lookback,
        limit=limit,
        offset=offset,
        func=_run,
    )
