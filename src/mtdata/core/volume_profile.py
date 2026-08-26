from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, Literal, Optional

from pydantic import Field

from ..services.data_service import fetch_candles, fetch_ticks
from ..shared.constants import TIMEFRAME_SECONDS
from ..shared.schema import DetailLiteral, TimeframeLiteral
from ..shared.symbols import is_probably_crypto_symbol
from ..utils.freshness import closed_session_context, standard_weekend_window
from ..utils.mt5 import (
    MT5ConnectionError,
    _symbol_ready_guard,
    ensure_mt5_connection_or_raise,
    resolve_public_symbol,
)
from ..utils.time import _format_datetime_explicit, bar_close_epoch
from ..utils.utils import (
    _parse_end_datetime,
    _parse_start_datetime,
    _positive_float_attr,
    validate_historical_range,
)
from ..utils.volume_profile import (
    VolumeProfileConfig,
    VolumeProfilePriceSourceLiteral,
    VolumeProfileVolumeSourceLiteral,
    compute_volume_profile,
)
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway
from .runtime_metadata import attach_mt5_source

logger = logging.getLogger(__name__)

VolumeProfileSourceLiteral = Literal["auto", "ticks", "m1_bars"]

_DEFAULT_MAX_TICK_WINDOW_DAYS = 1
_DEFAULT_MAX_TICKS = 50_000
_DEFAULT_MAX_M1_BARS = 20_000
_DEFAULT_PROFILE_LIMIT = 200
_MIN_TICK_PRICE_COVERAGE_RATIO = 0.5
_TICK_WINDOW_TOLERANCE_SECONDS = 1.0


def _m1_tick_count_unsupported_error(source: str) -> Dict[str, Any]:
    return {
        "success": False,
        "error": (
            "volume_source=tick_count is not available with source=m1_bars. "
            "M1 approximation creates three synthetic low/close/high rows per "
            "bar; counting those rows is not a market tick count."
        ),
        "error_code": "volume_profile_tick_count_unavailable_for_m1_bars",
        "parameter": "volume_source",
        "source": source,
        "volume_source": "tick_count",
        "remediation": (
            "Use volume_source=tick_volume with source=m1_bars, or source=ticks "
            "with volume_source=tick_count."
        ),
        "valid_values": {
            "volume_source": [
                "auto",
                "real_volume",
                "tick_volume",
                "volume_real",
                "volume",
            ],
            "source": ["ticks", "auto"],
        },
    }


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _positive_int_attr(obj: Any, *names: str) -> Optional[int]:
    value = _positive_float_attr(obj, *names)
    if value is None:
        return None
    return int(value)


def _window_days(start: Optional[str], end: Optional[str]) -> Optional[float]:
    start_dt = _parse_start_datetime(start) if start else None
    end_dt = _parse_end_datetime(end) if end else None
    if start and start_dt is None:
        return None
    if end and end_dt is None:
        return None
    if start_dt is None and end_dt is None:
        return None
    if end_dt is None:
        end_dt = _utc_now_naive()
    if start_dt is None:
        return None
    seconds = max(0.0, float((end_dt - start_dt).total_seconds()))
    return seconds / 86400.0


def _exceeds_tick_window(days: Optional[float], max_days: int) -> bool:
    if days is None:
        return False
    return (
        float(days) * 86400.0
        > float(max_days) * 86400.0 + _TICK_WINDOW_TOLERANCE_SECONDS
    )


def _resolve_profile_window(
    *,
    start: Optional[str],
    end: Optional[str],
    timeframe: Optional[str],
    lookback: Optional[int],
) -> Dict[str, Any]:
    if start:
        return {"start": start, "end": end}
    if timeframe is None and lookback is None:
        end_dt = _parse_end_datetime(end) if end else _utc_now_naive()
        if end and end_dt is None:
            return {"error": f"Could not parse end datetime {end!r}"}
        assert end_dt is not None
        start_dt = end_dt - timedelta(days=1)
        return {
            "start": start_dt.isoformat(sep=" ", timespec="seconds"),
            "end": end if end else end_dt.isoformat(sep=" ", timespec="seconds"),
        }
    if not timeframe:
        return {"error": "timeframe is required when lookback is provided"}
    tf = str(timeframe).strip().upper()
    seconds = TIMEFRAME_SECONDS.get(tf)
    if seconds is None:
        return {"error": f"Invalid timeframe {timeframe!r}"}
    if lookback is None:
        bars = _DEFAULT_PROFILE_LIMIT
    else:
        try:
            bars = int(lookback)
        except (TypeError, ValueError):
            bars = 0
        if bars <= 0:
            return {
                "error": (
                    "lookback must be a positive integer when timeframe is provided; "
                    f"omit lookback to use the default {int(_DEFAULT_PROFILE_LIMIT)} bars."
                )
            }
    end_dt = _parse_end_datetime(end) if end else _utc_now_naive()
    if end and end_dt is None:
        return {"error": f"Could not parse end datetime {end!r}"}
    assert end_dt is not None
    return {
        "start": None,
        "end": end if end else end_dt.isoformat(sep=" ", timespec="seconds"),
        "requested_bars": bars,
    }


def _resolve_profile_bar_window(
    *,
    symbol: str,
    timeframe: str,
    bars: int,
    end: Optional[str],
) -> Dict[str, Any]:
    payload = fetch_candles(
        symbol=symbol,
        timeframe=timeframe,  # type: ignore[arg-type]
        limit=max(1, int(bars)),
        start=None,
        end=end,
        ohlcv="close",
        time_as_epoch=True,
        include_incomplete=False,
        allow_stale=True,
    )
    if payload.get("error"):
        return {
            "error": (
                "Could not resolve the volume-profile bar window: "
                f"{payload.get('error')}"
            ),
            "error_code": "volume_profile_bar_window_failed",
        }

    end_dt = _parse_end_datetime(end) if end else _utc_now_naive()
    if end_dt is None:
        return {"error": f"Could not parse end datetime {end!r}"}
    end_epoch = _profile_datetime_epoch(end_dt)
    assert end_epoch is not None

    resolved_bars: list[tuple[float, float]] = []
    for row in _table_rows(payload):
        if not isinstance(row, dict):
            continue
        raw_time = row.get("time")
        try:
            open_epoch = float(raw_time)
        except (TypeError, ValueError):
            parsed = _parse_start_datetime(str(raw_time or ""))
            parsed_epoch = _profile_datetime_epoch(parsed)
            if parsed_epoch is None:
                continue
            open_epoch = parsed_epoch
        try:
            close_epoch = bar_close_epoch(open_epoch, timeframe)
        except (TypeError, ValueError):
            continue
        if close_epoch <= end_epoch + 0.001:
            resolved_bars.append((open_epoch, close_epoch))

    resolved_bars.sort(key=lambda item: item[0])
    if len(resolved_bars) < int(bars):
        return {
            "error": (
                f"Only {len(resolved_bars)} completed {timeframe} bars are "
                f"available; {int(bars)} are required for the requested profile."
            ),
            "error_code": "volume_profile_insufficient_bar_history",
            "requested_bars": int(bars),
            "available_bars": len(resolved_bars),
        }

    selected = resolved_bars[-int(bars) :]
    return {
        "start": _format_window_timestamp(selected[0][0]),
        "end": _format_window_timestamp(selected[-1][1]),
        "bar_window": {
            "timeframe": timeframe,
            "requested_bars": int(bars),
            "resolved_bars": len(selected),
            "first_bar_open": _format_window_timestamp(selected[0][0]),
            "last_bar_close": _format_window_timestamp(selected[-1][1]),
            "boundary_basis": "actual_completed_timeframe_bars",
        },
    }


def _table_rows(payload: Dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("data")
    return rows if isinstance(rows, list) else []


def _format_window_timestamp(
    value: Any,
    *,
    end_bound: bool = False,
) -> Optional[str]:
    if value in (None, ""):
        return None
    timespec = "seconds"
    parsed: Optional[datetime]
    if isinstance(value, datetime):
        parsed = value
        if value.microsecond:
            timespec = "milliseconds"
    elif isinstance(value, (int, float)):
        try:
            epoch = float(value)
        except (TypeError, ValueError):
            parsed = None
        else:
            if not math.isfinite(epoch):
                return None
            parsed = datetime.fromtimestamp(epoch, tz=timezone.utc)
            if not float(epoch).is_integer():
                timespec = "milliseconds"
    else:
        text = str(value).strip()
        if not text:
            return None
        if "." in text:
            timespec = "milliseconds"
        parser = _parse_end_datetime if end_bound else _parse_start_datetime
        parsed = parser(text)
        if parsed is None:
            return text
        if parsed.microsecond:
            timespec = (
                "milliseconds"
                if parsed.microsecond % 1000 == 0
                else "microseconds"
            )
    return _format_datetime_explicit(parsed, timespec=timespec)


def _utc_datetime(value: datetime) -> datetime:
    return (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )


def _weekend_tick_coverage_context(
    symbol: str,
    *,
    requested_start: datetime,
    requested_end: datetime,
    observed_start: Optional[datetime],
    observed_end: Optional[datetime],
    endpoint_tolerance_seconds: float = 300.0,
) -> Dict[str, Any]:
    """Identify endpoint gaps fully explained by the standard weekend closure."""
    if is_probably_crypto_symbol(symbol):
        return {}
    request_start_utc = _utc_datetime(requested_start)
    request_end_utc = _utc_datetime(requested_end)
    start_window = standard_weekend_window(request_start_utc)
    end_probe = request_end_utc - timedelta(microseconds=1)
    end_window = standard_weekend_window(end_probe)
    closures: list[Dict[str, Any]] = []
    start_gap_expected = False
    end_gap_expected = False

    if start_window is not None and observed_start is not None:
        close_utc, reopen_utc = start_window
        seconds_after_reopen = (
            _utc_datetime(observed_start) - reopen_utc
        ).total_seconds()
        start_gap_expected = (
            0.0 <= seconds_after_reopen <= float(endpoint_tolerance_seconds)
        )
        if start_gap_expected:
            closures.append(
                {
                    "reason": "standard_weekend_closure",
                    "start": _format_datetime_explicit(close_utc, timespec="seconds"),
                    "end": _format_datetime_explicit(reopen_utc, timespec="seconds"),
                }
            )

    if end_window is not None and observed_end is not None:
        close_utc, reopen_utc = end_window
        seconds_before_close = (
            close_utc - _utc_datetime(observed_end)
        ).total_seconds()
        end_gap_expected = (
            0.0 <= seconds_before_close <= float(endpoint_tolerance_seconds)
        )
        closure = {
            "reason": "standard_weekend_closure",
            "start": _format_datetime_explicit(close_utc, timespec="seconds"),
            "end": _format_datetime_explicit(reopen_utc, timespec="seconds"),
        }
        if end_gap_expected and closure not in closures:
            closures.append(closure)

    entirely_closed = bool(
        start_window is not None
        and end_window is not None
        and start_window == end_window
    )
    if entirely_closed and not closures:
        close_utc, reopen_utc = start_window
        closures.append(
            {
                "reason": "standard_weekend_closure",
                "start": _format_datetime_explicit(close_utc, timespec="seconds"),
                "end": _format_datetime_explicit(reopen_utc, timespec="seconds"),
            }
        )
    return {
        "start_gap_expected": start_gap_expected,
        "end_gap_expected": end_gap_expected,
        "entire_request_closed": entirely_closed,
        "scheduled_closures": closures,
        "basis": "standard_weekend_hours",
    }


def _observed_profile_window(
    rows: list[dict[str, Any]],
    *,
    fallback_start: Optional[str],
    fallback_end: Optional[str],
) -> Dict[str, Optional[str]]:
    times = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        formatted = _format_window_timestamp(row.get("time"))
        if formatted not in (None, ""):
            times.append(formatted)
    if not times:
        return {
            "start": _format_window_timestamp(fallback_start),
            "end": _format_window_timestamp(fallback_end),
        }
    return {"start": times[0], "end": times[-1]}


def _profile_datetime_epoch(value: Optional[datetime]) -> Optional[float]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.timestamp()


def _fetch_tick_rows(
    *,
    symbol: str,
    start: Optional[str],
    end: Optional[str],
    max_ticks: int,
) -> Dict[str, Any]:
    # ``fetch_ticks`` is start-anchored when both bounds are supplied.  A
    # capped profile must retain the most recent flow, so make the provider
    # query end-anchored and then discard any rows preceding the requested
    # start locally.
    payload = fetch_ticks(
        symbol=symbol,
        limit=max(1, int(max_ticks)),
        start=None,
        end=end,
        format="full_rows",
    )
    if payload.get("error"):
        return payload
    provider_rows = _table_rows(payload)
    parsed_start = _parse_start_datetime(start) if start else None
    start_epoch = _profile_datetime_epoch(parsed_start)
    rows = provider_rows
    if start_epoch is not None:
        rows = [
            row
            for row in provider_rows
            if (
                (row_epoch := _profile_datetime_epoch(
                    _parse_start_datetime(str(row.get("time") or ""))
                ))
                is None
                or row_epoch >= start_epoch
            )
        ]
    provider_limit_reached = bool(payload.get("limit_reached")) or len(
        provider_rows
    ) >= int(max_ticks)
    earliest_provider_epoch = (
        _profile_datetime_epoch(
            _parse_start_datetime(str(provider_rows[0].get("time") or ""))
        )
        if provider_rows
        else None
    )
    limit_reached = bool(
        provider_limit_reached
        and (
            start_epoch is None
            or earliest_provider_epoch is None
            or earliest_provider_epoch >= start_epoch
        )
    )
    return {
        "success": True,
        "source": "ticks",
        "rows": rows,
        "fetch_payload": payload,
        "diagnostics": {
            "tick_rows": int(len(rows)),
            "provider_tick_rows": int(len(provider_rows)),
            "requested_max_ticks": int(max_ticks),
            "tick_limit_reached": limit_reached,
            "retained": "latest",
        },
    }


def _fetch_m1_rows(
    *,
    symbol: str,
    start: Optional[str],
    end: Optional[str],
    max_m1_bars: int,
) -> Dict[str, Any]:
    # fetch_candles with start+end+limit is start-anchored first_n. A capped
    # profile must retain the most recent bars, so query end-anchored and then
    # discard any rows preceding the requested start locally.
    max_bars = max(1, int(max_m1_bars))
    payload = fetch_candles(
        symbol=symbol,
        timeframe="M1",
        limit=max_bars,
        start=None,
        end=end,
        ohlcv="OHLCV",
        include_incomplete=False,
    )
    if payload.get("error"):
        return payload
    provider_candles = [
        candle for candle in _table_rows(payload) if isinstance(candle, dict)
    ]
    parsed_start = _parse_start_datetime(start) if start else None
    start_epoch = _profile_datetime_epoch(parsed_start)
    candles = provider_candles
    if start_epoch is not None:
        candles = [
            candle
            for candle in provider_candles
            if (
                (
                    row_epoch := _profile_datetime_epoch(
                        _parse_start_datetime(str(candle.get("time") or ""))
                    )
                )
                is None
                or row_epoch >= start_epoch
            )
        ]
    truncated = len(candles) > max_bars
    if truncated:
        candles = candles[-max_bars:]
    provider_limit_reached = bool(payload.get("limit_reached")) or len(
        provider_candles
    ) >= max_bars
    earliest_provider_epoch = (
        _profile_datetime_epoch(
            _parse_start_datetime(str(provider_candles[0].get("time") or ""))
        )
        if provider_candles
        else None
    )
    truncated = bool(
        truncated
        or (
            provider_limit_reached
            and (
                start_epoch is None
                or earliest_provider_epoch is None
                or earliest_provider_epoch >= start_epoch
            )
        )
    )
    rows = []
    for candle in candles:
        if not isinstance(candle, dict):
            continue
        volume = candle.get("real_volume")
        try:
            real_volume = float(volume)
        except (TypeError, ValueError):
            real_volume = 0.0
        if real_volume <= 0.0:
            volume = candle.get("tick_volume")
        try:
            weight = float(volume)
        except (TypeError, ValueError):
            weight = 0.0
        prices = []
        for key in ("low", "close", "high"):
            try:
                value = float(candle.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0.0:
                prices.append(value)
        if not prices:
            continue
        per_price_weight = weight / float(len(prices)) if weight > 0.0 else 0.0
        candle_time = _format_window_timestamp(candle.get("time"))
        for price in prices:
            rows.append(
                {
                    "time": candle_time,
                    "mid": price,
                    "tick_volume": per_price_weight,
                    "real_volume": real_volume / float(len(prices)) if real_volume > 0.0 else 0.0,
                }
            )
    return {
        "success": True,
        "source": "m1_bars",
        "rows": rows,
        "fetch_payload": payload,
        "diagnostics": {
            "m1_bars": int(len(candles)),
            "profile_rows": int(len(rows)),
            "requested_max_m1_bars": int(max_m1_bars),
            "truncated": truncated,
            "selection": "latest_n" if truncated else "all",
            "approximation": "M1 bar volume split across low/close/high prices.",
        },
        "warnings": [
            "Volume profile used an L/C/H equal-weight proxy from M1 bars "
            "(open omitted; each bar's volume split equally across low, close, "
            "and high); intrabar volume location is estimated.",
            *(
                [f"M1 input exceeded max_m1_bars={max_bars}; the latest {max_bars} bars were retained."]
                if truncated
                else []
            ),
        ],
    }


def _row_finite_positive(row: Any, key: str) -> Optional[float]:
    if isinstance(row, dict):
        value = row.get(key)
    else:
        value = getattr(row, key, None)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(numeric) and numeric > 0.0:
        return numeric
    return None


def _tick_price_quality(rows: list[dict[str, Any]], price_source: str) -> Dict[str, Any]:
    source = str(price_source or "mid").strip().lower()
    total = len(rows)
    valid = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        price = _row_finite_positive(row, source)
        if source == "mid" and price is None:
            bid = _row_finite_positive(row, "bid")
            ask = _row_finite_positive(row, "ask")
            if bid is not None and ask is not None:
                price = (bid + ask) / 2.0
        if price is not None:
            valid += 1
    ratio = (valid / total) if total else 0.0
    return {
        "price_source": source,
        "input_rows": int(total),
        "valid_price_rows": int(valid),
        "dropped_price_rows": int(max(0, total - valid)),
        "valid_price_ratio": round(ratio, 4),
    }


def _should_fallback_from_tick_prices(quality: Dict[str, Any]) -> bool:
    input_rows = int(quality.get("input_rows") or 0)
    if input_rows <= 0:
        return True
    valid_rows = int(quality.get("valid_price_rows") or 0)
    if valid_rows <= 0:
        return True
    ratio = float(quality.get("valid_price_ratio") or 0.0)
    return ratio < _MIN_TICK_PRICE_COVERAGE_RATIO


def _select_profile_rows(
    *,
    symbol: str,
    start: Optional[str],
    end: Optional[str],
    source: str,
    price_source: str,
    max_tick_window_days: int,
    max_ticks: int,
    max_m1_bars: int,
) -> Dict[str, Any]:
    source_value = str(source or "auto").strip().lower()
    if source_value not in {"auto", "ticks", "m1_bars"}:
        return {"error": "source must be one of: auto, ticks, m1_bars"}
    days = _window_days(start, end)
    use_m1 = source_value == "m1_bars"
    window_exceeds_tick_budget = _exceeds_tick_window(days, max_tick_window_days)
    if source_value == "auto" and window_exceeds_tick_budget:
        use_m1 = True

    if use_m1:
        selected = _fetch_m1_rows(
            symbol=symbol,
            start=start,
            end=end,
            max_m1_bars=max_m1_bars,
        )
        diagnostics = selected.setdefault("diagnostics", {})
        if isinstance(diagnostics, dict) and window_exceeds_tick_budget:
            diagnostics["tick_window_days"] = round(days, 4)
            diagnostics["max_tick_window_days"] = int(max_tick_window_days)
            if source_value == "auto":
                diagnostics["auto_fallback_reason"] = (
                    "requested window exceeds bounded tick window"
                )
            else:
                diagnostics["tick_window_budget_exceeded"] = True
        return selected

    tick_result = _fetch_tick_rows(
        symbol=symbol,
        start=start,
        end=end,
        max_ticks=max_ticks,
    )
    if not tick_result.get("error"):
        rows = tick_result.get("rows")
        if not isinstance(rows, list):
            rows = []
        quality = _tick_price_quality(rows, price_source)
        diagnostics = tick_result.setdefault("diagnostics", {})
        if isinstance(diagnostics, dict):
            diagnostics["tick_price_quality"] = quality
        if (
            source_value == "auto"
            and isinstance(diagnostics, dict)
            and diagnostics.get("tick_limit_reached") is True
        ):
            fallback = _fetch_m1_rows(
                symbol=symbol,
                start=start,
                end=end,
                max_m1_bars=max_m1_bars,
            )
            fallback_diagnostics = fallback.setdefault("diagnostics", {})
            if isinstance(fallback_diagnostics, dict):
                fallback_diagnostics.update(
                    {
                        "auto_fallback_reason": "max_ticks",
                        "tick_rows": diagnostics.get("tick_rows"),
                        "requested_max_ticks": diagnostics.get(
                            "requested_max_ticks"
                        ),
                        "tick_limit_reached": True,
                        "tick_price_quality": quality,
                    }
                )
            return fallback
        if source_value == "auto" and _should_fallback_from_tick_prices(quality):
            fallback = _fetch_m1_rows(
                symbol=symbol,
                start=start,
                end=end,
                max_m1_bars=max_m1_bars,
            )
            fallback_diagnostics = fallback.setdefault("diagnostics", {})
            if isinstance(fallback_diagnostics, dict):
                fallback_diagnostics["auto_fallback_reason"] = (
                    "tick price coverage below threshold"
                )
                fallback_diagnostics["tick_price_quality"] = quality
                fallback_diagnostics["min_tick_price_coverage_ratio"] = (
                    _MIN_TICK_PRICE_COVERAGE_RATIO
                )
            return fallback
        if isinstance(diagnostics, dict) and source_value == "auto":
            diagnostics["auto_source_reason"] = (
                "tick data within bounded window with adequate price coverage"
            )
        return tick_result
    if source_value == "ticks":
        return tick_result
    fallback = _fetch_m1_rows(
        symbol=symbol,
        start=start,
        end=end,
        max_m1_bars=max_m1_bars,
    )
    diagnostics = fallback.setdefault("diagnostics", {})
    if isinstance(diagnostics, dict):
        diagnostics["auto_fallback_reason"] = "tick fetch failed"
        diagnostics["tick_error"] = tick_result.get("error")
    return fallback


def _merge_profile_warnings(*groups: Any) -> list[str]:
    merged: list[str] = []
    for group in groups:
        items = [group] if isinstance(group, str) else group
        if not isinstance(items, list):
            continue
        for item in items:
            text = str(item).strip()
            if text and text not in merged:
                merged.append(text)
    return merged


def _profile_detail_payload(profile: Dict[str, Any], detail: str) -> Dict[str, Any]:
    detail_value = str(detail or "compact").strip().lower()
    if detail_value in {"summary"}:
        detail_value = "compact"
    if detail_value not in {"compact", "standard", "full"}:
        detail_value = "compact"
    keys = [
        "success",
        "symbol",
        "profile_source",
        "source",
        "source_decision",
        "volume_profile_accuracy",
        "volume_source_quality",
        "is_synthetic",
        "source_note",
        "window",
        "requested_window",
        "bar_window",
        "price_source",
        "price_source_requested",
        "price_source_effective",
        "proxy_prices",
        "allocation_method",
        "volume_is_synthetic",
        "volume_kind",
        "bucket_size",
        "requested_bucket_size",
        "effective_bucket_size",
        "value_area_pct",
        "price_point",
        "price_digits",
        "total_volume",
        "poc",
        "vah",
        "val",
        "levels",
        "value_area",
        "diagnostics",
        "truncated",
        "truncation_reason",
        "data_quality",
        "coverage_note",
        "scheduled_closures",
        "warnings",
        "as_of",
        "data_as_of",
        "fetched_at",
        "timezone",
        "data_age_seconds",
        "observation_age_seconds",
        "data_stale",
        "stale_after_seconds",
        "freshness_basis",
        "freshness_applicability",
        "freshness_state",
        "market_status",
        "market_status_reason",
        "note",
        "query_type",
        "units",
    ]
    out = {key: profile[key] for key in keys if key in profile}
    value_area = out.get("value_area")
    if isinstance(value_area, dict):
        compact_value_area = dict(value_area)
        bucket_indexes = compact_value_area.get("bucket_indexes")
        if isinstance(bucket_indexes, list):
            compact_value_area["bucket_count"] = len(bucket_indexes)
        if detail_value != "full":
            compact_value_area.pop("bucket_indexes", None)
        out["value_area"] = compact_value_area
    if detail_value == "compact":
        out.pop("levels", None)
        out.pop("units", None)
        window = out.get("window")
        if isinstance(window, dict) and not any(
            value not in (None, "") for value in window.values()
        ):
            out.pop("window", None)
    else:
        out["detail"] = detail_value
    if detail_value == "standard":
        out["buckets"] = profile.get("buckets", [])[:50]
        out["bucket_note"] = "First 50 buckets returned; use detail='full' for all buckets."
    elif detail_value == "full":
        out["buckets"] = profile.get("buckets", [])
        fetch_payload = profile.get("fetch_payload")
        if isinstance(fetch_payload, dict):
            out["fetch_meta"] = {
                key: fetch_payload.get(key)
                for key in ("count", "start", "end", "timezone", "price_precision", "price_currency")
                if key in fetch_payload
            }
    return out


def _profile_freshness_meta(
    fetch_payload: Any,
    *,
    data_as_of: Optional[str],
    historical_query: bool,
    timeframe: Optional[str] = None,
    window_seconds: Optional[float] = None,
    profile_source: Optional[str] = None,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(fetch_payload, dict):
        fetch_payload = {}
    out: Dict[str, Any] = {}
    fetched_at = fetch_payload.get("data_fetched_at") or fetch_payload.get("as_of")
    if fetched_at not in (None, ""):
        out["fetched_at"] = fetched_at
    if data_as_of not in (None, ""):
        out["data_as_of"] = data_as_of
    for target, source_names in (("timezone", ("timezone",)),):
        for name in source_names:
            value = fetch_payload.get(name)
            if value not in (None, "", [], {}):
                out[target] = value
                break
    if data_as_of:
        observed_at = _parse_start_datetime(data_as_of)
        if observed_at is not None:
            timeframe_seconds = float(
                TIMEFRAME_SECONDS.get(str(timeframe or "").strip().upper(), 0)
                or 0
            )
            if timeframe_seconds > 0:
                stale_after_seconds = max(300.0, timeframe_seconds)
                freshness_basis = "completed_bar_close_timeframe_window"
            else:
                stale_after_seconds = 300.0
                freshness_basis = "latest_observation_fixed_5m"
            age_seconds = max(
                0.0,
                (_utc_now_naive() - observed_at).total_seconds(),
            )
            out.update(
                {
                    "as_of": data_as_of,
                    "data_age_seconds": round(age_seconds, 3),
                    "query_type": "historical" if historical_query else "latest",
                }
            )
            if historical_query:
                out["data_stale"] = None
                out["observation_age_seconds"] = round(age_seconds, 3)
                out["freshness_basis"] = "historical_window_not_applicable"
                out["freshness_applicability"] = "historical_query"
            else:
                data_stale = age_seconds > stale_after_seconds
                session = closed_session_context(
                    symbol or fetch_payload.get("symbol") or out.get("symbol"),
                    now_epoch=_utc_now_naive().replace(tzinfo=timezone.utc).timestamp(),
                    item="volume profile",
                    data_age_seconds=age_seconds,
                )
                if session:
                    data_stale = True
                    out["market_status"] = session.get("market_status")
                    out["market_status_reason"] = session.get("market_status_reason")
                    out["freshness_state"] = "closed_weekend_snapshot"
                    note = session.get("note")
                    if note:
                        out["note"] = note
                out["data_stale"] = data_stale
                out["stale_after_seconds"] = stale_after_seconds
                out["freshness_basis"] = freshness_basis
        return out
    for target, source_names in (
        ("as_of", ("as_of", "data_fetched_at")),
        ("data_age_seconds", ("data_age_seconds", "data_freshness_seconds")),
        ("data_stale", ("data_stale",)),
    ):
        for name in source_names:
            value = fetch_payload.get(name)
            if value not in (None, "", [], {}):
                out[target] = value
                break
    if "data_age_seconds" not in out:
        meta = fetch_payload.get("meta")
        diagnostics = meta.get("diagnostics") if isinstance(meta, dict) else None
        freshness = diagnostics.get("freshness") if isinstance(diagnostics, dict) else None
        if isinstance(freshness, dict):
            age_seconds = freshness.get("data_freshness_seconds")
            if age_seconds is not None:
                out["data_age_seconds"] = age_seconds
            within_policy = freshness.get("last_bar_within_policy_window")
            if within_policy is not None:
                out["data_stale"] = not bool(within_policy)
    out.setdefault("query_type", "latest")
    return out


def _profile_units(profile: Dict[str, Any]) -> Dict[str, str]:
    volume_kind = str(profile.get("volume_kind") or "volume_weight").strip() or "volume_weight"
    return {
        "price": "absolute_price",
        "bucket_size": "absolute_price",
        "poc.price": "absolute_price",
        "vah.price": "absolute_price",
        "val.price": "absolute_price",
        "volume": volume_kind,
        "total_volume": volume_kind,
        "value_area.volume": volume_kind,
    }


def _profile_source_quality(source: Any) -> Dict[str, Any]:
    source_value = str(source or "").strip().lower()
    if source_value == "m1_bars":
        return {
            "profile_source": "m1_bars",
            "volume_profile_accuracy": "approximated_from_m1_bars",
            "volume_source_quality": "estimated_m1_bar_proxy",
            "is_synthetic": True,
            "source_note": (
                "Volume profile is approximated from M1 bars; use source='ticks' "
                "for tick-precise profiles when the window is small enough."
            ),
        }
    if source_value == "ticks":
        return {
            "profile_source": "ticks",
            "volume_profile_accuracy": "tick_precise",
            "volume_source_quality": "raw_ticks",
            "is_synthetic": False,
        }
    return {}


def compute_volume_profile_payload(  # noqa: C901
    *,
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    timeframe: Optional[TimeframeLiteral] = None,
    lookback: Annotated[Optional[int], Field(ge=1)] = None,
    source: VolumeProfileSourceLiteral = "auto",
    price_source: VolumeProfilePriceSourceLiteral = "mid",
    volume_source: VolumeProfileVolumeSourceLiteral = "auto",
    bucket_size: Optional[float] = None,
    bucket_points: Optional[float] = None,
    bucket_count: Optional[int] = None,
    max_buckets: Annotated[int, Field(ge=1)] = 120,
    value_area_pct: float = 70.0,
    reference_price: Optional[float] = None,
    max_tick_window_days: int = _DEFAULT_MAX_TICK_WINDOW_DAYS,
    max_ticks: int = _DEFAULT_MAX_TICKS,
    max_m1_bars: int = _DEFAULT_MAX_M1_BARS,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    bucket_controls = [
        name
        for name, value in (
            ("bucket_size", bucket_size),
            ("bucket_points", bucket_points),
            ("bucket_count", bucket_count),
        )
        if value is not None
    ]
    if len(bucket_controls) > 1:
        return {
            "error": (
                "Choose exactly one volume-profile bucket control: bucket_size, "
                "bucket_points, or bucket_count."
            ),
            "error_code": "volume_profile_conflicting_bucket_controls",
            "conflicting_parameters": bucket_controls,
            "parameter": "bucket_size,bucket_points,bucket_count",
            "remediation": (
                "Pass exactly one of bucket_size, bucket_points, or bucket_count."
            ),
        }
    if start is not None and (timeframe is not None or lookback is not None):
        return {
            "error": (
                "Choose one volume-profile window mode: start/end calendar "
                "bounds, or timeframe/lookback bars. Do not combine start with "
                "timeframe or lookback."
            ),
            "error_code": "volume_profile_conflicting_window_selectors",
            "conflicting_parameters": [
                name
                for name, value in (
                    ("start", start),
                    ("timeframe", timeframe),
                    ("lookback", lookback),
                )
                if value is not None
            ],
            "remediation": (
                "Use start/end or timeframe/lookback, not both."
            ),
        }
    range_error = validate_historical_range(start, end)
    if range_error is not None:
        return range_error
    source_value = str(source or "auto").strip().lower()
    volume_source_value = str(volume_source or "auto").strip().lower()
    if source_value == "m1_bars" and volume_source_value == "tick_count":
        return _m1_tick_count_unsupported_error(source_value)
    try:
        value_area_value = float(value_area_pct)
    except (TypeError, ValueError):
        value_area_value = math.nan
    if not math.isfinite(value_area_value) or not 0.0 < value_area_value <= 100.0:
        return {
            "error": (
                "value_area_pct must be in (0, 100] percent; "
                f"got {value_area_pct!r}"
            ),
            "error_code": "volume_profile_invalid_value_area_pct",
        }
    if lookback is not None and not timeframe:
        return {
            "error": (
                "lookback is a bar count and requires timeframe; "
                "use max_ticks to cap tick rows."
            )
        }
    window = _resolve_profile_window(
        start=start,
        end=end,
        timeframe=timeframe,
        lookback=lookback,
    )
    if window.get("error"):
        return {"error": window["error"]}
    resolved_start = window.get("start")
    resolved_end = window.get("end")
    bar_window: Optional[Dict[str, Any]] = None
    mt5_gateway = create_mt5_gateway(
        ensure_connection_impl=ensure_mt5_connection_or_raise
    )
    mt5_gateway.ensure_connection()
    symbol, symbol_input = resolve_public_symbol(symbol, gateway=mt5_gateway)
    requested_bars = window.get("requested_bars")
    if requested_bars is not None and timeframe is not None:
        # fetch_candles owns its symbol-readiness guard. Resolve the requested
        # bar window before taking this function's guard so the Windows file
        # lock is never recursively acquired through a second file handle.
        resolved_bar_window = _resolve_profile_bar_window(
            symbol=symbol,
            timeframe=str(timeframe),
            bars=int(requested_bars),
            end=resolved_end,
        )
        if resolved_bar_window.get("error"):
            return resolved_bar_window
        resolved_start = resolved_bar_window.get("start")
        resolved_end = resolved_bar_window.get("end")
        if isinstance(resolved_bar_window.get("bar_window"), dict):
            bar_window = dict(resolved_bar_window["bar_window"])
    with _symbol_ready_guard(symbol) as (err, info):
        if err:
            return {"error": err}
        price_digits = _positive_int_attr(info, "digits")
        price_point = _positive_float_attr(info, "point", "trade_tick_size")
    selected = _select_profile_rows(
        symbol=symbol,
        start=resolved_start,
        end=resolved_end,
        source=source,
        price_source=price_source,
        max_tick_window_days=max_tick_window_days,
        max_ticks=max_ticks,
        max_m1_bars=max_m1_bars,
    )
    if selected.get("error"):
        return selected
    selected_source = str(selected.get("source") or "").strip().lower()
    if selected_source == "m1_bars" and volume_source_value == "tick_count":
        return _m1_tick_count_unsupported_error(selected_source)
    requested_price_source = str(price_source or "mid").strip().lower()
    config = VolumeProfileConfig(
        price_source="mid" if selected_source == "m1_bars" else price_source,
        volume_source=volume_source,
        bucket_size=bucket_size,
        bucket_points=bucket_points,
        bucket_count=bucket_count,
        max_buckets=max_buckets,
        value_area_fraction=value_area_value / 100.0,
        price_point=price_point,
        price_digits=price_digits,
        reference_price=reference_price,
    )
    profile = compute_volume_profile(selected.get("rows", []), config)
    if profile.get("error"):
        profile["symbol"] = symbol
        profile.update(_profile_source_quality(selected.get("source")))
        profile["price_point"] = price_point
        profile["price_digits"] = price_digits
        profile["diagnostics"] = {
            **(selected.get("diagnostics") or {}),
            **(profile.get("diagnostics") or {}),
        }
        requested_start = _parse_start_datetime(resolved_start) if resolved_start else None
        requested_end = _parse_end_datetime(resolved_end) if resolved_end else None
        if requested_start is not None and requested_end is not None:
            closure_context = _weekend_tick_coverage_context(
                symbol,
                requested_start=requested_start,
                requested_end=requested_end,
                observed_start=None,
                observed_end=None,
            )
            if closure_context.get("entire_request_closed"):
                profile.update(
                    {
                        "no_data_reason": "market_closed_weekend",
                        "market_status": "closed",
                        "market_status_reason": "weekend",
                        "market_status_source": "standard_weekend_hours",
                        "scheduled_closures": closure_context.get(
                            "scheduled_closures"
                        ),
                        "requested_window": {
                            "start": _format_window_timestamp(resolved_start),
                            "end": _format_window_timestamp(
                                resolved_end,
                                end_bound=True,
                            ),
                        },
                        "data_quality": {
                            "status": "not_applicable",
                            "reason": "market_closed_weekend",
                        },
                    }
                )
        return profile
    profile["symbol"] = symbol
    profile.update(_profile_source_quality(selected.get("source")))
    profile["price_source_requested"] = requested_price_source
    profile["price_source_effective"] = (
        "lch_equal_weight_proxy" if selected_source == "m1_bars" else requested_price_source
    )
    profile["price_source"] = profile["price_source_effective"]
    if selected_source == "m1_bars":
        profile["proxy_prices"] = ["low", "close", "high"]
        profile["allocation_method"] = "equal_weight"
        profile["volume_is_synthetic"] = True
    selected_diagnostics = selected.get("diagnostics")
    selected_reason = None
    if isinstance(selected_diagnostics, dict):
        selected_reason = selected_diagnostics.get(
            "auto_fallback_reason"
        ) or selected_diagnostics.get("auto_source_reason")
    profile["source_decision"] = {
        "requested": str(source),
        "selected": selected.get("source"),
        "reason": selected_reason or "explicit_source",
    }
    profile["window"] = _observed_profile_window(
        selected.get("rows", []),
        fallback_start=resolved_start,
        fallback_end=resolved_end,
    )
    profile["requested_window"] = {
        "start": _format_window_timestamp(resolved_start),
        "end": _format_window_timestamp(resolved_end, end_bound=True),
    }
    if bar_window is not None:
        profile["bar_window"] = bar_window
    profile["diagnostics"] = {
        **(selected.get("diagnostics") or {}),
        **(profile.get("diagnostics") or {}),
    }
    if (
        str(selected.get("source") or "").lower() == "ticks"
        and profile["diagnostics"].get("tick_limit_reached") is True
    ):
        profile["truncated"] = True
        profile["truncation_reason"] = "max_ticks"
        profile["volume_profile_accuracy"] = "tick_truncated"
        profile["volume_source_quality"] = "partial_raw_ticks"
        profile["source_note"] = (
            "Prices and volumes come from raw ticks, but max_ticks truncated the "
            "requested window; POC and value-area levels describe only the retained sample."
        )
        tick_rows = int(profile["diagnostics"].get("tick_rows") or 0)
        max_ticks_value = int(profile["diagnostics"].get("requested_max_ticks") or tick_rows)
        profile["data_quality"] = {
            "status": "partial",
            "reason": "max_ticks",
        }
        profile["coverage_note"] = (
            f"Profile uses only the latest {tick_rows} ticks because max_ticks="
            f"{max_ticks_value} was reached; it does not represent the full requested window."
        )
    elif str(selected.get("source") or "").lower() == "ticks":
        requested_start = _parse_start_datetime(resolved_start) if resolved_start else None
        requested_end = _parse_end_datetime(resolved_end) if resolved_end else None
        observed_start = _parse_start_datetime(profile["window"].get("start"))
        observed_end = _parse_end_datetime(profile["window"].get("end"))
        if all(
            value is not None
            for value in (requested_start, requested_end, observed_start, observed_end)
        ):
            assert requested_start is not None
            assert requested_end is not None
            assert observed_start is not None
            assert observed_end is not None
            requested_seconds = max(
                0.0,
                float((requested_end - requested_start).total_seconds()),
            )
            start_gap = max(
                0.0,
                float((observed_start - requested_start).total_seconds()),
            )
            end_gap = max(
                0.0,
                float((requested_end - observed_end).total_seconds()),
            )
            gap_tolerance = max(300.0, requested_seconds * 0.05)
            closure_context = _weekend_tick_coverage_context(
                symbol,
                requested_start=requested_start,
                requested_end=requested_end,
                observed_start=observed_start,
                observed_end=observed_end,
            )
            start_gap_expected = bool(
                closure_context.get("start_gap_expected")
            )
            end_gap_expected = bool(closure_context.get("end_gap_expected"))
            profile["diagnostics"].update(
                {
                    "requested_window_seconds": round(requested_seconds, 3),
                    "observed_start_gap_seconds": round(start_gap, 3),
                    "observed_end_gap_seconds": round(end_gap, 3),
                    "window_gap_tolerance_seconds": round(gap_tolerance, 3),
                    "start_gap_explained_by_scheduled_closure": start_gap_expected,
                    "end_gap_explained_by_scheduled_closure": end_gap_expected,
                }
            )
            scheduled_closures = closure_context.get("scheduled_closures")
            if scheduled_closures:
                profile["scheduled_closures"] = scheduled_closures
                profile["coverage_note"] = (
                    "Tick-window coverage excludes the disclosed scheduled "
                    "weekend closure from endpoint-gap checks."
                )
            if (
                (start_gap > gap_tolerance and not start_gap_expected)
                or (end_gap > gap_tolerance and not end_gap_expected)
            ):
                profile["truncated"] = True
                profile["truncation_reason"] = "incomplete_tick_window"
                profile["volume_profile_accuracy"] = "tick_partial_window"
                profile["volume_source_quality"] = "partial_raw_ticks"
                profile["data_quality"] = {
                    "status": "partial",
                    "reason": "incomplete_tick_window",
                }
                profile["coverage_note"] = (
                    "Raw ticks do not cover the requested profile window within "
                    f"the {round(gap_tolerance, 3)} second tolerance; POC and value "
                    "area describe only the observed window."
                )
    fetch_payload = selected.get("fetch_payload")
    window = profile.get("window") if isinstance(profile.get("window"), dict) else {}
    window_days = _window_days(window.get("start"), window.get("end"))
    profile_source_name = str(
        profile.get("profile_source") or selected.get("source") or ""
    )
    bar_window_meta = profile.get("bar_window")
    data_as_of = None
    if profile_source_name == "m1_bars":
        if isinstance(bar_window_meta, dict):
            data_as_of = bar_window_meta.get("last_bar_close")
        if not data_as_of:
            last_open = _parse_start_datetime(window.get("end"))
            if last_open is not None:
                data_as_of = _format_window_timestamp(
                    last_open + timedelta(minutes=1)
                )
    if not data_as_of:
        data_as_of = window.get("end")
    profile.update(
        _profile_freshness_meta(
            fetch_payload,
            data_as_of=data_as_of,
            historical_query=(
                start is not None or end is not None
            ),
            timeframe=timeframe,
            window_seconds=(
                None if window_days is None else float(window_days) * 86400.0
            ),
            profile_source=profile_source_name,
            symbol=str(profile.get("symbol") or ""),
        )
    )
    profile["units"] = _profile_units(profile)
    merged_warnings = _merge_profile_warnings(
        profile.get("warnings"),
        selected.get("warnings"),
        profile.get("warning"),
    )
    if merged_warnings:
        profile["warnings"] = merged_warnings
    profile["fetch_payload"] = fetch_payload
    profile["symbol"] = symbol
    if symbol_input is not None:
        profile["symbol_input"] = symbol_input
    return attach_mt5_source(
        _profile_detail_payload(profile, detail),
        gateway=mt5_gateway,
    )


@mcp.tool()
def volume_profile_levels(  # noqa: PLR0913
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    timeframe: Optional[TimeframeLiteral] = None,
    lookback: Annotated[Optional[int], Field(ge=1)] = None,
    source: VolumeProfileSourceLiteral = "auto",
    price_source: VolumeProfilePriceSourceLiteral = "mid",
    volume_source: VolumeProfileVolumeSourceLiteral = "auto",
    bucket_size: Optional[float] = None,
    bucket_points: Optional[float] = None,
    bucket_count: Optional[int] = None,
    max_buckets: Annotated[int, Field(ge=1)] = 120,
    value_area_pct: Annotated[float, Field(gt=0.0, le=100.0)] = 70.0,
    reference_price: Optional[float] = None,
    max_tick_window_days: int = _DEFAULT_MAX_TICK_WINDOW_DAYS,
    max_ticks: int = _DEFAULT_MAX_TICKS,
    max_m1_bars: int = _DEFAULT_MAX_M1_BARS,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Compute volume-profile POC, VAH, and VAL from ticks or M1-bar approximation.

    With no window arguments, the profile covers the latest 24 hours and fetches
    at most 50,000 ticks. `source="auto"` uses bounded raw ticks for short windows and falls back to
    M1-bar approximation for larger windows. `lookback` is always a bar count and
    requires `timeframe`; use `max_ticks` to cap tick rows. When `timeframe` is
    provided without `lookback`, the window defaults to 200 bars. `price_source="mid"`
    is the safe default for FX tick data where `last` is often unavailable. M1
    approximation always reports an `lch_equal_weight_proxy` effective price source.
    """

    def _run() -> Dict[str, Any]:
        try:
            detail_value = str(detail or "compact").strip().lower()
            return compute_volume_profile_payload(
                symbol=symbol,
                start=start,
                end=end,
                timeframe=timeframe,
                lookback=lookback,
                source=source,
                price_source=price_source,
                volume_source=volume_source,
                bucket_size=bucket_size,
                bucket_points=bucket_points,
                bucket_count=bucket_count,
                max_buckets=max_buckets,
                value_area_pct=value_area_pct,
                reference_price=reference_price,
                max_tick_window_days=max_tick_window_days,
                max_ticks=max_ticks,
                max_m1_bars=max_m1_bars,
                detail=detail_value,  # type: ignore[arg-type]
            )
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error computing volume profile levels: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="volume_profile_levels",
        symbol=symbol,
        start=start,
        end=end,
        timeframe=timeframe,
        lookback=lookback,
        source=source,
        price_source=price_source,
        volume_source=volume_source,
        bucket_size=bucket_size,
        bucket_points=bucket_points,
        bucket_count=bucket_count,
        max_buckets=max_buckets,
        value_area_pct=value_area_pct,
        max_tick_window_days=max_tick_window_days,
        max_ticks=max_ticks,
        max_m1_bars=max_m1_bars,
        detail=detail,
        func=_run,
    )
