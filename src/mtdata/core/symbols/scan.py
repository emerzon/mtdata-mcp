"""Market-scan and top-markets MCP tools."""

import logging
import math
import re
import time
from datetime import datetime, timezone
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
)

from pydantic import Field

from ...services.data_service.candles import _drop_incomplete_tail
from ...shared.constants import (
    FETCH_RETRY_ATTEMPTS,
    FETCH_RETRY_DELAY,
    TIMEFRAME_MAP,
    TIMEFRAME_SECONDS,
)
from ...shared.market_units import forex_points_per_pip
from ...shared.schema import (
    DetailLiteral,
    TimeframeLiteral,
)
from ...shared.symbols import _alnum_upper
from ...shared.validators import invalid_timeframe_error
from ...utils.freshness import (
    QUOTE_STALE_SECONDS,
    closed_session_context,
    format_age_seconds,
    format_freshness_label,
)
from ...utils.market_metadata import (
    FRESHNESS_ANCHOR_WALL_CLOCK,
    FRESHNESS_METRIC_LAST_COMPLETED_BAR_AGE,
    TICK_VOLUME_SEMANTICS,
    TICK_VOLUME_UNIT,
    build_tick_freshness_context,
)
from ...utils.mt5 import (
    MT5ConnectionError,
    _ensure_symbol_ready,
    _mt5_copy_rates_from,
    _mt5_copy_rates_from_pos,
    _symbol_ready_guard,
    _symbol_visibility_snapshot_guard,
    account_currency_from_gateway,
    ensure_mt5_connection_or_raise,
    mt5,
)
from ...utils.quote import (
    compute_spread_metrics,
    enforce_quote_execution_readiness,
    resolve_quote_tick,
    tick_epoch,
    tick_value,
)
from ...utils.symbol import _extract_group_path as _extract_group_path_util
from ...utils.symbol import (
    _normalize_group_path_query,
)
from ...utils.time import (
    _format_time_minimal,
    _format_time_second_explicit,
    bar_close_epoch,
)
from ...utils.utils import (
    _normalize_limit,
    _table_from_rows,
)
from .._mcp_instance import mcp
from ..error_envelope import build_error_payload
from ..execution_logging import run_logged_operation
from ..mt5_gateway import create_mt5_gateway
from ..output_contract import (
    attach_collection_contract,
    build_pagination_meta,
    normalize_output_verbosity_detail,
)
from ..runtime_metadata import build_mt5_source_provenance
from .classify import (
    _case_insensitive_sort_key,
    _clean_broker_text,
    _invalid_symbol_category_error,
    _normalize_symbol_category_filter,
    _symbol_category,
    _symbol_suggestion_from_info,
)

logger = logging.getLogger("mtdata.core.symbols")

_MARKET_SCAN_STALE_BAR_SECONDS = 7 * 24 * 60 * 60

_MARKET_SCAN_STALE_QUOTE_SECONDS = QUOTE_STALE_SECONDS

_TOP_MARKETS_MAX_CANDIDATES = 250

_TOP_MARKETS_DEFAULT_SCAN_BUDGET_SECONDS = 30.0

_MARKET_SCAN_MAX_CANDIDATES = _TOP_MARKETS_MAX_CANDIDATES

def _market_scan_is_tradable(symbol: Any) -> bool:
    disabled_trade_mode = getattr(mt5, "SYMBOL_TRADE_MODE_DISABLED", None)
    if disabled_trade_mode is None:
        return True
    return getattr(symbol, "trade_mode", None) != disabled_trade_mode

def _market_scan_base_row(symbol: Any) -> Dict[str, Any]:
    return {
        "symbol": _clean_broker_text(getattr(symbol, "name", None)),
        "group": _clean_broker_text(_extract_group_path_util(symbol)),
        "asset_class": _symbol_category(symbol),
        "description": _clean_broker_text(getattr(symbol, "description", None)),
    }

def _market_scan_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out

def _market_scan_bar_int(value: Any) -> Optional[int]:
    try:
        out = int(value)
    except Exception:
        return None
    return out

def _market_scan_round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), max(0, int(digits)))

def _market_scan_points(value: Optional[float]) -> Optional[int | float]:
    rounded = _market_scan_round(value, digits=4)
    if rounded is None:
        return None
    nearest = round(float(rounded))
    if math.isclose(float(rounded), float(nearest), abs_tol=1e-9):
        return int(nearest)
    return rounded

def _market_scan_stale_bar_seconds(timeframe: Optional[str]) -> int:
    seconds = TIMEFRAME_SECONDS.get(str(timeframe or "").strip().upper())
    if seconds:
        return max(1, int(seconds))
    return int(_MARKET_SCAN_STALE_BAR_SECONDS)

def _market_scan_freshness_fields(
    bar_time: Optional[float],
    *,
    timeframe: Optional[str] = None,
    symbol: Any = None,
) -> Dict[str, Any]:
    if bar_time is None:
        return {}
    try:
        now_epoch = float(time.time())
        close_epoch = bar_close_epoch(bar_time, str(timeframe))
        age_seconds = max(0.0, now_epoch - close_epoch)
    except Exception:
        return {}
    stale_after_seconds = _market_scan_stale_bar_seconds(timeframe)
    data_stale = age_seconds > stale_after_seconds
    symbol_name = getattr(symbol, "name", symbol)
    session_symbol = (
        None if _symbol_category(symbol) == "crypto" else symbol_name
    )
    closed_session = closed_session_context(
        session_symbol,
        now_epoch=now_epoch,
        item="bar",
        data_age_seconds=age_seconds,
    )
    fields: Dict[str, Any] = {
        "data_freshness_seconds": _market_scan_round(age_seconds, digits=3),
        "data_freshness_anchor": FRESHNESS_ANCHOR_WALL_CLOCK,
        "data_freshness_metric": FRESHNESS_METRIC_LAST_COMPLETED_BAR_AGE,
        "stale_after_seconds": stale_after_seconds,
        "bar_age_hours": _market_scan_round(age_seconds / 3600.0, digits=3),
        "data_stale": data_stale,
        "history_policy_ok": not data_stale and not bool(closed_session),
        "freshness": format_freshness_label(
            data_stale=data_stale,
            age_seconds=age_seconds,
            item="bar",
        ),
    }
    if closed_session:
        fields.update(closed_session)
        policy_relaxed = bool(closed_session.get("freshness_policy_relaxed"))
        fields["freshness"] = format_freshness_label(
            data_stale=fields.get("data_stale"),
            market_status=fields.get("market_status") if policy_relaxed else None,
            market_status_reason=(
                fields.get("market_status_reason") if policy_relaxed else None
            ),
            age_seconds=age_seconds,
            item="bar",
        )
    elif not data_stale:
        age_text = format_age_seconds(age_seconds)
        fields["freshness"] = f"latest completed bar, {age_text} ago"
    if fields["data_stale"]:
        fields["stale_warning"] = (
            "Completed bar data may be stale; latest bar is "
            f"{fields['bar_age_hours']} hours old."
        )
    return fields

def _quote_staleness_fields(
    tick_time: Optional[float],
    *,
    symbol: Any = None,
) -> Dict[str, Any]:
    if tick_time is None:
        return {}
    try:
        now_epoch = float(time.time())
        signed_age_seconds = now_epoch - float(tick_time)
    except Exception:
        return {}
    fields = build_tick_freshness_context(
        symbol,
        tick_epoch=tick_time,
        now_epoch=now_epoch,
        item="tick",
        stale_after_seconds=_MARKET_SCAN_STALE_QUOTE_SECONDS,
        age_rounder=lambda value: _market_scan_round(value, digits=3),
    )
    fields["data_age"] = (
        f"{format_age_seconds(-signed_age_seconds)} ahead of wall clock"
        if signed_age_seconds < 0.0
        else format_age_seconds(signed_age_seconds)
    )
    if fields.get("timestamp_in_future"):
        fields["warning"] = fields.get("timestamp_warning")
    elif fields["data_stale"]:
        fields["warning"] = (
            "Live quote timestamp is older than "
            f"{int(_MARKET_SCAN_STALE_QUOTE_SECONDS)} seconds."
        )
    return fields

_MARKET_SCAN_BAR_FRESHNESS_FIELDS = {
    "data_freshness_seconds": "bar_age_seconds",
    "data_freshness_anchor": "bar_freshness_anchor",
    "data_freshness_metric": "bar_freshness_metric",
    "stale_after_seconds": "bar_stale_after_seconds",
    "bar_age_hours": "bar_age_hours",
    "data_stale": "bar_stale",
    "history_policy_ok": "history_policy_ok",
    "freshness": "bar_freshness",
    "market_status": "bar_market_status",
    "market_status_reason": "bar_market_status_reason",
    "freshness_policy_relaxed": "bar_freshness_policy_relaxed",
    "stale_warning": "bar_stale_warning",
}

def _market_scan_bar_freshness_fields(
    bar_time: Optional[float],
    *,
    timeframe: Optional[str] = None,
    symbol: Any = None,
) -> Dict[str, Any]:
    """Return bar-policy fields that cannot overwrite quote safety fields."""
    fields = _market_scan_freshness_fields(
        bar_time,
        timeframe=timeframe,
        symbol=symbol,
    )
    return {
        output_name: fields[input_name]
        for input_name, output_name in _MARKET_SCAN_BAR_FRESHNESS_FIELDS.items()
        if input_name in fields
    }

def _market_scan_quote_freshness_fields(
    tick_time: Optional[float],
    *,
    symbol: Any = None,
) -> Dict[str, Any]:
    if tick_time is None:
        return {}
    return {
        "tick_time": _format_time_second_explicit(tick_time),
        **_quote_staleness_fields(tick_time, symbol=symbol),
    }

def _market_scan_points_per_pip(symbol: Any, *, point: float, digits: int) -> Optional[float]:
    return forex_points_per_pip(
        str(getattr(symbol, "name", "") or ""),
        path=str(getattr(symbol, "path", "") or ""),
        point=point,
        digits=digits,
    )

def _build_market_scan_spread_row(
    symbol: Any,
    mt5_gateway: Any,
    *,
    spread_cost_currency: Optional[str] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    raw_tick = mt5_gateway.symbol_info_tick(symbol.name)
    tick, quote_source = resolve_quote_tick(
        mt5_gateway,
        symbol.name,
        raw_tick,
        now_epoch=time.time(),
        stale_after_seconds=_MARKET_SCAN_STALE_QUOTE_SECONDS,
    )
    if tick is None:
        return None, f"Failed to get tick data: {mt5_gateway.last_error()}"

    bid = _market_scan_float(tick_value(tick, "bid"))
    ask = _market_scan_float(tick_value(tick, "ask"))
    tick_time = tick_epoch(tick)
    point = _market_scan_float(getattr(symbol, "point", 0.0)) or 0.0
    tick_size = _market_scan_float(getattr(symbol, "trade_tick_size", 0.0)) or 0.0
    trade_tick_value = (
        _market_scan_float(getattr(symbol, "trade_tick_value", 0.0)) or 0.0
    )
    digits = max(0, int(getattr(symbol, "digits", 0) or 0))

    points_per_pip = _market_scan_points_per_pip(symbol, point=point, digits=digits)
    spread_metrics = compute_spread_metrics(
        bid,
        ask,
        point=point,
        points_per_pip=points_per_pip,
        tick_size=tick_size,
        tick_value_money=trade_tick_value,
        account_currency=spread_cost_currency,
    )
    spread_quality = spread_metrics["spread_quality"]
    if spread_quality == "one_sided":
        return None, "Bid/ask quote is unavailable."
    if spread_quality == "inverted":
        return None, "Bid/ask quote is invalid."
    spread_valid = spread_metrics["spread_valid"]
    spread_abs = spread_metrics["spread"]
    mid = spread_metrics["mid"]
    spread_points = spread_metrics["spread_points"]
    spread_pips = spread_metrics["spread_pips"]
    spread_pct = spread_metrics["spread_pct"]
    spread_cost_per_lot = spread_metrics["spread_cost_per_lot"]
    pricing_basis = spread_metrics["pricing_basis"]

    quote_freshness = _market_scan_quote_freshness_fields(
        tick_time,
        symbol=symbol.name,
    )
    quote_freshness.update(quote_source)
    enforce_quote_execution_readiness(
        quote_freshness,
        bid=bid,
        ask=ask,
        quote_source_conflict=quote_source.get("quote_source_conflict"),
        point=point,
    )

    row = _market_scan_base_row(symbol)
    row.update(
        {
            "bid": _market_scan_round(bid, digits=digits),
            "ask": _market_scan_round(ask, digits=digits),
            "mid": _market_scan_round(mid, digits=digits + 1),
            "quote_as_of": (
                _format_time_second_explicit(tick_time)
                if tick_time is not None
                else None
            ),
            **quote_source,
            **quote_freshness,
            "spread": _market_scan_round(spread_abs, digits=digits),
            "spread_points": _market_scan_points(spread_points),
            "spread_pips": _market_scan_round(spread_pips, digits=4),
            "spread_pct": _market_scan_round(spread_pct, digits=6),
            "spread_cost_per_lot": _market_scan_round(spread_cost_per_lot, digits=6),
            "spread_valid": spread_valid,
            "spread_quality": spread_quality,
            "pricing_basis": pricing_basis,
        }
    )
    if spread_cost_per_lot is not None and spread_cost_currency:
        row["spread_cost_currency"] = spread_cost_currency
    return row, None

def _market_scan_completed_bar_age(
    rates: Any,
    *,
    timeframe: str,
    now_epoch: float,
) -> Optional[float]:
    if rates is None or len(rates) < 1:
        return None
    latest_time = _market_scan_float(rates[-1]["time"])
    if latest_time is None:
        return None
    latest_close = bar_close_epoch(latest_time, timeframe)
    if latest_close is None:
        return None
    return max(0.0, now_epoch - latest_close)


def _market_scan_open_session_stale(
    symbol: str,
    *,
    timeframe: str,
    rates: Any,
    now_epoch: float,
) -> bool:
    age_seconds = _market_scan_completed_bar_age(
        rates,
        timeframe=timeframe,
        now_epoch=now_epoch,
    )
    if (
        age_seconds is None
        or age_seconds <= _market_scan_stale_bar_seconds(timeframe)
    ):
        return False
    closed = closed_session_context(
        symbol,
        now_epoch=now_epoch,
        item="bar",
        data_age_seconds=age_seconds,
    )
    return not bool(closed and closed.get("freshness_policy_relaxed"))


def _market_scan_completed_rates(
    symbol: str,
    *,
    timeframe: str,
    mt5_timeframe: Any,
    count: int,
) -> Any:
    requested = max(1, int(count))

    def _completed(raw_rates: Any, *, now_epoch: float) -> Any:
        if raw_rates is None or len(raw_rates) < 1:
            return raw_rates
        completed = _drop_incomplete_tail(
            raw_rates,
            timeframe,
            current_time_epoch=now_epoch,
        )
        if len(completed) > requested:
            completed = completed[-requested:]
        return completed

    now_epoch = float(time.time())
    _ensure_symbol_ready(symbol)
    rates = _completed(
        _mt5_copy_rates_from_pos(symbol, mt5_timeframe, 0, requested + 1),
        now_epoch=now_epoch,
    )
    if rates is None or len(rates) < 1:
        return rates
    if not _market_scan_open_session_stale(
        symbol,
        timeframe=timeframe,
        rates=rates,
        now_epoch=now_epoch,
    ):
        return rates
    saw_refresh = False
    for attempt in range(FETCH_RETRY_ATTEMPTS):
        if attempt:
            time.sleep(FETCH_RETRY_DELAY)
        refreshed = _completed(
            _mt5_copy_rates_from(
                symbol,
                mt5_timeframe,
                datetime.fromtimestamp(now_epoch, tz=timezone.utc),
                requested + 1,
            ),
            now_epoch=now_epoch,
        )
        if refreshed is None or len(refreshed) < 1:
            continue
        saw_refresh = True
        rates = refreshed
        if not _market_scan_open_session_stale(
            symbol,
            timeframe=timeframe,
            rates=rates,
            now_epoch=now_epoch,
        ):
            return rates
    if saw_refresh:
        return None
    return rates

def _project_market_scan_completed_bars(
    symbol: Any,
    timeframe: str,
    latest_bar: Any,
    previous_bar: Any,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    open_price = _market_scan_float(latest_bar["open"])
    close_price = _market_scan_float(latest_bar["close"])
    previous_close = _market_scan_float(previous_bar["close"])
    if open_price is None or close_price is None:
        return None, "Completed bar is missing open/close prices."
    if previous_close is None:
        return None, "Previous completed bar is missing its close price."
    if previous_close == 0:
        return None, "Previous completed bar close price is zero."

    digits = max(0, int(getattr(symbol, "digits", 0) or 0))
    bar_time = _market_scan_float(latest_bar["time"])
    row = _market_scan_base_row(symbol)
    row.update(
        {
            "timeframe": timeframe,
            "data_source": f"{timeframe}_bars",
            "time": _format_time_minimal(bar_time) if bar_time is not None else None,
            **_market_scan_bar_freshness_fields(
                bar_time,
                timeframe=timeframe,
                symbol=symbol,
            ),
            "previous_close": _market_scan_round(previous_close, digits=digits),
            "open": _market_scan_round(open_price, digits=digits),
            "close": _market_scan_round(close_price, digits=digits),
            "bar_close": _market_scan_round(close_price, digits=digits),
            "price_currency": str(
                getattr(symbol, "currency_profit", "") or ""
            ).strip()
            or None,
            "price_basis": "mt5_latest_completed_bar_close",
            "price_point": _market_scan_float(getattr(symbol, "point", None)),
            "tick_volume": _market_scan_bar_int(latest_bar["tick_volume"]),
            "real_volume": _market_scan_bar_int(latest_bar["real_volume"]),
            "price_change_pct": _market_scan_round(
                ((close_price - previous_close) / previous_close) * 100.0,
                digits=6,
            ),
            "price_change_basis": "previous_completed_close_to_latest_completed_close",
            "price_change_period": {
                "bars": 1,
                "timeframe": timeframe,
                "bar_state": "completed",
            },
            "gap_pct": _market_scan_round(
                ((open_price - previous_close) / previous_close) * 100.0,
                digits=6,
            ),
            "gap_basis": "previous_completed_close_to_latest_completed_open",
        }
    )
    return row, None


def _build_market_scan_bar_row(
    symbol: Any,
    timeframe: str,
    mt5_timeframe: Any,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    rates = _market_scan_completed_rates(
        symbol.name,
        timeframe=timeframe,
        mt5_timeframe=mt5_timeframe,
        count=2,
    )
    if rates is None or len(rates) < 2:
        return None, f"At least two completed {timeframe} bars are required."
    return _project_market_scan_completed_bars(
        symbol,
        timeframe,
        rates[-1],
        rates[-2],
    )

def _market_scan_table(
    headers: List[str],
    rows: List[Dict[str, Any]],
    *,
    include_contract_meta: bool = True,
) -> Dict[str, Any]:
    ordered_rows = [[row.get(header) for header in headers] for row in rows]
    result = _table_from_rows(headers, ordered_rows)
    return attach_collection_contract(
        result,
        collection_kind="table",
        rows=result.get("data"),
        include_contract_meta=include_contract_meta,
    )

def _market_scan_contract_table(
    headers: List[str],
    rows: List[Dict[str, Any]],
    *,
    include_columns: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "rows": [dict(row) for row in rows],
        "row_count": int(len(rows)),
    }
    if include_columns:
        columns = [str(header) for header in headers]
        seen = set(columns)
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in row.keys():
                column = str(key)
                if column not in seen:
                    columns.append(column)
                    seen.add(column)
        out["columns"] = columns
    return out

def _project_market_scan_rows(
    headers: List[str],
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    projected: List[Dict[str, Any]] = []
    for row in rows:
        out = {header: row.get(header) for header in headers}
        projected.append(out)
    return projected

def _compact_market_scan_projection(
    headers: List[str],
    rows: List[Dict[str, Any]],
) -> tuple[List[str], Dict[str, Any]]:
    """Prune empty columns and hoist metadata repeated across scan rows."""
    projected_headers = [
        header
        for header in headers
        if any(row.get(header) is not None for row in rows)
    ]
    shared: Dict[str, Any] = {}
    if len(rows) > 1:
        for header in (
            "price_basis",
            "bar_market_status_reason",
            "bar_freshness_policy_relaxed",
        ):
            values = [row.get(header) for row in rows]
            first = values[0]
            if first is not None and all(value == first for value in values[1:]):
                shared[header] = first
                projected_headers = [
                    candidate
                    for candidate in projected_headers
                    if candidate != header
                ]

        timestamp_warnings = [
            str(row.get("timestamp_warning"))
            for row in rows
            if row.get("timestamp_warning") not in (None, "")
        ]
        unique_warnings = list(dict.fromkeys(timestamp_warnings))
        if len(timestamp_warnings) > len(unique_warnings):
            shared["warnings"] = unique_warnings
            projected_headers = [
                header
                for header in projected_headers
                if header != "timestamp_warning"
            ]
    return projected_headers, shared

_MARKET_SCAN_UNITS = {
    "bid": "price",
    "ask": "price",
    "mid": "price",
    "close": "price",
    "bar_close": "price",
    "previous_close": "price",
    "price_change_pct": "percent (1.0 = 1%)",
    "live_price_change_pct": "percent (1.0 = 1%)",
    "gap_pct": "percent (1.0 = 1%)",
    "tick_volume": TICK_VOLUME_UNIT,
    "real_volume": "traded_volume",
    "spread_points": "broker_points",
    "spread_pips": "pips (forex_only; null when not applicable)",
    "spread_pct": "percent (1.0 = 1%)",
    "spread_cost_per_lot": "currency_per_lot_estimate",
    "rsi": "0_100",
    "sma_distance_pct": "percent (1.0 = 1%)",
    "price_point": "broker_price_increment",
    "data_age_seconds": "seconds",
    "data_freshness_seconds": "seconds",
    "stale_after_seconds": "seconds",
    "bar_age_seconds": "seconds",
    "bar_stale_after_seconds": "seconds",
    "bar_age_hours": "hours",
}

def _market_scan_ranking_basis(rank_by: str) -> str:
    if rank_by in {"live_price_change_pct", "abs_live_price_change_pct"}:
        return "previous_completed_close_to_live_quote_mid"
    if rank_by in {"spread_pct", "spread"}:
        return "live_quote_bid_ask"
    return "completed_bar_metric"

def _attach_market_scan_rank_gap_warning(
    payload: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> None:
    rank_by = str(payload.get("rank_by") or "")
    if rank_by not in {"price_change_pct", "abs_price_change_pct"}:
        return
    max_gap = 0.0
    for row in rows:
        close = _market_scan_float(row.get("close"))
        mid = _market_scan_float(row.get("mid"))
        if mid is None:
            bid = _market_scan_float(row.get("bid"))
            ask = _market_scan_float(row.get("ask"))
            if bid is not None and ask is not None:
                mid = (bid + ask) / 2.0
        if close is None or mid is None or mid <= 0:
            continue
        max_gap = max(max_gap, abs(close - mid) / mid)
    if max_gap < 0.0015:
        return
    warning = (
        "Ranking uses completed-bar close (price_change_pct). Live quotes diverge "
        "from that close; use --rank-by abs_live_price_change_pct for executable "
        "prices."
    )
    existing = payload.get("warnings")
    if isinstance(existing, list):
        if warning not in existing:
            payload["warnings"] = [*existing, warning]
    elif existing:
        payload["warnings"] = [existing, warning]
    else:
        payload["warnings"] = [warning]

def _market_scan_units_for_rows(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    seen_fields = {
        str(key)
        for row in rows
        if isinstance(row, dict)
        for key, value in row.items()
        if value is not None
    }
    units = {
        key: unit
        for key, unit in _MARKET_SCAN_UNITS.items()
        if key in seen_fields
    }
    if "spread_pips" in units:
        has_non_fx_spread_row = any(
            isinstance(row, dict)
            and row.get("spread_points") is not None
            and row.get("spread_pips") is None
            for row in rows
        )
        if has_non_fx_spread_row:
            units["spread_pips"] = "pips (forex_only; omitted when not applicable)"
    return units

def _attach_top_markets_units(
    out: Dict[str, Any],
    *row_groups: List[Dict[str, Any]],
    headers: Optional[List[str]] = None,
) -> None:
    rows = [
        row
        for group in row_groups
        for row in group
        if isinstance(row, dict)
    ]
    units = _market_scan_units_for_rows(rows)
    allowed_headers = [str(header) for header in (headers or []) if str(header)]
    if not allowed_headers:
        columns = out.get("columns")
        if isinstance(columns, list):
            allowed_headers = [str(header) for header in columns if str(header)]
    if allowed_headers:
        allowed = set(allowed_headers)
        units = {key: unit for key, unit in units.items() if key in allowed}
    if units:
        out["units"] = units
    _attach_market_scan_volume_semantics(out, units)

def _attach_market_scan_volume_semantics(
    out: Dict[str, Any],
    units: Dict[str, str],
) -> None:
    if units.get("tick_volume") in {TICK_VOLUME_UNIT, "broker_tick_count"}:
        out["volume_type"] = "tick_volume"
        out["volume_semantics"] = TICK_VOLUME_SEMANTICS


def _attach_tick_volume_comparability(
    out: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> None:
    asset_classes = sorted(
        {
            str(row.get("asset_class") or "").strip()
            for row in rows
            if isinstance(row, dict) and str(row.get("asset_class") or "").strip()
        }
    )
    if not asset_classes:
        return
    comparable = len(asset_classes) == 1
    out["rank_comparable"] = comparable
    out["ranking_asset_classes"] = asset_classes
    if comparable:
        return
    warning = (
        "Raw broker tick counts measure feed-update activity and are not a "
        "comparable traded-liquidity measure across asset classes."
    )
    out["comparison_warning"] = warning
    out["ranking_remediation"] = (
        "Retry with --category or --group to rank a homogeneous universe."
    )
    warnings = out.setdefault("warnings", [])
    if warning not in warnings:
        warnings.append(warning)

def _market_scan_contract_meta(
    *,
    request: Dict[str, Any],
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "tool": "market_scan",
        "request": {
            key: value for key, value in request.items() if value is not None
        },
        "runtime": {},
    }
    if stats:
        out["stats"] = {
            key: value for key, value in stats.items() if value is not None
        }
    return out

def _market_scan_error(
    message: str,
    *,
    code: str,
    request: Dict[str, Any],
    stats: Optional[Dict[str, Any]] = None,
    details: Any = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    out = build_error_payload(
        message,
        code=str(code),
        operation="market_scan",
    )
    out["meta"] = _market_scan_contract_meta(request=request, stats=stats)
    if details not in (None, [], {}):
        out["details"] = details
    if warnings:
        out["warnings"] = warnings
    return out

def _attach_market_scan_source(payload: Dict[str, Any], gateway: Any = None) -> Dict[str, Any]:
    """Attach broker identity to MT5-backed scan outcomes, including failures."""
    out = dict(payload)
    out["source"] = build_mt5_source_provenance(gateway)
    return out

def _market_scan_freshness_summary(
    rows: List[Dict[str, Any]],
    *,
    include_stale_symbols: bool = True,
) -> Dict[str, Any]:
    if not rows:
        return {}
    def _row_stale(row: Dict[str, Any]) -> bool:
        return bool(row.get("bar_stale")) or bool(
            row.get("quote_stale", row.get("data_stale"))
        )

    stale_count = sum(1 for row in rows if _row_stale(row))
    stale_bar_count = sum(1 for row in rows if bool(row.get("bar_stale")))
    unsafe_quote_count = sum(
        1
        for row in rows
        if bool(row.get("quote_stale", row.get("data_stale")))
        or row.get(
            "quote_usable_for_live_trading",
            row.get("usable_for_live_trading"),
        ) is False
    )
    row_count = len(rows)
    if stale_count == row_count:
        freshness = "stale"
    elif stale_count:
        freshness = f"mixed, {stale_count}/{row_count} stale"
    else:
        freshness = "fresh"

    out: Dict[str, Any] = {
        "freshness": freshness,
        "stale_rows": int(stale_count),
        "freshness_basis": "conservative_quote_or_bar",
        "stale_bar_rows": int(stale_bar_count),
        "unsafe_quote_rows": int(unsafe_quote_count),
    }
    if stale_count and include_stale_symbols:
        out["stale_symbols"] = [
            str(row.get("symbol"))
            for row in rows
            if _row_stale(row) and str(row.get("symbol") or "").strip()
        ]
    row_times = [
        str(row.get("time") or "").strip()
        for row in rows
        if str(row.get("time") or "").strip()
    ]
    bar_rows = [
        row
        for row in rows
        if str(row.get("data_source") or "").strip().endswith("_bars")
        and str(row.get("time") or "").strip()
    ]
    if bar_rows:
        symbols_by_time: Dict[str, List[str]] = {}
        for row in bar_rows:
            timestamp = str(row.get("time") or "").strip()
            symbol = str(row.get("symbol") or "").strip()
            symbols_by_time.setdefault(timestamp, [])
            if symbol and symbol not in symbols_by_time[timestamp]:
                symbols_by_time[timestamp].append(symbol)
        bar_times = sorted(symbols_by_time)
        aligned = len(bar_times) == 1
        out["bar_time_alignment"] = {
            "status": "aligned" if aligned else "mixed",
            "comparable": aligned,
            "distinct_timestamps": len(bar_times),
            "basis": "latest_completed_bar_open_per_symbol",
        }
        out["bar_rank_comparable"] = aligned
        out["price_change_comparable"] = aligned
        if aligned:
            out["data_as_of"] = bar_times[0]
            out["bar_as_of"] = bar_times[0]
            out["data_as_of_basis"] = "shared_latest_completed_bar_open"
        else:
            time_range = {
                "oldest": bar_times[0],
                "newest": bar_times[-1],
            }
            out["data_as_of_range"] = time_range
            out["bar_as_of_range"] = dict(time_range)
            out["data_as_of_basis"] = "per_row_latest_completed_bar_open"
            out["bar_time_alignment"]["groups"] = [
                {
                    "time": timestamp,
                    "symbols": symbols_by_time[timestamp],
                }
                for timestamp in bar_times
            ]
            out["comparison_warning"] = (
                "Completed-bar timestamps differ across returned symbols; one-bar "
                "price-change and volume ranks are not clock-aligned. Use each row's "
                "time before comparing ranks."
            )
    elif row_times:
        out["data_as_of"] = max(row_times)
        out["data_as_of_basis"] = "latest_source_timestamp"
        if min(row_times) != max(row_times):
            out["data_as_of_range"] = {
                "oldest": min(row_times),
                "newest": max(row_times),
            }
    quote_times = [
        str(row.get("quote_as_of") or "").strip()
        for row in rows
        if str(row.get("quote_as_of") or "").strip()
    ]
    if quote_times:
        quote_times = sorted(quote_times)
        aligned_quotes = len(set(quote_times)) == 1
        out["quote_as_of"] = quote_times[-1]
        out["quote_as_of_range"] = {
            "oldest": quote_times[0],
            "newest": quote_times[-1],
        }
        out["quote_time_alignment"] = {
            "status": "aligned" if aligned_quotes else "mixed",
            "comparable": aligned_quotes,
            "atomic": len(rows) <= 1,
            "sampling": "sequential_per_symbol",
            "distinct_timestamps": len(set(quote_times)),
        }
        out["quote_rank_comparable"] = aligned_quotes

    now_epoch = time.time()
    closed_count = sum(
        1
        for row in rows
        if closed_session_context(row.get("symbol"), now_epoch=now_epoch)
    )
    if closed_count == row_count:
        out["session_status"] = "closed_weekend"
        if stale_count == 0:
            out["freshness"] = "closed_weekend_snapshot"
    elif closed_count:
        out["session_status"] = f"mixed, {closed_count}/{row_count} closed_weekend"
        if stale_count == 0:
            out["freshness"] = f"mixed, {closed_count}/{row_count} closed_weekend_snapshot"
    return out

def _namespace_market_scan_quote_freshness(row: Dict[str, Any]) -> None:
    """Keep quote freshness distinct from the bar that supplies scan prices."""
    field_map = {
        "tick_time": "quote_time",
        "data_age_seconds": "quote_age_seconds",
        "data_age_anchor": "quote_age_anchor",
        "data_age_metric": "quote_age_metric",
        "stale_after_seconds": "quote_stale_after_seconds",
        "data_stale": "quote_stale",
        "freshness_reason": "quote_freshness_reason",
        "timestamp_ahead_of_wall_clock": "quote_timestamp_ahead_of_wall_clock",
        "timestamp_in_future": "quote_timestamp_in_future",
        "timestamp_skew_seconds": "quote_timestamp_skew_seconds",
        "timestamp_skew_tolerance_seconds": "quote_timestamp_skew_tolerance_seconds",
        "timestamp_warning": "quote_timestamp_warning",
        "warning": "quote_warning",
        "freshness": "quote_freshness",
        "usable_for_live_trading": "quote_usable_for_live_trading",
        "usable_for_live_trading_basis": "quote_usable_for_live_trading_basis",
    }
    for source, target in field_map.items():
        if source in row:
            row[target] = row.pop(source)
    row["price_as_of"] = row.get("time")
    row["price_freshness"] = row.get("bar_freshness")

def _attach_market_scan_live_change(row: Dict[str, Any]) -> None:
    previous_close = _market_scan_float(row.get("previous_close"))
    live_mid = _market_scan_float(row.get("mid"))
    if previous_close is None or previous_close == 0.0 or live_mid is None:
        return
    row["live_price_change_pct"] = _market_scan_round(
        ((live_mid - previous_close) / previous_close) * 100.0,
        digits=6,
    )
    row["live_price_change_basis"] = (
        "previous_completed_close_to_live_quote_mid"
    )
    bar_change = _market_scan_float(row.get("price_change_pct"))
    live_change = _market_scan_float(row.get("live_price_change_pct"))
    if (
        bar_change is not None
        and live_change is not None
        and bar_change != 0.0
        and live_change != 0.0
        and (bar_change > 0) != (live_change > 0)
    ):
        bar_dir = "up" if bar_change > 0 else "down"
        live_dir = "up" if live_change > 0 else "down"
        row["direction_divergence"] = f"bar_{bar_dir}_live_{live_dir}"

def _market_scan_quote_exclusion_reason(row: Dict[str, Any]) -> str:
    if isinstance(row.get("quote_source_conflict"), dict):
        return "quote_source_conflict"
    freshness_reason = str(row.get("quote_freshness_reason") or "").strip().lower()
    if freshness_reason and freshness_reason != "live_quote":
        return freshness_reason
    spread_quality = str(row.get("spread_quality") or "").strip().lower()
    if spread_quality and spread_quality != "two_sided":
        return f"quote_{spread_quality}"
    if row.get("quote_stale") is True:
        return "stale_quote"
    if freshness_reason:
        return freshness_reason
    return "quote_not_usable_for_live_trading"

_TOP_MARKETS_COMPACT_BASE_HEADERS = [
    "symbol",
    "group",
    "asset_class",
    "timeframe",
    "data_source",
    "time",
    "data_stale",
    "freshness",
    "spread_valid",
    "spread_quality",
    "usable_for_live_trading",
]

_TOP_MARKETS_COMPACT_SPREAD_HEADERS = [
    "quote_as_of",
    "bid",
    "ask",
    "mid",
    "spread_pct",
    "spread_points",
    "spread_pips",
]

_TOP_MARKETS_COMPACT_BAR_HEADERS = [
    "bar_stale",
    "bar_freshness",
    "quote_as_of",
    "bid",
    "ask",
    "mid",
    "bar_close",
    "tick_volume",
    "price_change_pct",
    "live_price_change_pct",
    "direction_divergence",
]

_TOP_MARKETS_COMPACT_HEADERS = [
    *_TOP_MARKETS_COMPACT_BASE_HEADERS,
    "bar_stale",
    "bar_freshness",
    "quote_as_of",
    "bid",
    "ask",
    "mid",
    "bar_close",
    "spread_pct",
    "spread_points",
    "tick_volume",
    "price_change_pct",
    "live_price_change_pct",
    "direction_divergence",
]

_TOP_MARKETS_FULL_BASE_HEADERS = [
    "symbol",
    "group",
    "asset_class",
    "description",
    "timeframe",
    "data_source",
    "time",
    "data_age_seconds",
    "data_age_anchor",
    "data_age_metric",
    "stale_after_seconds",
    "data_stale",
    "freshness_reason",
    "timestamp_ahead_of_wall_clock",
    "timestamp_in_future",
    "timestamp_skew_seconds",
    "timestamp_skew_tolerance_seconds",
    "timestamp_warning",
    "freshness",
    "warning",
    "spread_valid",
    "spread_quality",
    "usable_for_live_trading",
]

_TOP_MARKETS_COMPACT_VOLUME_HEADERS = [
    "symbol",
    "group",
    "asset_class",
    "timeframe",
    "data_source",
    "time",
    "bar_stale",
    "bar_freshness",
    "bar_close",
    "tick_volume",
    "price_change_pct",
]

_TOP_MARKETS_FULL_SPREAD_HEADERS = [
    "quote_as_of",
    "bid",
    "ask",
    "mid",
    "spread",
    "spread_points",
    "spread_pct",
    "spread_cost_per_lot",
    "spread_cost_currency",
    "pricing_basis",
]

_TOP_MARKETS_FULL_BAR_HEADERS = [
    "bar_age_seconds",
    "bar_freshness_anchor",
    "bar_freshness_metric",
    "bar_stale_after_seconds",
    "bar_age_hours",
    "bar_stale",
    "bar_market_status",
    "bar_market_status_reason",
    "bar_freshness_policy_relaxed",
    "bar_freshness",
    "bar_stale_warning",
    "previous_close",
    "open",
    "close",
    "quote_as_of",
    "bid",
    "ask",
    "mid",
    "tick_volume",
    "real_volume",
    "price_change_pct",
    "live_price_change_pct",
    "direction_divergence",
    "live_price_change_basis",
]

_TOP_MARKETS_FULL_HEADERS = [
    *_TOP_MARKETS_FULL_BASE_HEADERS,
    *_TOP_MARKETS_FULL_SPREAD_HEADERS,
    *_TOP_MARKETS_FULL_BAR_HEADERS,
]

def _top_markets_headers(metric: str, *, detail_mode: str) -> List[str]:
    if metric == "spread":
        if detail_mode == "compact":
            return [
                *_TOP_MARKETS_COMPACT_BASE_HEADERS,
                *_TOP_MARKETS_COMPACT_SPREAD_HEADERS,
            ]
        return [
            *_TOP_MARKETS_FULL_BASE_HEADERS,
            *_TOP_MARKETS_FULL_SPREAD_HEADERS,
        ]
    if metric == "volume" and detail_mode == "compact":
        return list(_TOP_MARKETS_COMPACT_VOLUME_HEADERS)
    if detail_mode == "compact":
        return [
            *_TOP_MARKETS_COMPACT_BASE_HEADERS,
            *_TOP_MARKETS_COMPACT_BAR_HEADERS,
        ]
    return [
        *_TOP_MARKETS_FULL_BASE_HEADERS,
        *_TOP_MARKETS_FULL_BAR_HEADERS,
    ]

def _top_markets_all_headers(*, detail_mode: str) -> List[str]:
    compact_headers = [
        "rank_category",
        "rank",
        *_TOP_MARKETS_COMPACT_HEADERS,
    ]
    if detail_mode == "compact":
        return compact_headers
    return [
        "rank_category",
        "rank",
        *_TOP_MARKETS_FULL_HEADERS,
    ]

def _ranked_top_market_rows(
    ranking: str,
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "rank_category": ranking,
            "rank": rank,
            **row,
        }
        for rank, row in enumerate(rows, start=1)
    ]

def _top_market_data_source(metric: str, timeframe: str) -> str:
    return "live_tick" if metric == "spread" else f"{timeframe}_bars"

def _top_market_data_time_key(metric: str) -> str:
    return "tick_time" if metric == "spread" else "time"

def _top_market_rows_with_data_context(
    metric: str,
    rows: List[Dict[str, Any]],
    *,
    timeframe: str,
) -> List[Dict[str, Any]]:
    data_source = _top_market_data_source(metric, timeframe)
    data_time_key = _top_market_data_time_key(metric)
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        mapped = dict(row)
        mapped["data_source"] = data_source
        mapped["time"] = row.get(data_time_key)
        normalized.append(mapped)
    return normalized

def _parse_market_scan_symbols(symbols: Optional[str]) -> List[str]:
    text = str(symbols or "").replace(";", ",").replace("\n", ",")
    parsed: List[str] = []
    seen: set[str] = set()
    for chunk in text.split(","):
        name = chunk.strip()
        if not name:
            continue
        upper = name.upper()
        if upper in seen:
            continue
        seen.add(upper)
        parsed.append(name)
    return parsed

def _market_scan_group_matches_query(group_path: str, requested: str) -> bool:
    group_normalized = _normalize_group_path_query(group_path).casefold()
    requested_normalized = _normalize_group_path_query(requested).casefold()
    group_compact = re.sub(r"[^a-z0-9]+", "", group_normalized)
    requested_compact = re.sub(r"[^a-z0-9]+", "", requested_normalized)
    if requested_normalized == group_normalized or requested_compact == group_compact:
        return True

    def _tokens(value: str) -> set[str]:
        result: set[str] = set()
        for token in re.split(r"[^a-z0-9]+", value):
            if not token:
                continue
            if token.endswith("ies") and len(token) > 3:
                token = token[:-3] + "y"
            elif token.endswith("s") and len(token) > 1:
                token = token[:-1]
            result.add(token)
        return result

    group_tokens = _tokens(group_normalized)
    requested_tokens = _tokens(requested_normalized)
    generic_tokens = {"stock", "cfd", "market", "instrument", "symbol"}
    informative_tokens = requested_tokens - generic_tokens
    if not informative_tokens:
        return False
    return informative_tokens.issubset(group_tokens)

def _resolve_market_scan_group_path(
    all_symbols: List[Any],
    group: str,
) -> tuple[List[str], Optional[str]]:
    requested = str(group or "").strip()
    if not requested:
        return [], "group must not be empty."

    groups: Dict[str, str] = {}
    for symbol in all_symbols:
        group_path = str(_extract_group_path_util(symbol) or "").strip()
        if not group_path:
            continue
        groups.setdefault(_normalize_group_path_query(group_path).lower(), group_path)

    requested_lower = _normalize_group_path_query(requested).lower()
    exact = groups.get(requested_lower)
    if exact is not None:
        return [exact], None

    partial_matches = sorted(
        (
            value for value in groups.values()
            if _market_scan_group_matches_query(value, requested_lower)
        ),
        key=_case_insensitive_sort_key,
    )
    if len(partial_matches) == 1:
        return [partial_matches[0]], None
    if partial_matches:
        return partial_matches, None
    available = sorted(groups.values(), key=_case_insensitive_sort_key)
    if available:
        preview = ", ".join(available[:5])
        suffix = ", ..." if len(available) > 5 else ""
        return [], f"No symbol group matched '{requested}'. Available groups: {preview}{suffix}"
    return [], f"No symbol group matched '{requested}'."


def _scan_symbol_suggestions(
    requested: str,
    tradable_symbols: List[Any],
    *,
    limit: int = 5,
) -> List[Any]:
    query_upper = str(requested or "").strip().upper()
    query_compact = _alnum_upper(requested)
    if not query_upper and not query_compact:
        return []
    ranked: List[tuple[tuple[int, str], Any]] = []
    seen: set[str] = set()
    for symbol in tradable_symbols:
        name = str(getattr(symbol, "name", "") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        name_upper = name.upper()
        name_compact = _alnum_upper(name)
        if name_upper == query_upper or (
            query_compact and name_compact == query_compact
        ):
            score = 0
        elif name_upper.startswith(query_upper):
            score = 1
        elif query_compact and name_compact.startswith(query_compact):
            score = 2
        elif (
            query_compact
            and name_compact
            and query_compact.startswith(name_compact)
            and len(name_compact) >= 5
        ):
            score = 2
        elif query_upper in name_upper:
            score = 3
        else:
            continue
        ranked.append(((score, name_upper), symbol))
    ranked.sort(key=lambda item: item[0])
    return [
        _symbol_suggestion_from_info(symbol)
        for _key, symbol in ranked[: max(1, int(limit))]
    ]


def _select_market_scan_symbols(
    all_symbols: List[Any],
    *,
    symbols: Optional[str] = None,
    group: Optional[str],
    universe: str,
) -> tuple[List[Any], Dict[str, Any], Optional[str]]:
    requested_names = _parse_market_scan_symbols(symbols)
    selection_meta: Dict[str, Any] = {}
    if requested_names:
        selection_meta["symbols_input"] = list(requested_names)
    if requested_names and group:
        return [], selection_meta, "Provide either symbols or group, not both."

    tradable_symbols = [symbol for symbol in all_symbols if _market_scan_is_tradable(symbol)]

    if requested_names:
        by_upper: Dict[str, Any] = {}
        by_compact: Dict[str, List[Any]] = {}
        for tradable_symbol in tradable_symbols:
            name = str(getattr(tradable_symbol, "name", "") or "").strip()
            if not name:
                continue
            by_upper.setdefault(name.upper(), tradable_symbol)
            compact_name = _alnum_upper(name)
            if compact_name:
                by_compact.setdefault(compact_name, []).append(tradable_symbol)
        selected: List[Any] = []
        missing: List[str] = []
        for name in requested_names:
            symbol_obj = by_upper.get(name.upper())
            if symbol_obj is None:
                compact_matches = by_compact.get(_alnum_upper(name), [])
                if len(compact_matches) == 1:
                    symbol_obj = compact_matches[0]
            if symbol_obj is None:
                missing.append(name)
                continue
            selected.append(symbol_obj)
        if not selected:
            suggestions = []
            for requested in missing:
                suggestions.extend(
                    _scan_symbol_suggestions(requested, tradable_symbols)
                )
            if suggestions:
                selection_meta["did_you_mean"] = suggestions[:5]
            return [], selection_meta, "None of the requested symbols matched the MT5 symbol list."
        if missing:
            suggestions = []
            for requested in missing:
                suggestions.extend(
                    _scan_symbol_suggestions(requested, tradable_symbols)
                )
            if suggestions:
                selection_meta["did_you_mean"] = suggestions[:5]
        return selected, {
            **selection_meta,
            "scope": "symbols",
            "requested_symbols": requested_names,
            "missing_symbols": missing,
        }, None

    if group:
        resolved_groups, group_error = _resolve_market_scan_group_path(tradable_symbols, group)
        if group_error or not resolved_groups:
            return [], {}, group_error
        resolved_group_set = {
            _normalize_group_path_query(str(group_path).strip()).lower()
            for group_path in resolved_groups
        }
        selected = sorted(
            [
                symbol for symbol in tradable_symbols
                if (
                    _normalize_group_path_query(
                        str(_extract_group_path_util(symbol) or "").strip()
                    ).lower()
                    in resolved_group_set
                )
                and (universe == "all" or bool(getattr(symbol, "visible", False)))
            ],
            key=lambda symbol: _case_insensitive_sort_key(getattr(symbol, "name", "")),
        )
        return selected, {
            **selection_meta,
            "scope": "group",
            "group": resolved_groups[0] if len(resolved_groups) == 1 else str(group).strip(),
            "groups": resolved_groups,
        }, None

    selected = sorted(
        [
            symbol for symbol in tradable_symbols
            if universe == "all" or bool(getattr(symbol, "visible", False))
        ],
        key=lambda symbol: _case_insensitive_sort_key(getattr(symbol, "name", "")),
    )
    return selected, {**selection_meta, "scope": "universe"}, None

def _market_scan_compute_rsi(closes: List[float], length: int) -> Optional[float]:
    if length <= 0 or len(closes) < (length + 1):
        return None

    gains: List[float] = []
    losses: List[float] = []
    for prev_close, close in zip(closes[:-1], closes[1:], strict=False):
        delta = float(close - prev_close)
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains[:length]) / float(length)
    avg_loss = sum(losses[:length]) / float(length)
    for gain, loss in zip(gains[length:], losses[length:], strict=False):
        avg_gain = ((avg_gain * float(length - 1)) + float(gain)) / float(length)
        avg_loss = ((avg_loss * float(length - 1)) + float(loss)) / float(length)

    if avg_loss <= 0.0:
        if avg_gain <= 0.0:
            return 50.0
        return 100.0

    relative_strength = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))

def _build_market_scan_signal_row(
    symbol: Any,
    *,
    timeframe: str,
    mt5_timeframe: Any,
    lookback: int,
    rsi_length: int,
    sma_period: int,
    include_rsi: bool,
    include_sma: bool,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    rsi_warmup_bars = (
        max(50, int(rsi_length) * 25) if include_rsi else 0
    )
    fetch_count = max(int(lookback), rsi_warmup_bars)
    rates = _market_scan_completed_rates(
        symbol.name,
        timeframe=timeframe,
        mt5_timeframe=mt5_timeframe,
        count=fetch_count,
    )
    if rates is None or len(rates) < 2:
        return None, f"At least two completed {timeframe} bars are required."

    row, error = _project_market_scan_completed_bars(
        symbol,
        timeframe,
        rates[-1],
        rates[-2],
    )
    if row is None:
        return None, error

    close_values: List[float] = []
    for bar in rates:
        close_value = _market_scan_float(bar["close"])
        if close_value is not None:
            close_values.append(close_value)

    digits = max(0, int(getattr(symbol, "digits", 0) or 0))
    close_price = _market_scan_float(rates[-1]["close"])
    sma_value = None
    if include_sma and len(close_values) >= max(1, int(sma_period)):
        sma_window = close_values[-int(sma_period):]
        sma_value = float(sum(sma_window) / len(sma_window))
    rsi_values = (
        close_values[-rsi_warmup_bars:]
        if include_rsi and rsi_warmup_bars
        else close_values
    )
    rsi_value = (
        _market_scan_compute_rsi(rsi_values, int(rsi_length))
        if include_rsi
        else None
    )
    sma_distance_pct = None
    if sma_value is not None and sma_value != 0 and close_price is not None:
        sma_distance_pct = ((close_price - sma_value) / sma_value) * 100.0

    row["rsi_warmup_bars"] = rsi_warmup_bars if include_rsi else None
    if include_rsi:
        row["rsi"] = _market_scan_round(rsi_value, digits=4)
    if include_sma:
        row["sma_value"] = _market_scan_round(sma_value, digits=digits)
        row["sma_distance_pct"] = _market_scan_round(sma_distance_pct, digits=6)
    return row, None

def _market_scan_missing_required_metric(
    row: Dict[str, Any],
    *,
    rank_by: str,
    rsi_above: Optional[float],
    rsi_below: Optional[float],
    price_vs_sma: Optional[str],
    max_spread_pct: Optional[float],
    min_tick_volume: Optional[int],
    min_price_change_pct: Optional[float],
    max_price_change_pct: Optional[float],
    min_gap_pct: Optional[float] = None,
    max_gap_pct: Optional[float] = None,
    rsi_length: int,
    sma_period: int,
) -> Optional[str]:
    requirements: List[tuple[str, str]] = []
    if rank_by in {"abs_price_change_pct", "price_change_pct"}:
        requirements.append(("price_change_pct", "price-change data is unavailable."))
    elif rank_by in {"abs_live_price_change_pct", "live_price_change_pct"}:
        requirements.append(
            ("live_price_change_pct", "Live price-change data is unavailable.")
        )
    elif rank_by == "gap_pct":
        requirements.append(("gap_pct", "Gap data is unavailable."))
    elif rank_by == "tick_volume":
        requirements.append(("tick_volume", "Tick-volume data is unavailable."))
    elif rank_by == "rsi":
        requirements.append(("rsi", f"Not enough history to compute RSI({int(rsi_length)})."))
    elif rank_by == "spread_pct":
        requirements.append(("spread_pct", "Spread data is unavailable."))

    if min_price_change_pct is not None or max_price_change_pct is not None:
        requirements.append(("price_change_pct", "price-change data is unavailable."))
    if min_gap_pct is not None or max_gap_pct is not None:
        requirements.append(("gap_pct", "Gap data is unavailable."))
    if max_spread_pct is not None:
        requirements.append(("spread_pct", "Spread data is unavailable."))
    if min_tick_volume is not None:
        requirements.append(("tick_volume", "Tick-volume data is unavailable."))
    if rsi_above is not None or rsi_below is not None:
        requirements.append(("rsi", f"Not enough history to compute RSI({int(rsi_length)})."))
    if price_vs_sma is not None:
        requirements.append(("sma_value", f"Not enough history to compute SMA({int(sma_period)})."))

    for key, message in requirements:
        if row.get(key) is None:
            return message
    return None

def _market_scan_row_matches_filters(
    row: Dict[str, Any],
    *,
    min_price_change_pct: Optional[float],
    max_price_change_pct: Optional[float],
    max_spread_pct: Optional[float],
    min_tick_volume: Optional[int],
    rsi_below: Optional[float],
    rsi_above: Optional[float],
    price_vs_sma: Optional[str],
    min_gap_pct: Optional[float] = None,
    max_gap_pct: Optional[float] = None,
) -> bool:
    price_change_pct = _market_scan_float(row.get("price_change_pct"))
    gap_pct = _market_scan_float(row.get("gap_pct"))
    spread_pct = _market_scan_float(row.get("spread_pct"))
    tick_volume = _market_scan_bar_int(row.get("tick_volume"))
    rsi_value = _market_scan_float(row.get("rsi"))
    close_price = _market_scan_float(row.get("close"))
    sma_value = _market_scan_float(row.get("sma_value"))

    if min_price_change_pct is not None and (price_change_pct is None or price_change_pct < float(min_price_change_pct)):
        return False
    if max_price_change_pct is not None and (price_change_pct is None or price_change_pct > float(max_price_change_pct)):
        return False
    if min_gap_pct is not None and (gap_pct is None or gap_pct < float(min_gap_pct)):
        return False
    if max_gap_pct is not None and (gap_pct is None or gap_pct > float(max_gap_pct)):
        return False
    if max_spread_pct is not None and (
        row.get("spread_valid") is not True
        or spread_pct is None
        or spread_pct > float(max_spread_pct)
    ):
        return False
    if min_tick_volume is not None and (tick_volume is None or tick_volume < int(min_tick_volume)):
        return False
    if rsi_below is not None and (rsi_value is None or rsi_value > float(rsi_below)):
        return False
    if rsi_above is not None and (rsi_value is None or rsi_value < float(rsi_above)):
        return False
    if price_vs_sma == "above" and (close_price is None or sma_value is None or close_price <= sma_value):
        return False
    if price_vs_sma == "below" and (close_price is None or sma_value is None or close_price >= sma_value):
        return False
    return True

def _larger_abs_metric_cut_for_freshness(
    ranked_rows: List[Dict[str, Any]],
    visible_rows: List[Dict[str, Any]],
    metric_key: str,
) -> bool:
    """True when freshness-first sort dropped a larger-magnitude mover."""
    if len(visible_rows) >= len(ranked_rows):
        return False
    visible_symbols = {
        str(row.get("symbol") or "")
        for row in visible_rows
        if isinstance(row, dict)
    }
    visible_mags = [
        abs(value)
        for row in visible_rows
        if isinstance(row, dict)
        for value in (_market_scan_float(row.get(metric_key)),)
        if value is not None
    ]
    if not visible_mags:
        return False
    best_visible = max(visible_mags)
    for row in ranked_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("symbol") or "") in visible_symbols:
            continue
        omitted = _market_scan_float(row.get(metric_key))
        if omitted is not None and abs(omitted) > best_visible:
            return True
    return False


def _market_scan_ranking_policy(rank_by: str, rank_order: str) -> List[str]:
    priorities = []
    if rank_by == "spread_pct":
        priorities.append("valid_spreads_first")
    if rank_by in {"live_price_change_pct", "abs_live_price_change_pct"}:
        priorities.append("usable_quotes_first")
    return [*priorities, "fresh_bars_first", "non_missing_values_first", f"{rank_by}_{rank_order}", "symbol_asc"]


def _market_scan_sort_rows(
    rows: List[Dict[str, Any]],
    *,
    rank_by: str,
    rank_order: str,
    rsi_above: Optional[float],
    rsi_below: Optional[float],
) -> None:
    order = _market_scan_effective_rank_order(
        rank_by,
        rank_order=rank_order,
        rsi_above=rsi_above,
        rsi_below=rsi_below,
    )

    if rank_by == "abs_price_change_pct":
        rows.sort(
            key=lambda row: (
                bool(row.get("bar_stale")),
                row.get("price_change_pct") is None,
                (
                    abs(float(row.get("price_change_pct") or 0.0))
                    if order == "asc"
                    else -abs(float(row.get("price_change_pct") or 0.0))
                ),
                row.get("symbol") or "",
            )
        )
        return
    if rank_by == "abs_live_price_change_pct":
        rows.sort(
            key=lambda row: (
                row.get("quote_usable_for_live_trading") is not True,
                bool(row.get("bar_stale")),
                row.get("live_price_change_pct") is None,
                (
                    abs(float(row.get("live_price_change_pct") or 0.0))
                    if order == "asc"
                    else -abs(float(row.get("live_price_change_pct") or 0.0))
                ),
                row.get("symbol") or "",
            )
        )
        return

    missing_value = float("inf") if order == "asc" else 0.0

    rows.sort(
        key=lambda row: (
            rank_by == "spread_pct" and row.get("spread_valid") is not True,
            (
                rank_by == "live_price_change_pct"
                and row.get("quote_usable_for_live_trading") is not True
            ),
            bool(row.get("bar_stale")),
            row.get(rank_by) is None,
            (
                float(row.get(rank_by) if row.get(rank_by) is not None else missing_value)
                if order == "asc"
                else -(float(row.get(rank_by) or 0.0))
            ),
            row.get("symbol") or "",
        )
    )

_RANK_BY_ALIASES = {
    "abs_price_change": "abs_price_change_pct",
    "abs_live_price_change": "abs_live_price_change_pct",
    "live_price_change": "live_price_change_pct",
    "price_change": "price_change_pct",
    "spread": "spread_pct",
}

_SYMBOLS_TOP_MARKETS_INTERNAL_RANK_BY = {
    "abs_price_change_pct": "abs_price_change",
    "price_change_pct": "price_change",
    "live_price_change_pct": "live_price_change",
    "abs_live_price_change_pct": "abs_live_price_change",
    "tick_volume": "volume",
    "spread_pct": "spread",
}

_MARKET_SCAN_RANK_BY_CHOICES = (
    "abs_price_change_pct",
    "abs_price_change",
    "abs_live_price_change_pct",
    "abs_live_price_change",
    "live_price_change_pct",
    "live_price_change",
    "price_change_pct",
    "price_change",
    "gap_pct",
    "tick_volume",
    "rsi",
    "spread_pct",
    "spread",
)

def _normalize_market_scan_rank_by(value: Any) -> tuple[str, Optional[str]]:
    raw_value = str(value or "abs_price_change_pct").strip().lower()
    return _RANK_BY_ALIASES.get(raw_value, raw_value), raw_value

def _normalize_market_scan_rank_order(value: Any) -> tuple[str, Optional[str]]:
    raw_value = str(value or "auto").strip().lower()
    aliases = {"ascending": "asc", "descending": "desc"}
    return aliases.get(raw_value, raw_value), raw_value

def _market_scan_effective_rank_order(
    rank_by: str,
    *,
    rank_order: str,
    rsi_above: Optional[float] = None,
    rsi_below: Optional[float] = None,
) -> str:
    order = str(rank_order or "auto").strip().lower()
    if order != "auto":
        return order
    if rank_by == "spread_pct":
        return "asc"
    if rank_by == "rsi" and rsi_below is not None and rsi_above is None:
        return "asc"
    return "desc"

def _market_scan_ranking_label(
    rank_by: str,
    *,
    rank_order: str,
    rsi_above: Optional[float] = None,
    rsi_below: Optional[float] = None,
) -> str:
    order = _market_scan_effective_rank_order(
        rank_by,
        rank_order=rank_order,
        rsi_above=rsi_above,
        rsi_below=rsi_below,
    )
    if rank_by == "abs_price_change_pct":
        return (
            "smallest_abs_price_change_pct"
            if order == "asc"
            else "largest_abs_price_change_pct"
        )
    if rank_by == "price_change_pct":
        return "lowest_price_change_pct" if order == "asc" else "highest_price_change_pct"
    if rank_by == "abs_live_price_change_pct":
        return (
            "smallest_abs_live_price_change_pct"
            if order == "asc"
            else "largest_abs_live_price_change_pct"
        )
    if rank_by == "live_price_change_pct":
        return (
            "lowest_live_price_change_pct"
            if order == "asc"
            else "highest_live_price_change_pct"
        )
    if rank_by == "gap_pct":
        return "lowest_gap_pct" if order == "asc" else "highest_gap_pct"
    if rank_by == "tick_volume":
        return "lowest_tick_volume" if order == "asc" else "highest_tick_volume"
    if rank_by == "spread_pct":
        return "lowest_spread_pct" if order == "asc" else "highest_spread_pct"
    if rank_by == "rsi":
        return "lowest_rsi" if order == "asc" else "highest_rsi"
    return str(rank_by)

@mcp.tool()
def symbols_top_markets(  # noqa: C901
    rank_by: Literal[
        "all",
        "spread",
        "price_change",
        "spread_pct",
        "tick_volume",
        "price_change_pct",
        "abs_price_change",
        "abs_price_change_pct",
        "live_price_change",
        "live_price_change_pct",
        "abs_live_price_change",
        "abs_live_price_change_pct",
    ] = "abs_price_change_pct",  # type: ignore
    limit: Annotated[int, Field(ge=1)] = 10,
    universe: Literal["visible", "all"] = "visible",  # type: ignore
    timeframe: TimeframeLiteral = "H1",
    group: Optional[str] = None,
    category: Optional[str] = None,
    candidate_offset: Annotated[int, Field(ge=0)] = 0,
    candidate_limit: Annotated[
        Optional[int], Field(ge=1, le=_TOP_MARKETS_MAX_CANDIDATES)
    ] = None,
    scan_budget_seconds: Annotated[
        float,
        Field(
            ge=0,
            description=(
                "Maximum wall-clock seconds for candidate sampling. Use 0 to "
                "wait for an exact full-universe ranking."
            ),
        ),
    ] = _TOP_MARKETS_DEFAULT_SCAN_BUDGET_SECONDS,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Quick MT5 market overview ranked by spread, tick volume, or price change.

    Defaults to visible tradable symbols for responsiveness. Set `group` or
    `category` (forex, crypto, indices, commodities, stocks, bonds, etfs) to keep
    rankings inside a comparable asset universe. Set `universe="all"` to include
    hidden tradable symbols too; that mode is slower because MT5 may need to
    activate quotes for instruments that are not already visible. Defaults to a
    single absolute-price-change leaderboard; set `rank_by="price_change"` for
    gainers only or `rank_by="all"` for spread, volume, signed price-change,
    and absolute price-change leaderboards. Row `time` follows `data_source`:
    `live_tick` (quote time) for spread rankings, otherwise the selected
    `timeframe`'s completed-bar time. Volume and price-change rankings use
    the most recent completed bar on `timeframe`. Uses compact leaderboard rows by default. Set
    `detail="full"` for the expanded row shape and collection metadata. Use
    Large universes are scanned globally within `scan_budget_seconds`; a result
    that exhausts the budget is explicitly partial. Set the budget to 0 to wait
    for the exact global top-N in one invocation. `candidate_limit` and
    `candidate_offset` remain available for deterministic recovery partitions.
    Resume at `candidate_page.next_offset` after a partial page, and merge each
    page's top-N rows by the selected ranking value. Keep the same candidate
    universe and filters throughout pagination. Use
    `market_scan` instead when you need explicit symbol inputs, RSI/SMA filters,
    or a single flat scanner table. Locked or invalid quotes are marked unsafe
    and rank after valid two-sided quotes in spread leaderboards.
    """

    detail_mode = normalize_output_verbosity_detail(detail, default="compact")

    def _run() -> Dict[str, Any]:  # noqa: C901
        try:
            if limit is not None:
                try:
                    if int(limit) <= 0:
                        return {"error": "limit must be a positive integer when provided."}
                except (TypeError, ValueError):
                    return {"error": "limit must be a positive integer when provided."}
            try:
                scan_budget_value = float(scan_budget_seconds)
            except (TypeError, ValueError):
                return {"error": "scan_budget_seconds must be a non-negative number."}
            if not math.isfinite(scan_budget_value) or scan_budget_value < 0.0:
                return {"error": "scan_budget_seconds must be a finite number >= 0."}
            raw_rank_by_value = str(rank_by or "abs_price_change_pct").strip().lower()
            if raw_rank_by_value == "volume":
                return {
                    "error": (
                        "rank_by='volume' is ambiguous; use rank_by='tick_volume' "
                        "for broker tick counts."
                    )
                }
            rank_by_value = _RANK_BY_ALIASES.get(
                raw_rank_by_value,
                raw_rank_by_value,
            )
            rank_kind = _SYMBOLS_TOP_MARKETS_INTERNAL_RANK_BY.get(
                rank_by_value,
                rank_by_value,
            )
            if rank_kind not in {
                "all",
                "spread",
                "volume",
                "price_change",
                "abs_price_change",
                "live_price_change",
                "abs_live_price_change",
            }:
                return {
                    "error": (
                        "rank_by must be one of: all, spread/spread_pct, "
                        "tick_volume, price_change/price_change_pct, "
                        "abs_price_change/abs_price_change_pct, "
                        "live_price_change/live_price_change_pct, "
                        "abs_live_price_change/abs_live_price_change_pct."
                    )
                }

            universe_value = str(universe or "visible").strip().lower()
            if universe_value not in {"visible", "all"}:
                return {"error": "universe must be 'visible' or 'all'."}
            group_filter = _normalize_group_path_query(group) if group else None
            category_filter = _normalize_symbol_category_filter(category)
            if category and not category_filter:
                return _invalid_symbol_category_error(
                    category,
                    operation="symbols_top_markets",
                )

            timeframe_value = str(timeframe or "H1").strip().upper()
            needs_bar_data = rank_kind in {
                "all",
                "volume",
                "price_change",
                "abs_price_change",
                "live_price_change",
                "abs_live_price_change",
            }
            if needs_bar_data and timeframe_value not in TIMEFRAME_MAP:
                return {"error": invalid_timeframe_error(timeframe_value, TIMEFRAME_MAP)}
            mt5_timeframe = TIMEFRAME_MAP.get(timeframe_value)

            mt5_gateway = create_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            source = build_mt5_source_provenance(mt5_gateway)
            spread_cost_currency = (
                account_currency_from_gateway(mt5_gateway)
                if rank_kind != "volume"
                else None
            )

            with _symbol_visibility_snapshot_guard():
                raw_symbols = mt5_gateway.symbols_get()
            if raw_symbols is None:
                return {"error": f"Failed to get symbols: {mt5_gateway.last_error()}"}
            all_symbols = list(raw_symbols)
            tradable_symbols = [
                symbol
                for symbol in all_symbols
                if _market_scan_is_tradable(symbol)
            ]

            selected_symbols, selection_meta, selection_error = _select_market_scan_symbols(
                all_symbols,
                group=group_filter,
                universe=universe_value,
            )
            if selection_error:
                return {
                    "error": selection_error
                    or f"No symbol group matched '{group_filter}'.",
                }
            filters: Dict[str, Any] = {}
            group_has_no_universe_members = bool(group_filter) and not selected_symbols
            if group_filter:
                resolved_groups = list(selection_meta.get("groups") or [])
                filters["group"] = selection_meta.get("group") or str(group_filter).strip()
                if len(resolved_groups) > 1:
                    filters["groups"] = resolved_groups
            if category_filter:
                selected_symbols = [
                    symbol
                    for symbol in selected_symbols
                    if _symbol_category(symbol) == category_filter
                ]
                filters["category"] = category_filter
            selected_symbols = sorted(
                selected_symbols,
                key=lambda symbol: _case_insensitive_sort_key(getattr(symbol, "name", "")),
            )

            try:
                candidate_offset_value = int(candidate_offset or 0)
            except (TypeError, ValueError):
                return {"error": "candidate_offset must be a non-negative integer."}
            if candidate_offset_value < 0:
                return {"error": "candidate_offset must be >= 0."}
            candidate_limit_value: Optional[int] = None
            if candidate_limit is not None:
                try:
                    candidate_limit_value = int(candidate_limit)
                except (TypeError, ValueError):
                    return {
                        "error": (
                            "candidate_limit must be an integer between 1 and "
                            f"{_TOP_MARKETS_MAX_CANDIDATES}."
                        )
                    }
                if not 1 <= candidate_limit_value <= _TOP_MARKETS_MAX_CANDIDATES:
                    return {
                        "error": (
                            "candidate_limit must be between 1 and "
                            f"{_TOP_MARKETS_MAX_CANDIDATES}."
                        )
                    }

            candidate_total = len(selected_symbols)
            partition_requested = bool(
                candidate_limit_value is not None or candidate_offset_value
            )
            if partition_requested:
                page_size = candidate_limit_value or _TOP_MARKETS_MAX_CANDIDATES
                selected_symbols = selected_symbols[
                    candidate_offset_value : candidate_offset_value + page_size
                ]

            ranking_universe_size = len(selected_symbols)
            limit_value = _normalize_limit(limit) or 10
            started_at = time.perf_counter()
            scan_started_epoch = time.time()

            spread_rows: List[Dict[str, Any]] = []
            volume_rows: List[Dict[str, Any]] = []
            price_change_rows: List[Dict[str, Any]] = []
            metric_issues: Dict[str, List[Dict[str, str]]] = {
                "spread": [],
                "volume": [],
                "price_change": [],
            }
            metric_skips: Dict[str, int] = {
                "spread": 0,
                "volume": 0,
                "price_change": 0,
            }

            def _record_issue(metric_name: str, symbol_name: str, reason: str) -> None:
                metric_skips[metric_name] += 1
                if len(metric_issues[metric_name]) < 10:
                    metric_issues[metric_name].append(
                        {"symbol": symbol_name, "reason": reason}
                    )

            def _collect_for_symbol(symbol: Any) -> None:
                symbol_name = str(getattr(symbol, "name", "") or "")

                spread_row = None
                if rank_kind != "volume":
                    spread_row, spread_error = _build_market_scan_spread_row(
                        symbol,
                        mt5_gateway,
                        spread_cost_currency=spread_cost_currency,
                    )
                    if spread_error and rank_kind in {"all", "spread"}:
                        _record_issue("spread", symbol_name, spread_error)
                    elif spread_row is not None and rank_kind in {"all", "spread"}:
                        spread_rows.append(spread_row)

                if needs_bar_data and mt5_timeframe is not None:
                    bar_row, bar_error = _build_market_scan_bar_row(
                        symbol,
                        timeframe=timeframe_value,
                        mt5_timeframe=mt5_timeframe,
                    )
                    if bar_error:
                        if rank_kind in {"all", "volume"}:
                            _record_issue("volume", symbol_name, bar_error)
                        if rank_kind in {
                            "all",
                            "price_change",
                            "abs_price_change",
                            "live_price_change",
                            "abs_live_price_change",
                        }:
                            _record_issue("price_change", symbol_name, bar_error)
                    elif bar_row is not None:
                        combined_bar_row = dict(spread_row or {})
                        combined_bar_row.update(bar_row)
                        _attach_market_scan_live_change(combined_bar_row)
                        if rank_kind in {"all", "volume"}:
                            volume_rows.append(dict(combined_bar_row))
                        if rank_kind in {
                            "all",
                            "price_change",
                            "abs_price_change",
                            "live_price_change",
                            "abs_live_price_change",
                        }:
                            price_change_rows.append(dict(combined_bar_row))

            scanned_symbol_count = 0
            scan_budget_exhausted = False
            for symbol in selected_symbols:
                if (
                    scanned_symbol_count > 0
                    and scan_budget_value > 0.0
                    and (time.perf_counter() - started_at) >= scan_budget_value
                ):
                    scan_budget_exhausted = True
                    break
                scanned_symbol_count += 1
                symbol_name = str(getattr(symbol, "name", "") or "")
                is_hidden = not bool(getattr(symbol, "visible", False))
                if universe_value == "all" and is_hidden:
                    with _symbol_ready_guard(symbol_name, info_before=symbol) as (err, _):
                        if err:
                            if rank_kind in {"all", "spread"}:
                                _record_issue("spread", symbol_name, err)
                            if rank_kind in {"all", "volume"}:
                                _record_issue("volume", symbol_name, err)
                            if rank_kind in {
                                "all",
                                "price_change",
                                "abs_price_change",
                                "live_price_change",
                                "abs_live_price_change",
                            }:
                                _record_issue("price_change", symbol_name, err)
                            continue
                        _collect_for_symbol(symbol)
                    continue
                _collect_for_symbol(symbol)

            scan_finished_epoch = time.time()
            scan_completed = scanned_symbol_count == ranking_universe_size

            spread_rows.sort(
                key=lambda row: (
                    row.get("spread_valid") is not True,
                    bool(row.get("data_stale")),
                    row.get("spread_pct") is None,
                    row.get("spread_pct") if row.get("spread_pct") is not None else float("inf"),
                    row.get("symbol") or "",
                )
            )
            volume_rows.sort(
                key=lambda row: (
                    bool(row.get("data_stale")) or bool(row.get("bar_stale")),
                    row.get("tick_volume") is None,
                    -(row.get("tick_volume") or 0),
                    row.get("symbol") or "",
                )
            )
            live_rank = rank_kind in {"live_price_change", "abs_live_price_change"}
            if live_rank:
                price_change_rows = [
                    row
                    for row in price_change_rows
                    if row.get("quote_usable_for_live_trading") is not False
                    and row.get("live_price_change_pct") is not None
                ]
            change_key = "live_price_change_pct" if live_rank else "price_change_pct"
            abs_rank = rank_kind in {"abs_price_change", "abs_live_price_change"}
            price_change_rows.sort(
                key=lambda row: (
                    bool(row.get("data_stale")) or bool(row.get("bar_stale")),
                    row.get(change_key) is None,
                    (
                        -abs(float(row.get(change_key) or 0.0))
                        if abs_rank
                        else -(float(row.get(change_key) or 0.0))
                    ),
                    row.get("symbol") or "",
                )
            )
            abs_price_change_rows = [dict(row) for row in price_change_rows]
            abs_price_change_rows.sort(
                key=lambda row: (
                    bool(row.get("data_stale")) or bool(row.get("bar_stale")),
                    row.get("price_change_pct") is None,
                    -abs(float(row.get("price_change_pct") or 0.0)),
                    row.get("symbol") or "",
                )
            )

            evaluated_counts = {
                "spread": len(spread_rows),
                "volume": len(volume_rows),
                "price_change": len(price_change_rows),
                "abs_price_change": len(abs_price_change_rows),
            }
            freshness_cut_larger_mover = _larger_abs_metric_cut_for_freshness(
                abs_price_change_rows,
                abs_price_change_rows[:limit_value],
                "price_change_pct",
            )

            spread_rows = _top_market_rows_with_data_context(
                "spread",
                spread_rows[:limit_value],
                timeframe=timeframe_value,
            )
            volume_rows = _top_market_rows_with_data_context(
                "volume",
                volume_rows[:limit_value],
                timeframe=timeframe_value,
            )
            price_change_rows = _top_market_rows_with_data_context(
                "price_change",
                price_change_rows[:limit_value],
                timeframe=timeframe_value,
            )
            abs_price_change_rows = _top_market_rows_with_data_context(
                "price_change",
                abs_price_change_rows[:limit_value],
                timeframe=timeframe_value,
            )

            def _scope_fields(
                metric_name: str,
                rows: List[Dict[str, Any]],
            ) -> Dict[str, Any]:
                available_count = int(evaluated_counts[metric_name])
                returned_count = int(len(rows))
                fields: Dict[str, Any] = {
                    "requested_limit": int(limit_value),
                    "universe_size": int(ranking_universe_size),
                    "available_count": available_count,
                }
                if detail_mode == "full":
                    fields["returned_count"] = returned_count
                if returned_count < int(limit_value):
                    fields["note"] = (
                        f"Requested {int(limit_value)} rows but only "
                        f"{available_count} symbols provided {metric_name} data "
                        f"in the {universe_value} universe."
                    )
                return fields

            scan_meta = {
                "success": True,
                "source": source,
                "universe": universe_value,
                "broker_symbol_count": len(all_symbols),
                "tradable_symbol_count": len(tradable_symbols),
                "visible_count": sum(
                    1 for symbol in tradable_symbols if bool(getattr(symbol, "visible", False))
                ),
                "ranking_scope": (
                    "candidate_partition"
                    if partition_requested
                    else "global"
                    if scan_completed
                    else "partial_global"
                ),
                "ranking_complete": bool(
                    scan_completed
                    and not freshness_cut_larger_mover
                    and (
                        not partition_requested
                        or (
                            candidate_offset_value == 0
                            and ranking_universe_size == candidate_total
                        )
                    )
                ),
                "candidate_progress": build_pagination_meta(
                    total=ranking_universe_size,
                    returned=scanned_symbol_count,
                    offset=0,
                    limit=max(1, ranking_universe_size),
                ),
                "sampling_window": {
                    "started_at": _format_time_minimal(scan_started_epoch),
                    "ended_at": _format_time_minimal(scan_finished_epoch),
                    "duration_seconds": round(
                        max(0.0, scan_finished_epoch - scan_started_epoch),
                        3,
                    ),
                    "basis": "sequential_per_symbol",
                    "atomic": False,
                    "comparable": scanned_symbol_count <= 1,
                },
            }
            if scan_budget_exhausted:
                scan_meta.update(
                    {
                        "partial": True,
                        "scan_status": "time_budget_exhausted",
                        "warning": (
                            "The scan reached its wall-clock budget before all "
                            "candidates were sampled; returned ranks are partial."
                        ),
                        "remediation": (
                            "Retry with a larger scan_budget_seconds value or set "
                            "scan_budget_seconds=0 to wait for the exact global ranking."
                        ),
                    }
                )
            if any(metric_skips.values()):
                scan_meta["partial_failure"] = True
                scan_meta["ranking_complete"] = False
                scan_meta["evaluation_failures"] = {
                    metric: {"count": count, "examples": metric_issues[metric]}
                    for metric, count in metric_skips.items() if count
                }
                scan_meta["success"] = any(evaluated_counts.values())
                scan_meta["scan_status"] = "evaluation_incomplete"
                scan_meta["remediation"] = "Inspect evaluation_failures and correct missing quote or history data before relying on a complete ranking."
                if not scan_meta["success"]:
                    scan_meta["error_code"] = "market_scan_incomplete"
                    scan_meta["error"] = "No requested leaderboard could be evaluated."
            if universe_value == "visible" and len(tradable_symbols) > len(selected_symbols):
                scan_meta["note"] = (
                    f"Ranked visible Market Watch symbols only ({len(selected_symbols)} of "
                    f"{len(tradable_symbols)} tradable symbols from "
                    f"{len(all_symbols)} broker symbols); pass --universe all "
                    "to scan the full catalog."
                )
            if filters:
                scan_meta["filters"] = filters
            if group_has_no_universe_members:
                scan_meta["status"] = "no_group_members_in_universe"
                scan_meta["remediation"] = (
                    "The exact broker group has no members in the selected universe; "
                    "retry with --universe all to include hidden symbols."
                )
            if partition_requested or scan_budget_exhausted:
                scan_meta["candidate_page"] = build_pagination_meta(
                    total=candidate_total,
                    returned=scanned_symbol_count,
                    offset=candidate_offset_value,
                    limit=(
                        candidate_limit_value
                        or (_TOP_MARKETS_MAX_CANDIDATES if partition_requested else max(1, ranking_universe_size))
                    ),
                )
                next_offset = candidate_offset_value + scanned_symbol_count
                scan_meta["candidate_page"]["next_offset"] = (
                    next_offset if next_offset < candidate_total else None
                )
                scan_meta["candidate_page"]["aggregation_required"] = bool(
                    candidate_offset_value > 0 or scanned_symbol_count < candidate_total
                )
                scan_meta["candidate_page"]["aggregation_note"] = (
                    "Merge top-N rows from every candidate partition by the selected "
                    "ranking value to compute the global leaderboard. Resume at "
                    "next_offset; keep the same candidate universe and filters."
                )
            if detail_mode == "full":
                scan_meta.update(
                    {
                        "rank_by": rank_by_value,
                        "rank_by_input": raw_rank_by_value
                        if raw_rank_by_value != rank_by_value
                        else None,
                        "limit": limit_value,
                        "universe": universe_value,
                        "detail": detail_mode,
                        "timeframe": timeframe_value if needs_bar_data else None,
                        "timeframe_requested": timeframe_value,
                        "timeframe_used": timeframe_value if needs_bar_data else None,
                        "scanned_symbols": scanned_symbol_count,
                        "scan_budget_seconds": scan_budget_value,
                        "query_latency_ms": round(
                            (time.perf_counter() - started_at) * 1000.0,
                            3,
                        ),
                    }
                )

            if rank_kind == "spread":
                spread_headers = _top_markets_headers("spread", detail_mode=detail_mode)
                out = _market_scan_table(
                    spread_headers,
                    spread_rows,
                    include_contract_meta=detail_mode == "full",
                )
                out.update(scan_meta)
                out["ranking"] = "lowest_spread"
                out.update(_scope_fields("spread", spread_rows))
                out.update(
                    _market_scan_freshness_summary(
                        spread_rows,
                        include_stale_symbols=detail_mode == "full",
                    )
                )
                _attach_top_markets_units(out, spread_rows, headers=spread_headers)
                if detail_mode == "full":
                    out["evaluated_symbols"] = evaluated_counts["spread"]
                    out["skipped_symbols"] = metric_skips["spread"]
                    out["skipped_examples"] = metric_issues["spread"]
                return out

            if rank_kind == "volume":
                volume_headers = _top_markets_headers("volume", detail_mode=detail_mode)
                out = _market_scan_table(
                    volume_headers,
                    volume_rows,
                    include_contract_meta=detail_mode == "full",
                )
                out.update(scan_meta)
                out["ranking"] = "highest_tick_volume"
                out.update(_scope_fields("volume", volume_rows))
                out.update(
                    _market_scan_freshness_summary(
                        volume_rows,
                        include_stale_symbols=detail_mode == "full",
                    )
                )
                _attach_top_markets_units(out, volume_rows, headers=volume_headers)
                _attach_tick_volume_comparability(out, volume_rows)
                if detail_mode == "full":
                    out["evaluated_symbols"] = evaluated_counts["volume"]
                    out["skipped_symbols"] = metric_skips["volume"]
                    out["skipped_examples"] = metric_issues["volume"]
                return out

            if rank_kind in {
                "price_change",
                "abs_price_change",
                "live_price_change",
                "abs_live_price_change",
            }:
                price_change_headers = _top_markets_headers(
                    "price_change", detail_mode=detail_mode
                )
                out = _market_scan_table(
                    price_change_headers,
                    price_change_rows,
                    include_contract_meta=detail_mode == "full",
                )
                out.update(scan_meta)
                out["ranking"] = (
                    "largest_abs_live_price_change_pct"
                    if rank_kind == "abs_live_price_change"
                    else "highest_live_price_change_pct"
                    if rank_kind == "live_price_change"
                    else "largest_abs_price_change_pct"
                    if rank_kind == "abs_price_change"
                    else "highest_price_change_pct"
                )
                out["price_change_basis"] = (
                    "previous_completed_close_to_latest_completed_close"
                )
                out["live_price_change_basis"] = (
                    "previous_completed_close_to_live_quote_mid"
                )
                out["price_change_period"] = {
                    "bars": 1,
                    "timeframe": timeframe,
                    "bar_state": "completed",
                }
                out.update(_scope_fields("price_change", price_change_rows))
                out.update(
                    _market_scan_freshness_summary(
                        price_change_rows,
                        include_stale_symbols=detail_mode == "full",
                    )
                )
                _attach_top_markets_units(
                    out, price_change_rows, headers=price_change_headers
                )
                if detail_mode == "full":
                    out["evaluated_symbols"] = evaluated_counts["price_change"]
                    out["skipped_symbols"] = metric_skips["price_change"]
                    out["skipped_examples"] = metric_issues["price_change"]
                return attach_collection_contract(
                    out,
                    collection_kind="table",
                    rows=out.get("data"),
                    include_contract_meta=detail_mode == "full",
                )

            all_rows = [
                *_ranked_top_market_rows("lowest_spread", spread_rows),
                *_ranked_top_market_rows("highest_tick_volume", volume_rows),
                *_ranked_top_market_rows("highest_price_change_pct", price_change_rows),
                *_ranked_top_market_rows(
                    "largest_abs_price_change_pct",
                    abs_price_change_rows,
                ),
            ]
            all_headers = _top_markets_all_headers(detail_mode=detail_mode)
            out = _market_scan_table(
                all_headers,
                all_rows,
                include_contract_meta=detail_mode == "full",
            )
            out.update(scan_meta)
            out["ranking"] = "all"
            out["price_change_basis"] = (
                "previous_completed_close_to_latest_completed_close"
            )
            out["live_price_change_basis"] = (
                "previous_completed_close_to_live_quote_mid"
            )
            out["price_change_period"] = {
                "bars": 1,
                "timeframe": timeframe,
                "bar_state": "completed",
            }
            out["rank_categories"] = [
                "lowest_spread",
                "highest_tick_volume",
                "highest_price_change_pct",
                "largest_abs_price_change_pct",
            ]
            out["rank_by_categories"] = {
                "spread": "lowest_spread",
                "spread_pct": "lowest_spread",
                "tick_volume": "highest_tick_volume",
                "price_change": "highest_price_change_pct",
                "price_change_pct": "highest_price_change_pct",
                "abs_price_change": "largest_abs_price_change_pct",
                "abs_price_change_pct": "largest_abs_price_change_pct",
            }
            out["requested_limit"] = int(limit_value)
            out["universe_size"] = int(ranking_universe_size)
            out["returned_counts"] = {
                "lowest_spread": len(spread_rows),
                "highest_tick_volume": len(volume_rows),
                "highest_price_change_pct": len(price_change_rows),
                "largest_abs_price_change_pct": len(abs_price_change_rows),
            }
            out["available_counts"] = {
                "lowest_spread": evaluated_counts["spread"],
                "highest_tick_volume": evaluated_counts["volume"],
                "highest_price_change_pct": evaluated_counts["price_change"],
                "largest_abs_price_change_pct": evaluated_counts[
                    "abs_price_change"
                ],
            }
            out.update(
                _market_scan_freshness_summary(
                    all_rows,
                    include_stale_symbols=detail_mode == "full",
                )
            )
            ranking_rows = {
                "lowest_spread": spread_rows,
                "highest_tick_volume": volume_rows,
                "highest_price_change_pct": price_change_rows,
                "largest_abs_price_change_pct": abs_price_change_rows,
            }
            ranking_context: Dict[str, Dict[str, Any]] = {}
            ranking_times: List[str] = []
            for ranking_name, rows_for_ranking in ranking_rows.items():
                freshness_context = _market_scan_freshness_summary(
                    rows_for_ranking,
                    include_stale_symbols=False,
                )
                context = {
                    key: freshness_context[key]
                    for key in (
                        "data_as_of",
                        "data_as_of_range",
                        "bar_time_alignment",
                        "bar_rank_comparable",
                        "price_change_comparable",
                        "freshness",
                        "stale_rows",
                        "stale_bar_rows",
                        "unsafe_quote_rows",
                    )
                    if key in freshness_context
                }
                if rows_for_ranking:
                    context["data_source"] = rows_for_ranking[0].get("data_source")
                    context["timeframe"] = rows_for_ranking[0].get("timeframe")
                ranking_time = str(context.get("data_as_of") or "").strip()
                if ranking_time:
                    ranking_times.append(ranking_time)
                ranking_range = context.get("data_as_of_range")
                if isinstance(ranking_range, dict):
                    ranking_times.extend(
                        str(ranking_range.get(key) or "").strip()
                        for key in ("oldest", "newest")
                        if str(ranking_range.get(key) or "").strip()
                    )
                ranking_context[ranking_name] = context
            out["ranking_context"] = ranking_context
            if ranking_times:
                distinct_ranking_times = sorted(set(ranking_times))
                out["data_as_of_range"] = {
                    "oldest": distinct_ranking_times[0],
                    "newest": distinct_ranking_times[-1],
                }
                if len(distinct_ranking_times) == 1:
                    out["data_as_of"] = distinct_ranking_times[0]
                    out["data_as_of_basis"] = "shared_source_timestamp_across_rankings"
                else:
                    out.pop("data_as_of", None)
                    out.pop("bar_as_of", None)
                    out["data_as_of_basis"] = "per_ranking_source_timestamps"
                out["data_time_alignment"] = {
                    "status": (
                        "aligned" if len(distinct_ranking_times) == 1 else "mixed"
                    ),
                    "comparable": len(distinct_ranking_times) == 1,
                    "distinct_timestamps": len(distinct_ranking_times),
                }
            if detail_mode == "full":
                out["scan_stats"] = {
                    "spread": {
                        "evaluated_symbols": evaluated_counts["spread"],
                        "skipped_symbols": metric_skips["spread"],
                        "skipped_examples": metric_issues["spread"],
                    },
                    "volume": {
                        "evaluated_symbols": evaluated_counts["volume"],
                        "skipped_symbols": metric_skips["volume"],
                        "skipped_examples": metric_issues["volume"],
                    },
                    "price_change": {
                        "evaluated_symbols": evaluated_counts["price_change"],
                        "skipped_symbols": metric_skips["price_change"],
                        "skipped_examples": metric_issues["price_change"],
                    },
                    "abs_price_change": {
                        "evaluated_symbols": evaluated_counts["abs_price_change"],
                        "skipped_symbols": metric_skips["price_change"],
                        "skipped_examples": metric_issues["price_change"],
                    },
                }
            _attach_top_markets_units(out, all_rows, headers=all_headers)
            _attach_tick_volume_comparability(out, volume_rows)
            return out
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error scanning top markets: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="symbols_top_markets",
        rank_by=rank_by,
        limit=limit,
        universe=universe,
        timeframe=timeframe,
        detail=detail_mode,
        func=_run,
    )

_MARKET_SCAN_PRESETS: Dict[str, Dict[str, Any]] = {
    "oversold": {"rsi_below": 30.0, "min_tick_volume": 1000, "rank_by": "rsi"},
    "overbought": {"rsi_above": 70.0, "min_tick_volume": 1000, "rank_by": "rsi"},
    "high_volume": {"rank_by": "tick_volume"},
    "tight_spread": {"max_spread_pct": 0.01, "min_tick_volume": 500, "rank_by": "spread_pct"},
    "gap_up": {"min_gap_pct": 2.0, "rank_by": "gap_pct"},
    "gap_down": {"max_gap_pct": -2.0, "rank_by": "gap_pct"},
}

@mcp.tool()
def market_scan(  # noqa: C901
    symbols: Optional[str] = None,
    group: Optional[str] = None,
    preset: Optional[str] = None,
    limit: Annotated[int, Field(ge=1)] = 10,
    offset: Annotated[int, Field(ge=0)] = 0,
    universe: Literal["visible", "all"] = "visible",  # type: ignore
    timeframe: TimeframeLiteral = "H1",
    detail: DetailLiteral = "compact",
    lookback: Annotated[int, Field(ge=2)] = 100,
    rsi_length: Annotated[int, Field(ge=1)] = 14,
    sma_period: Annotated[int, Field(ge=1)] = 20,
    min_price_change_pct: Optional[float] = None,
    max_price_change_pct: Optional[float] = None,
    max_spread_pct: Annotated[Optional[float], Field(ge=0.0)] = None,
    min_tick_volume: Annotated[Optional[int], Field(ge=0)] = None,
    rsi_below: Annotated[Optional[float], Field(ge=0.0, le=100.0)] = None,
    rsi_above: Annotated[Optional[float], Field(ge=0.0, le=100.0)] = None,
    price_vs_sma: Optional[Literal["above", "below"]] = None,  # type: ignore
    rank_by: Literal["abs_price_change_pct", "abs_price_change", "abs_live_price_change_pct", "abs_live_price_change", "live_price_change_pct", "live_price_change", "price_change_pct", "price_change", "gap_pct", "tick_volume", "rsi", "spread_pct", "spread"] = "abs_price_change_pct",  # type: ignore
    rank_order: Literal["auto", "asc", "desc", "ascending", "descending"] = "auto",  # type: ignore
    quote_usable_only: Optional[bool] = None,
    allow_partial: bool = True,
) -> Dict[str, Any]:
    """Filtered MT5 market scanner with one flat table and technical filters.

    Pass `symbols` for one instrument or a comma-separated list. `price_change_pct`
    is the return from the previous completed bar close to the latest completed
    bar close. Use `live_price_change_pct` or `abs_live_price_change_pct` to rank
    from the previous completed close to the current quote midpoint. `data` is
    the canonical flat row payload. Compact detail is the
    default; use
    `detail="full"` when you also want the explicit `columns` ordering hint
    for compatibility. Broad scans use the visible universe; `universe="all"`
    must be combined with `symbols` or `group` to avoid unbounded hidden-symbol
    activation. Use `symbols_top_markets` for a quick all-market overview with
    separate spread, volume, and mover leaderboards. `quote_usable_only` defaults
    to true for spread and live-price-change rankings and the `tight_spread`
    preset, excluding stale, future, locked, inverted, and one-sided quotes
    before pagination. Set it explicitly for other rankings. Locked or invalid
    quotes cannot satisfy a maximum-spread filter. `allow_partial` defaults true
    so unknown requested names are dropped with `missing_symbols` and a warning;
    explicit lists are permissive by default. Set false to fail closed when any
    requested symbol is missing or cannot be evaluated. Evaluation failures
    produce partial_failure=true and ranking_complete=false. A scan where no
    symbol can be evaluated fails even when allow_partial=true.
    """

    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    preset_value = str(preset or "").strip().lower().replace("-", "_")
    preset_error = None
    preset_config = _MARKET_SCAN_PRESETS.get(preset_value) if preset_value else None
    min_gap_pct: Optional[float] = None
    max_gap_pct: Optional[float] = None
    if preset_value and preset_config is None:
        preset_error = (
            "preset must be one of: "
            + ", ".join(sorted(_MARKET_SCAN_PRESETS))
            + "."
        )
    elif preset_config:
        min_gap_pct = preset_config.get("min_gap_pct")
        max_gap_pct = preset_config.get("max_gap_pct")
        if min_price_change_pct is None and "min_price_change_pct" in preset_config:
            min_price_change_pct = preset_config["min_price_change_pct"]
        if max_price_change_pct is None and "max_price_change_pct" in preset_config:
            max_price_change_pct = preset_config["max_price_change_pct"]
        if max_spread_pct is None and "max_spread_pct" in preset_config:
            max_spread_pct = preset_config["max_spread_pct"]
        if min_tick_volume is None and "min_tick_volume" in preset_config:
            min_tick_volume = preset_config["min_tick_volume"]
        if rsi_below is None and "rsi_below" in preset_config:
            rsi_below = preset_config["rsi_below"]
        if rsi_above is None and "rsi_above" in preset_config:
            rsi_above = preset_config["rsi_above"]
        if rank_by in {None, "abs_price_change_pct"} and "rank_by" in preset_config:
            rank_by = preset_config["rank_by"]

    normalized_rank_by, _ = _normalize_market_scan_rank_by(rank_by)
    quote_usable_only_value = (
        normalized_rank_by
        in {"spread_pct", "live_price_change_pct", "abs_live_price_change_pct"}
        or preset_value == "tight_spread"
        if quote_usable_only is None
        else bool(quote_usable_only)
    )

    def _run() -> Dict[str, Any]:  # noqa: C901
        request: Dict[str, Any] = {
            "symbols": symbols,
            "group": group,
            "preset": preset_value or None,
            "limit": limit,
            "offset": offset,
            "universe": universe,
            "timeframe": timeframe,
            "detail": detail_mode,
            "lookback": lookback,
            "rank_by": rank_by,
            "rank_order": rank_order,
            "quote_usable_only": quote_usable_only_value,
            "allow_partial": bool(allow_partial),
            "filters": {
                key: value
                for key, value in {
                    "min_price_change_pct": min_price_change_pct,
                    "max_price_change_pct": max_price_change_pct,
                    "min_gap_pct": min_gap_pct,
                    "max_gap_pct": max_gap_pct,
                    "max_spread_pct": max_spread_pct,
                    "min_tick_volume": min_tick_volume,
                    "rsi_below": rsi_below,
                    "rsi_above": rsi_above,
                    "price_vs_sma": price_vs_sma,
                    "rsi_length": rsi_length,
                    "sma_period": sma_period,
                }.items()
                if value is not None
            },
        }
        try:
            if preset_error:
                return _market_scan_error(
                    preset_error,
                    code="invalid_input",
                    request=request,
                )

            universe_value = str(universe or "visible").strip().lower()
            request["universe"] = universe_value
            if universe_value not in {"visible", "all"}:
                return _market_scan_error(
                    "universe must be 'visible' or 'all'.",
                    code="invalid_input",
                    request=request,
                )

            if symbols is not None and not any(
                item.strip()
                for item in str(symbols).replace(";", ",").split(",")
            ):
                return _market_scan_error(
                    (
                        "symbols was supplied but contains no symbols; omit it "
                        "to scan the visible universe."
                    ),
                    code="empty_symbol_selector",
                    request=request,
                )
            symbols_value = str(symbols or "").strip()
            symbols_filter = symbols_value or None
            if universe_value == "all" and not symbols_filter and not group:
                return _market_scan_error(
                    (
                        "market_scan universe='all' requires symbols or group "
                        "to bound the scan. Use universe='visible' for broad scans "
                        "or symbols_top_markets for quick all-market leaderboards."
                    ),
                    code="invalid_input",
                    request=request,
                )

            timeframe_value = str(timeframe or "H1").strip().upper()
            request["timeframe"] = timeframe_value
            if timeframe_value not in TIMEFRAME_MAP:
                return _market_scan_error(
                    invalid_timeframe_error(timeframe_value, TIMEFRAME_MAP),
                    code="invalid_timeframe",
                    request=request,
                )
            mt5_timeframe = TIMEFRAME_MAP[timeframe_value]

            rank_by_value, rank_by_input = _normalize_market_scan_rank_by(rank_by)
            request["rank_by"] = rank_by_value
            if rank_by_input != rank_by_value:
                request["rank_by_input"] = rank_by_input
            if rank_by_value not in {"abs_price_change_pct", "abs_live_price_change_pct", "live_price_change_pct", "price_change_pct", "gap_pct", "tick_volume", "rsi", "spread_pct"}:
                return _market_scan_error(
                    (
                        "rank_by must be one of: "
                        f"{', '.join(_MARKET_SCAN_RANK_BY_CHOICES)}."
                    ),
                    code="invalid_input",
                    request=request,
                )
            rank_order_value, rank_order_input = _normalize_market_scan_rank_order(rank_order)
            request["rank_order"] = rank_order_value
            if rank_order_input != rank_order_value:
                request["rank_order_input"] = rank_order_input
            if rank_order_value not in {"auto", "asc", "desc"}:
                return _market_scan_error(
                    "rank_order must be one of: auto, asc, desc, ascending, descending.",
                    code="invalid_input",
                    request=request,
                )

            price_vs_sma_value = None
            if price_vs_sma is not None:
                price_vs_sma_value = str(price_vs_sma).strip().lower()
                request["filters"] = {
                    **dict(request.get("filters", {})),
                    "price_vs_sma": price_vs_sma_value,
                }
                if price_vs_sma_value not in {"above", "below"}:
                    return _market_scan_error(
                        "price_vs_sma must be 'above' or 'below'.",
                        code="invalid_input",
                        request=request,
                    )

            try:
                lookback_value = int(lookback)
                rsi_length_value = int(rsi_length)
                sma_period_value = int(sma_period)
            except Exception:
                return _market_scan_error(
                    "lookback, rsi_length, and sma_period must be integers.",
                    code="invalid_input",
                    request=request,
                )
            request["lookback"] = lookback_value
            request["filters"] = {
                **dict(request.get("filters", {})),
                "rsi_length": rsi_length_value,
                "sma_period": sma_period_value,
            }
            if lookback_value < 2:
                return _market_scan_error(
                    "lookback must be at least 2.",
                    code="invalid_input",
                    request=request,
                )
            if rsi_length_value < 1:
                return _market_scan_error(
                    "rsi_length must be at least 1.",
                    code="invalid_input",
                    request=request,
                )
            if sma_period_value < 1:
                return _market_scan_error(
                    "sma_period must be at least 1.",
                    code="invalid_input",
                    request=request,
                )

            required_lookback = 2
            if rank_by_value == "rsi" or rsi_above is not None or rsi_below is not None:
                required_lookback = max(required_lookback, rsi_length_value + 1)
            if price_vs_sma_value is not None:
                required_lookback = max(required_lookback, sma_period_value)
            if lookback_value < required_lookback:
                return _market_scan_error(
                    (
                        f"lookback={lookback_value} is too small for the requested filters; "
                        f"need at least {required_lookback} bars."
                    ),
                    code="invalid_input",
                    request=request,
                )

            if limit is None:
                limit_value = 10
            else:
                try:
                    limit_value = int(limit)
                except Exception:
                    limit_value = 0
                if limit_value <= 0:
                    return _market_scan_error(
                        "limit must be a positive integer.",
                        code="invalid_input",
                        request=request,
                    )
            request["limit"] = limit_value
            try:
                offset_value = int(offset or 0)
            except Exception:
                return _market_scan_error(
                    "offset must be a non-negative integer.",
                    code="invalid_input",
                    request=request,
                )
            if offset_value < 0:
                return _market_scan_error(
                    "offset must be >= 0.",
                    code="invalid_input",
                    request=request,
                )
            request["offset"] = offset_value

            for filter_name, filter_value in (
                ("min_price_change_pct", min_price_change_pct),
                ("max_price_change_pct", max_price_change_pct),
                ("max_spread_pct", max_spread_pct),
                ("rsi_below", rsi_below),
                ("rsi_above", rsi_above),
            ):
                if filter_value is None:
                    continue
                numeric = _market_scan_float(filter_value)
                invalid = numeric is None
                if filter_name == "max_spread_pct":
                    invalid = invalid or numeric < 0.0
                elif filter_name in {"rsi_below", "rsi_above"}:
                    invalid = invalid or not 0.0 <= numeric <= 100.0
                if invalid:
                    if filter_name == "max_spread_pct":
                        expected = "a non-negative percentage"
                    elif filter_name in {"rsi_below", "rsi_above"}:
                        expected = "between 0 and 100"
                    else:
                        expected = "a finite number"
                    return _market_scan_error(
                        f"{filter_name} must be {expected}.",
                        code="invalid_input",
                        request=request,
                    )
            min_change = _market_scan_float(min_price_change_pct)
            max_change = _market_scan_float(max_price_change_pct)
            if (
                min_change is not None
                and max_change is not None
                and min_change > max_change
            ):
                return _market_scan_error(
                    "min_price_change_pct must be <= max_price_change_pct.",
                    code="contradictory_filters",
                    request=request,
                )
            rsi_floor = _market_scan_float(rsi_above)
            rsi_ceiling = _market_scan_float(rsi_below)
            if (
                rsi_floor is not None
                and rsi_ceiling is not None
                and rsi_floor >= rsi_ceiling
            ):
                return _market_scan_error(
                    "rsi_above must be < rsi_below.",
                    code="contradictory_filters",
                    request=request,
                )
            if min_tick_volume is not None:
                try:
                    min_tick_volume_value = int(min_tick_volume)
                except Exception:
                    min_tick_volume_value = -1
                if min_tick_volume_value < 0:
                    return _market_scan_error(
                        "min_tick_volume must be a non-negative integer.",
                        code="invalid_input",
                        request=request,
                    )

            include_rsi = (
                detail_mode != "compact"
                or rank_by_value == "rsi"
                or rsi_above is not None
                or rsi_below is not None
            )
            include_sma = detail_mode != "compact" or price_vs_sma_value is not None
            signal_lookback = lookback_value if (include_rsi or include_sma) else 2

            mt5_gateway = create_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            spread_cost_currency = account_currency_from_gateway(mt5_gateway)

            with _symbol_visibility_snapshot_guard():
                raw_symbols = mt5_gateway.symbols_get()
            if raw_symbols is None:
                return _attach_market_scan_source(
                    _market_scan_error(
                        f"Failed to get symbols: {mt5_gateway.last_error()}",
                        code="data_fetch_failed",
                        request=request,
                    ),
                    mt5_gateway,
                )
            all_symbols = list(raw_symbols)
            broker_symbol_count = len(all_symbols)
            visible_symbol_count = sum(
                bool(getattr(symbol, "visible", False)) for symbol in all_symbols
            )

            selected_symbols, selection_meta, selection_error = _select_market_scan_symbols(
                all_symbols,
                symbols=symbols_filter,
                group=group,
                universe=universe_value,
            )
            if selection_error:
                error_code = "invalid_input"
                if group and selection_meta.get("symbols_input") is None:
                    error_code = "symbol_group_error"
                return _attach_market_scan_source(
                    _market_scan_error(
                        selection_error,
                        code=error_code,
                        request=request,
                        details={
                            "did_you_mean": selection_meta.get("did_you_mean", []),
                            "search_hint": (
                                "Browse matching broker symbols with the symbols_list "
                                "tool (CLI: mtdata-cli symbols_list --search-term TERM)."
                            ),
                        }
                        if selection_meta.get("did_you_mean")
                        else None,
                    ),
                    mt5_gateway,
                )

            if selection_meta.get("symbols_input") is not None:
                request["symbols_input"] = selection_meta.get("symbols_input")
            request["scope"] = selection_meta.get("scope")
            if selection_meta.get("group") is not None:
                request["group"] = selection_meta.get("group")
            if selection_meta.get("groups") is not None:
                request["groups"] = selection_meta.get("groups")
            if selection_meta.get("requested_symbols") is not None:
                request["requested_symbols"] = selection_meta.get("requested_symbols")
            if selection_meta.get("missing_symbols") is not None:
                request["missing_symbols"] = selection_meta.get("missing_symbols")
            missing_requested = list(selection_meta.get("missing_symbols") or [])
            if missing_requested and not allow_partial:
                return _attach_market_scan_source(
                    _market_scan_error(
                        "Requested symbol(s) not found: "
                        + ", ".join(missing_requested)
                        + ".",
                        code="missing_symbols",
                        request=request,
                        details={
                            "missing_symbols": missing_requested,
                            "did_you_mean": selection_meta.get("did_you_mean", []),
                        },
                    ),
                    mt5_gateway,
                )
            if len(selected_symbols) > _MARKET_SCAN_MAX_CANDIDATES:
                return _attach_market_scan_source(
                    _market_scan_error(
                        (
                            f"The filtered universe contains {len(selected_symbols)} candidates, "
                            f"above the safe synchronous cap of {_MARKET_SCAN_MAX_CANDIDATES}. "
                            "Narrow the exact scan with symbols or group."
                        ),
                        code="candidate_universe_too_large",
                        request=request,
                        stats={
                            "candidate_count": len(selected_symbols),
                            "candidate_cap": _MARKET_SCAN_MAX_CANDIDATES,
                        },
                    ),
                    mt5_gateway,
                )
            started_at = time.perf_counter()
            matched_rows: List[Dict[str, Any]] = []
            skipped_examples: List[Dict[str, str]] = []
            skipped_symbols = 0
            skipped_reason_counts: Dict[str, int] = {}
            failed_symbols: List[Dict[str, str]] = []
            evaluated_symbols = 0
            quote_eligibility_excluded = 0
            quote_eligibility_reasons: Dict[str, int] = {}
            quote_eligibility_examples: List[Dict[str, str]] = []

            def _record_issue(symbol_name: str, reason: str) -> None:
                nonlocal skipped_symbols
                skipped_symbols += 1
                failed_symbols.append({"symbol": symbol_name, "reason": str(reason)})
                reason_key = str(reason or "unknown")
                skipped_reason_counts[reason_key] = (
                    skipped_reason_counts.get(reason_key, 0) + 1
                )
                if len(skipped_examples) < 10:
                    skipped_examples.append({"symbol": symbol_name, "reason": reason})

            for missing_symbol in selection_meta.get("missing_symbols", []):
                _record_issue(missing_symbol, "Requested symbol not found.")

            def _evaluate_symbol(symbol_obj: Any) -> None:
                nonlocal evaluated_symbols, quote_eligibility_excluded
                symbol_name = str(getattr(symbol_obj, "name", "") or "")

                spread_row, spread_error = _build_market_scan_spread_row(
                    symbol_obj,
                    mt5_gateway,
                    spread_cost_currency=spread_cost_currency,
                )
                if spread_error or spread_row is None:
                    _record_issue(symbol_name, spread_error or "Spread data is unavailable.")
                    return

                signal_row, signal_error = _build_market_scan_signal_row(
                    symbol_obj,
                    timeframe=timeframe_value,
                    mt5_timeframe=mt5_timeframe,
                    lookback=signal_lookback,
                    rsi_length=rsi_length_value,
                    sma_period=sma_period_value,
                    include_rsi=include_rsi,
                    include_sma=include_sma,
                )
                if signal_error or signal_row is None:
                    _record_issue(symbol_name, signal_error or "Bar data is unavailable.")
                    return

                row = dict(spread_row)
                row.update(signal_row)
                _namespace_market_scan_quote_freshness(row)
                _attach_market_scan_live_change(row)
                metric_error = _market_scan_missing_required_metric(
                    row,
                    rank_by=rank_by_value,
                    rsi_above=rsi_above,
                    rsi_below=rsi_below,
                    price_vs_sma=price_vs_sma_value,
                    max_spread_pct=max_spread_pct,
                    min_tick_volume=min_tick_volume,
                    min_price_change_pct=min_price_change_pct,
                    max_price_change_pct=max_price_change_pct,
                    min_gap_pct=min_gap_pct,
                    max_gap_pct=max_gap_pct,
                    rsi_length=rsi_length_value,
                    sma_period=sma_period_value,
                )
                if metric_error:
                    _record_issue(symbol_name, metric_error)
                    return

                evaluated_symbols += 1
                if (
                    quote_usable_only_value
                    and row.get("quote_usable_for_live_trading") is not True
                ):
                    reason = _market_scan_quote_exclusion_reason(row)
                    quote_eligibility_excluded += 1
                    quote_eligibility_reasons[reason] = (
                        quote_eligibility_reasons.get(reason, 0) + 1
                    )
                    if len(quote_eligibility_examples) < 10:
                        quote_eligibility_examples.append(
                            {"symbol": symbol_name, "reason": reason}
                        )
                    return
                if not _market_scan_row_matches_filters(
                    row,
                    min_price_change_pct=min_price_change_pct,
                    max_price_change_pct=max_price_change_pct,
                    max_spread_pct=max_spread_pct,
                    min_tick_volume=min_tick_volume,
                    rsi_below=rsi_below,
                    rsi_above=rsi_above,
                    price_vs_sma=price_vs_sma_value,
                    min_gap_pct=min_gap_pct,
                    max_gap_pct=max_gap_pct,
                ):
                    return
                matched_rows.append(row)

            for symbol_obj in selected_symbols:
                symbol_name = str(getattr(symbol_obj, "name", "") or "")
                is_hidden = not bool(getattr(symbol_obj, "visible", False))
                if is_hidden:
                    with _symbol_ready_guard(symbol_name, info_before=symbol_obj) as (err, _):
                        if err:
                            _record_issue(symbol_name, err)
                            continue
                        _evaluate_symbol(symbol_obj)
                    continue
                _evaluate_symbol(symbol_obj)

            effective_rank_order = _market_scan_effective_rank_order(
                rank_by_value,
                rank_order=rank_order_value,
                rsi_above=rsi_above,
                rsi_below=rsi_below,
            )
            if (
                rank_order_value == "auto"
                and rank_by_value == "gap_pct"
                and max_gap_pct is not None
                and float(max_gap_pct) < 0.0
            ):
                effective_rank_order = "asc"
            _market_scan_sort_rows(
                matched_rows,
                rank_by=rank_by_value,
                rank_order=effective_rank_order,
                rsi_above=rsi_above,
                rsi_below=rsi_below,
            )
            total_matches = len(matched_rows)
            limited_rows = matched_rows[offset_value : offset_value + limit_value]
            freshness_cut_larger_mover = False
            if rank_by_value in {
                "abs_price_change_pct",
                "price_change_pct",
                "abs_live_price_change_pct",
                "live_price_change_pct",
            }:
                metric_key = (
                    "live_price_change_pct"
                    if "live_price_change" in rank_by_value
                    else "price_change_pct"
                )
                freshness_cut_larger_mover = _larger_abs_metric_cut_for_freshness(
                    matched_rows,
                    limited_rows,
                    metric_key,
                )

            full_headers = [
                "symbol",
                "group",
                "asset_class",
                "description",
                "timeframe",
                "time",
                "price_as_of",
                "price_freshness",
                "quote_time",
                "quote_as_of",
                "bid",
                "ask",
                "mid",
                "quote_age_seconds",
                "quote_age_anchor",
                "quote_age_metric",
                "quote_stale_after_seconds",
                "quote_stale",
                "quote_freshness_reason",
                "quote_timestamp_in_future",
                "quote_timestamp_skew_seconds",
                "quote_timestamp_warning",
                "quote_warning",
                "quote_freshness",
                "quote_usable_for_live_trading",
                "spread_valid",
                "spread_quality",
                "bar_age_seconds",
                "bar_freshness_anchor",
                "bar_freshness_metric",
                "bar_stale_after_seconds",
                "bar_age_hours",
                "bar_stale",
                "bar_market_status",
                "bar_market_status_reason",
                "bar_freshness_policy_relaxed",
                "bar_freshness",
                "bar_stale_warning",
                "previous_close",
                "close",
                "price_currency",
                "price_basis",
                "price_point",
                "price_change_pct",
                "live_price_change_pct",
                "direction_divergence",
                "live_price_change_basis",
                "gap_pct",
                "tick_volume",
                "spread_pct",
                "spread_cost_per_lot",
                "spread_cost_currency",
                "rsi",
                "sma_value",
                "sma_distance_pct",
            ]
            compact_headers = [
                "symbol",
                "asset_class",
                "bar_close",
                "bar_stale",
                "time",
                "quote_as_of",
                "quote_age_seconds",
                "quote_freshness_reason",
                "quote_timestamp_ahead_of_wall_clock",
                "quote_timestamp_in_future",
                "quote_timestamp_skew_seconds",
                "quote_timestamp_warning",
                "bid",
                "ask",
                "spread_quality",
                "quote_source_state",
                "quote_usable_for_live_trading",
                "price_change_pct",
                "live_price_change_pct",
                "direction_divergence",
                "spread_pips",
                "spread_pct",
            ]
            if rank_by_value == "tick_volume" or min_tick_volume is not None:
                compact_headers.append("tick_volume")
            if rank_by_value == "gap_pct" or min_gap_pct is not None or max_gap_pct is not None:
                compact_headers.append("gap_pct")
            if include_rsi:
                compact_headers.append("rsi")
            if include_sma:
                compact_headers.append("sma_distance_pct")
            compact_shared_fields: Dict[str, Any] = {}
            if detail_mode == "compact":
                compact_headers, compact_shared_fields = (
                    _compact_market_scan_projection(compact_headers, limited_rows)
                )
            headers = compact_headers if detail_mode == "compact" else full_headers
            output_rows = (
                _project_market_scan_rows(headers, limited_rows)
                if detail_mode == "compact"
                else limited_rows
            )
            request["filters"] = {
                key: value
                for key, value in {
                    "min_price_change_pct": min_price_change_pct,
                    "max_price_change_pct": max_price_change_pct,
                    "min_gap_pct": min_gap_pct,
                    "max_gap_pct": max_gap_pct,
                    "max_spread_pct": max_spread_pct,
                    "min_tick_volume": min_tick_volume,
                    "rsi_below": rsi_below,
                    "rsi_above": rsi_above,
                    "price_vs_sma": price_vs_sma_value,
                    "rsi_length": rsi_length_value
                    if (rsi_above is not None or rsi_below is not None or rank_by_value == "rsi")
                    else None,
                    "sma_period": sma_period_value
                    if price_vs_sma_value is not None
                    else None,
                }.items()
                if value is not None
            }
            stats = {
                "scanned_symbols": len(selected_symbols),
                "evaluated_symbols": evaluated_symbols,
                "matched_symbols": total_matches,
                "filtered_out_symbols": max(0, evaluated_symbols - total_matches),
                "skipped_symbols": skipped_symbols,
                "skipped_examples": skipped_examples,
                "skipped_reason_counts": skipped_reason_counts,
                "quote_eligibility_excluded_symbols": quote_eligibility_excluded,
                "query_latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            }
            table_payload = _market_scan_contract_table(
                headers,
                output_rows,
                include_columns=detail_mode == "full",
            )
            freshness_summary = _market_scan_freshness_summary(
                limited_rows,
                include_stale_symbols=detail_mode == "full",
            )
            stale_rows = int(freshness_summary.get("stale_rows") or 0)
            returned_count = int(table_payload["row_count"])
            missing_symbols = list(selection_meta.get("missing_symbols") or [])
            requested_symbols = list(selection_meta.get("requested_symbols") or [])
            if missing_symbols and requested_symbols:
                message = (
                    f"Returned {max(0, len(requested_symbols) - len(missing_symbols))} "
                    f"of {len(requested_symbols)} requested symbols; dropped "
                    + ", ".join(missing_symbols)
                    + "."
                )
            elif total_matches > returned_count:
                message = (
                    f"Showing {returned_count} of {int(total_matches)} symbols matching "
                    "the requested market scan filters."
                )
            elif total_matches > 0:
                message = (
                    f"Returned all {returned_count} symbol(s) matching the requested "
                    "market scan filters."
                )
            else:
                message = "No symbols matched the requested market scan filters."
                filter_reason_parts = []
                if rsi_below is not None:
                    filter_reason_parts.append(f"RSI < {rsi_below:g}")
                if rsi_above is not None:
                    filter_reason_parts.append(f"RSI > {rsi_above:g}")
                if min_tick_volume is not None:
                    filter_reason_parts.append(
                        f"tick_volume >= {min_tick_volume:g}"
                    )
                evaluated = int(stats["evaluated_symbols"])
                if evaluated > 0 and filter_reason_parts:
                    message = (
                        f"No symbols matched the requested market scan filters "
                        f"(all {evaluated} evaluated; none had "
                        + " and ".join(filter_reason_parts)
                        + ")."
                    )
            if stale_rows:
                message = (
                    f"{message} Returned rows: "
                    f"{stale_rows}/{int(table_payload['row_count'])} stale."
                )
            out: Dict[str, Any] = {
                "success": True,
                "status": "ok" if total_matches > 0 else "no_matches",
                "message": message,
                "data": table_payload["rows"],
                "count": table_payload["row_count"],
                "rank_by": rank_by_value,
                "rank_order": effective_rank_order,
                "ranking_policy": _market_scan_ranking_policy(rank_by_value, effective_rank_order),
                "ranking_complete": not freshness_cut_larger_mover and not skipped_symbols,
                "ranking": _market_scan_ranking_label(
                    rank_by_value,
                    rank_order=effective_rank_order,
                    rsi_above=rsi_above,
                    rsi_below=rsi_below,
                ),
                "quote_usable_only": quote_usable_only_value,
                "quote_eligibility": {
                    "basis": "quote_usable_for_live_trading",
                    "required": quote_usable_only_value,
                    "excluded_symbols": int(quote_eligibility_excluded),
                    "excluded_reasons": dict(sorted(quote_eligibility_reasons.items())),
                    "excluded_examples": quote_eligibility_examples,
                },
                "price_change_basis": "previous_completed_close_to_latest_completed_close",
                "live_price_change_basis": (
                    "previous_completed_close_to_live_quote_mid"
                ),
                "ranking_basis": _market_scan_ranking_basis(rank_by_value),
                "price_change_period": {
                    "bars": 1,
                    "timeframe": timeframe,
                    "bar_state": "completed",
                },
                "gap_basis": "previous_completed_close_to_latest_completed_open",
                "pagination": build_pagination_meta(
                    total=total_matches,
                    returned=table_payload["row_count"],
                    offset=offset_value,
                    limit=limit_value,
                ),
                "universe_size": int(len(selected_symbols)),
                "summary": {
                    "counts": {
                        "scanned_symbols": int(stats["scanned_symbols"]),
                        "evaluated_symbols": int(stats["evaluated_symbols"]),
                        "filtered_out_symbols": int(stats["filtered_out_symbols"]),
                        "skipped_symbols": int(stats["skipped_symbols"]),
                        "quote_eligibility_excluded_symbols": int(
                            quote_eligibility_excluded
                        ),
                    }
                },
                "meta": _market_scan_contract_meta(request=request, stats=stats),
            }
            if rank_order_value != effective_rank_order:
                out["rank_order_requested"] = rank_order_value
            if skipped_examples:
                out["summary"]["skipped_examples"] = skipped_examples
            if skipped_reason_counts:
                out["summary"]["skipped_reason_counts"] = dict(
                    sorted(
                        skipped_reason_counts.items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                )
            compact_warnings = compact_shared_fields.pop("warnings", None)
            out.update(compact_shared_fields)
            if compact_warnings:
                out["warnings"] = list(compact_warnings)
            if preset_value:
                out["preset"] = preset_value
                out["preset_filters"] = {
                    key: value
                    for key, value in dict(preset_config or {}).items()
                    if key != "rank_by"
                }
                out["preset_rank_by"] = rank_by_value
            if detail_mode == "full":
                out["returned_count"] = int(table_payload["row_count"])
                out["summary"]["counts"]["matched_symbols"] = int(
                    stats["matched_symbols"]
                )
            missing_symbols = list(selection_meta.get("missing_symbols") or [])
            if missing_symbols:
                out["missing_symbols"] = missing_symbols
                if returned_count > 0:
                    out["partial_failure"] = True
                out.setdefault("warnings", []).append(
                    "Requested symbol(s) not found and excluded from the scan: "
                    + ", ".join(missing_symbols)
                    + "."
                )
            out.update(freshness_summary)
            _attach_market_scan_rank_gap_warning(out, table_payload["rows"])
            units = _market_scan_units_for_rows(table_payload["rows"])
            if units:
                out["units"] = units
            _attach_market_scan_volume_semantics(out, units)
            if rank_by_value == "tick_volume":
                _attach_tick_volume_comparability(out, table_payload["rows"])
            if "columns" in table_payload:
                out["columns"] = table_payload["columns"]
            if (
                selection_meta.get("scope") == "universe"
                and len(selected_symbols) < int(limit_value)
            ):
                out["note"] = (
                    f"Requested {int(limit_value)} rows but only "
                    f"{len(selected_symbols)} symbols were available in the "
                    f"{universe_value} universe."
                )
            if total_matches == 0:
                out["summary"]["empty"] = True
                out["visible_symbols"] = int(visible_symbol_count)
                out["broker_symbols"] = int(broker_symbol_count)
                evaluated = int(stats["evaluated_symbols"])
                if universe_value == "visible" and evaluated == 0:
                    out["remediation"] = (
                        "The default scan covers Market Watch symbols. Add symbols "
                        "to Market Watch, pass explicit --symbols, or use --universe "
                        "all with --symbols/--group to widen the scan safely."
                    )
                    out["message"] = (
                        f"{out['message']} Market Watch has "
                        f"{visible_symbol_count} visible symbol(s) out of "
                        f"{broker_symbol_count} broker symbol(s)."
                    )
                elif evaluated > 0:
                    out["remediation"] = (
                        "Relax the scan filters first (for example --rsi-below or "
                        "--min-tick-volume). Widen Market Watch or pass --universe "
                        "all only when evaluated_symbols is 0."
                    )
            if failed_symbols:
                out["partial_failure"] = True
                out["failed_symbols"] = failed_symbols
                out["success"] = bool(allow_partial and evaluated_symbols > 0)
                out["status"] = "partial" if out["success"] else "failed"
                out["message"] = (
                    f"Evaluated {evaluated_symbols} symbol(s); {len(failed_symbols)} "
                    "symbol(s) could not be evaluated. Ranking is incomplete."
                )
                out["remediation"] = "Inspect failed_symbols and correct the symbol or history errors. Relaxing scan filters does not repair evaluation failures."
                out.setdefault("warnings", []).append(out["message"])
                if not out["success"]:
                    out["error_code"] = "market_scan_incomplete"
                    out["error"] = out["message"]
            return _attach_market_scan_source(
                attach_collection_contract(
                    out,
                    collection_kind="table",
                    rows=output_rows,
                    include_contract_meta=detail_mode == "full",
                ),
                mt5_gateway,
            )
        except MT5ConnectionError as exc:
            return _attach_market_scan_source(
                _market_scan_error(
                    str(exc),
                    code="mt5_connection_error",
                    request=request,
                )
            )
        except Exception as exc:
            return _attach_market_scan_source(
                _market_scan_error(
                    f"Error running market scan: {str(exc)}",
                    code="market_scan_failed",
                    request=request,
                )
            )

    return run_logged_operation(
        logger,
        operation="market_scan",
        symbols=symbols,
        group=group,
        limit=limit,
        offset=offset,
        universe=universe,
        timeframe=timeframe,
        func=_run,
    )
