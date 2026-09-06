
import logging
import math
import os
import time
from typing import Any, Dict, Literal, Optional

from ..shared.market_units import (
    UNIT_BROKER_POINTS,
    UNIT_PERCENT,
    UNIT_PIPS,
    forex_points_per_pip,
)
from ..shared.schema import DetailLiteral
from ..utils.coercion import coerce_finite_float as _optional_finite_float
from ..utils.coercion import round_finite
from ..utils.freshness import (
    QUOTE_STALE_SECONDS,
    format_age_seconds,
)
from ..utils.market_metadata import (
    FRESHNESS_ANCHOR_WALL_CLOCK,
    FRESHNESS_METRIC_LAST_TICK_AGE,
    build_tick_freshness_context,
)
from ..utils.mt5 import (
    MT5ConnectionError,
    _ensure_symbol_ready,
    account_currency_from_gateway,
    describe_mt5_time_normalization,
    ensure_mt5_connection_or_raise,
    mt5,
    resolve_broker_symbol_name,
    resolve_public_symbol,
    symbol_price_currency,
    symbol_price_digits,
    symbol_price_point,
)
from ..utils.quote import (
    QUOTE_EXECUTION_SOURCE_AGREEMENT_BASIS,
    compute_spread_metrics,
    enforce_quote_execution_readiness,
    resolve_quote_tick,
    tick_epoch,
    tick_value,
)
from ..utils.symbol import (
    find_live_extended_session_symbols,
    symbol_suggestions_from_gateway,
)
from ..utils.time import (
    _format_time_second_explicit,
    _format_time_second_explicit_local,
    _resolve_client_tz,
    _use_client_tz,
    display_timezone_label,
)
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway
from .output_contract import ensure_common_meta, normalize_output_verbosity_detail
from .runtime_metadata import attach_mt5_source

logger = logging.getLogger(__name__)
_MARKET_DEPTH_ENABLE_ENV = "MTDATA_ENABLE_MARKET_DEPTH_FETCH"
_MARKET_TICKER_STALE_SECONDS = QUOTE_STALE_SECONDS
_MARKET_DEPTH_INITIAL_SNAPSHOT_ATTEMPTS = 20
_MARKET_DEPTH_INITIAL_SNAPSHOT_INTERVAL_SECONDS = 0.01
_MARKET_DEPTH_BOOK_UNITS = {
    "volume": "book_volume",
    "volume_real": "book_volume_real",
}
_MARKET_DEPTH_TICK_UNITS = {"volume": "mt5_tick_volume"}
_MARKET_DEPTH_SPREAD_UNITS = {
    "spread": "absolute_price",
    "spread_points": UNIT_BROKER_POINTS,
    "spread_pct": UNIT_PERCENT,
    "spread_cost_per_lot": "account_currency_per_broker_lot_estimate",
}
def _round_market_ticker_value(value: Any, *, digits: int) -> Any:
    return round_finite(value, digits, on_invalid="passthrough")


def _market_ticker_age_seconds(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return round(max(0.0, numeric), 1)


def _market_ticker_age_display(seconds: Any) -> Optional[str]:
    try:
        numeric = float(seconds)
    except Exception:
        return None
    if numeric < 0.0:
        return f"{format_age_seconds(-numeric)} ahead of wall clock"
    return format_age_seconds(numeric)


def _market_ticker_stale_warning(payload: Dict[str, Any], tick_time: Any) -> str:
    if payload.get("timestamp_in_future"):
        return str(payload.get("timestamp_warning"))
    display_time = payload.get("time") or payload.get("time_display")
    if display_time in (None, ""):
        try:
            display_time = _format_time_second_explicit(float(tick_time))
        except (TypeError, ValueError, OverflowError, OSError):
            display_time = "unavailable"
    return (
        "Tick data may be stale; last tick time is "
        f"{display_time}."
    )


def _market_ticker_points_per_pip(
    symbol_info: Any,
    *,
    symbol: str,
    point: float,
    digits: int,
) -> Optional[float]:
    return forex_points_per_pip(
        symbol,
        path=str(getattr(symbol_info, "path", "") or ""),
        point=point,
        digits=digits,
    )


def _market_depth_fetch_enabled() -> bool:
    raw = os.getenv(_MARKET_DEPTH_ENABLE_ENV)
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _market_depth_level_field(level: Any, *names: str) -> Any:
    for name in names:
        if isinstance(level, dict) and name in level:
            return level.get(name)
        try:
            return level[name]  # type: ignore[index]
        except Exception:
            pass
        try:
            return getattr(level, name)
        except Exception:
            pass
    return None


def _compact_market_ticker_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    skew_active = (
        payload.get("timestamp_ahead_of_wall_clock") is True
        or payload.get("timestamp_in_future") is True
        or str(payload.get("freshness_reason") or "") == "future_timestamp"
    )
    compact_keys = (
        "success",
        "symbol",
        "symbol_input",
        "type",
        "price_currency",
        "bid",
        "ask",
        "mid",
        "spread_valid",
        "spread_quality",
        "market_status",
        "market_status_reason",
        "freshness",
        "freshness_state",
        "freshness_reason",
        "data_stale",
        "usable_for_live_trading",
        "usable_for_live_trading_basis",
        "related_live_symbols",
        "timestamp_warning",
        "warning",
        "quote_source_state",
        "send_path_tick_fresh",
        "send_path_freshness_error",
        "data_age_seconds",
        "quote_as_of",
        "time",
        "timezone",
        "quote_source",
        "source",
    )
    if skew_active:
        compact_keys = compact_keys + (
            "timestamp_ahead_of_wall_clock",
            "timestamp_in_future",
            "timestamp_skew_seconds",
            "timestamp_skew_tolerance_seconds",
        )
    for key in compact_keys:
        if key == "freshness":
            # The full quote builder classifies live/recent/delayed/stale using
            # execution thresholds.  Keep that label rather than reducing it to
            # a binary fresh/stale view in compact output.
            value = payload.get(key)
        elif key == "time":
            value = payload.get("time_display") or payload.get("time")
        else:
            value = payload.get(key)
        if value is not None:
            out[key] = value
    quote_conflict = payload.get("quote_source_conflict")
    if isinstance(quote_conflict, dict):
        conflict_pips = quote_conflict.get("max_disagreement_pips")
        if conflict_pips is not None:
            out["quote_conflict_pips"] = conflict_pips
        cached_pair = quote_conflict.get("symbol_info_tick")
        if isinstance(cached_pair, dict):
            if cached_pair.get("bid") is not None:
                out["alternate_bid"] = cached_pair.get("bid")
            if cached_pair.get("ask") is not None:
                out["alternate_ask"] = cached_pair.get("ask")
    spread_fields = {
        key: payload[key]
        for key in ("spread", "spread_points", "spread_pips", "spread_pct")
        if payload.get(key) is not None
    }
    if spread_fields:
        reordered = {}
        inserted_spread = False
        for key, value in out.items():
            if not inserted_spread and key in {"spread_valid", "spread_quality", "market_status"}:
                reordered.update(spread_fields)
                inserted_spread = True
            reordered[key] = value
        if not inserted_spread:
            reordered.update(spread_fields)
        out = reordered
    if "spread_points" in out and payload.get("point") is not None:
        out["point"] = payload.get("point")
    units = payload.get("units")
    if isinstance(units, dict):
        compact_units = {
            key: units[key]
            for key in (
                "bid",
                "ask",
                "spread",
                "spread_points",
                "spread_pips",
                "spread_pct",
            )
            if key in out and units.get(key) is not None
        }
        if compact_units:
            out["units"] = compact_units
    if out.get("warning") == out.get("timestamp_warning"):
        out.pop("warning", None)
    return out


def _market_ticker_refresh_tick(
    gateway: Any,
    symbol: str,
    tick: Any,
    *,
    now_epoch: float,
) -> tuple[Any, Dict[str, Any]]:
    return resolve_quote_tick(
        gateway,
        symbol,
        tick,
        now_epoch=now_epoch,
        stale_after_seconds=_MARKET_TICKER_STALE_SECONDS,
    )


def _positive_market_ticker_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _format_mt5_error_text(error: Any) -> str:
    if isinstance(error, tuple) and len(error) >= 2:
        code = error[0]
        message = error[1]
        if message not in (None, ""):
            return f"{code}: {message}"
        return str(code)
    return str(error)


def _describe_symbol_select_error(symbol: str, error: Any) -> str:
    text = str(_format_mt5_error_text(error or "")).strip()
    lowered = text.lower()
    if lowered in {"success", "1: success"}:
        return f"Symbol '{symbol}' was not found or is not available in MT5."
    if any(marker in lowered for marker in ("call failed", "unknown symbol", "not found", "invalid symbol")):
        return f"Symbol '{symbol}' was not found or is not available in MT5."
    if text:
        return f"Failed to select symbol {symbol}: {text}"
    return f"Failed to select symbol {symbol}."


def _market_ticker_error(
    message: Any,
    *,
    code: str,
    remediation: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = build_error_payload(
        message,
        code=code,
        operation="market_ticker",
        details=details,
    )
    if remediation:
        payload["remediation"] = remediation
    return payload


def _market_ticker_live_symbol_recovery(
    mt5_gateway: Any,
    symbol: str,
    *,
    default_remediation: str,
) -> tuple[str, Optional[Dict[str, Any]]]:
    related_live_symbols = find_live_extended_session_symbols(mt5_gateway, symbol)
    if not related_live_symbols:
        return default_remediation, None
    live_symbol = related_live_symbols[0]["symbol"]
    return (
        f"A live extended-session contract is available: call market_ticker for "
        f"{live_symbol}, then use that exact symbol for current quotes.",
        {"symbol": symbol, "related_live_symbols": related_live_symbols},
    )


def _market_depth_disabled_payload() -> Dict[str, Any]:
    return {
        "success": False,
        "error": (
            "market_depth_fetch is disabled. "
            f"Set {_MARKET_DEPTH_ENABLE_ENV}=1 to enable it."
        ),
        "error_code": "feature_disabled",
        "feature": "market_depth_fetch",
        "env_var": _MARKET_DEPTH_ENABLE_ENV,
        "enable_instructions": f"Set {_MARKET_DEPTH_ENABLE_ENV}=1 to enable market_depth_fetch.",
        "why_disabled": "Market depth requires broker DOM support and is disabled by default.",
        "recommended_alternative": "market_ticker",
    }


def _market_depth_fetch_impl(symbol: str, spread: bool = False, require_dom: bool = False) -> Dict[str, Any]:  # noqa: C901
    """Return DOM if available; otherwise current bid/ask snapshot for `symbol`.

    Parameters: symbol
    """
    def _run() -> Dict[str, Any]:  # noqa: C901
        mt5_gateway: Any = None
        symbol_input: Optional[str] = None

        def _finalize(payload: Dict[str, Any]) -> Dict[str, Any]:
            if symbol_input is not None:
                payload.setdefault("symbol_input", symbol_input)
            return ensure_common_meta(
                attach_mt5_source(payload, gateway=mt5_gateway),
                tool_name="market_depth_fetch",
            )

        try:
            mt5_gateway = create_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            broker_symbol, symbol_input = resolve_public_symbol(
                symbol,
                gateway=mt5_gateway,
            )
            started = time.perf_counter()
            if not mt5_gateway.symbol_select(broker_symbol, True):
                return {
                    "error": _describe_symbol_select_error(
                        broker_symbol,
                        mt5_gateway.last_error(),
                    )
                }

            symbol_info = mt5_gateway.symbol_info(broker_symbol)
            if symbol_info is None:
                return {"error": f"Symbol {broker_symbol} not found"}

            digits = symbol_price_digits(symbol_info)
            point = symbol_price_point(symbol_info) or 0.0
            tick_size = float(getattr(symbol_info, "trade_tick_size", 0.0) or 0.0)
            trade_tick_value = float(
                getattr(symbol_info, "trade_tick_value", 0.0) or 0.0
            )
            price_currency = symbol_price_currency(symbol_info)
            spread_cost_currency = account_currency_from_gateway(mt5_gateway)

            def _compute_spread_metrics(bid: Any, ask: Any) -> Dict[str, Any] | None:
                metrics = compute_spread_metrics(
                    bid,
                    ask,
                    point=point,
                    tick_size=tick_size,
                    tick_value_money=trade_tick_value,
                    account_currency=spread_cost_currency,
                )
                if metrics["spread"] is None:
                    return None
                return {
                    key: metrics[key]
                    for key in (
                        "spread",
                        "spread_points",
                        "spread_pct",
                        "spread_cost_per_lot",
                        "pricing_basis",
                    )
                }

            def _price_display(value: Any) -> Any:
                if value is None:
                    return None
                try:
                    val = float(value)
                    if digits > 0:
                        return f"{val:.{digits}f}"
                    return str(val)
                except Exception:
                    return None

            def _attach_tick_freshness(
                payload: Dict[str, Any],
                tick_value_obj: Any,
                quote_metadata: Optional[Dict[str, Any]] = None,
            ) -> None:
                quote_now = time.time()
                epoch = tick_epoch(tick_value_obj)
                payload["observed_at"] = _format_time_second_explicit(quote_now)
                payload["observed_at_epoch"] = quote_now
                if quote_metadata:
                    payload.update(quote_metadata)
                context = build_tick_freshness_context(
                    broker_symbol,
                    tick_epoch=epoch,
                    now_epoch=quote_now,
                    item="depth reference tick",
                    age_rounder=lambda value: round(value, 3),
                )
                if context:
                    payload.update(context)
                else:
                    payload.update(
                        {
                            "freshness_state": "unknown",
                            "data_stale": None,
                            "usable_for_live_trading": False,
                            "usable_for_live_trading_basis": (
                                "depth_snapshot_and_reference_tick_freshness"
                            ),
                        }
                    )

                bid = tick_value(tick_value_obj, "bid")
                ask = tick_value(tick_value_obj, "ask")
                enforce_quote_execution_readiness(
                    payload,
                    bid=bid,
                    ask=ask,
                    quote_source_conflict=payload.get("quote_source_conflict"),
                    point=point,
                )
                blocker = None
                if (
                    payload.get("usable_for_live_trading") is False
                    and payload.get("usable_for_live_trading_basis")
                    == QUOTE_EXECUTION_SOURCE_AGREEMENT_BASIS
                ):
                    blocker = "quote_source_conflict"
                elif payload.get("spread_valid") is False:
                    blocker = "reference_quote_not_executable"
                if blocker is not None:
                    blockers = list(payload.get("execution_blockers") or [])
                    if blocker not in blockers:
                        blockers.append(blocker)
                    payload["execution_blockers"] = blockers

            book_subscription_active = False
            try:
                book_subscription_active = bool(
                    mt5_gateway.market_book_add(broker_symbol)
                )
            except Exception:
                book_subscription_active = False
            try:
                depth = None
                attempts = (
                    _MARKET_DEPTH_INITIAL_SNAPSHOT_ATTEMPTS
                    if book_subscription_active
                    else 1
                )
                for attempt in range(attempts):
                    depth = mt5_gateway.market_book_get(broker_symbol)
                    if depth is not None and len(depth) > 0:
                        break
                    if attempt + 1 < attempts:
                        time.sleep(_MARKET_DEPTH_INITIAL_SNAPSHOT_INTERVAL_SECONDS)
            finally:
                if book_subscription_active:
                    try:
                        if not mt5_gateway.market_book_release(broker_symbol):
                            logger.warning(
                                "market_book_release failed for %s",
                                broker_symbol,
                            )
                    except Exception as exc:
                        logger.warning(
                            "market_book_release failed for %s: %s",
                            broker_symbol,
                            exc,
                        )

            if depth is not None and len(depth) > 0:
                buy_orders = []
                sell_orders = []
                unclassified_orders = []

                for level in depth:
                    try:
                        price = float(_market_depth_level_field(level, "price"))
                        volume = float(_market_depth_level_field(level, "volume"))
                        volume_real = float(
                            _market_depth_level_field(
                                level,
                                "volume_real",
                                "volume_dbl",
                            )
                        )
                        level_type = int(_market_depth_level_field(level, "type"))
                    except (KeyError, TypeError, ValueError):
                        continue
                    order_data = {
                        "price": price,
                        "price_display": _price_display(price),
                        "volume": volume,
                        "volume_real": volume_real,
                    }

                    if level_type in {2, 4}:
                        buy_orders.append(order_data)
                    elif level_type in {1, 3}:
                        sell_orders.append(order_data)
                    else:
                        unclassified_orders.append(
                            {**order_data, "book_type": level_type}
                        )

                out = {
                    "success": True,
                    "symbol": broker_symbol,
                    "type": "full_depth",
                    "price_precision": digits,
                    "price_currency": price_currency,
                    "spread_cost_currency": spread_cost_currency,
                    "capabilities": {
                        "dom_available": True,
                        "depth_status": "available",
                        "depth_source": "market_book_get",
                        "spread_overlay_requested": bool(spread),
                    },
                    "data": {
                        "buy_orders": buy_orders,
                        "sell_orders": sell_orders,
                        "depth_levels": {
                            "buy": int(len(buy_orders)),
                            "sell": int(len(sell_orders)),
                            "total": int(len(buy_orders) + len(sell_orders)),
                        },
                        **(
                            {"unclassified_orders": unclassified_orders}
                            if unclassified_orders
                            else {}
                        ),
                    },
                    "units": dict(_MARKET_DEPTH_BOOK_UNITS),
                }

                if unclassified_orders:
                    out["capabilities"]["unclassified_depth_levels"] = int(
                        len(unclassified_orders)
                    )
                    out["warnings"] = [
                        "Depth rows with unknown MT5 book types were not assigned "
                        "to either side."
                    ]
                if spread and buy_orders and sell_orders:
                    valid_buy_prices = [
                        float(row.get("price"))
                        for row in buy_orders
                        if row.get("price") is not None
                    ]
                    valid_sell_prices = [
                        float(row.get("price"))
                        for row in sell_orders
                        if row.get("price") is not None
                    ]
                    if valid_buy_prices and valid_sell_prices:
                        best_bid = max(valid_buy_prices)
                        best_ask = min(valid_sell_prices)
                        spread_metrics = _compute_spread_metrics(best_bid, best_ask)
                        if spread_metrics is not None:
                            out["data"]["best_bid"] = best_bid
                            out["data"]["best_ask"] = best_ask
                            out["data"].update(spread_metrics)
                            out["capabilities"]["spread_overlay_applied"] = True
                            out["units"].update(_MARKET_DEPTH_SPREAD_UNITS)
                depth_tick = mt5_gateway.symbol_info_tick(broker_symbol)
                depth_now = time.time()
                depth_tick, quote_metadata = resolve_quote_tick(
                    mt5_gateway,
                    broker_symbol,
                    depth_tick,
                    now_epoch=depth_now,
                    stale_after_seconds=_MARKET_TICKER_STALE_SECONDS,
                )
                _attach_tick_freshness(out, depth_tick, quote_metadata)
                out["freshness_basis"] = "depth_snapshot_and_reference_tick"
                out["query_latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
                return _finalize(out)

            tick = mt5_gateway.symbol_info_tick(broker_symbol)
            if tick is None:
                return {"error": f"Failed to get tick data for {broker_symbol}"}
            if require_dom:
                return {
                    "error": f"DOM not available for {broker_symbol}. Use market_ticker for bid/ask snapshot instead.",
                    "error_code": "dom_unavailable",
                    "symbol": broker_symbol,
                    "depth_status": "unavailable",
                    "capabilities": {
                        "dom_available": False,
                        "depth_status": "unavailable",
                        "fallback_reason": "market_book_get returned no levels",
                    },
                    "recommended_alternative": "market_ticker",
                }
            quote_now = time.time()
            tick, quote_metadata = resolve_quote_tick(
                mt5_gateway,
                broker_symbol,
                tick,
                now_epoch=quote_now,
                stale_after_seconds=_MARKET_TICKER_STALE_SECONDS,
            )

            out = {
                "success": True,
                "symbol": broker_symbol,
                "type": "quote_fallback",
                "depth_status": "unavailable",
                "price_precision": digits,
                "price_currency": price_currency,
                "spread_cost_currency": spread_cost_currency,
                "recommended_alternative": "market_ticker",
                "capabilities": {
                    "dom_available": False,
                    "depth_status": "unavailable",
                    "depth_source": "symbol_info_tick",
                    "spread_overlay_requested": bool(spread),
                    "fallback_reason": "market_book_get returned no levels",
                },
                "data": {
                    "bid": _optional_finite_float(tick_value(tick, "bid")),
                    "ask": _optional_finite_float(tick_value(tick, "ask")),
                    "last": _positive_market_ticker_float(tick_value(tick, "last")),
                    "volume": int(tick_value(tick, "volume")) if tick_value(tick, "volume") else None,
                    "note": "Full market depth not available, showing current bid/ask snapshot.",
                },
                "units": dict(_MARKET_DEPTH_TICK_UNITS),
            }
            if out["data"]["last"] is None:
                out["data"]["last_unavailable"] = True
            if spread:
                spread_metrics = _compute_spread_metrics(
                    out["data"].get("bid"),
                    out["data"].get("ask"),
                )
                if spread_metrics is not None:
                    out["data"].update(spread_metrics)
                    out["capabilities"]["spread_overlay_applied"] = True
                    out["units"].update(_MARKET_DEPTH_SPREAD_UNITS)
            _use_ctz = _use_client_tz()
            tick_time = tick_epoch(tick)
            if tick_time and _use_ctz:
                out["data"]["time"] = _format_time_second_explicit_local(tick_time)
                out["data"]["time_epoch"] = tick_time
            elif tick_time:
                out["data"]["time"] = _format_time_second_explicit(tick_time)
                out["data"]["time_epoch"] = tick_time
            _attach_tick_freshness(out, tick, quote_metadata)
            out["timezone"] = display_timezone_label(
                use_client_tz=_use_ctz,
                resolve_client_tz=_resolve_client_tz,
            )
            out["query_latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            return _finalize(out)
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error getting market depth: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="market_depth_fetch",
        symbol=symbol,
        spread=spread,
        require_dom=require_dom,
        func=_run,
    )


def market_depth_fetch(symbol: str, spread: bool = False, require_dom: bool = False) -> Dict[str, Any]:
    """Return DOM when `MTDATA_ENABLE_MARKET_DEPTH_FETCH=1`; otherwise disabled.

    When enabled, returns broker DOM if available and otherwise falls back to
    the current bid/ask snapshot unless `require_dom=True`.

    Set `spread=True` to compute and include bid/ask spread metrics from the DOM
    or fallback quote. Returned spread prices use symbol price units; points use
    the broker symbol's point size, and pips are included when applicable.

    Parameters: symbol, spread, require_dom
    """
    if not _market_depth_fetch_enabled():
        return run_logged_operation(
            logger,
            operation="market_depth_fetch",
            symbol=symbol,
            spread=spread,
            require_dom=require_dom,
            func=_market_depth_disabled_payload,
        )
    return _market_depth_fetch_impl(symbol, spread=spread, require_dom=require_dom)


market_depth_fetch = mcp.tool()(market_depth_fetch)


@mcp.tool()
def market_ticker(  # noqa: C901
    symbol: str,
    detail: DetailLiteral = "compact",
    price_field: Optional[Literal["bid", "ask", "mid", "last", "spread"]] = None,
) -> Dict[str, Any]:
    """Return a lightweight quote snapshot with bid/ask/spread for `symbol`.

    Parameters: symbol
    Use `detail="compact"` to keep only the most operational bid/ask/spread fields.
    Set `price_field` to bid, ask, mid, last, or spread for a simple price result.
    """
    detail_mode = normalize_output_verbosity_detail(detail, default="compact")

    def _run() -> Dict[str, Any]:  # noqa: C901
        restore_symbol: Optional[str] = None
        mt5_gateway: Any = None

        def _finalize(payload: Dict[str, Any]) -> Dict[str, Any]:
            return ensure_common_meta(
                attach_mt5_source(payload, gateway=mt5_gateway),
                tool_name="market_ticker",
            )

        try:
            mt5_gateway = create_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            resolved_symbol = resolve_broker_symbol_name(symbol, gateway=mt5_gateway)
            started = time.perf_counter()
            info_before = mt5_gateway.symbol_info(resolved_symbol)
            was_visible = (
                bool(getattr(info_before, "visible", True))
                if info_before is not None
                else None
            )
            if not mt5_gateway.symbol_select(resolved_symbol, True):
                suggestions = symbol_suggestions_from_gateway(mt5_gateway, symbol)
                return _finalize(
                    _market_ticker_error(
                        _describe_symbol_select_error(
                            resolved_symbol, mt5_gateway.last_error()
                        ),
                        code="symbol_not_found",
                        remediation=(
                            f"Verify the broker symbol name with symbols_list(search_term='{symbol}') "
                            "or symbols_top_markets, then retry with an available MT5 symbol."
                        ),
                        details={"symbol": symbol, "did_you_mean": suggestions},
                    )
                )
            if was_visible is False:
                restore_symbol = resolved_symbol
                _ensure_symbol_ready(resolved_symbol)

            symbol_info = mt5_gateway.symbol_info(resolved_symbol)
            if symbol_info is None:
                suggestions = symbol_suggestions_from_gateway(mt5_gateway, symbol)
                return _finalize(
                    _market_ticker_error(
                        f"Symbol {resolved_symbol} not found",
                        code="symbol_not_found",
                        remediation=(
                            f"Verify the broker symbol name with symbols_list(search_term='{symbol}') "
                            "or symbols_describe."
                        ),
                        details={"symbol": symbol, "did_you_mean": suggestions},
                    )
                )

            tick = mt5_gateway.symbol_info_tick(resolved_symbol)
            if tick is None:
                remediation, details = _market_ticker_live_symbol_recovery(
                    mt5_gateway,
                    resolved_symbol,
                    default_remediation=(
                        "Ensure the symbol is visible in Market Watch and that the market is open "
                        "or recent ticks are available from the broker."
                    ),
                )
                return _finalize(
                    _market_ticker_error(
                        f"Failed to get tick data for {resolved_symbol}",
                        code="market_ticker_tick_unavailable",
                        remediation=remediation,
                        details=details,
                    )
                )

            quote_now_epoch = float(time.time())
            tick, quote_source_metadata = _market_ticker_refresh_tick(
                mt5_gateway,
                resolved_symbol,
                tick,
                now_epoch=quote_now_epoch,
            )

            digits = symbol_price_digits(symbol_info)
            point = symbol_price_point(symbol_info) or 0.0
            tick_size = float(getattr(symbol_info, "trade_tick_size", 0.0) or 0.0)
            tick_value_money = float(
                getattr(symbol_info, "trade_tick_value", 0.0) or 0.0
            )
            price_currency = symbol_price_currency(symbol_info)
            spread_cost_currency = account_currency_from_gateway(mt5_gateway)
            contract_size = _positive_market_ticker_float(
                getattr(symbol_info, "trade_contract_size", None)
            )

            bid_raw = tick_value(tick, "bid")
            ask_raw = tick_value(tick, "ask")
            last_raw = tick_value(tick, "last")
            bid = _optional_finite_float(bid_raw)
            ask = _optional_finite_float(ask_raw)
            last = _positive_market_ticker_float(last_raw)
            tick_time = tick_epoch(tick)
            if tick_time is None and not any(
                value not in (None, 0.0) for value in (bid, ask, last)
            ):
                remediation, details = _market_ticker_live_symbol_recovery(
                    mt5_gateway,
                    resolved_symbol,
                    default_remediation=(
                        "Verify the broker symbol name and ensure the symbol has a recent tick "
                        "in Market Watch."
                    ),
                )
                return _finalize(
                    _market_ticker_error(
                        (
                            f"No usable quote data for {resolved_symbol}. "
                            f"Use symbols_list(search_term='{symbol}') to find broker-specific names and suffixes."
                        ),
                        code="market_ticker_quote_unavailable",
                        remediation=remediation,
                        details=details,
                    )
                )
            tick_volume = tick_value(tick, "volume")
            if tick_volume is not None:
                try:
                    tick_volume = int(tick_volume)
                except (TypeError, ValueError, OverflowError):
                    tick_volume = None

            points_per_pip = _market_ticker_points_per_pip(
                symbol_info,
                symbol=resolved_symbol,
                point=point,
                digits=digits,
            )
            spread_metrics = compute_spread_metrics(
                bid,
                ask,
                point=point,
                points_per_pip=points_per_pip,
                tick_size=tick_size,
                tick_value_money=tick_value_money,
                account_currency=spread_cost_currency,
            )
            mid = spread_metrics["mid"]
            spread_abs = _round_market_ticker_value(
                spread_metrics["spread"], digits=digits
            )
            spread_points = _round_market_ticker_value(
                spread_metrics["spread_points"], digits=4
            )
            spread_pips = _round_market_ticker_value(
                spread_metrics["spread_pips"], digits=4
            )
            spread_pct = _round_market_ticker_value(
                spread_metrics["spread_pct"], digits=6
            )
            spread_cost_per_lot = _round_market_ticker_value(
                spread_metrics["spread_cost_per_lot"], digits=6
            )
            spread_valid = spread_metrics["spread_valid"]
            spread_quality = spread_metrics["spread_quality"]
            pricing_basis = spread_metrics["pricing_basis"]

            _use_ctz = _use_client_tz()

            out: Dict[str, Any] = {
                "success": True,
                "symbol": resolved_symbol,
                "type": "quote",
                "price_precision": digits,
                "bid_ask_precision": digits,
                "mid_precision": digits + 1,
                "point": point if point > 0 else None,
                "price_currency": price_currency,
                "bid": _round_market_ticker_value(bid, digits=digits),
                "ask": _round_market_ticker_value(ask, digits=digits),
                "mid": _round_market_ticker_value(mid, digits=digits + 1),
                "last": _round_market_ticker_value(last, digits=digits),
                "spread": spread_abs,
                "spread_points": spread_points,
                "spread_pips": spread_pips,
                "spread_pct": spread_pct,
                "spread_cost_per_lot": spread_cost_per_lot,
                "spread_valid": spread_valid,
                "spread_quality": spread_quality,
                "pricing_basis": pricing_basis,
                "units": {
                    "bid": "absolute_price",
                    "ask": "absolute_price",
                    "mid": "absolute_price",
                    "last": "absolute_price",
                    "point": "price_increment",
                    "spread": "absolute_price",
                    "spread_points": UNIT_BROKER_POINTS,
                    "spread_pct": UNIT_PERCENT,
                    "spread_cost_per_lot": "currency_per_lot_estimate",
                },
            }
            if last is None:
                out["last_unavailable"] = True
            out.update(quote_source_metadata)
            quote_conflict = out.get("quote_source_conflict")
            if isinstance(quote_conflict, dict) and point > 0:
                cached_pair = quote_conflict.get("symbol_info_tick")
                stream_pair = quote_conflict.get("stream_tick")
                if isinstance(cached_pair, dict) and isinstance(stream_pair, dict):
                    disagreements = []
                    for side in ("bid", "ask"):
                        try:
                            disagreements.append(
                                abs(float(cached_pair[side]) - float(stream_pair[side]))
                            )
                        except (KeyError, TypeError, ValueError):
                            continue
                    if disagreements:
                        max_disagreement_points = max(disagreements) / point
                        quote_conflict["max_disagreement_points"] = round(
                            max_disagreement_points, 4
                        )
                        if points_per_pip:
                            quote_conflict["max_disagreement_pips"] = round(
                                max_disagreement_points / points_per_pip, 4
                            )
            time_normalization = describe_mt5_time_normalization(
                symbol=resolved_symbol
            )
            out.update(time_normalization)
            if contract_size is not None:
                out["contract_size"] = _round_market_ticker_value(contract_size, digits=6)
                out["lot_definition"] = "1 broker lot equals contract_size contract units."
                out["units"]["contract_size"] = "contract_units_per_lot"
                out["units"]["lot"] = "broker_lot"
                if pricing_basis == "per_1_lot_estimate":
                    out["pricing_basis_units"] = "broker_lot"
            if spread_pips is not None:
                out["units"]["spread_pips"] = UNIT_PIPS
            if tick_volume not in (None, 0):
                out["tick_volume"] = tick_volume
            if spread_cost_per_lot is not None and spread_cost_currency:
                out["spread_cost_currency"] = spread_cost_currency
            if tick_time is not None:
                out["time_epoch"] = float(tick_time)
                out["quote_as_of"] = _format_time_second_explicit(float(tick_time))
                if _use_ctz:
                    out["time"] = _format_time_second_explicit_local(float(tick_time))
                else:
                    out["time"] = _format_time_second_explicit(float(tick_time))
            age_seconds = None
            now_epoch = None
            if tick_time is not None:
                try:
                    now_epoch = quote_now_epoch
                    age_seconds = max(0.0, now_epoch - float(tick_time))
                except Exception:
                    age_seconds = None
            if age_seconds is not None:
                freshness_context = build_tick_freshness_context(
                    resolved_symbol,
                    tick_epoch=tick_time,
                    now_epoch=now_epoch,
                    item="tick",
                    stale_after_seconds=_MARKET_TICKER_STALE_SECONDS,
                    age_rounder=_market_ticker_age_seconds,
                )
                rounded_age_seconds = freshness_context.get("data_age_seconds")
                out.update(freshness_context)
                age_display = _market_ticker_age_display(rounded_age_seconds)
                if age_display is not None:
                    out["data_age"] = age_display
                if out["data_stale"]:
                    out["warning"] = _market_ticker_stale_warning(out, tick_time)
            enforce_quote_execution_readiness(
                out,
                bid=bid,
                ask=ask,
                quote_source_conflict=out.get("quote_source_conflict"),
                point=point,
            )
            if out.get("usable_for_live_trading") is False or out.get("data_stale") is True:
                related_live_symbols = find_live_extended_session_symbols(
                    mt5_gateway,
                    resolved_symbol,
                )
                if related_live_symbols:
                    out["related_live_symbols"] = related_live_symbols
            diagnostics = {
                "source": out.get("quote_source", "mt5.symbol_info_tick"),
                "cache_used": False,
                "data_freshness_seconds": _market_ticker_age_seconds(age_seconds),
                "data_freshness_anchor": FRESHNESS_ANCHOR_WALL_CLOCK,
                "data_freshness_metric": FRESHNESS_METRIC_LAST_TICK_AGE,
                "timestamp_mode": time_normalization.get("timestamp_mode"),
                "time_normalization": time_normalization.get("time_normalization"),
                "query_latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
            meta = out.get("meta")
            if not isinstance(meta, dict):
                meta = {}
            meta["diagnostics"] = dict(diagnostics)
            out["meta"] = meta
            out["timezone"] = display_timezone_label(
                use_client_tz=_use_ctz,
                resolve_client_tz=_resolve_client_tz,
            )
            if price_field is not None:
                field_value = str(price_field or "").strip().lower()
                price_values = {
                    "bid": out.get("bid"),
                    "ask": out.get("ask"),
                    "mid": out.get("mid"),
                    "last": out.get("last"),
                    "spread": out.get("spread"),
                }
                if field_value not in price_values:
                    return _finalize(
                        _market_ticker_error(
                            "price_field must be one of: bid, ask, mid, last, spread.",
                            code="market_ticker_invalid_price_field",
                            remediation="Choose one of bid, ask, mid, last, or spread.",
                        )
                    )
                price = price_values.get(field_value)
                if price is None:
                    return _finalize(
                        _market_ticker_error(
                            f"{field_value} price is unavailable for {resolved_symbol}.",
                            code="market_ticker_price_unavailable",
                            remediation=(
                                "Use bid, ask, mid, or spread when the broker does not publish "
                                "the requested price field for this symbol."
                            ),
                        )
                    )
                quote_usable = out.get("usable_for_live_trading") is True
                selected_executable = field_value in {"bid", "ask"}
                selected_precision = digits + 1 if field_value == "mid" else digits
                simple: Dict[str, Any] = {
                    "success": True,
                    "symbol": resolved_symbol,
                    "type": "spread" if field_value == "spread" else "price",
                    "field": field_value,
                    "price": price,
                    "price_precision": selected_precision,
                    "price_currency": price_currency,
                    "price_currency_basis": "quote_currency_not_cash_cost",
                    "unit": out["units"].get(field_value, "absolute_price"),
                    "point": out.get("point"),
                    "price_basis": field_value,
                    "quote_usable_for_live_trading": quote_usable,
                    "selected_value_executable": bool(
                        selected_executable and quote_usable
                    ),
                    "units": {
                        "price": out["units"].get(field_value, "absolute_price"),
                        "point": out["units"]["point"],
                    },
                }
                if selected_executable:
                    simple["execution_side"] = field_value
                if field_value == "spread":
                    for key in ("spread_points", "spread_pips", "spread_pct"):
                        if out.get(key) is not None:
                            simple[key] = out[key]
                            if key in out["units"]:
                                simple["units"][key] = out["units"][key]
                copy_keys = [
                    "quote_as_of",
                    "time",
                    "time_epoch",
                    "timezone",
                    "data_age_seconds",
                    "data_age_anchor",
                    "data_age_metric",
                    "data_age",
                    "stale_after_seconds",
                    "data_stale",
                    "freshness_basis",
                    "freshness",
                    "freshness_state",
                    "freshness_reason",
                    "related_live_symbols",
                    "live_max_age_seconds",
                    "timestamp_ahead_of_wall_clock",
                    "timestamp_in_future",
                    "timestamp_skew_seconds",
                    "timestamp_skew_tolerance_seconds",
                    "timestamp_warning",
                    "market_status",
                    "market_status_reason",
                    "market_status_source",
                    "freshness_policy_relaxed",
                    "note",
                    "warning",
                    "quote_source",
                    "quote_source_state",
                    "quote_refresh_attempted",
                    "send_path_tick_fresh",
                    "send_path_freshness_error",
                    "spread_valid",
                    "spread_quality",
                    "usable_for_live_trading",
                    "usable_for_live_trading_basis",
                ]
                for key in copy_keys:
                    if out.get(key) is not None:
                        simple[key] = out.get(key)
                return _finalize(simple)
            if resolved_symbol != str(symbol or "").strip():
                out["symbol_input"] = symbol
            if detail_mode == "compact":
                out = _compact_market_ticker_payload(out)
            return _finalize(out)
        except MT5ConnectionError as exc:
            return _finalize(
                _market_ticker_error(
                    str(exc),
                    code="market_ticker_mt5_connection",
                    remediation="Ensure MetaTrader 5 is running, logged in, and reachable.",
                )
            )
        except Exception as exc:
            return _finalize(
                _market_ticker_error(
                    f"Error getting quote snapshot: {str(exc)}",
                    code="market_ticker_error",
                    remediation="Retry after checking the MT5 terminal and symbol availability.",
                )
            )
        finally:
            if restore_symbol is not None:
                try:
                    mt5.symbol_select(restore_symbol, False)
                except Exception:
                    pass

    return run_logged_operation(
        logger,
        operation="market_ticker",
        symbol=symbol,
        detail=detail_mode,
        price_field=price_field,
        func=_run,
    )
