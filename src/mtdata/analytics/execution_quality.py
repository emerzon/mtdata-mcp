"""Statistical engines for the advanced MT5-native analytics tools."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.analytics_requests import (
    TradeExecutionQualityRequest,
)
from ..core.error_envelope import invalid_minutes_back_payload
from ..core.trading.validation import snapshot_unavailable_error
from ..shared.symbols import (
    is_probably_crypto_symbol,
    is_probably_fx_session_symbol,
)
from ..utils.freshness import (
    format_age_seconds,
)
from ..utils.sessions import market_session_label, session_definition_for_clock
from ..utils.time import MAX_TRADING_MINUTES_BACK, format_epoch_utc
from ..utils.utils import (
    validate_historical_range,
)
from .engine_common import (
    _analysis_window_metadata,
    _bootstrap_mean_ci,
    _mapping,
    _percentiles,
    _round_execution_stat,
    _tick_frame,
    _window,
)

logger = logging.getLogger(__name__)


def _execution_percentiles(values: Iterable[float]) -> Dict[str, Optional[float]]:
    return {
        key: _round_execution_stat(value)
        for key, value in _percentiles(values).items()
    }


def _metric_family_has_observations(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        observations = value.get("observations")
        if observations is not None:
            try:
                return int(observations) > 0
            except (TypeError, ValueError):
                return False
        stat_keys = ("mean", "median", "p90", "p95", "p99", "max")
        if any(key in value for key in stat_keys):
            return any(value.get(key) is not None for key in stat_keys)
        nested = [
            child
            for child in value.values()
            if isinstance(child, dict)
        ]
        if nested:
            return any(_metric_family_has_observations(child) for child in nested)
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return any(_metric_family_has_observations(item) for item in value)
    return True



def _execution_duration_display(
    stats: Dict[str, Optional[float]],
) -> Dict[str, str]:
    """Format millisecond duration statistics for quick human inspection."""
    out: Dict[str, str] = {}
    for key, value in stats.items():
        if value is None:
            continue
        milliseconds = max(0.0, float(value))
        if milliseconds < 1000.0:
            display = f"{int(round(milliseconds))}ms"
        elif milliseconds < 60_000.0:
            seconds = milliseconds / 1000.0
            display = f"{seconds:.2f}".rstrip("0").rstrip(".") + "s"
        else:
            display = format_age_seconds(milliseconds / 1000.0)
        if display is not None:
            out[str(key)] = display
    return out



def _execution_bootstrap_mean_ci(
    values: Sequence[float],
    samples: int,
) -> Optional[List[float]]:
    interval = _bootstrap_mean_ci(values, samples)
    if interval is None:
        return None
    return [_round_execution_stat(value) for value in interval]



class _ExecutionTickCache:
    """Cache bounded quote chunks so nearby fills share broker reads."""

    _CHUNK_SECONDS = 300

    def __init__(self, gateway: Any) -> None:
        self.gateway = gateway
        self.frames: Dict[Tuple[str, int], pd.DataFrame] = {}
        self.queries = 0
        self.cache_hits = 0
        self.truncated_chunks = 0

    def get(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        first_chunk = (
            int(math.floor(start.timestamp() / self._CHUNK_SECONDS))
            * self._CHUNK_SECONDS
        )
        last_chunk = (
            int(math.floor(end.timestamp() / self._CHUNK_SECONDS))
            * self._CHUNK_SECONDS
        )
        frames: List[pd.DataFrame] = []
        for chunk_start in range(
            first_chunk,
            last_chunk + self._CHUNK_SECONDS,
            self._CHUNK_SECONDS,
        ):
            key = (symbol.casefold(), chunk_start)
            frame = self.frames.get(key)
            if frame is None:
                chunk_start_dt = datetime.fromtimestamp(chunk_start, tz=timezone.utc)
                chunk_end_dt = chunk_start_dt + timedelta(
                    seconds=self._CHUNK_SECONDS
                )
                frame, truncated = _tick_frame(
                    self.gateway,
                    symbol,
                    chunk_start_dt,
                    chunk_end_dt,
                    50_000,
                )
                self.frames[key] = frame
                self.queries += 1
                self.truncated_chunks += int(truncated)
            else:
                self.cache_hits += 1
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        if combined.empty:
            return combined
        dedupe_columns = [
            column
            for column in (
                "epoch",
                "bid",
                "ask",
                "last",
                "volume",
                "volume_real",
                "flags",
            )
            if column in combined.columns
        ]
        combined = combined.drop_duplicates(subset=dedupe_columns, keep="last")
        return combined[
            (combined["epoch"] >= start.timestamp())
            & (combined["epoch"] <= end.timestamp())
        ].sort_values("epoch", kind="stable")

    def metadata(self) -> Dict[str, Any]:
        return {
            "strategy": "symbol_5_minute_chunk_cache",
            "broker_queries": self.queries,
            "chunks_loaded": len(self.frames),
            "cache_hits": self.cache_hits,
            "truncated_chunks": self.truncated_chunks,
        }



def _deal_side(row: Dict[str, Any], gateway: Any) -> Optional[str]:
    value = row.get("type")
    text = str(value).lower()
    if text in {"buy", "0", str(getattr(gateway, "DEAL_TYPE_BUY", 0))}:
        return "buy"
    if text in {"sell", "1", str(getattr(gateway, "DEAL_TYPE_SELL", 1))}:
        return "sell"
    return None



def _order_type_label(value: Any, gateway: Any) -> str:
    order_types = (
        ("ORDER_TYPE_BUY", 0),
        ("ORDER_TYPE_SELL", 1),
        ("ORDER_TYPE_BUY_LIMIT", 2),
        ("ORDER_TYPE_SELL_LIMIT", 3),
        ("ORDER_TYPE_BUY_STOP", 4),
        ("ORDER_TYPE_SELL_STOP", 5),
        ("ORDER_TYPE_BUY_STOP_LIMIT", 6),
        ("ORDER_TYPE_SELL_STOP_LIMIT", 7),
        ("ORDER_TYPE_CLOSE_BY", 8),
    )
    for name, fallback in order_types:
        code = getattr(gateway, name, fallback)
        if value == code or str(value) == str(code):
            return name.removeprefix("ORDER_TYPE_")
    return "UNKNOWN"



def _execution_symbol_catalog(gateway: Any) -> Dict[str, Dict[str, Any]]:
    try:
        raw_symbols = list(gateway.symbols_get() or [])
    except Exception:
        return {}
    catalog: Dict[str, Dict[str, Any]] = {}
    for item in raw_symbols:
        row = _mapping(item)
        name = str(row.get("name") or getattr(item, "name", "") or "").strip()
        if name:
            catalog[name.casefold()] = {
                "name": name,
                "path": str(row.get("path") or getattr(item, "path", "") or ""),
            }
    return catalog



def _execution_session_calendar(
    symbol: str,
    *,
    gateway: Any,
    catalog: Dict[str, Dict[str, Any]],
) -> tuple[str, Optional[str]]:
    metadata = catalog.get(str(symbol).casefold(), {})
    path = str(metadata.get("path") or "")
    if not path:
        try:
            info = gateway.symbol_info(symbol)
        except Exception:
            info = None
        path = str(getattr(info, "path", "") or "")
    if is_probably_crypto_symbol(symbol):
        return "continuous_24_7", path or None
    if is_probably_fx_session_symbol(symbol, path=path):
        return "fx", path or None
    return "utc_hour_only", path or None



def _execution_session_definition(calendar: str) -> Dict[str, Any]:
    if calendar == "fx":
        return session_definition_for_clock("UTC", "fx")
    if calendar == "continuous_24_7":
        return {
            "basis": "continuous_market",
            "calendar": "continuous_24_7",
            "clock": "UTC",
            "continuous": "All UTC hours; no off-session bucket is applied.",
        }
    return {
        "basis": "utc_hour_only",
        "calendar": "utc_hour_only",
        "clock": "UTC",
        "note": (
            "No reliable venue calendar was available; use by_hour_utc and do "
            "not interpret named geographic sessions."
        ),
    }



def _execution_contract_size(symbol: str, gateway: Any) -> Optional[float]:
    try:
        info = gateway.symbol_info(symbol)
        size = float(getattr(info, "trade_contract_size", 0.0) or 0.0)
    except Exception:
        return None
    if not math.isfinite(size) or size <= 0.0:
        return None
    return size


def _execution_fill_notional(
    *,
    volume: float,
    price: float,
    contract_size: Optional[float],
) -> Optional[float]:
    if contract_size is None or volume <= 0.0 or price <= 0.0:
        return None
    notional = volume * contract_size * price
    if not math.isfinite(notional) or notional <= 0.0:
        return None
    return notional


def analyze_execution_quality(  # noqa: C901
    request: TradeExecutionQualityRequest, gateway: Any
) -> Dict[str, Any]:
    range_error = validate_historical_range(request.start, request.end)
    if range_error is not None:
        return range_error
    if int(request.minutes_back) > MAX_TRADING_MINUTES_BACK:
        return invalid_minutes_back_payload(
            request.minutes_back,
            operation="trade_execution_quality",
            max_minutes_back=MAX_TRADING_MINUTES_BACK,
        )
    try:
        start, end = _window(request.start, request.end, request.minutes_back)
    except ValueError as exc:
        message = str(exc)
        if "minutes_back" in message:
            return invalid_minutes_back_payload(
                request.minutes_back,
                operation="trade_execution_quality",
                max_minutes_back=MAX_TRADING_MINUTES_BACK,
                reason=message,
            )
        raise
    analysis_window = _analysis_window_metadata(request, start, end)
    account_currency = None
    account_info = getattr(gateway, "account_info", None)
    if callable(account_info):
        try:
            account_currency = str(
                getattr(account_info(), "currency", "") or ""
            ).strip() or None
        except Exception:
            account_currency = None
    symbol_catalog = _execution_symbol_catalog(gateway)
    resolved_symbol = None
    if request.symbol:
        from ..utils.mt5 import resolve_broker_symbol_name

        resolved_symbol = resolve_broker_symbol_name(request.symbol, gateway=gateway)
        exact = symbol_catalog.get(str(resolved_symbol).casefold())
        if symbol_catalog and exact is None:
            return {
                "error": f"Symbol {request.symbol!r} was not found by MT5.",
                "error_code": "symbol_not_found",
                "symbol": request.symbol,
                "remediation": (
                    "Use symbols_list to discover the exact broker symbol name."
                ),
            }
        if exact is not None:
            resolved_symbol = str(exact["name"])
        elif not symbol_catalog:
            try:
                symbol_info = gateway.symbol_info(resolved_symbol)
            except Exception:
                symbol_info = None
            if symbol_info is None:
                return {
                    "error": f"Symbol {request.symbol!r} was not found by MT5.",
                    "error_code": "symbol_not_found",
                    "symbol": request.symbol,
                    "remediation": (
                        "Use symbols_list to discover the exact broker symbol name."
                    ),
                }
    kwargs = {"group": resolved_symbol} if resolved_symbol else {}
    deal_rows = gateway.history_deals_get(start, end, **kwargs)
    if deal_rows is None:
        return snapshot_unavailable_error(
            gateway, snapshot="history_deals", context="analyze execution quality"
        )
    raw_deals = [_mapping(row) for row in deal_rows]
    order_rows = gateway.history_orders_get(start, end, **kwargs)
    if order_rows is None:
        return snapshot_unavailable_error(
            gateway, snapshot="history_orders", context="analyze execution quality"
        )
    raw_orders = [_mapping(row) for row in order_rows]
    if resolved_symbol:
        deals = [
            row
            for row in raw_deals
            if str(row.get("symbol") or "").casefold()
            == resolved_symbol.casefold()
        ]
        orders = [
            row
            for row in raw_orders
            if not str(row.get("symbol") or "").strip()
            or str(row.get("symbol") or "").casefold()
            == resolved_symbol.casefold()
        ]
    else:
        deals = raw_deals
        orders = raw_orders
    order_by_ticket = {int(row.get("ticket") or 0): row for row in orders if row.get("ticket")}
    fills = []
    skipped = {
        "non_trade": 0,
        "filter": 0,
        "unbenchmarked": 0,
        "missing_markout": 0,
        "future_timestamp": 0,
    }
    eligible_deals = []
    for deal in deals:
        side = _deal_side(deal, gateway)
        volume = float(deal.get("volume") or 0.0)
        symbol = str(deal.get("symbol") or "").strip()
        if side is None or volume <= 0 or not symbol:
            skipped["non_trade"] += 1
            continue
        if request.side and side != request.side:
            skipped["filter"] += 1
            continue
        if request.magic is not None and int(deal.get("magic") or 0) != int(request.magic):
            skipped["filter"] += 1
            continue
        eligible_deals.append(deal)

    eligible_deals.sort(
        key=lambda row: (
            float(row.get("time_msc") or 0),
            int(row.get("ticket") or 0),
        ),
        reverse=True,
    )
    benchmark_sources = {
        "arrival_quote": 0,
        "pending_order_price": 0,
        "order_price": 0,
        "order_price_fallback": 0,
    }
    arrival_quote_observations = 0
    processed_candidates = 0
    tick_cache = _ExecutionTickCache(gateway)
    contract_size_by_symbol: Dict[str, Optional[float]] = {}
    observed_epoch = datetime.now(timezone.utc).timestamp()
    future_tolerance_seconds = 300.0
    order_fill_totals: Dict[Any, Dict[str, float]] = {}
    order_completion_deals = 0
    for deal in eligible_deals:
        fill_epoch = (
            float(deal.get("time_msc") or 0) / 1000.0
            or float(deal.get("time") or 0)
        )
        if fill_epoch > observed_epoch + future_tolerance_seconds:
            continue
        order_ticket = int(deal.get("order") or 0)
        order = order_by_ticket.get(order_ticket)
        if not order:
            continue
        try:
            volume = float(deal.get("volume") or 0.0)
            initial_volume = float(order.get("volume_initial") or 0.0)
        except (TypeError, ValueError):
            continue
        if volume <= 0.0 or initial_volume <= 0.0:
            continue
        state = order_fill_totals.setdefault(
            order_ticket,
            {"filled_volume": 0.0, "initial_volume": initial_volume},
        )
        state["filled_volume"] += volume
        state["initial_volume"] = max(state["initial_volume"], initial_volume)
        order_completion_deals += 1
    for deal in eligible_deals:
        processed_candidates += 1
        side = _deal_side(deal, gateway)
        volume = float(deal.get("volume") or 0.0)
        symbol = str(deal.get("symbol") or "").strip()
        order = order_by_ticket.get(int(deal.get("order") or 0), {})
        fill_epoch = float(deal.get("time_msc") or 0) / 1000.0 or float(deal.get("time") or 0)
        if fill_epoch > observed_epoch + future_tolerance_seconds:
            skipped["future_timestamp"] += 1
            continue
        time_setup_msc = float(order.get("time_setup_msc") or 0.0)
        if not time_setup_msc and order.get("time_setup"):
            time_setup_msc = float(order["time_setup"]) * 1000.0
        setup_epoch = time_setup_msc / 1000.0 if time_setup_msc else None
        qstart = datetime.fromtimestamp(fill_epoch - request.quote_window_seconds, tz=timezone.utc)
        qend = datetime.fromtimestamp(fill_epoch + max(request.markout_seconds) + 5, tz=timezone.utc)
        ticks = tick_cache.get(symbol, qstart, qend)
        before = ticks[(ticks["epoch"] <= fill_epoch) & np.isfinite(ticks["mid"])]
        fill_time_quote = None
        if len(before):
            fill_tick = before.iloc[-1]
            fill_time_quote = float(fill_tick["ask"] if side == "buy" else fill_tick["bid"])
        order_type_value = order.get("type")
        order_type_label = _order_type_label(order_type_value, gateway)
        market_order_types = {
            getattr(gateway, "ORDER_TYPE_BUY", 0),
            getattr(gateway, "ORDER_TYPE_SELL", 1),
        }
        # Historical gateways can omit the originating order type. In that
        # ambiguous case, retain the conservative market-fill benchmark rather
        # than silently treating the deal as a pending-order fill.
        is_market_order = (
            order_type_value is None or order_type_value in market_order_types
        )
        order_price = float(
            order.get("price_open") or order.get("price_current") or 0.0
        )
        arrival_quote = None
        arrival_quote_epoch = None
        benchmark_price = None
        benchmark_epoch = None
        benchmark_source = None
        if request.benchmark == "arrival_quote" and setup_epoch is not None:
            arrival_start = datetime.fromtimestamp(
                setup_epoch - request.quote_window_seconds,
                tz=timezone.utc,
            )
            arrival_end = datetime.fromtimestamp(setup_epoch, tz=timezone.utc)
            arrival_ticks = tick_cache.get(symbol, arrival_start, arrival_end)
            arrival_before = arrival_ticks[
                (arrival_ticks["epoch"] <= setup_epoch)
                & np.isfinite(arrival_ticks["mid"])
            ]
            if len(arrival_before):
                latest = arrival_before.iloc[-1]
                arrival_quote = float(
                    latest["ask"] if side == "buy" else latest["bid"]
                )
                arrival_quote_epoch = float(latest["epoch"])
                arrival_quote_observations += 1
        if request.benchmark == "order_price":
            if order_price > 0:
                benchmark_price = order_price
                benchmark_source = "order_price"
        elif not is_market_order:
            if order_price > 0:
                benchmark_price = order_price
                benchmark_source = "pending_order_price"
        elif arrival_quote and arrival_quote > 0:
            benchmark_price = arrival_quote
            benchmark_epoch = arrival_quote_epoch
            benchmark_source = "arrival_quote"
        elif request.benchmark_fallback == "order_price" and order_price > 0:
            benchmark_price = order_price
            benchmark_source = "order_price_fallback"
        fill_price = float(deal.get("price") or 0.0)
        if not benchmark_price or fill_price <= 0:
            skipped["unbenchmarked"] += 1
            continue
        benchmark_sources[str(benchmark_source)] += 1
        sign = 1.0 if side == "buy" else -1.0
        slippage_bps = (
            sign * (fill_price - benchmark_price) / benchmark_price * 10_000.0
        )
        pending_arrival_shortfall_bps = (
            sign * (fill_price - arrival_quote) / arrival_quote * 10_000.0
            if not is_market_order and arrival_quote and arrival_quote > 0
            else None
        )
        markouts: Dict[str, Optional[float]] = {}
        for horizon in request.markout_seconds:
            candidates = ticks[(ticks["epoch"] >= fill_epoch + horizon) & (ticks["epoch"] <= fill_epoch + horizon + 5) & np.isfinite(ticks["mid"])]
            if len(candidates):
                markouts[str(horizon)] = float(sign * (float(candidates.iloc[0]["mid"]) - fill_price) / fill_price * 10_000.0)
            else:
                markouts[str(horizon)] = None
                skipped["missing_markout"] += 1
        initial_volume = float(order.get("volume_initial") or volume)
        order_to_fill_duration_ms = (
            max(
                0.0,
                float(deal.get("time_msc") or fill_epoch * 1000.0)
                - time_setup_msc,
            )
            if time_setup_msc
            else None
        )
        item = {
            "deal_ticket": deal.get("ticket"),
            "order_ticket": deal.get("order"),
            "position_id": deal.get("position_id"),
            "symbol": symbol,
            "side": side,
            "volume": volume,
            "fill_price": fill_price,
            "benchmark_price": benchmark_price,
            "benchmark_source": benchmark_source,
            "benchmark_epoch": benchmark_epoch,
            "benchmark_time": (
                format_epoch_utc(benchmark_epoch)
                if benchmark_epoch is not None
                else None
            ),
            "fill_time_quote": fill_time_quote,
            "slippage_bps": slippage_bps,
            "price_improved": slippage_bps < 0,
            **(
                {
                    "arrival_quote_price": arrival_quote,
                    "arrival_quote_epoch": arrival_quote_epoch,
                    "arrival_quote_time": format_epoch_utc(arrival_quote_epoch),
                    "arrival_implementation_shortfall_bps": (
                        pending_arrival_shortfall_bps
                    ),
                }
                if pending_arrival_shortfall_bps is not None
                and arrival_quote_epoch is not None
                else {}
            ),
            "order_to_fill_duration_ms": order_to_fill_duration_ms,
            "fill_timing_basis": (
                "market_fill_latency" if is_market_order else "pending_time_to_fill"
            ),
            "is_market_order": is_market_order,
            "deal_fill_ratio": min(1.0, volume / initial_volume) if initial_volume > 0 else None,
            "commission": float(deal.get("commission") or 0.0),
            "fee": float(deal.get("fee") or 0.0),
            "commission_fee_cash": max(
                0.0,
                -(
                    float(deal.get("commission") or 0.0)
                    + float(deal.get("fee") or 0.0)
                ),
            ),
            "commission_fee_per_lot": max(
                0.0,
                -(
                    float(deal.get("commission") or 0.0)
                    + float(deal.get("fee") or 0.0)
                ),
            )
            / volume,
            "markout_bps": markouts,
            "fill_epoch": fill_epoch,
            "order_type": order_type_label,
            "order_type_code": order_type_value,
            "hour_utc": datetime.fromtimestamp(fill_epoch, tz=timezone.utc).hour,
        }
        if symbol not in contract_size_by_symbol:
            contract_size_by_symbol[symbol] = _execution_contract_size(
                symbol, gateway
            )
        contract_size = contract_size_by_symbol[symbol]
        if contract_size is not None:
            item["contract_size"] = contract_size
        notional = _execution_fill_notional(
            volume=volume,
            price=fill_price,
            contract_size=contract_size,
        )
        if notional is not None:
            item["notional"] = notional
            cash_fee = float(item["commission_fee_cash"])
            item["commission_fee_bps"] = cash_fee / notional * 10_000.0
        session_calendar, symbol_path = _execution_session_calendar(
            symbol,
            gateway=gateway,
            catalog=symbol_catalog,
        )
        item["session_calendar"] = session_calendar
        item["session"] = None
        if symbol_path:
            item["symbol_path"] = symbol_path
        if session_calendar == "fx":
            item["session"] = market_session_label(
                datetime.fromtimestamp(fill_epoch, tz=timezone.utc),
                session_calendar="fx",
            )
        elif session_calendar == "continuous_24_7":
            item["session"] = "continuous"
        try:
            action = getattr(gateway, "ORDER_TYPE_BUY", 0) if side == "buy" else getattr(gateway, "ORDER_TYPE_SELL", 1)
            shortfall = gateway.order_calc_profit(
                action, symbol, volume, benchmark_price, fill_price
            )
            if shortfall is not None:
                item["execution_shortfall_currency_estimate"] = float(shortfall)
        except Exception as exc:
            item["execution_shortfall_error"] = str(exc)
            logger.warning("execution shortfall estimate failed: %s", exc)
        fills.append(item)
        if len(fills) >= request.limit:
            break
    fills.sort(
        key=lambda item: (
            float(item.get("fill_epoch") or 0),
            int(item.get("deal_ticket") or 0),
        )
    )
    market_order_fills = [item for item in fills if item.get("is_market_order")]
    non_market_order_fills = [item for item in fills if not item.get("is_market_order")]
    arrival_quote_market_fills = [
        item
        for item in market_order_fills
        if item.get("benchmark_source") == "arrival_quote"
    ]
    order_price_market_fills = [
        item
        for item in market_order_fills
        if item.get("benchmark_source") in {"order_price", "order_price_fallback"}
    ]
    market_slippages = [
        float(item["slippage_bps"]) for item in market_order_fills
    ]
    market_vs_arrival_slippages = [
        float(item["slippage_bps"]) for item in arrival_quote_market_fills
    ]
    market_vs_order_slippages = [
        float(item["slippage_bps"]) for item in order_price_market_fills
    ]
    pending_slippages = [
        float(item["slippage_bps"]) for item in non_market_order_fills
    ]
    pending_arrival_shortfalls = [
        float(item["arrival_implementation_shortfall_bps"])
        for item in non_market_order_fills
        if item.get("arrival_implementation_shortfall_bps") is not None
    ]
    if not fills:
        headline_fills = []
        slippage_basis = "not_applicable"
    elif request.benchmark == "order_price":
        headline_fills = fills
        slippage_basis = "explicit_order_price_all_fills"
    elif market_order_fills:
        headline_fills = market_order_fills
        slippage_basis = (
            "market_arrival_quote_with_order_price_fallback"
            if any(
                item.get("benchmark_source") == "order_price_fallback"
                for item in market_order_fills
            )
            else "market_arrival_quote"
        )
    else:
        headline_fills = non_market_order_fills
        slippage_basis = "pending_order_price_no_market_fills"
    headline_slippages = [
        float(item["slippage_bps"]) for item in headline_fills
    ]
    partial_orders = sum(
        state["initial_volume"] > 0.0
        and state["filled_volume"] < state["initial_volume"] * 0.999
        for state in order_fill_totals.values()
    )
    summary = {
        "fills": len(fills),
        "orders": len({item["order_ticket"] for item in fills}),
        "market_order_fills": len(market_order_fills),
        "non_market_order_fills": len(non_market_order_fills),
        "slippage_basis": slippage_basis,
        "slippage_bps": _execution_percentiles(headline_slippages),
        "market_fill_slippage_bps": _execution_percentiles(market_slippages),
        "market_fill_vs_arrival_quote_bps": _execution_percentiles(
            market_vs_arrival_slippages
        ),
        "market_fill_vs_order_price_bps": _execution_percentiles(
            market_vs_order_slippages
        ),
        "pending_fill_vs_order_bps": _execution_percentiles(pending_slippages),
        "pending_arrival_implementation_shortfall_bps": _execution_percentiles(
            pending_arrival_shortfalls
        ),
        "mean_slippage_ci_95": _execution_bootstrap_mean_ci(
            headline_slippages, 500
        ),
        "price_improvement_pct": _round_execution_stat(
            100.0 * np.mean([item["price_improved"] for item in headline_fills])
        ) if headline_fills else None,
        "partial_fill_pct": _round_execution_stat(
            100.0 * (partial_orders / len(order_fill_totals))
        ) if order_fill_totals else None,
        "partial_orders": int(partial_orders),
        "orders_evaluated_for_partial_fills": len(order_fill_totals),
        "partial_fill_deals_evaluated": order_completion_deals,
        "partial_fill_rate_basis": "all_eligible_deals_in_requested_window",
        "market_fill_latency_ms": _execution_percentiles(
            item["order_to_fill_duration_ms"]
            for item in market_order_fills
            if item.get("order_to_fill_duration_ms") is not None
        ),
        "pending_time_to_fill_ms": _execution_percentiles(
            item["order_to_fill_duration_ms"]
            for item in non_market_order_fills
            if item.get("order_to_fill_duration_ms") is not None
        ),
        "order_to_fill_duration_ms": _execution_percentiles(
            item["order_to_fill_duration_ms"]
            for item in fills
            if item.get("order_to_fill_duration_ms") is not None
        ),
        "commission_fee": _execution_percentiles(
            item["commission_fee_cash"] for item in fills
        ),
        "total_commission_fee": _round_execution_stat(
            sum(float(item.get("commission_fee_cash") or 0.0) for item in fills)
        ),
    }
    fill_symbols = sorted(
        {str(item.get("symbol")) for item in fills if item.get("symbol")}
    )
    mixed_contract_lots = len(fill_symbols) > 1
    if not mixed_contract_lots:
        summary["commission_fee_per_lot"] = _execution_percentiles(
            item["commission_fee_per_lot"] for item in fills
        )
    fee_bps_values = [
        float(item["commission_fee_bps"])
        for item in fills
        if item.get("commission_fee_bps") is not None
    ]
    if fee_bps_values:
        summary["commission_fee_bps"] = _execution_percentiles(fee_bps_values)
    duration_display = {
        name.removesuffix("_ms"): display
        for name in ("pending_time_to_fill_ms", "order_to_fill_duration_ms")
        if (
            display := _execution_duration_display(
                summary.get(name) if isinstance(summary.get(name), dict) else {}
            )
        )
    }
    if duration_display:
        summary["duration_display"] = duration_display
    insufficient_markout_horizons: List[str] = []
    for horizon in request.markout_seconds:
        horizon_key = str(horizon)
        values = [
            float(value)
            for item in fills
            if (value := item["markout_bps"].get(horizon_key)) is not None
        ]
        observations = len(values)
        missing = max(0, len(fills) - observations)
        markout_summary = _execution_percentiles(values)
        markout_summary.update(
            {
                "observations": observations,
                "missing": missing,
                "coverage_pct": _round_execution_stat(
                    observations / len(fills) * 100.0
                )
                if fills
                else 0.0,
                "minimum": request.min_sample,
                "sample_status": (
                    "not_applicable"
                    if not fills
                    else "ok"
                    if observations >= request.min_sample
                    else "insufficient"
                ),
            }
        )
        summary.setdefault("markout_bps", {})[horizon_key] = markout_summary
        if fills and observations < request.min_sample:
            insufficient_markout_horizons.append(horizon_key)
    breakdowns: Dict[str, List[Dict[str, Any]]] = {}
    if fills:
        fill_frame = pd.DataFrame(fills)
        for keys, label in ((["symbol", "side"], "by_symbol_side"), (["order_type"], "by_order_type"), (["session_calendar", "session"], "by_session"), (["hour_utc"], "by_hour_utc")):
            breakdowns[label] = []
            source_frame = fill_frame
            if label == "by_session":
                source_frame = fill_frame[fill_frame["session"].notna()]
            for group_key, items in source_frame.groupby(keys):
                labels = group_key if isinstance(group_key, tuple) else (group_key,)
                row = {name: value for name, value in zip(keys, labels)}
                row.update({"fills": len(items), "slippage_bps": _execution_percentiles(items["slippage_bps"])})
                if label == "by_symbol_side":
                    row["commission_fee_per_lot"] = _execution_percentiles(
                        items["commission_fee_per_lot"]
                    )
                    row["commission_fee"] = _execution_percentiles(
                        items["commission_fee_cash"]
                    )
                    row["total_commission_fee"] = _round_execution_stat(
                        float(items["commission_fee_cash"].sum())
                    )
                if label == "by_order_type":
                    codes = [
                        value
                        for value in items["order_type_code"].dropna().unique().tolist()
                    ]
                    if len(codes) == 1:
                        row["order_type_code"] = codes[0]
                    row["order_to_fill_duration_ms"] = _execution_percentiles(
                        items["order_to_fill_duration_ms"]
                    )
                breakdowns[label].append(row)
        breakdowns["by_symbol"] = []
        for symbol_name, items in fill_frame.groupby("symbol"):
            row = {
                "symbol": symbol_name,
                "fills": len(items),
                "commission_fee_per_lot": _execution_percentiles(
                    items["commission_fee_per_lot"]
                ),
                "commission_fee": _execution_percentiles(
                    items["commission_fee_cash"]
                ),
                "total_commission_fee": _round_execution_stat(
                    float(items["commission_fee_cash"].sum())
                ),
            }
            if "contract_size" in items:
                sizes = [
                    value
                    for value in items["contract_size"].dropna().unique().tolist()
                ]
                if len(sizes) == 1:
                    row["contract_size"] = sizes[0]
            breakdowns["by_symbol"].append(row)
    sample_start = format_epoch_utc(fills[0]["fill_epoch"]) if fills else None
    sample_end = format_epoch_utc(fills[-1]["fill_epoch"]) if fills else None
    benchmark_attempts = max(
        0, processed_candidates - skipped["future_timestamp"]
    )
    fallback_count = benchmark_sources["order_price_fallback"]
    warnings = []
    if fallback_count:
        warnings.append(
            f"{fallback_count} fill(s) used order price because no arrival quote was available."
        )
    if non_market_order_fills:
        warnings.append(
            "pending_time_to_fill_ms measures intentional limit/stop order wait, not "
            "broker execution latency; order_to_fill_duration_ms is a mixed duration."
        )
        if request.benchmark == "arrival_quote":
            warnings.append(
                "Pending fills use their order price for fill-quality slippage; "
                "setup-to-fill market movement is reported separately as "
                "pending_arrival_implementation_shortfall_bps."
            )
    if market_order_fills and non_market_order_fills and request.benchmark == "arrival_quote":
        warnings.append(
            "Headline slippage_bps and price_improvement_pct use market-order fills "
            "only; pending fill quality is reported separately."
        )
    if skipped["future_timestamp"]:
        warnings.append(
            f"Skipped {skipped['future_timestamp']} fill(s) whose broker timestamp "
            "was more than 5 minutes ahead of the observation clock."
        )
    if insufficient_markout_horizons:
        warnings.append(
            "Markout evidence is below min_sample for horizon(s) "
            + ", ".join(f"{horizon}s" for horizon in insufficient_markout_horizons)
            + "; descriptive statistics are retained but marked insufficient."
        )
    if mixed_contract_lots:
        warnings.append(
            "Account-wide commission_fee_per_lot is omitted because broker lots "
            "are not comparable across symbols; per-lot fees are reported in "
            "breakdowns.by_symbol and account-wide fees use cash and, when "
            "notional is available, basis points."
        )
    session_calendars = sorted(
        {
            str(item.get("session_calendar"))
            for item in fills
            if item.get("session_calendar")
        }
    )
    if not session_calendars and resolved_symbol:
        fallback_calendar, _ = _execution_session_calendar(
            resolved_symbol,
            gateway=gateway,
            catalog=symbol_catalog,
        )
        session_calendars = [fallback_calendar]
    session_definitions = {
        calendar: _execution_session_definition(calendar)
        for calendar in session_calendars
    }
    if "utc_hour_only" in session_calendars:
        warnings.append(
            "One or more symbols have no reliable venue calendar; use by_hour_utc for those fills."
        )
    if tick_cache.truncated_chunks:
        warnings.append(
            f"{tick_cache.truncated_chunks} cached quote chunk(s) exceeded 50000 "
            "ticks; quote coverage within those chunks is incomplete."
        )
    eligible_symbols = sorted(
        {
            str(item.get("symbol"))
            for item in eligible_deals
            if item.get("symbol")
        }
    )
    analyzed_symbols = sorted(
        {str(item.get("symbol")) for item in fills if item.get("symbol")}
    )
    filters_applied: Dict[str, Any] = {}
    if request.side is not None:
        filters_applied["side"] = request.side
    if request.magic is not None:
        filters_applied.update(
            {
                "magic": int(request.magic),
                "magic_exact": str(request.magic),
            }
        )
    fill_sample_quality = {
        "status": (
            "not_applicable"
            if not fills
            else "ok"
            if len(fills) >= request.min_sample
            else "insufficient"
        ),
        "minimum": request.min_sample,
        "observed": len(fills),
        "scope": "matched_fills_for_fill_level_metrics",
    }
    truncated = processed_candidates < len(eligible_deals)
    if truncated:
        warnings.append(
            "Headline execution metrics use the latest "
            f"{len(fills)} matched fill(s) of {len(eligible_deals)} eligible "
            "trade deals in the requested window; they are not full-window "
            "aggregates. Raise --limit to analyze more fills. Partial-fill "
            "metrics are the exception and use all eligible deals in the window."
        )
    sample = {
        "selection_order": "latest_first",
        "display_order": "chronological",
        "total_eligible": len(eligible_deals),
        "matched_fills": len(fills),
        "sample_start": sample_start,
        "sample_end": sample_end,
        "truncated": truncated,
        "limit": request.limit,
    }
    summary_scope = (
        f"latest_{len(fills)}_of_{len(eligible_deals)}"
        if truncated
        else "requested_window"
    )
    effective_analysis_window = {
        "start": sample_start,
        "end": sample_end,
        "timezone": "UTC",
        "scope": summary_scope,
        "selection_order": "latest_first",
    }
    benchmark_quality = {
        "requested": request.benchmark,
        "fallback_policy": request.benchmark_fallback,
        "source_counts": benchmark_sources,
        "fallback_count": fallback_count,
        "arrival_quote_coverage": (
            arrival_quote_observations / benchmark_attempts
            if request.benchmark == "arrival_quote" and benchmark_attempts
            else 0.0
            if request.benchmark == "arrival_quote"
            else None
        ),
    }
    common = {
        "success": True,
        **(
            {
                "symbol_filter": {
                    "requested": request.symbol,
                    "resolved": resolved_symbol,
                    "match_mode": "exact",
                }
            }
            if request.symbol
            else {}
        ),
        **({"currency": account_currency} if account_currency else {}),
        "window": analysis_window,
        "effective_analysis_window": effective_analysis_window,
        "summary_scope": summary_scope,
        "filters_applied": filters_applied,
    }
    price_quality_definition = {
        "slippage_bps": slippage_basis,
        "market_fill_slippage_bps": (
            "not_applicable"
            if not fills
            else "market_fill_vs_submitted_order_price"
            if request.benchmark == "order_price"
            else "market_fill_vs_arrival_executable_quote"
        ),
        "market_fill_vs_arrival_quote_bps": (
            "not_applicable"
            if not arrival_quote_market_fills
            else "market_fill_vs_arrival_executable_quote"
        ),
        "market_fill_vs_order_price_bps": (
            "not_applicable"
            if not order_price_market_fills
            else "market_fill_vs_submitted_order_price"
        ),
        "pending_fill_vs_order_bps": (
            "not_applicable"
            if not non_market_order_fills
            else "pending_fill_vs_submitted_order_price"
        ),
        "pending_arrival_implementation_shortfall_bps": (
            "not_applicable"
            if not pending_arrival_shortfalls
            else "pending_fill_vs_order_setup_executable_quote_not_broker_slippage"
        ),
        "markout_bps": "post_fill_midpoint_markout_positive_is_favorable",
    }
    execution_units = {
        "slippage_bps": "basis_points_positive_is_worse",
        "market_fill_slippage_bps": "basis_points_positive_is_worse",
        "market_fill_vs_arrival_quote_bps": "basis_points_positive_is_worse",
        "market_fill_vs_order_price_bps": "basis_points_positive_is_worse",
        "pending_fill_vs_order_bps": "basis_points_positive_is_worse",
        "pending_arrival_implementation_shortfall_bps": (
            "basis_points_positive_is_worse"
        ),
        "markout_bps": "basis_points_positive_is_favorable",
        "market_fill_latency_ms": "milliseconds",
        "pending_time_to_fill_ms": "milliseconds",
        "order_to_fill_duration_ms": "milliseconds",
        "commission": "account_currency",
        "fee": "account_currency",
        "commission_fee": "account_currency",
        "total_commission_fee": "account_currency",
        "commission_fee_per_lot": "account_currency_per_broker_lot",
        "price_improvement_pct": "percent_0_to_100",
        "partial_fill_pct": "percent_0_to_100",
        "commission_fee_bps": "basis_points_of_notional",
        "notional": "account_currency",
        "contract_size": "contract_units_per_broker_lot",
        "execution_shortfall_currency_estimate": (
            "account_currency_positive_is_worse"
        ),
    }
    if not fills:
        empty_message = "No matching fills in the requested window"
        if request.symbol:
            empty_message += f" for {request.symbol}"
        if minutes_label := (
            f" in the last {int(request.minutes_back)} minute(s)"
            if not request.start and not request.end
            else ""
        ):
            empty_message += minutes_label
        empty_message += "."
        common["empty"] = True
        common["status"] = "no_matching_fills"
        common["message"] = empty_message
    if request.detail not in {"compact", "summary"}:
        common["requested_window"] = analysis_window
    if request.detail in {"compact", "summary"}:
        compact_summary_keys = (
            "fills",
            "orders",
            "market_order_fills",
            "non_market_order_fills",
            "slippage_basis",
            "slippage_bps",
            "market_fill_vs_arrival_quote_bps",
            "market_fill_vs_order_price_bps",
            "price_improvement_pct",
            "partial_fill_pct",
            "market_fill_latency_ms",
            "pending_time_to_fill_ms",
            "commission_fee_per_lot",
            "commission_fee",
            "total_commission_fee",
            "commission_fee_bps",
            "markout_bps",
        )
        compact_summary: Dict[str, Any] = {}
        omitted_metrics: List[str] = []
        count_keys = {
            "fills",
            "orders",
            "market_order_fills",
            "non_market_order_fills",
            "slippage_basis",
            "price_improvement_pct",
            "partial_fill_pct",
        }
        for key in compact_summary_keys:
            if key not in summary:
                continue
            value = summary[key]
            if key in count_keys or _metric_family_has_observations(value):
                compact_summary[key] = value
            else:
                omitted_metrics.append(key)
        compact_out = {
            **common,
            "summary": compact_summary,
            "fill_sample_quality": fill_sample_quality,
            "data_quality": {
                "eligible_symbol_count": len(eligible_symbols),
                "analyzed_symbol_count": len(analyzed_symbols),
                "eligible_trade_deals": len(eligible_deals),
                "processed_candidates": processed_candidates,
                "matched_fills": len(fills),
                "skipped": skipped,
                "benchmark": {
                    "requested": benchmark_quality.get("requested"),
                    "fallback_count": benchmark_quality.get("fallback_count"),
                    "arrival_quote_coverage": benchmark_quality.get(
                        "arrival_quote_coverage"
                    ),
                },
            },
            "sample": {
                "selection_order": sample["selection_order"],
                "total_eligible": sample["total_eligible"],
                "matched_fills": sample["matched_fills"],
                "truncated": sample["truncated"],
                "limit": sample["limit"],
                "sample_start": sample["sample_start"],
                "sample_end": sample["sample_end"],
            },
            "warnings": warnings,
        }
        compact_units = {
            key: execution_units[key]
            for key in compact_summary
            if key in execution_units
        }
        if compact_units:
            compact_out["units"] = compact_units
        compact_price_definitions = {
            key: price_quality_definition[key]
            for key in compact_summary
            if key in price_quality_definition
        }
        if compact_price_definitions:
            compact_out["price_quality_definition"] = compact_price_definitions
        if omitted_metrics:
            compact_out["omitted_metrics"] = omitted_metrics
        return compact_out
    return {
        **common,
        "summary": summary,
        "breakdowns": breakdowns,
        **({"items": fills} if request.detail == "full" else {}),
        "fill_sample_quality": fill_sample_quality,
        "data_quality": {
            "history_deals": len(deals),
            "history_orders": len(orders),
            "history_deals_before_exact_filter": len(raw_deals),
            "history_orders_before_exact_filter": len(raw_orders),
            "eligible_symbols": eligible_symbols,
            "analyzed_symbols": analyzed_symbols,
            "eligible_trade_deals": len(eligible_deals),
            "processed_candidates": processed_candidates,
            "matched_fills": len(fills),
            "skipped": skipped,
            "benchmark": benchmark_quality,
            "quote_reads": tick_cache.metadata(),
            **(
                {"session_definition": next(iter(session_definitions.values()))}
                if len(session_definitions) == 1
                else {"session_definitions": session_definitions}
            ),
        },
        "sample": sample,
        "timing_definition": {
            "market_fill_latency_ms": "market_order_setup_to_fill_elapsed_time",
            "pending_time_to_fill_ms": "pending_order_setup_to_fill_wait_duration_not_execution_latency",
            "order_to_fill_duration_ms": "all_order_setup_to_fill_mixed_duration_not_execution_latency",
        },
        "price_quality_definition": price_quality_definition,
        "units": execution_units,
        "warnings": warnings,
    }
