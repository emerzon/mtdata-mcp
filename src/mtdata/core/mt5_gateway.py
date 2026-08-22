"""Shared MT5 gateway helpers for core entrypoints."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ..utils.mt5 import MT5ConnectionError, ensure_mt5_connection_or_raise, mt5_adapter
from .error_envelope import build_error_payload
from .execution_logging import (
    log_operation_exception,
    log_operation_finish,
    log_operation_start,
)

logger = logging.getLogger(__name__)

_MT5_CONSTANT_NAMES = frozenset(
    {
        "ORDER_TIME_GTC",
        "ORDER_TIME_SPECIFIED",
        "ORDER_FILLING_FOK",
        "ORDER_FILLING_IOC",
        "ORDER_FILLING_RETURN",
        "ORDER_TYPE_BUY",
        "ORDER_TYPE_BUY_LIMIT",
        "ORDER_TYPE_BUY_STOP",
        "ORDER_TYPE_BUY_STOP_LIMIT",
        "ORDER_TYPE_SELL",
        "ORDER_TYPE_SELL_LIMIT",
        "ORDER_TYPE_SELL_STOP",
        "ORDER_TYPE_SELL_STOP_LIMIT",
        "POSITION_TYPE_BUY",
        "POSITION_TYPE_SELL",
        "DEAL_ENTRY_IN",
        "DEAL_ENTRY_INOUT",
        "DEAL_ENTRY_OUT",
        "DEAL_ENTRY_OUT_BY",
        "DEAL_REASON_CLIENT",
        "DEAL_REASON_SL",
        "DEAL_REASON_TP",
        "DEAL_TYPE_BUY",
        "DEAL_TYPE_SELL",
        "COPY_TICKS_ALL",
        "SYMBOL_FILLING_FOK",
        "SYMBOL_FILLING_IOC",
        "SYMBOL_FILLING_RETURN",
        "SYMBOL_TRADE_MODE_CLOSEONLY",
        "SYMBOL_TRADE_MODE_DISABLED",
        "SYMBOL_TRADE_MODE_FULL",
        "SYMBOL_TRADE_MODE_LONGONLY",
        "SYMBOL_TRADE_MODE_SHORTONLY",
        "TIMEFRAME_M1",
        "TRADE_ACTION_DEAL",
        "TRADE_ACTION_MODIFY",
        "TRADE_ACTION_PENDING",
        "TRADE_ACTION_REMOVE",
        "TRADE_ACTION_SLTP",
        "TRADE_RETCODE_DONE",
        "TRADE_RETCODE_PRICE_CHANGED",
    }
)


@dataclass
class MT5Gateway:
    """Thin adapter boundary around MT5 access plus connection enforcement."""

    adapter: Any
    ensure_connection_impl: Callable[[], None]

    def ensure_connection(self) -> None:
        started_at = time.perf_counter()
        adapter_type = type(self.adapter).__name__
        log_operation_start(
            logger,
            operation="mt5_ensure_connection",
            adapter_type=adapter_type,
        )
        try:
            self.ensure_connection_impl()
        except Exception as exc:
            log_operation_exception(
                logger,
                operation="mt5_ensure_connection",
                started_at=started_at,
                exc=exc,
                adapter_type=adapter_type,
            )
            raise
        log_operation_finish(
            logger,
            operation="mt5_ensure_connection",
            started_at=started_at,
            success=True,
            adapter_type=adapter_type,
        )

    def _adapter_attr(self, name: str) -> Any:
        return getattr(self.adapter, name)

    def __getattr__(self, name: str) -> Any:
        if name in _MT5_CONSTANT_NAMES:
            return self._adapter_attr(name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def account_info(self) -> Any:
        return self.adapter.account_info()

    def copy_rates_from(self, symbol: str, timeframe: Any, dt_from: Any, count: int) -> Any:
        return self.adapter.copy_rates_from(symbol, timeframe, dt_from, count)

    def copy_rates_from_pos(self, symbol: str, timeframe: Any, start_pos: int, count: int) -> Any:
        return self.adapter.copy_rates_from_pos(symbol, timeframe, start_pos, count)

    def copy_rates_range(self, symbol: str, timeframe: Any, dt_from: Any, dt_to: Any) -> Any:
        return self.adapter.copy_rates_range(symbol, timeframe, dt_from, dt_to)

    def copy_ticks_from(self, symbol: str, dt_from: Any, count: int, flags: Any) -> Any:
        return self.adapter.copy_ticks_from(symbol, dt_from, count, flags)

    def copy_ticks_range(self, symbol: str, dt_from: Any, dt_to: Any, flags: Any) -> Any:
        return self.adapter.copy_ticks_range(symbol, dt_from, dt_to, flags)

    def history_deals_get(self, dt_from: Any, dt_to: Any, **kwargs: Any) -> Any:
        return self.adapter.history_deals_get(dt_from, dt_to, **kwargs)

    def history_orders_get(self, dt_from: Any, dt_to: Any, **kwargs: Any) -> Any:
        return self.adapter.history_orders_get(dt_from, dt_to, **kwargs)

    def last_error(self) -> Any:
        return self.adapter.last_error()

    def market_book_add(self, symbol: str) -> Any:
        return self.adapter.market_book_add(symbol)

    def market_book_get(self, symbol: str) -> Any:
        return self.adapter.market_book_get(symbol)

    def market_book_release(self, symbol: str) -> Any:
        return self.adapter.market_book_release(symbol)

    def order_send(self, request: Any) -> Any:
        return self.adapter.order_send(request)

    def order_calc_margin(self, action: Any, symbol: str, volume: float, price: float) -> Any:
        return self.adapter.order_calc_margin(action, symbol, volume, price)

    def order_calc_profit(
        self,
        action: Any,
        symbol: str,
        volume: float,
        price_open: float,
        price_close: float,
    ) -> Any:
        return self.adapter.order_calc_profit(action, symbol, volume, price_open, price_close)

    def orders_get(self, **kwargs: Any) -> Any:
        return self.adapter.orders_get(**kwargs)

    def positions_get(self, **kwargs: Any) -> Any:
        return self.adapter.positions_get(**kwargs)

    def retcode_name(self, code: Any) -> Any:
        return self.adapter.retcode_name(code)

    def symbol_info(self, symbol: str) -> Any:
        return self.adapter.symbol_info(symbol)

    def symbol_info_tick(self, symbol: str) -> Any:
        return self.adapter.symbol_info_tick(symbol)

    def symbol_select(self, symbol: str, visible: bool = True) -> Any:
        return self.adapter.symbol_select(symbol, visible)

    def symbols_get(self, *args: Any, **kwargs: Any) -> Any:
        return self.adapter.symbols_get(*args, **kwargs)

    def terminal_info(self) -> Any:
        return self.adapter.terminal_info()


def create_mt5_gateway(
    *,
    adapter: Any = mt5_adapter,
    ensure_connection_impl: Callable[[], None] = ensure_mt5_connection_or_raise,
) -> MT5Gateway:
    return MT5Gateway(
        adapter=adapter,
        ensure_connection_impl=ensure_connection_impl,
    )


def mt5_connection_error(
    gateway: Optional[Any] = None,
    *,
    operation: str = "mt5_ensure_connection",
) -> Optional[Dict[str, Any]]:
    try:
        mt5_gateway = gateway if gateway is not None else create_mt5_gateway()
        mt5_gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return build_error_payload(
            str(exc),
            code="mt5_connection_error",
            operation=operation,
        )
    return None
