"""Trading functions for MetaTrader integration."""

import logging

from .._mcp_instance import mcp
from ..error_envelope import new_request_id
from ..runtime_metadata import run_mt5_logged_operation
from . import time, validation
from .account import (
    lookup_trade_ticket_history,
    trade_account_info,
    trade_history,
    trade_journal_analyze,
)
from .context import trade_session_context
from .execution import (
    _cancel_pending,
    _close_positions,
    _modify_pending_order,
    _modify_position,
    _resolve_close_dry_run_target,
)
from .gateway import create_trading_gateway, resolve_trading_symbol_request
from .ideas import trade_idea_compose
from .orders import (
    _place_market_order,
    _place_pending_order,
    build_trade_place_dry_run_preview,
)
from .positions import trade_get_open, trade_get_pending
from .requests import TradeCloseRequest, TradeModifyRequest, TradePlaceRequest
from .risk import trade_risk_analyze, trade_stress_test, trade_var_cvar_calculate
from .use_cases import run_trade_close, run_trade_modify, run_trade_place

logger = logging.getLogger(__name__)


@mcp.tool()
def trade_place(request: TradePlaceRequest) -> dict:
    """Place a market or pending order.

    Defaults to preview mode. Set `dry_run=false` explicitly to send an order.
    Required inputs: symbol, volume, order_type.
    - BUY/SELL: market orders; omit `price`.
    - BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP: pending (requires `price`).
    - BUY_STOP_LIMIT/SELL_STOP_LIMIT: pending; `price` is the trigger and
      `stop_limit_price` is the limit order price activated by that trigger.
    - dry_run: validate routing and preview the order without sending it to MT5.
      Use `detail="compact"|"standard"|"full"` to control preview depth.
    - require_sl_tp: for market orders, require both SL and TP inputs before order
      submission. Requested SL/TP levels are sent atomically with the order.
      Defaults to True for safer automation behavior.
    - If a filled market order is reported without the requested TP/SL
      protection, the tool always attempts to close it defensively.
    - Environment guardrails can block orders before MT5 submission based on
      configured symbol policies, volume caps, or wallet/account risk limits.
    - idempotency_key: optional durable dedupe key with a configurable 24-hour
      retention window. The SQLite store is shared across restarts and workers.
      Reusing a key with the same payload replays the prior outcome instead of
      sending another order; changed payloads require a new key.
    - Responses include a correlation_id shared with execution logs. Idempotent
      replays also identify the original invocation.
    """
    correlation_id = new_request_id()

    def _run() -> dict:
        gateway = create_trading_gateway()
        resolved_request, symbol_input = resolve_trading_symbol_request(
            request,
            gateway,
        )
        result = run_trade_place(
            resolved_request,
            normalize_order_type_input=validation._normalize_order_type_input,
            normalize_pending_expiration=time._normalize_pending_expiration,
            prevalidate_trade_place_market_input=validation._prevalidate_trade_place_market_input,
            place_market_order=_place_market_order,
            place_pending_order=_place_pending_order,
            close_positions=_close_positions,
            safe_int_ticket=validation._safe_int_ticket,
            build_dry_run_preview=build_trade_place_dry_run_preview,
            correlation_id=correlation_id,
        )
        if isinstance(result, dict):
            result = dict(result)
            result["symbol"] = resolved_request.symbol
            if symbol_input is not None:
                result["symbol_input"] = symbol_input
        return result

    return run_mt5_logged_operation(
        logger,
        operation="trade_place",
        correlation_id=correlation_id,
        symbol=request.symbol,
        order_type=request.order_type,
        volume=request.volume,
        func=_run,
    )


@mcp.tool()
def trade_modify(request: TradeModifyRequest) -> dict:
    """Modify an open position or pending order by ticket.

    Supply at least one of price, stop_limit_price, stop_loss, take_profit,
    expiration, clear_stop_loss, or clear_take_profit. MT5 cannot retag an
    existing ticket; set the comment when placing or closing instead.
    Defaults to preview mode. Set `dry_run=false` explicitly to send a live
    modify request.
    Risk-increasing pending-order and position-protection changes can be
    blocked by configured trade guardrails, while close/reduce flows remain
    allowed.
    Optional idempotency_key values suppress duplicate retries for the same
    payload using a durable SQLite store with a configurable 24-hour retention
    window. This does not provide broker-side idempotency.
    Responses include a correlation_id shared with execution logs.
    """
    correlation_id = new_request_id()
    return run_mt5_logged_operation(
        logger,
        operation="trade_modify",
        correlation_id=correlation_id,
        ticket=request.ticket,
        func=lambda: run_trade_modify(
            request,
            normalize_pending_expiration=time._normalize_pending_expiration,
            modify_pending_order=_modify_pending_order,
            modify_position=_modify_position,
            correlation_id=correlation_id,
        ),
    )


@mcp.tool()
def trade_close(request: TradeCloseRequest) -> dict:
    """Close positions, cancel pending orders, or flatten both object classes.

    `target=positions` (default) never cancels pending orders.
    `target=pending` only cancels pending orders.
    `target=all_exposure` closes positions and cancels pending orders as separate
    legs; it is valid only for a bulk symbol, magic, or account scope.
    `magic` is a standalone bulk selector. Use `close_all=true` only to select
    the whole account when ticket, symbol, and magic are omitted.
    Set `volume` only to partially close a specific open position by ticket.
    `volume` is invalid without `ticket`.
    Defaults to preview mode. Set `dry_run=false` explicitly to send a live
    close/cancel request.
    Optional `idempotency_key` values durably suppress duplicate retries for
    the same close/cancel payload, including ambiguous broker responses.
    Responses include a correlation_id shared with execution logs.
    """
    correlation_id = new_request_id()
    return run_mt5_logged_operation(
        logger,
        operation="trade_close",
        correlation_id=correlation_id,
        ticket=request.ticket,
        symbol=request.symbol,
        target=request.target,
        func=lambda: run_trade_close(
            request,
            close_positions=_close_positions,
            cancel_pending=_cancel_pending,
            lookup_ticket_history=lookup_trade_ticket_history,
            resolve_close_target=_resolve_close_dry_run_target,
            correlation_id=correlation_id,
        ),
    )
