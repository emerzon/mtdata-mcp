"""Trading risk analysis."""

from __future__ import annotations

import logging

from mtdata.shared.runtime_metadata import run_mt5_logged_operation

from ...utils.mt5 import ensure_mt5_connection_or_raise, mt5_adapter
from .._mcp_instance import mcp
from .gateway import create_trading_gateway
from .requests import (
    TradeRiskAnalyzeRequest,
    TradeStressTestRequest,
    TradeVarCvarRequest,
)
from .use_cases import (
    run_trade_risk_analyze,
    run_trade_stress_test,
    run_trade_var_cvar_calculate,
)

logger = logging.getLogger(__name__)


@mcp.tool()
def trade_risk_analyze(request: TradeRiskAnalyzeRequest) -> dict:
    """Analyze risk exposure for existing positions and calculate position sizing for new trades.

    Use this for symbol-level trade planning: current exposure, stop-loss risk,
    reward/risk, and optional lot sizing from nested `sizing`, `stop_loss`, and
    an optional `entry`. Select `sizing.method="kelly"` to derive the risk
    percentage from win rate and average win/loss metrics before broker rounding.
    When entry is omitted with symbol and stop_loss, it defaults to the live
    tick price: ask for long, bid for short, or mid when direction is omitted.
    It is not a portfolio tail-risk model; use `trade_var_cvar_calculate` for
    VaR/CVaR across open positions.
    Compact detail keeps the sizing decision fields; full detail includes
    broker volume rounding diagnostics and incomplete-sizing context.

    When sizing a proposed trade, pass direction='long' or direction='short' to
    validate that proposed SL/TP are on the correct side of the entry. New-trade
    sizing is opt-in: fixed_fraction requires `sizing.risk_pct` and stop_loss;
    Kelly requires its win/loss inputs in `sizing` plus stop_loss.
    strict_risk=True blocks positive suggested volume when broker minimum volume
    would exceed the requested risk.
    """
    return run_mt5_logged_operation(
        logger,
        operation="trade_risk_analyze",
        symbol=request.symbol,
        func=lambda: run_trade_risk_analyze(
            request,
            gateway=create_trading_gateway(
                adapter=mt5_adapter,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            ),
        ),
    )


@mcp.tool()
def trade_var_cvar_calculate(request: TradeVarCvarRequest) -> dict:
    """Estimate VaR/CVaR for current open MT5 positions.

    By default this performs account-level tail-risk analysis over the full
    current portfolio. Pass `symbol` to scope the calculation to open positions
    in one instrument; the response identifies that narrower scope.
    The risk horizon defaults to one bar of the requested timeframe.
    Pass ``horizon_bars`` to request a multi-bar overlapping holding period
    comparable with ``portfolio_risk_decompose``.
    For stop-loss exposure and new-trade lot sizing rather than statistical
    return-distribution risk, use `trade_risk_analyze`. For a lightweight
    execution snapshot that includes account, quote, open positions, and pending
    orders, use `trade_session_context`.
    """
    return run_mt5_logged_operation(
        logger,
        operation="trade_var_cvar_calculate",
        symbol=request.symbol,
        timeframe=request.timeframe,
        func=lambda: run_trade_var_cvar_calculate(
            request,
            gateway=create_trading_gateway(
                adapter=mt5_adapter,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            ),
        ),
    )


@mcp.tool()
def trade_stress_test(request: TradeStressTestRequest) -> dict:
    """Apply deterministic percentage shocks to current open MT5 positions.

    This tool is read-only. It estimates position-level and aggregate P&L impact
    using current position marks and broker tick-value metadata.
    """
    return run_mt5_logged_operation(
        logger,
        operation="trade_stress_test",
        shocks=request.shocks,
        func=lambda: run_trade_stress_test(
            request,
            gateway=create_trading_gateway(
                adapter=mt5_adapter,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            ),
        ),
    )
