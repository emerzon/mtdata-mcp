"""Read-only advanced analytics MCP tools backed exclusively by MT5 data."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from mtdata.analytics.requests import (
    MarketMicrostructureRequest,
    MarketRelativeStrengthRequest,
    PortfolioRiskDecomposeRequest,
    StrategyValidateRequest,
    TradeExecutionQualityRequest,
)
from mtdata.shared.execution_logging import run_logged_operation
from mtdata.shared.output_contract import ensure_common_meta
from mtdata.shared.runtime_metadata import attach_mt5_source

from ..analytics.engines import (
    analyze_execution_quality,
    analyze_microstructure,
    decompose_portfolio_risk,
    rank_relative_strength,
    validate_strategies,
)
from ..utils.mt5 import ensure_mt5_connection_or_raise, mt5_adapter
from ._mcp_instance import mcp
from .mt5_gateway import create_mt5_gateway

logger = logging.getLogger(__name__)


def _run(tool_name: str, request: Any, engine: Callable[[Any, Any], Dict[str, Any]]) -> Dict[str, Any]:
    def execute() -> Dict[str, Any]:
        gateway = create_mt5_gateway(
            adapter=mt5_adapter,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )
        gateway.ensure_connection()
        return ensure_common_meta(
            attach_mt5_source(engine(request, gateway), gateway=gateway),
            tool_name=tool_name,
        )

    return run_logged_operation(
        logger,
        operation=tool_name,
        symbol=getattr(request, "symbol", None),
        timeframe=getattr(request, "timeframe", None),
        func=execute,
    )


@mcp.tool()
def market_microstructure_analyze(request: MarketMicrostructureRequest) -> Dict[str, Any]:
    """Analyze broker-feed tick liquidity and applicable trade-flow proxies."""
    return _run("market_microstructure_analyze", request, analyze_microstructure)


@mcp.tool()
def trade_execution_quality(request: TradeExecutionQualityRequest) -> Dict[str, Any]:
    """Measure fill slippage, latency, partial fills, fees, and post-fill markouts."""
    return _run("trade_execution_quality", request, analyze_execution_quality)


@mcp.tool()
def strategy_validate(request: StrategyValidateRequest) -> Dict[str, Any]:
    """Run anchored fixed-candidate OOS validation with horizon-safe outcomes."""
    return _run("strategy_validate", request, validate_strategies)


@mcp.tool()
def portfolio_risk_decompose(request: PortfolioRiskDecomposeRequest) -> Dict[str, Any]:
    """Decompose current-position tail risk and explicit portfolio stresses."""
    return _run("portfolio_risk_decompose", request, decompose_portfolio_risk)


@mcp.tool()
def market_relative_strength(request: MarketRelativeStrengthRequest) -> Dict[str, Any]:
    """Rank MT5 symbols by robust, factor-adjusted multi-horizon relative strength."""
    return _run("market_relative_strength", request, rank_relative_strength)
