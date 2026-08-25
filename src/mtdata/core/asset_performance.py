"""Delayed cross-asset performance context (not live broker quotes)."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, Literal, Optional

from pydantic import Field

from ..services.research.capabilities import PERFORMANCE, ResearchSourcePin
from ..services.research.errors import finviz_only_source_error
from ..services.research.payload import stamp_provider
from ..shared.schema import DetailLiteral
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation

logger = logging.getLogger(__name__)

PerformanceUniverse = Literal["forex", "crypto", "futures", "insider"]
PerformanceRankBy = Literal[
    "5min",
    "hour",
    "day",
    "week",
    "month",
    "quarter",
    "half",
    "year",
    "ytd",
]
PerformanceRankOrder = Literal["desc", "asc"]


def _fetch_finviz_performance(
    *,
    universe: str,
    symbol: Optional[str],
    option: str,
    limit: int,
    offset: int,
    page: int,
    detail: str,
    rank_by: Optional[str],
    order: Optional[str],
) -> Dict[str, Any]:
    from . import finviz as finviz_impl

    if universe == "forex":
        return finviz_impl.finviz_forex(
            symbol=symbol,
            limit=limit,
            offset=offset,
            detail=detail,  # type: ignore[arg-type]
            rank_by=rank_by,  # type: ignore[arg-type]
            order=order,  # type: ignore[arg-type]
        )
    if universe == "crypto":
        return finviz_impl.finviz_crypto(
            symbol=symbol,
            limit=limit,
            offset=offset,
            detail=detail,  # type: ignore[arg-type]
            rank_by=rank_by,  # type: ignore[arg-type]
            order=order,  # type: ignore[arg-type]
        )
    if universe == "futures":
        return finviz_impl.finviz_futures(
            symbol=symbol,
            limit=limit,
            offset=offset,
            detail=detail,  # type: ignore[arg-type]
            rank_by=rank_by,  # type: ignore[arg-type]
            order=order,  # type: ignore[arg-type]
        )
    return finviz_impl.finviz_insider_activity(
        option=option,  # type: ignore[arg-type]
        limit=limit,
        page=page,
        detail=detail,  # type: ignore[arg-type]
    )


@mcp.tool()
def asset_performance(
    universe: Annotated[
        PerformanceUniverse,
        Field(description="Context table: forex, crypto, futures, or market-wide insider."),
    ] = "forex",
    symbol: Annotated[
        Optional[str],
        Field(
            description=(
                "Optional forex, crypto, or futures filter such as EURUSD, "
                "BTCUSD/BTC, or the provider ticker/name."
            )
        ),
    ] = None,
    option: Annotated[
        Literal[
            "latest",
            "latest buys",
            "latest sales",
            "top week",
            "top week buys",
            "top week sales",
            "top owner trade",
            "top owner buys",
            "top owner sales",
        ],
        Field(description="Insider-activity slice when universe=insider."),
    ] = "latest",
    limit: Annotated[int, Field(ge=1, description="Max rows per page.")] = 20,
    offset: Annotated[int, Field(ge=0, description="Zero-based offset for forex/crypto/futures.")] = 0,
    page: Annotated[int, Field(ge=1, description="One-based page for insider activity.")] = 1,
    rank_by: Annotated[
        Optional[PerformanceRankBy],
        Field(
            description=(
                "Rank the fetched forex/crypto/futures snapshot by a performance "
                "horizon before paging. Forex and crypto accept 5min, hour, day, "
                "week, month, quarter, half, year, ytd. Futures currently only "
                "has day. Omit to keep provider table order."
            )
        ),
    ] = None,
    order: Annotated[
        Optional[PerformanceRankOrder],
        Field(
            description=(
                "Rank direction when rank_by is set: desc (default) or asc."
            )
        ),
    ] = None,
    detail: DetailLiteral = "compact",
    source: Annotated[
        ResearchSourcePin,
        Field(
            description="Adapter pin. auto uses every source that can serve this query."
        ),
    ] = "auto",
) -> Dict[str, Any]:
    """Fetch delayed cross-asset performance or market-wide insider context.

    This is research context, not a live executable quote. Use
    ``symbols_top_markets`` or ``market_ticker`` for broker prices. Finviz is
    the current adapter; ``source="mt5"`` returns a capability error.
    """

    def _run() -> Dict[str, Any]:
        pin_error = finviz_only_source_error(
            source,
            capability=PERFORMANCE,
            operation="asset_performance",
        )
        if pin_error is not None:
            return pin_error
        universe_key = str(universe)
        invalid: list[str] = []
        if universe_key == "insider" and symbol is not None:
            invalid.append("symbol")
        if universe_key != "insider" and str(option) != "latest":
            invalid.append("option")
        if universe_key != "insider" and int(page) != 1:
            invalid.append("page")
        if universe_key == "insider" and int(offset) != 0:
            invalid.append("offset")
        if universe_key == "insider":
            if rank_by is not None:
                invalid.append("rank_by")
            if order is not None:
                invalid.append("order")
        elif order is not None and rank_by is None:
            return build_error_payload(
                "order requires rank_by.",
                code="parameter_dependency_missing",
                operation="asset_performance",
                details={"parameter": "order", "requires": ["rank_by"]},
                valid_values={"rank_by": [
                    "5min", "hour", "day", "week", "month", "quarter",
                    "half", "year", "ytd",
                ]},
                remediation="Set --rank-by, or omit --order to keep provider order.",
            )
        if invalid:
            valid_by_universe = {
                "forex": ["symbol", "offset", "limit", "rank_by", "order", "detail"],
                "crypto": ["symbol", "offset", "limit", "rank_by", "order", "detail"],
                "futures": ["symbol", "offset", "limit", "rank_by", "order", "detail"],
                "insider": ["option", "page", "limit", "detail"],
            }
            return build_error_payload(
                "Universe '"
                + universe_key
                + "' does not use "
                + ", ".join(invalid)
                + ".",
                code="incompatible_parameters",
                operation="asset_performance",
                details={"invalid": invalid, "universe": universe_key},
                valid_values={
                    "universe": list(valid_by_universe),
                    "controls": valid_by_universe.get(universe_key, []),
                },
                remediation=(
                    "Drop the listed selectors, or switch --universe to one that "
                    "implements them."
                ),
            )
        payload = _fetch_finviz_performance(
            universe=str(universe),
            symbol=symbol,
            option=str(option),
            limit=int(limit),
            offset=int(offset),
            page=int(page),
            detail=str(detail or "compact"),
            rank_by=rank_by,
            order=order,
        )
        out = stamp_provider(payload, provider="finviz")
        if isinstance(out, dict):
            if out.get("success") is False or out.get("error"):
                provider_operation = out.get("operation")
                if provider_operation not in (None, "", "asset_performance"):
                    out["provider_operation"] = provider_operation
                out["operation"] = "asset_performance"
            out.setdefault("universe", universe)
            out.setdefault(
                "quote_role",
                "research_context_not_live_broker_quote",
            )
        return out

    return run_logged_operation(
        logger,
        operation="asset_performance",
        universe=universe,
        source=source,
        func=_run,
    )
