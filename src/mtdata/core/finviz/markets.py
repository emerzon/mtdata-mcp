"""Finviz forex, crypto, and futures performance adapters."""

from typing import (
    Annotated,
    Any,
    Dict,
    Literal,
    Optional,
)

from pydantic import Field

from mtdata.core.finviz.common import (
    _FINVIZ_PERFORMANCE_RANK_BY_SUPPORTED,
    _finviz_error_payload,
    _normalize_finviz_market_payload,
    _resolve_finviz_performance_rank,
    _run_logged_tool,
    _validate_finviz_crypto_symbol_filter,
    _validate_finviz_detail,
    _validate_positive_finviz_limit,
)
from mtdata.services.finviz import (
    get_crypto_performance,
    get_forex_performance,
    get_futures_performance,
)
from mtdata.shared.schema import DetailLiteral
from mtdata.shared.symbols import finviz_forex_symbol_to_mt5

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


def finviz_forex(
    symbol: Optional[str] = None,
    limit: Annotated[int, Field(ge=1)] = 20,
    offset: Annotated[int, Field(ge=0)] = 0,
    detail: DetailLiteral = "compact",  # type: ignore
    rank_by: Optional[PerformanceRankBy] = None,
    order: Optional[PerformanceRankOrder] = None,
) -> Dict[str, Any]:
    """
    Get forex currency pairs performance from Finviz.
    
    Returns performance data for major currency pairs including
    daily change, weekly change, and other metrics.

    Use offset with limit to retrieve consecutive non-overlapping rows.
    
    Returns
    -------
    dict
        Forex pairs performance data
    """
    request = {
        "symbol": symbol,
        "limit": limit,
        "offset": offset,
        "detail": detail,
        "rank_by": rank_by,
        "order": order,
    }

    def _run() -> Dict[str, Any]:
        symbol_norm = None
        if symbol not in (None, ""):
            symbol_norm = finviz_forex_symbol_to_mt5(symbol)
            if symbol_norm is None:
                return _finviz_error_payload(
                    (
                        f"Invalid forex symbol: {symbol}. Use a six-letter fiat "
                        "pair such as EURUSD or a slash pair such as EUR/USD."
                    ),
                    code="finviz_forex_invalid_symbol",
                    operation="finviz_forex",
                    details={"symbol": symbol},
                )
        limit_error = _validate_positive_finviz_limit(
            limit,
            operation="finviz_forex",
        )
        if limit_error is not None:
            return limit_error
        detail_error = _validate_finviz_detail(detail, operation="finviz_forex")
        if detail_error is not None:
            return detail_error
        *_, rank_error = _resolve_finviz_performance_rank(
            rank_by,
            order,
            operation="finviz_forex",
            supported=_FINVIZ_PERFORMANCE_RANK_BY_SUPPORTED["pairs"],
        )
        if rank_error is not None:
            return rank_error
        return _normalize_finviz_market_payload(
            get_forex_performance(),
            rows_key="pairs",
            limit=limit,
            offset=offset,
            detail=detail,
            tool="finviz_forex",
            request=request,
            symbol_filter=symbol_norm,
        )

    return _run_logged_tool("finviz_forex", request, _run)


def finviz_crypto(
    symbol: Optional[str] = None,
    limit: Annotated[int, Field(ge=1)] = 20,
    offset: Annotated[int, Field(ge=0)] = 0,
    detail: DetailLiteral = "compact",  # type: ignore
    rank_by: Optional[PerformanceRankBy] = None,
    order: Optional[PerformanceRankOrder] = None,
) -> Dict[str, Any]:
    """
    Get cryptocurrency performance from Finviz.
    
    Returns performance data for major cryptocurrencies including
    price, daily change, volume, and market cap.

    Use offset with limit to retrieve consecutive non-overlapping rows.
    
    Returns
    -------
    dict
        Crypto performance data
    """
    request = {
        "symbol": symbol,
        "limit": limit,
        "offset": offset,
        "detail": detail,
        "rank_by": rank_by,
        "order": order,
    }

    def _run() -> Dict[str, Any]:
        detail_error = _validate_finviz_detail(detail, operation="finviz_crypto")
        if detail_error is not None:
            return detail_error
        quote_error = _validate_finviz_crypto_symbol_filter(
            symbol,
            operation="finviz_crypto",
        )
        if quote_error is not None:
            return quote_error
        *_, rank_error = _resolve_finviz_performance_rank(
            rank_by,
            order,
            operation="finviz_crypto",
            supported=_FINVIZ_PERFORMANCE_RANK_BY_SUPPORTED["coins"],
        )
        if rank_error is not None:
            return rank_error
        return _normalize_finviz_market_payload(
            get_crypto_performance(),
            rows_key="coins",
            limit=limit,
            offset=offset,
            detail=detail,
            tool="finviz_crypto",
            request=request,
            symbol_filter=symbol,
        )

    return _run_logged_tool("finviz_crypto", request, _run)


def finviz_futures(
    symbol: Optional[str] = None,
    limit: Annotated[int, Field(ge=1)] = 20,
    offset: Annotated[int, Field(ge=0)] = 0,
    detail: DetailLiteral = "compact",  # type: ignore
    rank_by: Optional[PerformanceRankBy] = None,
    order: Optional[PerformanceRankOrder] = None,
) -> Dict[str, Any]:
    """
    Get futures market performance from Finviz.
    
    This endpoint is a performance-only Finviz source. It returns daily percent
    moves for major futures contracts across commodities, indices, bonds, and
    currencies, but Finviz does not expose current price or volume in this
    source. The response includes data_limitations.price when price is absent.
    Use offset with limit to retrieve consecutive non-overlapping rows.
    
    Returns
    -------
    dict
        Futures performance data
    """
    request = {
        "symbol": symbol,
        "limit": limit,
        "offset": offset,
        "detail": detail,
        "rank_by": rank_by,
        "order": order,
    }

    def _run() -> Dict[str, Any]:
        detail_error = _validate_finviz_detail(detail, operation="finviz_futures")
        if detail_error is not None:
            return detail_error
        *_, rank_error = _resolve_finviz_performance_rank(
            rank_by,
            order,
            operation="finviz_futures",
            supported=_FINVIZ_PERFORMANCE_RANK_BY_SUPPORTED["futures"],
        )
        if rank_error is not None:
            return rank_error
        return _normalize_finviz_market_payload(
            get_futures_performance(),
            rows_key="futures",
            limit=limit,
            offset=offset,
            detail=detail,
            tool="finviz_futures",
            request=request,
            symbol_filter=symbol,
        )

    return _run_logged_tool("finviz_futures", request, _run)
