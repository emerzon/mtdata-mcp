"""Finviz forex, crypto, and futures performance adapters."""

from typing import (
    Annotated,
    Any,
    Callable,
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


def _run_finviz_performance_market(
    *,
    operation: str,
    fetch: Callable[[], Any],
    rows_key: str,
    rank_supported: Any,
    symbol: Optional[str],
    limit: int,
    offset: int,
    detail: DetailLiteral,
    rank_by: Optional[PerformanceRankBy],
    order: Optional[PerformanceRankOrder],
    prepare_symbol: Optional[Callable[[Optional[str]], tuple[Optional[str], Optional[Dict[str, Any]]]]] = None,
    validate_limit: bool = False,
) -> Dict[str, Any]:
    request = {
        "symbol": symbol,
        "limit": limit,
        "offset": offset,
        "detail": detail,
        "rank_by": rank_by,
        "order": order,
    }

    def _run() -> Dict[str, Any]:
        symbol_filter: Optional[str] = symbol
        if prepare_symbol is not None:
            symbol_filter, prepare_error = prepare_symbol(symbol)
            if prepare_error is not None:
                return prepare_error
        if validate_limit:
            limit_error = _validate_positive_finviz_limit(
                limit,
                operation=operation,
            )
            if limit_error is not None:
                return limit_error
        detail_error = _validate_finviz_detail(detail, operation=operation)
        if detail_error is not None:
            return detail_error
        *_, rank_error = _resolve_finviz_performance_rank(
            rank_by,
            order,
            operation=operation,
            supported=rank_supported,
        )
        if rank_error is not None:
            return rank_error
        return _normalize_finviz_market_payload(
            fetch(),
            rows_key=rows_key,
            limit=limit,
            offset=offset,
            detail=detail,
            tool=operation,
            request=request,
            symbol_filter=symbol_filter,
        )

    return _run_logged_tool(operation, request, _run)


def _prepare_forex_symbol(symbol: Optional[str]) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    if symbol in (None, ""):
        return None, None
    symbol_norm = finviz_forex_symbol_to_mt5(symbol)
    if symbol_norm is None:
        return None, _finviz_error_payload(
            (
                f"Invalid forex symbol: {symbol}. Use a six-letter fiat "
                "pair such as EURUSD or a slash pair such as EUR/USD."
            ),
            code="finviz_forex_invalid_symbol",
            operation="finviz_forex",
            details={"symbol": symbol},
        )
    return symbol_norm, None


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
    return _run_finviz_performance_market(
        operation="finviz_forex",
        fetch=get_forex_performance,
        rows_key="pairs",
        rank_supported=_FINVIZ_PERFORMANCE_RANK_BY_SUPPORTED["pairs"],
        symbol=symbol,
        limit=limit,
        offset=offset,
        detail=detail,
        rank_by=rank_by,
        order=order,
        prepare_symbol=_prepare_forex_symbol,
        validate_limit=True,
    )


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
    def _prepare_crypto_symbol(
        value: Optional[str],
    ) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        quote_error = _validate_finviz_crypto_symbol_filter(
            value,
            operation="finviz_crypto",
        )
        if quote_error is not None:
            return None, quote_error
        return value, None

    return _run_finviz_performance_market(
        operation="finviz_crypto",
        fetch=get_crypto_performance,
        rows_key="coins",
        rank_supported=_FINVIZ_PERFORMANCE_RANK_BY_SUPPORTED["coins"],
        symbol=symbol,
        limit=limit,
        offset=offset,
        detail=detail,
        rank_by=rank_by,
        order=order,
        prepare_symbol=_prepare_crypto_symbol,
    )


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
    return _run_finviz_performance_market(
        operation="finviz_futures",
        fetch=get_futures_performance,
        rows_key="futures",
        rank_supported=_FINVIZ_PERFORMANCE_RANK_BY_SUPPORTED["futures"],
        symbol=symbol,
        limit=limit,
        offset=offset,
        detail=detail,
        rank_by=rank_by,
        order=order,
    )
