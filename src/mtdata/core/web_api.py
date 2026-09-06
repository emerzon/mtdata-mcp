"""FastAPI app exposing WebUI-ready endpoints that wrap existing mtdata tools."""

from __future__ import annotations

import hmac
import logging
from functools import lru_cache
from typing import Annotated, Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..bootstrap.runtime import is_loopback_host, load_web_api_runtime_settings
from ..bootstrap.settings import load_environment, mt5_config
from ..forecast.forecast import get_forecast_methods_data as _get_methods_impl
from ..forecast.requests import (
    ForecastBacktestRequest,
    ForecastGenerateRequest,
    ForecastVolatilityEstimateRequest,
)
from ..forecast.volatility import (
    get_volatility_methods_data as _get_vol_methods,
)
from ..services.data_service import fetch_candles as _fetch_candles_impl
from ..shared.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..shared.schema import DetailLiteral
from ..utils.denoise import get_denoise_methods_data as _get_denoise_methods
from ..utils.denoise import normalize_denoise_spec as _norm_dn
from ..utils.dimred import list_dimred_methods as _list_dimred_methods
from ..utils.mt5 import (
    ensure_mt5_connection_or_raise,
    mt5,
)
from ..utils.volume_profile import (
    VolumeProfilePriceSourceLiteral,
    VolumeProfileVolumeSourceLiteral,
)
from .error_envelope import build_error_payload
from .forecast import (
    forecast_backtest_run as _forecast_backtest_tool,
)
from .forecast import (
    forecast_generate as _forecast_generate_tool,
)
from .forecast import (
    forecast_volatility_estimate as _forecast_volatility_tool,
)
from .forecast_tasks import forecast_models_list as _forecast_models_list_tool
from .market_depth import market_ticker as _market_ticker_tool
from .mt5_gateway import create_mt5_gateway, mt5_connection_error
from .pivot import confluence_levels, pivot_compute_points, support_resistance_levels
from .symbols.catalog import symbols_list
from .tool_calling import call_tool_sync_structured, unwrap_tool_callable
from .trading.ideas_requests import TradeIdeaComposeRequest
from .trading.positions import trade_get_open, trade_get_pending
from .volume_profile import (
    VolumeProfileSourceLiteral,
    volume_profile_levels,
)
from .web_api_geometry import (
    get_confluence_response as _get_confluence_response,
)
from .web_api_geometry import (
    get_exposure_response as _get_exposure_response,
)
from .web_api_geometry import (
    get_volume_profile_response as _get_volume_profile_response,
)
from .web_api_handlers import (
    get_denoise_methods_response as _get_denoise_methods_response,
)
from .web_api_handlers import (
    get_dimred_methods_response as _get_dimred_methods_response,
)
from .web_api_handlers import (
    get_history_response as _get_history_response,
)
from .web_api_handlers import (
    get_instruments_response as _get_instruments_response,
)
from .web_api_handlers import (
    get_methods_response as _get_methods_response,
)
from .web_api_handlers import (
    get_models_response as _get_models_response,
)
from .web_api_handlers import (
    get_pivots_response as _get_pivots_response,
)
from .web_api_handlers import (
    get_support_resistance_response as _get_support_resistance_response,
)
from .web_api_handlers import (
    get_tick_response as _get_tick_response,
)
from .web_api_handlers import (
    get_vol_methods_response as _get_vol_methods_response,
)
from .web_api_handlers import (
    get_wavelets_response as _get_wavelets_response,
)
from .web_api_handlers import (
    post_backtest_response as _post_backtest_response,
)
from .web_api_handlers import (
    post_forecast_price_response as _post_forecast_price_response,
)
from .web_api_handlers import (
    post_forecast_volatility_response as _post_forecast_volatility_response,
)
from .web_api_handlers import (
    post_trade_idea_response as _post_trade_idea_response,
)
from .web_api_models import (
    ToolInvokeBody,
)
from .web_api_radar import (
    get_radar_response as _get_radar_response,
)
from .web_api_radar import (
    get_session_strip_response as _get_session_strip_response,
)
from .web_api_runtime import (
    SafeJSONResponse,
    create_web_api_app,
    mount_webui,
    run_webapi,
)
from .web_api_tools import (
    TOOLS_CATALOG_DEFAULT_LIMIT,
    TOOLS_CATALOG_MAX_LIMIT,
    ToolCatalogCategory,
    ToolCatalogDetail,
)
from .web_api_tools import (
    get_tool_for_webapi as _get_tool_for_webapi,
)
from .web_api_tools import (
    invoke_tool_for_webapi as _invoke_tool_for_webapi,
)
from .web_api_tools import (
    list_tools_for_webapi as _list_tools_for_webapi,
)

API_PREFIXES = ("/api", "/api/v1")

logger = logging.getLogger(__name__)
_bearer_auth = HTTPBearer(auto_error=False)


def _raise_auth_error(status_code: int, message: str, *, code: str, headers: Optional[Dict[str, str]] = None) -> None:
    payload = build_error_payload(message, code=code, operation="web_api_auth")
    logger.warning(
        "transport=web_api operation=%s request_id=%s status=%s error=%s",
        "web_api_auth",
        payload["request_id"],
        status_code,
        payload["error"],
    )
    raise HTTPException(status_code=status_code, detail=payload, headers=headers)


def _is_local_api_client(request: Request) -> bool:
    headers = getattr(request, "headers", None)
    forwarded = None
    if headers is not None:
        try:
            forwarded = (
                headers.get("x-forwarded-for")
                or headers.get("forwarded")
                or headers.get("x-real-ip")
            )
        except Exception:
            forwarded = None
    if isinstance(forwarded, str) and forwarded.strip():
        return False
    client_host = getattr(getattr(request, "client", None), "host", None)
    client_text = str(client_host or "").strip().lower()
    return client_text == "testclient" or is_loopback_host(client_text)


@lru_cache(maxsize=1)
def _get_api_access_runtime_settings():
    return load_web_api_runtime_settings()


def _clear_api_access_runtime_settings_cache() -> None:
    _get_api_access_runtime_settings.cache_clear()


def _require_api_access(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_auth),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    runtime = _get_api_access_runtime_settings()
    configured_token = str(runtime.auth_token or "").strip()
    supplied_token = None
    if isinstance(credentials, HTTPAuthorizationCredentials):
        scheme = str(credentials.scheme or "").strip().lower()
        token = str(credentials.credentials or "").strip()
        if scheme == "bearer" and token:
            supplied_token = token
    if not supplied_token and isinstance(x_api_key, str) and x_api_key.strip():
        supplied_token = x_api_key.strip()

    if configured_token:
        if supplied_token and hmac.compare_digest(supplied_token, configured_token):
            return
        _raise_auth_error(
            401,
            "Missing or invalid API token.",
            code="web_api_auth_required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if _is_local_api_client(request):
        return

    _raise_auth_error(
        403,
        "Remote API access requires WEBAPI_AUTH_TOKEN.",
        code="web_api_remote_forbidden",
    )

load_environment()
app = create_web_api_app()
api_router = APIRouter(dependencies=[Depends(_require_api_access)])


def _call_tool_raw(func: Any) -> Any:
    return unwrap_tool_callable(func)


def _get_models_impl(*, method: Optional[str] = None, detail: str = "compact") -> Any:
    return _call_tool_raw(_forecast_models_list_tool)(method=method, detail=detail)


def _run_forecast_generate_impl(request: Any) -> Dict[str, Any]:
    return call_tool_sync_structured(_forecast_generate_tool, request=request)


def _run_forecast_backtest_impl(request: Any) -> Dict[str, Any]:
    return call_tool_sync_structured(_forecast_backtest_tool, request=request)


def _forecast_vol_impl(request: Any) -> Dict[str, Any]:
    return call_tool_sync_structured(_forecast_volatility_tool, request=request)


def _web_api_gateway():
    return create_mt5_gateway(
        adapter=mt5,
        ensure_connection_impl=ensure_mt5_connection_or_raise,
    )


def _readiness_payload() -> tuple[Dict[str, Any], int]:
    connection_error = mt5_connection_error(_web_api_gateway())
    if connection_error is None:
        return (
            {
                "service": "mtdata-webui",
                "status": "ok",
                "ready": True,
                "components": {
                    "mt5_connection": {
                        "status": "ok",
                    }
                },
            },
            200,
        )
    return (
        {
            "service": "mtdata-webui",
            "status": "degraded",
            "ready": False,
            "components": {
                "mt5_connection": {
                    "status": "error",
                    "error_code": connection_error.get("error_code")
                    or "mt5_connection_error",
                }
            },
        },
        503,
    )


@api_router.get("/timeframes")
def get_timeframes() -> Dict[str, Any]:
    return {
        "timeframes": list(TIMEFRAME_MAP),
        "seconds": dict(TIMEFRAME_SECONDS),
    }


@api_router.get("/instruments")
def get_instruments(search: Optional[str] = Query(None), limit: Optional[int] = Query(None, ge=1)) -> Dict[str, Any]:
    return _get_instruments_response(
        search=search,
        limit=limit,
        symbols_list_tool=symbols_list,
        call_tool_raw=_call_tool_raw,
    )


@api_router.get("/methods")
def get_methods(
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    return _get_methods_response(get_methods_impl=_get_methods_impl, detail=detail)


@api_router.get("/models")
def get_models(
    method: Optional[str] = Query(None),
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    return _get_models_response(
        get_models_impl=_get_models_impl,
        method=method,
        detail=detail,
    )


@api_router.get("/volatility/methods")
def get_vol_methods() -> Dict[str, Any]:
    return _get_vol_methods_response(get_vol_methods=_get_vol_methods)


@api_router.get("/denoise/methods")
def get_denoise_methods() -> Dict[str, Any]:
    return _get_denoise_methods_response(get_denoise_methods=_get_denoise_methods)


@api_router.get("/dimred/methods")
def get_dimred_methods() -> Dict[str, Any]:
    return _get_dimred_methods_response(list_dimred_methods=_list_dimred_methods)


@api_router.get("/denoise/wavelets")
def get_wavelets() -> Dict[str, Any]:
    return _get_wavelets_response()


@api_router.get("/history")
def get_history(
    symbol: Annotated[str, Query()],
    timeframe: Annotated[str, Query()] = "H1",
    limit: Annotated[
        Optional[int],
        Query(
            ge=1,
            le=100000,
            description=(
                "Maximum bars to return. Defaults to 20 for latest-N and bounded "
                "start/end range queries."
            ),
        ),
    ] = None,
    start: Annotated[Optional[str], Query()] = None,
    end: Annotated[Optional[str], Query()] = None,
    ohlcv: Annotated[Optional[str], Query()] = "ohlc",
    include_spread: Annotated[
        bool,
        Query(description="Append historical candle spread to each row."),
    ] = False,
    include_incomplete: Annotated[
        bool,
        Query(description="Include the latest forming candle."),
    ] = False,
    allow_stale: Annotated[
        bool,
        Query(description="Return data even when freshness checks fail."),
    ] = False,
    indicators: Annotated[
        Optional[str],
        Query(description="Indicator specification forwarded to data_fetch_candles."),
    ] = None,
    timestamp_format: Literal["epoch", "iso", "iso_utc"] = "iso_utc",
    detail: DetailLiteral = "compact",
    denoise_method: Annotated[
        Optional[str],
        Query(description="Denoise method name; if set, returns extra *_dn columns."),
    ] = None,
    denoise_params: Annotated[
        Optional[str],
        Query(description="JSON or k=v list of denoise params."),
    ] = None,
) -> Dict[str, Any]:
    return _get_history_response(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
        start=start,
        end=end,
        ohlcv=ohlcv,
        include_spread=include_spread,
        include_incomplete=include_incomplete,
        allow_stale=allow_stale,
        indicators=indicators,
        timestamp_format=timestamp_format,
        detail=detail,
        denoise_method=denoise_method,
        denoise_params=denoise_params,
        fetch_candles_impl=_fetch_candles_impl,
        get_denoise_methods=_get_denoise_methods,
        normalize_denoise_spec=_norm_dn,
        gateway=_web_api_gateway(),
        mt5_config=mt5_config,
    )


@api_router.get("/pivots")
def get_pivots(
    symbol: str = Query(...),
    timeframe: str = Query("H1"),
    method: str = Query("classic"),
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    return _get_pivots_response(
        symbol=symbol,
        timeframe=timeframe,
        method=method,
        detail=detail,
        pivot_tool=pivot_compute_points,
        call_tool_raw=_call_tool_raw,
    )


@api_router.get("/support-resistance")
def get_support_resistance(
    symbol: str = Query(...),
    timeframe: str = Query("H1"),
    lookback: Optional[int] = Query(None, ge=100, le=20000),
    tolerance_pct: float = Query(0.15, ge=0.0, le=5.0),
    min_touches: int = Query(2, ge=1),
    max_levels: int = Query(4, ge=1, le=20),
    max_distance_pct: Optional[float] = Query(5.0, ge=0.0, le=100.0),
    volume_weighting: Literal["off", "auto"] = Query("off"),
    reaction_bars: int = Query(6, ge=1),
    adx_period: int = Query(14, ge=1),
    decay_half_life_bars: Optional[int] = Query(None, ge=1),
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    return _get_support_resistance_response(
        symbol=symbol,
        timeframe=timeframe,
        lookback=lookback,
        tolerance_pct=tolerance_pct,
        min_touches=min_touches,
        max_levels=max_levels,
        max_distance_pct=max_distance_pct,
        volume_weighting=volume_weighting,
        reaction_bars=reaction_bars,
        adx_period=adx_period,
        decay_half_life_bars=decay_half_life_bars,
        detail=detail,
        support_resistance_tool=support_resistance_levels,
        call_tool_raw=_call_tool_raw,
    )


@api_router.get("/confluence")
def get_confluence(
    symbol: str = Query(...),
    pivot_timeframe: str = Query("D1"),
    sr_timeframe: str = Query("auto"),
) -> Dict[str, Any]:
    return _get_confluence_response(
        symbol=symbol,
        pivot_timeframe=pivot_timeframe,
        sr_timeframe=sr_timeframe,
        confluence_tool=confluence_levels,
    )


@api_router.get("/volume-profile")
def get_volume_profile(
    symbol: str = Query(...),
    timeframe: Optional[str] = Query("H1"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    lookback: Optional[int] = Query(None, ge=1, le=20_000),
    source: VolumeProfileSourceLiteral = Query("auto"),
    price_source: VolumeProfilePriceSourceLiteral = Query("mid"),
    volume_source: VolumeProfileVolumeSourceLiteral = Query("auto"),
    bucket_size: Optional[float] = Query(None),
    bucket_points: Optional[float] = Query(None),
    bucket_count: Optional[int] = Query(None),
    max_buckets: int = Query(120, ge=1, le=2_000),
    value_area_pct: float = Query(70.0, gt=0.0, le=100.0),
    reference_price: Optional[float] = Query(None),
    max_tick_window_days: int = Query(1, ge=1, le=90),
    max_ticks: int = Query(50_000, ge=1, le=200_000),
    max_m1_bars: int = Query(20_000, ge=1, le=100_000),
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    return _get_volume_profile_response(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        lookback=lookback,
        source=source,
        price_source=price_source,
        volume_source=volume_source,
        bucket_size=bucket_size,
        bucket_points=bucket_points,
        bucket_count=bucket_count,
        max_buckets=max_buckets,
        value_area_pct=value_area_pct,
        reference_price=reference_price,
        max_tick_window_days=max_tick_window_days,
        max_ticks=max_ticks,
        max_m1_bars=max_m1_bars,
        detail=detail,
        volume_profile_tool=volume_profile_levels,
    )


@api_router.get("/exposure")
def get_exposure(symbol: str = Query(...)) -> Dict[str, Any]:
    return _get_exposure_response(
        symbol=symbol,
        open_tool=trade_get_open,
        pending_tool=trade_get_pending,
    )


@api_router.get("/radar")
def get_radar(
    symbols: Optional[str] = Query(None),
    timeframe: str = Query("H1"),
    rank_by: str = Query("watchlist"),
    limit: int = Query(20, ge=1, le=20),
) -> Dict[str, Any]:
    return _get_radar_response(
        symbols=symbols,
        timeframe=timeframe,
        rank_by=rank_by,
        limit=limit,
    )


@api_router.get("/session-strip")
def get_session_strip(symbol: Optional[str] = Query(None)) -> Dict[str, Any]:
    return _get_session_strip_response(symbol=symbol)


@api_router.get("/tick")
def get_tick(
    symbol: str = Query(...),
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    return _get_tick_response(
        symbol=symbol,
        detail=detail,
        market_ticker_tool=_market_ticker_tool,
        call_tool_raw=_call_tool_raw,
    )


@api_router.post("/forecast/price", response_model=None)
def post_forecast_price(body: ForecastGenerateRequest) -> Dict[str, Any] | SafeJSONResponse:
    result = _post_forecast_price_response(
        body=body,
        forecast_generate_use_case=_run_forecast_generate_impl,
    )
    if (
        body.async_mode
        and isinstance(result, dict)
        and result.get("task_id")
        and result.get("status") in {"pending", "running"}
    ):
        return SafeJSONResponse(status_code=202, content=result)
    return result


@api_router.post("/forecast/volatility")
def post_forecast_volatility(body: ForecastVolatilityEstimateRequest) -> Dict[str, Any]:
    return _post_forecast_volatility_response(body=body, forecast_vol_impl=_forecast_vol_impl)


@api_router.post("/backtest")
def post_backtest(body: ForecastBacktestRequest) -> Dict[str, Any]:
    return _post_backtest_response(body=body, backtest_use_case=_run_forecast_backtest_impl)


@api_router.post("/trade-ideas")
def post_trade_idea(body: TradeIdeaComposeRequest) -> Dict[str, Any]:
    from .trading.ideas import run_trade_idea_compose

    return _post_trade_idea_response(body=body, compose_impl=run_trade_idea_compose)


@api_router.get("/health")
def health() -> Dict[str, Any]:
    return {"service": "mtdata-webui", "status": "ok"}


@api_router.get("/ready")
def ready() -> SafeJSONResponse:
    payload, status_code = _readiness_payload()
    return SafeJSONResponse(status_code=status_code, content=payload)


@api_router.get("/tools")
def list_tools(
    category: Annotated[Optional[ToolCatalogCategory], Query()] = None,
    search: Annotated[Optional[str], Query()] = None,
    detail: ToolCatalogDetail = Query("compact"),
    include_fields: bool = Query(False),
    limit: Annotated[int, Query(ge=1, le=TOOLS_CATALOG_MAX_LIMIT)] = TOOLS_CATALOG_DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Dict[str, Any]:
    """List MCP tools with surface classification for the Web UI runner."""
    return _list_tools_for_webapi(
        category=category,
        search=search,
        detail=detail,
        include_fields=include_fields,
        limit=limit,
        offset=offset,
    )


@api_router.get("/tools/{tool_name}")
def get_tool(
    tool_name: str,
    detail: ToolCatalogDetail = Query("compact"),
    include_fields: bool = Query(True),
) -> Dict[str, Any]:
    """Return one tool with parameter field descriptors for the form runner."""
    return _get_tool_for_webapi(
        tool_name,
        detail=detail,
        include_fields=include_fields,
    )


@api_router.post("/tools/{tool_name}/invoke")
def invoke_tool(tool_name: str, body: ToolInvokeBody) -> Dict[str, Any]:
    """Invoke a registered MCP tool.

    Confirm is required only when the prepared call can mutate state
    (`dry_run=false` for live trade and destructive store tools; always for
    mutating tools that have no dry-run preview). Domain failures return
    HTTP 4xx/5xx with `success=false`.
    """
    return _invoke_tool_for_webapi(
        tool_name,
        arguments=body.arguments,
        confirm=bool(body.confirm),
    )


@app.get("/health")
def health_root() -> Dict[str, Any]:
    return health()


@app.get("/ready")
def ready_root() -> SafeJSONResponse:
    return ready()


@app.get("/")
def root() -> Dict[str, Any]:
    return health()


for _prefix in API_PREFIXES:
    app.include_router(api_router, prefix=_prefix)


mount_webui(app)


def main_webapi() -> None:
    """Entry point to run the FastAPI web server."""
    load_environment()
    _clear_api_access_runtime_settings_cache()
    run_webapi(app)
