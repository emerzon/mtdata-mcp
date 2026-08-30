"""Main entry point for the MCP server."""

import atexit
import logging
import os
from typing import Any, Literal, Optional, cast

from ..bootstrap.runtime import (
    McpRuntimeSettings,
    apply_mcp_runtime_settings,
    load_mcp_runtime_settings,
)
from ..bootstrap.settings import load_environment
from ..bootstrap.tools import bootstrap_tools
from ..shared.constants import SERVICE_NAME
from ..utils.mt5 import mt5_connection
from ._mcp_instance import mcp
from .mt5_gateway import mt5_connection_error


def _mcp_readiness_payload() -> tuple[dict[str, Any], int]:
    """Return MCP readiness without invoking a user-facing tool."""
    connection_error = mt5_connection_error()
    if connection_error is None:
        return (
            {
                "service": "mtdata-mcp",
                "status": "ok",
                "ready": True,
                "components": {"mt5_connection": {"status": "ok"}},
            },
            200,
        )
    return (
        {
            "service": "mtdata-mcp",
            "status": "degraded",
            "ready": False,
            "components": {
                "mt5_connection": {
                    "status": "error",
                    "error_code": "mt5_connection_error",
                }
            },
        },
        503,
    )


@mcp.custom_route("/live", methods=["GET"], include_in_schema=False)
async def _mcp_live(_request: Any):
    """Report whether the MCP HTTP process can serve requests."""
    from starlette.responses import JSONResponse

    return JSONResponse({"service": "mtdata-mcp", "status": "ok"})


@mcp.custom_route("/ready", methods=["GET"], include_in_schema=False)
async def _mcp_ready(_request: Any):
    """Report whether the MCP process can establish its MT5 dependency."""
    from starlette.concurrency import run_in_threadpool
    from starlette.responses import JSONResponse

    payload, status_code = await run_in_threadpool(_mcp_readiness_payload)
    return JSONResponse(payload, status_code=status_code)


def _run_prefixed_sse(runtime: McpRuntimeSettings) -> None:
    """Mount the SSE app so advertised and routed message URLs agree."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route

    prefix = "/" + runtime.mount_path.strip("/")
    inner = mcp.sse_app(mount_path=prefix)
    # Custom FastMCP routes live inside ``inner``. Keep standard probe paths at
    # the server root even when the MCP protocol itself has a mount prefix.
    app = Starlette(
        routes=[
            Route("/live", endpoint=_mcp_live, methods=["GET"]),
            Route("/ready", endpoint=_mcp_ready, methods=["GET"]),
            Mount(prefix, app=inner),
        ]
    )
    uvicorn.run(
        app,
        host=runtime.host,
        port=runtime.port,
        log_level=runtime.log_level.lower(),
    )


@atexit.register
def _disconnect_mt5():
    mt5_connection.disconnect()


def _warm_windows_joblib_cpu_cache() -> None:
    """Resolve joblib CPU topology before MCP tools enter worker threads."""
    if os.name != "nt":
        return

    try:
        import joblib

        joblib.cpu_count(only_physical_cores=True)
    except Exception:
        logging.getLogger(__name__).warning(
            "Could not warm joblib CPU topology cache.",
            exc_info=True,
        )


def main(
    *,
    transport: Optional[Literal["stdio", "sse", "streamable-http"]] = None,
    runtime_settings: Optional[McpRuntimeSettings] = None,
):
    """Main entry point for the MCP server"""
    load_environment()
    runtime = runtime_settings or load_mcp_runtime_settings(transport_override=transport)
    # sklearn asks joblib for the physical CPU count before every first
    # KMeans fit. On Windows, resolving that count from an asyncio worker can
    # leave joblib waiting indefinitely on its PowerShell topology probe.
    _warm_windows_joblib_cpu_cache()
    bootstrap_tools()
    apply_mcp_runtime_settings(mcp, runtime)
    settings = getattr(mcp, 'settings', None)
    if settings is not None:
        log_level = getattr(logging, str(getattr(settings, 'log_level', 'INFO')).upper(), logging.INFO)
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)
    transport_name = runtime.transport
    mount_path = runtime.mount_path if runtime.transport == "sse" and runtime.mount_path not in ("", "/") else None
    logger.info(f"Starting {SERVICE_NAME} server... transport={transport_name}")

    if transport_name == "sse" and settings is not None:
        base_path = str(getattr(settings, 'mount_path', '') or '').rstrip("/") or "/"
        logger.info(
            "SSE listening at http://%s:%s%s (event path %s, message path %s)",
            getattr(settings, 'host', '127.0.0.1'),
            getattr(settings, 'port', 8000),
            base_path,
            getattr(settings, 'sse_path', '/sse'),
            getattr(settings, 'message_path', '/message'),
        )
    elif transport_name == "streamable-http" and settings is not None:
        logger.info(
            "Streamable HTTP listening at http://%s:%s%s",
            getattr(settings, 'host', '127.0.0.1'),
            getattr(settings, 'port', 8000),
            getattr(settings, 'streamable_http_path', runtime.mount_path),
        )

    if transport_name in {"sse", "streamable-http"} and settings is not None:
        logger.info(
            "MCP probes available at http://%s:%s/live and /ready",
            getattr(settings, 'host', '127.0.0.1'),
            getattr(settings, 'port', 8000),
        )

    run_fn = getattr(mcp, 'run', None)
    if run_fn is not None:
        if transport_name == "sse" and mount_path:
            _run_prefixed_sse(runtime)
            return
        transport_literal = cast(Literal['stdio', 'sse', 'streamable-http'], transport_name)
        run_fn(transport=transport_literal, mount_path=mount_path if transport_name == "sse" else None)


def main_stdio():
    """Entry point for stdio mode (forced)"""
    main(transport="stdio")


def main_sse():
    """Entry point for SSE mode (forced)"""
    main(transport="sse")


def main_streamable_http():
    """Entry point for streamable HTTP mode (forced)."""
    main(transport="streamable-http")


if __name__ == "__main__":
    main()
