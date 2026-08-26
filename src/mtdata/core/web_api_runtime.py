"""FastAPI runtime assembly helpers for the Web API."""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.responses import JSONResponse, Response
from starlette.staticfiles import StaticFiles

from ..bootstrap.runtime import WebApiRuntimeSettings, load_web_api_runtime_settings
from .error_envelope import build_error_payload
from .output_serialization import sanitize_json
from .request_context import current_request_id, normalize_request_id, request_id_scope

logger = logging.getLogger(__name__)

WEBUI_MOUNT_PATH = "/app"
_MISSING_UI_STATUS = 503
_PACKAGE_DISTRIBUTION = "mtdata-mcp"


def _package_version() -> str:
    try:
        return importlib_metadata.version(_PACKAGE_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError:
        return "0+unknown"


class SafeJSONResponse(JSONResponse):
    """JSON response that preserves strict JSON by converting NaN/inf to null."""

    def render(self, content: Any) -> bytes:
        return super().render(sanitize_json(content))


@dataclass(frozen=True)
class WebUiMountResult:
    """Outcome of attempting to mount the production SPA."""

    mounted: bool
    directory: str
    reason: str | None = None


def create_web_api_app(settings: WebApiRuntimeSettings | None = None) -> FastAPI:
    """Create the shared FastAPI app with configured CORS middleware."""
    runtime = settings or load_web_api_runtime_settings()
    app = FastAPI(
        title="mtdata-webui",
        version=_package_version(),
        default_response_class=SafeJSONResponse,
    )
    origins = list(runtime.cors_origins)
    if not origins:
        origins = ["http://127.0.0.1:5173", "http://localhost:5173"]
    if any(str(origin).strip() == "*" for origin in origins):
        raise ValueError(
            "CORS_ORIGINS cannot include '*' while credentialed requests are enabled; specify explicit origins."
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> SafeJSONResponse:
        path = request.url.path.rstrip("/")
        operation = "web_api_request"
        if "/tools/" in path and path.endswith("/invoke"):
            operation = path.rsplit("/", 2)[-2]
        elif path.endswith("/tools"):
            operation = "tools_list"
        elif "/tools/" in path:
            operation = "tools_get"
        issues = [
            {
                "location": [str(part) for part in error.get("loc", ())],
                "type": str(error.get("type") or "validation_error"),
                "message": str(error.get("msg") or "Invalid request value."),
            }
            for error in exc.errors()
        ]
        payload = build_error_payload(
            "Request validation failed.",
            code="web_api_validation_error",
            request_id=current_request_id(),
            operation=operation,
            details={"issues": issues},
        )
        return SafeJSONResponse(status_code=422, content={"detail": payload})

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next: Any) -> Response:
        request_id = normalize_request_id(request.headers.get("x-request-id"))
        if request_id is None:
            from .error_envelope import new_request_id

            request_id = new_request_id()
        with request_id_scope(request_id):
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    return app


def resolve_webui_dist(directory: str | Path) -> Path | None:
    """Return the dist path when it contains a production `index.html`, else None."""
    path = Path(directory)
    try:
        if path.is_dir() and (path / "index.html").is_file():
            return path.resolve()
    except OSError:
        return None
    return None


def missing_webui_payload(directory: str) -> dict[str, Any]:
    """Structured payload describing how to enable the bundled Web UI."""
    return {
        "service": "mtdata-webui",
        "status": "ui_not_built",
        "error_code": "webui_dist_missing",
        "message": (
            "The Web UI production build was not found. "
            "Build it once, then restart mtdata-webapi and open /app/."
        ),
        "directory": directory,
        "enable": {
            "commands": [
                "cd webui",
                "npm install",
                "npm run build",
            ],
            "open": "http://127.0.0.1:8000/app/",
            "override_env": "WEBUI_DIST_DIR",
            "docs": "docs/WEB_API.md",
        },
    }


def missing_webui_html(directory: str) -> str:
    """Professional HTML guidance page when the SPA dist is missing."""
    payload = missing_webui_payload(directory)
    commands = "\n".join(payload["enable"]["commands"])
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>MTData — Web UI not built</title>
    <style>
      :root {{ color-scheme: dark; }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        min-height: 100vh;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        background: #020617;
        color: #e2e8f0;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem 1rem;
      }}
      main {{
        width: min(40rem, 100%);
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid #1e293b;
        border-radius: 1rem;
        padding: 1.75rem 1.5rem;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45);
      }}
      .brand {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #38bdf8;
        font-weight: 600;
        margin-bottom: 0.75rem;
      }}
      h1 {{
        margin: 0 0 0.75rem;
        font-size: 1.35rem;
        font-weight: 650;
        color: #f8fafc;
      }}
      p {{
        margin: 0 0 1rem;
        line-height: 1.55;
        color: #94a3b8;
      }}
      pre {{
        margin: 0 0 1rem;
        padding: 0.9rem 1rem;
        overflow-x: auto;
        border-radius: 0.65rem;
        background: #0f172a;
        border: 1px solid #1e293b;
        color: #bae6fd;
        font-size: 0.85rem;
        line-height: 1.5;
      }}
      code {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.85em;
        color: #7dd3fc;
      }}
      ul {{
        margin: 0;
        padding-left: 1.15rem;
        color: #94a3b8;
        line-height: 1.6;
      }}
      li + li {{ margin-top: 0.35rem; }}
      .path {{
        word-break: break-all;
        color: #cbd5e1;
      }}
    </style>
  </head>
  <body>
    <main>
      <div class="brand">MTData · Chart Workspace</div>
      <h1>Web UI is not built yet</h1>
      <p>
        The API is running, but the production SPA was not found at
        <span class="path"><code>{html.escape(directory, quote=True)}</code></span>.
        Build the frontend once, restart the server, then open
        <code>{html.escape(payload["enable"]["open"], quote=True)}</code>.
      </p>
      <pre>{html.escape(commands, quote=True)}</pre>
      <ul>
        <li>Override the dist path with <code>WEBUI_DIST_DIR</code> if needed.</li>
        <li>For live frontend development: <code>cd webui && npm run dev</code> (proxies API on :8000).</li>
        <li>Full details: <code>docs/WEB_API.md</code> and <code>docs/SETUP.md</code>.</li>
      </ul>
    </main>
  </body>
</html>
"""


def build_missing_webui_response(request: Request | None, directory: str) -> Response:
    """Return HTML or JSON guidance when the production UI is not available."""
    payload = missing_webui_payload(directory)
    accept = ""
    if request is not None:
        accept = (request.headers.get("accept") or "").lower()
    prefers_json = "application/json" in accept and "text/html" not in accept
    if prefers_json:
        return JSONResponse(payload, status_code=_MISSING_UI_STATUS)
    return HTMLResponse(missing_webui_html(directory), status_code=_MISSING_UI_STATUS)


def mount_webui(
    app: FastAPI,
    *,
    directory: str | None = None,
    settings: WebApiRuntimeSettings | None = None,
) -> WebUiMountResult:
    """Mount the built SPA when present; otherwise expose a clear missing-UI path."""
    runtime = settings or load_web_api_runtime_settings()
    target = directory if directory is not None else runtime.webui_directory
    resolved = resolve_webui_dist(target)
    if resolved is not None:
        try:
            app.mount(
                WEBUI_MOUNT_PATH,
                StaticFiles(directory=str(resolved), html=True),
                name="webui",
            )
        except Exception as exc:
            logger.warning("Failed to mount Web UI from %s: %s", resolved, exc)
            _register_missing_webui_routes(app, str(target))
            return WebUiMountResult(mounted=False, directory=str(target), reason=str(exc))
        logger.info("Mounted Web UI from %s at %s/", resolved, WEBUI_MOUNT_PATH)
        return WebUiMountResult(mounted=True, directory=str(resolved), reason=None)

    reason = "dist_missing"
    logger.warning(
        "Web UI dist not found at %s; serving enablement guidance at %s",
        target,
        WEBUI_MOUNT_PATH,
    )
    _register_missing_webui_routes(app, str(target))
    return WebUiMountResult(mounted=False, directory=str(target), reason=reason)


def _register_missing_webui_routes(app: FastAPI, directory: str) -> None:
    async def missing_webui(request: Request) -> Response:
        return build_missing_webui_response(request, directory)

    # Explicit paths so /app and /app/ both work without a silent framework 404.
    app.add_api_route(
        WEBUI_MOUNT_PATH,
        missing_webui,
        methods=["GET", "HEAD"],
        include_in_schema=False,
        name="webui_missing",
    )
    app.add_api_route(
        f"{WEBUI_MOUNT_PATH}/",
        missing_webui,
        methods=["GET", "HEAD"],
        include_in_schema=False,
        name="webui_missing_slash",
    )
    app.add_api_route(
        f"{WEBUI_MOUNT_PATH}/{{rest:path}}",
        missing_webui,
        methods=["GET", "HEAD"],
        include_in_schema=False,
        name="webui_missing_rest",
    )


def run_webapi(app: FastAPI, settings: WebApiRuntimeSettings | None = None) -> None:
    """Run the FastAPI app with the configured host and port."""
    import uvicorn

    runtime = settings or load_web_api_runtime_settings()
    uvicorn.run(app, host=runtime.host, port=runtime.port)
