"""Web API bridge for MCP tool catalog listing and safe generic invoke."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from pydantic import ValidationError

from ..forecast.exceptions import ForecastError
from ..utils.denoise import DenoiseCausalityError
from ..utils.mt5 import MT5ConnectionError
from ._mcp_tools import (
    _prepare_public_tool_call,
    _shape_public_tool_output,
    filter_tool_catalog_rows,
    get_tool_functions,
    registered_tool_catalog,
)
from .cli.runtime.commands import LIVE_TRADE_MUTATION_TOOLS, LIVE_TRADE_MUTATION_WARNING
from .schema_attach import get_public_tool_schema
from .tool_calling import resolve_sync_tool_result, unwrap_tool_callable
from .web_api_handlers import _http_error

logger = logging.getLogger(__name__)

# Account / store mutations that must never be one-click unguarded from the SPA.
MUTATING_TOOLS: frozenset[str] = frozenset(
    {
        *LIVE_TRADE_MUTATION_TOOLS,
        "forecast_models_delete",
        "forecast_models_cleanup",
        "forecast_task_cancel",
        "forecast_task_cancel_all",
    }
)

# High-traffic research tools with dedicated chart-workspace UX.
DEDICATED_UI_TOOLS: dict[str, str] = {
    "data_fetch_candles": "chart-workspace/history",
    "market_ticker": "chart-workspace/live-quotes",
    "pivot_compute_points": "chart-workspace/pivot-overlay",
    "support_resistance_levels": "chart-workspace/sr-overlay",
    "denoise_list_methods": "chart-workspace/denoise-modal",
    "denoise_describe": "chart-workspace/denoise-modal",
    "forecast_generate": "forecast-panel/price",
    "forecast_volatility_estimate": "forecast-panel/volatility",
    "forecast_backtest_run": "forecast-panel/backtest",
    "forecast_list_methods": "forecast-panel/methods",
    "forecast_models_list": "forecast-panel/models-browser",
    "tools_list": "tools-runner/catalog",
    "trade_idea_compose": "idea-panel/compose",
    "market_radar": "radar-panel/watchlist",
    "confluence_levels": "chart-workspace/confluence-overlay",
    "volume_profile_levels": "chart-workspace/volume-profile-overlay",
    "trade_get_open": "chart-workspace/exposure-overlay",
    "trade_get_pending": "chart-workspace/exposure-overlay",
}

# Product rationale for tools that stay out of the synchronous generic invoke path.
INTENTIONAL_OMIT_TOOLS: dict[str, str] = {
    "forecast_tune_genetic": (
        "Long-running optimization has no HTTP progress or cancellation contract. "
        "Run it through CLI or MCP instead."
    ),
    "forecast_tune_optuna": (
        "Long-running optimization has no HTTP progress or cancellation contract. "
        "Run it through CLI or MCP instead."
    ),
    "wait_event": (
        "Blocking waits have no HTTP progress or cancellation contract. "
        "Run wait_event through CLI or MCP instead."
    ),
}


def ensure_tools_bootstrapped() -> None:
    """Load the full MCP tool surface once for catalog/invoke endpoints."""
    from ..bootstrap.tools import bootstrap_tools

    bootstrap_tools()


def classify_tool_surface(name: str) -> str:
    """Return dedicated_ui | generic_runner | intentional_omit."""
    key = str(name or "").strip()
    if key in INTENTIONAL_OMIT_TOOLS:
        return "intentional_omit"
    if key in DEDICATED_UI_TOOLS:
        return "dedicated_ui"
    return "generic_runner"


def tool_requires_confirmation(name: str) -> bool:
    return str(name or "").strip() in MUTATING_TOOLS


def tool_safety_meta(name: str) -> Dict[str, Any]:
    key = str(name or "").strip()
    meta: Dict[str, Any] = {
        "requires_confirmation": tool_requires_confirmation(key),
        "is_live_trade_mutation": key in LIVE_TRADE_MUTATION_TOOLS,
        "surface": classify_tool_surface(key),
    }
    if key in DEDICATED_UI_TOOLS:
        meta["dedicated_path"] = DEDICATED_UI_TOOLS[key]
    if key in INTENTIONAL_OMIT_TOOLS:
        meta["omit_rationale"] = INTENTIONAL_OMIT_TOOLS[key]
    if key in LIVE_TRADE_MUTATION_TOOLS:
        meta["warning"] = LIVE_TRADE_MUTATION_WARNING
    elif key in MUTATING_TOOLS:
        meta["warning"] = (
            "This tool mutates stored state (models/tasks). "
            "Confirm explicitly before running."
        )
    return meta


def _enrich_catalog_row(row: Dict[str, Any], *, include_fields: bool = False) -> Dict[str, Any]:
    name = str(row.get("name") or "")
    out = dict(row)
    out["surface"] = classify_tool_surface(name)
    out["safety"] = tool_safety_meta(name)
    if include_fields:
        out["input_schema"] = get_public_tool_schema(name)
    return out


def list_tools_for_webapi(
    *,
    category: Optional[str] = None,
    search: Optional[str] = None,
    detail: str = "standard",
    include_fields: bool = False,
) -> Dict[str, Any]:
    ensure_tools_bootstrapped()
    catalog = registered_tool_catalog(detail=detail if detail in {"compact", "standard", "full"} else "standard")
    tools = catalog.get("tools") if isinstance(catalog, dict) else []
    if not isinstance(tools, list):
        tools = []

    enriched: List[Dict[str, Any]] = []
    for row in filter_tool_catalog_rows(
        tools,
        category=category,
        search=search,
    ):
        enriched.append(_enrich_catalog_row(row, include_fields=include_fields))

    categories: Dict[str, List[str]] = {}
    for row in enriched:
        categories.setdefault(str(row.get("category") or "other"), []).append(str(row.get("name") or ""))

    surfaces = {"dedicated_ui": 0, "generic_runner": 0, "intentional_omit": 0}
    for row in enriched:
        surfaces[str(row.get("surface") or "generic_runner")] = surfaces.get(
            str(row.get("surface") or "generic_runner"), 0
        ) + 1

    return {
        "success": True,
        "detail": catalog.get("detail") if isinstance(catalog, dict) else detail,
        "count": len(enriched),
        "categories": categories,
        "surfaces": surfaces,
        "tools": enriched,
    }


def get_tool_for_webapi(tool_name: str) -> Dict[str, Any]:
    ensure_tools_bootstrapped()
    name = str(tool_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="tool_name is required")

    catalog = registered_tool_catalog(detail="full")
    tools = catalog.get("tools") if isinstance(catalog, dict) else []
    match = None
    if isinstance(tools, list):
        for row in tools:
            if isinstance(row, dict) and str(row.get("name") or "") == name:
                match = row
                break
    if match is None:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {name}")

    enriched = _enrich_catalog_row(match, include_fields=True)
    return {"success": True, "tool": enriched}


def invoke_tool_for_webapi(
    tool_name: str,
    *,
    arguments: Optional[Dict[str, Any]] = None,
    confirm: bool = False,
) -> Dict[str, Any]:
    ensure_tools_bootstrapped()
    name = str(tool_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="tool_name is required")

    if name in INTENTIONAL_OMIT_TOOLS:
        raise HTTPException(
            status_code=403,
            detail={
                "error": f"Tool {name} is not available from the Web UI.",
                "rationale": INTENTIONAL_OMIT_TOOLS[name],
            },
        )

    if tool_requires_confirmation(name) and not confirm:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Tool {name} requires explicit confirmation.",
                "requires_confirmation": True,
                "safety": tool_safety_meta(name),
                "hint": "Re-submit with confirm=true after reviewing parameters.",
            },
        )

    funcs = get_tool_functions()
    fn = funcs.get(name)
    if fn is None:
        # market_depth_fetch may be registered only when env-enabled
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Tool {name} is not registered or is disabled.",
                "safety": tool_safety_meta(name),
            },
        )

    args = dict(arguments or {})
    # Strip UI-only keys if callers leak them
    args.pop("confirm", None)
    args.pop("__confirm", None)
    if "extras" in args:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "extras was removed; use the tool's detail parameter.",
                "parameter": "extras",
                "replacement": "detail",
            },
        )
    # The HTTP surface is always structured JSON; consume presentation-only
    # parameters here instead of leaking them into the raw domain callable.
    args.pop("json", None)
    output_fields = args.pop("output_fields", None)

    try:
        target = unwrap_tool_callable(fn)
        contract_state = _prepare_public_tool_call(
            target,
            args,
            json_output=True,
        )
        result = resolve_sync_tool_result(target(**args))
        result = _shape_public_tool_output(
            result,
            tool_name=name,
            contract_state=contract_state,
            output_fields=output_fields,
        )
    except DenoiseCausalityError as exc:
        raise _http_error(
            400, str(exc), code="tool_domain_error", operation=name
        ) from exc
    except (TypeError, ValueError, ValidationError) as exc:
        raise _http_error(
            422,
            f"Invalid parameters for {name}: {exc}",
            code="tool_param_error",
            operation=name,
        ) from exc
    except MT5ConnectionError as exc:
        raise _http_error(
            503, str(exc), code="mt5_connection_error", operation=name
        ) from exc
    except ForecastError as exc:
        raise _http_error(
            400, str(exc), code="tool_domain_error", operation=name
        ) from exc
    except Exception as exc:
        logger.exception("Web API tool invoke failed for %s", name)
        raise _http_error(
            500,
            "Tool invocation failed.",
            code="tool_invoke_internal_error",
            operation=name,
        ) from exc

    return {
        "success": True,
        "tool": name,
        "surface": classify_tool_surface(name),
        "result": result,
    }


def coverage_inventory_rows() -> List[Dict[str, Any]]:
    """Build complete inventory rows for docs/tests (includes gated tools)."""
    ensure_tools_bootstrapped()
    catalog = registered_tool_catalog(detail="compact")
    tools = catalog.get("tools") if isinstance(catalog, dict) else []
    rows: List[Dict[str, Any]] = []
    if isinstance(tools, list):
        for row in tools:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "")
            if not name:
                continue
            surface = classify_tool_surface(name)
            entry: Dict[str, Any] = {
                "name": name,
                "category": row.get("category"),
                "description": row.get("description"),
                "surface": surface,
                "frontend": (
                    DEDICATED_UI_TOOLS.get(name)
                    if surface == "dedicated_ui"
                    else (
                        INTENTIONAL_OMIT_TOOLS.get(name)
                        if surface == "intentional_omit"
                        else "tools-runner/generic"
                    )
                ),
                "requires_confirmation": tool_requires_confirmation(name),
            }
            if name == "market_depth_fetch":
                entry["gated"] = True
                entry["enable_env"] = row.get("enable_env") or "MTDATA_ENABLE_MARKET_DEPTH_FETCH"
                entry["enabled"] = row.get("enabled")
            rows.append(entry)
    rows.sort(key=lambda r: str(r.get("name") or ""))
    return rows
