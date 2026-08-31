"""Web API bridge for MCP tool catalog listing and safe generic invoke."""

from __future__ import annotations

import copy
import difflib
import inspect
import logging
from typing import Any, Dict, List, Literal, Optional, Sequence, get_args

from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from ..forecast.exceptions import ForecastError
from ..shared.tool_categories import TOOL_CATEGORY_IDS
from ..utils.coercion import UNPARSED_BOOL, parse_strict_bool
from ..utils.denoise import DenoiseCausalityError
from ..utils.mt5 import MT5ConnectionError
from ._mcp_tools import (
    _prepare_public_tool_call,
    _shape_public_tool_output,
    filter_tool_catalog_rows,
    get_tool_functions,
    registered_tool_catalog,
    registered_tool_catalog_entry,
)
from .cli.runtime.commands import LIVE_TRADE_MUTATION_TOOLS, LIVE_TRADE_MUTATION_WARNING
from .execution_logging import infer_result_success
from .output_contract import build_pagination_meta
from .schema_attach import get_public_tool_schema
from .tool_calling import resolve_sync_tool_result, unwrap_tool_callable
from .web_api_handlers import _http_error, _http_status_for_error

logger = logging.getLogger(__name__)

ToolCatalogCategory = Literal[*TOOL_CATEGORY_IDS]
ToolCatalogDetail = Literal["compact", "standard", "full"]
TOOL_CATALOG_CATEGORIES = TOOL_CATEGORY_IDS
TOOL_CATALOG_DETAILS = get_args(ToolCatalogDetail)
TOOLS_CATALOG_DEFAULT_LIMIT = 20
TOOLS_CATALOG_MAX_LIMIT = 1000

# Account / store mutations that must never be one-click unguarded from the SPA.
MUTATING_TOOLS: frozenset[str] = frozenset(
    {
        *LIVE_TRADE_MUTATION_TOOLS,
        "forecast_train",
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
    """Catalog flag: this tool can mutate and may require confirm=true."""
    return str(name or "").strip() in MUTATING_TOOLS


def _coerce_dry_run_flag(value: Any) -> Optional[bool]:
    parsed = parse_strict_bool(value, allow_none=True)
    if parsed is UNPARSED_BOOL or parsed is None:
        return None
    return bool(parsed)


def _dry_run_from_value(value: Any) -> Optional[bool]:
    if isinstance(value, BaseModel):
        fields = getattr(type(value), "model_fields", {})
        if "dry_run" in fields:
            return bool(value.dry_run)
        return None
    if isinstance(value, dict) and "dry_run" in value:
        return _coerce_dry_run_flag(value.get("dry_run"))
    return None


def _effective_dry_run(
    prepared_args: Dict[str, Any],
    *,
    target: Any = None,
) -> Optional[bool]:
    """Return the prepared dry_run flag, or None if the call has no preview mode."""
    if "dry_run" in prepared_args:
        flag = _coerce_dry_run_flag(prepared_args.get("dry_run"))
        if flag is not None:
            return flag
    for value in prepared_args.values():
        flag = _dry_run_from_value(value)
        if flag is not None:
            return flag
    if target is None:
        return None
    try:
        bound = inspect.signature(target).bind_partial(**prepared_args)
        bound.apply_defaults()
    except (TypeError, ValueError):
        return None
    if "dry_run" in bound.arguments:
        flag = _coerce_dry_run_flag(bound.arguments.get("dry_run"))
        if flag is not None:
            return flag
    for value in bound.arguments.values():
        flag = _dry_run_from_value(value)
        if flag is not None:
            return flag
    return None


def _invocation_requires_confirmation(
    name: str,
    prepared_args: Dict[str, Any],
    *,
    target: Any = None,
) -> bool:
    """Confirm only when the prepared call can mutate state."""
    if not tool_requires_confirmation(name):
        return False
    return _effective_dry_run(prepared_args, target=target) is not True


def _http_status_for_tool_result(result: Any) -> int:
    """Map a structured tool failure onto HTTP 4xx/5xx."""
    if not isinstance(result, dict):
        return 400
    code = str(result.get("error_code") or "").strip().lower()
    if (
        "param" in code
        or "validation" in code
        or code in {"invalid_params", "invalid_argument", "tool_param_error"}
    ):
        return 422
    if "not_found" in code:
        return 404
    if "internal" in code:
        return 500
    return _http_status_for_error(result, default=400)


def _raise_failed_tool_result(result: Any, *, operation: str) -> None:
    """Raise an HTTP error whose success flag matches the domain outcome."""
    status = _http_status_for_tool_result(result)
    if isinstance(result, dict):
        error_text = result.get("error")
        code = str(result.get("error_code") or "tool_error")
        if isinstance(error_text, str) and error_text.strip():
            raise _http_error(
                status, result, code=code, operation=operation
            )
        raise _http_error(
            status,
            "Tool invocation failed.",
            code=code,
            operation=operation,
            details={"result": result},
        )
    raise _http_error(
        status,
        "Tool invocation failed.",
        code="tool_error",
        operation=operation,
        details={"result": result} if result is not None else None,
    )


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


def _invalid_catalog_query_error(
    *,
    parameter: str,
    value: str,
    valid: Sequence[str],
    operation: str,
) -> HTTPException:
    valid_values = [str(item) for item in valid]
    requested = str(value or "").strip()
    matches = difflib.get_close_matches(requested, valid_values, n=1, cutoff=0.5)
    suggestion = matches[0] if matches else None
    message = (
        f"Invalid {parameter} '{requested}'. "
        f"Valid values: {', '.join(valid_values)}."
    )
    if suggestion:
        message += f" Did you mean '{suggestion}'?"
    details: Dict[str, Any] = {
        "parameter": parameter,
        "value": requested,
        "valid_values": valid_values,
    }
    if suggestion:
        details["suggestion"] = suggestion
    return _http_error(
        422,
        message,
        code="tool_param_error",
        operation=operation,
        details=details,
    )


def _catalog_detail_mode(detail: str, *, default: str, operation: str) -> str:
    requested = str(detail or default).strip().lower()
    if requested in TOOL_CATALOG_DETAILS:
        return requested
    raise _invalid_catalog_query_error(
        parameter="detail",
        value=str(detail or ""),
        valid=TOOL_CATALOG_DETAILS,
        operation=operation,
    )


def _catalog_category_filter(category: Optional[str], *, operation: str) -> Optional[str]:
    requested = str(category or "").strip().lower()
    if not requested:
        return None
    if requested in TOOL_CATALOG_CATEGORIES:
        return requested
    raise _invalid_catalog_query_error(
        parameter="category",
        value=str(category or ""),
        valid=TOOL_CATALOG_CATEGORIES,
        operation=operation,
    )


def _catalog_limit_value(limit: Any) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError) as exc:
        raise _http_error(
            422,
            "limit must be a positive integer.",
            code="tool_param_error",
            operation="tools_list",
            details={"parameter": "limit", "value": limit},
        ) from exc
    if value < 1 or value > TOOLS_CATALOG_MAX_LIMIT:
        raise _http_error(
            422,
            (
                "limit must be between 1 and "
                f"{TOOLS_CATALOG_MAX_LIMIT}."
            ),
            code="tool_param_error",
            operation="tools_list",
            details={
                "parameter": "limit",
                "value": value,
                "valid_values": {"min": 1, "max": TOOLS_CATALOG_MAX_LIMIT},
            },
        )
    return value


def _catalog_offset_value(offset: Any) -> int:
    try:
        value = int(offset or 0)
    except (TypeError, ValueError) as exc:
        raise _http_error(
            422,
            "offset must be a non-negative integer.",
            code="tool_param_error",
            operation="tools_list",
            details={"parameter": "offset", "value": offset},
        ) from exc
    if value < 0:
        raise _http_error(
            422,
            "offset must be a non-negative integer.",
            code="tool_param_error",
            operation="tools_list",
            details={"parameter": "offset", "value": value},
        )
    return value


def _enrich_catalog_row(row: Dict[str, Any], *, include_fields: bool = False) -> Dict[str, Any]:
    name = str(row.get("name") or "")
    out = dict(row)
    out["surface"] = classify_tool_surface(name)
    out["safety"] = tool_safety_meta(name)
    if include_fields:
        schema = copy.deepcopy(get_public_tool_schema(name))
        properties = schema.get("properties") if isinstance(schema, dict) else None
        if isinstance(properties, dict):
            properties.pop("json", None)
        required = schema.get("required") if isinstance(schema, dict) else None
        if isinstance(required, list):
            schema["required"] = [item for item in required if item != "json"]
        out["input_schema"] = schema
    return out


def list_tools_for_webapi(
    *,
    category: Optional[str] = None,
    search: Optional[str] = None,
    detail: str = "compact",
    include_fields: bool = False,
    limit: int = TOOLS_CATALOG_DEFAULT_LIMIT,
    offset: int = 0,
) -> Dict[str, Any]:
    ensure_tools_bootstrapped()
    detail_mode = _catalog_detail_mode(detail, default="compact", operation="tools_list")
    category_filter = _catalog_category_filter(category, operation="tools_list")
    limit_value = _catalog_limit_value(limit)
    offset_value = _catalog_offset_value(offset)
    catalog = registered_tool_catalog(detail=detail_mode)
    tools = catalog.get("tools") if isinstance(catalog, dict) else []
    if not isinstance(tools, list):
        tools = []

    searchable_names: Optional[set[str]] = None
    search_filter = str(search or "").strip()
    if search_filter and detail_mode != "full":
        search_catalog = registered_tool_catalog(detail="full")
        searchable_rows = (
            search_catalog.get("tools") if isinstance(search_catalog, dict) else []
        )
        if not isinstance(searchable_rows, list):
            searchable_rows = []
        searchable_names = {
            str(row.get("name") or "")
            for row in filter_tool_catalog_rows(
                searchable_rows,
                category=category_filter,
                search=search_filter,
            )
            if isinstance(row, dict)
        }

    filtered = filter_tool_catalog_rows(
        tools,
        category=category_filter,
        search=None if searchable_names is not None else search_filter,
    )
    if searchable_names is not None:
        filtered = [
            row for row in filtered
            if str(row.get("name") or "") in searchable_names
        ]
    start = min(offset_value, len(filtered))
    paged = filtered[start : start + limit_value]
    enriched = [_enrich_catalog_row(row, include_fields=include_fields) for row in paged]

    categories: Dict[str, List[str]] = {}
    surfaces = {"dedicated_ui": 0, "generic_runner": 0, "intentional_omit": 0}
    for row in filtered:
        name = str(row.get("name") or "")
        categories.setdefault(str(row.get("category") or "other"), []).append(name)
        surface = classify_tool_surface(name)
        surfaces[surface] = surfaces.get(surface, 0) + 1

    return {
        "success": True,
        "detail": catalog.get("detail") if isinstance(catalog, dict) else detail_mode,
        "count": len(enriched),
        "categories": categories,
        "surfaces": surfaces,
        "pagination": build_pagination_meta(
            total=len(filtered),
            returned=len(enriched),
            offset=offset_value,
            limit=limit_value,
        ),
        "tools": enriched,
    }


def get_tool_for_webapi(
    tool_name: str,
    *,
    detail: str = "compact",
    include_fields: bool = True,
) -> Dict[str, Any]:
    ensure_tools_bootstrapped()
    name = str(tool_name or "").strip()
    if not name:
        raise _http_error(
            400,
            "tool_name is required",
            code="tool_param_error",
            operation="tools_get",
        )

    detail_mode = _catalog_detail_mode(detail, default="compact", operation="tools_get")
    match = registered_tool_catalog_entry(name, detail=detail_mode)
    if match is None:
        raise _http_error(
            404,
            f"Unknown tool: {name}",
            code="tool_not_found",
            operation=name,
        )

    enriched = _enrich_catalog_row(match, include_fields=include_fields)
    return {"success": True, "detail": detail_mode, "tool": enriched}


def invoke_tool_for_webapi(
    tool_name: str,
    *,
    arguments: Optional[Dict[str, Any]] = None,
    confirm: bool = False,
) -> Dict[str, Any]:
    ensure_tools_bootstrapped()
    name = str(tool_name or "").strip()
    if not name:
        raise _http_error(
            400,
            "tool_name is required",
            code="tool_param_error",
            operation="tools_invoke",
        )

    if name in INTENTIONAL_OMIT_TOOLS:
        raise _http_error(
            403,
            f"Tool {name} is not available from the Web UI.",
            code="tool_not_available",
            operation=name,
            details={
                "rationale": INTENTIONAL_OMIT_TOOLS[name],
                "safety": tool_safety_meta(name),
            },
        )

    funcs = get_tool_functions()
    fn = funcs.get(name)
    if fn is None:
        # market_depth_fetch may be registered only when env-enabled
        raise _http_error(
            404,
            f"Tool {name} is not registered or is disabled.",
            code="tool_not_found",
            operation=name,
            details={"safety": tool_safety_meta(name)},
        )

    args = dict(arguments or {})
    # Strip UI-only keys if callers leak them
    args.pop("confirm", None)
    args.pop("__confirm", None)
    if "extras" in args:
        raise _http_error(
            400,
            "extras was removed; use the tool's detail parameter.",
            code="tool_param_error",
            operation=name,
            details={"parameter": "extras", "replacement": "detail"},
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
        if _invocation_requires_confirmation(name, args, target=target) and not confirm:
            raise _http_error(
                400,
                f"Tool {name} requires explicit confirmation.",
                code="confirmation_required",
                operation=name,
                details={
                    "requires_confirmation": True,
                    "safety": tool_safety_meta(name),
                    "hint": (
                        "Re-submit with confirm=true after reviewing parameters. "
                        "Preview calls with dry_run=true do not need confirm."
                    ),
                },
            )
        result = resolve_sync_tool_result(target(**args))
        result = _shape_public_tool_output(
            result,
            tool_name=name,
            contract_state=contract_state,
            output_fields=output_fields,
        )
    except HTTPException:
        raise
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

    if not infer_result_success(result):
        _raise_failed_tool_result(result, operation=name)

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
