"""Tool discovery catalog."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, Literal, Optional

from pydantic import Field

from ..shared.schema import DetailLiteral
from ..shared.tool_categories import TOOL_CATEGORY_IDS
from ._mcp_instance import mcp
from ._mcp_tools import filter_tool_catalog_rows, registered_tool_catalog
from .execution_logging import run_logged_operation
from .output_contract import build_pagination_meta

logger = logging.getLogger(__name__)

_TOOLS_LIST_DEFAULT_LIMIT = 20

ToolCategory = Literal[*TOOL_CATEGORY_IDS]


@mcp.tool()
def tools_list(
    category: Optional[ToolCategory] = None,
    search: Optional[str] = None,
    limit: Annotated[int, Field(ge=1)] = _TOOLS_LIST_DEFAULT_LIMIT,
    offset: Annotated[int, Field(ge=0)] = 0,
    include_related: bool = False,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """List mtdata tools with filters, pagination, and optional parameter summaries."""

    def _run() -> Dict[str, Any]:
        catalog = registered_tool_catalog(detail=detail)
        tools = catalog.get("tools") if isinstance(catalog, dict) else []
        if not isinstance(tools, list):
            return catalog
        try:
            offset_value = int(offset or 0)
        except (TypeError, ValueError):
            return {"error": "offset must be a non-negative integer."}
        if offset_value < 0:
            return {"error": "offset must be a non-negative integer."}
        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            return {"error": "limit must be a positive integer."}
        if limit_value < 1:
            return {"error": "limit must be a positive integer."}
        category_filter = str(category or "").strip().lower()
        search_filter = str(search or "").strip().lower()
        known_categories = set(TOOL_CATEGORY_IDS)
        detail_mode = str(catalog.get("detail") or detail or "compact").strip().lower()
        searchable_names: Optional[set[str]] = None
        if search_filter and detail_mode != "full":
            search_catalog = registered_tool_catalog(detail="full")
            searchable_rows = search_catalog.get("tools")
            exact_names = {
                str(row.get("name") or "")
                for row in searchable_rows if isinstance(row, dict)
                if str(row.get("name") or "").strip().lower() == search_filter
            }
            searchable_names = exact_names or {
                str(row.get("name") or "")
                for row in filter_tool_catalog_rows(
                    searchable_rows,
                    category=category_filter,
                    search=search_filter,
                )
            }
        filtered = []
        filtered_gated = []
        exact_names = {
            str(row.get("name") or "")
            for row in tools if isinstance(row, dict)
            if str(row.get("name") or "").strip().lower() == search_filter
        }
        for row in filter_tool_catalog_rows(
            tools,
            category=category_filter,
            search=(
                None
                if searchable_names is not None or exact_names
                else search_filter
            ),
        ):
            if exact_names and str(row.get("name") or "") not in exact_names:
                continue
            if searchable_names is not None and str(row.get("name") or "") not in searchable_names:
                continue
            if row.get("enabled") is False or row.get("status") == "disabled":
                filtered_gated.append(row)
            else:
                filtered.append(row)

        start = min(offset_value, len(filtered))
        paged = filtered[start : start + limit_value]
        gated_tools: list[Dict[str, Any]] = []
        slimmed: list[Dict[str, Any]] = []
        compact_mode = detail_mode == "compact"
        row_optional_keys = (
            "enabled",
            "enable_env",
            "status",
            "why_disabled",
            "recommended_alternative",
        )
        for row in paged:
            out_row = dict(row)
            if not include_related:
                out_row.pop("related_tools", None)
            if compact_mode:
                for key in row_optional_keys:
                    out_row.pop(key, None)
            slimmed.append(out_row)

        for row in filtered_gated:
            if detail_mode == "full":
                gated = dict(row)
            else:
                gated = {
                    key: row.get(key)
                    for key in row_optional_keys
                    if key in row
                }
                gated["name"] = str(row.get("name") or "")
                gated["category"] = str(row.get("category") or "other")
            gated_tools.append(gated)

        categories: Dict[str, list[str]] = {}
        for row in filtered:
            row_category = str(row.get("category") or "other")
            categories.setdefault(row_category, []).append(str(row.get("name") or ""))
        catalog = dict(catalog)
        catalog["tools"] = slimmed
        catalog["row_key"] = "tools"
        if compact_mode:
            for key in (
                "categories",
                "output_extras",
                "parameter_schema",
                "mcp_trading",
                "schema_version",
                "detail",
            ):
                catalog.pop(key, None)
        else:
            catalog["categories"] = categories
        catalog["count"] = len(slimmed)
        catalog["pagination"] = build_pagination_meta(
            total=len(filtered),
            returned=len(slimmed),
            offset=offset_value,
            limit=limit_value,
        )
        if category_filter or search_filter or not compact_mode:
            catalog["filters"] = {
                "category": category_filter or None,
                "search": search_filter or None,
            }
        if category_filter and category_filter not in known_categories:
            return {
                "success": False,
                "error": (
                    f"Unknown category '{category}'. Valid categories: "
                    + ", ".join(sorted(known_categories))
                    + "."
                ),
                "error_code": "invalid_category",
                "operation": "tools_list",
                "valid_categories": sorted(known_categories),
            }
        if gated_tools and (not compact_mode or bool(category_filter or search_filter)):
            catalog["gated_tools"] = gated_tools
        if gated_tools:
            catalog["gated_count"] = len(gated_tools)
        catalog["catalog_source"] = "rebuilt"
        return catalog

    return run_logged_operation(
        logger,
        operation="tools_list",
        category=category,
        search=search,
        limit=limit,
        offset=offset,
        include_related=include_related,
        detail=detail,
        func=_run,
    )
