"""Provider-agnostic equity screener."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, Literal, Optional, Union

from pydantic import Field

from ..services.research.capabilities import SCREENER, FinvizResearchSourcePin
from ..services.research.errors import finviz_only_source_error
from ..services.research.payload import stamp_provider
from ..shared.schema import DetailLiteral
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation

logger = logging.getLogger(__name__)

ScreenerView = Literal[
    "overview",
    "valuation",
    "financial",
    "ownership",
    "performance",
    "technical",
]


@mcp.tool()
def screener(
    filters: Annotated[
        Optional[Union[str, Dict[str, Any]]],
        Field(
            description=(
                "Screener filters as JSON, a dict, or provider shorthand. "
                "Filter names are provider-defined."
            )
        ),
    ] = None,
    order: Annotated[
        str,
        Field(
            description=(
                "Sort key. Default -marketcap (largest first). Use price for "
                "ascending price. Pagination follows this provider order."
            )
        ),
    ] = "-marketcap",
    view: Annotated[
        ScreenerView,
        Field(description="Screener column set."),
    ] = "overview",
    list_filters: Annotated[
        bool,
        Field(description="List valid filter names and values instead of screening."),
    ] = False,
    search: Annotated[
        Optional[str],
        Field(description="Filter-catalog search when list_filters is true."),
    ] = None,
    filter_name: Annotated[
        Optional[str],
        Field(description="Exact filter name to describe when list_filters is true."),
    ] = None,
    value_limit: Annotated[
        Optional[int],
        Field(
            ge=1,
            description=(
                "Maximum nested accepted values for an exact filter. Compact "
                "detail defaults to 20; full detail defaults to all values."
            ),
        ),
    ] = None,
    value_offset: Annotated[
        int,
        Field(ge=0, description="Offset within an exact filter's accepted values."),
    ] = 0,
    limit: Annotated[int, Field(ge=1, description="Max rows per page.")] = 20,
    page: Annotated[int, Field(ge=1, description="One-based results page.")] = 1,
    offset: Annotated[
        int,
        Field(
            ge=0,
            description=(
                "Zero-based offset for the filter catalog when list_filters is "
                "true. Not valid in results mode; use page instead."
            ),
        ),
    ] = 0,
    detail: DetailLiteral = "compact",
    source: Annotated[
        FinvizResearchSourcePin,
        Field(
            description="Adapter pin. auto uses every source that can serve this query."
        ),
    ] = "auto",
) -> Dict[str, Any]:
    """Screen equities or list valid screener filters.

    Filter keys stay provider-defined. Finviz is the current adapter;
    ``source="mt5"`` returns a capability error.
    """

    def _run() -> Dict[str, Any]:
        pin_error = finviz_only_source_error(
            source,
            capability=SCREENER,
            operation="screener",
        )
        if pin_error is not None:
            return pin_error
        from .finviz import finviz_filters_list, finviz_screen

        if list_filters:
            payload = finviz_filters_list(
                search=search,
                filter_name=filter_name,
                limit=int(limit),
                offset=int(offset),
                value_limit=value_limit,
                value_offset=int(value_offset),
                detail=str(detail or "compact"),
            )
        elif int(offset) != 0 or value_limit is not None or int(value_offset) != 0:
            invalid = []
            if int(offset) != 0:
                invalid.append("offset")
            if value_limit is not None:
                invalid.append("value_limit")
            if int(value_offset) != 0:
                invalid.append("value_offset")
            return build_error_payload(
                "Value-catalog controls are only valid with list_filters. Use "
                "--page for screener results.",
                code="incompatible_parameters",
                operation="screener",
                details={"invalid": invalid},
                valid_values={
                    "results": [
                        "filters",
                        "order",
                        "view",
                        "limit",
                        "page",
                        "detail",
                    ],
                    "list_filters": [
                        "search",
                        "filter_name",
                        "limit",
                        "offset",
                        "value_limit",
                        "value_offset",
                        "detail",
                    ],
                },
                remediation=(
                    "Drop --offset, or pass --list-filters true to page the "
                    "filter catalog. Use --page for result rows."
                ),
            )
        else:
            payload = finviz_screen(
                filters=filters,
                order=order,
                limit=int(limit),
                page=int(page),
                view=str(view),
                detail=str(detail or "compact"),
            )
        out = stamp_provider(payload, provider="finviz")
        if isinstance(out, dict) and (out.get("success") is False or out.get("error")):
            provider_operation = out.get("operation")
            if provider_operation not in (None, "", "screener"):
                out["provider_operation"] = provider_operation
            out["operation"] = "screener"
        return out

    return run_logged_operation(
        logger,
        operation="screener",
        list_filters=list_filters,
        source=source,
        func=_run,
    )
