"""Denoise discovery MCP tools."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import Field

from ..utils.denoise import get_denoise_methods_data
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .output_contract import build_pagination_meta, normalize_output_detail

logger = logging.getLogger(__name__)


_DENOISE_METHOD_DEFAULT_LIMIT = 30


def _summary_denoise_method(row: Dict[str, Any]) -> Dict[str, Any]:
    supports = row.get("supports") if isinstance(row.get("supports"), dict) else {}
    causality = supports.get("causality") if isinstance(supports, dict) else None
    params = row.get("params") if isinstance(row.get("params"), list) else []
    return {
        "method": row.get("method"),
        "available": bool(row.get("available", False)),
        "causality": list(causality) if isinstance(causality, list) else [],
        "requires_causality_opt_in": bool(
            row.get("requires_causality_opt_in", False)
        ),
        "requires": row.get("requires") or None,
        "params": list(params),
    }


def _denoise_methods(*, available_only: bool = False) -> List[Dict[str, Any]]:
    data = get_denoise_methods_data()
    rows = data.get("methods") if isinstance(data, dict) else []
    methods = [dict(row) for row in rows if isinstance(row, dict)]
    if available_only:
        methods = [row for row in methods if bool(row.get("available", False))]
    return methods


def _filter_denoise_methods(
    methods: List[Dict[str, Any]],
    *,
    causality: Optional[Literal["causal", "zero_phase"]] = None,
    core_only: bool = False,
) -> List[Dict[str, Any]]:
    filtered = list(methods)
    if core_only:
        filtered = [row for row in filtered if not str(row.get("requires") or "").strip()]
    causality_value = str(causality or "").strip().lower()
    if causality_value:
        if causality_value not in {"causal", "zero_phase"}:
            raise ValueError("Invalid causality filter. Use 'causal' or 'zero_phase'.")
        filtered = [
            row
            for row in filtered
            if causality_value
            in (
                row.get("supports", {}).get("causality", [])
                if isinstance(row.get("supports"), dict)
                else []
            )
        ]
    return filtered


@mcp.tool()
def denoise_list_methods(
    detail: Literal["compact", "standard", "full"] = "compact",
    available_only: bool = False,
    causality: Optional[Literal["causal", "zero_phase"]] = None,
    core_only: bool = False,
    limit: Annotated[int, Field(ge=1)] = _DENOISE_METHOD_DEFAULT_LIMIT,
    offset: Annotated[
        int,
        Field(ge=0, description="Zero-based method offset for pagination."),
    ] = 0,
) -> Dict[str, Any]:
    """List denoise methods, optional dependencies, causality support, and auto params."""

    def _run() -> Dict[str, Any]:
        detail_mode = normalize_output_detail(detail)
        methods = _filter_denoise_methods(
            _denoise_methods(available_only=available_only),
            causality=causality,
            core_only=core_only,
        )
        limit_value = int(limit)
        if limit_value < 1:
            return build_error_payload(
                "limit must be at least 1.",
                code="denoise_invalid_limit",
                operation="denoise_list_methods",
            )
        offset_value = int(offset)
        if offset_value < 0:
            return build_error_payload(
                "offset must be at least 0.",
                code="denoise_invalid_offset",
                operation="denoise_list_methods",
            )
        visible = methods[offset_value : offset_value + limit_value]
        hidden = max(0, len(methods) - offset_value - len(visible))
        compact_mode = detail_mode == "compact"
        method_rows = (
            [
                {
                    "method": row.get("method"),
                    "available": bool(row.get("available", False)),
                    "causality": list(
                        row.get("supports", {}).get("causality", [])
                        if isinstance(row.get("supports"), dict)
                        else []
                    ),
                    "requires_causality_opt_in": bool(
                        row.get("requires_causality_opt_in", False)
                    ),
                }
                for row in visible
            ]
            if compact_mode
            else (
                [_summary_denoise_method(row) for row in visible]
                if detail_mode == "standard"
                else visible
            )
        )
        columns = [
            "method",
            "available",
            "causality",
            "requires_causality_opt_in",
        ]
        if not compact_mode:
            columns.extend(["requires", "params"])
        out = {
            "success": True,
            "detail": detail_mode,
            "available_only": bool(available_only),
            "count": len(visible),
            "pagination": build_pagination_meta(
                total=len(methods),
                returned=len(visible),
                offset=offset_value,
                limit=limit_value,
            ),
            "columns": columns,
            "causality": str(causality).strip().lower() if causality else None,
            "core_only": bool(core_only),
            "methods": method_rows,
            "row_key": "methods",
            "describe_hint": "Use denoise_describe(method) for descriptions and defaults.",
        }
        if hidden > 0:
            out["list_all_hint"] = (
                f"Pass offset={offset_value + len(visible)} to continue listing methods."
            )
        return out

    return run_logged_operation(
        logger,
        operation="denoise_list_methods",
        detail=detail,
        available_only=available_only,
        causality=causality,
        core_only=core_only,
        limit=limit,
        offset=offset,
        func=_run,
    )


@mcp.tool()
def denoise_describe(method: str) -> Dict[str, Any]:
    """Describe one denoise method and its supported options.

    Use ``denoise_list_methods`` to browse method names available in this install.
    """

    def _run() -> Dict[str, Any]:
        wanted = str(method or "").strip().lower()
        methods = _denoise_methods()
        for row in methods:
            if str(row.get("method") or "").strip().lower() == wanted:
                described = dict(row)
                supports = described.get("supports")
                causality = supports.get("causality") if isinstance(supports, dict) else None
                if isinstance(causality, list):
                    has_causal = "causal" in causality
                    has_zero_phase = "zero_phase" in causality
                    if has_zero_phase and not has_causal:
                        described["causality_note"] = (
                            "Zero-phase (non-causal) only: uses future bars, so it introduces "
                            "look-ahead bias. Do not use for live signals or forward testing."
                        )
                    elif has_zero_phase and has_causal:
                        described["causality_note"] = (
                            "Supports causal and zero-phase modes. Use causal for live signals; "
                            "zero-phase uses future bars (look-ahead) and is analysis-only."
                        )
                    elif has_causal:
                        described["causality_note"] = (
                            "Causal: uses only past/current bars; safe for live signals."
                        )
                return {"success": True, "method": described}
        return build_error_payload(
            f"Unknown denoise method {method!r}.",
            code="denoise_method_unknown",
            operation="denoise_describe",
            details={"available_methods": [row.get("method") for row in methods]},
        )

    return run_logged_operation(
        logger,
        operation="denoise_describe",
        method=method,
        func=_run,
    )
