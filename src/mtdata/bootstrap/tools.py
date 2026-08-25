"""Explicit tool-module bootstrap for transport adapters."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Final, Iterable, Optional

from ..core._mcp_instance import mcp
from ..core._mcp_tools import _TOOL_OBJECT_REGISTRY, _TOOL_REGISTRY
from ..core.schema_attach import attach_schemas_to_tools
from ..shared.schema import get_shared_enum_lists

TOOL_MODULE_NAMES: Final[tuple[str, ...]] = (
    "mtdata.core.data",
    "mtdata.core.forecast",
    "mtdata.core.forecast_tasks",
    "mtdata.core.causal",
    "mtdata.core.analytics",
    "mtdata.core.diagnostics",
    "mtdata.core.denoise",
    "mtdata.core.indicators",
    "mtdata.core.market_depth",
    "mtdata.core.market_snapshot",
    "mtdata.core.options",
    "mtdata.core.patterns",
    "mtdata.core.pivot",
    "mtdata.core.volume_profile",
    "mtdata.core.symbols",
    "mtdata.core.regime",
    "mtdata.core.labels",
    "mtdata.core.market_status",
    "mtdata.core.radar",
    "mtdata.core.report",
    "mtdata.core.trading",
    "mtdata.core.temporal",
    "mtdata.core.calendar",
    "mtdata.core.equity_profile",
    "mtdata.core.screener",
    "mtdata.core.asset_performance",
    "mtdata.core.news",
    "mtdata.core.tools",
)

_BOOTSTRAPPED_MODULES: dict[str, ModuleType] = {}
_BOOTSTRAPPED_TOOL_FUNCTIONS: dict[str, object] = {}
_BOOTSTRAPPED_TOOL_OBJECTS: dict[str, object] = {}


def cli_tool_module_names(command: str) -> Optional[tuple[str, ...]]:
    """Return the minimal bootstrap module for a known CLI command family."""
    name = str(command or "").strip().lower().replace("-", "_")
    if not name or name == "tools_list":
        return None
    special = {
        "market_microstructure_analyze": "mtdata.core.analytics",
        "market_relative_strength": "mtdata.core.analytics",
        "portfolio_risk_decompose": "mtdata.core.analytics",
        "strategy_validate": "mtdata.core.analytics",
        "trade_execution_quality": "mtdata.core.analytics",
        "strategy_backtest": "mtdata.core.forecast",
        "volatility_term_structure": "mtdata.core.diagnostics",
        "market_ticker": "mtdata.core.market_depth",
        "market_depth_fetch": "mtdata.core.market_depth",
        "market_snapshot": "mtdata.core.market_snapshot",
        "market_status": "mtdata.core.market_status",
        "market_scan": "mtdata.core.symbols",
        "market_radar": "mtdata.core.radar",
        "support_resistance_levels": "mtdata.core.pivot",
        "confluence_levels": "mtdata.core.pivot",
        "pivot_compute_points": "mtdata.core.pivot",
        "volume_profile_levels": "mtdata.core.volume_profile",
        "news": "mtdata.core.news",
        "calendar": "mtdata.core.calendar",
        "equity_profile": "mtdata.core.equity_profile",
        "screener": "mtdata.core.screener",
        "asset_performance": "mtdata.core.asset_performance",
        "report_generate": "mtdata.core.report",
        "regime_detect": "mtdata.core.regime",
        "patterns_detect": "mtdata.core.patterns",
        "labels_triple_barrier": "mtdata.core.labels",
        "temporal_analyze": "mtdata.core.temporal",
        "wait_event": "mtdata.core.data",
    }
    module_name = special.get(name)
    if module_name is None:
        if name.startswith(("forecast_task_", "forecast_models_")) or name == "forecast_train":
            module_name = "mtdata.core.forecast_tasks"
        elif name.startswith("forecast_"):
            module_name = "mtdata.core.forecast"
        elif name.startswith("trade_"):
            module_name = "mtdata.core.trading"
        elif name.startswith("data_"):
            module_name = "mtdata.core.data"
        elif name.startswith("causal_") or name in {
            "cointegration_test",
            "correlation_matrix",
            "cross_correlation",
        }:
            module_name = "mtdata.core.causal"
        elif name.startswith("denoise_"):
            module_name = "mtdata.core.denoise"
        elif name.startswith("indicators_"):
            module_name = "mtdata.core.indicators"
        elif name.startswith("options_"):
            module_name = "mtdata.core.options"
        elif name.startswith("symbols_"):
            module_name = "mtdata.core.symbols"
        elif name in {"outliers_detect", "seasonality_detect", "stationarity_test"}:
            module_name = "mtdata.core.diagnostics"
    return (module_name,) if module_name in TOOL_MODULE_NAMES else None


def bootstrap_tools(module_names: Optional[Iterable[str]] = None) -> tuple[ModuleType, ...]:
    """Load selected tool modules, or the complete surface when none are selected."""
    requested = tuple(module_names) if module_names is not None else TOOL_MODULE_NAMES
    unknown = [name for name in requested if name not in TOOL_MODULE_NAMES]
    if unknown:
        raise ValueError(f"Unknown tool bootstrap module(s): {', '.join(unknown)}")
    imported_any = False
    for name in requested:
        if name not in _BOOTSTRAPPED_MODULES:
            module = import_module(name)
            _BOOTSTRAPPED_MODULES[name] = module
            module_tools = {
                tool_name: candidate
                for tool_name, candidate in vars(module).items()
                if callable(candidate)
                and str(getattr(candidate, "__module__", "")) == name
                and getattr(candidate, "_mcp_tool_object", None) is not None
            }
            _BOOTSTRAPPED_TOOL_FUNCTIONS.update(module_tools)
            _BOOTSTRAPPED_TOOL_OBJECTS.update(
                {
                    tool_name: candidate._mcp_tool_object
                    for tool_name, candidate in module_tools.items()
                }
            )
            imported_any = True

    # Tests, notebooks, and plugin discovery can register a temporary function
    # under an existing name after the official module has loaded. Reassert the
    # canonical function view so a repeated bootstrap remains deterministic.
    requested_set = set(requested)
    canonical_functions = {
        tool_name: function
        for tool_name, function in _BOOTSTRAPPED_TOOL_FUNCTIONS.items()
        if str(getattr(function, "__module__", "")) in requested_set
    }
    canonical_objects = {
        tool_name: _BOOTSTRAPPED_TOOL_OBJECTS[tool_name]
        for tool_name in canonical_functions
        if tool_name in _BOOTSTRAPPED_TOOL_OBJECTS
    }
    repaired_any = any(
        _TOOL_REGISTRY.get(tool_name) is not function
        for tool_name, function in canonical_functions.items()
    ) or any(
        _TOOL_OBJECT_REGISTRY.get(tool_name) is not tool_object
        for tool_name, tool_object in canonical_objects.items()
    )
    _TOOL_REGISTRY.update(canonical_functions)
    _TOOL_OBJECT_REGISTRY.update(canonical_objects)

    # A warm shell may load another module later. That import triggers one new
    # attachment pass for the expanded registry; repeated calls over the same
    # module set do not need to deep-copy every schema again.
    if imported_any or repaired_any:
        attach_schemas_to_tools(mcp, get_shared_enum_lists())
    return tuple(_BOOTSTRAPPED_MODULES[name] for name in requested)
