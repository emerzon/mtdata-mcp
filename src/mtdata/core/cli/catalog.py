"""Static command catalog used by the lightweight CLI entry point.

Keep this module free of imports from the tool graph.  A parity test guards the
catalog against drift from the dynamically registered MCP tools.
"""

from __future__ import annotations

import os

from ...shared.tool_categories import TOOL_CATEGORY_IDS, tool_catalog_category

# Shared by --help search and the unknown-command execute path.
COMMAND_SUGGESTION_CUTOFF = 0.45

CLI_COMMAND_NAMES = (
    "asset_performance",
    "calendar",
    "causal_discover_signals",
    "cointegration_test",
    "confluence_levels",
    "correlation_matrix",
    "cross_correlation",
    "data_fetch_candles",
    "data_fetch_ticks",
    "denoise_describe",
    "denoise_list_methods",
    "equity_profile",
    "forecast_backtest_run",
    "forecast_barrier_optimize",
    "forecast_barrier_prob",
    "forecast_conformal_intervals",
    "forecast_generate",
    "forecast_list_library_models",
    "forecast_list_methods",
    "forecast_models_cleanup",
    "forecast_models_delete",
    "forecast_models_list",
    "forecast_optimize_hints",
    "forecast_task_cancel",
    "forecast_task_cancel_all",
    "forecast_task_list",
    "forecast_task_status",
    "forecast_task_wait",
    "forecast_train",
    "forecast_tune_genetic",
    "forecast_tune_optuna",
    "forecast_volatility_estimate",
    "indicators_describe",
    "indicators_list",
    "labels_triple_barrier",
    "market_microstructure_analyze",
    "market_radar",
    "market_relative_strength",
    "market_scan",
    "market_snapshot",
    "market_status",
    "market_ticker",
    "market_depth_fetch",
    "news",
    "options_barrier_price",
    "options_chain",
    "options_expirations",
    "options_heston_calibrate",
    "options_provider_status",
    "outliers_detect",
    "patterns_detect",
    "pivot_compute_points",
    "portfolio_risk_decompose",
    "regime_detect",
    "report_generate",
    "screener",
    "seasonality_detect",
    "stationarity_test",
    "strategy_backtest",
    "strategy_validate",
    "support_resistance_levels",
    "symbols_describe",
    "symbols_list",
    "symbols_top_markets",
    "temporal_analyze",
    "tools_list",
    "trade_account_info",
    "trade_close",
    "trade_execution_quality",
    "trade_get_open",
    "trade_get_pending",
    "trade_history",
    "trade_idea_compose",
    "trade_journal_analyze",
    "trade_modify",
    "trade_place",
    "trade_risk_analyze",
    "trade_session_context",
    "trade_stress_test",
    "trade_var_cvar_calculate",
    "volatility_term_structure",
    "volume_profile_levels",
    "wait_event",
)

_OPTIONAL_COMMAND_ENV = {
    "market_depth_fetch": "MTDATA_ENABLE_MARKET_DEPTH_FETCH",
}

MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS = frozenset(
    {
        "causal_discover_signals",
        "correlation_matrix",
        "cointegration_test",
        "cross_correlation",
        "market_radar",
        "market_relative_strength",
        "market_scan",
    }
)

def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def known_command_names() -> tuple[str, ...]:
    """Return the discoverable CLI command catalog."""
    return tuple(sorted(CLI_COMMAND_NAMES))


def display_program_name(argv0: object) -> str:
    """Return a copy-pasteable CLI program name for help and recovery text."""
    basename = os.path.basename(str(argv0 or ""))
    if basename == "__main__.py":
        return "python -m mtdata"
    return basename or "mtdata-cli"


def current_cli_program_name(argv0: object | None = None) -> str:
    """Return the program name for copy-paste remediations in this process."""
    import sys

    raw = argv0 if argv0 is not None else (sys.argv[0] if sys.argv else "")
    path_text = str(raw or "").replace("\\", "/")
    basename = os.path.basename(path_text).lower()
    if basename in {
        "mtdata",
        "mtdata.exe",
        "mtdata-cli",
        "mtdata-cli.exe",
        "cli.py",
    }:
        return display_program_name(raw)
    if basename == "__main__.py":
        parent = os.path.basename(os.path.dirname(path_text)).lower()
        if parent == "mtdata":
            return "python -m mtdata"
    return "mtdata-cli"


def format_root_help(program: str) -> str:
    """Render root help without importing any tool implementation modules."""
    names = known_command_names()
    grouped_names: set[str] = set()
    lines = [
        f"usage: {program} [-h] [-V] [--json] [--output-fields FIELD[,FIELD...]]",
        "                  [--precision MODE]",
        "                  [--timeframe TIMEFRAME] <command> ...",
        "",
        "MetaTrader 5 research, forecasting, and trading CLI. One-shot commands",
        "load only their tool family; use shell for repeated commands in one warm",
        "process. TOON is the default output format; pass --json for JSON.",
        "",
        "Catalog categories (the same IDs are accepted by tools_list):",
        "  shell  Run interactive commands or a stdin batch in one warm process",
    ]
    for category in TOOL_CATEGORY_IDS:
        rows = [name for name in names if tool_catalog_category(name) == category]
        if not rows:
            continue
        grouped_names.update(rows)
        lines.extend(
            (
                "",
                f"  {category.upper()} [tools_list --category {category}]:",
                f"    {' '.join(rows)}",
            )
        )
    remaining = [name for name in names if name not in grouped_names]
    if remaining:
        lines.extend(("", "  OTHER:", f"    {' '.join(remaining)}"))
    lines.extend(
        (
            "",
            "  market_status scope: bare command = major-equity exchange calendar;",
            "                       pass a broker symbol for MT5 tradability.",
        )
    )
    lines.extend(
        (
            "",
            "Global options:",
            "  -h, --help              Show this help and exit",
            "  -V, --version           Show installed mtdata version and exit",
            "  --json                  Output structured JSON instead of TOON",
            "  --output-fields FIELD,...  Select output fields; nested columns",
            "                          use dotted paths such as data.close.",
            "                          Quote trust fields stay with price",
            "                          projections.",
            "  --precision MODE        TOON numeric display precision",
            "  --timeframe TIMEFRAME   Default MT5 timeframe for commands with a",
            "                          timeframe parameter; command-level",
            "                          --timeframe overrides it.",
            "                          For confluence_levels, it defaults",
            "                          --pivot-timeframe instead.",
            "                          For forecast_optimize_hints, it sets a",
            "                          single-item --timeframes search.",
            "",
            f"Run '{program} <command> --help' for command arguments.",
            f"Run '{program} --help <keyword>' to search detailed command help.",
            "Kebab-case command spellings are also accepted.",
        )
    )
    disabled = [
        f"{command} (set {env_name}=1)"
        for command, env_name in _OPTIONAL_COMMAND_ENV.items()
        if not _env_enabled(env_name)
    ]
    if disabled:
        lines.extend(("", "  DISABLED FEATURES:", f"    {'; '.join(disabled)}"))
    return "\n".join(lines)


__all__ = [
    "CLI_COMMAND_NAMES",
    "COMMAND_SUGGESTION_CUTOFF",
    "MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS",
    "known_command_names",
    "format_root_help",
]
