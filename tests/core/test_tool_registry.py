"""Regression test: verify every MCP tool is registered after bootstrap."""

import asyncio
import re
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace

from mtdata.bootstrap.tools import bootstrap_tools, mcp
from mtdata.core._mcp_tools import get_tool_registry, registered_tool_catalog
from mtdata.core.cli.api import _invoke_cli_tool_function
from mtdata.core.error_envelope import build_error_payload
from mtdata.core.request_context import current_request_id
from mtdata.core.tools import tools_list

EXPECTED_TOOL_NAMES = frozenset(
    {
        "asset_performance",
        "calendar",
        "causal_discover_signals",
        "cointegration_test",
        "cross_correlation",
        "confluence_levels",
        "correlation_matrix",
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
        "forecast_task_cancel_all",
        "forecast_task_cancel",
        "forecast_task_list",
        "forecast_task_status",
        "forecast_task_wait",
        "forecast_train",
        "forecast_tune_genetic",
        "forecast_tune_optuna",
        "forecast_volatility_estimate",
        "indicators_describe",
        "indicators_list",
        "outliers_detect",
        "labels_triple_barrier",
        "market_status",
        "market_snapshot",
        "market_radar",
        "market_ticker",
        "market_microstructure_analyze",
        "market_relative_strength",
        "news",
        "options_barrier_price",
        "options_chain",
        "options_expirations",
        "options_heston_calibrate",
        "options_provider_status",
        "patterns_detect",
        "pivot_compute_points",
        "portfolio_risk_decompose",
        "regime_detect",
        "report_generate",
        "screener",
        "strategy_backtest",
        "strategy_validate",
        "stationarity_test",
        "support_resistance_levels",
        "symbols_describe",
        "symbols_list",
        "symbols_top_markets",
        "market_scan",
        "temporal_analyze",
        "seasonality_detect",
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
        "trade_var_cvar_calculate",
        "trade_session_context",
        "trade_stress_test",
        "tools_list",
        "volume_profile_levels",
        "volatility_term_structure",
        "wait_event",
    }
)

_MCP_TOP_LEVEL_FORBIDDEN_SCHEMA_KEYS = frozenset(
    {"oneOf", "anyOf", "allOf", "enum", "not"}
)


def test_tool_count_matches_snapshot():
    bootstrap_tools()
    registered = {t.name for t in mcp._tool_manager.list_tools()}
    expected = set(EXPECTED_TOOL_NAMES)
    if "market_depth_fetch" in registered:
        expected.add("market_depth_fetch")
    missing = expected - registered
    extra = registered - expected
    assert not missing, f"Tools disappeared: {sorted(missing)}"
    assert not extra, f"New tools not in snapshot (update EXPECTED_TOOL_NAMES): {sorted(extra)}"
    assert len(registered) == len(expected)


def test_mcp_disabled_mode_hides_live_mutation_tools(monkeypatch):
    bootstrap_tools()
    monkeypatch.setenv("MTDATA_MCP_TRADING_MODE", "disabled")

    advertised = {tool.name for tool in mcp._tool_manager.list_tools()}
    protocol_advertised = {tool.name for tool in asyncio.run(mcp.list_tools())}

    assert not {"trade_place", "trade_modify", "trade_close"} & advertised
    assert not {"trade_place", "trade_modify", "trade_close"} & protocol_advertised
    assert "trade_get_open" in advertised


def _finish_request_id(caplog):
    finish = next(
        record.message
        for record in caplog.records
        if "event=finish operation=tools_list success=True" in record.message
    )
    match = re.search(r"\brequest_id=([a-f0-9]{12})\b", finish)
    assert match is not None
    return match.group(1)


def test_mcp_tool_invocation_binds_request_id_to_operation_logs(caplog):
    async_wrapper = getattr(tools_list, "_mcp_async_wrapper", None)
    assert callable(async_wrapper)

    with caplog.at_level("DEBUG", logger="mtdata.core.tools"):
        result = asyncio.run(async_wrapper(json=True, limit=1))

    assert result["success"] is True
    assert _finish_request_id(caplog)
    assert current_request_id() is None


def test_cli_tool_invocation_binds_request_id_to_operation_logs(caplog):
    with caplog.at_level("DEBUG", logger="mtdata.core.cli.api"):
        result = _invoke_cli_tool_function(
            tools_list,
            args=None,
            cmd_name="tools_list",
            kwargs={"json": True, "limit": 1},
        )

    assert result["success"] is True
    assert _finish_request_id(caplog)
    assert current_request_id() is None


def test_cli_error_envelope_matches_transport_log_request_id(caplog):
    def fail():
        return build_error_payload("broken", code="test_error")

    with caplog.at_level("WARNING", logger="mtdata.core.cli.api"):
        result = _invoke_cli_tool_function(
            fail,
            args=None,
            cmd_name="failure_probe",
            kwargs={},
        )

    finish = next(
        record.message
        for record in caplog.records
        if "event=finish operation=failure_probe success=False" in record.message
    )
    assert f"request_id={result['request_id']}" in finish
    assert current_request_id() is None


def test_cli_normalizes_bare_invalid_input_envelope():
    def fail():
        return {
            "success": False,
            "error": "Provide at least one symbol or MT5 group for correlation analysis.",
            "error_code": "invalid_input",
        }

    result = _invoke_cli_tool_function(
        fail,
        args=None,
        cmd_name="correlation_matrix",
        kwargs={},
    )

    assert result["error_code"] == "invalid_input"
    assert result["operation"] == "correlation_matrix"
    assert result["request_id"]
    assert "correlation_matrix --help" in result["remediation"]


def test_cli_report_progress_replays_only_structured_stderr():
    def report_probe(*, request):
        print("incidental stdout")
        print("incidental stderr", file=__import__("sys").stderr)
        print(
            "report_generate progress operation=context state=started elapsed=0.1s",
            file=__import__("sys").stderr,
        )
        return {"success": True}

    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = _invoke_cli_tool_function(
            report_probe,
            args=None,
            cmd_name="report_generate",
            kwargs={"request": SimpleNamespace(progress=True)},
        )

    assert result["success"] is True
    assert stdout.getvalue() == ""
    assert stderr.getvalue().strip() == (
        "report_generate progress operation=context state=started elapsed=0.1s"
    )


def test_cli_report_progress_stays_silent_when_not_requested():
    def report_probe(*, request):
        print(
            "report_generate progress operation=context state=started elapsed=0.1s",
            file=__import__("sys").stderr,
        )
        return {"success": True}

    stderr = StringIO()
    with redirect_stderr(stderr):
        _invoke_cli_tool_function(
            report_probe,
            args=None,
            cmd_name="report_generate",
            kwargs={"request": SimpleNamespace(progress=False)},
        )

    assert stderr.getvalue() == ""


def test_tool_public_schemas_match_mcp_top_level_subset():
    bootstrap_tools()
    tool_map = getattr(getattr(mcp, "_tool_manager", None), "_tools", None)
    assert isinstance(tool_map, dict) and tool_map

    issues: list[str] = []
    for name, tool in sorted(tool_map.items()):
        parameters = getattr(tool, "parameters", None)
        if not isinstance(parameters, dict):
            issues.append(f"{name}: parameters is {type(parameters).__name__}")
            continue

        if parameters.get("type") != "object":
            issues.append(
                f"{name}: top-level type is {parameters.get('type')!r}, expected 'object'"
            )

        forbidden_keys = sorted(
            key for key in _MCP_TOP_LEVEL_FORBIDDEN_SCHEMA_KEYS if key in parameters
        )
        if forbidden_keys:
            issues.append(
                f"{name}: forbidden top-level schema keys present: {forbidden_keys}"
            )

    assert not issues, "Invalid MCP tool schemas:\n" + "\n".join(issues)


def test_tools_catalog_standard_detail_includes_parameter_summaries():
    bootstrap_tools()

    compact = registered_tool_catalog(detail="compact")
    standard = registered_tool_catalog(detail="standard")
    full = registered_tool_catalog(detail="full")

    compact_market_scan = next(row for row in compact["tools"] if row["name"] == "market_scan")
    standard_market_scan = next(row for row in standard["tools"] if row["name"] == "market_scan")
    full_market_scan = next(row for row in full["tools"] if row["name"] == "market_scan")

    assert compact["detail"] == "compact"
    assert standard["detail"] == "standard"
    assert full["detail"] == "full"
    assert "parameters" not in compact_market_scan
    assert standard_market_scan["parameters"]["timeframe"] == "optional"
    assert "module" not in standard_market_scan
    assert full["schema_version"] == "1.0"
    assert compact["parameter_schema"]["available_in_detail"] == "full"
    assert full_market_scan["schema_version"] == "1.0"
    assert full_market_scan["parameters"]["timeframe"]["required"] is False
    assert full_market_scan["parameters"]["timeframe"]["type"] == "string"
    assert full_market_scan["parameters"]["timeframe"]["default"] == "H1"
    assert full_market_scan["parameters"]["timeframe"]["cli"]["forms"] == [
        {"kind": "option", "token": "--timeframe"}
    ]
    assert full_market_scan["module"] == "mtdata.core.symbols.scan"


def test_tools_list_full_exposes_nested_invocation_schema_and_cli_forms():
    from mtdata.core.analytics_requests import PortfolioRiskDecomposeRequest

    bootstrap_tools()
    raw_tools_list = getattr(tools_list, "__wrapped__", tools_list)

    out = raw_tools_list(
        search="portfolio_risk_decompose",
        detail="full",
    )

    assert out["schema_version"] == "1.0"
    assert out["count"] == 1
    row = out["tools"][0]
    proposed = row["parameters"]["proposed_trade"]
    assert proposed["$ref"] == "#/$defs/ProposedTrade"
    assert proposed["required"] is False
    assert proposed["default"] is None
    assert proposed["cli"]["available"] is True
    assert proposed["cli"]["value_format"] == "json_object"
    assert proposed["cli"]["forms"] == [
        {"kind": "option", "token": "--proposed-trade"},
        {"kind": "option", "token": "--proposed-trade-params", "role": "alias"},
        {
            "kind": "option",
            "token": "--proposed_trade_params",
            "role": "compatibility_alias",
        },
    ]
    proposed_schema = row["input_schema"]["$defs"]["ProposedTrade"]
    assert proposed_schema["required"] == ["symbol", "side", "volume"]
    assert proposed_schema["properties"]["side"]["enum"] == ["buy", "sell"]
    assert proposed_schema["properties"]["volume"]["exclusiveMinimum"] == 0.0

    # Construct a representative invocation using only discoverable schema data.
    invocation = {
        "proposed_trade": {
            "symbol": "EURUSD",
            "side": proposed_schema["properties"]["side"]["enum"][0],
            "volume": proposed_schema["properties"]["volume"]["exclusiveMinimum"]
            + 0.01,
        }
    }
    request = PortfolioRiskDecomposeRequest(**invocation)
    assert request.proposed_trade is not None
    assert request.proposed_trade.side == "buy"
    assert request.proposed_trade.volume == 0.01


def test_tools_list_full_includes_parser_only_controls_and_aliases():
    bootstrap_tools()
    full = registered_tool_catalog(detail="full")
    rows = {row["name"]: row for row in full["tools"]}

    forecast = rows["forecast_generate"]
    assert {"--set", "--print-config"} <= set(forecast["cli"]["public_tokens"])
    assert {"set_overrides", "print_config"} <= set(
        forecast["cli"]["parser_only_controls"]
    )

    candles = rows["data_fetch_candles"]
    candle_tokens = set(candles["cli"]["public_tokens"])
    assert {"--denoise-params", "--simplify-params", "--set"} <= candle_tokens
    assert {form["token"] for form in candles["parameters"]["denoise"]["cli"]["forms"]} >= {
        "--denoise",
        "--denoise-params",
    }

    history = rows["trade_history"]
    minutes_forms = history["parameters"]["minutes_back"]["cli"]["forms"]
    days = next(form for form in minutes_forms if form["token"] == "--days")
    assert days == {
        "kind": "option",
        "token": "--days",
        "role": "alias",
        "value_transform": "days_to_minutes",
    }
    days_binding = next(
        binding
        for binding in history["cli"]["bindings"]
        if binding["destination"] == "_trade_days"
    )
    assert days_binding["value_transform"] == {
        "operation": "multiply",
        "factor": 1440,
        "target": "minutes_back",
    }


def test_tools_list_cli_inventory_matches_built_command_parsers():
    import argparse

    from mtdata.core.cli.api import _add_tool_command_arguments, get_function_info

    bootstrap_tools()
    full = registered_tool_catalog(detail="full")
    rows = {row["name"]: row for row in full["tools"]}
    registry = get_tool_registry()

    for name in (
        "forecast_generate",
        "data_fetch_candles",
        "trade_history",
        "market_status",
        "forecast_list_library_models",
    ):
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        _add_tool_command_arguments(
            parser,
            cmd_name=name,
            func_info=get_function_info(registry[name]),
        )
        parser_tokens = {
            token
            for action in parser._actions
            for token in action.option_strings
        }
        assert set(rows[name]["cli"]["accepted_tokens"]) == parser_tokens


def test_tools_catalog_full_exposes_trading_defaults_and_venue_namespace():
    bootstrap_tools()
    full = registered_tool_catalog(detail="full")

    trade_place = next(
        row for row in full["tools"] if row["name"] == "trade_place"
    )
    assert trade_place["parameters"]["dry_run"]["default"] is True
    assert trade_place["parameters"]["require_sl_tp"]["default"] is True
    assert "Preview" in trade_place["parameters"]["dry_run"]["description"]
    assert trade_place["parameters"]["volume"]["exclusiveMinimum"] == 0.0
    assert trade_place["parameters"]["symbol"]["cli"]["forms"] == [
        {"kind": "positional", "token": "SYMBOL"},
        {"kind": "option", "token": "--symbol"},
    ]

    market_status = next(
        row for row in full["tools"] if row["name"] == "market_status"
    )
    assert market_status["parameters"]["venue"]["enum"] == [
        "NYSE",
        "NASDAQ",
        "LSE",
        "XETRA",
        "EURONEXT",
        "TSE",
        "HKEX",
        "SSE",
        "ASX",
    ]
    assert market_status["parameters"]["venue"]["cli"]["forms"] == [
        {"kind": "option", "token": "--venue"}
    ]


def test_bootstrap_repairs_temporary_tool_name_overwrite(monkeypatch):
    from mtdata.bootstrap import tools as bootstrap_module
    from mtdata.core import trading
    from mtdata.core._mcp_tools import _TOOL_OBJECT_REGISTRY, _TOOL_REGISTRY

    def trade_place():
        return {"success": True}

    monkeypatch.setattr(bootstrap_module, "_BOOTSTRAPPED_MODULES", {})
    monkeypatch.setattr(bootstrap_module, "_BOOTSTRAPPED_TOOL_FUNCTIONS", {})
    monkeypatch.setattr(bootstrap_module, "_BOOTSTRAPPED_TOOL_OBJECTS", {})
    monkeypatch.setitem(_TOOL_REGISTRY, "trade_place", trade_place)
    monkeypatch.setitem(_TOOL_OBJECT_REGISTRY, "trade_place", trade_place)
    bootstrap_module.bootstrap_tools(("mtdata.core.trading",))

    assert _TOOL_REGISTRY["trade_place"] is trading.trade_place
    assert (
        _TOOL_OBJECT_REGISTRY["trade_place"]
        is trading.trade_place._mcp_tool_object
    )
    full = registered_tool_catalog(detail="full")
    row = next(item for item in full["tools"] if item["name"] == "trade_place")
    assert row["parameters"]["dry_run"]["default"] is True


def test_tools_catalog_full_parameter_contracts_are_machine_described():
    bootstrap_tools()
    full = registered_tool_catalog(detail="full")

    issues = []
    for row in full["tools"]:
        properties = row["input_schema"].get("properties", {})
        for name, parameter in row["parameters"].items():
            if name not in properties:
                issues.append(f"{row['name']}.{name}: absent from input_schema")
            if not isinstance(parameter.get("required"), bool):
                issues.append(f"{row['name']}.{name}: required is not boolean")
            if not str(parameter.get("description") or "").strip():
                issues.append(f"{row['name']}.{name}: description missing")
            if str(parameter.get("description") or "").startswith("Value for "):
                issues.append(f"{row['name']}.{name}: placeholder description")
            cli = parameter.get("cli")
            if not isinstance(cli, dict) or "value_format" not in cli:
                issues.append(f"{row['name']}.{name}: CLI binding missing")

    assert not issues, "Invalid full catalog parameter contracts:\n" + "\n".join(issues)


def test_tools_catalog_documents_consequential_parameter_units_and_policies():
    bootstrap_tools()
    full = registered_tool_catalog(detail="full")
    tools = {row["name"]: row for row in full["tools"]}

    def description(tool: str, parameter: str) -> str:
        return tools[tool]["parameters"][parameter]["description"]

    assert "round-trip" in description("strategy_backtest", "spread_bps")
    assert "basis points" in description("strategy_backtest", "spread_bps")
    assert "after denoise and indicators" in description(
        "data_fetch_candles", "ohlcv"
    )
    assert "full source OHLCV" in description("data_fetch_candles", "ohlcv")
    assert all(
        policy in description("labels_triple_barrier", "same_bar_policy")
        for policy in ("sl_first", "tp_first", "neutral")
    )
    assert "bars of the requested timeframe" in description(
        "portfolio_risk_decompose", "ewma_half_life"
    )
    assert "seconds" in description("trade_execution_quality", "markout_seconds")
    assert "between 0 and 100" in description(
        "volatility_term_structure", "percentiles"
    )
    assert (
        "[{\"id\":\"cross\",\"type\":\"builtin_strategy\","
        "\"strategy\":\"ema_cross\"}]"
        in description("strategy_validate", "candidates")
    )


def test_tools_list_filters_and_paginates_rows():
    bootstrap_tools()
    raw_tools_list = getattr(tools_list, "__wrapped__", tools_list)

    out = raw_tools_list(category="forecast", limit=3, offset=1)

    assert out["success"] is True
    assert out["row_key"] == "tools"
    assert out["filters"] == {"category": "forecast", "search": None}
    assert out["count"] == 3
    assert out["pagination"]["total"] > out["count"]
    assert out["pagination"] == {
        "total": out["pagination"]["total"],
        "returned": 3,
        "offset": 1,
        "limit": 3,
        "has_more": True,
        "more_available": out["pagination"]["total"] - 4,
    }
    assert not {"total_count", "offset", "limit", "has_more"} & out.keys()
    assert all(row["category"] == "forecast" for row in out["tools"])
    assert "categories" not in out
    assert "output_extras" not in out


def test_tools_list_defaults_to_short_page_and_allows_full_explicit_limit():
    bootstrap_tools()
    raw_tools_list = getattr(tools_list, "__wrapped__", tools_list)

    default_page = raw_tools_list()
    full_catalog = raw_tools_list(limit=10_000)

    assert default_page["count"] == 20
    assert default_page["pagination"]["limit"] == 20
    assert default_page["pagination"]["has_more"] is True
    assert default_page["pagination"]["more_available"] > 0
    assert full_catalog["count"] == full_catalog["pagination"]["total"]
    assert full_catalog["pagination"]["has_more"] is False


def test_tools_catalog_full_descriptions_preserve_abbreviations_and_guidance():
    bootstrap_tools()
    catalog = registered_tool_catalog(detail="full")
    tools_by_name = {row["name"]: row for row in catalog["tools"]}

    expected_fragments = {
        ("market_relative_strength", "symbols"): ("e.g. EURUSD,GBPUSD", "group"),
        ("options_chain", "expiration"): ("e.g. 2026-07-17", "Omit"),
        ("wait_event", "symbol"): ("e.g. EURUSD", "max_wait_seconds"),
    }
    for (tool_name, parameter), fragments in expected_fragments.items():
        row = tools_by_name[tool_name]
        schema_description = row["input_schema"]["properties"][parameter][
            "description"
        ]
        catalog_description = row["parameters"][parameter]["description"]
        assert schema_description == catalog_description
        assert not schema_description.endswith("e.g.")
        assert all(fragment in schema_description for fragment in fragments)


def test_tools_catalog_publishes_enforced_numeric_lower_bounds():
    bootstrap_tools()
    catalog = registered_tool_catalog(detail="full")
    tools_by_name = {row["name"]: row for row in catalog["tools"]}

    expected = {
        ("cointegration_test", "min_overlap"): 2,
        ("indicators_list", "offset"): 0,
        ("labels_triple_barrier", "horizon"): 1,
        ("market_scan", "lookback"): 2,
        ("outliers_detect", "lookback"): 20,
        ("seasonality_detect", "lookback"): 31,
        ("stationarity_test", "lookback"): 21,
        ("trade_journal_analyze", "min_sample"): 1,
    }
    for (tool_name, parameter), minimum in expected.items():
        row = tools_by_name[tool_name]
        assert row["input_schema"]["properties"][parameter]["minimum"] == minimum
        assert row["parameters"][parameter]["minimum"] == minimum


def test_tools_list_standard_includes_catalog_metadata():
    bootstrap_tools()
    raw_tools_list = getattr(tools_list, "__wrapped__", tools_list)

    out = raw_tools_list(limit=1, detail="standard")

    assert out["categories"]
    assert out["output_extras"]


def test_tools_list_keeps_disabled_tools_out_of_callable_rows(monkeypatch):
    monkeypatch.delenv("MTDATA_ENABLE_MARKET_DEPTH_FETCH", raising=False)
    bootstrap_tools()
    raw_tools_list = getattr(tools_list, "__wrapped__", tools_list)

    out = raw_tools_list(category="market", search="depth")

    assert out["success"] is True
    assert out["tools"] == []
    assert out["count"] == 0
    assert out["pagination"]["total"] == 0
    assert out["gated_count"] == 1
    assert out["gated_tools"] == [
        {
            "enabled": False,
            "enable_env": "MTDATA_ENABLE_MARKET_DEPTH_FETCH",
            "status": "disabled",
            "why_disabled": "Requires broker Level 2/DOM support and is off by default.",
            "recommended_alternative": "market_ticker",
            "name": "market_depth_fetch",
            "category": "market",
        }
    ]


def test_tools_list_searches_parameters_and_preserves_full_gated_schema(monkeypatch):
    monkeypatch.delenv("MTDATA_ENABLE_MARKET_DEPTH_FETCH", raising=False)
    bootstrap_tools()
    raw_tools_list = getattr(tools_list, "__wrapped__", tools_list)

    parameter_match = raw_tools_list(search="max_wait_seconds")
    assert [row["name"] for row in parameter_match["tools"]] == ["wait_event"]

    gated = raw_tools_list(search="market_depth_fetch", detail="full")
    assert gated["count"] == 0
    assert gated["gated_count"] == 1
    assert "input_schema" in gated["gated_tools"][0]
    assert "cli" in gated["gated_tools"][0]


def test_tools_list_compact_prioritizes_rows_and_pagination():
    bootstrap_tools()
    raw_tools_list = getattr(tools_list, "__wrapped__", tools_list)

    out = raw_tools_list(limit=1)

    assert out["count"] == 1
    assert len(out["tools"]) == 1
    assert "pagination" in out
    assert "parameter_schema" not in out
    assert "mcp_trading" not in out
    assert "gated_tools" not in out


def test_tools_list_related_tools_are_opt_in():
    bootstrap_tools()
    raw_tools_list = getattr(tools_list, "__wrapped__", tools_list)

    hidden = raw_tools_list(search="data_fetch_candles")
    shown = raw_tools_list(search="data_fetch_candles", include_related=True)

    assert "related_tools" not in hidden["tools"][0]
    assert shown["tools"][0]["related_tools"]
