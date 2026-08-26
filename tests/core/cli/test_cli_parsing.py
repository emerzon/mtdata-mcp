"""Argument parsing tests for mtdata.core.cli module.

Tests argument parsing, parameter coercion, and CLI input normalization.
"""

import argparse
import copy
import json
import logging
import os
import re
import sys
import types
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union
from unittest.mock import MagicMock, call, patch

import pytest
from pydantic import Field, ValidationError

from mtdata.core.data.requests import DataFetchCandlesRequest
from mtdata.core.patterns_requests import PatternsDetectRequest
from mtdata.core.trading.requests import (
    TradeCloseRequest,
    TradeGetOpenRequest,
    TradeHistoryRequest,
    TradeModifyRequest,
    TradePlaceRequest,
    TradeRiskAnalyzeRequest,
)
from mtdata.forecast.requests import (
    ForecastBarrierOptimizeRequest,
    ForecastGenerateRequest,
)
from mtdata.shared.schema import AutoTimeframeLiteral

# ---------------------------------------------------------------------------
# Fixture: ensure the cli module is importable with heavy deps mocked
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Clear env vars that influence debug/colour behaviour between tests."""
    monkeypatch.delenv("MTDATA_CLI_DEBUG", raising=False)
    monkeypatch.delenv("MTDATA_OUTPUT_FORMAT", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("MT5_TIME_OFFSET_MINUTES", raising=False)


# We import lazily inside tests where heavy server machinery is needed,
# but the pure-logic helpers can be imported directly.
from mtdata.core.cli.api import (
    _add_forecast_generate_args,
    _coerce_cli_scalar,
    _example_value,
    _merge_dict,
    _normalize_cli_argv_aliases,
    _normalize_cli_list_value,
    _parse_kv_string,
    _parse_set_overrides,
    _resolve_param_kwargs,
    _resolve_raw_cli_command,
    add_dynamic_arguments,
    get_function_info,
)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def test_non_bar_commands_do_not_receive_global_timeframe() -> None:
    from mtdata.core.cli import api as cli_api

    assert {
        "data_fetch_ticks",
        "symbols_list",
        "tools_list",
    }.issubset(cli_api._TIMEFRAMELESS_GLOBAL_COMMANDS)


@pytest.mark.parametrize(
    ("command", "command_default"),
    [
        ("wait_event", None),
        ("patterns_detect", "ALL"),
        ("volume_profile_levels", None),
    ],
)
def test_global_timeframe_applies_to_mode_switch_commands(
    command: str,
    command_default: Optional[str],
) -> None:
    from mtdata.core.cli import api as cli_api

    args = argparse.Namespace(
        command=command,
        timeframe=command_default,
        _global_timeframe="H4",
        _trade_days=None,
    )
    functions = {
        command: {"_cli_func_info": {"params": [{"name": "timeframe"}]}}
    }

    result = cli_api._apply_global_cli_overrides(
        args,
        ["--timeframe", "H4", command],
        functions=functions,
    )

    assert result.timeframe == "H4"


@pytest.mark.parametrize(
    "command",
    ["trade_history", "trade_journal_analyze", "trade_execution_quality"],
)
def test_trade_lookback_aliases_are_mutually_exclusive(command: str) -> None:
    from mtdata.core.cli import api as cli_api

    args = argparse.Namespace(
        command=command,
        _global_timeframe=None,
        _trade_days=1.0,
        minutes_back=1440,
    )

    with pytest.raises(ValueError, match="cannot be used together"):
        cli_api._apply_global_cli_overrides(
            args,
            [command, "--days", "1", "--minutes-back", "1440"],
        )


def test_shell_timeframe_support_matches_one_shot_inheritance_policy() -> None:
    from mtdata.core.cli import api as cli_api

    def timeframe_tool(timeframe: str = "H1") -> None:
        pass

    functions = {
        name: {
            "func": timeframe_tool,
            "_cli_func_info": {"params": [{"name": "timeframe"}]},
        }
        for name in (
            "data_fetch_candles",
            "wait_event",
            "patterns_detect",
            "volume_profile_levels",
        )
    }

    supported = cli_api._shell_timeframe_commands(functions)

    assert "data_fetch_candles" in supported
    assert "forecast_optimize_hints" in supported
    assert {
        "wait_event",
        "patterns_detect",
        "volume_profile_levels",
    }.issubset(supported)


@pytest.mark.parametrize(
    "argv",
    [
        ["data_fetch_candles", "EURUSD"],
        ["--json", "data_fetch_candles", "EURUSD"],
        ["--output-fields", "success,data", "data_fetch_candles", "EURUSD"],
        ["--output-fields=success,data", "--precision", "full", "data_fetch_candles"],
        ["--timeframe", "H1", "--json", "data_fetch_candles", "EURUSD"],
    ],
)
def test_selective_bootstrap_finds_command_after_global_options(argv) -> None:
    assert _resolve_raw_cli_command(argv) == "data_fetch_candles"


def test_selective_bootstrap_keeps_unknown_leading_option_on_full_discovery() -> None:
    assert _resolve_raw_cli_command(["--bogus", "data_fetch_candles"]) == ""


def test_required_symbol_help_shows_positional_and_flag_forms() -> None:
    from mtdata.core.cli import api as cli_api

    def sample_tool(symbol: str) -> None:
        """Sample tool."""

    parser = cli_api._CLIArgumentParser(
        prog="mtdata-cli sample_tool",
        formatter_class=cli_api._CLIHelpFormatter,
    )
    cli_api.add_dynamic_arguments(
        parser,
        cli_api.get_function_info(sample_tool),
        cmd_name="sample_tool",
    )

    help_text = re.sub(r"\x1b\[[0-9;]*m", "", parser.format_help())
    assert (
        "usage: mtdata-cli sample_tool (SYMBOL | --symbol SYMBOL) [options]"
        in help_text
    )
    assert "Trading symbol (e.g. EURUSD). (required)" in help_text


def test_disabled_market_depth_parse_error_explains_gate(monkeypatch, capsys):
    from mtdata.core.cli import api as cli_api

    monkeypatch.delenv("MTDATA_ENABLE_MARKET_DEPTH_FETCH", raising=False)
    monkeypatch.setattr(sys, "argv", ["mtdata-cli", "market_depth_fetch", "--json"])
    parser = cli_api._CLIArgumentParser(prog="mtdata-cli")

    with pytest.raises(SystemExit, match="2"):
        parser.error("invalid choice: 'market_depth_fetch'")

    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "feature_disabled"
    assert payload["details"]["enable_env"] == "MTDATA_ENABLE_MARKET_DEPTH_FETCH"
    assert "Level 2/DOM" in payload["error"]


def test_disabled_market_depth_help_is_available(monkeypatch):
    from mtdata.core.cli import api as cli_api

    monkeypatch.delenv("MTDATA_ENABLE_MARKET_DEPTH_FETCH", raising=False)

    def sample_tool(symbol: str, spread: bool = False) -> None:
        """Return DOM when enabled."""

    parser = cli_api._CLIArgumentParser(
        prog="mtdata-cli market_depth_fetch",
        formatter_class=cli_api._CLIHelpFormatter,
    )
    cli_api.add_dynamic_arguments(
        parser,
        cli_api.get_function_info(sample_tool),
        cmd_name="market_depth_fetch",
    )
    help_text = parser.format_help()
    assert "usage:" in help_text.lower()
    assert "--spread" in help_text


@pytest.mark.parametrize(
    ("message", "error_code", "error_fragment"),
    [
        (
            "argument --timeframe: invalid choice: 'H7'",
            "cli_invalid_arguments",
            "invalid choice",
        ),
        (
            "the following arguments are required: symbol",
            "cli_missing_required",
            "Missing required argument(s): symbol.",
        ),
        (
            "unrecognized arguments: --bogus 1",
            "cli_invalid_arguments",
            "unrecognized arguments",
        ),
    ],
)
def test_default_toon_parse_errors_use_structured_stdout_envelope(
    monkeypatch,
    capsys,
    message,
    error_code,
    error_fragment,
):
    from mtdata.core.cli import api as cli_api

    monkeypatch.setattr(sys, "argv", ["mtdata-cli", "sample_tool"])
    parser = cli_api._CLIArgumentParser(prog="mtdata-cli sample_tool")

    with pytest.raises(SystemExit, match="2"):
        parser.error(message)

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "success: false" in captured.out
    assert f"error_code: {error_code}" in captured.out
    assert "operation: sample_tool" in captured.out
    assert "remediation:" in captured.out
    assert error_fragment in captured.out


@pytest.mark.parametrize(
    ("command", "message", "argv", "remediation_fragment"),
    [
        (
            "patterns_detect",
            "unrecognized arguments: --limit 100",
            ["mtdata-cli", "patterns_detect", "--limit", "100"],
            "Use --lookback for history bars",
        ),
        (
            "regime_detect",
            "unrecognized arguments: --limit 100",
            ["mtdata-cli", "regime_detect", "--limit", "100"],
            "Use --fetch-limit",
        ),
        (
            "trade_risk_analyze",
            "unrecognized arguments: --volume 0.1",
            ["mtdata-cli", "trade_risk_analyze", "EURUSD", "--volume", "0.1"],
            "computes suggested_volume",
        ),
        (
            "market_microstructure_analyze",
            "unrecognized arguments: --timeframe M5",
            ["mtdata-cli", "market_microstructure_analyze", "EURUSD", "--timeframe", "M5"],
            "Use --minutes-back",
        ),
    ],
)
def test_near_miss_flag_parse_errors_use_specific_remediation(
    monkeypatch,
    capsys,
    command,
    message,
    argv,
    remediation_fragment,
):
    from mtdata.core.cli import api as cli_api

    monkeypatch.setattr(sys, "argv", argv)
    parser = cli_api._CLIArgumentParser(prog=f"mtdata-cli {command}")

    with pytest.raises(SystemExit, match="2"):
        parser.error(message)

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "success: false" in captured.out
    assert "error_code: cli_invalid_arguments" in captured.out
    assert remediation_fragment in captured.out


def test_missing_barriers_remediation_includes_example(monkeypatch, capsys):
    from mtdata.core.cli import api as cli_api

    monkeypatch.setattr(
        sys,
        "argv",
        ["mtdata-cli", "labels_triple_barrier", "EURUSD", "--timeframe", "H1"],
    )
    parser = cli_api._CLIArgumentParser(prog="mtdata-cli labels_triple_barrier")

    with pytest.raises(SystemExit, match="2"):
        parser.error("the following arguments are required: --barriers")

    captured = capsys.readouterr()
    assert "error_code: cli_missing_required" in captured.out
    assert "unit=pct take_profit=0.5 stop_loss=0.5" in captured.out


def test_missing_order_type_mentions_side_alias(monkeypatch, capsys):
    from mtdata.core.cli import api as cli_api

    monkeypatch.setattr(
        sys,
        "argv",
        ["mtdata-cli", "trade_place", "EURUSD", "--volume", "0.01"],
    )
    parser = cli_api._CLIArgumentParser(prog="mtdata-cli trade_place")

    with pytest.raises(SystemExit, match="2"):
        parser.error("the following arguments are required: --order-type")

    captured = capsys.readouterr()
    assert "error_code: cli_missing_required" in captured.out
    assert "--order-type BUY or SELL" in captured.out
    assert "--side buy/sell" in captured.out


def test_dynamic_cli_help_has_no_placeholder_param_text():
    from mtdata.bootstrap.settings import load_environment
    from mtdata.core.cli import api as cli_api

    load_environment()
    functions = cli_api.discover_tools()
    parser = cli_api._CLIArgumentParser(prog="mtdata-cli")
    cli_api.add_global_args_to_parser(parser, exclude_params=["timeframe"])
    parser.add_argument(
        "--timeframe",
        dest="_global_timeframe",
        default=argparse.SUPPRESS,
        metavar="TIMEFRAME",
    )
    subparsers = parser.add_subparsers(dest="command")
    command_parsers = {}
    forecast_tool = None

    for cmd_name, tool in sorted(functions.items()):
        func = tool["func"]
        func_info = tool.setdefault("_cli_func_info", cli_api.get_function_info(func))
        cli_api._apply_schema_overrides(tool, func_info)
        if cmd_name == "forecast_generate":
            forecast_tool = (tool, func_info)
            continue

        cmd_parser = subparsers.add_parser(cmd_name)
        exclude_globals = [p["name"] for p in func_info["params"]]
        if cmd_name == "report_generate":
            exclude_globals.append("timeframe")
        if cmd_name in {
            "news",
            "calendar",
            "equity_profile",
            "screener",
            "asset_performance",
        } or cmd_name in cli_api._TIMEFRAMELESS_GLOBAL_COMMANDS:
            exclude_globals.append("timeframe")
        cli_api.add_global_args_to_parser(
            cmd_parser,
            exclude_params=exclude_globals,
            suppress_defaults=True,
        )
        cli_api.add_dynamic_arguments(
            cmd_parser,
            func_info,
            (tool.get("meta") or {}).get("param_docs"),
            cmd_name=cmd_name,
        )
        command_parsers[cmd_name] = cmd_parser

    if forecast_tool is not None:
        cmd_parser = subparsers.add_parser("forecast_generate")
        cli_api.add_global_args_to_parser(
            cmd_parser,
            exclude_params=["symbol", "timeframe"],
            suppress_defaults=True,
        )
        cli_api._add_forecast_generate_args(cmd_parser)
        command_parsers["forecast_generate"] = cmd_parser

    placeholder = re.compile(r"^[A-Za-z_][A-Za-z0-9_]* parameter$")
    offenders = []
    for cmd_name, cmd_parser in sorted(command_parsers.items()):
        for action in cmd_parser._actions:
            help_text = getattr(action, "help", None)
            if isinstance(help_text, str) and placeholder.match(help_text):
                offenders.append(f"{cmd_name}.{action.dest}: {help_text}")

    assert offenders == []


# ========================================================================
# _add_forecast_generate_args
# ========================================================================


class TestAddForecastGenerateArgs:
    def test_adds_args(self):
        parser = argparse.ArgumentParser()
        _add_forecast_generate_args(parser)
        # Should parse without error when given required args
        args = parser.parse_args(["EURUSD"])
        assert args.symbol_positional == "EURUSD"
        assert not hasattr(args, "library")
        assert args.method == "theta"
        assert args.timeframe == "H1"
        assert args.horizon == 12
        assert args.detail == "compact"

    def test_all_options(self):
        parser = argparse.ArgumentParser()
        _add_forecast_generate_args(parser)
        args = parser.parse_args(
            [
                "GBPUSD",
                "--library",
                "pretrained",
                "--method",
                "chronos2",
                "--timeframe",
                "D1",
                "--horizon",
                "24",
                "--lookback",
                "200",
                "--quantity",
                "return",
                "--ci-alpha",
                "0.1",
                "--denoise",
                "wavelet",
                "--print-config",
            ]
        )
        assert args.symbol_positional == "GBPUSD"
        assert args.library == "pretrained"
        assert args.method == "chronos2"
        assert args.horizon == 24
        assert args.lookback == 200
        assert args.quantity == "return"
        assert args.ci_alpha == 0.1
        assert args.detail == "compact"
        assert args.print_config is True

    def test_symbol_accepts_flag_alias(self):
        parser = argparse.ArgumentParser()
        _add_forecast_generate_args(parser)
        args = parser.parse_args(["--symbol", "GBPUSD"])
        assert args.symbol == "GBPUSD"

    def test_choice_values_are_case_insensitive(self):
        parser = argparse.ArgumentParser()
        _add_forecast_generate_args(parser)
        args = parser.parse_args(
            [
                "EURUSD",
                "--library",
                "NATIVE",
                "--timeframe",
                "h1",
                "--quantity",
                "RETURN",
                "--proxy",
                "ABS_RETURN",
                "--detail",
                "FULL",
                "--model-cache",
                "EPHEMERAL",
            ]
        )

        assert args.library == "native"
        assert args.timeframe == "H1"
        assert args.quantity == "return"
        assert args.proxy == "abs_return"
        assert args.detail == "full"
        assert args.model_cache == "ephemeral"

    def test_detail_accepts_summary(self):
        parser = argparse.ArgumentParser()
        _add_forecast_generate_args(parser)
        args = parser.parse_args(["EURUSD", "--detail", "summary"])
        assert args.detail == "summary"

    def test_method_help_lists_registered_methods_without_restricting_custom_paths(self):
        parser = argparse.ArgumentParser()
        _add_forecast_generate_args(parser)

        help_text = parser.format_help()
        assert "Registered" in help_text
        assert "built-in methods:" in help_text
        assert "theta" in help_text
        assert "forecast_list_methods" in help_text

        args = parser.parse_args(
            ["EURUSD", "--method", "sklearn.ensemble.RandomForestRegressor"]
        )
        assert args.method == "sklearn.ensemble.RandomForestRegressor"

    def test_parser_includes_typed_value_help_epilog(self):
        parser = argparse.ArgumentParser()
        _add_forecast_generate_args(parser)

        assert "Typed Value Formats:" in parser.epilog
        assert "--denoise PRESET|JSON" in parser.epilog

    def test_parser_allows_bare_presence_for_typed_flags(self):
        parser = argparse.ArgumentParser()
        _add_forecast_generate_args(parser)

        args = parser.parse_args(["BTCUSD", "--denoise", "--params"])

        assert args.denoise == "__PRESENT__"
        assert args.params == "__PRESENT__"

    def test_parser_accepts_denoise_params_companion(self):
        parser = argparse.ArgumentParser()
        _add_forecast_generate_args(parser)
        args = parser.parse_args(
            ["EURUSD", "--denoise", "ema", "--denoise-params", "alpha=0.2"]
        )
        assert args.denoise == "ema"
        assert args.denoise_params == "alpha=0.2"

    def test_parser_exposes_detail_flag(self):
        parser = argparse.ArgumentParser()
        _add_forecast_generate_args(parser)

        help_text = parser.format_help()
        assert "--detail" in help_text
        args = parser.parse_args(["BTCUSD", "--detail", "full"])
        assert args.detail == "full"

    def test_horizon_help_explains_forecast_time_and_value_semantics(self):
        parser = argparse.ArgumentParser()
        _add_forecast_generate_args(parser)

        help_text = next(
            action.help for action in parser._actions if action.dest == "horizon"
        )
        assert "forecast_time identifies each target bar's open" in help_text
        assert "currently forming bar" in help_text
        assert "bar_state distinguishes forming from future" in help_text

    @patch("mtdata.core.cli.api.discover_tools")
    def test_main_shows_targeted_error_for_bare_denoise(self, mock_discover, capsys):
        from mtdata.core.cli.api import main

        mock_fn = MagicMock(return_value="ok")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."
        mock_discover.return_value = {
            "forecast_generate": {"func": mock_fn, "meta": {"description": "Generate forecasts"}},
        }

        with patch("sys.argv", ["cli.py", "forecast_generate", "BTCUSD", "--denoise"]), pytest.raises(SystemExit):
            main()

        captured = capsys.readouterr()
        assert captured.err == ""
        assert "success: false" in captured.out
        assert "error_code: cli_invalid_arguments" in captured.out
        assert "--denoise expects a value." in captured.out
        assert "--denoise ema" in captured.out
        mock_fn.assert_not_called()

    @patch("mtdata.core.cli.api.discover_tools")
    def test_main_allows_bare_denoise_when_set_supplies_value(self, mock_discover):
        from mtdata.core.cli.api import main

        mock_fn = MagicMock(return_value="ok")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."
        mock_discover.return_value = {
            "forecast_generate": {"func": mock_fn, "meta": {"description": "Generate forecasts"}},
        }

        with patch("sys.argv", ["cli.py", "forecast_generate", "BTCUSD", "--denoise", "--set", "denoise.method=ema"]):
            result = main()

        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert request.denoise == {"method": "ema"}

    @patch("mtdata.core.cli.api.discover_tools")
    def test_main_accepts_denoise_params_companion(self, mock_discover):
        from mtdata.core.cli.api import main

        mock_fn = MagicMock(return_value="ok")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."
        mock_discover.return_value = {
            "forecast_generate": {"func": mock_fn, "meta": {"description": "Generate forecasts"}},
        }

        with patch(
            "sys.argv",
            [
                "cli.py",
                "forecast_generate",
                "EURUSD",
                "--denoise",
                "ema",
                "--denoise-params",
                "alpha=0.2",
            ],
        ):
            result = main()

        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert request.denoise == {"method": "ema", "params": {"alpha": 0.2}}


@pytest.mark.parametrize(
    ("cmd_name", "param_name", "annotation", "expected_flag"),
    [
        ("market_status", "symbol", Optional[str], "--symbol"),
        ("correlation_matrix", "symbols", Optional[str], "--symbols"),
        (
            "forecast_list_library_models",
            "library",
            Optional[Literal["native", "all"]],
            "--library",
        ),
    ],
)
def test_optional_positional_named_forms_are_visible(
    cmd_name, param_name, annotation, expected_flag,
):
    parser = argparse.ArgumentParser()
    func_info = {
        "params": [
            {
                "name": param_name,
                "type": annotation,
                "required": False,
                "default": None,
            }
        ]
    }

    add_dynamic_arguments(parser, func_info, cmd_name=cmd_name)

    help_text = _strip_ansi(parser.format_help())
    assert expected_flag in help_text


# ========================================================================
# add_dynamic_arguments
# ========================================================================


class TestAddDynamicArguments:
    def test_adds_required_positional(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
            ]
        }
        add_dynamic_arguments(parser, func_info)
        args = parser.parse_args(["EURUSD"])
        assert args.symbol == "EURUSD"

    def test_adds_optional_flags(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {"name": "count", "type": int, "required": False, "default": 10},
            ]
        }
        add_dynamic_arguments(parser, func_info)
        args = parser.parse_args(["EURUSD", "--count", "20"])
        assert args.count == 20

    def test_trade_place_marks_volume_and_order_type_required(self):
        parser = argparse.ArgumentParser()

        def tool(request: TradePlaceRequest):
            pass

        func_info = get_function_info(tool)
        add_dynamic_arguments(parser, func_info, cmd_name="trade_place")

        with pytest.raises(SystemExit):
            parser.parse_args(["EURUSD"])
        args = parser.parse_args(
            ["EURUSD", "--volume", "0.01", "--order-type", "BUY"]
        )
        assert args.volume == 0.01
        assert args.order_type == "BUY"
        assert parser.parse_args(
            ["EURUSD", "--volume", "0.01", "--order-type", "buy-stop"]
        ).order_type == "BUY_STOP"
        assert parser.parse_args(
            ["EURUSD", "--volume", "0.01", "--order-type", "BUY STOP"]
        ).order_type == "BUY_STOP"
        assert parser.parse_args(
            ["EURUSD", "--volume", "0.01", "--side", "buy"]
        ).order_type == "BUY"
        help_text = parser.format_help()
        assert "volume" in help_text and "(required)" in help_text
        assert "--side" in help_text

    def test_bool_param(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "flag", "type": bool, "required": False, "default": None},
            ]
        }
        add_dynamic_arguments(parser, func_info)
        args = parser.parse_args(["--flag"])
        assert args.flag == "true"
        args = parser.parse_args(["--flag", "true"])
        assert args.flag == "true"
        args = parser.parse_args(["--flag", "True"])
        assert args.flag == "true"
        args = parser.parse_args(["--flag", "false"])
        assert args.flag == "false"
        args = parser.parse_args(["--flag", "FALSE"])
        assert args.flag == "false"
        args = parser.parse_args(["--no-flag"])
        assert args.flag == "false"
        help_text = _strip_ansi(parser.format_help())
        assert "--flag [{true,false}]" in help_text
        assert "[bool]" not in help_text

    def test_data_fetch_candle_bool_help_mentions_defaults(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "include_spread",
                    "type": bool,
                    "required": False,
                    "default": False,
                },
                {
                    "name": "allow_stale",
                    "type": bool,
                    "required": False,
                    "default": False,
                },
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="data_fetch_candles")

        help_text = _strip_ansi(parser.format_help())

        assert "--include-spread [{true,false}]" in help_text
        assert "defaults to false" in help_text
        assert "--allow-stale [{true,false}]" in help_text
        assert "freshness" in help_text
        assert "freshness_policy_relaxed" in help_text

    def test_include_incomplete_bool_param_uses_canonical_hyphen_flag(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "include_incomplete",
                    "type": bool,
                    "required": False,
                    "default": False,
                },
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="data_fetch_candles")

        canonical_action = next(
            action
            for action in parser._actions
            if action.dest == "include_incomplete" and action.help != argparse.SUPPRESS
        )
        hidden_alias_action = next(
            action
            for action in parser._actions
            if action.dest == "include_incomplete"
            and action.help == argparse.SUPPRESS
            and "--include_incomplete" in action.option_strings
        )

        assert canonical_action.option_strings == ["--include-incomplete"]
        assert "--include_incomplete" in hidden_alias_action.option_strings
        assert not any(
            action.help != argparse.SUPPRESS
            and action.dest == "include_incomplete"
            and "--no-include-incomplete" in action.option_strings
            for action in parser._actions
        )
        assert parser.parse_args(["--include_incomplete"]).include_incomplete == "true"
        assert (
            parser.parse_args(["--no_include_incomplete"]).include_incomplete == "false"
        )
        for token in ("1", "yes", "on"):
            assert (
                parser.parse_args(["--include-incomplete", token]).include_incomplete
                == "true"
            )
        for token in ("0", "no", "off"):
            assert (
                parser.parse_args(["--include-incomplete", token]).include_incomplete
                == "false"
            )

    def test_market_scan_help_uses_plural_symbols_parameter(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "symbols",
                    "type": Optional[str],
                    "required": False,
                    "default": None,
                },
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="market_scan")

        help_text = _strip_ansi(parser.format_help())

        assert "--symbols SYMBOLS" in help_text
        assert "Comma-separated MT5 symbols" in help_text
        assert parser.parse_args(["EURUSD", "GBPUSD"]).symbols == ["EURUSD", "GBPUSD"]
        assert (
            parser.parse_args(["--symbols", "EURUSD,GBPUSD"])._cli_option_symbols
            == ["EURUSD,GBPUSD"]
        )

    def test_market_radar_accepts_optional_positional_symbols(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "symbols",
                    "type": Optional[str],
                    "required": False,
                    "default": None,
                },
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="market_radar")

        help_text = _strip_ansi(parser.format_help())

        assert "[symbols ...]" in help_text
        assert "--symbols SYMBOLS" in help_text
        assert parser.parse_args(["EURUSD", "GBPUSD"]).symbols == ["EURUSD", "GBPUSD"]
        assert (
            parser.parse_args(["--symbols", "EURUSD,GBPUSD"])._cli_option_symbols
            == ["EURUSD,GBPUSD"]
        )

    def test_non_positional_required_parameters_are_required_options(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "spot",
                    "type": float,
                    "required": True,
                    "default": None,
                },
                {
                    "name": "strike",
                    "type": float,
                    "required": True,
                    "default": None,
                },
                {
                    "name": "barrier",
                    "type": float,
                    "required": True,
                    "default": None,
                },
            ]
        }

        add_dynamic_arguments(parser, func_info, cmd_name="options_barrier_price")

        help_text = _strip_ansi(parser.format_help())
        assert "--strike STRIKE" in help_text
        assert "--barrier BARRIER" in help_text
        assert help_text.count("(required)") == 3
        with pytest.raises(SystemExit):
            parser.parse_args(["100"])
        parsed = parser.parse_args(["100", "--strike", "105", "--barrier", "90"])
        assert parsed.strike == 105.0
        assert parsed.barrier == 90.0

    def test_list_param(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "items",
                    "type": List[str],
                    "required": False,
                    "default": None,
                },
            ]
        }
        add_dynamic_arguments(parser, func_info)
        args = parser.parse_args(["--items", "a", "b", "c"])
        assert args.items == ["a", "b", "c"]

    def test_mapping_param_adds_companion(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "simplify",
                    "type": Dict[str, Any],
                    "required": False,
                    "default": None,
                },
            ]
        }
        add_dynamic_arguments(parser, func_info)
        args = parser.parse_args(
            ["--simplify", "lttb", "--simplify-params", "points=100"]
        )
        assert args.simplify == "lttb"
        assert args.simplify_params == "points=100"

    def test_required_mapping_option_requires_value(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {"name": "barrier", "type": Dict[str, Any], "required": True, "default": None},
            ]
        }

        add_dynamic_arguments(parser, func_info, cmd_name="forecast_barrier_prob")

        help_text = _strip_ansi(parser.format_help())
        assert "--barrier BARRIER" in help_text
        with pytest.raises(SystemExit):
            parser.parse_args(["EURUSD", "--barrier"])
        parsed = parser.parse_args(["EURUSD", "--barrier", "kind=tp_sl"])
        assert parsed.barrier == "kind=tp_sl"

    def test_required_symbols_option_consumes_space_separated_values(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "symbols", "type": str, "required": True, "default": None},
            ]
        }

        add_dynamic_arguments(parser, func_info, cmd_name="cross_correlation")

        parsed = parser.parse_args(["--symbols", "EURUSD", "GBPUSD"])
        assert parsed._cli_option_symbols == ["EURUSD", "GBPUSD"]

    def test_mapping_param_adds_set_override(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "params",
                    "type": Dict[str, Any],
                    "required": False,
                    "default": None,
                },
            ]
        }
        add_dynamic_arguments(parser, func_info)
        help_text = _strip_ansi(parser.format_help())
        args = parser.parse_args(["--params", "alpha=0.5", "--set", "params.beta=0.2"])
        assert args.params == "alpha=0.5"
        assert args.set_overrides == ["params.beta=0.2"]
        assert "--set" in help_text
        assert "--params-params" not in help_text

    def test_nullable_float_accepts_none_and_null_tokens(self):
        parser = argparse.ArgumentParser()

        def levels(
            max_distance_pct: Annotated[Optional[float], Field(ge=0.0)] = 5.0,
        ):
            return max_distance_pct

        func_info = get_function_info(levels)
        add_dynamic_arguments(parser, func_info, cmd_name="support_resistance_levels")
        assert parser.parse_args(["--max-distance-pct", "none"]).max_distance_pct is None
        assert parser.parse_args(["--max-distance-pct", "NULL"]).max_distance_pct is None
        assert parser.parse_args(["--max-distance-pct", "2.5"]).max_distance_pct == 2.5
        with pytest.raises(SystemExit):
            parser.parse_args(["--max-distance-pct", "-1"])

    def test_annotated_bool_parser_accepts_false_token(self):
        parser = argparse.ArgumentParser()

        def calendar_like(
            upcoming: Annotated[Optional[bool], Field(description="Keep unreleased.")] = None,
        ):
            return upcoming

        func_info = get_function_info(calendar_like)
        add_dynamic_arguments(parser, func_info, cmd_name="calendar")
        args = parser.parse_args(["--upcoming", "false"])
        assert args.upcoming == "false"

    def test_omitted_required_symbol_alias_has_no_namespace_attribute(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="symbols_describe")
        args = parser.parse_args([])
        assert not hasattr(args, "symbol")
        assert not hasattr(args, "_cli_option_symbol")

    def test_first_required_param_accepts_flag_alias(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {"name": "count", "type": int, "required": False, "default": 10},
            ]
        }
        add_dynamic_arguments(parser, func_info)
        args = parser.parse_args(["--symbol", "EURUSD", "--count", "20"])
        assert args._cli_option_symbol == "EURUSD"
        assert args.count == 20

    def test_first_required_param_has_positional_and_flag_actions(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
            ]
        }
        add_dynamic_arguments(parser, func_info)
        symbol_actions = [
            action
            for action in parser._actions
            if action.dest in {"symbol", "_cli_option_symbol"}
        ]
        assert len(symbol_actions) == 2
        assert any(action.option_strings == [] for action in symbol_actions)
        assert any(action.option_strings == ["--symbol"] for action in symbol_actions)

    def test_single_word_flag_is_not_duplicated(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "ticket", "type": int, "required": False, "default": None},
            ]
        }
        add_dynamic_arguments(parser, func_info)
        ticket_action = next(
            action for action in parser._actions if action.dest == "ticket"
        )
        assert ticket_action.option_strings == ["--ticket"]

    def test_limit_exposes_only_canonical_limit_for_bar_command(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "limit", "type": int, "required": False, "default": 100},
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="data_fetch_candles")
        args = parser.parse_args(["--limit", "250"])
        assert args.limit == 250
        limit_action = next(
            action for action in parser._actions if action.dest == "limit"
        )
        assert "--bars" not in limit_action.option_strings

    def test_temporal_and_research_window_aliases(self):
        temporal = argparse.ArgumentParser()
        add_dynamic_arguments(
            temporal,
            {
                "params": [
                    {"name": "group_by", "type": Literal["dow", "hour", "month", "session", "all"], "required": False, "default": "dow"},
                ]
            },
            cmd_name="temporal_analyze",
        )
        assert temporal.parse_args(["--by", "hour"]).group_by == "hour"
        assert temporal.parse_args(["--group-by", "day_of_week"]).group_by == "dow"

        research = argparse.ArgumentParser()
        add_dynamic_arguments(
            research,
            {
                "params": [
                    {"name": "window_bars", "type": int, "required": False, "default": 500},
                ]
            },
            cmd_name="cointegration_test",
        )
        assert research.parse_args(["--lookback", "200"]).window_bars == 200

    def test_trade_place_accepts_side_alias_for_order_type(self):
        parser = argparse.ArgumentParser()

        def tool(request: TradePlaceRequest):
            pass

        add_dynamic_arguments(
            parser,
            get_function_info(tool),
            cmd_name="trade_place",
        )
        args = parser.parse_args(["EURUSD", "--side", "sell", "--volume", "0.02"])
        assert args.order_type == "SELL"
        assert args.volume == 0.02

    def test_news_accepts_optional_positional_symbol(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": False, "default": None},
                {"name": "detail", "type": str, "required": False, "default": "compact"},
                {"name": "limit", "type": int, "required": False, "default": None},
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="news")
        args = parser.parse_args(["AAPL", "--limit", "5"])
        assert args.symbol == "AAPL"
        assert args.limit == 5

    def test_indicators_list_category_accepts_mixed_case(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "category",
                    "type": Literal["momentum", "trend", "volatility"],
                    "required": False,
                    "default": None,
                },
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="indicators_list")
        args = parser.parse_args(["--category", "Trend"])
        assert args.category == "trend"

    def test_trade_history_position_ticket_accepts_ticket_alias(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "position_ticket",
                    "type": int,
                    "required": False,
                    "default": None,
                },
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="trade_history")
        args = parser.parse_args(["--ticket", "123456"])
        assert args.position_ticket == 123456

    @pytest.mark.parametrize(
        "command",
        ["forecast_backtest_run", "forecast_tune_genetic", "forecast_tune_optuna"],
    )
    def test_forecast_multi_methods_accepts_method_alias(self, command):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "methods",
                    "type": Optional[List[str]],
                    "required": False,
                    "default": None,
                },
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name=command)
        args = parser.parse_args(["--method", "theta"])
        assert args.methods == ["theta"]

    def test_market_depth_exposes_require_dom(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "require_dom", "type": bool, "required": False, "default": False},
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="market_depth_fetch")
        args = parser.parse_args(["--require-dom"])
        assert args.require_dom == "true"
        require_dom_action = next(
            action for action in parser._actions if action.dest == "require_dom"
        )
        assert "--require-dom" in require_dom_action.option_strings

    def test_market_depth_spread_help_describes_boolean_output_control(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "spread", "type": bool, "required": False, "default": False},
            ]
        }

        add_dynamic_arguments(parser, func_info, cmd_name="market_depth_fetch")
        args = parser.parse_args(["--spread", "true"])
        spread_action = next(
            action for action in parser._actions if action.dest == "spread"
        )

        assert args.spread == "true"
        assert "Boolean output control" in spread_action.help
        assert "fallback quote" in spread_action.help
        assert "spread value" not in spread_action.help.lower()

    @pytest.mark.parametrize(
        ("command", "parameter", "expected"),
        [
            ("options_barrier_price", "barrier", "knock-in/knock-out"),
            ("strategy_validate", "candidates", "builtin_strategy"),
            ("strategy_validate", "candidates", "forecast_threshold"),
            ("strategy_validate", "candidates", "0.005"),
            ("strategy_validate", "barrier", "next-open execution"),
        ],
    )
    def test_specialized_barrier_help_is_domain_specific(
        self, command, parameter, expected
    ):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": parameter,
                    "type": str,
                    "required": False,
                    "default": None,
                }
            ]
        }

        add_dynamic_arguments(parser, func_info, cmd_name=command)

        action = next(action for action in parser._actions if action.dest == parameter)
        assert expected in action.help

    def test_wait_event_exposes_symbol_without_instrument_alias(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": False, "default": None},
            ]
        }

        add_dynamic_arguments(parser, func_info, cmd_name="wait_event")

        assert parser.parse_args(["EURUSD"]).symbol == "EURUSD"
        assert not any(
            action.dest == "instrument"
            for action in parser._actions
        )
        symbol_action = next(
            action for action in parser._actions if action.dest == "symbol"
        )
        assert "Timer-only duration" in symbol_action.help

    def test_wait_event_help_explains_bounded_boundary_mode(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "timeframe", "type": str, "required": False, "default": None},
                {
                    "name": "max_wait_seconds",
                    "type": float,
                    "required": False,
                    "default": None,
                },
                {
                    "name": "symbols",
                    "type": Optional[List[str]],
                    "required": False,
                    "default": None,
                },
                {
                    "name": "poll_interval_seconds",
                    "type": float,
                    "required": False,
                    "default": None,
                },
            ]
        }

        add_dynamic_arguments(parser, func_info, cmd_name="wait_event")
        help_text = _strip_ansi(parser.format_help())
        compact_help = " ".join(help_text.split())

        assert "optional safety cap" in compact_help
        assert "defaults to the timeframe length plus 60 seconds" in compact_help
        assert "Basket of 1-12 trading symbols" in compact_help
        assert "Cannot be combined with symbol" in compact_help
        assert "must be at least 0.1" in compact_help
        assert "Defaults to M1" not in compact_help
        assert "--timeout" in help_text
        assert parser.parse_args(["--timeout", "1"]).max_wait_seconds == 1.0
        assert parser.parse_args(["--symbols", "EURUSD", "GBPUSD"]).symbols == [
            "EURUSD",
            "GBPUSD",
        ]

    def test_wait_event_watch_for_help_lists_watcher_schemas(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "watch_for",
                    "type": Optional[list],
                    "required": False,
                    "default": None,
                },
            ]
        }

        add_dynamic_arguments(parser, func_info, cmd_name="wait_event")
        help_text = _strip_ansi(parser.format_help())
        compact_help = " ".join(help_text.split())

        assert "price_touch_level" in compact_help
        assert "order_filled" in compact_help
        assert "window.kind=minutes|ticks" in compact_help
        assert "threshold_mode=fixed_pct" in compact_help
        assert '{"type":"order_filled","symbol":"EURUSD"}' in compact_help

    def test_calendar_prefers_start_end_and_hides_legacy_date_flags(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "start", "type": str, "required": False, "default": None},
                {"name": "end", "type": str, "required": False, "default": None},
                {"name": "date_from", "type": str, "required": False, "default": None},
                {"name": "date_to", "type": str, "required": False, "default": None},
            ]
        }

        add_dynamic_arguments(parser, func_info, cmd_name="calendar")

        args = parser.parse_args(["--start", "2026-01-05", "--end", "2026-01-12"])
        assert args.start == "2026-01-05"
        assert args.end == "2026-01-12"
        assert not any(
            action.dest in {"date_from", "date_to"} and action.help != argparse.SUPPRESS
            for action in parser._actions
        )

    def test_labels_triple_barrier_uses_canonical_detail_choices(
        self,
    ):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "detail",
                    "type": Literal["compact", "standard", "summary", "full"],
                    "required": False,
                    "default": "compact",
                },
            ]
        }

        add_dynamic_arguments(parser, func_info, cmd_name="labels_triple_barrier")

        detail_action = next(action for action in parser._actions if action.dest == "detail")
        assert list(detail_action.choices) == ["compact", "standard", "summary", "full"]
        assert not any(action.dest == "summary_only" for action in parser._actions)
        args = parser.parse_args(["--detail", "standard"])
        assert args.detail == "standard"

    def test_patterns_detect_detail_choices_follow_canonical_order(self):
        parser = argparse.ArgumentParser()

        def tool(request):
            pass

        tool.__annotations__ = {"request": PatternsDetectRequest}
        func_info = get_function_info(tool)
        add_dynamic_arguments(parser, func_info, cmd_name="patterns_detect")

        detail_action = next(action for action in parser._actions if action.dest == "detail")
        assert list(detail_action.choices) == ["compact", "standard", "summary", "full"]
        args = parser.parse_args(["EURUSD", "--detail", "summary"])
        assert args.detail == "summary"

    def test_trading_order_commands_expose_canonical_detail(self):
        for cmd_name, model_type, detail_value, argv in (
            (
                "trade_place",
                TradePlaceRequest,
                "standard",
                [
                    "EURUSD",
                    "--volume",
                    "0.01",
                    "--order-type",
                    "BUY",
                    "--detail",
                    "standard",
                ],
            ),
            (
                "trade_modify",
                TradeModifyRequest,
                "summary",
                ["--ticket", "123", "--detail", "summary"],
            ),
            ("trade_close", TradeCloseRequest, "summary", ["--detail", "summary"]),
        ):
            parser = argparse.ArgumentParser()

            def tool(request):
                pass

            tool.__annotations__ = {"request": model_type}
            func_info = get_function_info(tool)
            add_dynamic_arguments(parser, func_info, cmd_name=cmd_name)

            assert any(action.dest == "detail" for action in parser._actions)
            assert not any(action.dest == "preview_detail" for action in parser._actions)
            args = parser.parse_args(argv)
            assert args.detail == detail_value

    def test_news_detail_choices_are_compact_and_full(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "detail",
                    "type": Literal["compact", "full"],
                    "required": False,
                    "default": "compact",
                },
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="news")
        detail_action = next(action for action in parser._actions if action.dest == "detail")
        assert list(detail_action.choices) == ["compact", "full"]

    def test_trade_modify_requires_named_ticket(self, capsys):
        parser = argparse.ArgumentParser(allow_abbrev=False)

        def tool(request):
            pass

        tool.__annotations__ = {"request": TradeModifyRequest}
        func_info = get_function_info(tool)
        add_dynamic_arguments(parser, func_info, cmd_name="trade_modify")

        assert parser.parse_args(["--ticket", "123"]).ticket == 123
        with pytest.raises(SystemExit):
            parser.parse_args(["123"])
        assert "--ticket" in capsys.readouterr().err

    def test_partial_flag_prefix_is_rejected_when_abbrev_disabled(self, capsys):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        func_info = {
            "params": [
                {
                    "name": "search_term",
                    "type": str,
                    "required": False,
                    "default": None,
                },
            ]
        }

        add_dynamic_arguments(parser, func_info)

        with pytest.raises(SystemExit):
            parser.parse_args(["--search", "BTC"])

        assert "unrecognized arguments: --search BTC" in capsys.readouterr().err


# ========================================================================
# _parse_kv_string
# ========================================================================


class TestParseKvString:
    def test_kv_pairs(self):
        result = _parse_kv_string("a=1,b=2")
        assert result is not None
        assert "a" in result

    def test_json_string(self):
        result = _parse_kv_string('{"a": 1}')
        assert result is not None
        assert result["a"] == 1

    @patch("mtdata.utils.utils.parse_kv_or_json", side_effect=Exception("fail"))
    def test_exception_returns_none(self, mock_parse):
        result = _parse_kv_string("bad")
        assert result is None


# ========================================================================
# _resolve_param_kwargs
# ========================================================================


class TestResolveParamKwargs:
    def test_forecast_method_profile_help_matches_public_choices(self):
        kwargs, _ = _resolve_param_kwargs(
            {
                "name": "profile",
                "type": Literal["quickstart", "core", "all"],
                "required": False,
                "default": "all",
            },
            None,
            cmd_name="forecast_list_methods",
        )

        assert all(value in kwargs["help"] for value in ("quickstart", "core", "all"))
        assert all(
            value not in kwargs["help"]
            for value in ("fast", "statistical", "machine_learning", "pretrained")
        )

    @pytest.mark.parametrize(
        ("command", "parameter", "expected"),
        [
            ("forecast_list_library_models", "limit", "compact output uses 20"),
            (
                "forecast_models_delete",
                "confirm_model_id",
                "required with --dry-run false",
            ),
            ("volatility_term_structure", "horizons", "horizons in bars"),
            ("market_relative_strength", "weights", "matching --horizons"),
            ("market_relative_strength", "limit", "ranked symbols"),
            (
                "patterns_detect",
                "include_completed",
                "Candlestick mode always scans closed-bar detections",
            ),
            ("patterns_detect", "top_k", "Full detail returns every surviving row"),
            ("options_chain", "limit", "option contracts"),
            ("volume_profile_levels", "lookback", "Historical bar count"),
            ("patterns_detect", "lookback", "Historical bars to scan for patterns"),
            ("forecast_generate", "lookback", "native theta/fourier_ols"),
            ("cointegration_test", "limit", "compact/summary output uses 10"),
            ("outliers_detect", "limit", "anomalous bars"),
            ("temporal_analyze", "limit", "time buckets"),
            ("temporal_analyze", "session_calendar", "auto, fx, equity, or continuous_24_7"),
            ("temporal_analyze", "time_range", "in --timezone"),
            ("temporal_analyze", "return_basis", "overnight/session gaps"),
            ("options_heston_calibrate", "valuation_date", "chain snapshot date"),
            (
                "options_heston_calibrate",
                "calendar",
                "valuation timezone",
            ),
            ("seasonality_detect", "max_period", "period in bars"),
        ],
    )
    def test_command_help_explains_reported_units_and_objects(
        self, command, parameter, expected
    ):
        kwargs, _ = _resolve_param_kwargs(
            {"name": parameter, "type": str, "required": False, "default": None},
            None,
            cmd_name=command,
        )

        assert expected in kwargs["help"]

    def test_strategy_barrier_help_uses_documented_percent_term(self):
        kwargs, _ = _resolve_param_kwargs(
            {"name": "barrier", "type": dict, "required": False, "default": None},
            None,
            cmd_name="strategy_validate",
        )

        assert "percent values" in kwargs["help"]
        assert "0.5 means 0.5" in kwargs["help"]
        assert "percentage points" not in kwargs["help"]

    def test_basic_str_param(self):
        param = {"name": "symbol", "type": str, "required": True, "default": None}
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert kwargs["type"] is str
        assert is_mapping is False

    def test_int_param(self):
        param = {"name": "count", "type": int, "required": False, "default": 10}
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert kwargs["type"] is int
        assert kwargs["default"] == 10

    def test_float_param(self):
        param = {"name": "alpha", "type": float, "required": False, "default": 0.05}
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert kwargs["type"] is float

    def test_bool_param(self):
        param = {"name": "verbose", "type": bool, "required": False, "default": None}
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert kwargs["choices"] == ["true", "false"]
        assert "metavar" not in kwargs
        assert kwargs["type"]("True") == "true"
        assert kwargs["type"]("FALSE") == "false"

    def test_research_bools_still_accept_aliases(self):
        param = {"name": "include_spread", "type": bool, "required": False, "default": False}
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="data_fetch_candles")
        assert kwargs["type"]("yes") == "true"
        assert kwargs["type"]("0") == "false"

    @pytest.mark.parametrize("alias", ["no", "off", "0", "yes", "on", "1", "y", "n"])
    def test_trading_mutation_bools_reject_undocumented_aliases(self, alias):
        kwargs, _ = _resolve_param_kwargs(
            {"name": "dry_run", "type": bool, "required": False, "default": True},
            None,
            cmd_name="trade_place",
        )
        with pytest.raises(argparse.ArgumentTypeError, match="true or false"):
            kwargs["type"](alias)

    def test_trading_mutation_bools_accept_canonical_true_false(self):
        kwargs, _ = _resolve_param_kwargs(
            {"name": "dry_run", "type": bool, "required": False, "default": True},
            None,
            cmd_name="trade_place",
        )
        assert kwargs["type"]("true") == "true"
        assert kwargs["type"]("FALSE") == "false"

    @pytest.mark.parametrize("alias", ["no", "off", "0", "yes", "on", "1"])
    def test_trade_place_request_rejects_live_aliases_for_dry_run(self, alias):
        with pytest.raises(ValidationError, match="true or false"):
            TradePlaceRequest(
                symbol="EURUSD",
                volume=0.01,
                order_type="BUY",
                stop_loss=1.10,
                take_profit=1.12,
                dry_run=alias,
            )

    def test_trade_history_limit_has_safety_cap(self):
        with pytest.raises(ValidationError):
            TradeHistoryRequest(limit=1_000_000)
        request = TradeHistoryRequest(limit=500)
        assert request.limit == 500

    def test_optional_int_accepts_integer_and_null_tokens(self):
        param = {
            "name": "count",
            "type": Optional[int],
            "required": False,
            "default": None,
        }
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert kwargs["type"]("42") == 42
        assert kwargs["type"]("none") is None
        assert kwargs["type"]("NULL") is None

    def test_dict_param_is_mapping(self):
        param = {
            "name": "params",
            "type": Dict[str, Any],
            "required": False,
            "default": None,
        }
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert is_mapping is True

    def test_literal_type(self):
        param = {
            "name": "mode",
            "type": Literal["a", "b", "c"],
            "required": False,
            "default": "a",
        }
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert kwargs["choices"] == ["a", "b", "c"]
        assert kwargs["type"]("B") == "b"

    def test_literal_type_preserves_canonical_choice_case(self):
        param = {
            "name": "timeframe",
            "type": Literal["M1", "H1", "D1"],
            "required": False,
            "default": "H1",
        }

        kwargs, _ = _resolve_param_kwargs(param, None)

        assert kwargs["type"]("h1") == "H1"
        assert kwargs["type"]("D1") == "D1"
        assert kwargs["type"]("bad") == "bad"

    def test_union_of_literals_exposes_all_choices(self):
        param = {
            "name": "timeframe",
            "type": AutoTimeframeLiteral,
            "required": False,
            "default": "H1",
        }

        kwargs, _ = _resolve_param_kwargs(
            param,
            None,
            cmd_name="support_resistance_levels",
        )

        assert "H1" in kwargs["choices"]
        assert "auto" in kwargs["choices"]
        assert kwargs["type"]("AUTO") == "auto"

    def test_patterns_mode_choices_are_explicit(self):
        param = {
            "name": "mode",
            "type": PatternsDetectRequest.model_fields["mode"].annotation,
            "required": False,
            "default": "candlestick",
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="patterns_detect")
        assert kwargs["choices"] == [
            "candlestick",
            "classic",
            "harmonic",
            "fractal",
            "elliott",
            "all",
        ]
        assert "fractals" not in kwargs["choices"]

    def test_static_method_and_transform_choices_are_exposed_per_command(self):
        method_param = {
            "name": "method",
            "type": Literal["pearson", "spearman"],
            "required": False,
            "default": "pearson",
        }
        correlation, _ = _resolve_param_kwargs(
            method_param,
            None,
            cmd_name="correlation_matrix",
        )
        assert correlation["choices"] == ["pearson", "spearman"]
        assert correlation["type"]("SPEARMAN") == "spearman"

        var_method, _ = _resolve_param_kwargs(
            {**method_param, "type": Literal["historical", "parametric"]},
            None,
            cmd_name="trade_var_cvar_calculate",
        )
        assert var_method["choices"] == [
            "historical",
            "parametric",
        ]

        transform_param = {
            "name": "transform",
            "type": Literal["log_return", "pct", "diff", "level", "log_level"],
            "required": False,
            "default": "log_return",
        }
        transform, _ = _resolve_param_kwargs(
            transform_param,
            None,
            cmd_name="cross_correlation",
        )
        assert transform["choices"] == [
            "log_return",
            "pct",
            "diff",
            "level",
            "log_level",
        ]

    def test_dynamic_and_comma_composed_choices_explain_discovery_in_help(self):
        tests_param = {
            "name": "tests",
            "type": str,
            "required": False,
            "default": "adf,kpss,pp",
        }
        stationarity, _ = _resolve_param_kwargs(
            tests_param,
            None,
            cmd_name="stationarity_test",
        )
        assert "adf, kpss, pp" in stationarity["help"]
        assert "choices" not in stationarity

        method_param = {
            "name": "method",
            "type": str,
            "required": True,
            "default": None,
        }
        denoise, _ = _resolve_param_kwargs(
            method_param,
            None,
            cmd_name="denoise_describe",
        )
        assert "denoise_list_methods" in denoise["help"]

    def test_var_cvar_symbol_help_explains_it_requires_open_exposure(self):
        symbol_param = {
            "name": "symbol",
            "type": Optional[str],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(
            symbol_param,
            None,
            cmd_name="trade_var_cvar_calculate",
        )

        assert "currently open positions" in kwargs["help"]
        assert "full open portfolio" in kwargs["help"]

    def test_report_template_choices_are_explicit(self):
        from mtdata.core.report.requests import ReportTemplateLiteral

        param = {
            "name": "template",
            "type": ReportTemplateLiteral,
            "required": False,
            "default": "basic",
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="report_generate")
        assert kwargs["choices"] == [
            "minimal",
            "basic",
            "advanced",
            "scalping",
            "intraday",
            "swing",
            "position",
        ]
        assert "Report template" in kwargs["help"]
        assert "scalping" in kwargs["help"]
        assert "Typical warm runtimes" in kwargs["help"]

    def test_forecast_train_quantity_help_does_not_advertise_volatility(self):
        param = {
            "name": "quantity",
            "type": Literal["price", "return"],
            "required": False,
            "default": "price",
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="forecast_train")
        assert kwargs["choices"] == ["price", "return"]
        assert "forecast_volatility_estimate" in kwargs["help"]
        assert "price/return/volatility" not in kwargs["help"]

    def test_trade_idea_compose_template_help_describes_quick_and_standard(self):
        param = {
            "name": "template",
            "type": Literal["quick", "standard"],
            "required": False,
            "default": "quick",
        }
        kwargs, _ = _resolve_param_kwargs(
            param, None, cmd_name="trade_idea_compose"
        )
        assert kwargs["choices"] == ["quick", "standard"]
        assert "quick" in kwargs["help"]
        assert "standard" in kwargs["help"]
        assert "minimal" not in kwargs["help"]
        assert "scalping" not in kwargs["help"]

    def test_trade_modify_stop_help_does_not_mention_trade_place(self):
        param = {
            "name": "stop_loss",
            "type": Optional[float],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="trade_modify")
        assert "trade_place" not in kwargs["help"]
        assert "require-sl-tp" not in kwargs["help"]
        assert "clear-stop-loss" in kwargs["help"]

    def test_asset_performance_universe_help_is_not_symbol_scan(self):
        param = {
            "name": "universe",
            "type": Literal["forex", "crypto", "futures", "insider"],
            "required": False,
            "default": "forex",
        }
        kwargs, _ = _resolve_param_kwargs(
            param, None, cmd_name="asset_performance"
        )
        assert "forex" in kwargs["help"]
        assert "insider" in kwargs["help"]
        assert "Symbol scan universe" not in kwargs["help"]

    def test_list_type(self):
        param = {"name": "items", "type": List[str], "required": False, "default": None}
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert kwargs["nargs"] == "+"

    def test_param_docs_used(self):
        param = {"name": "symbol", "type": str, "required": True, "default": None}
        docs = {"symbol": "The trading symbol"}
        kwargs, _ = _resolve_param_kwargs(param, docs)
        assert kwargs["help"] == "The trading symbol"

    def test_no_default_for_required(self):
        param = {"name": "sym", "type": str, "required": True, "default": None}
        kwargs, _ = _resolve_param_kwargs(param, None)
        assert "default" not in kwargs

    def test_list_of_literals(self):
        param = {
            "name": "methods",
            "type": List[Literal["a", "b"]],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None)
        assert "choices" not in kwargs
        assert kwargs["metavar"] == "{a,b}"
        assert kwargs["nargs"] == "+"
        assert kwargs["type"]("A") == "a"
        assert kwargs["type"]("A,b") == "a,b"
        with pytest.raises(argparse.ArgumentTypeError, match="invalid choice"):
            kwargs["type"]("a,c")

    def test_forecast_method_help_avoids_massive_choices(self):
        param = {"name": "method", "type": str, "required": False, "default": None}
        kwargs, _ = _resolve_param_kwargs(
            param, None, cmd_name="forecast_conformal_intervals"
        )
        assert "choices" not in kwargs
        assert kwargs["metavar"] == "METHOD"
        assert "forecast_list_methods" in kwargs["help"]
        assert kwargs["help"].count("forecast_list_methods") == 1

    def test_forecast_method_literal_help_uses_method_browser_hint(self):
        param = {
            "name": "method",
            "type": Literal["theta", "arima"],
            "required": False,
            "default": "theta",
        }
        kwargs, _ = _resolve_param_kwargs(param, None)
        assert "choices" not in kwargs
        assert kwargs["metavar"] == "METHOD"
        assert "forecast_list_methods" in kwargs["help"]

    def test_non_forecast_method_help_does_not_mention_forecast_browser(self):
        param = {"name": "method", "type": str, "required": False, "default": None}
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="correlation_matrix")

        assert kwargs["help"] == "Correlation coefficient: pearson or spearman."
        assert "forecast_list_methods" not in kwargs["help"]

    def test_patterns_detect_lookback_help_omits_forecast_defaults(self):
        kwargs, _ = _resolve_param_kwargs(
            {"name": "lookback", "type": int, "required": False, "default": None},
            None,
            cmd_name="patterns_detect",
        )

        assert "Historical bars to scan for patterns" in kwargs["help"]
        assert "theta" not in kwargs["help"].lower()
        assert "fourier" not in kwargs["help"].lower()

    @pytest.mark.parametrize(
        ("param_name", "expected"),
        [
            ("lookback", "after applying the requested time window"),
            ("as_of", "Cannot be combined with start/end"),
            ("start", "Cannot be combined with as_of"),
            ("end", "Cannot be combined with as_of"),
            ("quantity", "forecast_volatility_estimate"),
        ],
    )
    def test_forecast_train_help_explains_training_window_and_targets(
        self,
        param_name,
        expected,
    ):
        from mtdata.core.forecast_tasks import ForecastTrainRequest

        description = ForecastTrainRequest.model_fields[param_name].description
        assert description is not None
        assert expected in description

    def test_forecast_train_wait_help_explains_context_defaults(self):
        kwargs, _ = _resolve_param_kwargs(
            {
                "name": "wait",
                "type": bool,
                "required": False,
                "default": False,
            },
            None,
            cmd_name="forecast_train",
        )

        assert "One-shot CLI and stdin shell batches always wait" in kwargs["help"]
        assert "--wait false is rejected there" in kwargs["help"]
        assert "CLI default: true" in kwargs["help"]
        assert "interactive shell, MCP, and Web API" in kwargs["help"]

    @pytest.mark.parametrize(
        ("cmd_name", "param_name", "expected"),
        [
            ("forecast_tune_optuna", "n_trials", "200 rolling backtests"),
            ("forecast_tune_optuna", "timeout", "wall-clock search limit"),
            ("forecast_optimize_hints", "timeframes", "cheaper exploratory run"),
            ("forecast_optimize_hints", "population", "190 rolling backtests"),
            ("forecast_optimize_hints", "generations", "population*generations*steps"),
            (
                "forecast_optimize_hints",
                "max_search_time_seconds",
                "wall-clock search limit",
            ),
            ("forecast_tune_genetic", "population", "600 rolling backtests"),
            ("forecast_tune_genetic", "generations", "600 at the defaults"),
            (
                "forecast_tune_genetic",
                "max_search_time_seconds",
                "wall-clock search limit",
            ),
            ("market_status", "allow_partial", "strict completion"),
            ("patterns_detect", "allow_partial", "strict completion"),
            ("forecast_task_wait", "timeout_seconds", "Maximum 86400"),
        ],
    )
    def test_forecast_search_help_quantifies_work(
        self,
        cmd_name,
        param_name,
        expected,
    ):
        kwargs, _ = _resolve_param_kwargs(
            {
                "name": param_name,
                "type": str,
                "required": False,
                "default": None,
            },
            None,
            cmd_name=cmd_name,
        )

        assert expected in kwargs["help"]

    @pytest.mark.parametrize(
        ("cmd_name", "param_name", "expected"),
        [
            ("data_fetch_candles", "timestamp_format", "ISO in CLIENT_TZ"),
            ("data_fetch_ticks", "timestamp_format", "ISO in CLIENT_TZ"),
            ("market_ticker", "price_field", "default bid/ask/spread quote snapshot"),
            ("patterns_detect", "timeframe", "elliott scans H1/H4/D1"),
            ("symbols_list", "universe", "searches use the full broker catalog"),
            ("volume_profile_levels", "source", "labeled M1-bar approximation"),
        ],
    )
    def test_command_specific_help_explains_conditional_semantics(
        self,
        cmd_name,
        param_name,
        expected,
    ):
        kwargs, _ = _resolve_param_kwargs(
            {
                "name": param_name,
                "type": str,
                "required": False,
                "default": None,
            },
            None,
            cmd_name=cmd_name,
        )

        assert expected in kwargs["help"]

    def test_common_analysis_params_have_specific_help(self):
        transform_kwargs, _ = _resolve_param_kwargs(
            {"name": "transform", "type": str, "required": False, "default": None},
            None,
            cmd_name="correlation_matrix",
        )
        min_regime_kwargs, _ = _resolve_param_kwargs(
            {"name": "min_regime_bars", "type": int, "required": False, "default": None},
            None,
            cmd_name="regime_detect",
        )
        correlation_limit_kwargs, _ = _resolve_param_kwargs(
            {"name": "limit", "type": int, "required": False, "default": None},
            None,
            cmd_name="correlation_matrix",
        )
        correlation_window_kwargs, _ = _resolve_param_kwargs(
            {"name": "window_bars", "type": int, "required": False, "default": 500},
            None,
            cmd_name="correlation_matrix",
        )
        causal_limit_kwargs, _ = _resolve_param_kwargs(
            {"name": "limit", "type": int, "required": False, "default": None},
            None,
            cmd_name="causal_discover_signals",
        )
        causal_window_kwargs, _ = _resolve_param_kwargs(
            {"name": "window_bars", "type": int, "required": False, "default": 500},
            None,
            cmd_name="causal_discover_signals",
        )
        regime_fetch_limit_kwargs, _ = _resolve_param_kwargs(
            {"name": "fetch_limit", "type": int, "required": False, "default": 100},
            None,
            cmd_name="regime_detect",
        )

        assert "Price transform" in transform_kwargs["help"]
        assert transform_kwargs["help"] != "transform parameter"
        assert "Minimum bars a detected regime must span" in min_regime_kwargs["help"]
        assert min_regime_kwargs["help"] != "min_regime_bars parameter"
        assert "Max correlation pair rows" in correlation_limit_kwargs["help"]
        assert "Historical bars per symbol" in correlation_window_kwargs["help"]
        assert causal_limit_kwargs["help"] == "Max causal link rows to return."
        assert "Historical bars per symbol" in causal_window_kwargs["help"]
        assert "max_regimes" in regime_fetch_limit_kwargs["help"]

    def test_stationarity_trend_help_describes_adf_deterministic_term(self):
        kwargs, _ = _resolve_param_kwargs(
            {"name": "trend", "type": str, "required": False, "default": "c"},
            None,
            cmd_name="stationarity_test",
        )

        assert "ADF" in kwargs["help"]
        assert "constant+trend" in kwargs["help"]
        assert "Cointegration" not in kwargs["help"]

    @pytest.mark.parametrize(
        "cmd_name",
        [
            "causal_discover_signals",
            "cointegration_test",
            "correlation_matrix",
        ],
    )
    def test_pairwise_symbol_help_describes_comma_separated_symbols(self, cmd_name):
        kwargs, _ = _resolve_param_kwargs(
            {"name": "symbols", "type": str, "required": False, "default": None},
            None,
            cmd_name=cmd_name,
        )

        assert "Comma- or space-separated MT5 symbols" in kwargs["help"]
        assert "Optional with --group" in kwargs["help"]

    @pytest.mark.parametrize(
        ("cmd_name", "required"),
        [
            ("causal_discover_signals", False),
            ("correlation_matrix", False),
            ("cointegration_test", False),
            ("cross_correlation", True),
        ],
    )
    def test_pairwise_symbol_positionals_accept_space_separated_values(
        self,
        cmd_name,
        required,
    ):
        parser = argparse.ArgumentParser()
        add_dynamic_arguments(
            parser,
            {
                "params": [
                    {
                        "name": "symbols",
                        "type": str,
                        "required": required,
                        "default": None,
                    }
                ]
            },
            cmd_name=cmd_name,
        )

        assert parser.parse_args(["EURUSD", "GBPUSD"]).symbols == ["EURUSD", "GBPUSD"]

    @pytest.mark.parametrize(
        ("cmd_name", "param_name", "alias"),
        [
            ("screener", "search", "--search-term"),
            ("forecast_list_methods", "search_term", "--search"),
        ],
    )
    def test_catalog_search_flags_accept_both_spellings(
        self,
        cmd_name,
        param_name,
        alias,
    ):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        add_dynamic_arguments(
            parser,
            {
                "params": [
                    {
                        "name": param_name,
                        "type": str,
                        "required": False,
                        "default": None,
                    }
                ]
            },
            cmd_name=cmd_name,
        )

        assert getattr(parser.parse_args([alias, "cap"]), param_name) == "cap"

    def test_labels_triple_barrier_limit_and_lookback_help_distinguish_roles(self):
        limit_kwargs, _ = _resolve_param_kwargs(
            {"name": "limit", "type": int, "required": False, "default": 1200},
            None,
            cmd_name="labels_triple_barrier",
        )
        lookback_kwargs, _ = _resolve_param_kwargs(
            {"name": "lookback", "type": int, "required": False, "default": 300},
            None,
            cmd_name="labels_triple_barrier",
        )

        assert "recent resolved TP/SL examples" in limit_kwargs["help"]
        assert "tail is entirely neutral" in limit_kwargs["help"]
        assert "full returns the complete labeled series" in limit_kwargs["help"]
        assert "Optional tail cap of labeled entries" in lookback_kwargs["help"]
        assert "lookback plus horizon" in lookback_kwargs["help"]
        assert "explicit date range is analyzed in full" in lookback_kwargs["help"]

    def test_labels_triple_barrier_barriers_help_documents_json_and_units(self):
        barriers_kwargs, _ = _resolve_param_kwargs(
            {"name": "barriers", "type": dict, "required": True, "default": None},
            None,
            cmd_name="labels_triple_barrier",
        )

        assert '"unit":"pct"' in barriers_kwargs["help"]
        assert "price values are absolute levels" in barriers_kwargs["help"]

    def test_patterns_engine_help_names_mode_scope(self):
        kwargs, _ = _resolve_param_kwargs(
            {"name": "engine", "type": str, "required": False, "default": None},
            None,
            cmd_name="patterns_detect",
        )

        assert "Classic-mode engine" in kwargs["help"]
        assert "invalid for other modes" in kwargs["help"]

    def test_temporal_lookback_help_discloses_auto_window(self):
        kwargs, _ = _resolve_param_kwargs(
            {"name": "lookback", "type": int, "required": False, "default": None},
            None,
            cmd_name="temporal_analyze",
        )

        assert "timeframe-aware seasonal window" in kwargs["help"]
        assert "H1 session: 1440 bars" in kwargs["help"]

    def test_temporal_limit_help_discloses_per_dimension_paging(self):
        kwargs, _ = _resolve_param_kwargs(
            {"name": "limit", "type": int, "required": False, "default": None},
            None,
            cmd_name="temporal_analyze",
        )

        assert "single group_by" in kwargs["help"]
        assert "each of the four breakdowns" in kwargs["help"]
        assert "dimension_pagination" in kwargs["help"]
        assert "groups_analyzed" in kwargs["help"]
        assert "unpaged total" in kwargs["help"]

    def test_trade_history_minutes_back_help_mentions_default_lookback(self):
        kwargs, _ = _resolve_param_kwargs(
            {"name": "minutes_back", "type": int, "required": False, "default": None},
            None,
            cmd_name="trade_history",
        )

        assert "Defaults to 10080 minutes" in kwargs["help"]
        assert "7 days" in kwargs["help"]

    def test_trade_journal_minutes_back_help_mentions_default_lookback(self):
        kwargs, _ = _resolve_param_kwargs(
            {"name": "minutes_back", "type": int, "required": False, "default": None},
            None,
            cmd_name="trade_journal_analyze",
        )

        assert "Defaults to 10080 minutes" in kwargs["help"]
        assert "7 days" in kwargs["help"]

    @pytest.mark.parametrize(
        ("cmd_name", "param_name", "expected"),
        [
            ("data_fetch_candles", "limit", "default to a 20-bar page"),
            ("data_fetch_ticks", "limit", "maximum 50000"),
            ("market_status", "symbol", "static major-equity-exchange calendar"),
            ("forecast_task_cancel_all", "status_filter", "all, pending, or running"),
            ("trade_journal_analyze", "limit", "realized exit deals"),
            ("trade_execution_quality", "minutes_back", "7 days"),
            ("trade_execution_quality", "limit", "eligible fills"),
            ("trade_history", "side", "position_side"),
            ("trade_journal_analyze", "side", "realized position"),
            ("asset_performance", "option", "latest buys/sales"),
        ],
    )
    def test_command_specific_help_describes_effective_contract(
        self,
        cmd_name,
        param_name,
        expected,
    ):
        kwargs, _ = _resolve_param_kwargs(
            {
                "name": param_name,
                "type": str,
                "required": False,
                "default": None,
            },
            None,
            cmd_name=cmd_name,
        )

        assert expected in kwargs["help"]

    def test_trading_execution_flags_have_actionable_help(self):
        place_dry_run_kwargs, _ = _resolve_param_kwargs(
            {"name": "dry_run", "type": bool, "required": False, "default": True},
            None,
            cmd_name="trade_place",
        )
        require_sl_tp_kwargs, _ = _resolve_param_kwargs(
            {"name": "require_sl_tp", "type": bool, "required": False, "default": True},
            None,
            cmd_name="trade_place",
        )
        close_all_kwargs, _ = _resolve_param_kwargs(
            {"name": "close_all", "type": bool, "required": False, "default": False},
            None,
            cmd_name="trade_close",
        )
        close_target_kwargs, _ = _resolve_param_kwargs(
            {"name": "target", "type": str, "required": False, "default": "positions"},
            None,
            cmd_name="trade_close",
        )
        close_dry_run_kwargs, _ = _resolve_param_kwargs(
            {"name": "dry_run", "type": bool, "required": False, "default": True},
            None,
            cmd_name="trade_close",
        )
        modify_key_kwargs, _ = _resolve_param_kwargs(
            {
                "name": "idempotency_key",
                "type": str,
                "required": False,
                "default": None,
            },
            None,
            cmd_name="trade_modify",
        )

        assert "without sending it to the broker" in place_dry_run_kwargs["help"]
        assert "require_sl_tp parameter" != require_sl_tp_kwargs["help"]
        assert "stop_loss and take_profit" in require_sl_tp_kwargs["help"]
        assert "whole account" in close_all_kwargs["help"]
        assert "never cancels pending orders" in close_target_kwargs["help"]
        assert close_dry_run_kwargs["default"] is True
        assert "dedupe key" in modify_key_kwargs["help"]

    def test_report_generate_format_help_is_removed_output_help(self):
        param = {
            "name": "format",
            "type": str,
            "required": False,
            "default": "legacy",
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="report_generate")
        assert kwargs["help"] == "Domain-specific shape selector when supported; TOON/JSON selection uses json."

    def test_screener_filters_help_is_command_specific(self):
        param = {
            "name": "filters",
            "type": Optional[str],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="screener")
        assert "NASDAQ" in kwargs["help"]
        assert "Sector" in kwargs["help"]

    def test_screener_order_help_is_command_specific(self):
        param = {
            "name": "order",
            "type": Optional[str],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="screener")
        assert (
            kwargs["help"]
            == (
                "Sort key. Default -marketcap (largest first). Use --order=price for "
                "ascending price. Pagination follows this provider order."
            )
        )

    def test_screener_descending_order_uses_parser_safe_equals_form(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)

        def tool(order: Optional[str] = None) -> None:
            pass

        add_dynamic_arguments(
            parser,
            get_function_info(tool),
            cmd_name="screener",
        )

        assert parser.parse_args(["--order=-marketcap"]).order == "-marketcap"

    @pytest.mark.parametrize("cmd_name", ["trade_modify", "trade_get_pending"])
    def test_trading_parameter_help_matches_command_contract(self, cmd_name):
        name = "price" if cmd_name == "trade_modify" else "order_type"
        kwargs, _ = _resolve_param_kwargs(
            {"name": name, "type": Optional[str], "required": False, "default": None},
            None,
            cmd_name=cmd_name,
        )

        if cmd_name == "trade_modify":
            assert "Omit when only stop_loss/take_profit change" in kwargs["help"]
            assert "required" not in kwargs["help"]
        else:
            assert "BUY_STOP_LIMIT" in kwargs["help"]
            assert "SELL_STOP_LIMIT" in kwargs["help"]
            assert "market" not in kwargs["help"].lower()

    @pytest.mark.parametrize(
        "cmd_name",
        [
            "equity_profile",
        ],
    )
    def test_equity_profile_symbol_help_uses_an_equity_ticker(self, cmd_name):
        kwargs, _ = _resolve_param_kwargs(
            {"name": "symbol", "type": str, "required": True, "default": None},
            None,
            cmd_name=cmd_name,
        )

        assert "AAPL" in kwargs["help"]
        assert "EURUSD" not in kwargs["help"]

    def test_market_scan_limit_help_is_command_specific(self):
        param = {
            "name": "limit",
            "type": Optional[int],
            "required": False,
            "default": 20,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="market_scan")
        assert kwargs["help"] == "Max matching symbols to return."

    def test_market_scan_rank_by_help_lists_actual_options(self):
        param = {
            "name": "rank_by",
            "type": Optional[str],
            "required": False,
            "default": "abs_price_change_pct",
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="market_scan")
        assert "completed-bar" in kwargs["help"]
        assert "abs_live_price_change_pct" in kwargs["help"]

    def test_symbols_top_markets_limit_help_is_command_specific(self):
        param = {
            "name": "limit",
            "type": Optional[int],
            "required": False,
            "default": 10,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="symbols_top_markets")
        assert kwargs["help"] == (
            "Max symbols for the selected ranking; per leaderboard when rank_by=all."
        )

    def test_symbols_top_markets_rank_by_help_lists_actual_options(self):
        param = {
            "name": "rank_by",
            "type": Optional[str],
            "required": False,
            "default": "all",
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="symbols_top_markets")
        assert "abs_price_change_pct (default)" in kwargs["help"]
        assert "all, spread/spread_pct" in kwargs["help"]
        assert "tick_volume" in kwargs["help"]
        assert "volume/tick_volume" not in kwargs["help"]
        assert "price_change/price_change_pct" in kwargs["help"]
        assert "abs_price_change/abs_price_change_pct" in kwargs["help"]
        assert "rsi" not in kwargs["help"]

    def test_news_limit_help_is_command_specific(self):
        param = {"name": "limit", "type": int, "required": False, "default": 20}
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="news")
        assert "Global maximum across all news/event buckets" in kwargs["help"]
        assert "defaults to 10" in kwargs["help"]
        assert "compact" in kwargs["help"].lower()

    def test_market_microstructure_minutes_back_help_discloses_default(self):
        kwargs, _ = _resolve_param_kwargs(
            {"name": "minutes_back", "type": int, "required": False, "default": None},
            None,
            cmd_name="market_microstructure_analyze",
        )
        assert "Defaults to 60" in kwargs["help"]
        assert "start/end" in kwargs["help"]

    def test_trade_stress_test_shocks_help_has_json_examples(self):
        param = {
            "name": "shocks",
            "type": Dict[str, float],
            "required": True,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="trade_stress_test")
        assert kwargs["help"] == (
            "JSON object mapping symbols to percentage shocks greater than -100. "
            "Examples: '{\"*\":-2}' or '{\"EURUSD\":-1,\"XAUUSD\":-3}'. "
            "-100 is rejected because it would imply a zero or negative price; "
            "use a near-total shock such as -99.99 to model almost complete loss."
        )

    def test_trade_stress_test_requires_named_shocks_option(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "shocks",
                    "type": Dict[str, float],
                    "required": True,
                    "default": None,
                },
            ]
        }

        add_dynamic_arguments(parser, func_info, cmd_name="trade_stress_test")

        parsed = parser.parse_args(["--shocks", '{"*":-2}'])
        assert parsed.shocks == '{"*":-2}'
        with pytest.raises(SystemExit):
            parser.parse_args(['{"*":-2}'])

    def test_calendar_start_help_is_command_specific(self):
        param = {
            "name": "start",
            "type": Optional[str],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="calendar")
        assert "YYYY-MM-DD" in kwargs["help"]
        assert "today" in kwargs["help"]

    def test_calendar_end_help_is_command_specific(self):
        param = {
            "name": "end",
            "type": Optional[str],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="calendar")
        assert "YYYY-MM-DD" in kwargs["help"]
        assert "today" in kwargs["help"]

    def test_temporal_min_bars_help_describes_group_filter(self):
        param = {
            "name": "min_bars",
            "type": Optional[int],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="temporal_analyze")
        assert kwargs["help"] == "Exclude grouped rows with fewer than this many bars."

    def test_report_include_sections_help_lists_template_section_names(self):
        param = {
            "name": "include_sections",
            "type": Optional[List[str]],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="report_generate")
        assert "context" in kwargs["help"]
        assert "confluence" in kwargs["help"]
        assert "volume_profile" in kwargs["help"]
        assert "availability varies by template" in kwargs["help"]

    def test_options_symbol_help_is_underlying_specific(self):
        param = {"name": "symbol", "type": str, "required": True, "default": None}
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="options_chain")
        assert "Underlying symbol" in kwargs["help"]
        assert "EURUSD" not in kwargs["help"]

    def test_forecast_tune_optuna_search_space_help_is_command_specific(self):
        param = {
            "name": "search_space",
            "type": Dict[str, Any],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="forecast_tune_optuna")
        assert kwargs["help"] == "Optuna search space (JSON or k=v)."

    def test_data_fetch_candles_indicators_help_mentions_named_and_underscore_syntax(
        self,
    ):
        param = {
            "name": "indicators",
            "type": Optional[List[Dict[str, Any]]],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="data_fetch_candles")
        assert "rsi_14" in kwargs["help"]
        assert "sma=20" in kwargs["help"]
        assert "rsi(length=14)" in kwargs["help"]
        assert "On PowerShell" in kwargs["help"]
        assert '--indicators "rsi(14)"' in kwargs["help"]
        assert "Catalog names" in kwargs["help"]
        assert "indicators_describe" in kwargs["help"]

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("start", "broker-session calendar periods"),
            ("end", "broker-local calendar-period boundary"),
            ("limit", "first-N"),
        ],
    )
    def test_data_fetch_candles_range_help_states_bound_contract(
        self,
        name: str,
        expected: str,
    ):
        param = {
            "name": name,
            "type": Optional[str] if name != "limit" else int,
            "required": False,
            "default": None if name != "limit" else 20,
        }

        kwargs, _ = _resolve_param_kwargs(
            param,
            None,
            cmd_name="data_fetch_candles",
        )

        assert expected in kwargs["help"]

    def test_include_incomplete_help_names_compact_forming_status(self):
        param = {
            "name": "include_incomplete",
            "type": bool,
            "required": False,
            "default": False,
        }

        kwargs, _ = _resolve_param_kwargs(
            param,
            None,
            cmd_name="data_fetch_candles",
        )
        help_text = kwargs["help"]

        assert "forming_candle_status=skipped" in help_text
        assert "full detail" in help_text

    def test_denoise_help_mentions_json_example(self):
        param = {
            "name": "denoise",
            "type": Optional[Dict[str, Any]],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None)
        assert "--denoise kalman" in kwargs["help"]
        assert '"method":"kalman"' in kwargs["help"]

    def test_params_help_mentions_json_and_key_value_examples(self):
        param = {
            "name": "params",
            "type": Optional[Dict[str, Any]],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None)
        assert "--params alpha=0.3,beta=0.1" in kwargs["help"]
        assert '"alpha":0.3' in kwargs["help"]

    def test_features_help_mentions_json_and_key_value_examples(self):
        param = {
            "name": "features",
            "type": Optional[Dict[str, Any]],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None)
        assert "--features lag=3,rolling=5" in kwargs["help"]
        assert '"lag":3' in kwargs["help"]

    def test_forecast_barrier_optimize_method_has_cli_choices(self):
        param = {
            "name": "method",
            "type": ForecastBarrierOptimizeRequest.model_fields["method"].annotation,
            "required": False,
            "default": "auto",
        }
        kwargs, _ = _resolve_param_kwargs(
            param, None, cmd_name="forecast_barrier_optimize"
        )
        assert kwargs["choices"] == [
            "auto",
            "bootstrap",
            "garch",
            "heston",
            "hmm_mc",
            "jump_diffusion",
            "mc_gbm",
            "mc_gbm_bb",
            "ensemble",
        ]
        assert "Barrier simulation method" in kwargs["help"]

    def test_forecast_barrier_optimize_params_help_names_grid_keys(self):
        param = {
            "name": "params",
            "type": Optional[Dict[str, Any]],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(
            param, None, cmd_name="forecast_barrier_optimize"
        )
        help_text = kwargs["help"]
        assert "tp_min" in help_text
        assert "sl_min" in help_text
        assert "ticks" in help_text
        assert "--params" in help_text


# ========================================================================
# _normalize_cli_argv_aliases
# ========================================================================


class TestNormalizeCliArgvAliases:
    def test_normalizes_first_alias_command_token(self):
        functions = {
            "symbols_list": {"func": lambda: None},
            "market_ticker": {"func": lambda: None},
        }

        out = _normalize_cli_argv_aliases(
            ["--timeframe", "H1", "symbols-list", "--search-term", "BTC"],
            functions,
        )

        assert out == ["--timeframe", "H1", "symbols_list", "--search-term", "BTC"]

    def test_normalizes_help_query_alias_keyword(self):
        functions = {
            "trade_place": {"func": lambda: None},
        }

        out = _normalize_cli_argv_aliases(["--help", "trade-place"], functions)

        assert out == ["--help", "trade_place"]

    def test_maps_confluence_command_timeframe_to_pivot_timeframe(self):
        functions = {"confluence_levels": {"func": lambda: None}}

        out = _normalize_cli_argv_aliases(
            ["confluence_levels", "EURUSD", "--timeframe", "H1"],
            functions,
        )

        assert out == [
            "confluence_levels",
            "EURUSD",
            "--pivot-timeframe",
            "H1",
        ]

    def test_keeps_explicit_confluence_pivot_timeframe_when_timeframe_differs(self):
        functions = {"confluence_levels": {"func": lambda: None}}

        with pytest.raises(ValueError, match="both set and differ"):
            _normalize_cli_argv_aliases(
                [
                    "confluence_levels",
                    "EURUSD",
                    "--pivot-timeframe",
                    "D1",
                    "--timeframe",
                    "H1",
                ],
                functions,
            )

    def test_does_not_rewrite_confluence_timeframe_when_pivot_present(self):
        functions = {"confluence_levels": {"func": lambda: None}}

        out = _normalize_cli_argv_aliases(
            [
                "confluence_levels",
                "EURUSD",
                "--pivot-timeframe",
                "D1",
                "--timeframe",
                "D1",
            ],
            functions,
        )

        assert out == [
            "confluence_levels",
            "EURUSD",
            "--pivot-timeframe",
            "D1",
        ]

    def test_keeps_global_confluence_timeframe_for_global_override(self):
        functions = {"confluence_levels": {"func": lambda: None}}

        out = _normalize_cli_argv_aliases(
            ["--timeframe=H1", "confluence_levels", "EURUSD"],
            functions,
        )

        assert out == ["--timeframe=H1", "confluence_levels", "EURUSD"]


# ========================================================================
# _example_value
# ========================================================================


class TestExampleValue:
    def test_known_hint(self):
        param = {"name": "symbol", "type": str, "default": None}
        assert _example_value(param, prefer_default=False) == "EURUSD"

    def test_default_preferred(self):
        param = {"name": "unknown_param", "type": str, "default": "mydefault"}
        assert _example_value(param, prefer_default=True) == "mydefault"

    def test_int_type_fallback(self):
        param = {"name": "weird", "type": int, "default": None}
        assert _example_value(param, prefer_default=False) == "10"

    def test_float_type_fallback(self):
        param = {"name": "weird", "type": float, "default": None}
        assert _example_value(param, prefer_default=False) == "0.1"

    def test_bool_type_fallback(self):
        param = {"name": "weird", "type": bool, "default": None}
        assert _example_value(param, prefer_default=False) == "true"

    def test_list_type_fallback(self):
        param = {"name": "weird", "type": list, "default": None}
        assert _example_value(param, prefer_default=False) == "a,b"

    def test_unknown_type_fallback(self):
        param = {"name": "weird", "type": str, "default": None}
        result = _example_value(param, prefer_default=False)
        assert isinstance(result, str)


# ========================================================================
# _coerce_cli_scalar
# ========================================================================


class TestCoerceCliScalar:
    def test_true(self):
        assert _coerce_cli_scalar("true") is True

    def test_false(self):
        assert _coerce_cli_scalar("false") is False

    def test_null(self):
        assert _coerce_cli_scalar("null") is None

    def test_none_string(self):
        assert _coerce_cli_scalar("none") is None

    def test_integer(self):
        assert _coerce_cli_scalar("42") == 42

    def test_float(self):
        assert _coerce_cli_scalar("3.14") == 3.14

    def test_json_object(self):
        assert _coerce_cli_scalar('{"a": 1}') == {"a": 1}

    def test_json_array(self):
        assert _coerce_cli_scalar("[1, 2]") == [1, 2]

    def test_python_literal_array(self):
        assert _coerce_cli_scalar("[{'type': 'candle_close'}]") == [
            {"type": "candle_close"}
        ]

    def test_plain_string(self):
        assert _coerce_cli_scalar("hello") == "hello"

    def test_empty_string(self):
        assert _coerce_cli_scalar("") == ""

    def test_whitespace_string(self):
        assert _coerce_cli_scalar("  ") == ""

    def test_json_string_with_quotes(self):
        assert _coerce_cli_scalar('"hello"') == "hello"

    def test_TRUE_uppercase(self):
        assert _coerce_cli_scalar("TRUE") is True

    def test_False_mixed_case(self):
        assert _coerce_cli_scalar("False") is False


# ========================================================================
# _normalize_cli_list_value
# ========================================================================


class TestNormalizeCliListValue:
    def test_none(self):
        assert _normalize_cli_list_value(None) is None

    def test_string_comma_separated(self):
        assert _normalize_cli_list_value("a,b,c") == ["a", "b", "c"]

    def test_nested_commas_are_not_split(self):
        assert _normalize_cli_list_value(
            'rsi(14),macd(12,26,9),label="fast,slow"'
        ) == ["rsi(14)", "macd(12,26,9)", 'label="fast,slow"']

    def test_string_space_separated(self):
        assert _normalize_cli_list_value("a b c") == ["a", "b", "c"]

    def test_json_array(self):
        assert _normalize_cli_list_value('["x","y"]') == ["x", "y"]

    def test_python_literal_array(self):
        assert _normalize_cli_list_value(
            "[{'type':'price_change','threshold_value':0.1}]"
        ) == [{"type": "price_change", "threshold_value": 0.1}]

    def test_list_passthrough(self):
        assert _normalize_cli_list_value(["a", "b"]) == ["a", "b"]

    def test_empty_string(self):
        assert _normalize_cli_list_value("") == []

    def test_tuple_input(self):
        assert _normalize_cli_list_value(("a", "b")) == ["a", "b"]

    def test_non_string_non_list(self):
        assert _normalize_cli_list_value(42) == 42

    def test_nested_list_with_strings(self):
        result = _normalize_cli_list_value(["a,b", "c"])
        assert "a" in result and "b" in result and "c" in result

    def test_list_with_non_string_items(self):
        result = _normalize_cli_list_value([1, 2])
        assert 1 in result and 2 in result

    def test_list_with_none_items(self):
        result = _normalize_cli_list_value(["a", None, "b"])
        assert result == ["a", "b"]


# ========================================================================
# _parse_set_overrides
# ========================================================================


class TestParseSetOverrides:
    def test_none(self):
        assert _parse_set_overrides(None) == {}

    def test_empty_list(self):
        assert _parse_set_overrides([]) == {}

    def test_single_override(self):
        result = _parse_set_overrides(["method.sp=24"])
        assert result == {"method": {"sp": 24}}

    def test_multiple_overrides(self):
        result = _parse_set_overrides(["method.sp=24", "method.max_epochs=20"])
        assert result["method"]["sp"] == 24
        assert result["method"]["max_epochs"] == 20

    def test_multiple_sections(self):
        result = _parse_set_overrides(["method.sp=24", "denoise.method=wavelet"])
        assert "method" in result
        assert "denoise" in result

    def test_nested_override(self):
        result = _parse_set_overrides(["params.model.window=64"])
        assert result == {"params": {"model": {"window": 64}}}

    def test_invalid_no_equals(self):
        with pytest.raises(ValueError, match="expected section.key=value"):
            _parse_set_overrides(["bad_override"])

    def test_invalid_no_dot(self):
        with pytest.raises(ValueError, match="expected section.key=value"):
            _parse_set_overrides(["key=value"])

    def test_empty_string_items_skipped(self):
        result = _parse_set_overrides(["", "method.x=1", "  "])
        assert result == {"method": {"x": 1}}

    def test_non_string_items_skipped(self):
        result = _parse_set_overrides([None, 123])
        assert result == {}

    def test_boolean_value_coercion(self):
        result = _parse_set_overrides(["method.flag=true"])
        assert result["method"]["flag"] is True

    def test_null_value_coercion(self):
        result = _parse_set_overrides(["method.param=null"])
        assert result["method"]["param"] is None


# ========================================================================
# _merge_dict
# ========================================================================


class TestMergeDict:
    def test_both_none(self):
        assert _merge_dict(None, None) == {}

    def test_dst_only(self):
        assert _merge_dict({"a": 1}, None) == {"a": 1}

    def test_src_only(self):
        assert _merge_dict(None, {"b": 2}) == {"b": 2}

    def test_merge(self):
        assert _merge_dict({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_src_overwrites_dst(self):
        assert _merge_dict({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_merge(self):
        assert _merge_dict({"a": {"x": 1}}, {"a": {"y": 2}}) == {"a": {"x": 1, "y": 2}}


# ========================================================================
# Parameterised tests for broader coverage of _coerce_cli_scalar
# ========================================================================


class TestCoerceCliScalarParameterized:
    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("True", True),
            ("FALSE", False),
            ("Null", None),
            ("NONE", None),
            ("0", 0),
            ("1", 1),
            ("-1", -1),
            ("0.0", 0.0),
            ("-3.14", -3.14),
            ("hello world", "hello world"),
        ],
    )
    def test_coerce_values(self, input_val, expected):
        assert _coerce_cli_scalar(input_val) == expected


# ========================================================================
# Parameterised tests for _normalize_cli_list_value
# ========================================================================


class TestNormalizeCliListParameterized:
    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("a b c", ["a", "b", "c"]),
            ("a,b,c", ["a", "b", "c"]),
            ('["x"]', ["x"]),
            (["a b", "c,d"], ["a", "b", "c", "d"]),
            (None, None),
            ([], []),
        ],
    )
    def test_normalize(self, input_val, expected):
        assert _normalize_cli_list_value(input_val) == expected


def test_multi_value_symbol_commands_parse_and_join_to_comma_string():
    from mtdata.core.cli.catalog import MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS
    from mtdata.core.cli.runtime.commands import join_cli_symbol_values

    required = {"cross_correlation"}
    for cmd_name in sorted(MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS):
        parser = argparse.ArgumentParser()
        add_dynamic_arguments(
            parser,
            {
                "params": [
                    {
                        "name": "symbols",
                        "type": str,
                        "required": cmd_name in required,
                        "default": None,
                    }
                ]
            },
            cmd_name=cmd_name,
        )
        parsed = parser.parse_args(["EURUSD", "GBPUSD"])
        assert parsed.symbols == ["EURUSD", "GBPUSD"]
        assert join_cli_symbol_values(cmd_name, parsed.symbols) == "EURUSD,GBPUSD"
