"""Integration tests for mtdata.core.cli module.

Tests main entry point, command execution, and end-to-end CLI workflows.
"""

import argparse
import copy
import io
import json
import logging
import sys
import types
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from unittest.mock import MagicMock, call, patch

import pytest

from mtdata.core.analytics_requests import (
    MarketMicrostructureRequest,
    TradeExecutionQualityRequest,
)
from mtdata.core.data.requests import DataFetchCandlesRequest
from mtdata.core.output_serialization import sanitize_json
from mtdata.core.trading.requests import (
    TradeCloseRequest,
    TradeGetOpenRequest,
    TradeHistoryRequest,
    TradeRiskAnalyzeRequest,
)
from mtdata.forecast.requests import ForecastGenerateRequest

pytestmark = pytest.mark.usefixtures("_isolate_env")

# We import lazily inside tests where heavy server machinery is needed,
# but the pure-logic helpers can be imported directly.
from mtdata.core.cli.api import (
    _CLI_DESCRIPTION,
    _argparse_color_enabled,
    _build_epilog,
    _build_usage_examples,
    _coerce_cli_scalar,
    _extract_help_query,
    _first_line,
    _format_cli_literal,
    _format_result_for_cli,
    _is_typed_dict_type,
    _match_commands,
    _merge_dict,
    _normalize_cli_list_value,
    _parse_set_overrides,
    _print_extended_help,
    _quote_cli_value,
    _sort_subparser_help_choices,
    _suggest_commands,
    _type_name,
    create_command_function,
    get_function_info,
    main,
)

# ========================================================================
# main()
# ========================================================================


class TestMain:
    def test_root_option_error_is_not_reclassified_as_disabled_feature(self, capsys):
        from mtdata.core.cli.api import _CLIArgumentParser

        parser = _CLIArgumentParser(prog="mtdata-cli")
        subparsers = parser.add_subparsers(dest="command")
        subparsers.add_parser("market_depth_fetch")
        subparsers.add_parser("tools_list")

        argv = ["--precison", "full", "tools_list", "--json"]
        with patch("sys.argv", ["cli.py", *argv]), pytest.raises(SystemExit):
            parser.parse_args(argv)

        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "cli_invalid_arguments"
        assert payload["operation"] == "cli"
        assert "market_depth_fetch is disabled" not in payload["error"]

    def test_standard_parser_classifies_missing_required_arguments(self, capsys):
        from mtdata.core.cli.api import _CLIArgumentParser

        parser = _CLIArgumentParser(prog="mtdata-cli indicators_describe")
        parser.add_argument("name")

        with (
            patch("sys.argv", ["cli.py", "indicators_describe", "--json"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            parser.parse_args([])

        assert exc_info.value.code == 2
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "cli_missing_required"
        assert payload["operation"] == "indicators_describe"
        assert payload["error"] == "Missing required argument(s): name."
        assert payload["details"] == {"missing_arguments": ["name"]}
        assert payload["remediation"] == "Provide: name."

    @pytest.mark.parametrize(
        ("command_name", "request_model", "argv", "expected_minutes"),
        [
            (
                "market_microstructure_analyze",
                MarketMicrostructureRequest,
                [
                    "cli.py",
                    "market_microstructure_analyze",
                    "EURUSD",
                    "--start",
                    "2026-08-20T14:00:00Z",
                    "--end",
                    "2026-08-20T14:05:00Z",
                ],
                60,
            ),
            (
                "trade_execution_quality",
                TradeExecutionQualityRequest,
                [
                    "cli.py",
                    "trade_execution_quality",
                    "--start",
                    "2026-08-19T14:00:00Z",
                    "--end",
                    "2026-08-19T15:00:00Z",
                ],
                10_080,
            ),
        ],
    )
    @patch("mtdata.core.cli.api.discover_tools")
    def test_explicit_analytics_windows_preserve_lookback_omission(
        self,
        mock_discover,
        command_name,
        request_model,
        argv,
        expected_minutes,
    ):
        mock_fn = MagicMock(return_value={"success": True})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = command_name
        mock_fn.__doc__ = "Analyze an explicit historical window."

        def command(request):
            pass

        command.__annotations__ = {"request": request_model}
        info = get_function_info(command)
        info["func"] = mock_fn
        mock_discover.return_value = {
            command_name: {
                "func": mock_fn,
                "meta": {"description": "Analyze an explicit historical window"},
                "_cli_func_info": info,
            }
        }

        with patch("sys.argv", argv):
            result = main()

        assert result == 0
        request = mock_fn.call_args.kwargs["request"]
        assert request.minutes_back == expected_minutes
        assert "minutes_back" not in request.model_fields_set
        assert {"start", "end"}.issubset(request.model_fields_set)

    @pytest.mark.parametrize(
        ("command_name", "request_model", "argv", "expected_minutes"),
        [
            (
                "market_microstructure_analyze",
                MarketMicrostructureRequest,
                ["cli.py", "market_microstructure_analyze", "EURUSD"],
                60,
            ),
            (
                "trade_execution_quality",
                TradeExecutionQualityRequest,
                ["cli.py", "trade_execution_quality"],
                10_080,
            ),
        ],
    )
    @patch("mtdata.core.cli.api.discover_tools")
    def test_analytics_commands_apply_omitted_lookback_default(
        self,
        mock_discover,
        command_name,
        request_model,
        argv,
        expected_minutes,
    ):
        mock_fn = MagicMock(return_value={"success": True})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = command_name
        mock_fn.__doc__ = "Analyze a default relative window."

        def command(request):
            pass

        command.__annotations__ = {"request": request_model}
        info = get_function_info(command)
        info["func"] = mock_fn
        mock_discover.return_value = {
            command_name: {
                "func": mock_fn,
                "meta": {"description": "Analyze a default relative window"},
                "_cli_func_info": info,
            }
        }

        with patch("sys.argv", argv):
            result = main()

        assert result == 0
        request = mock_fn.call_args.kwargs["request"]
        assert request.minutes_back == expected_minutes
        assert "minutes_back" not in request.model_fields_set

    @patch("mtdata.core.cli.api.discover_tools")
    def test_version_flag_exits_without_tool_discovery(self, mock_discover, capsys):
        with (
            patch("sys.argv", ["cli.py", "--version"]),
            patch("mtdata.core.cli.api._cli_version", return_value="9.8.7"),
        ):
            result = main()

        assert result == 0
        mock_discover.assert_not_called()
        assert capsys.readouterr().out.strip() == "mtdata-cli 9.8.7"

    @patch("mtdata.core.cli.api.discover_tools", return_value={})
    def test_no_tools(self, mock_discover, capsys):
        result = main()
        assert result == 1
        assert "No tools discovered" in capsys.readouterr().err

    def test_no_tools_reports_discovery_cause(self, capsys):
        from mtdata.core.cli import api as cli_api

        def fail_discovery():
            cli_api._DISCOVERY_ERRORS.append(
                "bootstrap_tools failed: missing optional package"
            )
            return {}

        with patch("mtdata.core.cli.api.discover_tools", side_effect=fail_discovery):
            result = main()

        error_output = capsys.readouterr().err
        assert result == 1
        assert "Discovery error: bootstrap_tools failed: missing optional package" in error_output
        assert "MTDATA_CLI_DEBUG=1" in error_output

    @patch("mtdata.core.cli.api.sys")
    @patch("mtdata.core.cli.api.discover_tools")
    def test_help_query_mode(self, mock_discover, mock_sys, capsys):
        def my_tool(symbol: str):
            """My tool."""
            pass

        info = get_function_info(my_tool)
        mock_discover.return_value = {
            "my_tool": {
                "func": my_tool,
                "meta": {"description": "My tool"},
                "_cli_func_info": info,
            },
        }
        mock_sys.argv = ["cli.py", "--help", "my_tool"]
        mock_sys.stderr = sys.stderr
        mock_sys.stdout = sys.stdout

        result = main()
        assert result == 0

    @patch("mtdata.core.cli.api.discover_tools")
    def test_no_command_shows_help(self, mock_discover, capsys):
        def my_tool(symbol: str):
            """My tool."""
            pass

        mock_discover.return_value = {
            "my_tool": {"func": my_tool, "meta": {"description": "My tool"}},
        }
        with patch("sys.argv", ["cli.py"]):
            result = main()
        assert result == 1
        out = capsys.readouterr().out
        assert "usage: cli.py" in out
        assert "<command>" in out
        assert "Catalog categories" in out
        assert "{my_tool,my-tool}" not in out

    @patch("mtdata.core.cli.api.discover_tools")
    def test_command_execution(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "my_tool"
        mock_fn.__doc__ = "My tool."

        def my_tool(symbol: str):
            """My tool."""
            pass

        info = get_function_info(my_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "my_tool": {
                "func": mock_fn,
                "meta": {"description": "My tool"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "my_tool", "EURUSD"]):
            result = main()
        assert result == 0
        mock_fn.assert_called_once()

    @patch("mtdata.core.cli.api.discover_tools")
    def test_hyphenated_command_alias_executes_tool(self, mock_discover):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "symbols_list"
        mock_fn.__doc__ = "List symbols."

        def symbols_list(search_term: str | None = None):
            """List symbols."""
            pass

        info = get_function_info(symbols_list)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "symbols_list": {
                "func": mock_fn,
                "meta": {"description": "List symbols"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "symbols-list", "--search-term", "BTC"]):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(search_term="BTC", __cli_raw=True)

    @patch("mtdata.core.cli.api.discover_tools")
    def test_correlation_matrix_keeps_optional_first_positional_symbols(
        self, mock_discover
    ):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "correlation_matrix"
        mock_fn.__doc__ = "Correlation matrix."

        def correlation_matrix(
            symbols: Optional[str] = None, group: Optional[str] = None
        ):
            """Correlation matrix."""
            pass

        info = get_function_info(correlation_matrix)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "correlation_matrix": {
                "func": mock_fn,
                "meta": {"description": "Correlation matrix"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "correlation_matrix", "EURUSD", "GBPUSD"]):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(symbols="EURUSD,GBPUSD", __cli_raw=True)

    @patch("mtdata.core.cli.api.discover_tools")
    def test_cross_correlation_accepts_space_separated_symbols_option(
        self, mock_discover
    ):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "cross_correlation"
        mock_fn.__doc__ = "Cross correlation."

        def cross_correlation(symbols: str):
            """Cross correlation."""
            pass

        info = get_function_info(cross_correlation)
        info["func"] = mock_fn
        mock_discover.return_value = {
            "cross_correlation": {
                "func": mock_fn,
                "meta": {"description": "Cross correlation"},
                "_cli_func_info": info,
            },
        }

        with patch(
            "sys.argv",
            ["cli.py", "cross_correlation", "--symbols", "EURUSD", "GBPUSD"],
        ):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(symbols="EURUSD,GBPUSD", __cli_raw=True)

    @patch("mtdata.core.cli.api.discover_tools")
    def test_market_relative_strength_accepts_positional_symbols(
        self, mock_discover
    ):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "market_relative_strength"
        mock_fn.__doc__ = "Relative strength."

        def market_relative_strength(
            symbols: Optional[str] = None, group: Optional[str] = None
        ):
            """Relative strength."""
            pass

        info = get_function_info(market_relative_strength)
        info["func"] = mock_fn
        mock_discover.return_value = {
            "market_relative_strength": {
                "func": mock_fn,
                "meta": {"description": "Relative strength"},
                "_cli_func_info": info,
            },
        }

        with patch(
            "sys.argv",
            ["cli.py", "market_relative_strength", "EURUSD", "GBPUSD"],
        ):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(symbols="EURUSD,GBPUSD", __cli_raw=True)

    @patch("mtdata.core.cli.api.discover_tools")
    def test_correlation_matrix_accepts_group_without_symbols(self, mock_discover):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "correlation_matrix"
        mock_fn.__doc__ = "Correlation matrix."

        def correlation_matrix(
            symbols: Optional[str] = None, group: Optional[str] = None
        ):
            """Correlation matrix."""
            pass

        info = get_function_info(correlation_matrix)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "correlation_matrix": {
                "func": mock_fn,
                "meta": {"description": "Correlation matrix"},
                "_cli_func_info": info,
            },
        }

        with patch(
            "sys.argv", ["cli.py", "correlation_matrix", "--group", "Forex\\Majors"]
        ):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(group="Forex\\Majors", __cli_raw=True)

    @patch("mtdata.core.cli.api.discover_tools")
    def test_causal_discover_signals_accepts_group_without_symbols(
        self,
        mock_discover,
    ):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "causal_discover_signals"
        mock_fn.__doc__ = "Causal discovery."

        def causal_discover_signals(
            symbols: Optional[str] = None, group: Optional[str] = None
        ):
            """Causal discovery."""
            pass

        info = get_function_info(causal_discover_signals)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "causal_discover_signals": {
                "func": mock_fn,
                "meta": {"description": "Causal discovery"},
                "_cli_func_info": info,
            },
        }

        with patch(
            "sys.argv",
            ["cli.py", "causal_discover_signals", "--group", "Forex\\Majors"],
        ):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(group="Forex\\Majors", __cli_raw=True)

    @patch("mtdata.core.cli.api.discover_tools")
    def test_causal_discover_signals_joins_space_separated_symbols(
        self,
        mock_discover,
    ):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "causal_discover_signals"

        def causal_discover_signals(symbols: Optional[str] = None):
            pass

        info = get_function_info(causal_discover_signals)
        info["func"] = mock_fn
        mock_discover.return_value = {
            "causal_discover_signals": {
                "func": mock_fn,
                "meta": {"description": "Causal discovery"},
                "_cli_func_info": info,
            },
        }

        with patch(
            "sys.argv",
            ["cli.py", "causal_discover_signals", "EURUSD", "GBPUSD"],
        ):
            status = main()

        assert status == 0
        mock_fn.assert_called_once_with(symbols="EURUSD,GBPUSD", __cli_raw=True)

    @patch("mtdata.core.cli.api.discover_tools")
    def test_cointegration_test_keeps_optional_first_positional_symbols(
        self, mock_discover
    ):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "cointegration_test"
        mock_fn.__doc__ = "Cointegration test."

        def cointegration_test(
            symbols: Optional[str] = None, group: Optional[str] = None
        ):
            """Cointegration test."""
            pass

        info = get_function_info(cointegration_test)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "cointegration_test": {
                "func": mock_fn,
                "meta": {"description": "Cointegration test"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "cointegration_test", "EURUSD", "GBPUSD"]):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(symbols="EURUSD,GBPUSD", __cli_raw=True)

    @patch("mtdata.core.cli.api.discover_tools")
    def test_cointegration_test_accepts_group_without_symbols(self, mock_discover):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "cointegration_test"
        mock_fn.__doc__ = "Cointegration test."

        def cointegration_test(
            symbols: Optional[str] = None, group: Optional[str] = None
        ):
            """Cointegration test."""
            pass

        info = get_function_info(cointegration_test)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "cointegration_test": {
                "func": mock_fn,
                "meta": {"description": "Cointegration test"},
                "_cli_func_info": info,
            },
        }

        with patch(
            "sys.argv", ["cli.py", "cointegration_test", "--group", "Forex\\Majors"]
        ):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(group="Forex\\Majors", __cli_raw=True)

    @patch("mtdata.core.cli.api.discover_tools")
    def test_trade_risk_analyze_accepts_optional_positional_symbol(
        self, mock_discover
    ):
        mock_fn = MagicMock(return_value={"success": True})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "trade_risk_analyze"
        mock_fn.__doc__ = "Analyze trade risk."

        def trade_risk_analyze(request: TradeRiskAnalyzeRequest):
            """Analyze trade risk."""
            pass

        info = get_function_info(trade_risk_analyze)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "trade_risk_analyze": {
                "func": mock_fn,
                "meta": {"description": "Analyze trade risk"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "trade_risk_analyze", "EURUSD"]):
            result = main()

        assert result == 0
        request = mock_fn.call_args.kwargs["request"]
        assert isinstance(request, TradeRiskAnalyzeRequest)
        assert request.symbol == "EURUSD"

    @pytest.mark.parametrize(
        ("command_name", "request_model"),
        [
            ("trade_close", TradeCloseRequest),
            ("trade_execution_quality", TradeExecutionQualityRequest),
        ],
    )
    @patch("mtdata.core.cli.api.discover_tools")
    def test_trade_commands_accept_nonfirst_optional_positional_symbol(
        self,
        mock_discover,
        command_name,
        request_model,
    ):
        mock_fn = MagicMock(return_value={"success": True})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = command_name
        mock_fn.__doc__ = "Trade command."

        def command(request):
            pass

        command.__annotations__ = {"request": request_model}
        info = get_function_info(command)
        info["func"] = mock_fn
        mock_discover.return_value = {
            command_name: {
                "func": mock_fn,
                "meta": {"description": "Trade command"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", command_name, "EURUSD"]):
            result = main()

        assert result == 0
        request = mock_fn.call_args.kwargs["request"]
        assert isinstance(request, request_model)
        assert request.symbol == "EURUSD"

    @patch("mtdata.core.cli.api.discover_tools")
    def test_trade_get_open_accepts_optional_positional_symbol(self, mock_discover):
        mock_fn = MagicMock(return_value={"success": True})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "trade_get_open"
        mock_fn.__doc__ = "Get open positions."

        def trade_get_open(request: TradeGetOpenRequest):
            """Get open positions."""
            pass

        info = get_function_info(trade_get_open)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "trade_get_open": {
                "func": mock_fn,
                "meta": {"description": "Get open positions"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "trade_get_open", "EURUSD"]):
            result = main()

        assert result == 0
        request = mock_fn.call_args.kwargs["request"]
        assert isinstance(request, TradeGetOpenRequest)
        assert request.symbol == "EURUSD"

    @patch("mtdata.core.cli.api.discover_tools")
    def test_market_scan_keeps_optional_first_positional_symbols(self, mock_discover):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "market_scan"
        mock_fn.__doc__ = "Market scan."

        def market_scan(symbols: Optional[str] = None, group: Optional[str] = None):
            """Market scan."""
            pass

        info = get_function_info(market_scan)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "market_scan": {
                "func": mock_fn,
                "meta": {"description": "Market scan"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "market_scan", "EURUSD", "GBPUSD"]):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(symbols="EURUSD,GBPUSD", __cli_raw=True)

    @patch("mtdata.core.cli.api.discover_tools")
    def test_market_radar_keeps_optional_first_positional_symbols(self, mock_discover):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "market_radar"
        mock_fn.__doc__ = "Market radar."

        def market_radar(symbols: Optional[str] = None):
            """Market radar."""
            pass

        info = get_function_info(market_radar)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "market_radar": {
                "func": mock_fn,
                "meta": {"description": "Market radar"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "market_radar", "EURUSD", "GBPUSD"]):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(symbols="EURUSD,GBPUSD", __cli_raw=True)

    @patch("mtdata.core.cli.api.discover_tools")
    def test_market_scan_accepts_group_without_symbols(self, mock_discover):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "market_scan"
        mock_fn.__doc__ = "Market scan."

        def market_scan(symbols: Optional[str] = None, group: Optional[str] = None):
            """Market scan."""
            pass

        info = get_function_info(market_scan)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "market_scan": {
                "func": mock_fn,
                "meta": {"description": "Market scan"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "market_scan", "--group", "Forex\\Majors"]):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(group="Forex\\Majors", __cli_raw=True)

    @patch("mtdata.core.cli.api.discover_tools")
    def test_market_status_keeps_optional_first_positional_symbol(self, mock_discover):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "market_status"
        mock_fn.__doc__ = "Market status."

        def market_status(symbol: Optional[str] = None, region: Optional[str] = "all"):
            """Market status."""
            pass

        info = get_function_info(market_status)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "market_status": {
                "func": mock_fn,
                "meta": {"description": "Market status"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "market_status", "EURUSD"]):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(symbol="EURUSD", region="all", __cli_raw=True)

    @patch("mtdata.core.cli.api.discover_tools")
    def test_global_timeframe_before_command_is_applied(self, mock_discover):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "my_tool"
        mock_fn.__doc__ = "My tool."

        def my_tool(symbol: str, timeframe: str = "H1"):
            """My tool."""
            pass

        info = get_function_info(my_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "my_tool": {
                "func": mock_fn,
                "meta": {"description": "My tool"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "--timeframe", "D1", "my_tool", "EURUSD"]):
            result = main()
        assert result == 0
        assert mock_fn.call_args[1]["timeframe"] == "D1"

    @patch("mtdata.core.cli.api.discover_tools")
    def test_command_timeframe_overrides_global_timeframe(self, mock_discover):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "my_tool"
        mock_fn.__doc__ = "My tool."

        def my_tool(symbol: str, timeframe: str = "H1"):
            """My tool."""
            pass

        info = get_function_info(my_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "my_tool": {
                "func": mock_fn,
                "meta": {"description": "My tool"},
                "_cli_func_info": info,
            },
        }
        with patch(
            "sys.argv",
            ["cli.py", "--timeframe", "D1", "my_tool", "EURUSD", "--timeframe", "H1"],
        ):
            result = main()
        assert result == 0
        assert mock_fn.call_args[1]["timeframe"] == "H1"

    @patch("mtdata.core.cli.api.discover_tools")
    def test_global_timeframe_rejects_command_without_timeframe(
        self, mock_discover, capsys
    ):
        mock_fn = MagicMock(return_value="output text")

        def catalog(limit: int = 10):
            return {"limit": limit}

        info = get_function_info(catalog)
        info["func"] = mock_fn
        mock_discover.return_value = {
            "catalog": {
                "func": mock_fn,
                "meta": {"description": "Catalog"},
                "_cli_func_info": info,
            }
        }

        with (
            patch("sys.argv", ["cli.py", "--timeframe", "D1", "catalog"]),
            pytest.raises(SystemExit, match="2"),
        ):
            main()

        captured = capsys.readouterr()
        assert captured.err == ""
        assert "success: false" in captured.out
        assert "error_code: cli_invalid_arguments" in captured.out
        assert "--timeframe is not supported" in captured.out
        mock_fn.assert_not_called()

    @patch("mtdata.core.cli.api.discover_tools")
    def test_global_timeframe_defaults_confluence_pivot_timeframe(
        self,
        mock_discover,
    ):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "confluence_levels"
        mock_fn.__doc__ = "Confluence levels."

        def confluence_levels(
            symbol: str,
            pivot_timeframe: str = "D1",
            sr_timeframe: str = "auto",
        ):
            """Confluence levels."""

        info = get_function_info(confluence_levels)
        info["func"] = mock_fn
        mock_discover.return_value = {
            "confluence_levels": {
                "func": mock_fn,
                "meta": {"description": "Confluence levels"},
                "_cli_func_info": info,
            },
        }

        with patch(
            "sys.argv",
            [
                "cli.py",
                "--timeframe",
                "H4",
                "confluence_levels",
                "EURUSD",
            ],
        ):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(
            symbol="EURUSD",
            pivot_timeframe="H4",
            sr_timeframe="auto",
            __cli_raw=True,
        )

    @patch("mtdata.core.cli.api.discover_tools")
    def test_global_timeframe_defaults_optimize_hints_timeframes(
        self,
        mock_discover,
    ):
        mock_fn = MagicMock(return_value={"success": True})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_optimize_hints"
        mock_fn.__doc__ = "Optimize hints."

        def forecast_optimize_hints(
            symbol: str,
            timeframes: Optional[List[str]] = None,
        ):
            """Optimize hints."""

        info = get_function_info(forecast_optimize_hints)
        info["func"] = mock_fn
        mock_discover.return_value = {
            "forecast_optimize_hints": {
                "func": mock_fn,
                "meta": {"description": "Optimize hints"},
                "_cli_func_info": info,
            },
        }

        with patch(
            "sys.argv",
            [
                "cli.py",
                "--timeframe",
                "H1",
                "forecast_optimize_hints",
                "EURUSD",
                "--json",
            ],
        ):
            result = main()

        assert result == 0
        assert mock_fn.call_args.kwargs["symbol"] == "EURUSD"
        assert mock_fn.call_args.kwargs["timeframes"] == ["H1"]

    @patch("mtdata.core.cli.api.discover_tools")
    def test_literal_list_accepts_comma_and_space_separated_tokens(
        self,
        mock_discover,
    ):
        mock_fn = MagicMock(return_value={"success": True})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_optimize_hints"
        mock_fn.__doc__ = "Optimize hints."

        def forecast_optimize_hints(
            symbol: str,
            timeframes: Optional[List[Literal["H4", "D1", "W1"]]] = None,
        ):
            """Optimize hints."""

        info = get_function_info(forecast_optimize_hints)
        info["func"] = mock_fn
        mock_discover.return_value = {
            "forecast_optimize_hints": {
                "func": mock_fn,
                "meta": {"description": "Optimize hints"},
                "_cli_func_info": info,
            },
        }

        with patch(
            "sys.argv",
            [
                "cli.py",
                "forecast_optimize_hints",
                "EURUSD",
                "--timeframes",
                "H4,D1",
                "W1",
                "--json",
            ],
        ):
            result = main()

        assert result == 0
        assert mock_fn.call_args.kwargs["timeframes"] == ["H4", "D1", "W1"]

    @patch("mtdata.core.cli.api.discover_tools")
    def test_explicit_confluence_pivot_timeframe_overrides_global(
        self,
        mock_discover,
    ):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "confluence_levels"
        mock_fn.__doc__ = "Confluence levels."

        def confluence_levels(
            symbol: str,
            pivot_timeframe: str = "D1",
        ):
            """Confluence levels."""

        info = get_function_info(confluence_levels)
        info["func"] = mock_fn
        mock_discover.return_value = {
            "confluence_levels": {
                "func": mock_fn,
                "meta": {"description": "Confluence levels"},
                "_cli_func_info": info,
            },
        }

        with patch(
            "sys.argv",
            [
                "cli.py",
                "--timeframe",
                "H4",
                "confluence_levels",
                "EURUSD",
                "--pivot-timeframe",
                "W1",
            ],
        ):
            result = main()

        assert result == 0
        assert mock_fn.call_args.kwargs["pivot_timeframe"] == "W1"

    @patch("mtdata.core.cli.api.discover_tools")
    def test_json_output_suppresses_mtdata_logs_in_non_verbose_mode(
        self, mock_discover, capsys
    ):
        def noisy_tool(**_kwargs):
            """Emit a log line before returning structured output."""
            logging.getLogger("mtdata.test_noise").error(
                "noise that should be suppressed"
            )
            return {"ok": True}

        info = get_function_info(noisy_tool)
        root_logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stderr)
        root_logger.addHandler(handler)
        previous_level = root_logger.level
        root_logger.setLevel(logging.INFO)

        mock_discover.return_value = {
            "noisy_tool": {
                "func": noisy_tool,
                "meta": {"description": "Noisy tool"},
                "_cli_func_info": info,
            },
        }

        try:
            with patch("sys.argv", ["cli.py", "--json", "noisy_tool"]):
                result = main()
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(previous_level)

        assert result == 0
        out = capsys.readouterr()
        assert json.loads(out.out)["ok"] is True
        assert "noise that should be suppressed" not in out.err

    @patch("mtdata.core.cli.api.discover_tools")
    def test_env_output_format_json_usessanitize_json(
        self, mock_discover, monkeypatch, capsys
    ):
        def simple_tool(**_kwargs):
            return {"ok": True}

        info = get_function_info(simple_tool)
        mock_discover.return_value = {
            "simple_tool": {
                "func": simple_tool,
                "meta": {"description": "Simple tool"},
                "_cli_func_info": info,
            },
        }

        monkeypatch.setenv("MTDATA_OUTPUT_FORMAT", "json")
        with patch("sys.argv", ["cli.py", "simple_tool"]):
            result = main()

        assert result == 0
        out = capsys.readouterr()
        assert json.loads(out.out)["ok"] is True

    @patch("mtdata.core.cli.api.discover_tools")
    def test_invalid_env_output_format_fails_before_discovery(
        self, mock_discover, monkeypatch, capsys
    ):
        monkeypatch.setenv("MTDATA_OUTPUT_FORMAT", "jsoon")

        with patch("sys.argv", ["cli.py", "tools_list"]):
            result = main()

        assert result == 2
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "cli_invalid_output_format"
        assert payload["operation"] == "cli"
        mock_discover.assert_not_called()

    @patch("mtdata.core.cli.api.discover_tools")
    def test_explicit_json_overrides_invalid_env_output_format(
        self, mock_discover, monkeypatch, capsys
    ):
        monkeypatch.setenv("MTDATA_OUTPUT_FORMAT", "jsoon")
        mock_discover.return_value = {}

        with patch("sys.argv", ["cli.py", "--json", "no-such-command"]):
            result = main()

        assert result == 1
        assert "No tools discovered" in capsys.readouterr().err
        mock_discover.assert_called_once()

    @patch("mtdata.core.cli.api.discover_tools")
    def test_json_output_suppresses_stream_noise_and_third_party_logs(
        self, mock_discover, capsys
    ):
        def noisy_tool(**_kwargs):
            print("stdout noise")
            print("stderr noise", file=sys.stderr)
            logging.getLogger("sentence_transformers").warning("third-party noise")
            return {"ok": True}

        info = get_function_info(noisy_tool)
        root_logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stderr)
        root_logger.addHandler(handler)
        previous_level = root_logger.level
        root_logger.setLevel(logging.INFO)

        mock_discover.return_value = {
            "noisy_tool": {
                "func": noisy_tool,
                "meta": {"description": "Noisy tool"},
                "_cli_func_info": info,
            },
        }

        try:
            with patch("sys.argv", ["cli.py", "--json", "noisy_tool"]):
                result = main()
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(previous_level)

        assert result == 0
        out = capsys.readouterr()
        assert json.loads(out.out)["ok"] is True
        assert "stdout noise" not in out.out
        assert "stderr noise" not in out.err
        assert "third-party noise" not in out.err

    @patch("mtdata.core.cli.api.discover_tools")
    def test_toon_output_suppresses_stream_noise_but_keeps_structured_warnings(
        self, mock_discover, capsys
    ):
        import warnings

        def noisy_tool(**_kwargs):
            print("stdout noise")
            print("stderr noise", file=sys.stderr)
            warnings.warn("runtime warning", RuntimeWarning)
            return {"ok": True}

        info = get_function_info(noisy_tool)
        mock_discover.return_value = {
            "noisy_tool": {
                "func": noisy_tool,
                "meta": {"description": "Noisy tool"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "noisy_tool"]):
            result = main()

        assert result == 0
        out = capsys.readouterr()
        assert "stdout noise" not in out.out
        assert "stderr noise" not in out.err
        assert "ok: true" in out.out
        assert "warnings[1]:" in out.out
        assert "runtime warning" in out.out

    @patch("mtdata.core.cli.api.discover_tools")
    def test_cli_ignores_deprecation_warnings_in_structured_output(
        self, mock_discover, capsys
    ):
        import warnings

        def noisy_tool(**_kwargs):
            warnings.warn("deprecated runtime path", DeprecationWarning)
            warnings.warn("pending deprecation path", PendingDeprecationWarning)
            warnings.warn("runtime warning", RuntimeWarning)
            return {"ok": True}

        info = get_function_info(noisy_tool)
        mock_discover.return_value = {
            "noisy_tool": {
                "func": noisy_tool,
                "meta": {"description": "Noisy tool"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "noisy_tool"]):
            result = main()

        assert result == 0
        out = capsys.readouterr()
        assert "runtime warning" in out.out
        assert "deprecated runtime path" not in out.out
        assert "pending deprecation path" not in out.out

    @patch("mtdata.core.cli.api.discover_tools")
    def test_cli_runtime_warnings_are_compact_and_path_free(
        self, mock_discover, capsys
    ):
        import warnings

        def noisy_tool(**_kwargs):
            warnings.warn_explicit(
                "optimizer input is\npoorly scaled",
                RuntimeWarning,
                filename=r"C:\\Users\\secret\\site-packages\\library.py",
                lineno=42,
            )
            return {"ok": True}

        info = get_function_info(noisy_tool)
        mock_discover.return_value = {
            "noisy_tool": {
                "func": noisy_tool,
                "meta": {"description": "Noisy tool"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "--json", "noisy_tool"]):
            result = main()

        assert result == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["warnings"] == ["optimizer input is poorly scaled"]

    @patch("mtdata.core.cli.api.discover_tools")
    def test_cli_ignores_import_warnings_in_structured_output(
        self, mock_discover, capsys
    ):
        import warnings

        def noisy_tool(**_kwargs):
            warnings.warn("optional package feature unavailable", ImportWarning)
            warnings.warn("runtime warning", RuntimeWarning)
            return {"ok": True}

        info = get_function_info(noisy_tool)
        mock_discover.return_value = {
            "noisy_tool": {
                "func": noisy_tool,
                "meta": {"description": "Noisy tool"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "noisy_tool"]):
            result = main()

        assert result == 0
        out = capsys.readouterr()
        assert "runtime warning" in out.out
        assert "optional package feature unavailable" not in out.out

    @patch("mtdata.core.cli.api.discover_tools")
    def test_cli_ignores_resource_warnings_in_structured_output(
        self, mock_discover, capsys
    ):
        import warnings

        def noisy_tool(**_kwargs):
            warnings.warn_explicit(
                "unclosed database in <sqlite3.Connection object>",
                ResourceWarning,
                filename=r"C:\repo\src\mtdata\forecast\gpu_runtime.py",
                lineno=83,
            )
            warnings.warn("runtime warning", RuntimeWarning)
            return {"ok": True}

        info = get_function_info(noisy_tool)
        mock_discover.return_value = {
            "noisy_tool": {
                "func": noisy_tool,
                "meta": {"description": "Noisy tool"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "noisy_tool"]):
            result = main()

        assert result == 0
        out = capsys.readouterr()
        assert "runtime warning" in out.out
        assert "ResourceWarning" not in out.out
        assert "unclosed database" not in out.out

    @patch("mtdata.core.cli.api.discover_tools")
    def test_cli_ignores_third_party_future_warnings_in_structured_output(
        self, mock_discover, capsys
    ):
        import warnings

        def noisy_tool(**_kwargs):
            warnings.warn_explicit(
                "pandas concat behavior is deprecated",
                FutureWarning,
                filename=r"C:\env\Lib\site-packages\finvizfinance\screener\base.py",
                lineno=134,
            )
            warnings.warn("runtime warning", RuntimeWarning)
            return {"ok": True}

        info = get_function_info(noisy_tool)
        mock_discover.return_value = {
            "noisy_tool": {
                "func": noisy_tool,
                "meta": {"description": "Noisy tool"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "noisy_tool"]):
            result = main()

        assert result == 0
        out = capsys.readouterr()
        assert "runtime warning" in out.out
        assert "pandas concat behavior is deprecated" not in out.out
        assert "site-packages" not in out.out

    @patch("mtdata.core.cli.api.discover_tools")
    def test_cli_ignores_first_party_future_warnings_in_structured_output(
        self, mock_discover, capsys
    ):
        import warnings

        def noisy_tool(**_kwargs):
            warnings.warn_explicit(
                "Downcasting behavior in `replace` is deprecated",
                FutureWarning,
                filename=r"C:\Users\Admin\Documents\Code\mtdata\src\mtdata\services\data_service\history.py",
                lineno=1069,
            )
            return {"ok": True}

        info = get_function_info(noisy_tool)
        mock_discover.return_value = {
            "noisy_tool": {
                "func": noisy_tool,
                "meta": {"description": "Noisy tool"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "noisy_tool"]):
            result = main()

        assert result == 0
        out = capsys.readouterr()
        assert "Downcasting behavior" not in out.out

    @patch("mtdata.core.cli.api.discover_tools")
    def test_help_hides_irrelevant_timeframe_for_trade_account_info(
        self, mock_discover, capsys
    ):
        mock_fn = MagicMock(return_value={"success": True})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "trade_account_info"
        mock_fn.__doc__ = "Trade account info."

        def trade_account_info():
            """Trade account info."""
            pass

        info = get_function_info(trade_account_info)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "trade_account_info": {
                "func": mock_fn,
                "meta": {"description": "Trade account info"},
                "_cli_func_info": info,
            },
        }
        with (
            patch("sys.argv", ["cli.py", "trade_account_info", "--help"]),
            pytest.raises(SystemExit),
        ):
            main()
        out = capsys.readouterr().out
        assert "--timeframe" not in out

    @patch("mtdata.core.cli.api.discover_tools")
    def test_root_help_documents_global_timeframe_precedence(self, mock_discover, capsys):
        def my_tool(symbol: str, timeframe: str = "H1"):
            """My tool."""
            pass

        mock_fn = MagicMock(return_value={"success": True})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "my_tool"
        mock_fn.__doc__ = "My tool."
        info = get_function_info(my_tool)
        info["func"] = mock_fn
        mock_discover.return_value = {
            "my_tool": {
                "func": mock_fn,
                "meta": {"description": "My tool"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "--help"]), pytest.raises(SystemExit):
            main()

        out = " ".join(capsys.readouterr().out.split())
        assert "Default MT5 timeframe for commands with a timeframe parameter" in out
        assert "command-level --timeframe overrides it" in out

    @patch("mtdata.core.cli.api.discover_tools")
    def test_help_exposes_equivalent_symbol_option_for_required_first_arg(
        self, mock_discover, capsys
    ):
        mock_fn = MagicMock(return_value={"success": True})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "my_tool"
        mock_fn.__doc__ = "My tool."

        def my_tool(symbol: str):
            """My tool."""
            pass

        info = get_function_info(my_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "my_tool": {
                "func": mock_fn,
                "meta": {"description": "My tool"},
                "_cli_func_info": info,
            },
        }
        with (
            patch("sys.argv", ["cli.py", "my_tool", "--help"]),
            pytest.raises(SystemExit),
        ):
            main()
        out = capsys.readouterr().out
        assert "positional arguments:" in out
        assert "symbol" in out
        assert "--symbol" in out

    @patch("mtdata.core.cli.api.discover_tools")
    def test_required_symbol_alias_does_not_depend_on_parameter_order(
        self, mock_discover, capsys
    ):
        def conformal(lookback: int = 100, *, symbol: str, **_kwargs):
            return {"success": True, "symbol": symbol, "lookback": lookback}

        mock_discover.return_value = {
            "conformal": {
                "func": conformal,
                "meta": {"description": "Conformal intervals"},
            },
        }

        with patch("sys.argv", ["cli.py", "conformal", "EURUSD", "--json"]):
            assert main() == 0
        assert json.loads(capsys.readouterr().out)["symbol"] == "EURUSD"

        with patch(
            "sys.argv", ["cli.py", "conformal", "--symbol", "EURUSD", "--json"]
        ):
            assert main() == 0
        assert json.loads(capsys.readouterr().out)["symbol"] == "EURUSD"

    @patch("mtdata.core.cli.api.discover_tools")
    def test_required_symbol_accepts_flag_form(self, mock_discover):
        def market_ticker(symbol: str, **_kwargs):
            return {"success": True, "symbol": symbol}

        mock_discover.return_value = {
            "market_ticker": {
                "func": market_ticker,
                "meta": {"description": "Get ticker"},
            }
        }

        with patch("sys.argv", ["cli.py", "market_ticker", "--symbol", "EURUSD"]):
            status = main()

        assert status == 0

    @patch("mtdata.core.cli.api.discover_tools")
    def test_trade_history_days_alias_converts_to_minutes(self, mock_discover):
        mock_fn = MagicMock(return_value=[])
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "trade_history"
        mock_fn.__doc__ = "Trade history."

        def trade_history(request: TradeHistoryRequest):
            """Trade history."""
            pass

        info = get_function_info(trade_history)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "trade_history": {
                "func": mock_fn,
                "meta": {"description": "Trade history"},
                "_cli_func_info": info,
            },
        }
        with patch(
            "sys.argv", ["cli.py", "trade_history", "--days", "2", "--ticket", "123456"]
        ):
            result = main()
        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert isinstance(request, TradeHistoryRequest)
        assert request.minutes_back == 2880
        assert str(request.position_ticket) == "123456"

    @patch("mtdata.core.cli.api.discover_tools")
    def test_trade_history_rejects_conflicting_lookback_aliases(self, mock_discover):
        mock_fn = MagicMock(return_value=[])
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "trade_history"
        mock_fn.__doc__ = "Trade history."

        def trade_history(request: TradeHistoryRequest):
            """Trade history."""
            pass

        info = get_function_info(trade_history)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "trade_history": {
                "func": mock_fn,
                "meta": {"description": "Trade history"},
                "_cli_func_info": info,
            },
        }
        with patch(
            "sys.argv",
            ["cli.py", "trade_history", "--days", "2", "--minutes-back", "60"],
        ), pytest.raises(SystemExit) as caught:
            main()

        assert caught.value.code == 2
        mock_fn.assert_not_called()

    @patch("mtdata.core.cli.api.discover_tools")
    def test_trade_history_side_flag_populates_request(self, mock_discover):
        mock_fn = MagicMock(return_value=[])
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "trade_history"
        mock_fn.__doc__ = "Trade history."

        def trade_history(request: TradeHistoryRequest):
            """Trade history."""
            pass

        info = get_function_info(trade_history)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "trade_history": {
                "func": mock_fn,
                "meta": {"description": "Trade history"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "trade_history", "--side", "short"]):
            result = main()
        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert isinstance(request, TradeHistoryRequest)
        assert request.side == "SHORT"

    @patch("mtdata.core.cli.api.discover_tools")
    def test_command_exception_handled(self, mock_discover, capsys):
        mock_fn = MagicMock(side_effect=RuntimeError("fail!"))
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "bad_tool"
        mock_fn.__doc__ = "Bad tool."

        def bad_tool(symbol: str):
            """Bad tool."""
            pass

        info = get_function_info(bad_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "bad_tool": {
                "func": mock_fn,
                "meta": {"description": "Bad tool"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "bad_tool", "X"]):
            result = main()
        assert result == 1
        assert "Error" in capsys.readouterr().err

    @patch("mtdata.core.cli.api.discover_tools")
    def test_command_exception_is_structured_in_json_mode(self, mock_discover, capsys):
        mock_fn = MagicMock(side_effect=RuntimeError("fail!"))
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "bad_tool"

        def bad_tool(symbol: str):
            pass

        info = get_function_info(bad_tool)
        info["func"] = mock_fn
        mock_discover.return_value = {
            "bad_tool": {
                "func": mock_fn,
                "meta": {"description": "Bad tool"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "bad_tool", "X", "--json"]):
            status = main()

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert status == 1
        assert captured.err == ""
        assert payload["success"] is False
        assert payload["error_code"] == "unexpected_error"
        assert payload["operation"] == "bad_tool"

    @patch("mtdata.core.cli.api.discover_tools")
    def test_command_tool_error_result_returns_nonzero(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={"error": "bad input"})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "bad_tool"
        mock_fn.__doc__ = "Bad tool."

        def bad_tool(symbol: str):
            """Bad tool."""
            pass

        info = get_function_info(bad_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "bad_tool": {
                "func": mock_fn,
                "meta": {"description": "Bad tool"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "bad_tool", "X"]):
            result = main()
        assert result == 1

    @patch("mtdata.core.cli.api.discover_tools")
    def test_command_blocked_trade_preview_returns_nonzero(
        self,
        mock_discover,
        capsys,
    ):
        mock_fn = MagicMock(
            return_value={
                "success": False,
                "status": "preview_blocked",
                "preview_ok": False,
                "blockers": ["missing_stop_loss"],
            }
        )
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "trade_place"
        mock_fn.__doc__ = "Preview a trade."

        def trade_place(symbol: str):
            """Preview a trade."""

        info = get_function_info(trade_place)
        info["func"] = mock_fn
        mock_discover.return_value = {
            "trade_place": {
                "func": mock_fn,
                "meta": {"description": "Preview a trade"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "trade_place", "EURUSD", "--json"]):
            result = main()

        assert result == 1
        assert json.loads(capsys.readouterr().out)["preview_ok"] is False

    @patch("mtdata.core.cli.api.discover_tools")
    def test_command_eligible_trade_preview_returns_zero(
        self,
        mock_discover,
        capsys,
    ):
        mock_fn = MagicMock(
            return_value={"success": True, "status": "preview", "preview_ok": True}
        )
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "trade_place"
        mock_fn.__doc__ = "Preview a trade."

        def trade_place(symbol: str):
            """Preview a trade."""

        info = get_function_info(trade_place)
        info["func"] = mock_fn
        mock_discover.return_value = {
            "trade_place": {
                "func": mock_fn,
                "meta": {"description": "Preview a trade"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "trade_place", "EURUSD", "--json"]):
            result = main()

        assert result == 0
        assert json.loads(capsys.readouterr().out)["preview_ok"] is True

    @patch("mtdata.core.cli.api.discover_tools")
    def test_command_no_action_result_returns_nonzero(self, mock_discover, capsys):
        mock_fn = MagicMock(
            return_value={"message": "No action taken", "no_action": True}
        )
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "noop_tool"
        mock_fn.__doc__ = "No-op tool."

        def noop_tool(symbol: str):
            """No-op tool."""
            pass

        info = get_function_info(noop_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "noop_tool": {
                "func": mock_fn,
                "meta": {"description": "No-op tool"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "noop_tool", "X"]):
            result = main()
        assert result == 1

    @patch("mtdata.core.cli.api.discover_tools")
    def test_command_successful_no_action_result_returns_zero(
        self, mock_discover, capsys
    ):
        mock_fn = MagicMock(
            return_value={
                "success": True,
                "message": "No open positions",
                "no_action": True,
                "count": 0,
                "items": [],
            }
        )
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "noop_tool"
        mock_fn.__doc__ = "No-op tool."

        def noop_tool(symbol: str, __cli_raw: bool = False):
            """No-op tool."""
            return {"success": True}

        info = get_function_info(noop_tool)
        info["func"] = mock_fn
        mock_discover.return_value = {
            "noop_tool": {
                "func": mock_fn,
                "meta": {"description": "No-op tool"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "noop_tool", "X"]):
            result = main()
        assert result == 0

    @patch("mtdata.core.cli.api.discover_tools")
    def test_keyboard_interrupt(self, mock_discover, capsys):
        mock_fn = MagicMock(side_effect=KeyboardInterrupt)
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "slow_tool"
        mock_fn.__doc__ = "Slow tool."

        def slow_tool(symbol: str):
            """Slow tool."""
            pass

        info = get_function_info(slow_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "slow_tool": {
                "func": mock_fn,
                "meta": {"description": "Slow tool"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "slow_tool", "X"]):
            result = main()
        assert result == 1
        assert "Aborted" in capsys.readouterr().err

    @patch("mtdata.core.cli.api.discover_tools")
    def test_debug_mode_traceback(self, mock_discover, capsys, monkeypatch):
        monkeypatch.setenv("MTDATA_CLI_DEBUG", "1")
        mock_fn = MagicMock(side_effect=ValueError("debug test"))
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "debug_tool"
        mock_fn.__doc__ = "Debug tool."

        def debug_tool(symbol: str):
            """Debug tool."""
            pass

        info = get_function_info(debug_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "debug_tool": {
                "func": mock_fn,
                "meta": {"description": "Debug tool"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "debug_tool", "X"]):
            result = main()
        assert result == 1
        err = capsys.readouterr().err
        assert "Traceback" in err or "Error" in err

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_train_waits_in_one_shot_process(
        self, mock_discover, capsys
    ):
        invoked = []

        def forecast_train(symbol: str, wait: bool = False, **_kwargs):
            invoked.append((symbol, wait))
            return {
                "success": True,
                "status": "completed",
                "task_id": "task-1",
                "model_id": "mlf_rf/EURUSD_H1/abc",
            }

        mock_discover.return_value = {
            "forecast_train": {
                "func": forecast_train,
                "meta": {"description": "Train a forecast model"},
            },
        }

        with patch("sys.argv", ["cli.py", "forecast_train", "EURUSD", "--json"]):
            result = main()

        payload = json.loads(capsys.readouterr().out)
        assert result == 0
        assert payload["status"] == "completed"
        assert payload["model_id"] == "mlf_rf/EURUSD_H1/abc"
        assert invoked == [("EURUSD", True)]

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_train_rejects_wait_false_in_one_shot_process(
        self, mock_discover, capsys
    ):
        invoked = []

        def forecast_train(symbol: str, wait: bool = False, **_kwargs):
            invoked.append((symbol, wait))
            return {"success": True, "status": "pending", "task_id": "task-1"}

        mock_discover.return_value = {
            "forecast_train": {
                "func": forecast_train,
                "meta": {"description": "Train a forecast model"},
            },
        }

        with patch("sys.argv", ["cli.py", "forecast_train", "EURUSD", "--wait", "false", "--json"]):
            result = main()

        payload = json.loads(capsys.readouterr().out)
        assert result != 0
        assert payload["error_code"] == "cli_background_process_required"
        assert "--wait false" in payload["error"]
        assert invoked == []

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_train_waits_in_stdin_shell_batch(
        self, mock_discover, monkeypatch, capsys
    ):
        from mtdata.core.cli import api

        invoked = []

        def forecast_train(symbol: str, wait: bool = False, **_kwargs):
            invoked.append((symbol, wait))
            return {
                "success": True,
                "status": "completed",
                "task_id": "task-1",
                "model_id": "mlf_rf/EURUSD_H1/abc",
            }

        mock_discover.return_value = {
            "forecast_train": {
                "func": forecast_train,
                "meta": {"description": "Train a forecast model"},
            },
        }
        monkeypatch.setattr(
            api.sys,
            "stdin",
            io.StringIO("forecast_train EURUSD --json\n"),
        )

        status = api.run_shell(interactive=False)

        assert status == 0
        record = json.loads(capsys.readouterr().out)
        assert record["result"]["status"] == "completed"
        assert record["result"]["model_id"] == "mlf_rf/EURUSD_H1/abc"
        assert invoked == [("EURUSD", True)]

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_train_is_allowed_in_interactive_shell(
        self, mock_discover, monkeypatch, capsys
    ):
        from mtdata.core.cli import api

        invoked = []
        commands = iter(["forecast_train EURUSD --json", "exit"])

        def forecast_train(symbol: str, wait: bool = False, **_kwargs):
            invoked.append((symbol, wait))
            return {"success": True, "status": "pending", "task_id": "task-1"}

        mock_discover.return_value = {
            "forecast_train": {
                "func": forecast_train,
                "meta": {"description": "Train a forecast model"},
            },
        }
        monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))

        status = api.run_shell(interactive=True)

        assert status == 0
        assert invoked == [("EURUSD", False)]
        assert '"status": "pending"' in capsys.readouterr().out


# ========================================================================
# Forecast generate custom parser integration
# ========================================================================


class TestForecastGenerateIntegration:
    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_help_documents_canonical_model_id(
        self, mock_discover, capsys
    ):
        mock_fn = MagicMock()
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."
        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            }
        }

        with patch("sys.argv", ["cli.py", "forecast_generate", "--help"]), pytest.raises(
            SystemExit, match="0"
        ):
            main()

        output = capsys.readouterr().out
        assert "canonical model_id" in output
        assert "method/data_scope/params_hash" in output
        assert "trained-model params_hash" not in output

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_basic(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={"forecast": [1.0, 2.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch("sys.argv", ["cli.py", "forecast_generate", "EURUSD"]):
            result = main()
        assert result == 0
        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args[1]
        request = call_kwargs["request"]
        assert isinstance(request, ForecastGenerateRequest)
        assert request.symbol == "EURUSD"
        assert request.library is None
        assert request.method == "theta"
        assert request.detail == "compact"
        assert request.ci_alpha == 0.0
        assert request.model_cache == "reuse"
        assert call_kwargs["__cli_raw"] is True

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_accepts_ephemeral_model_cache(self, mock_discover):
        mock_fn = MagicMock(return_value={"success": True})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."
        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }

        with patch(
            "sys.argv",
            [
                "cli.py",
                "forecast_generate",
                "EURUSD",
                "--model-cache",
                "ephemeral",
            ],
        ):
            result = main()

        assert result == 0
        assert mock_fn.call_args.kwargs["request"].model_cache == "ephemeral"

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_rejects_async_mode_in_one_shot_process(
        self, mock_discover, capsys
    ):
        mock_fn = MagicMock(return_value={"status": "pending", "task_id": "task-1"})
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."
        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }

        with patch(
            "sys.argv",
            ["cli.py", "forecast_generate", "EURUSD", "--async-mode", "--json"],
        ):
            result = main()

        payload = json.loads(capsys.readouterr().out)
        assert result == 1
        assert payload["error_code"] == "cli_background_process_required"
        assert payload["operation"] == "forecast_generate"
        mock_fn.assert_not_called()

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_allows_async_mode_in_shell_process(
        self, mock_discover
    ):
        mock_fn = MagicMock(return_value={"status": "pending", "task_id": "task-1"})
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."
        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }

        with (
            patch("mtdata.core.cli.api._SHELL_SESSION_DEPTH", 1),
            patch("mtdata.core.cli.api._INTERACTIVE_SHELL_SESSION_DEPTH", 1),
            patch(
                "sys.argv",
                ["cli.py", "forecast_generate", "EURUSD", "--async-mode"],
            ),
        ):
            result = main()

        assert result == 0
        assert mock_fn.call_args.kwargs["request"].async_mode is True

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_missing_symbol_is_usage_error(
        self, mock_discover, capsys
    ):
        mock_discover.return_value = {
            "forecast_generate": {
                "func": MagicMock(),
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch("sys.argv", ["cli.py", "forecast_generate", "--json"]):
            status = main()

        assert status == 2
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "cli_missing_required"
        assert payload["operation"] == "forecast_generate"
        assert "symbol" in payload["error"]

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_invalid_timeframe_is_usage_error(
        self, mock_discover, capsys
    ):
        mock_discover.return_value = {
            "forecast_generate": {
                "func": MagicMock(),
                "meta": {"description": "Generate forecasts"},
            },
        }
        with (
            patch(
                "sys.argv",
                ["cli.py", "forecast_generate", "EURUSD", "--timeframe", "M7", "--json"],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 2
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "cli_invalid_arguments"
        assert payload["operation"] == "forecast_generate"
        assert "invalid choice" in payload["error"]

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_maps_full_detail_to_request(self, mock_discover):
        mock_fn = MagicMock(return_value={"forecast": [1.0, 2.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch(
            "sys.argv",
            ["cli.py", "forecast_generate", "EURUSD", "--detail", "full"],
        ):
            result = main()
        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert request.detail == "full"

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_preserves_domain_detail(self, mock_discover):
        mock_fn = MagicMock(return_value={"forecast": [1.0, 2.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."
        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }

        with patch(
            "sys.argv",
            ["cli.py", "forecast_generate", "EURUSD", "--detail", "standard"],
        ):
            result = main()

        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert request.detail == "standard"

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_forwards_shared_output_controls(self, mock_discover):
        mock_fn = MagicMock(
            return_value={"success": True, "forecast": [1.0], "method": "theta"}
        )
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."
        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }

        with patch(
            "sys.argv",
            [
                "cli.py",
                "forecast_generate",
                "EURUSD",
                "--detail",
                "full",
                "--output-fields",
                "forecast,method",
            ],
        ):
            result = main()

        assert result == 0
        call_kwargs = mock_fn.call_args.kwargs
        assert call_kwargs["request"].detail == "full"
        assert call_kwargs["output_fields"] == "forecast,method"

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_accepts_symbol_flag_alias(self, mock_discover):
        mock_fn = MagicMock(return_value={"forecast": [1.0, 2.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch(
            "sys.argv", ["cli.py", "forecast_generate", "--symbol", "BTCUSD"]
        ):
            result = main()
        assert result == 0
        assert mock_fn.call_args.kwargs["request"].symbol == "BTCUSD"

    @pytest.mark.parametrize(
        "selectors",
        [
            ["EURUSD", "--symbol", "GBPUSD"],
            ["--symbol", "GBPUSD", "EURUSD"],
        ],
    )
    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_rejects_conflicting_symbol_forms(
        self, mock_discover, selectors, capsys
    ):
        mock_fn = MagicMock(return_value={"forecast": [1.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."
        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }

        with patch("sys.argv", ["cli.py", "forecast_generate", *selectors, "--json"]):
            status = main()

        assert status == 2
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "cli_invalid_arguments"
        assert "not both" in payload["error"]
        mock_fn.assert_not_called()

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_rejects_malformed_params(self, mock_discover):
        mock_fn = MagicMock(return_value={"forecast": [1.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."
        mock_discover.return_value = {
            "forecast_generate": {"func": mock_fn, "meta": {"description": "Forecast"}}
        }

        with (
            patch(
                "sys.argv",
                ["cli.py", "forecast_generate", "EURUSD", "--params", "bad"],
            ),
            pytest.raises(SystemExit, match="2"),
        ):
            main()

        mock_fn.assert_not_called()

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_rejects_unknown_set_section(self, mock_discover):
        mock_fn = MagicMock(return_value={"forecast": [1.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."
        mock_discover.return_value = {
            "forecast_generate": {"func": mock_fn, "meta": {"description": "Forecast"}}
        }

        with (
            patch(
                "sys.argv",
                ["cli.py", "forecast_generate", "EURUSD", "--set", "bogus.foo=1"],
            ),
            pytest.raises(SystemExit, match="2"),
        ):
            main()

        mock_fn.assert_not_called()

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_print_config(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={"forecast": [1.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch(
            "sys.argv", ["cli.py", "forecast_generate", "EURUSD", "--print-config"]
        ):
            result = main()
        assert result == 0
        # --print-config should NOT call the underlying function
        mock_fn.assert_not_called()
        out = capsys.readouterr().out
        assert "EURUSD" in out

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_print_config_honors_json(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={"forecast": [1.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."
        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }

        with patch(
            "sys.argv",
            ["cli.py", "forecast_generate", "EURUSD", "--print-config", "--json"],
        ):
            result = main()

        assert result == 0
        mock_fn.assert_not_called()
        payload = json.loads(capsys.readouterr().out)
        assert payload["success"] is True
        assert payload["forecast_generate"]["symbol"] == "EURUSD"

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_print_config_validates_and_projects(
        self, mock_discover, capsys
    ):
        mock_fn = MagicMock(return_value={"forecast": [1.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."
        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }

        with patch(
            "sys.argv",
            [
                "cli.py",
                "forecast_generate",
                "EURUSD",
                "--method",
                "theta",
                "--params",
                "p=2",
                "--print-config",
                "--json",
            ],
        ):
            invalid_status = main()
        invalid = json.loads(capsys.readouterr().out)
        assert invalid_status == 1
        assert invalid["error_code"] == "unknown_parameter"

        with patch(
            "sys.argv",
            [
                "cli.py",
                "forecast_generate",
                "EURUSD",
                "--print-config",
                "--output-fields",
                "forecast_generate.method",
                "--json",
            ],
        ):
            projected_status = main()
        projected = json.loads(capsys.readouterr().out)
        assert projected_status == 0
        assert projected == {
            "success": True,
            "forecast_generate": {"method": "theta"},
        }

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_json_format(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value="text forecast")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch("sys.argv", ["cli.py", "forecast_generate", "EURUSD", "--json"]):
            result = main()
        assert result == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["text"] == "text forecast"

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_invalid_horizon_returns_json_error(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={"forecast": [1.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."
        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }

        with (
            patch(
                "sys.argv",
                ["cli.py", "--json", "forecast_generate", "EURUSD", "--horizon", "0"],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 2
        mock_fn.assert_not_called()
        payload = json.loads(capsys.readouterr().out)
        assert payload["success"] is False
        assert payload["error_code"] == "cli_invalid_arguments"
        assert payload["operation"] == "forecast_generate"
        assert payload["error"] == "horizon must be between 1 and 500."
        assert "ForecastGenerateRequest" not in payload["error"]

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_tool_error_returns_nonzero(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={"error": "forecast failed"})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch("sys.argv", ["cli.py", "forecast_generate", "EURUSD"]):
            result = main()
        assert result == 1

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_with_overrides(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={"forecast": [1.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch(
            "sys.argv",
            [
                "cli.py",
                "forecast_generate",
                "EURUSD",
                "--set",
                "method.sp=24",
                "--set",
                "method.max_epochs=20",
            ],
        ):
            result = main()
        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert request.params["sp"] == 24
        assert request.params["max_epochs"] == 20

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_uses_global_timeframe_before_command(
        self, mock_discover
    ):
        mock_fn = MagicMock(return_value={"forecast": [1.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch(
            "sys.argv", ["cli.py", "--timeframe", "D1", "forecast_generate", "EURUSD"]
        ):
            result = main()
        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert request.timeframe == "D1"

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_with_denoise(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value="ok")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch(
            "sys.argv",
            ["cli.py", "forecast_generate", "EURUSD", "--denoise", "wavelet"],
        ):
            result = main()
        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert request.denoise == {"method": "wavelet"}

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_rejects_removed_verbose_flag(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={"forecast": [1.0, 2.0, 3.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch("sys.argv", ["cli.py", "forecast_generate", "EURUSD", "--verbose"]):
            with pytest.raises(SystemExit, match="2"):
                main()

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_omits_redundant_ci_block_when_bounds_rendered(
        self, mock_discover, capsys
    ):
        mock_fn = MagicMock(
            return_value={
                "times": ["2026-03-07 18:00"],
                "forecast_price": [67580.67],
                "lower_price": [63025.26],
                "upper_price": [71823.47],
                "ci_status": "available",
                "ci_alpha": 0.05,
            }
        )
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch(
            "sys.argv", ["cli.py", "forecast_generate", "BTCUSD", "--timeframe", "D1"]
        ):
            result = main()
        assert result == 0
        out = capsys.readouterr().out
        assert "forecast[1]{time,forecast,lower,upper}:" in out
        assert "\nci:" not in out

    @patch("mtdata.core.cli.api.discover_tools")
    def test_forecast_generate_default_output_includes_compact_context(
        self, mock_discover, capsys
    ):
        mock_fn = MagicMock(
            return_value={
                "forecast_time": ["2026-03-07 18:00"],
                "forecast_price": [1.17308],
                "method": "arima",
                "quantity": "price",
                "detail": "compact",
                "ci_status": "available",
                "ci_alpha": 0.05,
                "interval_summary": {
                    "first_low": 1.1702,
                    "first_high": 1.1729,
                    "median_width": 0.0027,
                },
            }
        )
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch("sys.argv", ["cli.py", "forecast_generate", "EURUSD"]):
            result = main()
        assert result == 0
        out = capsys.readouterr().out
        assert "method: arima" in out
        assert "quantity: price" in out
        assert "detail: compact" in out
        assert "ci:" in out
        assert "status: available" in out
        assert "interval_summary:" in out
        assert "forecast[1]{time,forecast}:" in out


# ========================================================================
# Edge cases / misc coverage
# ========================================================================


class TestEdgeCases:

    @patch("mtdata.core.cli.api.discover_tools")
    def test_invalid_extras_preserves_json_argument_error_contract(
        self, mock_discover, capsys
    ):
        def sample_tool():
            return {"success": True, "value": 1}

        mock_discover.return_value = {
            "sample_tool": {
                "func": sample_tool,
                "meta": {"description": "Sample tool"},
            }
        }

        with patch(
            "sys.argv",
            ["cli.py", "sample_tool", "--extras", "nonsense", "--json"],
        ), pytest.raises(SystemExit, match="2"):
            main()

        payload = json.loads(capsys.readouterr().out)
        assert payload["success"] is False
        assert payload["error_code"] == "cli_invalid_arguments"
        assert "extras" in payload["error"]

    @patch("mtdata.core.cli.api.discover_tools")
    def test_all_unresolved_output_fields_fail_with_structured_json(
        self, mock_discover, capsys
    ):
        def sample_tool(output_fields=None, **_kwargs):
            return {"success": True, "value": 1}

        mock_discover.return_value = {
            "sample_tool": {
                "func": sample_tool,
                "meta": {"description": "Sample tool"},
            }
        }

        with patch(
            "sys.argv",
            ["cli.py", "sample_tool", "--output-fields", "missing", "--json"],
        ):
            status = main()

        assert status == 1
        assert mock_discover.called
        payload = json.loads(capsys.readouterr().out)
        assert payload["success"] is False
        assert payload["error_code"] == "output_fields_unresolved"
        assert payload["output_fields_status"] == "failed"
        assert payload["unresolved_output_fields"] == ["missing"]
        assert payload["valid_output_fields"] == ["value"]
        assert "valid_output_fields" in payload["remediation"]

    @patch("mtdata.core.cli.api.discover_tools")
    def test_partial_output_fields_succeed_with_unresolved_list(
        self, mock_discover, capsys
    ):
        def sample_tool(output_fields=None, **_kwargs):
            return {"success": True, "symbol": "EURUSD", "value": 1}

        mock_discover.return_value = {
            "sample_tool": {
                "func": sample_tool,
                "meta": {"description": "Sample tool"},
            }
        }

        with patch(
            "sys.argv",
            ["cli.py", "sample_tool", "--output-fields", "symbol,missing", "--json"],
        ):
            status = main()

        assert status == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["success"] is True
        assert payload["symbol"] == "EURUSD"
        assert payload["output_fields_status"] == "partial"
        assert payload["unresolved_output_fields"] == ["missing"]

    @patch("mtdata.core.cli.api.discover_tools")
    def test_all_unresolved_output_fields_fail_in_toon(
        self, mock_discover, capsys, monkeypatch
    ):
        def sample_tool(output_fields=None, **_kwargs):
            return {"success": True, "value": 1}

        mock_discover.return_value = {
            "sample_tool": {
                "func": sample_tool,
                "meta": {"description": "Sample tool"},
            }
        }
        monkeypatch.setenv("MTDATA_OUTPUT_FORMAT", "toon")

        with patch(
            "sys.argv",
            ["cli.py", "sample_tool", "--output-fields", "missing"],
        ):
            status = main()

        assert status == 1
        output = capsys.readouterr().out
        assert "success: false" in output
        assert "error_code: output_fields_unresolved" in output
        assert "output_fields_status: failed" in output
        assert "unresolved_output_fields[1]: missing" in output
        assert "valid_output_fields[1]: value" in output
        assert "Choose one or more paths from valid_output_fields" in output

    @patch("mtdata.core.cli.api.discover_tools")
    def test_trade_projection_has_same_selected_fields_in_toon_and_json(
        self, mock_discover, capsys, monkeypatch
    ):
        def trade_risk_analyze(symbol: str, output_fields=None, **_kwargs):
            return {
                "success": True,
                "symbol": symbol,
                "account": {"equity": 10_000.0},
                "trade_evaluation": {
                    "status": "valid",
                    "reward_risk_ratio": 2.0,
                },
            }

        mock_discover.return_value = {
            "trade_risk_analyze": {
                "func": trade_risk_analyze,
                "meta": {"description": "Analyze trade risk"},
            }
        }
        monkeypatch.setenv("MTDATA_OUTPUT_FORMAT", "toon")

        with patch(
            "sys.argv",
            [
                "cli.py",
                "trade_risk_analyze",
                "EURUSD",
                "--output-fields",
                "trade_evaluation",
            ],
        ):
            toon_status = main()
        toon_output = capsys.readouterr().out

        with patch(
            "sys.argv",
            [
                "cli.py",
                "trade_risk_analyze",
                "EURUSD",
                "--output-fields",
                "trade_evaluation",
                "--json",
            ],
        ):
            json_status = main()
        json_output = json.loads(capsys.readouterr().out)

        assert toon_status == json_status == 0
        assert set(json_output) == {"success", "symbol", "trade_evaluation"}
        assert "success: true" in toon_output
        assert "symbol: EURUSD" in toon_output
        assert "trade_evaluation:" in toon_output
        assert "status: valid" in toon_output
        assert "reward_risk_ratio: 2.0" in toon_output
        assert "account" not in toon_output

    @patch("mtdata.core.cli.api.discover_tools")
    def test_empty_trade_collection_accepts_declared_row_path(
        self, mock_discover, capsys
    ):
        def trade_get_open(output_fields=None, **_kwargs):
            return {"success": True, "count": 0, "items": [], "empty": True}

        mock_discover.return_value = {
            "trade_get_open": {
                "func": trade_get_open,
                "meta": {"description": "Open positions"},
            }
        }

        with patch(
            "sys.argv",
            [
                "cli.py",
                "trade_get_open",
                "--output-fields",
                "items.symbol",
                "--json",
            ],
        ):
            status = main()

        assert status == 0
        assert json.loads(capsys.readouterr().out) == {
            "success": True,
            "items": [],
        }

    @patch("mtdata.core.cli.api.discover_tools")
    def test_json_mode_formats_argparse_failures_as_json(self, mock_discover, capsys):
        def choice_tool(mode: Literal["fast", "slow"] = "fast"):
            return {"success": True, "mode": mode}

        mock_discover.return_value = {
            "choice_tool": {
                "func": choice_tool,
                "meta": {"description": "Choice tool"},
            },
        }

        with patch(
            "sys.argv",
            ["cli.py", "choice_tool", "--mode", "invalid", "--json"],
        ), pytest.raises(SystemExit, match="2"):
            main()

        payload = json.loads(capsys.readouterr().out)
        assert payload["success"] is False
        assert payload["error_code"] == "cli_invalid_arguments"
        assert "invalid choice" in payload["error"]
    def test_coerce_cli_scalar_numeric_like_string(self):
        assert _coerce_cli_scalar("123abc") == "123abc"

    def test_coerce_cli_scalar_just_dot(self):
        result = _coerce_cli_scalar(".")
        assert result == "."

    def test_merge_dict_empty_both(self):
        assert _merge_dict({}, {}) == {}

    def test_format_cli_literal_complex_object(self):
        class Unserializable:
            def __str__(self):
                return "unserializable"

        # json.dumps should fail, fallback to str
        result = _format_cli_literal(Unserializable())
        assert result == "unserializable"

    def test_normalize_cli_list_value_bad_json(self):
        result = _normalize_cli_list_value("[not valid json")
        assert isinstance(result, list)
        # Should fallback to comma/space splitting
        assert len(result) >= 1

    def test_normalize_cli_list_value_empty_list(self):
        result = _normalize_cli_list_value([])
        assert result == []

    def test_parse_set_overrides_section_key_empty(self):
        with pytest.raises(ValueError):
            _parse_set_overrides([".key=value"])

    def test_parse_set_overrides_key_empty(self):
        with pytest.raises(ValueError):
            _parse_set_overrides(["section.=value"])

    def test_json_default_with_bad_bytes(self):
        result = sanitize_json(b"\xff\xfe")
        assert isinstance(result, str)

    def test_shape_public_tool_output_with_none_cmd(self):
        from mtdata.core._mcp_tools import shape_public_tool_output

        r = {"data": 1}
        out = shape_public_tool_output(r, tool_name=None, detail="full")
        assert "cli_meta" not in out
        assert "tool" not in out["meta"]
        assert "runtime" in out["meta"]

    def test_shape_public_tool_output_normalizes_news_without_import_error(self):
        from mtdata.core._mcp_tools import shape_public_tool_output

        out = shape_public_tool_output(
            {"success": True, "general_news": [], "symbol_news": []},
            tool_name="news",
            detail="compact",
        )

        assert out["success"] is True

    def test_render_cli_result_selects_requested_fields(self, capsys):
        from argparse import Namespace

        from mtdata.core.cli.api import _render_cli_result

        _render_cli_result(
            {"success": True, "bid": 1.1, "ask": 1.2, "symbol": "EURUSD"},
            args=Namespace(
                json=True,
                extras=None,
                fields="bid,ask",
                precision=None,
                verbose=False,
            ),
            cmd_name="market_ticker",
        )

        payload = json.loads(capsys.readouterr().out)
        assert payload == {
            "success": True,
            "symbol": "EURUSD",
            "bid": 1.1,
            "ask": 1.2,
        }

    def test_resolve_param_kwargs_type_resolution_failure(self):
        # A parameter with a weird type that causes exception
        from mtdata.core.cli.api import _resolve_param_kwargs
        class WeirdType:
            pass

        param = {"name": "x", "type": WeirdType, "required": False, "default": None}
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert kwargs["type"] is str  # fallback

    def test_create_command_mapping_companion_on_none_arg(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {
                    "name": "simplify",
                    "type": Dict[str, Any],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        # simplify_params present but simplify is None
        args = argparse.Namespace(
            simplify=None, simplify_params="points=100", json=False, verbose=False
        )
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        # companion params should create the dict
        assert isinstance(call_kwargs.get("simplify"), dict)

    def test_create_command_mapping_companion_merge(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {
                    "name": "simplify",
                    "type": Dict[str, Any],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(
            simplify='{"method":"lttb"}',
            simplify_params="points=100",
            json=False,
            verbose=False,
        )
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["simplify"]["method"] == "lttb"

    def test_denoise_companion_merges_into_method_params(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {
                    "name": "denoise",
                    "type": Dict[str, Any],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(
            denoise='{"method":"sma"}',
            denoise_params="window=7",
            json=False,
            verbose=False,
        )

        assert cmd_fn(args) == 0
        assert mock_fn.call_args.kwargs["denoise"] == {
            "method": "sma",
            "params": {"window": 7},
        }

    def test_denoise_companion_promotes_pipeline_controls(self):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {
                    "name": "denoise",
                    "type": Dict[str, Any],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(
            denoise='{"method":"kalman"}',
            denoise_params="keep_original=true,suffix=_filtered,process_var=0.1",
            json=False,
            verbose=False,
        )

        assert cmd_fn(args) == 0
        assert mock_fn.call_args.kwargs["denoise"] == {
            "method": "kalman",
            "keep_original": True,
            "suffix": "_filtered",
            "params": {"process_var": 0.1},
        }

    @pytest.mark.parametrize("cmd_name", ["data_fetch_candles", "data_fetch_ticks"])
    def test_ranged_data_commands_preserve_omitted_limit(self, cmd_name):
        mock_fn = MagicMock(return_value={"success": True})
        func_info = {
            "func": mock_fn,
            "params": [
                {
                    "name": "limit",
                    "type": int,
                    "required": False,
                    "default": 20,
                }
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name=cmd_name)

        assert cmd_fn(argparse.Namespace(json=False, verbose=False)) == 0
        assert "limit" not in mock_fn.call_args.kwargs

    def test_extract_help_query_no_args(self):
        assert _extract_help_query([]) is None

    def test_format_result_for_cli_empty_fmt(self):
        result = _format_result_for_cli(
            {"a": 1}, fmt="", verbose=False, cmd_name="test"
        )
        assert isinstance(result, str)

    def test_first_line_only_whitespace_lines(self):
        assert _first_line("   \n   \n   ") == ""

    def test_type_name_with_generic(self):
        result = _type_name(List[str])
        assert isinstance(result, str)

    def test_quote_cli_value_no_whitespace(self):
        assert _quote_cli_value("abc123") == "abc123"

    def test_example_value_with_tuple_type(self):
        from mtdata.core.cli.api import _example_value
        param = {"name": "weird", "type": tuple, "default": None}
        result = _example_value(param, prefer_default=False)
        assert isinstance(result, str)

    def test_build_usage_examples_optional_same_as_default(self):
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {"name": "timeframe", "type": str, "required": False, "default": "H1"},
            ]
        }
        base, advanced = _build_usage_examples("cmd", func_info)
        assert "cmd" in base

    def test_match_commands_multi_token_query(self):
        def fetch_data(symbol: str):
            """Fetch candle data."""
            pass

        info = get_function_info(fetch_data)
        fns = {
            "data_fetch_candles": {
                "func": fetch_data,
                "meta": {"description": "Fetch candle data"},
                "_cli_func_info": info,
            },
        }
        matches = _match_commands(fns, "candle data")
        assert len(matches) == 1

    def test_is_typed_dict_with_optional_keys(self):
        class FakeTD:
            __annotations__ = {"x": int}
            __optional_keys__ = frozenset({"x"})

        assert _is_typed_dict_type(FakeTD) is True

    def test_safe_tz_name_with_zone_none(self):
        from mtdata.core.runtime_metadata import _safe_tz_name

        class FakeTZ:
            zone = None

        # An object whose tz hints are all empty yields no name (not a repr string).
        result = _safe_tz_name(FakeTZ())
        assert result is None


# ========================================================================
# help suggestions
# ========================================================================


class TestHelpSuggestions:
    def test_suggest_commands_returns_close_matches(self):
        functions = {
            "forecast_generate": {"func": lambda: None},
            "indicators_list": {"func": lambda: None},
            "market_ticker": {"func": lambda: None},
        }
        assert _suggest_commands(functions, "forecast_generat") == ["forecast_generate"]

    def test_extended_help_shows_did_you_mean_for_typos(self, capsys):
        fns = {
            "indicators_list": {
                "func": lambda: None,
                "meta": {},
                "_cli_func_info": {"params": [], "doc": ""},
            },
            "market_ticker": {
                "func": lambda: None,
                "meta": {},
                "_cli_func_info": {"params": [], "doc": ""},
            },
        }
        _print_extended_help(fns, "indicatr_list")
        out = capsys.readouterr().out
        assert "No commands match 'indicatr_list'." in out
        assert "Did you mean: indicators_list" in out
        assert "Available commands:" not in out
        assert "market_ticker" not in out
        assert "tools_list --search indicatr_list --json" in out

    def test_unknown_help_uses_invoked_program_name(self, capsys, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [r"C:\Users\Admin\Documents\Code\mtdata\src\mtdata\__main__.py"],
        )
        fns = {
            "market_ticker": {
                "func": lambda: None,
                "meta": {},
                "_cli_func_info": {"params": [], "doc": ""},
            },
        }
        _print_extended_help(fns, "foobar")
        out = capsys.readouterr().out
        assert "python -m mtdata --help" in out
        assert "python -m mtdata tools_list --search foobar --json" in out

    def test_extended_help_examples_use_invoked_program_name(self, capsys, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [r"C:\Users\Admin\Documents\Code\mtdata\src\mtdata\__main__.py"],
        )

        def market_status(symbol: Optional[str] = None, venue: Optional[str] = None):
            return None

        fns = {
            "market_status": {
                "func": market_status,
                "meta": {"description": "Market status"},
                "_cli_func_info": get_function_info(market_status),
            },
        }
        _print_extended_help(fns, "market_status")
        out = capsys.readouterr().out
        assert "Example: python -m mtdata market_status --symbol EURUSD" in out
        assert "Example+: python -m mtdata market_status --venue NYSE" in out
        assert "mtdata-cli market_status" not in out


# ========================================================================
# _argparse_color_enabled
# ========================================================================


class TestArgparseColorEnabled:
    def test_no_color_env(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert _argparse_color_enabled() is False

    def test_no_tty(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
        assert _argparse_color_enabled() is False

    def test_tty(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
        assert _argparse_color_enabled() is True


# ========================================================================
# _build_epilog
# ========================================================================


class TestBuildEpilog:
    def test_basic(self):
        def my_func(symbol: str):
            """My func."""
            pass

        info = get_function_info(my_func)
        functions = {
            "my_func": {
                "func": my_func,
                "meta": {"description": "Do something"},
                "_cli_func_info": info,
            },
        }
        epilog = _build_epilog(functions)
        assert "my_func" in epilog
        assert "Commands and Arguments" in epilog

    def test_labels_triple_barrier_renders_canonical_literal_choices(self):
        def labels_triple_barrier(
            symbol: str,
            detail: Literal["compact", "standard", "summary", "full"] = "compact",
        ):
            """Label bars."""
            pass

        info = get_function_info(labels_triple_barrier)
        functions = {
            "labels_triple_barrier": {
                "func": labels_triple_barrier,
                "meta": {"description": "Label bars"},
                "_cli_func_info": info,
            },
        }

        epilog = _build_epilog(functions)

        assert "--summary-only" not in epilog
        assert "--detail{compact,standard,summary,full}" in epilog
        assert "<Literal>" not in epilog

    def test_description_promotes_warm_workflows(self):
        assert "One-shot commands initialize the requested tool family" in _CLI_DESCRIPTION
        assert "mtdata-cli shell" in _CLI_DESCRIPTION
        assert "long-lived stdio or HTTP server" in _CLI_DESCRIPTION

    def test_custom_subparser_help_choices_are_sorted(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        subparsers.add_parser("forecast_volatility_estimate", help="Volatility")
        subparsers.add_parser("forecast_generate", help="Forecast")

        _sort_subparser_help_choices(subparsers)

        assert [action.dest for action in subparsers._choices_actions] == [
            "forecast_generate",
            "forecast_volatility_estimate",
        ]

    def test_formats_required_args_like_runtime_parser(self):
        def trade_place_like(symbol: str, volume: float, order_type: str):
            """Place order."""

        info = get_function_info(trade_place_like)
        functions = {
            "trade_place_like": {
                "func": trade_place_like,
                "meta": {"description": "Place order"},
                "_cli_func_info": info,
            },
        }

        epilog = _build_epilog(functions)
        assert "trade_place_like: symbol<str> --volume<float> --order-type<str>" in epilog

    def test_groups_commands_by_category(self):
        def data_fetch_candles(symbol: str):
            """Fetch candles."""

        def forecast_generate(symbol: str):
            """Generate forecast."""

        def trade_place(symbol: str, volume: float):
            """Place order."""

        functions = {}
        for func in (data_fetch_candles, forecast_generate, trade_place):
            info = get_function_info(func)
            functions[func.__name__] = {
                "func": func,
                "meta": {"description": info["doc"].splitlines()[0]},
                "_cli_func_info": info,
            }

        epilog = _build_epilog(functions)

        assert "Commands and Arguments by Section:" in epilog
        assert "DATA ACCESS:" in epilog
        assert "FORECASTING:" in epilog
        assert "TRADING:" in epilog
        assert epilog.index("DATA ACCESS:") < epilog.index("- data_fetch_candles:")
        assert epilog.index("FORECASTING:") < epilog.index("- forecast_generate:")
        assert epilog.index("TRADING:") < epilog.index("- trade_place:")


# ========================================================================
# _print_extended_help
# ========================================================================


class TestPrintExtendedHelp:
    def _make_functions(self):
        def forecast_gen(symbol: str, horizon: int = 12):
            """Generate forecast."""
            pass

        info = get_function_info(forecast_gen)
        return {
            "forecast_generate": {
                "func": forecast_gen,
                "meta": {"description": "Generate forecast"},
                "_cli_func_info": info,
            },
        }

    def test_matching_query(self, capsys):
        fns = self._make_functions()
        _print_extended_help(fns, "forecast")
        out = capsys.readouterr().out
        assert "forecast_generate" in out
        assert "Example" in out

    def test_no_matching_query(self, capsys):
        fns = self._make_functions()
        _print_extended_help(fns, "nonexistent")
        out = capsys.readouterr().out
        assert "No commands match" in out

    def test_trade_place_help_surfaces_safety_flags(self, capsys):
        def trade_place(
            symbol: str,
            volume: float,
            order_type: str,
            price: float | None = None,
            stop_loss: float | None = None,
            take_profit: float | None = None,
            expiration: str | None = None,
            comment: str | None = None,
            deviation: int = 20,
            dry_run: bool = True,
            require_sl_tp: bool = True,
        ):
            """Place a market or pending order."""
            pass

        info = get_function_info(trade_place)
        fns = {
            "trade_place": {
                "func": trade_place,
                "meta": {"description": "Place a market or pending order"},
                "_cli_func_info": info,
            },
        }

        _print_extended_help(fns, "trade_place")
        out = capsys.readouterr().out
        assert "dry_run=true" in out
        assert "require_sl_tp=true" in out
        assert "market and pending orders default to require_sl_tp=true" in out
        assert "unprotected market fill is always closed defensively" in out
        assert (
            "dry_run=true is the default; set --dry-run false explicitly to send an order to MT5"
            in out
        )

    def test_forecast_train_topic_help_advertises_one_shot_wait(self, capsys):
        def forecast_train(symbol: str, wait: bool = False, method: str = "theta"):
            """Start a background training job."""
            pass

        info = get_function_info(forecast_train)
        fns = {
            "forecast_train": {
                "func": forecast_train,
                "meta": {"description": "Start a background training job"},
                "_cli_func_info": info,
            },
        }

        _print_extended_help(fns, "forecast_train")
        out = capsys.readouterr().out
        assert "wait=true" in out
        assert "wait=false" not in out
        assert "one-shot cli and stdin batches wait by default" in out.lower()
        assert "--wait false is rejected" in out.lower()


# ========================================================================
# _extract_help_query
# ========================================================================


class TestExtractHelpQuery:
    def test_help_with_query(self):
        assert _extract_help_query(["--help", "forecast"]) == "forecast"

    def test_help_with_multi_word_query(self):
        assert (
            _extract_help_query(["--help", "forecast", "generate"])
            == "forecast generate"
        )

    def test_help_without_query(self):
        assert _extract_help_query(["--help"]) is None

    def test_no_help(self):
        assert _extract_help_query(["forecast_generate", "EURUSD"]) is None

    def test_short_help_flag(self):
        assert _extract_help_query(["-h", "data"]) == "data"

    def test_help_followed_by_flag(self):
        assert _extract_help_query(["--help", "--verbose"]) is None

    def test_help_with_query_then_flag(self):
        assert _extract_help_query(["--help", "forecast", "--verbose"]) == "forecast"


# ========================================================================
# _build_usage_examples
# ========================================================================


class TestBuildUsageExamples:
    def test_basic(self):
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {"name": "timeframe", "type": str, "required": False, "default": "H1"},
            ]
        }
        base, advanced = _build_usage_examples("data_fetch_candles", func_info)
        assert "data_fetch_candles" in base
        # The first required param uses prefer_default=True; since default is None it uses hint or placeholder
        assert "symbol" in base.lower() or "EURUSD" in base or "<symbol>" in base

    def test_no_optional_params(self):
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
            ]
        }
        base, advanced = _build_usage_examples("test_cmd", func_info)
        assert advanced is None

    def test_multiple_required(self):
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {"name": "method", "type": str, "required": True, "default": None},
            ]
        }
        base, advanced = _build_usage_examples("test_cmd", func_info)
        assert "--method" in base


# ========================================================================
# _match_commands
# ========================================================================


class TestMatchCommands:
    def _make_functions(self):
        def my_forecast(symbol: str):
            """Generate forecast."""
            pass

        def my_data(symbol: str):
            """Fetch data."""
            pass

        info_f = get_function_info(my_forecast)
        info_d = get_function_info(my_data)
        return {
            "forecast_generate": {
                "func": my_forecast,
                "meta": {"description": "Generate forecast"},
                "_cli_func_info": info_f,
            },
            "data_fetch": {
                "func": my_data,
                "meta": {"description": "Fetch data"},
                "_cli_func_info": info_d,
            },
        }

    def test_match_forecast(self):
        fns = self._make_functions()
        matches = _match_commands(fns, "forecast")
        assert len(matches) == 1
        assert matches[0][0] == "forecast_generate"

    def test_match_data(self):
        fns = self._make_functions()
        matches = _match_commands(fns, "data")
        assert len(matches) == 1

    def test_no_match(self):
        fns = self._make_functions()
        matches = _match_commands(fns, "nonexistent")
        assert len(matches) == 0

    def test_empty_query(self):
        fns = self._make_functions()
        matches = _match_commands(fns, "")
        assert len(matches) == 0

    def test_match_all(self):
        fns = self._make_functions()
        # Both have "symbol" in their params context
        matches = _match_commands(fns, "generate")
        assert len(matches) == 1

    def test_command_family_name_matches_rank_before_loose_help_text(self):
        def command(symbol: str):
            pass

        info = get_function_info(command)
        functions = {
            "equity_profile": {
                "func": command,
                "meta": {"description": "Show insider trades"},
                "_cli_func_info": info,
            },
            "trade_place": {
                "func": command,
                "meta": {"description": "Place an order"},
                "_cli_func_info": info,
            },
            "denoise_describe": {
                "func": command,
                "meta": {"description": "Describe supported options"},
                "_cli_func_info": info,
            },
            "options_chain": {
                "func": command,
                "meta": {"description": "Fetch listed contracts"},
                "_cli_func_info": info,
            },
        }

        assert [row[0] for row in _match_commands(functions, "trade")][:2] == [
            "trade_place",
            "equity_profile",
        ]
        assert [row[0] for row in _match_commands(functions, "options")][:2] == [
            "options_chain",
            "denoise_describe",
        ]


def test_extended_help_surfaces_global_flags(capsys):
    from mtdata.core.cli.api import _print_extended_help
    _print_extended_help({}, 'precision')
    out = capsys.readouterr().out
    assert 'Global options' in out
    assert '--precision' in out


def test_extended_help_exact_global_flag_precedes_command_matches(capsys):
    from mtdata.core.cli.api import _print_extended_help

    def sample(json: bool = False):
        """A command whose documentation mentions json."""

    info = get_function_info(sample)
    functions = {
        "sample": {
            "func": sample,
            "meta": {"description": "Return json output"},
            "_cli_func_info": info,
        }
    }

    _print_extended_help(functions, "json")

    out = capsys.readouterr().out
    assert "Global options matching 'json'" in out
    assert "Extended help for query" not in out


def test_help_search_indexes_reviewed_examples():
    def trade_place(symbol: str):
        pass

    info = get_function_info(trade_place)
    functions = {
        "trade_place": {
            "func": trade_place,
            "meta": {"description": "Place an order"},
            "_cli_func_info": info,
        }
    }

    assert [row[0] for row in _match_commands(functions, "stop-loss 1.00")] == [
        "trade_place"
    ]


@pytest.mark.parametrize(
    ("command", "required_text", "forbidden_text"),
    [
        ("outliers_detect", "--method mad", "theta"),
        ("forecast_volatility_estimate", "--method ewma", "theta"),
        ("equity_profile", "AAPL", "EURUSD"),
        ("news", "AAPL", "EURUSD"),
        ("options_chain", "AAPL", "EURUSD"),
        ("options_expirations", "AAPL", "EURUSD"),
        ("options_heston_calibrate", "AAPL", "EURUSD"),
        ("report_generate", "--template minimal", "<template>"),
        ("portfolio_risk_decompose", "--method bootstrap_historical", "theta"),
        ("options_barrier_price", "150 --strike 155", "<"),
        ("wait_event", "EURUSD --timeframe H1", "mtdata-cli wait_event\n"),
        ("trade_stress_test", "--shocks", "<shocks>"),
        ("cross_correlation", "EURUSD GBPUSD", "<symbols>"),
        ("strategy_validate", "ema_cross", "<candidates>"),
    ],
)
def test_reviewed_usage_examples_are_concrete(command, required_text, forbidden_text):
    from mtdata.core.cli.api import _COMMAND_USAGE_EXAMPLES

    rendered = "\n".join(
        example for example in _COMMAND_USAGE_EXAMPLES[command] if example
    )
    assert required_text in rendered
    assert forbidden_text not in rendered


def test_match_global_flags_matches_prefix():
    from mtdata.core.cli.api import _match_global_flags
    assert [n for n, _ in _match_global_flags('prec')] == ['precision']
    assert _match_global_flags('xyzzy') == []
