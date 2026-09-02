from __future__ import annotations

import subprocess
import sys

from mtdata.core.schema_attach import get_public_tool_schema
from mtdata.core.schema_evaluation import (
    SchemaEvaluationReport,
    SchemaFinding,
    _evaluate_tool,
    evaluate_public_tool_schemas,
    format_schema_evaluation,
)


def test_mcp_singleton_initializes_with_warnings_as_errors() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            "-c",
            "from mtdata.core._mcp_instance import mcp; assert mcp is not None",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_schema_evaluation_report_is_sorted_and_fails_only_on_errors() -> None:
    warning = SchemaFinding(
        "warning",
        "candidate",
        "sample_tool",
        "limit",
        "Potential simplification.",
    )
    report = SchemaEvaluationReport(
        tool_count=91,
        expected_tool_count=91,
        findings=(warning,),
    )

    assert report.ok is True
    assert report.errors == ()
    assert report.warnings == (warning,)
    assert report.to_dict()["warning_count"] == 1
    assert format_schema_evaluation(report).startswith(
        "Schema evaluation PASS: 91/91 tools, 0 errors, 1 warnings"
    )


def test_schema_evaluation_report_error_is_machine_readable() -> None:
    error = SchemaFinding(
        "error",
        "signature_mismatch",
        "sample_tool",
        "symbol",
        "Schema and runtime differ.",
    )
    report = SchemaEvaluationReport(
        tool_count=90,
        expected_tool_count=91,
        findings=(error,),
    )

    assert report.ok is False
    assert report.to_dict()["error_count"] == 1
    assert "sample_tool.symbol" in format_schema_evaluation(report)


def test_schema_evaluation_rejects_generated_placeholder_descriptions() -> None:
    def sample_tool(symbol: str, json: bool = False, output_fields: list[str] | None = None):
        return symbol, json, output_fields

    findings: list[SchemaFinding] = []
    _evaluate_tool(
        "sample_tool",
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["symbol"],
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Value for symbol.",
                },
                "json": {
                    "type": "boolean",
                    "default": False,
                    "description": "Return JSON.",
                },
                "output_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to return.",
                },
            },
        },
        sample_tool,
        findings,
    )

    assert any(
        finding.code == "placeholder_description"
        and finding.parameter == "symbol"
        for finding in findings
    )


def test_schema_evaluation_rejects_unanchored_and_pcre_patterns() -> None:
    def sample_tool(
        symbols: str = "EURUSD",
        json: bool = False,
        output_fields: list[str] | None = None,
    ):
        return symbols, json, output_fields

    findings: list[SchemaFinding] = []
    _evaluate_tool(
        "sample_tool",
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "symbols": {
                    "type": "string",
                    "pattern": r".*\S.*",
                    "description": "Symbols to query.",
                    "default": "EURUSD",
                },
                "json": {
                    "type": "boolean",
                    "default": False,
                    "description": "Return JSON.",
                },
                "output_fields": {
                    "type": "array",
                    "items": {"type": "string", "pattern": r"^\s+$"},
                    "description": "Fields to return.",
                },
            },
        },
        sample_tool,
        findings,
    )

    assert any(
        finding.code == "unanchored_pattern" and finding.parameter == "symbols"
        for finding in findings
    )
    assert any(
        finding.code == "unsupported_pattern_escape"
        and finding.parameter == "output_fields"
        for finding in findings
    )


def test_schema_evaluation_rejects_pydantic_only_constraint_keywords() -> None:
    def sample_tool(
        limit: int = 10,
        json: bool = False,
        output_fields: list[str] | None = None,
    ):
        return limit, json, output_fields

    findings: list[SchemaFinding] = []
    _evaluate_tool(
        "sample_tool",
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "limit": {
                    "type": "integer",
                    "ge": 1,
                    "default": 10,
                    "description": "Maximum rows to return.",
                },
                "json": {
                    "type": "boolean",
                    "default": False,
                    "description": "Return JSON.",
                },
                "output_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to return.",
                },
            },
        },
        sample_tool,
        findings,
    )

    assert any(
        finding.code == "non_json_schema_constraint"
        and finding.parameter == "limit"
        for finding in findings
    )


def test_public_schema_evaluation_has_no_unbounded_numeric_parameters() -> None:
    report = evaluate_public_tool_schemas()

    assert not [
        finding
        for finding in report.findings
        if finding.code == "unbounded_numeric_parameter"
    ]
    assert not [
        finding
        for finding in report.findings
        if finding.code == "non_json_schema_constraint"
    ]
    assert not [
        finding
        for finding in report.findings
        if finding.code in {"unanchored_pattern", "unsupported_pattern_escape"}
    ]


def test_market_radar_allow_partial_has_public_description() -> None:
    evaluate_public_tool_schemas()
    properties = get_public_tool_schema("market_radar")["properties"]

    description = properties["allow_partial"]["description"]
    assert "fail closed" in description
    assert "missing" in description


def test_public_schema_evaluation_counts_always_registered_gated_tool() -> None:
    report = evaluate_public_tool_schemas()

    assert report.tool_count == report.expected_tool_count
    assert get_public_tool_schema("market_depth_fetch")
    assert not [
        finding
        for finding in report.findings
        if finding.code == "tool_count_mismatch"
    ]


def test_trade_modify_confirmation_flags_have_descriptions() -> None:
    evaluate_public_tool_schemas()
    properties = get_public_tool_schema("trade_modify")["properties"]

    assert properties["clear_stop_loss"]["description"] == (
        "Explicitly remove stop-loss protection from the ticket."
    )
    assert properties["clear_take_profit"]["description"] == (
        "Explicitly remove take-profit protection from the ticket."
    )


def test_forecast_lookback_schemas_disclose_har_rv_window_contract() -> None:
    for tool_name in (
        "forecast_backtest_run",
        "forecast_volatility_estimate",
    ):
        properties = get_public_tool_schema(tool_name)["properties"]
        description = str(properties["lookback"]["description"])

        assert "HAR-RV" in description
        assert "params.days" in description
