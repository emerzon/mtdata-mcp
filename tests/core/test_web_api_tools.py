"""Tests for Web API MCP tool catalog + invoke bridge."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from mtdata.core import web_api
from mtdata.core.report.requests import ReportGenerateRequest
from mtdata.core.web_api_tools import (
    DEDICATED_UI_TOOLS,
    MUTATING_TOOLS,
    classify_tool_surface,
    coverage_inventory_rows,
    invoke_tool_for_webapi,
    list_tools_for_webapi,
    tool_requires_confirmation,
)
from mtdata.forecast.exceptions import ForecastError
from mtdata.utils.denoise import DenoiseCausalityError
from mtdata.utils.mt5 import MT5ConnectionError

REPO_ROOT = Path(__file__).resolve().parents[2]


def _documented_tool_inventory() -> tuple[str, list[dict[str, str]]]:
    path = REPO_ROOT / "docs" / "WEBUI_TOOL_COVERAGE.md"
    text = path.read_text(encoding="utf-8")
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 6:
            continue
        rows.append(
            {
                "name": cells[0].strip("`"),
                "category": cells[1],
                "surface": cells[2],
                "frontend": cells[3],
                "confirm": cells[4],
            }
        )
    return text, rows


def _fresh_coverage_inventory_rows() -> list[dict[str, object]]:
    code = (
        "import json; "
        "from mtdata.core.web_api_tools import coverage_inventory_rows; "
        "print(json.dumps(coverage_inventory_rows()))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(completed.stdout)


class TestToolClassification:
    def test_dedicated_and_generic_and_confirm(self):
        assert classify_tool_surface("forecast_generate") == "dedicated_ui"
        assert classify_tool_surface("regime_detect") == "generic_runner"
        assert classify_tool_surface("forecast_tune_optuna") == "intentional_omit"
        assert classify_tool_surface("wait_event") == "intentional_omit"
        assert tool_requires_confirmation("trade_place") is True
        assert tool_requires_confirmation("tools_list") is False
        assert "trade_place" in MUTATING_TOOLS
        assert "forecast_generate" in DEDICATED_UI_TOOLS

    def test_inventory_covers_registry(self):
        rows = coverage_inventory_rows()
        names = {row["name"] for row in rows}
        assert "tools_list" in names
        assert "trade_place" in names
        assert "market_depth_fetch" in names  # gated but listed
        assert len(rows) == len(names)
        # zero unlisted surface values
        for row in rows:
            assert row["surface"] in {"dedicated_ui", "generic_runner", "intentional_omit"}

    def test_documented_inventory_matches_registry(self):
        expected = {
            str(row["name"]): {
                "name": str(row["name"]),
                "category": str(row.get("category") or ""),
                "surface": str(row["surface"]),
                "frontend": str(row["frontend"]),
                "confirm": "yes" if row["requires_confirmation"] else "no",
            }
            for row in _fresh_coverage_inventory_rows()
        }
        text, documented_rows = _documented_tool_inventory()
        documented_names = [row["name"] for row in documented_rows]

        assert len(documented_names) == len(set(documented_names)), (
            "WEBUI_TOOL_COVERAGE.md contains duplicate tool rows"
        )
        assert {row["name"]: row for row in documented_rows} == expected
        assert f"**Total tools in inventory:** {len(expected)} " in text


class TestListAndInvoke:
    def test_list_tools_includes_surface_meta(self):
        payload = list_tools_for_webapi(search="tools_list")
        assert payload["count"] >= 1
        tool = payload["tools"][0]
        assert tool["name"] == "tools_list"
        assert tool["surface"] == "dedicated_ui"
        assert "safety" in tool

    def test_invoke_tools_list(self):
        result = invoke_tool_for_webapi(
            "tools_list",
            arguments={"limit": 2, "detail": "compact"},
        )
        assert result["success"] is True
        assert result["tool"] == "tools_list"
        inner = result["result"]
        assert isinstance(inner, dict)
        assert inner.get("count") == 2

    def test_invoke_applies_public_output_contract_to_request_model(self):
        def report_generate(request: ReportGenerateRequest):
            return {
                "success": True,
                "symbol": request.symbol,
                "detail_seen": request.detail,
                "meta": {"domain": {"template": request.template}},
                "diagnostics": {"source": "test"},
            }

        with (
            patch("mtdata.core.web_api_tools.ensure_tools_bootstrapped"),
            patch(
                "mtdata.core.web_api_tools.get_tool_functions",
                return_value={"report_generate": report_generate},
            ),
        ):
            compact = invoke_tool_for_webapi(
                "report_generate",
                arguments={"symbol": "EURUSD", "template": "minimal"},
            )
            rich = invoke_tool_for_webapi(
                "report_generate",
                arguments={
                    "symbol": "EURUSD",
                    "template": "minimal",
                    "detail": "full",
                },
            )

        assert compact["result"]["detail_seen"] == "compact"
        assert "meta" not in compact["result"]
        assert "diagnostics" not in compact["result"]
        assert rich["result"]["detail_seen"] == "full"
        assert rich["result"]["meta"]["domain"]["template"] == "minimal"
        assert rich["result"]["diagnostics"] == {"source": "test"}

    def test_invoke_adds_guidance_only_when_requested(self):
        def market_ticker(detail: str = "compact"):
            return {
                "success": True,
                "detail_seen": detail,
                "meta": {"domain": {"symbol": "EURUSD"}},
            }

        with (
            patch("mtdata.core.web_api_tools.ensure_tools_bootstrapped"),
            patch(
                "mtdata.core.web_api_tools.get_tool_functions",
                return_value={"market_ticker": market_ticker},
            ),
        ):
            compact = invoke_tool_for_webapi("market_ticker")
            guided = invoke_tool_for_webapi(
                "market_ticker",
                arguments={"detail": "full"},
            )

        assert "related_tools" not in compact["result"]
        assert "meta" not in compact["result"]
        assert guided["result"]["detail_seen"] == "full"
        assert guided["result"]["related_tools"]
        assert "meta" in guided["result"]

    def test_invoke_applies_field_selection(self):
        def demo():
            return {
                "success": True,
                "symbol": "EURUSD",
                "bid": 1.1,
                "ask": 1.2,
            }

        with (
            patch("mtdata.core.web_api_tools.ensure_tools_bootstrapped"),
            patch(
                "mtdata.core.web_api_tools.get_tool_functions",
                return_value={"demo": demo},
            ),
        ):
            result = invoke_tool_for_webapi(
                "demo",
                arguments={"output_fields": "bid"},
            )

        assert result["result"] == {
            "success": True,
            "symbol": "EURUSD",
            "bid": 1.1,
        }

    def test_invoke_rejects_invalid_extras(self):
        with (
            patch("mtdata.core.web_api_tools.ensure_tools_bootstrapped"),
            patch(
                "mtdata.core.web_api_tools.get_tool_functions",
                return_value={"demo": lambda: {"success": True}},
            ),
            pytest.raises(HTTPException) as exc,
        ):
            invoke_tool_for_webapi(
                "demo",
                arguments={"extras": "not-a-real-extra"},
            )

        assert exc.value.status_code == 400
        assert exc.value.detail["parameter"] == "extras"
        assert exc.value.detail["replacement"] == "detail"

    def test_mutation_requires_confirm(self):
        with pytest.raises(HTTPException) as exc:
            invoke_tool_for_webapi("trade_place", arguments={"symbol": "EURUSD"}, confirm=False)
        assert exc.value.status_code == 400
        detail = exc.value.detail
        assert isinstance(detail, dict)
        assert detail.get("requires_confirmation") is True

    @pytest.mark.parametrize("tool_name", ["forecast_tune_genetic", "forecast_tune_optuna"])
    def test_long_running_tuning_is_omitted_from_sync_invoke(self, tool_name):
        with pytest.raises(HTTPException) as exc:
            invoke_tool_for_webapi(tool_name)

        assert exc.value.status_code == 403
        assert exc.value.detail["rationale"].startswith("Long-running optimization")

    def test_wait_event_is_omitted_from_sync_invoke(self):
        with pytest.raises(HTTPException) as exc:
            invoke_tool_for_webapi("wait_event")

        assert exc.value.status_code == 403
        assert "wait_event" in exc.value.detail["rationale"]

    @pytest.mark.parametrize(
        ("error", "status_code", "error_code"),
        [
            (TypeError("unexpected keyword"), 422, "tool_param_error"),
            (ValueError("bad interval"), 422, "tool_param_error"),
            (ForecastError("forecast rejected"), 400, "tool_domain_error"),
            (DenoiseCausalityError("future leak"), 400, "tool_domain_error"),
            (
                MT5ConnectionError("terminal unavailable"),
                503,
                "mt5_connection_error",
            ),
            (RuntimeError("secret internal detail"), 500, "tool_invoke_internal_error"),
        ],
    )
    def test_invoke_classifies_tool_exceptions(
        self, error, status_code, error_code
    ):
        with (
            patch("mtdata.core.web_api_tools.ensure_tools_bootstrapped"),
            patch(
                "mtdata.core.web_api_tools.get_tool_functions",
                return_value={"demo": lambda: None},
            ),
            patch(
                "mtdata.core.web_api_tools.resolve_sync_tool_result",
                side_effect=error,
            ),
            pytest.raises(HTTPException) as exc,
        ):
            invoke_tool_for_webapi("demo")

        assert exc.value.status_code == status_code
        assert exc.value.detail["error_code"] == error_code
        if status_code == 500:
            assert "secret internal detail" not in exc.value.detail["error"]


class TestWebApiRoutes:
    def setup_method(self):
        self.client = TestClient(web_api.app)

    def test_get_tools_route(self):
        res = self.client.get("/api/v1/tools", params={"search": "tools_list"})
        assert res.status_code == 200
        body = res.json()
        assert body["count"] >= 1
        assert any(t["name"] == "tools_list" for t in body["tools"])

    def test_get_tool_detail_route(self):
        res = self.client.get("/api/v1/tools/tools_list")
        assert res.status_code == 200
        body = res.json()
        assert body["tool"]["name"] == "tools_list"
        schema = body["tool"].get("input_schema")
        assert isinstance(schema, dict)
        assert {"json", "output_fields"}.issubset(schema["properties"])
        assert "extras" not in schema["properties"]

    def test_invoke_route(self):
        res = self.client.post(
            "/api/v1/tools/tools_list/invoke",
            json={"arguments": {"limit": 1, "detail": "compact"}, "confirm": False},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        assert body["result"]["count"] == 1

    def test_invoke_trade_without_confirm_blocked(self):
        res = self.client.post(
            "/api/v1/tools/trade_place/invoke",
            json={"arguments": {"symbol": "EURUSD"}, "confirm": False},
        )
        assert res.status_code == 400
        detail = res.json()["detail"]
        assert detail["requires_confirmation"] is True
