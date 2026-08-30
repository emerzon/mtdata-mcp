"""Tests for Web API MCP tool catalog + invoke bridge."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from mtdata.core import web_api
from mtdata.core.report.requests import ReportGenerateRequest
from mtdata.core.trading.requests import TradeModifyRequest, TradePlaceRequest
from mtdata.core.web_api_tools import (
    DEDICATED_UI_TOOLS,
    INTENTIONAL_OMIT_TOOLS,
    MUTATING_TOOLS,
    TOOL_CATALOG_CATEGORIES,
    TOOL_CATALOG_DETAILS,
    TOOLS_CATALOG_DEFAULT_LIMIT,
    classify_tool_surface,
    coverage_inventory_rows,
    get_tool_for_webapi,
    invoke_tool_for_webapi,
    list_tools_for_webapi,
    tool_requires_confirmation,
)
from mtdata.forecast.exceptions import ForecastError
from mtdata.utils.denoise import DenoiseCausalityError
from mtdata.utils.mt5 import MT5ConnectionError

_TRADE_PLACE_ARGS = {
    "symbol": "EURUSD",
    "volume": 0.01,
    "order_type": "BUY",
    "stop_loss": 1.05,
    "take_profit": 1.15,
}
_DOMAIN_FAILURES = [
    ({"success": False, "error": "Unknown symbol EURX", "error_code": "symbol_not_found"}, 404),
    ({"success": False, "error": "volume must be positive", "error_code": "tool_param_error"}, 422),
    ({"success": False, "error": "terminal unavailable", "error_code": "mt5_connection_error"}, 503),
    ({"success": False, "error": "internal boom", "error_code": "tool_invoke_internal_error"}, 500),
    ({"success": False, "error": "preview blocked", "error_code": "preview_blocked"}, 400),
]


def _fake_trade_place(request: TradePlaceRequest):
    return {
        "success": True,
        "dry_run": request.dry_run,
        "preview_ok": bool(request.dry_run),
        "would_send_order": not request.dry_run,
    }


def _trade_place_functions():
    return patch(
        "mtdata.core.web_api_tools.get_tool_functions",
        return_value={"trade_place": _fake_trade_place},
    )


def _assert_error_envelope(
    payload: dict,
    *,
    error_code: str,
    operation: str | None = None,
) -> None:
    assert payload["success"] is False
    assert payload["error_code"] == error_code
    assert isinstance(payload.get("error"), str) and payload["error"].strip()
    assert isinstance(payload.get("request_id"), str) and payload["request_id"].strip()
    if operation is not None:
        assert payload["operation"] == operation


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
        assert set(DEDICATED_UI_TOOLS) <= names
        assert set(INTENTIONAL_OMIT_TOOLS) <= names
        for row in rows:
            name = row["name"]
            surface = row["surface"]
            assert surface in {"dedicated_ui", "generic_runner", "intentional_omit"}
            assert classify_tool_surface(name) == surface
            if surface == "dedicated_ui":
                assert row["frontend"] == DEDICATED_UI_TOOLS[name]
            elif surface == "intentional_omit":
                assert row["frontend"] == INTENTIONAL_OMIT_TOOLS[name]
            else:
                assert row["frontend"] == "tools-runner/generic"


class TestListAndInvoke:
    def test_list_tools_includes_surface_meta(self):
        payload = list_tools_for_webapi(search="tools_list")
        assert payload["count"] >= 1
        tool = payload["tools"][0]
        assert tool["name"] == "tools_list"
        assert tool["surface"] == "dedicated_ui"
        assert "safety" in tool
        assert payload["detail"] == "compact"
        assert payload["pagination"]["limit"] == TOOLS_CATALOG_DEFAULT_LIMIT

    def test_list_tools_rejects_unknown_category(self):
        with pytest.raises(HTTPException) as exc:
            list_tools_for_webapi(category="tradng")
        assert exc.value.status_code == 422
        _assert_error_envelope(
            exc.value.detail, error_code="tool_param_error", operation="tools_list"
        )
        details = exc.value.detail["details"]
        assert details["parameter"] == "category"
        assert "trading" in details["valid_values"]
        assert details["suggestion"] == "trading"

    def test_list_tools_rejects_unknown_detail(self):
        with pytest.raises(HTTPException) as exc:
            list_tools_for_webapi(detail="verbose")
        assert exc.value.status_code == 422
        details = exc.value.detail["details"]
        assert details["parameter"] == "detail"
        assert set(details["valid_values"]) == set(TOOL_CATALOG_DETAILS)

    def test_list_tools_valid_filter_with_no_matches_is_empty(self):
        payload = list_tools_for_webapi(category="trading", search="zzz_no_such_tool")
        assert payload["success"] is True
        assert payload["count"] == 0
        assert payload["tools"] == []
        assert payload["pagination"]["total"] == 0
        assert payload["pagination"]["has_more"] is False

    def test_get_tool_rejects_unknown_detail(self):
        with pytest.raises(HTTPException) as exc:
            get_tool_for_webapi("trade_place", detail="verbose")
        assert exc.value.status_code == 422
        details = exc.value.detail["details"]
        assert details["parameter"] == "detail"
        assert set(details["valid_values"]) == set(TOOL_CATALOG_DETAILS)

    def test_list_tools_pages_compact_catalog(self):
        first = list_tools_for_webapi(limit=2, offset=0)
        second = list_tools_for_webapi(limit=2, offset=2)
        unbounded = list_tools_for_webapi(limit=TOOLS_CATALOG_DEFAULT_LIMIT, offset=0)
        assert first["count"] == 2
        assert first["pagination"]["limit"] == 2
        assert first["pagination"]["offset"] == 0
        assert first["pagination"]["total"] >= 4
        assert first["pagination"]["has_more"] is True
        assert [row["name"] for row in first["tools"]] != [
            row["name"] for row in second["tools"]
        ]
        assert unbounded["count"] <= TOOLS_CATALOG_DEFAULT_LIMIT

    def test_invoke_tools_list(self):
        result = invoke_tool_for_webapi(
            "tools_list",
            arguments={"limit": 2, "detail": "compact"},
        )
        assert result["success"] is True
        assert result["tool"] == "tools_list"
        inner = result["result"]
        assert isinstance(inner, dict)
        assert len(inner["tools"]) == 2
        assert "count" not in inner

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
        assert rich["result"]["meta"]["diagnostics"] == {"source": "test"}

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
        _assert_error_envelope(
            exc.value.detail, error_code="tool_param_error", operation="demo"
        )
        assert exc.value.detail["details"]["parameter"] == "extras"
        assert exc.value.detail["details"]["replacement"] == "detail"

    @pytest.mark.parametrize(
        ("tool_name", "arguments"),
        [
            ("trade_place", {**_TRADE_PLACE_ARGS, "dry_run": False}),
            ("forecast_task_cancel", {"task_id": "task-1"}),
        ],
    )
    def test_live_mutation_requires_confirm(self, tool_name, arguments):
        with pytest.raises(HTTPException) as exc:
            invoke_tool_for_webapi(tool_name, arguments=arguments, confirm=False)
        assert exc.value.status_code == 400
        _assert_error_envelope(
            exc.value.detail,
            error_code="confirmation_required",
            operation=tool_name,
        )
        details = exc.value.detail["details"]
        assert details["requires_confirmation"] is True
        assert "safety" in details
        assert "hint" in details

    @pytest.mark.parametrize("payload, status_code", _DOMAIN_FAILURES)
    def test_invoke_maps_domain_failure_to_http_error(self, payload, status_code):
        with (
            patch("mtdata.core.web_api_tools.ensure_tools_bootstrapped"),
            patch(
                "mtdata.core.web_api_tools.get_tool_functions",
                return_value={"demo": lambda: payload},
            ),
            pytest.raises(HTTPException) as exc,
        ):
            invoke_tool_for_webapi("demo")

        assert exc.value.status_code == status_code
        _assert_error_envelope(
            exc.value.detail,
            error_code=payload["error_code"],
            operation="demo",
        )
        assert exc.value.detail["error"] == payload["error"]

    @pytest.mark.parametrize("tool_name", ["forecast_tune_genetic", "forecast_tune_optuna"])
    def test_long_running_tuning_is_omitted_from_sync_invoke(self, tool_name):
        with pytest.raises(HTTPException) as exc:
            invoke_tool_for_webapi(tool_name)

        assert exc.value.status_code == 403
        _assert_error_envelope(
            exc.value.detail,
            error_code="tool_not_available",
            operation=tool_name,
        )
        assert exc.value.detail["details"]["rationale"].startswith(
            "Long-running optimization"
        )

    def test_wait_event_is_omitted_from_sync_invoke(self):
        with pytest.raises(HTTPException) as exc:
            invoke_tool_for_webapi("wait_event")

        assert exc.value.status_code == 403
        _assert_error_envelope(
            exc.value.detail,
            error_code="tool_not_available",
            operation="wait_event",
        )
        assert "wait_event" in exc.value.detail["details"]["rationale"]

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
        _assert_error_envelope(
            exc.value.detail, error_code=error_code, operation="demo"
        )
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
        assert body["detail"] == "compact"
        assert any(t["name"] == "tools_list" for t in body["tools"])
        assert body["pagination"]["returned"] == body["count"]

    def test_get_tools_default_is_bounded_compact(self):
        res = self.client.get("/api/v1/tools")
        assert res.status_code == 200
        body = res.json()
        assert body["detail"] == "compact"
        assert body["count"] <= TOOLS_CATALOG_DEFAULT_LIMIT
        assert body["pagination"]["limit"] == TOOLS_CATALOG_DEFAULT_LIMIT
        assert body["pagination"]["offset"] == 0
        assert "total" in body["pagination"]
        assert body["pagination"]["returned"] == body["count"]
        if body["pagination"]["total"] > TOOLS_CATALOG_DEFAULT_LIMIT:
            assert body["pagination"]["has_more"] is True

    def test_get_tools_rejects_invalid_category_and_detail(self):
        category = self.client.get("/api/v1/tools", params={"category": "tradng"})
        detail = self.client.get("/api/v1/tools", params={"detail": "verbose"})
        tool_detail = self.client.get(
            "/api/v1/tools/trade_place", params={"detail": "verbose"}
        )
        for res, parameter, known in (
            (category, "category", "trading"),
            (detail, "detail", "compact"),
            (tool_detail, "detail", "compact"),
        ):
            assert res.status_code == 422
            blob = json.dumps(res.json()).lower()
            assert parameter in blob
            assert known in blob

    def test_get_tools_valid_category_empty_search_is_200(self):
        res = self.client.get(
            "/api/v1/tools",
            params={"category": "trading", "search": "zzz_no_such_tool"},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["count"] == 0
        assert body["tools"] == []
        assert "trading" in TOOL_CATALOG_CATEGORIES

    def test_get_tool_detail_route(self):
        res = self.client.get("/api/v1/tools/tools_list")
        assert res.status_code == 200
        body = res.json()
        assert body["detail"] == "compact"
        tool = body["tool"]
        assert tool["name"] == "tools_list"
        assert "description" in tool
        assert "safety" in tool
        schema = tool.get("input_schema")
        assert isinstance(schema, dict)
        assert "json" not in schema["properties"]
        assert "output_fields" in schema["properties"]
        assert "extras" not in schema["properties"]
        assert "cli" not in tool
        assert "module" not in tool

    def test_get_tool_detail_levels_and_include_fields(self):
        compact = self.client.get("/api/v1/tools/data_fetch_candles")
        standard = self.client.get(
            "/api/v1/tools/data_fetch_candles",
            params={"detail": "standard"},
        )
        full = self.client.get(
            "/api/v1/tools/data_fetch_candles",
            params={"detail": "full"},
        )
        without_fields = self.client.get(
            "/api/v1/tools/data_fetch_candles",
            params={"include_fields": False},
        )

        assert compact.status_code == 200
        assert standard.status_code == 200
        assert full.status_code == 200
        assert without_fields.status_code == 200

        compact_tool = compact.json()["tool"]
        standard_tool = standard.json()["tool"]
        full_tool = full.json()["tool"]
        bare_tool = without_fields.json()["tool"]

        assert compact.json()["detail"] == "compact"
        assert standard.json()["detail"] == "standard"
        assert full.json()["detail"] == "full"
        assert {"name", "description", "safety", "input_schema"} <= set(compact_tool)
        assert "cli" not in compact_tool
        assert "module" not in compact_tool
        assert "parameters" in standard_tool
        assert "cli" not in standard_tool
        assert "cli" in full_tool
        assert "module" in full_tool
        assert "parameters" in full_tool
        assert "input_schema" not in bare_tool
        assert "cli" not in bare_tool

        compact_size = len(json.dumps(compact.json(), sort_keys=True))
        full_size = len(json.dumps(full.json(), sort_keys=True))
        assert compact_size < full_size

    def test_trade_place_catalog_is_available(self):
        res = self.client.get("/api/v1/tools/trade_place")

        assert res.status_code == 200
        tool = res.json()["tool"]
        assert tool["name"] == "trade_place"
        assert tool["safety"]["requires_confirmation"] is True

    @pytest.mark.parametrize(
        "body",
        [
            {"arguments": [], "confirm": False},
            {"arguments": {}, "confirm": "notabool"},
        ],
    )
    def test_invoke_request_validation_uses_error_envelope(self, body):
        res = self.client.post(
            "/api/v1/tools/tools_list/invoke",
            json=body,
        )

        assert res.status_code == 422
        envelope = res.json()["detail"]
        _assert_error_envelope(
            envelope,
            error_code="web_api_validation_error",
            operation="tools_list",
        )
        assert envelope["details"]["issues"]
        assert res.headers["x-request-id"] == envelope["request_id"]

    def test_catalog_request_validation_uses_error_envelope(self):
        res = self.client.get("/api/v1/tools", params={"category": "tradng"})

        assert res.status_code == 422
        envelope = res.json()["detail"]
        _assert_error_envelope(
            envelope,
            error_code="web_api_validation_error",
            operation="tools_list",
        )
        assert envelope["details"]["issues"][0]["location"][-1] == "category"
        assert res.headers["x-request-id"] == envelope["request_id"]

    def test_get_unknown_tool_uses_error_envelope(self):
        res = self.client.get("/api/v1/tools/not_a_real_tool")
        assert res.status_code == 404
        body = res.json()
        assert body.get("success") is not True
        envelope = body["detail"]
        _assert_error_envelope(
            envelope, error_code="tool_not_found", operation="not_a_real_tool"
        )
        assert res.headers.get("x-request-id") == envelope["request_id"]

    def test_invoke_route(self):
        res = self.client.post(
            "/api/v1/tools/tools_list/invoke",
            json={"arguments": {"limit": 1, "detail": "compact"}, "confirm": False},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        assert len(body["result"]["tools"]) == 1
        assert "count" not in body["result"]

    def test_invoke_trade_live_without_confirm_blocked(self):
        res = self.client.post(
            "/api/v1/tools/trade_place/invoke",
            json={
                "arguments": {**_TRADE_PLACE_ARGS, "dry_run": False},
                "confirm": False,
            },
        )
        assert res.status_code == 400
        body = res.json()
        assert body.get("success") is not True
        envelope = body["detail"]
        _assert_error_envelope(
            envelope,
            error_code="confirmation_required",
            operation="trade_place",
        )
        details = envelope["details"]
        assert details["requires_confirmation"] is True
        assert "safety" in details
        assert "hint" in details
        assert res.headers.get("x-request-id") == envelope["request_id"]

    @pytest.mark.parametrize("dry_run", [True, None])
    def test_invoke_trade_preview_without_confirm(self, dry_run):
        arguments = dict(_TRADE_PLACE_ARGS)
        if dry_run is not None:
            arguments["dry_run"] = dry_run
        with _trade_place_functions():
            res = self.client.post(
                "/api/v1/tools/trade_place/invoke",
                json={"arguments": arguments, "confirm": False},
            )
        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        assert body["result"]["dry_run"] is True
        assert body["result"]["would_send_order"] is False

    def test_invoke_trade_live_with_confirm(self):
        with _trade_place_functions():
            res = self.client.post(
                "/api/v1/tools/trade_place/invoke",
                json={
                    "arguments": {**_TRADE_PLACE_ARGS, "dry_run": False},
                    "confirm": True,
                },
            )
        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        assert body["result"]["dry_run"] is False

    def test_invoke_route_domain_failure_is_not_wrapped_as_success(self):
        payload, status_code = _DOMAIN_FAILURES[0]
        with patch(
            "mtdata.core.web_api_tools.get_tool_functions",
            return_value={"demo": lambda: payload},
        ):
            res = self.client.post(
                "/api/v1/tools/demo/invoke",
                json={"arguments": {}, "confirm": False},
            )
        assert res.status_code == status_code
        body = res.json()
        assert body.get("success") is not True
        envelope = body["detail"]
        _assert_error_envelope(
            envelope,
            error_code=payload["error_code"],
            operation="demo",
        )
        assert envelope["error"] == payload["error"]
        assert res.headers.get("x-request-id") == envelope["request_id"]

    def test_invoke_unknown_tool_uses_error_envelope(self):
        res = self.client.post(
            "/api/v1/tools/not_a_real_tool/invoke",
            json={"arguments": {}, "confirm": False},
        )
        assert res.status_code == 404
        body = res.json()
        assert body.get("success") is not True
        envelope = body["detail"]
        _assert_error_envelope(
            envelope,
            error_code="tool_not_found",
            operation="not_a_real_tool",
        )
        assert "safety" in envelope["details"]
        assert res.headers.get("x-request-id") == envelope["request_id"]

    def test_invoke_omitted_tool_uses_error_envelope(self):
        res = self.client.post(
            "/api/v1/tools/wait_event/invoke",
            json={"arguments": {}, "confirm": False},
        )
        assert res.status_code == 403
        body = res.json()
        assert body.get("success") is not True
        envelope = body["detail"]
        _assert_error_envelope(
            envelope,
            error_code="tool_not_available",
            operation="wait_event",
        )
        assert "wait_event" in envelope["details"]["rationale"]
        assert res.headers.get("x-request-id") == envelope["request_id"]

    def test_invoke_preserves_uint64_ticket_strings(self):
        def _fake_trade_modify(request: TradeModifyRequest):
            return {
                "success": True,
                "dry_run": request.dry_run,
                "ticket": request.ticket,
            }

        tickets = (
            "9007199254740991",
            "9007199254740993",
            "18446744073709551615",
        )
        with (
            patch("mtdata.core.web_api_tools.ensure_tools_bootstrapped"),
            patch(
                "mtdata.core.web_api_tools.get_tool_functions",
                return_value={"trade_modify": _fake_trade_modify},
            ),
        ):
            for ticket in tickets:
                res = self.client.post(
                    "/api/v1/tools/trade_modify/invoke",
                    json={
                        "arguments": {
                            "ticket": ticket,
                            "stop_loss": 1.0,
                            "dry_run": True,
                        },
                        "confirm": False,
                    },
                )
                assert res.status_code == 200, res.text
                assert res.json()["result"]["ticket"] == int(ticket)

    def test_invoke_extras_uses_error_envelope(self):
        with patch(
            "mtdata.core.web_api_tools.get_tool_functions",
            return_value={"demo": lambda: {"success": True}},
        ):
            res = self.client.post(
                "/api/v1/tools/demo/invoke",
                json={"arguments": {"extras": "not-a-real-extra"}, "confirm": False},
            )
        assert res.status_code == 400
        body = res.json()
        assert body.get("success") is not True
        envelope = body["detail"]
        _assert_error_envelope(
            envelope, error_code="tool_param_error", operation="demo"
        )
        assert envelope["details"]["parameter"] == "extras"
        assert envelope["details"]["replacement"] == "detail"
        assert res.headers.get("x-request-id") == envelope["request_id"]
