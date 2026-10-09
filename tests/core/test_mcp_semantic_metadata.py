"""Exercise semantic metadata through the actual async MCP output wrapper."""

import asyncio
from copy import deepcopy

import pytest

from mtdata.core import _mcp_tools


@pytest.fixture
def wrap_payload(monkeypatch):
    # Use the real transport wrapper without modifying the global tool catalog.
    monkeypatch.setattr(
        _mcp_tools, "_ORIG_TOOL_DECORATOR", lambda *a, **kw: lambda fn: fn
    )
    monkeypatch.setattr(_mcp_tools, "_upsert_tool_registration", lambda *a, **kw: None)

    def wrap(name, payload):
        def tool(detail: str = "compact"):
            return deepcopy(payload)

        tool.__name__ = name
        return _mcp_tools._recording_tool_decorator()(tool)._mcp_async_wrapper

    return wrap


@pytest.mark.parametrize("tool_name", ["forecast_list_methods", "denoise_list_methods"])
@pytest.mark.parametrize("detail", ["compact", "standard"])
@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("offset,total", [(0, 23), (20, 23), (0, 0)])
def test_method_catalog_keeps_page_and_filter_context(
    wrap_payload, tool_name, detail, json_output, offset, total
):
    returned = min(20, total - offset)
    has_more = offset + returned < total
    pagination = {
        "total": total,
        "returned": returned,
        "offset": offset,
        "limit": 20,
        "has_more": has_more,
    }
    if has_more:
        pagination["next_offset"] = offset + returned
    payload = {
        "success": True,
        "methods": [
            {"method": f"method_{idx}", "category": "classical"}
            for idx in range(offset, offset + returned)
        ],
        "pagination": pagination,
        "filters": {"search": "method"},
        "profile": "quickstart",
        "profile_hint": "Use profile=all to list all registered methods.",
    }
    call = wrap_payload(tool_name, payload)
    result = asyncio.run(call(detail=detail, json=json_output))

    if json_output:
        assert result["pagination"] == pagination
        assert len(result["methods"]) == returned
        assert result["filters"] == payload["filters"]
        assert result["profile"] == "quickstart"
    else:
        for key, value in pagination.items():
            expected = str(value).lower() if isinstance(value, bool) else str(value)
            assert f"{key}: {expected}" in result
        if not has_more:
            assert "next_offset" not in result
        assert "search: method" in result
        assert "profile: quickstart" in result
        assert payload["profile_hint"] in result
        for row in payload["methods"]:
            assert row["method"] in result


def test_method_catalog_retains_unavailable_reason(wrap_payload):
    call = wrap_payload(
        "forecast_list_methods",
        {
            "success": True,
            "methods": [
                {"method": "theta", "available": True},
                {
                    "method": "optional_model",
                    "available": False,
                    "unavailable_reason": "Requires optional dependency",
                },
            ],
        },
    )
    result = asyncio.run(call())
    assert "unavailable" in result
    assert "optional_model" in result
    assert "Requires optional dependency" in result


@pytest.mark.parametrize("detail", ["compact", "standard", "full"])
@pytest.mark.parametrize("json_output", [False, True])
def test_projected_trade_idea_keeps_partial_failure_context(
    wrap_payload, detail, json_output
):
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "actionability": "preview_only",
        "partial_failure": True,
        "failed_sections": ["volatility"],
        "section_errors": {"volatility": "No usable volatility observations"},
        "geometry": {"basis": "fallback", "stop_loss": 1.08},
        "unrelated_details": {"value": 123},
    }
    call = wrap_payload("trade_idea_compose", payload)
    result = asyncio.run(
        call(detail=detail, json=json_output, output_fields="geometry")
    )
    if json_output:
        for key in ("partial_failure", "failed_sections", "section_errors", "geometry"):
            assert result[key] == payload[key]
        assert result["success"] is True
        assert "unrelated_details" not in result
    else:
        assert "partial_failure: true" in result
        assert "failed_sections" in result
        assert "section_errors" in result
        assert "volatility" in result
        assert "No usable volatility observations" in result
        assert "basis: fallback" in result
        assert "unrelated_details" not in result
