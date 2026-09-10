from __future__ import annotations

from typing import Any

import pytest

from mtdata.core._mcp_tools import shape_public_tool_output
from mtdata.core.news import news
from mtdata.core.output_contract import build_pagination_meta


def _rows_for(collection_key: str) -> list[Any]:
    if collection_key == "expirations":
        return ["2026-10-16", "2026-10-23"]
    if collection_key == "tasks":
        return [
            {"task_id": "task-1", "status": "running"},
            {"task_id": "task-2", "status": "pending"},
        ]
    if collection_key == "models":
        return [{"model_id": "m/1"}, {"model_id": "m/2"}]
    if collection_key == "methods":
        return [{"method": "m1"}, {"method": "m2"}]
    if collection_key == "tools":
        return [
            {"name": "tool_1", "category": "market"},
            {"name": "tool_2", "category": "market"},
        ]
    if collection_key == "general_news":
        return [{"title": "One"}, {"title": "Two"}]
    if collection_key == "rows":
        return [{"symbol": "EURUSD"}, {"symbol": "GBPUSD"}]
    return [{"symbol": "EURUSD"}, {"symbol": "GBPUSD"}]


@pytest.mark.parametrize(
    ("tool_name", "collection_key"),
    [
        ("tools_list", "tools"),
        ("forecast_list_methods", "methods"),
        ("denoise_list_methods", "methods"),
        ("forecast_models_list", "models"),
        ("forecast_task_list", "tasks"),
        ("symbols_list", "data"),
        ("options_expirations", "expirations"),
        ("market_scan", "data"),
        ("market_radar", "rows"),
        ("news", "general_news"),
    ],
)
def test_offset_tools_share_compact_pagination_contract(
    tool_name: str,
    collection_key: str,
) -> None:
    result = shape_public_tool_output(
        {
            "success": True,
            "count": 2,
            collection_key: _rows_for(collection_key),
            "pagination": {
                "total": 9,
                "returned": 2,
                "offset": 4,
                "limit": 2,
                "has_more": True,
                "more_available": 3,
            },
        },
        tool_name=tool_name,
        detail="compact",
    )

    assert "count" not in result
    assert result["pagination"] == {
        "offset": 4,
        "limit": 2,
        "returned": 2,
        "has_more": True,
        "total": 9,
        "next_offset": 6,
    }


def test_filtered_provider_page_advances_by_consumed_source_rows() -> None:
    result = shape_public_tool_output(
        {
            "success": True,
            "items": [],
            "providers_queried": ["finviz"],
            "pagination": {
                "total": None,
                "returned": 0,
                "offset": 3,
                "limit": 3,
                "has_more": True,
                "more_available": None,
                "scope": "provider_page",
                "provider_total": 100,
                "provider_returned": 3,
            },
        },
        tool_name="news",
        detail="compact",
    )

    assert result["pagination"]["returned"] == 0
    assert result["pagination"]["next_offset"] == 6


@pytest.mark.parametrize(
    ("tool_name", "pagination_context"),
    [
        (
            "data_fetch_candles",
            {"continuation_direction": "forward"},
        ),
        (
            "data_fetch_ticks",
            {"selection": "last_n", "source_events_returned": 2},
        ),
    ],
)
def test_cursor_tools_keep_core_and_continuation_fields(
    tool_name: str,
    pagination_context: dict[str, Any],
) -> None:
    result = shape_public_tool_output(
        {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "M1",
            "count": 2,
            "data": [{"time": "2026-09-10T12:00:00Z"}] * 2,
            "pagination": {
                "total": None,
                "returned": 2,
                "offset": 2,
                "limit": 2,
                "has_more": True,
                "more_available": None,
                "total_lower_bound": 5,
                "next_cursor": "opaque",
                **pagination_context,
            },
        },
        tool_name=tool_name,
        detail="compact",
    )

    assert result["pagination"] == {
        "offset": 2,
        "limit": 2,
        "returned": 2,
        "has_more": True,
        "total": None,
        **pagination_context,
        "total_lower_bound": 5,
        "next_cursor": "opaque",
    }


@pytest.mark.parametrize(
    ("tool_name", "pagination_context"),
    [
        ("data_fetch_candles", {}),
        ("data_fetch_ticks", {"selection": "last_n"}),
    ],
)
def test_complete_cursor_pages_keep_core_without_continuation(
    tool_name: str,
    pagination_context: dict[str, Any],
) -> None:
    result = shape_public_tool_output(
        {
            "success": True,
            "symbol": "EURUSD",
            "data": [{"time": "2026-09-10T12:00:00Z"}],
            "count": 1,
            "pagination": {
                "total": 3,
                "returned": 1,
                "offset": 2,
                "limit": 2,
                "has_more": False,
                "more_available": 0,
                **pagination_context,
            },
        },
        tool_name=tool_name,
        detail="compact",
    )

    assert result["pagination"] == {
        "offset": 2,
        "limit": 2,
        "returned": 1,
        "has_more": False,
        "total": 3,
        **pagination_context,
    }
    assert "count" not in result


@pytest.mark.parametrize(
    ("total", "limit", "offset", "suggested_offset"),
    [
        (1, 20, 1, 0),
        (3, 2, 3, 2),
        (20, 10, 999, 10),
        (4, None, 8, 0),
    ],
)
def test_build_pagination_marks_successful_page_beyond_total(
    total: int,
    limit: int | None,
    offset: int,
    suggested_offset: int,
) -> None:
    pagination = build_pagination_meta(
        total=total,
        returned=0,
        offset=offset,
        limit=limit,
    )

    assert pagination["has_more"] is False
    assert pagination["page_beyond_total"] is True
    assert pagination["suggested_offset"] == suggested_offset


@pytest.mark.parametrize(
    ("tool_name", "collection_key"),
    [
        ("forecast_task_list", "tasks"),
        ("forecast_models_list", "models"),
        ("denoise_list_methods", "methods"),
        ("symbols_list", "data"),
        ("options_expirations", "expirations"),
        ("market_scan", "data"),
        ("market_radar", "rows"),
        ("news", "general_news"),
    ],
)
def test_page_beyond_total_keeps_compact_recovery_guidance(
    tool_name: str,
    collection_key: str,
) -> None:
    result = shape_public_tool_output(
        {
            "success": True,
            "count": 0,
            collection_key: [],
            "message": "Generic empty collection message.",
            "hint": "Generic empty collection hint.",
            "pagination": build_pagination_meta(
                total=5,
                returned=0,
                offset=9,
                limit=2,
            ),
        },
        tool_name=tool_name,
        detail="compact",
    )

    assert result["success"] is True
    assert result["empty"] is True
    assert result["empty_reason"] == "page_beyond_total"
    assert result["pagination"] == {
        "offset": 9,
        "limit": 2,
        "returned": 0,
        "has_more": False,
        "total": 5,
        "page_beyond_total": True,
        "suggested_offset": 4,
    }
    assert "offset=9" in result["message"]
    assert "offset=4" in result["hint"]
    assert "count" not in result


def test_news_page_beyond_total_keeps_pre_pagination_provider_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mtdata.core.news.fetch_unified_news",
        lambda **_kwargs: {
            "success": True,
            "sources_used": ["finviz", "mt5"],
            "source_details": {
                "finviz": {"success": True},
                "mt5": {"success": False, "error": "terminal unavailable"},
            },
            "general_news": [
                {"title": "One", "provider": "finviz"},
                {"title": "Two", "provider": "finviz"},
            ],
        },
    )
    raw = news
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__

    page = raw(limit=1, offset=99)
    result = shape_public_tool_output(page, tool_name="news", detail="compact")

    assert result["providers_queried"] == ["finviz", "mt5"]
    assert result["provider_failures"] == {"mt5": "terminal unavailable"}
    assert result["empty_reason"] == "page_beyond_total"
    assert result["pagination"]["suggested_offset"] == 1
