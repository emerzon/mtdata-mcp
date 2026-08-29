from mtdata.core._mcp_tools import shape_public_tool_output


def test_trade_preview_compact_preserves_every_execution_safety_gate() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "dry_run": True,
        "preview_ok": True,
        "would_send_order": True,
        "order_sent": False,
        "required_confirmation": True,
        "validation_passed": True,
        "trade_gate_passed": True,
        "blockers": [],
        "order_type": "BUY",
        "volume": 0.1,
        "stop_loss": 1.09,
        "take_profit": 1.12,
        "request_echo": {"deviation": 20},
        "data_stale": False,
        "freshness": "fresh",
        "data_age_seconds": 0.2,
        "source": {"provider": "mt5", "server": "Demo"},
        "meta": {"diagnostics": {"latency_ms": 3}},
    }

    result = shape_public_tool_output(
        payload,
        tool_name="trade_place",
        detail="compact",
    )

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "dry_run": True,
        "preview_ok": True,
        "would_send_order": True,
        "order_sent": False,
        "required_confirmation": True,
        "validation_passed": True,
        "trade_gate_passed": True,
        "blockers": [],
        "order_type": "BUY",
        "volume": 0.1,
        "stop_loss": 1.09,
        "take_profit": 1.12,
        "source": {"provider": "mt5"},
    }


def test_open_positions_compact_replaces_per_row_freshness_with_one_warning() -> None:
    payload = {
        "success": True,
        "count": 2,
        "row_key": "items",
        "items": [
            {
                "ticket": 1,
                "symbol": "EURUSD",
                "price_current": 1.1,
                "data_age_seconds": 900,
                "data_stale": True,
                "freshness_state": "stale",
                "usable_for_live_trading": False,
            },
            {
                "ticket": 2,
                "symbol": "USDJPY",
                "price_current": 150.0,
                "data_age_seconds": 0.1,
                "data_stale": False,
                "freshness_state": "live",
                "usable_for_live_trading": True,
            },
        ],
        "quote_freshness_summary": {
            "positions_enriched": 2,
            "stale_quotes": 1,
            "live_usable_quotes": 1,
            "recent_or_delayed_quotes": 0,
        },
    }

    result = shape_public_tool_output(
        payload,
        tool_name="trade_get_open",
        detail="compact",
    )

    assert "count" not in result
    assert "row_key" not in result
    assert all("data_stale" not in row for row in result["items"])
    assert result["warnings"] == [
        {
            "code": "stale_position_quotes",
            "scope": "trade_get_open",
            "message": "Some position quotes are stale and must not drive live execution.",
            "stale_quotes": 1,
            "positions_checked": 2,
        }
    ]


def test_trade_full_moves_sampling_request_and_quote_quality_to_meta() -> None:
    payload = {
        "success": True,
        "items": [],
        "request_echo": {"history_kind": "deals"},
        "sample_provenance": {"items_returned": 50, "items_truncated": True},
        "quote_freshness_summary": {"positions_enriched": 2, "stale_quotes": 0},
        "source": {"provider": "mt5", "server": "Demo"},
    }

    result = shape_public_tool_output(
        payload,
        tool_name="trade_history",
        detail="full",
    )

    assert "request_echo" not in result
    assert "sample_provenance" not in result
    assert "quote_freshness_summary" not in result
    assert result["meta"]["request"]["request_echo"] == {"history_kind": "deals"}
    assert result["meta"]["sampling"]["items_truncated"] is True
    assert result["meta"]["quality"]["quote_freshness"]["stale_quotes"] == 0


def test_compact_report_omits_nominal_runtime_and_section_status() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "detail": "compact",
        "content_detail": "summary_only",
        "section_run_status": "complete",
        "sections_status": {"summary": {"ok": 5, "partial": 0, "error": 0}},
        "summary_structured": {"market": {"close": 1.1}},
        "data_as_of": "2026-08-29T10:00:00Z",
        "as_of": "2026-08-29T10:01:00Z",
        "timezone": "UTC",
        "runtime_plan": {"actual_runtime_seconds": 2.5},
        "source": {"provider": "mt5", "server": "Demo"},
    }

    result = shape_public_tool_output(
        payload,
        tool_name="report_generate",
        detail="compact",
    )

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "summary_structured": {"market": {"close": 1.1}},
        "data_as_of": "2026-08-29T10:00:00Z",
        "source": {"provider": "mt5"},
    }


def test_compact_report_keeps_degraded_section_status_and_warning() -> None:
    payload = {
        "success": True,
        "section_run_status": "partial",
        "sections_status": {"summary": {"ok": 4, "partial": 1, "error": 0}},
        "summary_structured": {"market": {"close": 1.1}},
        "warnings": ["Forecast section timed out."],
    }

    result = shape_public_tool_output(
        payload,
        tool_name="report_generate",
        detail="compact",
    )

    assert result["section_run_status"] == "partial"
    assert result["sections_status"]["summary"]["partial"] == 1
    assert result["warnings"] == [
        {"code": "data_warning", "message": "Forecast section timed out."}
    ]
