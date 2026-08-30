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
        "as_of": "2026-08-29T12:00:00Z",
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
    assert result["as_of"] == "2026-08-29T12:00:00Z"
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


def test_trade_account_compact_keeps_distinct_safety_gates_once() -> None:
    payload = {
        "success": True,
        "retrieved_at": "2026-08-29T12:00:00Z",
        "source": {"provider": "mt5", "server": "Demo"},
        "account_context_id": "context-123",
        "account_type": "demo",
        "is_demo": True,
        "is_live": False,
        "balance": 10000.0,
        "equity": 9900.0,
        "profit": -100.0,
        "floating_pnl": -100.0,
        "pnl_basis": "floating_open_positions",
        "equity_balance_delta": -100.0,
        "margin": 500.0,
        "margin_free": 9400.0,
        "margin_level": 1980.0,
        "currency": "USD",
        "leverage": 100,
        "trade_allowed": False,
        "trade_allowed_basis": "strict_execution_and_margin",
        "broker_trade_allowed": True,
        "new_exposure_allowed": False,
        "readiness_scope": "account_and_terminal_not_symbol_session",
        "execution_ready": False,
        "execution_ready_scope": "account_and_terminal_enablement",
        "execution_hard_blockers": ["terminal_auto_trading_disabled"],
        "account_risk_status": "ok",
        "account_risk_reasons": [],
        "symbol_sessions_evaluated": False,
        "now_tradable": False,
        "now_tradable_means": "requires_symbol_session",
    }

    result = shape_public_tool_output(
        payload,
        tool_name="trade_account_info",
        detail="compact",
    )

    assert result["new_exposure_allowed"] is False
    assert result["as_of"] == "2026-08-29T12:00:00Z"
    assert result["execution_ready"] is False
    assert result["execution_hard_blockers"] == [
        "terminal_auto_trading_disabled"
    ]
    assert result["account_risk_status"] == "ok"
    assert "trade_allowed" not in result
    assert "floating_pnl" not in result
    assert "now_tradable" not in result


def test_trade_history_compact_hoists_uniform_basis_and_drops_request_prose() -> None:
    payload = {
        "success": True,
        "kind": "trade_history",
        "history_kind": "deals",
        "scope": "account_history",
        "currency": "USD",
        "period_source": "default_lookback",
        "minutes_back_effective": 10080,
        "broker_server_tz": "America/New_York",
        "broker_utc_offset_seconds": -14400,
        "defaults_applied": ["last_7_days"],
        "note": "Defaulted to the last seven days.",
        "items": [
            {
                "deal_ticket": 101,
                "order_ticket": 55,
                "position_ticket": 55,
                "symbol": "EURUSD",
                "fill_time": "2026-08-29T10:00:00Z",
                "position_action": "close_long",
                "deal_effect": "close",
                "fill_side": "sell",
                "position_side": "long",
                "volume": 0.1,
                "price": 1.1,
                "price_basis": "executed_fill",
                "price_currency": "USD",
                "profit": 25.0,
                "commission": 0.0,
                "swap": 0.0,
                "fee": 0.0,
            }
        ],
        "units": {
            "volume": "broker_lot",
            "profit": "account_currency",
            "price": "absolute_price",
        },
        "pagination": {
            "has_more": True,
            "next_cursor": "opaque-token",
            "returned": 1,
            "total": 20,
        },
    }

    result = shape_public_tool_output(
        payload,
        tool_name="trade_history",
        detail="compact",
    )

    assert result["price_basis"] == "executed_fill"
    assert result["pagination"] == {
        "has_more": True,
        "next_cursor": "opaque-token",
    }
    assert result["units"] == {"volume": "broker_lot"}
    assert result["items"] == [
            {
                "deal_ticket": 101,
                "order_ticket": 55,
                "position_ticket": 55,
            "symbol": "EURUSD",
            "fill_time": "2026-08-29T10:00:00Z",
            "position_action": "close_long",
            "volume": 0.1,
                "price": 1.1,
                "profit": 25.0,
                "commission": 0.0,
                "swap": 0.0,
                "fee": 0.0,
            }
    ]
    assert "note" not in result


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
