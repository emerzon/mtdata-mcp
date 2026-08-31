import json

from mtdata.core._mcp_tools import shape_public_tool_output


def _json_size(payload: dict) -> int:
    return len(json.dumps(payload, separators=(",", ":")))


def test_compact_trade_place_preview_error_keeps_only_actionable_state() -> None:
    warning = {
        "code": "data_warning",
        "message": "Dry run only. No order was sent to MT5.",
    }
    payload = {
        "success": False,
        "error": (
            "stop_loss and take_profit are required when require_sl_tp=true. "
            "Provide both levels or explicitly set require_sl_tp=false."
        ),
        "error_code": "preview_blocked",
        "operation": "trade_place",
        "status": "preview_blocked",
        "symbol": "EURUSD",
        "order_type": "BUY",
        "volume": 0.01,
        "dry_run": True,
        "no_action": True,
        "no_action_reason": "dry_run_validation_blocked",
        "would_send_order": False,
        "preview_ok": False,
        "validation_passed": False,
        "require_sl_tp": True,
        "blockers": ["missing_stop_loss", "missing_take_profit"],
        "remediation": "Provide both protection levels and retry.",
        "warnings": [warning],
        "account_state": {
            "equity": 10_000.0,
            "margin_free": 9_500.0,
            "margin_level": 400.0,
        },
        "candidate_risk": {
            "risk_currency": None,
            "notional_exposure": 1_100.0,
        },
        "validation": {
            "blockers": ["missing_stop_loss", "missing_take_profit"],
            "local_requirements_passed": False,
            "live_submission_eligible": False,
        },
        "validation_scope": "local_preview_plus_estimates",
        "preview_checks_performed": [
            "request_routing",
            "local_safety_requirements",
            "protection_level_preview",
        ],
        "broker_validation_not_performed": [
            "broker_acceptance",
            "broker_price_distance_enforcement",
            "broker_margin_reservation",
            "broker_fillability",
            "broker_sl_tp_attachment",
        ],
        "guardrails_preview": {
            "enabled": True,
            "blocked": False,
            "checks_not_performed": [],
        },
        "requested_comment": "strategy entry",
        "applied_comment": "strategy entry",
        "units": {
            "volume": "broker_lot",
            "margin_required": "account_currency",
        },
        "quote_context": {
            "bid": 1.0999,
            "ask": 1.1001,
            "observed_at": "2026-08-30T12:00:00Z",
            "freshness_state": "fresh",
            "usable_for_live_trading": True,
            "raw_tick": {"flags": 6, "volume_real": 0.0},
        },
    }

    compact = shape_public_tool_output(
        payload,
        tool_name="trade_place",
        detail="compact",
    )
    full = shape_public_tool_output(
        payload,
        tool_name="trade_place",
        detail="full",
    )

    assert compact == {
        "success": False,
        "error": payload["error"],
        "error_code": "preview_blocked",
        "operation": "trade_place",
        "status": "preview_blocked",
        "symbol": "EURUSD",
        "order_type": "BUY",
        "volume": 0.01,
        "dry_run": True,
        "no_action": True,
        "no_action_reason": "dry_run_validation_blocked",
        "would_send_order": False,
        "preview_ok": False,
        "validation_passed": False,
        "require_sl_tp": True,
        "blockers": ["missing_stop_loss", "missing_take_profit"],
        "remediation": "Provide both protection levels and retry.",
        "warnings": [warning],
    }
    assert _json_size(compact) < _json_size(full) * 0.5


def test_compact_trade_place_error_keeps_unsafe_quote_context() -> None:
    compact = shape_public_tool_output(
        {
            "success": False,
            "error": "The quote is stale and cannot be used for live execution.",
            "error_code": "preview_blocked",
            "symbol": "EURUSD",
            "dry_run": True,
            "no_action": True,
            "preview_ok": False,
            "blockers": ["quote_not_live_ready"],
            "quote_context": {
                "bid": 1.0999,
                "ask": 1.1001,
                "observed_at": "2026-08-30T11:00:00Z",
                "freshness_state": "stale",
                "usable_for_live_trading": False,
                "sizing_warning": "Quote is suitable only as a research reference.",
                "raw_tick": {"flags": 6, "volume_real": 0.0},
            },
        },
        tool_name="trade_place",
        detail="compact",
    )

    assert compact["quote_context"] == {
        "bid": 1.0999,
        "ask": 1.1001,
        "observed_at": "2026-08-30T11:00:00Z",
        "freshness_state": "stale",
        "usable_for_live_trading": False,
        "sizing_warning": "Quote is suitable only as a research reference.",
    }


def test_compact_trade_risk_error_deduplicates_candidate_diagnostics() -> None:
    message = "For long trades, stop_loss must be below entry."
    payload = {
        "success": False,
        "error": message,
        "error_code": "invalid_sl_for_direction",
        "candidate_valid": False,
        "candidate_status": "invalid",
        "geometry_valid": False,
        "sizing_eligible": False,
        "portfolio_snapshot_status": "available",
        "scope": {"mode": "symbol", "symbol": "EURUSD", "matched_positions": 0},
        "trade_evaluation": {
            "status": "invalid",
            "symbol": "EURUSD",
            "direction": "long",
            "direction_source": "explicit",
            "entry": 1.10,
            "sl": 1.20,
            "tp": None,
            "error": {
                "code": "invalid_sl_for_direction",
                "reason": message,
                "message": message,
                "field": "stop_loss",
                "entry": 1.10,
                "constraint": "stop_loss < entry",
                "stop_loss": 1.20,
            },
        },
        "position_sizing_error": {
            "code": "invalid_sl_for_direction",
            "reason": message,
            "message": message,
            "field": "stop_loss",
            "entry": 1.10,
            "constraint": "stop_loss < entry",
            "stop_loss": 1.20,
        },
        "account": {
            "account_context_id": "broker-demo:hash",
            "equity": 10_000.0,
            "margin_free": 9_500.0,
        },
        "scoped_risk": {
            "overall_risk_status": "low",
            "positions_count": 0,
            "pending_orders_count": 0,
            "notional_exposure": 0.0,
        },
        "sizing_risk_policy": {
            "strict_risk": True,
            "rounding": "down",
            "minimum_volume_behavior": "block",
        },
        "positions": [],
        "pending_orders": [],
        "quote_context": {
            "bid": 1.0999,
            "ask": 1.1001,
            "freshness_state": "fresh",
            "usable_for_live_trading": True,
            "raw_tick": {"flags": 6},
        },
    }

    compact = shape_public_tool_output(
        payload,
        tool_name="trade_risk_analyze",
        detail="compact",
    )
    full = shape_public_tool_output(
        payload,
        tool_name="trade_risk_analyze",
        detail="full",
    )

    assert compact == {
        "success": False,
        "error": message,
        "error_code": "invalid_sl_for_direction",
        "candidate_valid": False,
        "candidate_status": "invalid",
        "geometry_valid": False,
        "sizing_eligible": False,
        "portfolio_snapshot_status": "available",
        "scope": {"mode": "symbol", "symbol": "EURUSD"},
        "trade_evaluation": {
            "status": "invalid",
            "symbol": "EURUSD",
            "direction": "long",
            "direction_source": "explicit",
            "entry": 1.10,
            "sl": 1.20,
        },
        "position_sizing_error": {
            "code": "invalid_sl_for_direction",
            "field": "stop_loss",
            "entry": 1.10,
            "constraint": "stop_loss < entry",
            "stop_loss": 1.20,
        },
    }
    assert json.dumps(compact).count(message) == 1
    assert _json_size(compact) < _json_size(full) * 0.5


def test_compact_trade_risk_connection_error_keeps_generic_recovery_context() -> None:
    payload = {
        "success": False,
        "error": "Failed to connect to MetaTrader5.",
        "error_code": "mt5_connection_error",
        "operation": "trade_risk_analyze",
        "details": {"terminal": "not_running"},
        "remediation": "Start the MT5 terminal and retry.",
    }

    assert shape_public_tool_output(
        payload,
        tool_name="trade_risk_analyze",
        detail="compact",
    ) == payload
