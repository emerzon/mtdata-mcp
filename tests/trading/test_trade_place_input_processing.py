from __future__ import annotations

import logging
import os
import re
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

# Add src to path to ensure local package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from mtdata.core.trading import trade_modify as _trade_modify_tool
from mtdata.core.trading import trade_place as _trade_place_tool
from mtdata.core.trading.requests import (
    TradeCloseRequest,
    TradeGetOpenRequest,
    TradeGetPendingRequest,
    TradeHistoryRequest,
    TradeModifyRequest,
    TradePlaceRequest,
)
from mtdata.core.trading.validation import (
    MT5_UINT64_MAX,
    _normalize_order_type_input,
    _safe_int_magic,
    _safe_int_ticket,
)


def trade_place(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradePlaceRequest(**kwargs)
    return _trade_place_tool(request=request, __cli_raw=raw_output)


def trade_modify(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradeModifyRequest(**kwargs)
    return _trade_modify_tool(request=request, __cli_raw=raw_output)


def test_trading_order_requests_expose_canonical_detail_field() -> None:
    fields = TradePlaceRequest.model_fields
    base = {"symbol": "EURUSD", "volume": 0.01, "order_type": "BUY"}

    assert "detail" in fields
    assert "preview_detail" not in fields
    assert TradePlaceRequest(**base, detail="full").detail == "full"
    assert TradePlaceRequest(**base, detail="compact").detail == "compact"
    assert TradePlaceRequest(**base, detail="standard").detail == "standard"
    with pytest.raises(ValidationError, match="detail"):
        TradePlaceRequest(**base, detail="summary")
    assert TradeModifyRequest(ticket=100, detail="summary").detail == "summary"
    assert TradeCloseRequest(detail="summary").detail == "summary"


def test_execution_request_dry_run_defaults() -> None:
    # All trade mutators preview by default; pass dry_run=false to execute live.
    assert TradePlaceRequest(
        symbol="EURUSD", volume=0.01, order_type="BUY"
    ).dry_run is True
    assert TradeModifyRequest(ticket=100).dry_run is True
    assert TradeCloseRequest(ticket=100).dry_run is True


def test_trade_close_rejects_numeric_positional_symbol() -> None:
    with pytest.raises(ValidationError, match="--ticket"):
        TradeCloseRequest(symbol="1861825294")


@pytest.mark.parametrize(
    "request_factory",
    [
        lambda: TradePlaceRequest(
            symbol="EURUSD", volume=0.01, order_type="BUY", deviation=-1
        ),
        lambda: TradeCloseRequest(ticket=100, deviation=-1),
        lambda: TradeCloseRequest(ticket=100, volume=0),
        lambda: TradeCloseRequest(ticket=100, volume=-0.1),
        lambda: TradeModifyRequest(ticket="abc"),
        lambda: TradeModifyRequest(ticket=0),
    ],
)
def test_trade_requests_reject_invalid_execution_numerics(request_factory) -> None:
    with pytest.raises(ValidationError):
        request_factory()


def test_mt5_identifiers_preserve_full_uint64_precision() -> None:
    above_json_safe = (1 << 53) + 1

    assert _safe_int_ticket(str(above_json_safe)) == above_json_safe
    assert _safe_int_ticket(str(MT5_UINT64_MAX)) == MT5_UINT64_MAX
    assert _safe_int_ticket(float(above_json_safe)) is None
    assert _safe_int_ticket(str(MT5_UINT64_MAX + 1)) is None
    assert _safe_int_magic(0) == 0

    assert TradeModifyRequest(ticket=str(above_json_safe)).ticket == above_json_safe
    assert TradeCloseRequest(ticket=str(MT5_UINT64_MAX)).ticket == MT5_UINT64_MAX
    assert TradeHistoryRequest(deal_ticket=str(above_json_safe)).deal_ticket == above_json_safe


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: TradePlaceRequest(
            symbol="EURUSD", volume=0.01, order_type="BUY", magic=value
        ),
        lambda value: TradeCloseRequest(magic=value),
        lambda value: TradeHistoryRequest(magic=value),
        lambda value: TradeGetOpenRequest(magic=value),
        lambda value: TradeGetPendingRequest(magic=value),
    ],
)
def test_all_public_magic_fields_enforce_mt5_uint64(factory) -> None:
    assert factory(0).magic == 0
    assert factory(str(MT5_UINT64_MAX)).magic == MT5_UINT64_MAX
    for invalid in (-1, str(MT5_UINT64_MAX + 1), 1.0):
        with pytest.raises(ValidationError):
            factory(invalid)


def test_close_magic_zero_keeps_exact_bulk_scope() -> None:
    from mtdata.core.trading.execution import _close_positions

    gateway = MagicMock()
    gateway.ensure_connection.return_value = None
    gateway.POSITION_TYPE_BUY = 0
    gateway.ORDER_TYPE_BUY = 0
    gateway.positions_get.return_value = [
        SimpleNamespace(
            ticket=101,
            symbol="EURUSD",
            type=0,
            volume=0.1,
            profit=1.0,
            magic=0,
        ),
        SimpleNamespace(
            ticket=202,
            symbol="EURUSD",
            type=0,
            volume=0.1,
            profit=1.0,
            magic=42,
        ),
    ]
    gateway.symbol_info_tick.return_value = None

    result = _close_positions(magic=0, dry_run=True, gateway=gateway)

    assert result["matched_count"] == 1
    assert result["matched_positions"][0]["ticket"] == 101
    assert result["filters_applied"] == {"magic": 0}


def test_normalize_order_type_rejects_mt5_integer() -> None:
    normalized, error = _normalize_order_type_input(2)
    assert normalized is None
    assert "canonical string" in error


def test_normalize_order_type_rejects_prefixed_symbolic_name() -> None:
    normalized, error = _normalize_order_type_input("ORDER_TYPE_BUY_STOP")
    assert normalized is None
    assert "Unsupported" in error


def test_normalize_order_type_rejects_numeric_string() -> None:
    normalized, error = _normalize_order_type_input("4")
    assert normalized is None
    assert "canonical string" in error


def test_trade_place_rejects_numeric_order_type() -> None:
    with pytest.raises(ValidationError):
        TradePlaceRequest(symbol="BTCUSD", volume=0.03, order_type=2, price=68750)


def test_trade_place_rejects_prefixed_order_type() -> None:
    with pytest.raises(ValidationError, match="order_type"):
        TradePlaceRequest(
            symbol="BTCUSD",
            volume=0.03,
            order_type="ORDER_TYPE_BUY_STOP",
            price=70650,
        )


def test_trade_place_routes_prefixed_market_order_type() -> None:
    with patch("mtdata.core.trading._place_market_order", return_value={"ok": True}) as mock_market:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            require_sl_tp=False,
            dry_run=False,
            __cli_raw=True,
        )
        assert out["ok"] is True
        assert out["success"] is True
        assert re.fullmatch(r"[0-9a-f]{12}", out["correlation_id"])
        assert mock_market.call_args.kwargs["order_type"] == "BUY"


def test_trade_place_rejects_unknown_numeric_order_type() -> None:
    with pytest.raises(ValidationError):
        TradePlaceRequest(symbol="BTCUSD", volume=0.03, order_type=99, price=68750)


def test_trade_place_logs_finish_event(caplog) -> None:
    with patch("mtdata.core.trading._place_market_order", return_value={"success": True}), caplog.at_level(logging.DEBUG,
        logger="mtdata.core.trading",
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            require_sl_tp=False,
            dry_run=False,
            __cli_raw=True,
        )

    assert out["success"] is True
    assert any(
        "event=finish operation=trade_place success=True" in record.message
        for record in caplog.records
    )


def test_trade_place_missing_required_fields_returns_friendly_error() -> None:
    with pytest.raises(ValidationError, match="Field required"):
        TradePlaceRequest()


def test_trade_place_blank_expiration_keeps_market_routing() -> None:
    with patch("mtdata.core.trading._place_market_order", return_value={"ok": True}) as mock_market, patch(
        "mtdata.core.trading._place_pending_order", return_value={"pending": True}
    ) as mock_pending:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            expiration="",
            require_sl_tp=False,
            dry_run=False,
            __cli_raw=True,
        )
        assert out["ok"] is True
        assert out["success"] is True
        assert re.fullmatch(r"[0-9a-f]{12}", out["correlation_id"])
        mock_market.assert_called_once()
        mock_pending.assert_not_called()


def test_trade_place_require_sl_tp_needs_inputs_before_market_send() -> None:
    with patch(
        "mtdata.core.trading.validation._prevalidate_trade_place_market_input",
        return_value=None,
    ) as mock_prevalidate, patch("mtdata.core.trading._place_market_order") as mock_market:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            require_sl_tp=True,
            dry_run=False,
            __cli_raw=True,
        )
    assert "error" in out
    assert out.get("require_sl_tp") is True
    assert set(out.get("missing", [])) == {"stop_loss", "take_profit"}
    assert "trade_risk_analyze" in out.get("hint", "")
    assert out.get("related_tools") == [
        "trade_risk_analyze",
        "forecast_barrier_optimize",
    ]
    mock_prevalidate.assert_called_once_with("BTCUSD", 0.03)
    mock_market.assert_not_called()


def test_trade_place_reports_symbol_error_before_sl_tp_requirement() -> None:
    with patch(
        "mtdata.core.trading.validation._prevalidate_trade_place_market_input",
        return_value={"error": "Symbol FAKESYM not found"},
    ) as mock_prevalidate, patch("mtdata.core.trading._place_market_order") as mock_market:
        out = trade_place(
            symbol="FAKESYM",
            volume=0.03,
            order_type="BUY",
            require_sl_tp=True,
            dry_run=False,
            __cli_raw=True,
        )
    assert out.get("error") == "Symbol FAKESYM not found"
    mock_prevalidate.assert_called_once_with("FAKESYM", 0.03)
    mock_market.assert_not_called()


def test_trade_place_require_sl_tp_false_allows_market_without_sl_tp() -> None:
    with patch("mtdata.core.trading._place_market_order", return_value={"retcode": 10009}) as mock_market:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            require_sl_tp=False,
            dry_run=False,
            __cli_raw=True,
        )
    assert out.get("retcode") == 10009
    mock_market.assert_called_once()


def test_trade_place_dry_run_market_preview_skips_order_send() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={"bid": 64999.0, "ask": 65001.0, "estimated_fill_price": 65001.0},
    ) as mock_preview:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            dry_run=True,
            __cli_raw=True,
        )

    assert out.get("success") is True
    assert out.get("dry_run") is True
    assert out.get("pending") is False
    assert out.get("action") == "place_market_order"
    assert "preview_scope_summary" not in out
    assert "validation_not_performed" not in out
    assert out["warnings"][0].startswith("Dry run only.")
    assert out["validation_scope"] == "local_preview_plus_estimates"
    assert out["preview_ok"] is True
    assert out["preview_checks_performed"] == [
        "request_routing",
        "local_safety_requirements",
        "protection_level_preview",
    ]
    assert out["checks_not_performed"] == ["margin_estimate"]
    assert "broker_acceptance" in out["broker_validation_not_performed"]
    assert "trade_gate_passed" not in out
    assert out.get("message") == "Dry run only. No order was sent to MT5."
    assert out.get("bid") == 64999.0
    assert out.get("ask") == 65001.0
    assert out.get("estimated_fill_price") == 65001.0
    assert out["source"]["provider"] == "mt5"
    mock_preview.assert_called_once()
    mock_market.assert_not_called()


def test_trade_place_dry_run_market_preview_rejects_missing_sl_tp() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={"bid": 64999.0, "ask": 65001.0, "estimated_fill_price": 65001.0},
    ) as mock_preview:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            dry_run=True,
            __cli_raw=True,
        )

    assert out.get("success") is False
    assert out["error_code"] == "preview_blocked"
    assert "stop_loss and take_profit are required" in out["error"]
    assert out.get("blockers") == ["missing_stop_loss", "missing_take_profit"]
    assert out.get("no_action_reason") == "dry_run_validation_blocked"
    assert out.get("dry_run") is True
    assert out.get("require_sl_tp") is True
    assert "live submission with require_sl_tp=true would be rejected" in out.get(
        "dry_run_note", ""
    )
    assert out["validation"]["live_submission_eligible"] is False
    assert out["preview_ok"] is False
    assert out["validation_passed"] is False
    assert out["would_send_order"] is False
    assert out["no_action"] is True
    assert out["status"] == "preview_blocked"
    assert out.get("action") == "place_market_order"
    assert "requested_sl" not in out
    assert "requested_tp" not in out
    mock_preview.assert_called_once()
    assert mock_preview.call_args.kwargs["stop_loss"] is None
    assert mock_preview.call_args.kwargs["take_profit"] is None
    mock_market.assert_not_called()


def test_trade_place_dry_run_preview_detail_keeps_safety_lists() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={"bid": 64999.0, "ask": 65001.0, "estimated_fill_price": 65001.0},
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            dry_run=True,
            detail="compact",
            __cli_raw=True,
        )

    assert out.get("success") is True
    assert out.get("dry_run") is True
    assert "preview_scope_summary" not in out
    assert out["warnings"][0].startswith("Dry run only.")
    assert out["checks_not_performed"] == ["margin_estimate"]
    assert "broker_acceptance" in out["broker_validation_not_performed"]
    assert "validation_not_performed" not in out
    assert out["guardrails_preview"] == {
        "enabled": False,
        "blocked": False,
        "checks_not_performed": [],
    }
    assert out["guardrails_enabled"] is False
    assert out["validation_scope"] == "local_preview_plus_estimates"
    assert out["preview_ok"] is True
    assert "trade_gate_passed" not in out
    mock_market.assert_not_called()


def test_trade_place_dry_run_standard_detail_keeps_validation_context() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={"bid": 64999.0, "ask": 65001.0, "estimated_fill_price": 65001.0},
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            dry_run=True,
            detail="standard",
            __cli_raw=True,
        )

    assert out.get("success") is True
    assert out.get("dry_run") is True
    assert out.get("actionability") == "preview_only"
    assert "preview_scope_summary" in out
    assert "warnings" in out
    assert "guardrails_preview" in out
    assert out["guardrails_enabled"] is False
    assert out["validation_scope"] == "local_preview_plus_estimates"
    assert "trade_gate_passed" not in out
    mock_market.assert_not_called()


def test_trade_place_dry_run_preview_error_uses_standard_error_shape() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={"preview_error": "Failed to get current price for BTCUSD"},
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            dry_run=True,
            __cli_raw=True,
        )

    assert out.get("success") is False
    assert out.get("preview_ok") is False
    assert out.get("error") == "Failed to get current price for BTCUSD"
    assert out.get("error_code") == "trade_preview_error"
    assert out.get("operation") == "trade_place"
    assert out.get("preview_error") == out.get("error")
    assert out.get("no_action_reason") == "dry_run_preview_error"
    mock_market.assert_not_called()


def test_trade_place_unknown_symbol_replaces_protection_blockers() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={
            "preview_error": "Symbol NOTAREALXYZ not found",
            "preview_error_code": "symbol_not_found",
            "remediation": "Use symbols_list to find the broker's exact symbol name.",
            "related_tools": ["symbols_list"],
        },
    ):
        out = trade_place(
            symbol="NOTAREALXYZ",
            volume=0.01,
            order_type="BUY",
            dry_run=True,
            __cli_raw=True,
        )

    assert out["success"] is False
    assert out["error_code"] == "symbol_not_found"
    assert out["blockers"] == ["symbol_not_found"]
    assert "dry_run_note" not in out
    assert out["related_tools"] == ["symbols_list"]
    mock_market.assert_not_called()


def test_trade_place_dry_run_rejects_invalid_live_protection_preview() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={
            "bid": 65000.0,
            "ask": 65002.0,
            "estimated_fill_price": 65002.0,
            "sl_tp_valid": False,
            "sl_tp_error": "stop_loss must be below the live bid for BUY orders. sl=65100.0",
        },
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=65100,
            take_profit=68000,
            dry_run=True,
            __cli_raw=True,
        )

    assert out.get("success") is False
    assert out.get("error_code") == "preview_blocked"
    assert out.get("preview_ok") is False
    assert out.get("dry_run") is True
    assert out.get("validation_code") == "invalid_protection_levels"
    assert out.get("validation_error") == out.get("sl_tp_error")
    assert "stop_loss must be below the live bid" in out.get("validation_error", "")
    assert out.get("no_action_reason") == "dry_run_validation_blocked"
    assert out.get("blockers") == ["invalid_protection_levels"]
    assert "Local protection validation failed" in out["warnings"][0]
    assert "checks passed" not in out["warnings"][0]
    mock_market.assert_not_called()


def test_trade_modify_rejects_request_without_modification_fields() -> None:
    out = trade_modify(ticket=100, __cli_raw=True)

    assert out["success"] is False
    assert out["error_code"] == "no_modification_fields"
    assert "at least one field" in out["error"]
    assert out["ticket"] == 100


def test_trade_place_dry_run_orders_account_quote_and_protection_blockers() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={
            "account_blockers": ["no_free_margin", "critical_margin_stress"],
            "account_state": {
                "margin_free": -1.0,
                "margin_stress": {"status": "critical"},
            },
            "quote_context": {
                "usable_for_live_trading": False,
                "warning": "Quote is stale.",
            },
            "sl_tp_valid": False,
            "sl_tp_error": "Protection levels are invalid.",
            "margin_required": 20.0,
            "margin_free": -1.0,
            "margin_sufficient": False,
        },
    ):
        out = trade_place(
            symbol="EURUSD",
            volume=0.01,
            order_type="BUY",
            stop_loss=1.08,
            take_profit=1.12,
            dry_run=True,
            __cli_raw=True,
        )

    assert out["blockers"] == [
        "no_free_margin",
        "critical_margin_stress",
        "margin_insufficient",
        "quote_not_live_ready",
        "invalid_protection_levels",
    ]
    assert out["success"] is False
    assert out["error_code"] == "preview_blocked"
    assert out["preview_ok"] is False
    assert out["account_state"]["margin_stress"]["status"] == "critical"
    mock_market.assert_not_called()


def test_trade_place_dry_run_blocks_insufficient_estimated_margin() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={
            "quote_context": {"usable_for_live_trading": True},
            "sl_tp_valid": True,
            "margin_required": 200.0,
            "margin_free": 100.0,
            "margin_sufficient": False,
        },
    ):
        out = trade_place(
            symbol="EURUSD",
            volume=0.01,
            order_type="BUY",
            stop_loss=1.08,
            take_profit=1.12,
            dry_run=True,
            detail="standard",
            __cli_raw=True,
        )

    assert out["success"] is False
    assert out["error_code"] == "preview_blocked"
    assert out["preview_ok"] is False
    assert out["validation_passed"] is False
    assert out["blockers"] == ["margin_insufficient"]
    assert out["actionability"] == "blocked_by_margin_estimate"
    assert out["no_action_reason"] == "dry_run_validation_blocked"
    mock_market.assert_not_called()


def test_trade_place_dry_run_rejects_identical_protection_before_mt5() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview"
    ) as mock_preview:
        out = trade_place(
            symbol="EURUSD",
            volume=0.01,
            order_type="BUY",
            stop_loss=1.0,
            take_profit=1.0,
            dry_run=True,
            detail="standard",
            __cli_raw=True,
        )

    assert out["success"] is False
    assert out["error_code"] == "preview_blocked"
    assert out["validation_code"] == "invalid_protection_levels"
    assert out["validation_error"] == "stop_loss and take_profit must be different prices."
    assert out["validation"]["local_requirements_passed"] is False
    assert out["validation"]["live_submission_eligible"] is False
    assert "invalid_protection_levels" in out["validation"]["blockers"]
    assert out["no_action_reason"] == "dry_run_validation_blocked"
    mock_preview.assert_not_called()
    mock_market.assert_not_called()


@pytest.mark.parametrize(
    ("order_type", "stop_loss", "take_profit", "expected_error"),
    [
        ("BUY", 1.1, 1.0, "take_profit must be above stop_loss for BUY orders."),
        ("SELL", 1.0, 1.1, "stop_loss must be above take_profit for SELL orders."),
    ],
)
def test_trade_place_dry_run_rejects_reversed_protection_before_mt5(
    order_type: str,
    stop_loss: float,
    take_profit: float,
    expected_error: str,
) -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview"
    ) as mock_preview:
        out = trade_place(
            symbol="EURUSD",
            volume=0.01,
            order_type=order_type,
            stop_loss=stop_loss,
            take_profit=take_profit,
            dry_run=True,
            detail="standard",
            __cli_raw=True,
        )

    assert out["success"] is False
    assert out["error_code"] == "preview_blocked"
    assert out["validation_code"] == "invalid_protection_levels"
    assert out["validation_error"] == expected_error
    assert out["validation"]["local_requirements_passed"] is False
    assert out["no_action_reason"] == "dry_run_validation_blocked"
    mock_preview.assert_not_called()
    mock_market.assert_not_called()


def test_trade_place_dry_run_preserves_mt5_connection_error() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={
            "preview_error": "Failed to connect to MetaTrader5.",
            "preview_error_code": "mt5_connection_error",
        },
    ):
        out = trade_place(
            symbol="EURUSD",
            volume=0.01,
            order_type="BUY",
            stop_loss=1.08,
            take_profit=1.12,
            dry_run=True,
            detail="standard",
            __cli_raw=True,
        )

    assert out["success"] is False
    assert out["error_code"] == "mt5_connection_error"
    assert out["validation"]["local_requirements_passed"] is True
    assert out["validation"]["live_submission_eligible"] is False
    assert "mt5_connection_error" in out["validation"]["blockers"]
    mock_market.assert_not_called()


def test_trade_place_dry_run_rejects_bool_like_invalid_protection_preview() -> None:
    class BoolLikeFalse:
        def __bool__(self) -> bool:
            return False

    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={
            "sl_tp_valid": BoolLikeFalse(),
            "sl_tp_error": "take_profit must be above the live ask for BUY orders.",
        },
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            dry_run=True,
            __cli_raw=True,
        )

    assert out.get("success") is False
    assert out.get("error_code") == "preview_blocked"
    assert out.get("preview_ok") is False
    assert out.get("dry_run") is True
    assert out.get("validation_code") == "invalid_protection_levels"
    assert "take_profit must be above the live ask" in out.get(
        "validation_error", ""
    )
    assert out.get("blockers") == ["invalid_protection_levels"]
    mock_market.assert_not_called()


def test_trade_place_dry_run_pending_preview_skips_order_send() -> None:
    with patch("mtdata.core.trading._place_pending_order") as mock_pending, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={"bid": 64999.0, "ask": 65001.0, "entry_price": 64500.0},
    ), patch(
        "mtdata.core.trading.time._normalize_pending_expiration",
        return_value=(1787270399, True),
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY_LIMIT",
            price=64500,
            stop_loss=64000,
            take_profit=65500,
            expiration="2026-08-20",
            dry_run=True,
            __cli_raw=True,
        )

    assert out.get("success") is True
    assert out.get("dry_run") is True
    assert out.get("pending") is True
    assert out.get("action") == "place_pending_order"
    assert out["preview_ok"] is True
    assert "preview_scope_summary" not in out
    assert out["warnings"][0].startswith("Dry run only.")
    assert out["checks_not_performed"] == ["margin_estimate"]
    assert "broker_acceptance" in out["broker_validation_not_performed"]
    assert "trade_gate_passed" not in out
    assert out.get("requested_price") == 64500
    assert out.get("expiration") == "2026-08-20"
    assert out.get("expiration_normalized") == 1787270399
    assert out.get("expiration_resolved_utc") == "2026-08-20T23:59:59Z"
    assert out.get("expiration_policy") == "expires_at"
    assert out.get("expiration_explicit") is True
    mock_pending.assert_not_called()


def test_trade_place_pending_preview_blocks_live_use_of_unusable_quote() -> None:
    with patch("mtdata.core.trading._place_pending_order") as mock_pending, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={
            "bid": 1.1000,
            "ask": 1.1002,
            "entry_price": 1.0950,
            "quote_context": {
                "usable_for_live_trading": False,
                "freshness_state": "live",
                "warning": "Quote sources conflict.",
            },
        },
    ), patch(
        "mtdata.core.trading.time._normalize_pending_expiration",
        return_value=(1787270399, True),
    ):
        out = trade_place(
            symbol="EURUSD",
            volume=0.01,
            order_type="BUY_LIMIT",
            price=1.0950,
            stop_loss=1.0900,
            take_profit=1.1050,
            expiration="2026-08-20",
            dry_run=True,
            __cli_raw=True,
        )

    assert out["success"] is False
    assert out["error_code"] == "preview_blocked"
    assert out["preview_ok"] is False
    assert out["validation_passed"] is False
    assert out["validation"]["local_requirements_passed"] is True
    assert out["validation"]["live_submission_eligible"] is False
    assert out["validation"]["staging_valid"] is True
    assert out["staging_valid"] is True
    assert out["blockers"] == ["quote_not_live_ready"]
    assert "not usable for live submission" in out["error"]
    mock_pending.assert_not_called()


def test_trade_place_unusable_quote_does_not_make_invalid_pending_stageable() -> None:
    with patch("mtdata.core.trading._place_pending_order") as mock_pending, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={
            "quote_context": {"usable_for_live_trading": False},
            "sl_tp_valid": False,
            "sl_tp_error": "Pending protection levels are invalid.",
        },
    ), patch(
        "mtdata.core.trading.time._normalize_pending_expiration",
        return_value=(1787270399, True),
    ):
        out = trade_place(
            symbol="EURUSD",
            volume=0.01,
            order_type="BUY_LIMIT",
            price=1.0950,
            stop_loss=1.0900,
            take_profit=1.1050,
            expiration="2026-08-20",
            dry_run=True,
            __cli_raw=True,
        )

    assert out["success"] is False
    assert out["preview_ok"] is False
    assert out["validation"]["local_requirements_passed"] is False
    assert out["validation"]["live_submission_eligible"] is False
    assert out["validation"]["staging_valid"] is False
    assert out["staging_valid"] is False
    assert out["blockers"] == [
        "quote_not_live_ready",
        "invalid_protection_levels",
    ]
    mock_pending.assert_not_called()


def test_trade_place_stop_limit_preview_exposes_both_prices() -> None:
    with patch("mtdata.core.trading._place_pending_order") as mock_pending, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={
            "bid": 69990.0,
            "ask": 70000.0,
            "entry_price": 70050.0,
            "trigger_price": 70100.0,
            "stop_limit_price": 70050.0,
        },
    ) as mock_preview:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY_STOP_LIMIT",
            price=70100,
            stop_limit_price=70050,
            stop_loss=69000,
            take_profit=72000,
            dry_run=True,
            __cli_raw=True,
        )

    assert out["success"] is True
    assert out["trigger_price"] == 70100.0
    assert out["stop_limit_price"] == 70050.0
    assert out["requested_price"] == 70100
    assert out["requested_stop_limit_price"] == 70050
    assert mock_preview.call_args.kwargs["stop_limit_price"] == 70050
    mock_pending.assert_not_called()


@pytest.mark.parametrize(
    "price",
    [0, -1, float("nan"), float("inf"), float("-inf")],
)
def test_trade_place_pending_preview_fails_closed_for_invalid_price(price) -> None:
    with patch("mtdata.core.trading._place_pending_order") as mock_pending, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={
            "preview_error": (
                "price must be a strictly positive finite number after symbol normalization."
            ),
            "preview_error_code": "invalid_pending_price",
        },
    ):
        out = trade_place(
            symbol="EURUSD",
            volume=0.01,
            order_type="BUY_LIMIT",
            price=price,
            stop_loss=1.08,
            take_profit=1.12,
            dry_run=True,
            __cli_raw=True,
        )

    assert out["success"] is False
    assert out["error_code"] == "invalid_pending_price"
    assert out["preview_ok"] is False
    assert out["validation_passed"] is False
    assert out["validation"]["local_requirements_passed"] is False
    assert out["validation"]["live_submission_eligible"] is False
    assert "invalid_pending_price" in out["validation"]["blockers"]
    assert out["blockers"] == ["invalid_pending_price"]
    mock_pending.assert_not_called()


@pytest.mark.parametrize("order_type", ["BUY_STOP_LIMIT", "SELL_STOP_LIMIT"])
def test_trade_place_stop_limit_requires_second_price(order_type) -> None:
    out = trade_place(
        symbol="EURUSD",
        volume=0.01,
        order_type=order_type,
        price=1.101 if order_type.startswith("BUY") else 1.099,
        stop_loss=1.08 if order_type.startswith("BUY") else 1.12,
        take_profit=1.12 if order_type.startswith("BUY") else 1.08,
        dry_run=True,
        __cli_raw=True,
    )

    assert out["success"] is False
    assert out["error_code"] == "invalid_stop_limit_price"
    assert "stop_limit_price is required" in out["error"]


@pytest.mark.parametrize(
    "expiration",
    [0, -1, float("nan"), float("inf"), float("-inf"), "0", "-1", "nan", "inf", "-inf"],
)
def test_trade_place_pending_preview_rejects_invalid_expiration(
    expiration,
) -> None:
    with patch("mtdata.core.trading._place_pending_order") as mock_pending, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview"
    ) as mock_preview:
        out = trade_place(
            symbol="EURUSD",
            volume=0.01,
            order_type="BUY_LIMIT",
            price=1.09,
            stop_loss=1.08,
            take_profit=1.12,
            expiration=expiration,
            dry_run=True,
            __cli_raw=True,
        )

    assert out["success"] is False
    assert out["error_code"] == "invalid_pending_expiration"
    assert out["preview_ok"] is False
    assert out["validation_passed"] is False
    assert out["validation"]["live_submission_eligible"] is False
    assert out["blockers"] == ["invalid_pending_expiration"]
    mock_preview.assert_not_called()
    mock_pending.assert_not_called()


def test_trade_place_pending_preview_rejects_past_expiration_with_context() -> None:
    with patch("mtdata.core.trading._place_pending_order") as mock_pending:
        out = trade_place(
            symbol="EURUSD",
            volume=0.01,
            order_type="BUY_LIMIT",
            price=1.09,
            stop_loss=1.08,
            take_profit=1.12,
            expiration="2020-01-01T00:00:00+00:00",
            dry_run=True,
            __cli_raw=True,
        )

    assert out["error_code"] == "invalid_pending_expiration"
    assert out["expiration_context"]["reason"] == "not_in_future"
    assert out["expiration_context"]["expiration_resolved_utc"] == (
        "2020-01-01T00:00:00Z"
    )
    mock_pending.assert_not_called()


def test_trade_place_rejects_market_order_with_price() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading._place_pending_order"
    ) as mock_pending:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            price=64500,
            stop_loss=64000,
            take_profit=68000,
            __cli_raw=True,
        )

    assert "Conflicting arguments" in out["error"]
    assert "order_type=BUY is a market order" in out["error"]
    assert "BUY_LIMIT/BUY_STOP" in out["error"]
    assert out.get("pending") is None
    assert out.get("order_type") == "BUY"
    assert out.get("price") == 64500
    mock_market.assert_not_called()
    mock_pending.assert_not_called()


def test_trade_place_require_sl_tp_flags_unprotected_market_fill() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "sl_tp_result": {"status": "failed", "requested": {"sl": 64000.0, "tp": 68000.0}},
        },
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            require_sl_tp=True,
            dry_run=False,
            __cli_raw=True,
        )
    assert "error" in out
    assert out.get("require_sl_tp") is True
    assert out.get("protection_status") == "unprotected_position"
    assert any("CRITICAL" in str(w) for w in out.get("warnings", []))


def test_trade_place_preserves_scalar_warning_on_unprotected_market_fill() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "warnings": "broker warning",
            "sl_tp_result": {"status": "failed", "requested": {"sl": 64000.0, "tp": 68000.0}},
        },
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            require_sl_tp=True,
            dry_run=False,
            __cli_raw=True,
        )
    assert "broker warning" in out.get("warnings", [])
    assert "b" not in out.get("warnings", [])
    assert any("CRITICAL" in str(w) for w in out.get("warnings", []))


def test_trade_place_auto_closes_unverified_market_fill() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "warnings": ["verify protection"],
            "sl_tp_result": {"status": "unverified", "requested": {"sl": 64000.0, "tp": 68000.0}},
            "protection_status": "protection_unverified",
            "position_ticket_candidates": [456],
        },
    ), patch(
        "mtdata.core.trading._close_positions",
        return_value={"success": True, "closed_count": 1},
    ) as mock_close:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            require_sl_tp=True,
            dry_run=False,
            __cli_raw=True,
        )
    mock_close.assert_called_once_with(
        ticket=456,
        volume=0.03,
        comment="AUTO-CLOSE: TP/SL protection unresolved",
        deviation=20,
        require_live_quote=False,
    )
    assert out.get("error") == "Order was executed, but TP/SL protection could not be verified."
    assert out.get("error_code") == "protection_not_verified"
    assert out.get("protection_status") == "auto_closed_after_sl_tp_fail"
    assert "verify protection" in out.get("warnings", [])
    assert any("could not be verified" in warning.lower() for warning in out.get("warnings", [])), out


def test_trade_place_treats_unknown_protection_status_as_unverified() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "sl_tp_result": {
                "status": "unexpected",
                "requested": {"sl": 64000.0, "tp": 68000.0},
            },
            "position_ticket": 456,
        },
    ), patch(
        "mtdata.core.trading._close_positions",
        return_value={"success": True, "closed_count": 1},
    ) as mock_close:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            require_sl_tp=False,
            dry_run=False,
            __cli_raw=True,
        )

    mock_close.assert_called_once()
    assert out["success"] is False
    assert out["protection_status"] == "auto_closed_after_sl_tp_fail"
    assert out["error"] == (
        "Order was executed, but TP/SL protection could not be verified."
    )


def test_trade_place_does_not_treat_auto_close_not_found_as_closed() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "sl_tp_result": {"status": "failed", "requested": {"sl": 64000.0}},
            "position_ticket": 456,
        },
    ), patch(
        "mtdata.core.trading._close_positions",
        return_value={"error": "Position 456 not found"},
    ):
        out = trade_place(
            symbol="EURUSD",
            volume=1.0,
            order_type="BUY",
            stop_loss=1.0,
            take_profit=1.2,
            dry_run=False,
            __cli_raw=True,
        )

    assert out.get("protection_status") != "auto_closed_after_sl_tp_fail"
    assert out.get("auto_close_result", {}).get("already_closed") is not True
    assert any("AUTO-CLOSE FAILED" in warning for warning in out.get("warnings", [])), out


def test_trade_place_defaults_to_auto_closing_unprotected_market_fill() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "sl_tp_result": {"status": "failed", "requested": {"sl": 64000.0, "tp": 68000.0}},
            "position_ticket": 456,
        },
    ), patch(
        "mtdata.core.trading._close_positions",
        return_value={"success": True, "closed_count": 1},
    ) as mock_close:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            dry_run=False,
            __cli_raw=True,
        )
    mock_close.assert_called_once()
    assert "error" in out
    assert out.get("require_sl_tp") is True
    assert out.get("auto_close_on_sl_tp_fail") is True
    assert out.get("protection_status") == "auto_closed_after_sl_tp_fail"
    assert "TP/SL protection could not be applied" in str(out.get("error"))


def test_trade_place_auto_close_attempts_recovery_on_sl_tp_fail() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "sl_tp_result": {"status": "failed", "requested": {"sl": 64000.0, "tp": 68000.0}},
            "position_ticket": 789,
        },
    ), patch(
        "mtdata.core.trading._close_positions",
        return_value={"retcode": 10009, "ticket": 789},
    ) as mock_close:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            dry_run=False,
            __cli_raw=True,
        )
    mock_close.assert_called_once()
    assert out.get("auto_close_on_sl_tp_fail") is True
    assert out.get("protection_status") == "auto_closed_after_sl_tp_fail"
    assert out.get("auto_close_result", {}).get("retcode") == 10009


def test_trade_place_preserves_atomic_protection_status_without_fallback_fields() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "sl_tp_result": {
                "status": "applied",
                "requested": {"sl": 64000.0, "tp": 68000.0},
            },
            "protection_status": "protected",
        },
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            dry_run=False,
            __cli_raw=True,
        )
    assert out.get("protection_status") == "protected"
    assert "fallback_used" not in out.get("sl_tp_result", {})
    assert not any("fallback" in str(w).lower() for w in out.get("warnings", []))


def test_trade_modify_blank_expiration_keeps_position_path() -> None:
    with patch("mtdata.core.trading._modify_position", return_value={"success": True}) as mock_pos, patch(
        "mtdata.core.trading._modify_pending_order", return_value={"success": True}
    ) as mock_pending:
        out = trade_modify(ticket=123, stop_loss=1.0, expiration="", __cli_raw=True)
        assert out.get("success") is True
        mock_pos.assert_called_once()
        mock_pending.assert_not_called()


@pytest.mark.parametrize("expiration", [0, -1, "0", "-1", "2020-01-01"])
def test_trade_modify_rejects_invalid_expiration_before_order_lookup(
    expiration,
) -> None:
    with patch("mtdata.core.trading._modify_pending_order") as mock_pending, patch(
        "mtdata.core.trading._modify_position"
    ) as mock_position:
        out = trade_modify(
            ticket=123,
            expiration=expiration,
            dry_run=True,
            __cli_raw=True,
        )

    assert out["success"] is False
    assert out["error_code"] == "invalid_pending_expiration"
    assert out["preview_ok"] is False
    assert out["validation"]["live_submission_eligible"] is False
    mock_pending.assert_not_called()
    mock_position.assert_not_called()


def test_trade_modify_pending_not_found_reports_checked_scope() -> None:
    with patch(
        "mtdata.core.trading._modify_pending_order",
        return_value={"error": "Pending order 123 not found"},
    ):
        out = trade_modify(ticket=123, price=1.2, __cli_raw=True)
    assert "error" in out
    assert out.get("error_code") == "ticket_not_found"
    assert out.get("ticket") == 123
    assert out.get("checked_scopes") == ["pending_orders"]
    assert "trade_get_pending" in str(out.get("suggestion"))


def test_trade_modify_missing_ticket_reports_both_checked_scopes() -> None:
    with patch(
        "mtdata.core.trading._modify_position",
        return_value={"error": "Position 123 not found"},
    ), patch(
        "mtdata.core.trading._modify_pending_order",
        return_value={"error": "Pending order 123 not found"},
    ):
        out = trade_modify(ticket=123, stop_loss=1.0, __cli_raw=True)
    assert "error" in out
    assert out.get("error_code") == "ticket_not_found"
    assert out.get("ticket") == 123
    assert out.get("checked_scopes") == ["positions", "pending_orders"]
    assert "trade_get_open" in str(out.get("suggestion"))
    assert out.get("remediation")


def test_trade_place_pending_without_sl_tp_is_blocked_by_default() -> None:
    with patch("mtdata.core.trading._place_pending_order") as mock_pending, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={"bid": 1.17, "ask": 1.1702, "entry_price": 1.16},
    ):
        out = trade_place(
            symbol="EURUSD",
            volume=0.01,
            order_type="BUY_LIMIT",
            price=1.16,
            dry_run=True,
            __cli_raw=True,
        )

    assert out["success"] is False
    assert out["preview_ok"] is False
    assert out["error_code"] == "preview_blocked"
    assert "missing_stop_loss" in out["blockers"]
    assert "missing_take_profit" in out["blockers"]
    assert out["require_sl_tp"] is True
    mock_pending.assert_not_called()
