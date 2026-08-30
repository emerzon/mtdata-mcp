from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from mtdata.core._mcp_tools import _select_output_fields
from mtdata.core.trading.comments import (
    _attach_comment_preview_metadata,
    _comment_sanitization_info,
    _normalize_trade_comment,
)
from mtdata.core.trading.orders import build_trade_place_dry_run_preview
from mtdata.core.trading.requests import TradeCloseRequest, TradePlaceRequest
from mtdata.core.trading.time import _server_time_naive_to_mt5_timestamp
from mtdata.core.trading.use_cases import run_trade_close, run_trade_place
from mtdata.core.trading.validation import (
    _normalize_order_type_input,
    _normalize_price_for_symbol,
    _normalize_trade_price_inputs,
    _retcode_is_accepted,
    _safe_float_attr,
    _trade_accepted_codes,
    _validate_deviation,
    _validate_live_protection_levels,
    _validate_pending_order_levels,
    _validate_volume,
)


def test_normalize_order_type_rejects_bool_and_fractional_numeric():
    normalized, error = _normalize_order_type_input(True)
    assert normalized is None
    assert "Unsupported order_type" in error

    normalized, error = _normalize_order_type_input(2.5)
    assert normalized is None
    assert "Unsupported order_type" in error


def test_normalize_order_type_rejects_alias_and_prefixed_names():
    normalized, error = _normalize_order_type_input("long")
    assert normalized is None
    assert "Unsupported order_type" in error

    normalized, error = _normalize_order_type_input("mt5.order_type_sell_limit")
    assert normalized is None
    assert "Unsupported order_type" in error


def test_validate_volume_handles_bad_symbol_constraints_gracefully():
    symbol_info = SimpleNamespace(volume_min=-1, volume_max="bad", volume_step=0)

    volume, error = _validate_volume(0.13, symbol_info)

    assert error is None
    assert volume == 0.13


def test_validate_deviation_rejects_negative_and_non_numeric():
    value, error = _validate_deviation(-1)
    assert value is None
    assert error == "deviation must be >= 0"

    value, error = _validate_deviation("abc")
    assert value is None
    assert error == "deviation must be numeric"


def test_trade_done_helpers_use_safe_int_attr_and_cached_codes():
    mt5 = SimpleNamespace(
        TRADE_RETCODE_PLACED="10008",
        TRADE_RETCODE_DONE=True,
        TRADE_RETCODE_DONE_PARTIAL="10010",
    )

    accepted_codes = _trade_accepted_codes(mt5)

    assert accepted_codes == {10008, 10009, 10010}
    assert _retcode_is_accepted(mt5, "10008", accepted_codes) is True
    assert _retcode_is_accepted(mt5, "10010", accepted_codes) is True
    assert _retcode_is_accepted(mt5, 1, accepted_codes) is False


def test_normalize_price_for_symbol_rejects_negative_values():
    assert _normalize_price_for_symbol(-37.634, point=0.01, digits=2) is None


def test_normalize_price_for_symbol_removes_binary_tick_residue():
    assert _normalize_price_for_symbol(1.10001, point=0.00001, digits=5) == 1.10001
    assert _normalize_price_for_symbol(2654.56, point=0.01, digits=2) == 2654.56


def test_trade_prices_snap_to_trade_tick_size_when_coarser_than_point():
    symbol_info = SimpleNamespace(point=0.01, trade_tick_size=0.05, digits=2)

    normalized, error = _normalize_trade_price_inputs(
        symbol_info=symbol_info,
        price=2654.57,
        require_price=True,
        stop_loss=2650.02,
        take_profit=2660.03,
    )

    assert error is None
    assert normalized["point"] == 0.01
    assert normalized["price_increment"] == 0.05
    assert normalized["price"] == 2654.55
    assert normalized["stop_loss"] == 2650.0
    assert normalized["take_profit"] == 2660.05


def test_validate_live_protection_levels_accepts_negative_quotes():
    symbol_info = SimpleNamespace(point=0.01, trade_stops_level=0, trade_freeze_level=0)
    tick = SimpleNamespace(bid=-37.64, ask=-37.63)

    result = _validate_live_protection_levels(
        symbol_info=symbol_info,
        tick=tick,
        side="BUY",
        stop_loss=-37.80,
        take_profit=-37.20,
    )

    assert result is None


def test_protection_validators_accept_mapping_ticks():
    symbol_info = SimpleNamespace(
        point=0.0001,
        trade_stops_level=0,
        trade_freeze_level=0,
    )
    tick = {"bid": 1.1, "ask": 1.1002}

    live_result = _validate_live_protection_levels(
        symbol_info=symbol_info,
        tick=tick,
        side="BUY",
        stop_loss=1.09,
        take_profit=1.11,
    )
    pending_result = _validate_pending_order_levels(
        symbol_info=symbol_info,
        tick=tick,
        order_type_value=2,
        price=1.099,
        stop_loss=1.09,
        take_profit=1.11,
        mt5=SimpleNamespace(
            ORDER_TYPE_BUY_LIMIT=2,
            ORDER_TYPE_SELL_LIMIT=3,
            ORDER_TYPE_BUY_STOP=4,
            ORDER_TYPE_SELL_STOP=5,
        ),
    )

    assert live_result is None
    assert pending_result is None


@pytest.mark.parametrize(
    ("order_type", "trigger", "limit_price"),
    [(6, 1.1010, 1.1008), (7, 1.0990, 1.0992)],
)
def test_stop_limit_levels_accept_directional_price_relationship(
    order_type, trigger, limit_price
):
    result = _validate_pending_order_levels(
        symbol_info=SimpleNamespace(
            point=0.0001,
            trade_stops_level=0,
            trade_freeze_level=0,
        ),
        tick=SimpleNamespace(bid=1.1000, ask=1.1002),
        order_type_value=order_type,
        price=trigger,
        stop_limit_price=limit_price,
        stop_loss=1.09 if order_type == 6 else 1.11,
        take_profit=1.12 if order_type == 6 else 1.08,
        mt5=SimpleNamespace(
            ORDER_TYPE_BUY_LIMIT=2,
            ORDER_TYPE_SELL_LIMIT=3,
            ORDER_TYPE_BUY_STOP=4,
            ORDER_TYPE_SELL_STOP=5,
            ORDER_TYPE_BUY_STOP_LIMIT=6,
            ORDER_TYPE_SELL_STOP_LIMIT=7,
        ),
    )

    assert result is None


@pytest.mark.parametrize(
    ("order_type", "trigger", "limit_price", "message"),
    [
        (6, 1.1010, 1.1012, "at or below"),
        (7, 1.0990, 1.0988, "at or above"),
    ],
)
def test_stop_limit_levels_reject_reversed_price_relationship(
    order_type, trigger, limit_price, message
):
    result = _validate_pending_order_levels(
        symbol_info=SimpleNamespace(
            point=0.0001,
            trade_stops_level=0,
            trade_freeze_level=0,
        ),
        tick=SimpleNamespace(bid=1.1000, ask=1.1002),
        order_type_value=order_type,
        price=trigger,
        stop_limit_price=limit_price,
        stop_loss=None,
        take_profit=None,
        mt5=SimpleNamespace(
            ORDER_TYPE_BUY_LIMIT=2,
            ORDER_TYPE_SELL_LIMIT=3,
            ORDER_TYPE_BUY_STOP=4,
            ORDER_TYPE_SELL_STOP=5,
            ORDER_TYPE_BUY_STOP_LIMIT=6,
            ORDER_TYPE_SELL_STOP_LIMIT=7,
        ),
    )

    assert message in result["error"]


def test_run_trade_close_rejects_conflicting_profit_and_loss_filters():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TradeCloseRequest(close_all=True, profit_only=True, loss_only=True)


def test_run_trade_close_uses_history_lookup_when_ticket_is_already_closed():
    request = TradeCloseRequest(ticket=123, dry_run=False)
    close_positions = MagicMock(return_value={"error": "Position 123 not found"})
    cancel_pending = MagicMock(return_value={"error": "Pending order 123 not found"})
    lookup_ticket_history = MagicMock(
        return_value={
            "message": "Ticket 123 was a Buy position that has already been closed at 2026-03-29 10:00 UTC. No action taken.",
            "no_action": True,
            "checked_scopes": ["positions", "pending_orders", "history_deals"],
        }
    )

    result = run_trade_close(
        request,
        close_positions=close_positions,
        cancel_pending=cancel_pending,
        lookup_ticket_history=lookup_ticket_history,
    )

    assert result["no_action"] is True
    assert "already been closed" in result["message"]
    assert result["checked_scopes"] == ["positions", "pending_orders", "history_deals"]
    lookup_ticket_history.assert_called_once_with(123)


def test_run_trade_close_passes_magic_filter_to_all_exposure_legs():
    request = TradeCloseRequest(
        magic=987,
        target="all_exposure",
        confirm_close_all=True,
        dry_run=False,
    )
    close_positions = MagicMock(return_value={"closed_count": 1})
    cancel_pending = MagicMock(return_value={"cancelled_count": 1})

    result = run_trade_close(
        request,
        close_positions=close_positions,
        cancel_pending=cancel_pending,
    )

    assert result["success"] is True
    assert result["closed_count"] == 1
    assert result["cancelled_count"] == 1
    close_positions.assert_called_once()
    cancel_pending.assert_called_once()
    assert close_positions.call_args.kwargs["magic"] == 987
    assert cancel_pending.call_args.kwargs["magic"] == 987


def test_run_trade_close_preview_uses_transport_neutral_comment():
    result = run_trade_close(
        TradeCloseRequest(symbol="EURUSD", dry_run=True, confirm_close_all=True),
        close_positions=lambda **_kwargs: {"success": True, "matched_count": 1},
        cancel_pending=MagicMock(),
    )

    assert result["success"] is True
    assert result["comment"] == "mtdata close"
    assert result["applied_comment"] == "mtdata close"


def test_normalize_trade_comment_applies_default_and_suffix_length_caps():
    comment = _normalize_trade_comment(None, default="DefaultComment", suffix="-MKT")
    assert comment == "DefaultComment-MKT"

    long_comment = _normalize_trade_comment("x" * 50, default="ignored", suffix="-TP")
    assert len(long_comment) == 31
    assert long_comment.endswith("-TP")


def test_normalize_trade_comment_sanitizes_special_characters():
    comment = _normalize_trade_comment(
        "Short: EV+, R:R 9.65",
        default="ignored",
    )
    assert ":" not in comment
    assert "," not in comment
    assert "+" not in comment
    assert comment == "Short EV R R 9.65"


def test_comment_sanitization_info_reports_changes():
    info = _comment_sanitization_info(
        "Short: ETS bearish, barrier EV+, R:R 9.65",
        "Short ETS bearish barrier EV R R 9.65",
    )
    assert info == {
        "requested": "Short: ETS bearish, barrier EV+, R:R 9.65",
        "applied": "Short ETS bearish barrier EV R R 9.65",
    }


def test_comment_preview_reports_exact_close_comment_and_limit():
    requested = "Close 🚀 because this explanation is much too long"

    result = _attach_comment_preview_metadata(
        {"warnings": ["Existing warning"]},
        requested,
        default="MCP close",
        close=True,
    )

    assert result["requested_comment"] == requested
    assert result["applied_comment"] == "Close because this expla"
    assert result["comment"] == result["applied_comment"]
    assert result["comment_max_length"] == 24
    assert result["comment_changed"] is True
    assert result["comment_sanitization"]["requested"] == requested
    assert result["comment_truncation"]["max_length"] == 24
    assert result["warnings"][0] == "Existing warning"
    assert any("sanitized" in warning for warning in result["warnings"])
    assert any("truncated to 24" in warning for warning in result["warnings"])


def test_comment_preview_leaves_short_safe_comment_without_warning():
    result = _attach_comment_preview_metadata(
        {},
        "strategy 42",
        default="MCP order",
    )

    assert result["requested_comment"] == "strategy 42"
    assert result["applied_comment"] == "strategy 42"
    assert result["comment_max_length"] == 31
    assert "comment_changed" not in result
    assert "warnings" not in result


def test_server_time_naive_to_mt5_timestamp_strips_timezone():
    ts = _server_time_naive_to_mt5_timestamp(datetime(1970, 1, 1, 0, 1, 0, tzinfo=timezone.utc))
    assert ts == 60


def test_run_trade_place_logs_finish_event(caplog):
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        require_sl_tp=False,
        dry_run=False,
    )

    with caplog.at_level("DEBUG", logger="mtdata.core.trading.use_cases"):
        result = run_trade_place(
            request,
            normalize_order_type_input=lambda value: ("BUY", None),
            normalize_pending_expiration=lambda value: (value, False),
            prevalidate_trade_place_market_input=lambda symbol, volume: None,
            place_market_order=lambda **kwargs: {"success": True, "order_id": 7},
            place_pending_order=lambda **kwargs: {"success": True, "order_id": 8},
            close_positions=lambda **kwargs: {"closed_count": 1},
            safe_int_ticket=lambda value: value,
        )

    assert result["success"] is True
    assert result["order_id"] == 7
    assert result["guardrails_enabled"] is False
    assert any("without configured trade guardrails" in warning for warning in result["warnings"])
    assert any(
        "event=finish operation=trade_place success=True" in record.message
        for record in caplog.records
    )


def test_run_trade_place_ignores_gtc_for_market_buy_sell_without_price():
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        expiration="GTC",
        require_sl_tp=False,
        dry_run=False,
    )
    place_market_order = MagicMock(return_value={"success": True, "path": "market"})
    place_pending_order = MagicMock(return_value={"success": True, "path": "pending"})

    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (None, True),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=place_pending_order,
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
    )

    assert result["success"] is True
    assert result["path"] == "market"
    assert result["guardrails_enabled"] is False
    place_market_order.assert_called_once()
    place_pending_order.assert_not_called()


def test_run_trade_place_rejects_dated_market_expiration_without_price():
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        expiration="2026-04-01 12:00",
        require_sl_tp=False,
    )
    place_market_order = MagicMock(return_value={"success": True, "path": "market"})
    place_pending_order = MagicMock(return_value={"success": True, "path": "pending"})

    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (1711972800, True),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=place_pending_order,
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
    )

    assert "error" in result
    assert "expiration only applies to pending orders placed with a price" in result["error"]
    place_market_order.assert_not_called()
    place_pending_order.assert_not_called()


def test_run_trade_place_dry_run_returns_preview_without_execution():
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1.08,
        take_profit=1.12,
        comment="Preview 🚀 comment that exceeds thirty-one characters",
        dry_run=True,
        detail="full",
    )
    place_market_order = MagicMock(return_value={"success": True, "path": "market"})
    place_pending_order = MagicMock(return_value={"success": True, "path": "pending"})

    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: {"error": "should not run"},
        place_market_order=place_market_order,
        place_pending_order=place_pending_order,
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
    )

    assert result["dry_run"] is True
    assert result["no_action"] is True
    assert result["no_action_reason"] == "dry_run"
    assert result["would_send_order"] is False
    assert result["dry_run_simulated"] is True
    assert result["pending"] is False
    assert result["action"] == "place_market_order"
    assert result["validation_scope"] == "local_preview_plus_estimates"
    assert "trade_gate_passed" not in result
    assert result["actionability"] == "preview_only"
    assert "validation_not_performed" not in result
    assert "protection_level_preview" in result["preview_checks_performed"]
    assert "margin_estimate" not in result["preview_checks_performed"]
    assert "margin_estimate" in result["checks_not_performed"]
    assert "broker_acceptance" in result["broker_validation_not_performed"]
    assert result["stop_loss"] == 1.08
    assert result["take_profit"] == 1.12
    assert result["blockers"] == []
    assert result["requested_comment"].startswith("Preview 🚀")
    assert result["applied_comment"] == "Preview comment that exceeds th"
    assert result["comment"] == result["applied_comment"]
    assert result["comment_max_length"] == 31
    assert result["comment_sanitization"]["requested"] == request.comment
    assert result["comment_truncation"]["max_length"] == 31
    assert "requested_sl" not in result
    assert "requested_tp" not in result
    selected = _select_output_fields(result, "stop_loss,take_profit,blockers")
    assert selected["stop_loss"] == 1.08
    assert selected["take_profit"] == 1.12
    assert selected["blockers"] == []
    assert "unresolved_output_fields" not in selected
    place_market_order.assert_not_called()
    place_pending_order.assert_not_called()


def test_run_trade_place_dry_run_includes_quote_preview_when_available():
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1.08,
        take_profit=1.12,
        dry_run=True,
    )
    preview_builder = MagicMock(
        return_value={
            "bid": 1.0999,
            "ask": 1.1001,
            "spread_points": 2.0,
            "estimated_fill_price": 1.1001,
            "margin_required": 110.0,
            "margin_sufficient": True,
            "sl_tp_valid": True,
            "quote_context": {"usable_for_live_trading": True},
        }
    )
    place_market_order = MagicMock(return_value={"success": True, "path": "market"})
    place_pending_order = MagicMock(return_value={"success": True, "path": "pending"})

    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: {"error": "should not run"},
        place_market_order=place_market_order,
        place_pending_order=place_pending_order,
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
        build_dry_run_preview=preview_builder,
    )

    assert result["dry_run"] is True
    assert result["bid"] == 1.0999
    assert result["ask"] == 1.1001
    assert result["estimated_fill_price"] == 1.1001
    assert result["margin_required"] == 110.0
    assert result["sl_tp_valid"] is True
    assert result["magic"] == 234000
    assert result["comment"] == "mtdata order"
    preview_builder.assert_called_once_with(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        pending=False,
        price=None,
        stop_limit_price=None,
        stop_loss=1.08,
        take_profit=1.12,
    )
    place_market_order.assert_not_called()
    place_pending_order.assert_not_called()


def test_run_trade_place_compact_preview_slims_quote_and_validation():
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1.08,
        take_profit=1.12,
        dry_run=True,
        detail="compact",
    )
    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=MagicMock(),
        place_pending_order=MagicMock(),
        close_positions=MagicMock(),
        safe_int_ticket=lambda value: value,
        build_dry_run_preview=lambda **_kwargs: {
            "bid": 1.0999,
            "ask": 1.1001,
            "estimated_fill_price": 1.1001,
            "sl_tp_valid": True,
            "candidate_risk": {
                "status": "ok",
                "risk_currency": 20.0,
                "risk_pct_of_equity": 0.5,
                "reward_currency": 40.0,
                "reward_risk_ratio": 2.0,
                "basis": "tick_value_linear_sensitivity",
                "equity": 4000.0,
            },
            "quote_context": {
                "usable_for_live_trading": True,
                "freshness_state": "live",
                "quote_time": "2026-07-15T12:00:00Z",
                "quote_time_epoch": 1_784_113_200,
                "timestamp_skew_seconds": 0.2,
                "quote_source": "mt5.symbol_info_tick",
            },
        },
    )

    assert result["preview_ok"] is True
    assert result["quote_context"] == {
        "usable_for_live_trading": True,
        "freshness_state": "live",
        "quote_time": "2026-07-15T12:00:00Z",
    }
    assert result["validation"] == {
        "local_requirements_passed": True,
        "live_submission_eligible": True,
        "blockers": [],
    }
    assert "quote_time_epoch" not in result["quote_context"]
    assert result["stop_loss"] == 1.08
    assert result["take_profit"] == 1.12
    assert result["candidate_risk"] == {
        "status": "ok",
        "risk_currency": 20.0,
        "risk_pct_of_equity": 0.5,
        "reward_currency": 40.0,
        "reward_risk_ratio": 2.0,
    }


def test_run_trade_place_preview_marks_unprotected_by_request():
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        require_sl_tp=False,
        dry_run=True,
    )
    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=MagicMock(),
        place_pending_order=MagicMock(),
        close_positions=MagicMock(),
        safe_int_ticket=lambda value: value,
        build_dry_run_preview=lambda **_kwargs: {
            "bid": 1.0999,
            "ask": 1.1001,
            "estimated_fill_price": 1.1001,
            "quote_context": {"usable_for_live_trading": True, "freshness_state": "live"},
        },
    )

    assert result["preview_ok"] is True
    assert result["require_sl_tp"] is False
    assert result["protection_status"] == "unprotected_by_request"
    assert result["auto_close_on_sl_tp_fail"] is False
    assert any("unprotected_by_request" in str(item) for item in result["warnings"])


def test_run_trade_place_dry_run_blocks_untrusted_quote_preview():
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1.08,
        take_profit=1.12,
        dry_run=True,
        detail="standard",
    )
    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=MagicMock(),
        place_pending_order=MagicMock(),
        close_positions=MagicMock(),
        safe_int_ticket=lambda value: value,
        build_dry_run_preview=lambda **_kwargs: {
            "bid": 1.0999,
            "ask": 1.1001,
            "estimated_fill_price": 1.1001,
            "sl_tp_valid": True,
            "quote_context": {
                "freshness_state": "stale",
                "freshness_reason": "stale_age",
                "usable_for_live_trading": False,
            },
        },
    )

    assert result["preview_ok"] is False
    assert result["success"] is False
    assert result["error_code"] == "preview_blocked"
    assert result["status"] == "preview_blocked"
    assert result["validation_passed"] is False
    assert result["blockers"] == ["quote_not_live_ready"]
    assert result["no_action_reason"] == "dry_run_validation_blocked"
    assert result["quote_context"]["usable_for_live_trading"] is False


def test_run_trade_place_dry_run_names_weekend_closure_instead_of_refresh():
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.01,
        order_type="BUY",
        stop_loss=1.08,
        take_profit=1.12,
        dry_run=True,
        detail="standard",
    )
    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=MagicMock(),
        place_pending_order=MagicMock(),
        close_positions=MagicMock(),
        safe_int_ticket=lambda value: value,
        build_dry_run_preview=lambda **_kwargs: {
            "bid": 1.0999,
            "ask": 1.1001,
            "estimated_fill_price": 1.1001,
            "sl_tp_valid": True,
            "quote_context": {
                "freshness_state": "closed_weekend_snapshot",
                "freshness_reason": "market_closed",
                "usable_for_live_trading": False,
                "market_status": "closed",
                "market_status_reason": "weekend",
                "assumed_closure_end": "2026-08-31T21:00:00Z",
            },
        },
    )

    assert result["success"] is False
    assert result["blockers"] == ["market_closed_weekend"]
    assert result["market_status"] == "closed"
    assert result["market_status_reason"] == "weekend"
    assert result["next_market_open"] == "2026-08-31T21:00:00Z"
    assert "weekend" in result["error"].lower()
    assert "2026-08-31T21:00:00Z" in result["error"]
    assert "refresh it and retry" not in result["error"].lower()
    assert "refresh it and retry" not in result["remediation"].lower()
    assert "next market open" in result["remediation"]


def test_run_trade_place_preview_blocks_market_when_raw_send_tick_is_skewed():
    request = TradePlaceRequest(
        symbol="BTCUSD",
        volume=0.01,
        order_type="BUY",
        stop_loss=70000,
        take_profit=85000,
        dry_run=True,
        detail="standard",
    )
    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=MagicMock(),
        place_pending_order=MagicMock(),
        close_positions=MagicMock(),
        safe_int_ticket=lambda value: value,
        build_dry_run_preview=lambda **_kwargs: {
            "bid": 80000.0,
            "ask": 80010.0,
            "estimated_fill_price": 80010.0,
            "sl_tp_valid": True,
            "quote_context": {
                "usable_for_live_trading": True,
                "freshness_state": "live",
                "send_path_tick_fresh": False,
                "send_path_tick_age_status": "future",
                "send_path_freshness_error": (
                    "Tick for BTCUSD is 45.0s ahead of the wall clock "
                    "and is not safe for live trading."
                ),
            },
        },
    )

    assert result["preview_ok"] is False
    assert result["validation"]["live_submission_eligible"] is False
    assert result["blockers"] == ["raw_quote_timestamp_ahead_of_clock"]


def test_run_trade_place_pending_preview_allows_stale_session_snapshot():
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.01,
        order_type="BUY_LIMIT",
        price=1.14,
        stop_loss=1.13,
        take_profit=1.16,
        dry_run=True,
        detail="standard",
    )
    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY_LIMIT", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=MagicMock(),
        place_pending_order=MagicMock(),
        close_positions=MagicMock(),
        safe_int_ticket=lambda value: value,
        build_dry_run_preview=lambda **_kwargs: {
            "bid": 1.15809,
            "ask": 1.15825,
            "estimated_fill_price": 1.14,
            "sl_tp_valid": True,
            "quote_context": {
                "usable_for_live_trading": False,
                "freshness_state": "stale",
                "send_path_tick_fresh": False,
                "send_path_tick_age_status": "stale",
            },
        },
    )

    assert result["preview_ok"] is True
    assert result["validation"]["live_submission_eligible"] is True
    assert "quote_not_live_ready" not in result["blockers"]


def test_build_trade_place_dry_run_preview_uses_live_quote_and_margin():
    adapter = SimpleNamespace(
        ORDER_TYPE_BUY=0,
        order_calc_margin=MagicMock(return_value=123.45),
    )
    gateway = MagicMock()
    gateway.adapter = adapter
    gateway.ORDER_TYPE_BUY = 0
    gateway.symbol_info.return_value = SimpleNamespace(
        visible=True,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        point=0.00001,
        digits=5,
        trade_stops_level=10,
        trade_freeze_level=0,
    )
    fixed_now = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
    gateway.symbol_info_tick.return_value = SimpleNamespace(
        bid=1.0999,
        ask=1.1001,
        time=fixed_now.timestamp(),
    )
    gateway.account_info.return_value = SimpleNamespace(margin_free=1000.0)

    with patch("mtdata.core.trading.common.datetime", wraps=datetime) as mock_datetime, patch(
        "mtdata.core.trading.orders._stdlib_time.time",
        return_value=fixed_now.timestamp(),
    ):
        mock_datetime.now.return_value = fixed_now
        result = build_trade_place_dry_run_preview(
            symbol="EURUSD",
            volume=0.1,
            order_type="BUY",
            pending=False,
            price=None,
            stop_limit_price=None,
            stop_loss=1.08,
            take_profit=1.12,
            gateway=gateway,
        )

    assert result["bid"] == 1.0999
    assert result["ask"] == 1.1001
    assert result["estimated_fill_price"] == 1.1001
    assert result["spread_points"] == 20.0
    assert result["spread_pips"] == 2.0
    assert result["sl_distance_points"] == 2010.0
    assert result["sl_distance_pips"] == 201.0
    assert result["tp_distance_points"] == 1990.0
    assert result["tp_distance_pips"] == 199.0
    assert result["sl_distance_pct"] > 0
    assert result["tp_distance_pct"] > 0
    assert result["sl_tp_valid"] is True
    assert result["margin_required"] == 123.45
    assert result["margin_free"] == 1000.0
    assert result["margin_sufficient"] is True
    assert result["margin_action"] == "BUY"
    assert result["margin_estimate_basis"] == "market_fill_side_at_estimated_price"
    assert result["quote_context"]["usable_for_live_trading"] is True
    assert result["quote_context"]["freshness_state"] == "live"
    assert result["quote_context"]["send_path_tick_fresh"] is True
    assert result["quote_context"]["quote_timezone"] == "UTC"
    assert result["units"]["sl_distance_points"] == "broker_points"
    assert result["units"]["sl_distance_pips"] == "pips"
    adapter.order_calc_margin.assert_called_once_with(0, "EURUSD", 0.1, 1.1001)


def test_build_trade_place_dry_run_preview_reconciles_bounded_broker_clock_lead():
    adapter = SimpleNamespace(
        ORDER_TYPE_BUY=0,
        order_calc_margin=MagicMock(return_value=123.45),
    )
    gateway = MagicMock()
    gateway.adapter = adapter
    gateway.ORDER_TYPE_BUY = 0
    gateway.symbol_info.return_value = SimpleNamespace(
        visible=True,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        point=0.00001,
        digits=5,
        trade_stops_level=10,
        trade_freeze_level=0,
    )
    fixed_now = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
    gateway.symbol_info_tick.return_value = SimpleNamespace(
        bid=1.0999,
        ask=1.1001,
        time=fixed_now.timestamp() + 13.9,
        time_msc=(fixed_now.timestamp() + 13.9) * 1000.0,
    )
    gateway.account_info.return_value = SimpleNamespace(margin_free=1000.0)

    with patch("mtdata.core.trading.common.datetime", wraps=datetime) as mock_datetime, patch(
        "mtdata.core.trading.orders._stdlib_time.time",
        return_value=fixed_now.timestamp(),
    ):
        mock_datetime.now.return_value = fixed_now
        result = build_trade_place_dry_run_preview(
            symbol="BTCUSD",
            volume=0.01,
            order_type="BUY",
            pending=False,
            price=None,
            stop_limit_price=None,
            stop_loss=70000,
            take_profit=85000,
            gateway=gateway,
        )

    assert result["quote_context"]["send_path_tick_fresh"] is True
    assert result["quote_context"]["usable_for_live_trading"] is True
    assert result["quote_context"]["clock_reconciled"] is True
    assert result["quote_context"]["quote_clock_reference"] == (
        "broker_tick_at_acquisition"
    )
    assert result["quote_context"]["local_clock_lag_seconds"] == pytest.approx(
        13.9,
        abs=0.01,
    )
    assert result["quote_context"]["data_age_anchor"] == (
        "broker_tick_reconciled_clock"
    )


@pytest.mark.parametrize(
    ("order_type", "trigger", "limit_price", "side_action"),
    [
        ("BUY_STOP_LIMIT", 1.1010, 1.1008, 0),
        ("SELL_STOP_LIMIT", 1.0990, 1.0992, 1),
    ],
)
def test_stop_limit_dry_run_previews_trigger_and_limit_price(
    order_type, trigger, limit_price, side_action
):
    margin = MagicMock(return_value=50.0)
    gateway = MagicMock()
    gateway.adapter = SimpleNamespace(order_calc_margin=margin)
    gateway.ORDER_TYPE_BUY = 0
    gateway.ORDER_TYPE_SELL = 1
    gateway.ORDER_TYPE_BUY_STOP_LIMIT = 6
    gateway.ORDER_TYPE_SELL_STOP_LIMIT = 7
    gateway.symbol_info.return_value = SimpleNamespace(
        visible=True,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        point=0.0001,
        digits=4,
        trade_stops_level=0,
        trade_freeze_level=0,
    )
    quote_time = 4_102_444_800.0
    gateway.symbol_info_tick.return_value = SimpleNamespace(
        bid=1.1000,
        ask=1.1002,
        time=quote_time,
    )
    gateway.account_info.return_value = SimpleNamespace(margin_free=1_000.0)

    with patch(
        "mtdata.core.trading.orders._stdlib_time.time", return_value=quote_time
    ):
        result = build_trade_place_dry_run_preview(
            symbol="EURUSD",
            volume=0.1,
            order_type=order_type,
            pending=True,
            price=trigger,
            stop_limit_price=limit_price,
            stop_loss=1.09 if side_action == 0 else 1.11,
            take_profit=1.12 if side_action == 0 else 1.08,
            gateway=gateway,
        )

    assert result["pending_levels_valid"] is True
    assert result["trigger_price"] == pytest.approx(trigger)
    assert result["stop_limit_price"] == pytest.approx(limit_price)
    assert result["estimated_fill_price"] == pytest.approx(limit_price)
    margin.assert_called_once_with(side_action, "EURUSD", 0.1, limit_price)


@pytest.mark.parametrize(
    ("price", "stop_loss", "take_profit"),
    [
        (0, None, None),
        (-1, None, None),
        (float("nan"), None, None),
        (float("inf"), None, None),
        (float("-inf"), None, None),
        (0, 1.08, 1.12),
    ],
)
def test_pending_dry_run_rejects_invalid_price_before_quote_resolution(
    price,
    stop_loss,
    take_profit,
):
    gateway = MagicMock()
    gateway.account_info.return_value = None
    gateway.symbol_info.return_value = SimpleNamespace(
        visible=True,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        point=0.00001,
        digits=5,
    )

    result = build_trade_place_dry_run_preview(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY_LIMIT",
        pending=True,
        price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        gateway=gateway,
    )

    assert result["preview_error_code"] == "invalid_pending_price"
    assert "strictly positive finite" in result["preview_error"]
    gateway.symbol_info_tick.assert_not_called()


@pytest.mark.parametrize(
    ("order_type", "entry_price", "stop_loss", "take_profit"),
    [
        ("BUY_LIMIT", 1.1010, 1.0800, 1.1200),
        ("BUY_STOP", 1.0990, 1.0800, 1.1200),
        ("SELL_LIMIT", 1.0990, 1.1200, 1.0800),
        ("SELL_STOP", 1.1010, 1.1200, 1.0800),
    ],
)
def test_pending_entry_error_does_not_misclassify_valid_protection(
    order_type,
    entry_price,
    stop_loss,
    take_profit,
):
    now = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc).timestamp()
    gateway = MagicMock()
    gateway.adapter = SimpleNamespace(order_calc_margin=MagicMock(return_value=20.0))
    gateway.ORDER_TYPE_BUY = 0
    gateway.ORDER_TYPE_SELL = 1
    gateway.ORDER_TYPE_BUY_LIMIT = 2
    gateway.ORDER_TYPE_SELL_LIMIT = 3
    gateway.ORDER_TYPE_BUY_STOP = 4
    gateway.ORDER_TYPE_SELL_STOP = 5
    gateway.symbol_info.return_value = SimpleNamespace(
        visible=True,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        point=0.0001,
        digits=4,
        trade_stops_level=0,
        trade_freeze_level=0,
    )
    gateway.symbol_info_tick.return_value = SimpleNamespace(
        bid=1.0999,
        ask=1.1001,
        time=now,
    )
    gateway.account_info.return_value = SimpleNamespace(margin_free=1_000.0)

    with patch("mtdata.core.trading.orders._stdlib_time.time", return_value=now):
        result = build_trade_place_dry_run_preview(
            symbol="EURUSD",
            volume=0.1,
            order_type=order_type,
            pending=True,
            price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            gateway=gateway,
        )

    assert result["pending_levels_valid"] is False
    assert result["pending_entry_valid"] is False
    assert result["pending_entry_error"] == result["preview_error"]
    assert result["sl_tp_valid"] is True
    assert "sl_tp_error" not in result


@pytest.mark.parametrize(
    ("order_type", "entry_price", "expected_action"),
    [
        ("BUY_LIMIT", 1.099, 0),
        ("BUY_STOP", 1.101, 0),
        ("SELL_LIMIT", 1.101, 1),
        ("SELL_STOP", 1.099, 1),
    ],
)
def test_pending_margin_preview_uses_fill_side_action(
    order_type,
    entry_price,
    expected_action,
):
    order_calc_margin = MagicMock(return_value=220.0)
    gateway = MagicMock()
    gateway.adapter = SimpleNamespace(order_calc_margin=order_calc_margin)
    gateway.ORDER_TYPE_BUY = 0
    gateway.ORDER_TYPE_SELL = 1
    gateway.ORDER_TYPE_BUY_LIMIT = 2
    gateway.ORDER_TYPE_SELL_LIMIT = 3
    gateway.ORDER_TYPE_BUY_STOP = 4
    gateway.ORDER_TYPE_SELL_STOP = 5
    gateway.symbol_info.return_value = SimpleNamespace(
        visible=True,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        point=0.0001,
        digits=4,
        trade_stops_level=0,
        trade_freeze_level=0,
    )
    now = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc).timestamp()
    gateway.symbol_info_tick.return_value = SimpleNamespace(
        bid=1.0999,
        ask=1.1001,
        time=now,
    )
    gateway.account_info.return_value = SimpleNamespace(margin_free=100.0)

    with patch("mtdata.core.trading.orders._stdlib_time.time", return_value=now):
        result = build_trade_place_dry_run_preview(
            symbol="EURUSD",
            volume=0.1,
            order_type=order_type,
            pending=True,
            price=entry_price,
            stop_loss=None,
            take_profit=None,
            gateway=gateway,
        )

    order_calc_margin.assert_called_once_with(
        expected_action,
        "EURUSD",
        0.1,
        entry_price,
    )
    assert result["margin_required"] == 220.0
    assert result["margin_required_when_filled"] == 220.0
    assert result["margin_sufficient"] is False
    assert result["margin_action"] == (
        "BUY" if order_type.startswith("BUY") else "SELL"
    )
    assert result["margin_estimate_basis"] == "pending_fill_side_at_entry_price"


def test_trade_preview_does_not_emit_negative_metrics_for_inverted_quote():
    now = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc).timestamp()
    gateway = MagicMock()
    gateway.adapter = SimpleNamespace(
        ORDER_TYPE_BUY=0,
        order_calc_margin=MagicMock(return_value=10.0),
    )
    gateway.ORDER_TYPE_BUY = 0
    gateway.symbol_info.return_value = SimpleNamespace(
        visible=True,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        point=0.0001,
        digits=4,
        trade_stops_level=0,
        trade_freeze_level=0,
    )
    gateway.symbol_info_tick.return_value = SimpleNamespace(
        bid=1.1002,
        ask=1.1,
        time=now,
    )

    with patch("mtdata.core.trading.orders._stdlib_time.time", return_value=now):
        result = build_trade_place_dry_run_preview(
            symbol="EURUSD",
            volume=0.1,
            order_type="BUY",
            pending=False,
            price=None,
            stop_loss=None,
            take_profit=None,
            gateway=gateway,
        )

    assert "spread_points" not in result
    assert "spread_pips" not in result
    assert "spread_pct" not in result
    assert result["quote_context"]["usable_for_live_trading"] is False


def test_build_trade_place_dry_run_preview_exposes_account_blockers():
    adapter = SimpleNamespace(
        ORDER_TYPE_BUY=0,
        order_calc_margin=MagicMock(return_value=20.0),
    )
    gateway = MagicMock()
    gateway.adapter = adapter
    gateway.ORDER_TYPE_BUY = 0
    gateway.symbol_info.return_value = SimpleNamespace(
        visible=True,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        point=0.00001,
        digits=5,
        trade_stops_level=0,
        trade_freeze_level=0,
    )
    now = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc).timestamp()
    gateway.symbol_info_tick.return_value = SimpleNamespace(
        bid=1.0999,
        ask=1.1001,
        time=now,
    )
    gateway.account_info.return_value = SimpleNamespace(
        equity=100.0,
        margin=101.0,
        margin_free=-1.0,
        margin_level=99.0,
        trade_allowed=False,
    )

    with patch("mtdata.core.trading.orders._stdlib_time.time", return_value=now):
        result = build_trade_place_dry_run_preview(
            symbol="EURUSD",
            volume=0.1,
            order_type="BUY",
            pending=False,
            price=None,
            stop_loss=1.08,
            take_profit=1.12,
            gateway=gateway,
        )

    assert result["account_blockers"] == [
        "account_trading_disabled",
        "no_free_margin",
        "critical_margin_stress",
    ]
    assert result["account_state"]["trade_allowed"] is False
    assert result["account_state"]["broker_trade_allowed"] is False
    assert result["account_state"]["new_exposure_allowed"] is False
    assert result["account_state"]["margin_stress"]["status"] == "critical"
    assert result["margin_sufficient"] is False


def test_build_trade_place_dry_run_preview_combines_trade_allowed_gate():
    adapter = SimpleNamespace(
        ORDER_TYPE_BUY=0,
        order_calc_margin=MagicMock(return_value=20.0),
    )
    gateway = MagicMock()
    gateway.adapter = adapter
    gateway.ORDER_TYPE_BUY = 0
    gateway.build_trade_preflight.return_value = {"execution_ready_strict": True}
    gateway.symbol_info.return_value = SimpleNamespace(
        visible=True,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        point=0.00001,
        digits=5,
        trade_stops_level=0,
        trade_freeze_level=0,
    )
    now = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc).timestamp()
    gateway.symbol_info_tick.return_value = SimpleNamespace(
        bid=1.0999,
        ask=1.1001,
        time=now,
    )
    gateway.account_info.return_value = SimpleNamespace(
        equity=100.0,
        margin=101.0,
        margin_free=-1.0,
        margin_level=99.0,
        trade_allowed=True,
    )

    with patch("mtdata.core.trading.orders._stdlib_time.time", return_value=now):
        result = build_trade_place_dry_run_preview(
            symbol="EURUSD",
            volume=0.1,
            order_type="BUY",
            pending=False,
            price=None,
            stop_loss=1.08,
            take_profit=1.12,
            gateway=gateway,
        )

    assert result["account_state"]["broker_trade_allowed"] is True
    assert result["account_state"]["trade_allowed"] is False
    assert result["account_state"]["new_exposure_allowed"] is False
    assert result["account_state"]["trade_allowed_basis"] == [
        "broker_trade_allowed",
        "margin_not_critical",
        "execution_ready_strict",
    ]
    assert result["account_state"]["margin_stress"]["status"] == "critical"
    assert "critical_margin_stress" in result["account_blockers"]


def test_trade_preview_reconciles_equal_timestamp_quote_conflict():
    now = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc).timestamp()
    gateway = MagicMock()
    gateway.adapter = SimpleNamespace(
        ORDER_TYPE_BUY=0,
        order_calc_margin=MagicMock(return_value=123.45),
    )
    gateway.ORDER_TYPE_BUY = 0
    gateway.COPY_TICKS_ALL = 0
    gateway.symbol_info.return_value = SimpleNamespace(
        visible=True,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        point=0.00001,
        digits=5,
        trade_stops_level=10,
        trade_freeze_level=0,
    )
    gateway.symbol_info_tick.return_value = SimpleNamespace(
        bid=1.15304,
        ask=1.15326,
        time=now,
        time_msc=now * 1000,
    )
    gateway.copy_ticks_range.return_value = [
        {
            "bid": 1.15308,
            "ask": 1.15322,
            "time": now,
            "time_msc": now * 1000,
        }
    ]

    with patch("mtdata.core.trading.orders._stdlib_time.time", return_value=now):
        result = build_trade_place_dry_run_preview(
            symbol="EURUSD",
            volume=0.1,
            order_type="BUY",
            pending=False,
            price=None,
            stop_loss=1.14,
            take_profit=1.16,
            gateway=gateway,
        )

    assert result["bid"] == 1.15308
    assert result["ask"] == 1.15322
    assert result["estimated_fill_price"] == 1.15322
    assert result["sl_tp_valid"] is True
    assert result["quote_context"]["quote_source"] == "mt5.copy_ticks_range"
    assert result["quote_context"]["quote_source_conflict"]["reason"] == (
        "equal_timestamp_bid_ask_disagreement"
    )
    assert result["quote_context"]["usable_for_live_trading"] is True


def test_build_trade_place_dry_run_preview_preserves_zero_symbol_digits():
    adapter = SimpleNamespace(
        ORDER_TYPE_BUY=0,
        order_calc_margin=MagicMock(return_value=123.45),
    )
    gateway = MagicMock()
    gateway.adapter = adapter
    gateway.ORDER_TYPE_BUY = 0
    gateway.symbol_info.return_value = SimpleNamespace(
        visible=True,
        volume_min=1.0,
        volume_max=100.0,
        volume_step=1.0,
        point=1.0,
        digits=0,
        trade_stops_level=10,
        trade_freeze_level=0,
    )
    gateway.symbol_info_tick.return_value = SimpleNamespace(
        bid=12344.6,
        ask=12345.4,
        time=datetime.now(timezone.utc).timestamp(),
    )
    gateway.account_info.return_value = SimpleNamespace(margin_free=1000.0)

    result = build_trade_place_dry_run_preview(
        symbol="US30",
        volume=1.0,
        order_type="BUY",
        pending=False,
        price=None,
        stop_loss=None,
        take_profit=None,
        gateway=gateway,
    )

    assert result["bid"] == 12345.0
    assert result["ask"] == 12345.0
    assert result["estimated_fill_price"] == 12345.0


def test_run_trade_place_rejects_contradictory_sl_tp_safety_settings():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TradePlaceRequest(
            symbol="EURUSD",
            volume=0.1,
            order_type="BUY",
            stop_loss=1.08,
            take_profit=1.12,
            auto_close_on_sl_tp_fail=False,
            dry_run=False,
        )


def test_run_trade_place_auto_close_uses_candidate_ticket_when_primary_is_missing():
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1.08,
        take_profit=1.12,
        dry_run=False,
    )
    close_calls = []

    def close_positions(**kwargs):
        close_calls.append(kwargs)
        return {"success": True, "closed_count": 1}

    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=lambda **kwargs: {
            "retcode": 10009,
            "sl_tp_result": {"status": "failed", "requested": {"sl": 1.08, "tp": 1.12}},
            "position_ticket_candidates": [111, 222],
        },
        place_pending_order=lambda **kwargs: {"success": True, "path": "pending"},
        close_positions=close_positions,
        safe_int_ticket=lambda value: value,
    )

    assert close_calls and close_calls[0]["ticket"] == 111
    assert result["auto_close_on_sl_tp_fail"] is True
    assert result["protection_status"] == "auto_closed_after_sl_tp_fail"
    assert result["auto_close_result"]["closed_count"] == 1


def test_run_trade_place_without_any_ticket_guides_to_trade_get_open():
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1.08,
        take_profit=1.12,
        dry_run=False,
    )

    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=lambda **kwargs: {
            "retcode": 10009,
            "sl_tp_result": {"status": "failed", "requested": {"sl": 1.08, "tp": 1.12}},
        },
        place_pending_order=lambda **kwargs: {"success": True, "path": "pending"},
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
    )

    assert result["protection_status"] == "unprotected_position"
    assert any("trade_get_open" in str(w) for w in result.get("warnings", []))
    assert not any("trade_modify now" in str(w) for w in result.get("warnings", []))


# ---------------------------------------------------------------------------
# _safe_float_attr tests
# ---------------------------------------------------------------------------

class TestSafeFloatAttr:
    def test_normal_float_attribute(self):
        obj = SimpleNamespace(price=1.2345)
        assert _safe_float_attr(obj, "price") == 1.2345

    def test_int_attribute_coerced(self):
        obj = SimpleNamespace(volume=100)
        assert _safe_float_attr(obj, "volume") == 100.0

    def test_string_numeric_coerced(self):
        obj = SimpleNamespace(value="3.14")
        assert _safe_float_attr(obj, "value") == 3.14

    def test_missing_attribute_returns_default(self):
        obj = SimpleNamespace()
        assert _safe_float_attr(obj, "price") == 0.0
        assert _safe_float_attr(obj, "price", -1.0) == -1.0

    def test_none_attribute_returns_default(self):
        obj = SimpleNamespace(profit=None)
        assert _safe_float_attr(obj, "profit") == 0.0

    def test_bool_attribute_returns_default(self):
        obj = SimpleNamespace(flag=True)
        assert _safe_float_attr(obj, "flag") == 0.0

    def test_nan_returns_default(self):
        obj = SimpleNamespace(price=float("nan"))
        assert _safe_float_attr(obj, "price") == 0.0

    def test_inf_returns_default(self):
        obj = SimpleNamespace(price=float("inf"))
        assert _safe_float_attr(obj, "price") == 0.0

    def test_negative_inf_returns_default(self):
        obj = SimpleNamespace(price=float("-inf"))
        assert _safe_float_attr(obj, "price") == 0.0

    def test_non_numeric_string_returns_default(self):
        obj = SimpleNamespace(price="not_a_number")
        assert _safe_float_attr(obj, "price") == 0.0

    def test_zero_is_valid(self):
        obj = SimpleNamespace(profit=0.0)
        assert _safe_float_attr(obj, "profit") == 0.0

    def test_negative_is_valid(self):
        obj = SimpleNamespace(profit=-42.5)
        assert _safe_float_attr(obj, "profit") == -42.5

    def test_getattr_exception_returns_default(self):
        class Broken:
            def __getattr__(self, name):
                raise RuntimeError("boom")
        assert _safe_float_attr(Broken(), "price") == 0.0

    def test_custom_default(self):
        obj = SimpleNamespace()
        assert _safe_float_attr(obj, "bid", 99.0) == 99.0
