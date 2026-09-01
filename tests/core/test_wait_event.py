from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from mtdata.core.data import wait_events
from mtdata.core.data.requests import WaitEventRequest


def test_wait_event_request_defaults_watch_for_to_empty_list() -> None:
    request = WaitEventRequest(timeframe="M1")

    assert request.watch_for == []
    assert request.end_on == []
    assert request.max_wait_seconds == 61.0


def test_wait_event_request_default_watch_lists_are_independent() -> None:
    first = WaitEventRequest(timeframe="M1")
    second = WaitEventRequest(timeframe="M1")

    assert first.watch_for is not second.watch_for


def test_wait_event_request_rejects_combined_stopping_modes() -> None:
    with pytest.raises(
        ValidationError,
        match="Do not combine timeframe with max_wait_seconds",
    ):
        WaitEventRequest(timeframe="M1", max_wait_seconds=30)


def test_wait_event_request_rejects_non_positive_poll_interval() -> None:
    with pytest.raises(ValueError, match="Input should be greater than or equal to 0.1"):
        WaitEventRequest(
            watch_for=[{"type": "order_created"}],
            poll_interval_seconds=0.0,
        )


def test_wait_event_request_parses_market_event_specs() -> None:
    request = WaitEventRequest.model_validate(
        {
            "symbol": "EURUSD",
            "timeframe": "M5",
            "watch_for": [
                {
                    "type": "price_change",
                    "window": {"kind": "ticks", "value": 5},
                    "baseline_window": {"kind": "ticks", "value": 20},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 3.0,
                },
                {
                    "type": "volume_spike",
                    "window": {"kind": "minutes", "value": 3},
                    "baseline_window": {"kind": "minutes", "value": 30},
                    "source": "tick_count",
                    "threshold_mode": "zscore",
                    "threshold_value": 2.5,
                },
                {
                    "type": "tick_count_spike",
                    "window": {"kind": "minutes", "value": 2},
                    "baseline_window": {"kind": "minutes", "value": 20},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 3.0,
                },
                {
                    "type": "spread_spike",
                    "window": {"kind": "ticks", "value": 3},
                    "baseline_window": {"kind": "ticks", "value": 12},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 2.0,
                },
                {
                    "type": "tick_count_drought",
                    "window": {"kind": "minutes", "value": 2},
                    "baseline_window": {"kind": "minutes", "value": 20},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 0.5,
                },
                {
                    "type": "range_expansion",
                    "window": {"kind": "ticks", "value": 4},
                    "baseline_window": {"kind": "ticks", "value": 16},
                    "price_source": "mid",
                    "threshold_mode": "zscore",
                    "threshold_value": 2.0,
                },
                {
                    "type": "price_touch_level",
                    "level": 100.5,
                    "direction": "up",
                    "tolerance": 0.1,
                },
                {
                    "type": "price_break_level",
                    "level": 101.0,
                    "direction": "up",
                    "tolerance": 0.05,
                    "confirm_ticks": 2,
                },
                {
                    "type": "price_enter_zone",
                    "lower": 99.5,
                    "upper": 100.5,
                    "direction": "either",
                },
                {
                    "type": "pending_near_fill",
                    "distance": 0.2,
                },
                {
                    "type": "stop_threat",
                    "distance": 0.15,
                },
            ],
            "end_on": [{"type": "candle_close", "timeframe": "M5"}],
        }
    )

    assert len(request.watch_for) == 11
    assert request.watch_for[0].type == "price_change"
    assert request.watch_for[1].type == "volume_spike"
    assert request.watch_for[2].type == "tick_count_spike"
    assert request.watch_for[3].type == "spread_spike"
    assert request.watch_for[4].type == "tick_count_drought"
    assert request.watch_for[5].type == "range_expansion"
    assert request.watch_for[6].type == "price_touch_level"
    assert request.watch_for[7].type == "price_break_level"
    assert request.watch_for[8].type == "price_enter_zone"
    assert request.watch_for[9].type == "pending_near_fill"
    assert request.watch_for[10].type == "stop_threat"
    assert request.end_on[0].type == "candle_close"


def test_wait_event_request_rejects_unknown_event_names_with_standard_discriminator_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        WaitEventRequest.model_validate(
            {
                "symbol": "EURUSD",
                "watch_for": [
                    {"type": "price_level_touch", "level": 1.1454, "tolerance": 0.0002},
                ],
            }
        )

    error = exc_info.value.errors()[0]
    assert error["type"] == "union_tag_invalid"
    assert error["ctx"]["tag"] == "price_level_touch"


def test_wait_event_request_rejects_invalid_price_direction_with_standard_literal_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        WaitEventRequest.model_validate(
            {
                "symbol": "EURUSD",
                "watch_for": [
                    {"type": "price_touch_level", "level": 1.1454, "direction": "above"},
                ],
            }
        )

    error = exc_info.value.errors()[0]
    assert error["type"] == "literal_error"
    assert error["loc"] == ("watch_for", 0, "price_touch_level", "direction")
    assert error["msg"] == "Input should be 'up', 'down' or 'either'"


def test_wait_event_request_rejects_invalid_account_side_with_standard_literal_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        WaitEventRequest.model_validate(
            {
                "symbol": "EURUSD",
                "watch_for": [
                    {"type": "position_opened", "symbol": "EURUSD", "side": "long"},
                ],
            }
        )

    error = exc_info.value.errors()[0]
    assert error["type"] == "literal_error"
    assert error["loc"] == ("watch_for", 0, "position_opened", "side")
    assert error["msg"] == "Input should be 'buy' or 'sell'"


def test_wait_event_request_accepts_canonical_account_side_values() -> None:
    request = WaitEventRequest.model_validate(
        {
            "symbol": "EURUSD",
            "side": "buy",
            "max_wait_seconds": 5,
            "watch_for": [
                {"type": "position_opened", "symbol": "EURUSD", "side": "sell"},
            ],
        }
    )

    assert request.side == "buy"
    assert request.watch_for[0].side == "sell"


def test_wait_event_request_rejects_request_level_invalid_side_with_standard_literal_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        WaitEventRequest.model_validate(
            {
                "symbol": "EURUSD",
                "side": "long",
                "watch_for": [
                    {"type": "position_opened"},
                ],
            }
        )

    error = exc_info.value.errors()[0]
    assert error["type"] == "literal_error"
    assert error["loc"] == ("side",)
    assert error["msg"] == "Input should be 'buy' or 'sell'"


def test_wait_event_request_rejects_candle_close_in_watch_for_with_standard_discriminator_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        WaitEventRequest.model_validate(
            {
                "symbol": "EURUSD",
                "watch_for": [{"type": "candle_close", "timeframe": "M15"}],
            }
        )

    error = exc_info.value.errors()[0]
    assert error["type"] == "union_tag_invalid"
    assert error["ctx"]["tag"] == "candle_close"


def test_wait_event_request_rejects_string_shorthands_with_standard_model_errors() -> None:
    with pytest.raises(ValidationError) as exc_info:
        WaitEventRequest.model_validate(
            {
                "symbol": "EURUSD",
                "watch_for": ["order_filled", "new_bar"],
                "end_on": ["candle_close"],
            }
        )

    errors = exc_info.value.errors()
    assert [error["type"] for error in errors] == [
        "model_attributes_type",
        "model_attributes_type",
        "model_type",
    ]


def test_wait_event_request_rejects_non_boundary_events_in_end_on_with_standard_literal_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        WaitEventRequest.model_validate(
            {
                "symbol": "EURUSD",
                "end_on": [{"type": "order_created", "symbol": "EURUSD"}],
            }
        )

    error = exc_info.value.errors()[0]
    assert error["type"] == "literal_error"
    assert error["loc"] == ("end_on", 0, "type")
    assert error["msg"] == "Input should be 'candle_close'"


def test_wait_event_request_rejects_end_on_without_top_level_timeframe() -> None:
    with pytest.raises(ValidationError, match="end_on requires a top-level timeframe"):
        WaitEventRequest.model_validate(
            {
                "symbol": "EURUSD",
                "end_on": [{"type": "candle_close", "timeframe": "M15"}],
            }
        )


@patch("mtdata.core.data.create_mt5_gateway", return_value=object())
@patch("mtdata.core.data._compact_wait_event_public_result", side_effect=lambda result, **_: result)
@patch("mtdata.core.data.run_wait_event", return_value={"success": True})
def test_wait_event_accepts_explicit_watch_for_objects(
    mock_run_wait,
    _mock_compact,
    _mock_gateway,
) -> None:
    result = _raw_wait_event()(
        symbol="BTCUSD",
        timeframe="M15",
        watch_for=[
            {"type": "price_touch_level", "symbol": "BTCUSD", "level": 100000.0},
            {"type": "price_break_level", "symbol": "BTCUSD", "level": 100500.0},
            {"type": "price_enter_zone", "symbol": "BTCUSD", "lower": 99500.0, "upper": 100500.0},
            {"type": "pending_near_fill", "symbol": "BTCUSD"},
            {"type": "stop_threat", "symbol": "BTCUSD"},
        ],
    )

    assert result == {"success": True}
    request = mock_run_wait.call_args.args[0]
    assert [item.type for item in request.watch_for] == [
        "price_touch_level",
        "price_break_level",
        "price_enter_zone",
        "pending_near_fill",
        "stop_threat",
    ]


def test_wait_event_request_rejects_watch_for_string_with_standard_model_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        WaitEventRequest.model_validate(
            {
                "symbol": "BTCUSD",
                "watch_for": ["price_touch_level"],
            }
        )

    error = exc_info.value.errors()[0]
    assert error["type"] == "model_attributes_type"
    assert error["loc"] == ("watch_for", 0)


def _raw_wait_event():
    from mtdata.core.data import wait_event

    return wait_event.__wrapped__


@patch("mtdata.core.data.create_mt5_gateway", return_value=object())
@patch("mtdata.core.data._compact_wait_event_public_result", side_effect=lambda result, **_: result)
@patch("mtdata.core.data.run_wait_event", return_value={"success": True})
def test_wait_event_prefers_public_symbol_name(mock_run_wait, _mock_compact, _mock_gateway) -> None:
    result = _raw_wait_event()(symbol="EURUSD", timeframe="M1", watch_for=[])

    assert result == {"success": True}
    request = mock_run_wait.call_args.args[0]
    assert request.symbol == "EURUSD"


@patch("mtdata.core.data.create_mt5_gateway", return_value=object())
@patch("mtdata.core.data._compact_wait_event_public_result", side_effect=lambda result, **_: result)
@patch("mtdata.core.data.run_wait_event", return_value={"success": True})
def test_wait_event_builds_default_watchers_for_normalized_basket(
    mock_run_wait,
    _mock_compact,
    _mock_gateway,
) -> None:
    result = _raw_wait_event()(symbols=["eurusd", " gbpusd "], timeframe="M5")

    assert result == {"success": True}
    request = mock_run_wait.call_args.args[0]
    assert request.symbol is None
    assert request.symbols == ["EURUSD", "GBPUSD"]
    assert list(request.watch_for or []) == []
    assert request.max_wait_seconds == 301.0


@patch("mtdata.core.data.create_mt5_gateway", return_value=object())
@patch("mtdata.core.data._compact_wait_event_public_result", side_effect=lambda result, **_: result)
@patch("mtdata.core.data.run_wait_event", return_value={"success": True})
def test_wait_event_symbol_less_timeframe_builds_boundary_only_request(
    mock_run_wait,
    _mock_compact,
    _mock_gateway,
) -> None:
    result = _raw_wait_event()(timeframe="H1")

    assert result == {"success": True}
    request = mock_run_wait.call_args.args[0]
    assert request.symbol is None
    assert request.symbols is None
    assert request.timeframe == "H1"
    assert request.max_wait_seconds == 3601.0
    assert request.watch_for == []
    assert [(item.type, item.timeframe) for item in request.end_on] == [
        ("candle_close", "H1")
    ]


def test_wait_event_public_signature_requires_timeframe_and_hides_timing_controls() -> None:
    raw = _raw_wait_event()

    with pytest.raises(TypeError, match="timeframe"):
        raw(symbol="BTCUSD")
    with pytest.raises(TypeError, match="max_wait_seconds"):
        raw(symbol="BTCUSD", timeframe="M1", max_wait_seconds=30)
    with pytest.raises(TypeError, match="poll_interval_seconds"):
        raw(symbol="BTCUSD", timeframe="M1", poll_interval_seconds=1)


@patch("mtdata.core.data.create_mt5_gateway", return_value=object())
@patch("mtdata.core.data._compact_wait_event_public_result", side_effect=lambda result, **_: result)
@patch("mtdata.core.data.run_wait_event", return_value={"success": True})
def test_wait_event_forwards_accept_preexisting(
    mock_run_wait,
    _mock_compact,
    _mock_gateway,
) -> None:
    result = _raw_wait_event()(
        symbol="BTCUSD",
        timeframe="M5",
        accept_preexisting=True,
        watch_for=[{"type": "price_touch_level", "level": 70000}],
    )

    assert result == {"success": True}
    assert mock_run_wait.call_args.args[0].accept_preexisting is True


@patch("mtdata.core.data.create_mt5_gateway", return_value=object())
@patch("mtdata.core.data.run_wait_event", return_value={"success": True})
def test_wait_event_rejects_symbol_with_symbols(
    mock_run_wait,
    _mock_gateway,
) -> None:
    result = _raw_wait_event()(
        symbol="BTCUSD",
        symbols=["EURUSD", "GBPUSD"],
        timeframe="M1",
    )

    assert result == {
        "error": "symbol and symbols cannot be combined.",
        "error_code": "wait_event_invalid_request",
    }
    mock_run_wait.assert_not_called()


@patch("mtdata.core.data.create_mt5_gateway", return_value=object())
@patch("mtdata.core.data.run_wait_event", return_value={"success": True})
def test_wait_event_rejects_conflicting_end_on_timeframe(
    mock_run_wait,
    _mock_gateway,
) -> None:
    result = _raw_wait_event()(
        symbol="BTCUSD",
        timeframe="M1",
        watch_for=[{"type": "order_filled", "symbol": "BTCUSD"}],
        end_on=[{"type": "candle_close", "timeframe": "M5"}],
    )

    assert result["error_code"] == "wait_event_invalid_request"
    assert "end_on timeframes must match the top-level timeframe" in result["error"]
    mock_run_wait.assert_not_called()


@patch("mtdata.core.data.create_mt5_gateway", return_value=object())
@patch("mtdata.core.data.run_wait_event", return_value={"success": True})
def test_wait_event_symbol_timeframe_infers_wait_budget(
    mock_run_wait,
    _mock_gateway,
) -> None:
    result = _raw_wait_event()(symbol="EURUSD", timeframe="M1")

    assert result["success"] is True
    mock_run_wait.assert_called_once()
    assert mock_run_wait.call_args.args[0].max_wait_seconds == 61.0


def test_wait_event_request_rejects_instrument_as_extra_field() -> None:
    with pytest.raises(ValidationError) as exc_info:
        WaitEventRequest.model_validate(
            {
                "instrument": "EURUSD",
                "watch_for": [{"type": "order_created"}],
            }
        )

    error = exc_info.value.errors()[0]
    assert error["type"] == "extra_forbidden"
    assert error["loc"] == ("instrument",)


def test_watch_for_description_enumerates_event_types():
    desc = WaitEventRequest.model_fields["watch_for"].description or ""
    for event_type in ("price_change", "volume_spike", "tp_hit", "sl_hit", "stop_threat"):
        assert event_type in desc
    assert "Example" in desc


def test_watch_for_documented_example_is_valid():
    example = [
        {
            "type": "price_change",
            "direction": "up",
            "threshold_mode": "fixed_pct",
            "threshold_value": 0.1,
        }
    ]
    req = WaitEventRequest(
        symbol="EURUSD",
        timeframe="M1",
        watch_for=example,
    )
    assert req.watch_for[0].type == "price_change"
