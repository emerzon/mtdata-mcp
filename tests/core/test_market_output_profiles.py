from mtdata.core._mcp_tools import shape_public_tool_output


def _ticker_payload() -> dict:
    return {
        "success": True,
        "symbol": "EURUSD",
        "type": "quote",
        "bid": 1.1,
        "ask": 1.1002,
        "mid": 1.1001,
        "spread": 0.0002,
        "spread_points": 2,
        "spread_valid": True,
        "spread_quality": "two_sided",
        "price_precision": 5,
        "point": 0.00001,
        "units": {"bid": "absolute_price", "spread_points": "broker_points"},
        "quote_as_of": "2026-08-29T11:00:00Z",
        "time": "2026-08-29T11:00:00Z",
        "time_epoch": 1788001200.0,
        "timezone": "UTC",
        "freshness": "live, tick just now",
        "freshness_state": "live",
        "data_age_seconds": 0.1,
        "data_stale": False,
        "freshness_reason": "within_threshold",
        "usable_for_live_trading": True,
        "usable_for_live_trading_basis": "fresh_two_sided_quote",
        "source": {
            "provider": "mt5",
            "broker_company": "Example Broker",
            "server": "Demo",
        },
        "meta": {"diagnostics": {"query_latency_ms": 2.3}},
    }


def test_compact_ticker_keeps_quote_and_one_execution_gate() -> None:
    result = shape_public_tool_output(
        _ticker_payload(),
        tool_name="market_ticker",
        detail="compact",
    )

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "bid": 1.1,
        "ask": 1.1002,
        "mid": 1.1001,
        "spread": 0.0002,
        "spread_points": 2,
        "quote_as_of": "2026-08-29T11:00:00Z",
        "data_age_seconds": 0.1,
        "usable_for_live_trading": True,
        "source": {"provider": "mt5"},
    }


def test_compact_ticker_turns_stale_telemetry_into_one_warning() -> None:
    payload = _ticker_payload()
    payload.update(
        {
            "data_stale": True,
            "freshness_state": "stale",
            "freshness_reason": "age_exceeds_threshold",
            "usable_for_live_trading": False,
            "warning": "Quote is too old for live execution.",
        }
    )

    result = shape_public_tool_output(
        payload,
        tool_name="market_ticker",
        detail="compact",
    )

    assert result["usable_for_live_trading"] is False
    assert "freshness" not in result
    assert "data_stale" not in result
    assert result["warnings"] == [
        {
            "code": "data_stale",
            "scope": "market_ticker",
            "message": "Quote is too old for live execution.",
            "data_as_of": "2026-08-29T11:00:00Z",
            "age_seconds": 0.1,
        },
    ]


def test_market_snapshot_prunes_only_nested_quote_telemetry() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "snapshot": {
            "bid": 1.1,
            "ask": 1.2,
            "freshness_state": "live",
            "data_stale": False,
            "usable_for_live_trading": True,
            "timezone": "UTC",
            "regime": {"type": "trend", "confidence": 0.8},
            "status": {"timezone": "America/New_York", "status": "open"},
        },
    }

    result = shape_public_tool_output(
        payload,
        tool_name="market_snapshot",
        detail="compact",
    )

    snapshot = result["snapshot"]
    assert "freshness_state" not in snapshot
    assert "data_stale" not in snapshot
    assert snapshot["regime"]["type"] == "trend"
    assert snapshot["status"]["timezone"] == "America/New_York"


def test_full_ticker_consolidates_observation_metadata() -> None:
    result = shape_public_tool_output(
        _ticker_payload(),
        tool_name="market_ticker",
        detail="full",
    )

    assert "source" not in result
    assert "quote_as_of" not in result
    assert "freshness_state" not in result
    assert "units" not in result
    assert result["meta"]["source"]["provider"] == "mt5"
    assert result["meta"]["time"]["data_as_of"] == "2026-08-29T11:00:00Z"
    assert result["meta"]["freshness"][0]["status"] == "live"
    assert result["meta"]["units"]["bid"] == "absolute_price"
