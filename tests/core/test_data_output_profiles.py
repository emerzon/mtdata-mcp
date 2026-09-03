from mtdata.core._mcp_tools import shape_public_tool_output


def _candle_payload() -> dict:
    return {
        "success": True,
        "symbol": "BTCUSD",
        "timeframe": "M15",
        "data": [
            {
                "time": "2026-08-29T03:45Z",
                "open": 10,
                "close": 11,
                "bar_state": "closed",
                "gap_before": None,
            },
            {
                "time": "2026-08-29T04:00Z",
                "open": 11,
                "close": 12,
                "bar_state": "closed",
                "gap_before": "gap_seconds=3600",
            },
        ],
        "count": 2,
        "row_key": "data",
        "requested_limit": 2,
        "limit_satisfied": True,
        "forming_candle_status": "skipped",
        "indicator_columns": ["ema_20"],
        "indicators_spec": "ema(20)",
        "indicator_engine": {"effective_backend": "test"},
        "indicator_input": "raw_ohlcv",
        "indicator_warmup_bars": 100,
        "history_bars_fetched": 102,
        "processing_pipeline": ["fetch_ohlcv", "indicators"],
        "data_window": {"start": "2026-08-29T03:45Z", "end": "2026-08-29T04:00Z"},
        "as_of": "2026-08-29T04:24:00Z",
        "as_of_basis": "retrieval_time",
        "data_as_of": "2026-08-29T04:15:00Z",
        "data_as_of_basis": "completed_bar_close",
        "timestamp_format": "iso_utc",
        "timezone": "UTC",
        "freshness": "fresh, bar 9m ago",
        "data_age_seconds": 540,
        "data_stale": False,
        "history_policy_ok": True,
        "freshness_reason": "clock_skew_within_tolerance",
        "session_gaps": [
            {
                "from": "2026-08-29T02:45Z",
                "to": "2026-08-29T03:45Z",
                "missing_bars_est": 3,
                "context": "session break",
            }
        ],
        "source": {
            "provider": "mt5",
            "broker_company": "Example Broker",
            "server": "Demo",
            "source_context_id": "secret-ish-context",
            "context_available": True,
        },
        "units": {"tick_volume": "bid_update_count"},
    }


def test_compact_candles_omit_nominal_metadata_and_row_diagnostics() -> None:
    result = shape_public_tool_output(
        _candle_payload(),
        tool_name="data_fetch_candles",
        detail="compact",
    )

    assert set(result) == {
        "success",
        "symbol",
        "timeframe",
        "data",
        "data_as_of",
        "data_as_of_basis",
        "forming_candle_status",
        "indicator_input",
        "limit_satisfied",
        "timestamp_format",
        "source",
        "warnings",
    }
    assert result["forming_candle_status"] == "skipped"
    assert result["limit_satisfied"] is True
    assert result["data_as_of"] == "2026-08-29T04:15:00Z"
    assert result["data_as_of_basis"] == "completed_bar_close"
    assert result["timestamp_format"] == "iso_utc"
    assert result["source"] == {"provider": "mt5"}
    assert all("bar_state" not in row and "gap_before" not in row for row in result["data"])
    assert result["warnings"] == [
        {
            "code": "session_gap",
            "scope": "candles",
            "message": "The returned series contains an expected session or feed gap.",
            "count": 1,
            "first_from": "2026-08-29T02:45Z",
            "first_to": "2026-08-29T03:45Z",
            "missing_bars_est": 3,
            "context": "session break",
        }
    ]


def test_compact_candles_emit_one_actionable_stale_warning() -> None:
    payload = _candle_payload()
    payload["session_gaps"] = []
    payload["data_stale"] = True
    payload["history_policy_ok"] = False
    payload["freshness_reason"] = "latest_quote_stale"

    result = shape_public_tool_output(
        payload,
        tool_name="data_fetch_candles",
        detail="compact",
    )

    assert "freshness" not in result
    assert "data_stale" not in result
    assert result["warnings"] == [
        {
            "code": "data_stale",
            "scope": "candles",
            "message": "The latest data is outside the expected freshness window.",
            "data_as_of": "2026-08-29T04:15:00Z",
            "age_seconds": 540,
        }
    ]


def test_compact_candles_keep_skipped_forming_bar_hint() -> None:
    payload = _candle_payload()
    payload["session_gaps"] = []
    payload["hint"] = "Set include_incomplete=true to include the forming candle."

    result = shape_public_tool_output(
        payload,
        tool_name="data_fetch_candles",
        detail="compact",
    )

    assert result["forming_candle_status"] == "skipped"
    assert result["hint"] == (
        "Set include_incomplete=true to include the forming candle."
    )


def test_full_candles_consolidate_metadata_sections() -> None:
    result = shape_public_tool_output(
        _candle_payload(),
        tool_name="data_fetch_candles",
        detail="full",
    )

    assert "source" not in result
    assert "freshness" not in result
    assert "data_as_of" not in result
    assert "indicator_columns" not in result
    assert result["meta"]["source"]["provider"] == "mt5"
    assert result["meta"]["time"]["data_as_of"] == "2026-08-29T04:15:00Z"
    assert result["meta"]["freshness"][0]["status"] == "fresh"
    assert result["meta"]["processing"]["indicators"]["columns"] == ["ema_20"]
    assert result["meta"]["quality"]["session_gaps"][0]["missing_bars_est"] == 3


def test_compact_ticks_empty_closed_range_does_not_claim_data_is_shown() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "empty": True,
        "empty_reason": "market_closed_weekend",
        "count": 0,
        "data": [],
        "pagination": {"returned": 0, "has_more": False},
        "market_status": "closed",
        "market_status_reason": "weekend",
        "data_stale": False,
        "source": {"provider": "mt5", "server": "Demo"},
    }

    result = shape_public_tool_output(
        payload,
        tool_name="data_fetch_ticks",
        detail="compact",
    )

    assert result["empty"] is True
    assert result["warnings"][0]["code"] == "market_closed"
    assert "no ticks were returned" in result["warnings"][0]["message"]
    assert "latest completed data is shown" not in result["warnings"][0]["message"]


def test_compact_ticks_omit_healthy_freshness_and_keep_provider_only() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "count": 1,
        "data": [{"time": "2026-08-29T04:00:00Z", "bid": 1.1, "ask": 1.2}],
        "freshness": "fresh",
        "freshness_state": "live",
        "data_age_seconds": 0.1,
        "data_stale": False,
        "usable_for_live_trading": True,
        "timestamp_format": "iso_utc",
        "timezone": "UTC",
        "source": {"provider": "mt5", "server": "Demo"},
        "units": {"bid": "absolute_price"},
    }

    result = shape_public_tool_output(
        payload,
        tool_name="data_fetch_ticks",
        detail="compact",
    )

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "data": payload["data"],
        "source": {"provider": "mt5"},
    }


def test_compact_ticks_keep_incomplete_spread_sample_percentage() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "data": [{"time": "2026-08-29T04:00:00Z", "bid": 1.1, "ask": 1.1}],
        "quote_completeness_pct": 100.0,
        "valid_spread_sample_pct": 66.67,
        "source": {"provider": "mt5"},
    }

    result = shape_public_tool_output(
        payload,
        tool_name="data_fetch_ticks",
        detail="compact",
    )

    assert result["quote_completeness_pct"] == 100.0
    assert result["valid_spread_sample_pct"] == 66.67


def test_compact_ticks_omit_complete_spread_sample_percentage() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "data": [{"time": "2026-08-29T04:00:00Z", "bid": 1.1, "ask": 1.2}],
        "quote_completeness_pct": 100.0,
        "valid_spread_sample_pct": 100.0,
        "source": {"provider": "mt5"},
    }

    result = shape_public_tool_output(
        payload,
        tool_name="data_fetch_ticks",
        detail="compact",
    )

    assert "valid_spread_sample_pct" not in result
    assert result["quote_completeness_pct"] == 100.0


def test_compact_symbol_description_does_not_mislabel_quote_conflict_as_stale() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "details": {
            "time": "2026-08-30T22:00:00Z",
            "data_stale": False,
            "usable_for_live_trading": False,
            "quote_source_conflict": {
                "reason": "equal_timestamp_bid_ask_disagreement"
            },
            "warning": "Quote sources conflict.",
        },
    }

    result = shape_public_tool_output(
        payload,
        tool_name="symbols_describe",
        detail="compact",
    )

    assert result["warnings"] == [
        {
            "code": "quote_source_conflict",
            "scope": "symbols_describe",
            "message": "Quote sources conflict.",
            "data_as_of": "2026-08-30T22:00:00Z",
        }
    ]


def test_compact_symbol_description_uses_general_code_for_unknown_live_blocker() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "details": {
            "data_stale": False,
            "usable_for_live_trading": False,
            "warning": "Quote cannot be used for live execution.",
        },
    }

    result = shape_public_tool_output(
        payload,
        tool_name="symbols_describe",
        detail="compact",
    )

    assert result["warnings"][0]["code"] == "quote_not_live"


def test_compact_symbol_description_keeps_broker_lot_units() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "details": {
            "trade_contract_size": 100_000.0,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "lot_definition": (
                "1 broker lot equals trade_contract_size contract units."
            ),
            "units": {
                "trade_contract_size": "contract_units_per_broker_lot",
                "volume_min": "broker_lot",
                "volume_max": "broker_lot",
                "volume_step": "broker_lot",
            },
        },
    }

    result = shape_public_tool_output(
        payload,
        tool_name="symbols_describe",
        detail="compact",
    )

    assert result["details"]["lot_definition"].startswith("1 broker lot")
    assert result["details"]["units"] == payload["details"]["units"]
