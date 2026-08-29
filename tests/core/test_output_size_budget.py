import json

from mtdata.core._mcp_tools import shape_public_tool_output


def _large_candle_payload() -> dict:
    rows = []
    for index in range(160):
        rows.append(
            {
                "time": f"2026-08-29T{index // 4:02d}:{(index % 4) * 15:02d}Z",
                "open": 79000 + index,
                "high": 79010 + index,
                "low": 78990 + index,
                "close": 79005 + index,
                "tick_volume": 1000 + index,
                "ema_20": 79001 + index,
                "ema_50": 78980 + index,
                "bar_state": "closed",
                "gap_before": None,
            }
        )
    return {
        "success": True,
        "symbol": "BTCUSD",
        "timeframe": "M15",
        "data": rows,
        "count": 160,
        "row_key": "data",
        "requested_limit": 160,
        "limit_satisfied": True,
        "forming_candle_status": "skipped",
        "hint": "Set include_incomplete=true to include the forming candle.",
        "indicator_columns": ["ema_20", "ema_50"],
        "indicators_spec": "ema(20),ema(50)",
        "indicator_engine": {
            "pandas_ta": {"name": "pandas-ta-classic", "version": "0.6.52"},
            "talib": {"available": True, "version": "0.7.1"},
            "effective_backend": "pandas-ta-classic+talib",
        },
        "indicator_rounding": {
            "price_columns": ["ema_20", "ema_50"],
            "price_precision": 2,
            "policy": "symbol_price_precision",
        },
        "processing_pipeline": ["fetch_ohlcv", "indicators"],
        "data_window": {
            "start": "2026-08-27T12:15Z",
            "end": "2026-08-29T04:00Z",
            "latest_bar_complete": True,
        },
        "as_of": "2026-08-29T04:24:41Z",
        "data_as_of": "2026-08-29T04:15:00Z",
        "data_as_of_basis": "completed_bar_close",
        "timestamp_format": "iso_utc",
        "timezone": "UTC",
        "freshness": "fresh, bar 9m 41s ago",
        "data_age_seconds": 581,
        "data_stale": False,
        "history_policy_ok": True,
        "latest_quote_stale": False,
        "latest_quote_age_seconds": 0,
        "freshness_reason": "clock_skew_within_tolerance",
        "indicator_warmup_bars": 1250,
        "history_bars_fetched": 1411,
        "volume_semantics": "tick_volume_is_bid_update_count_not_lots",
        "source": {
            "provider": "mt5",
            "broker_company": "Example Broker",
            "server": "Demo",
            "source_context_id": "context-id",
            "context_available": True,
        },
        "units": {"tick_volume": "bid_update_count"},
    }


def test_compact_candle_output_stays_within_metadata_size_budget() -> None:
    raw = _large_candle_payload()
    compact = shape_public_tool_output(
        raw,
        tool_name="data_fetch_candles",
        detail="compact",
    )

    raw_size = len(json.dumps(raw, separators=(",", ":")))
    compact_size = len(json.dumps(compact, separators=(",", ":")))
    assert compact_size <= raw_size * 0.75
    assert set(compact) == {"success", "symbol", "timeframe", "data", "source"}
    assert all("bar_state" not in row and "gap_before" not in row for row in compact["data"])


def test_targeted_rich_selection_does_not_restore_full_metadata() -> None:
    raw = _large_candle_payload()
    selected = shape_public_tool_output(
        raw,
        tool_name="data_fetch_candles",
        detail="compact",
        output_fields="indicator_engine.effective_backend",
    )

    assert selected == {
        "success": True,
        "symbol": "BTCUSD",
        "timeframe": "M15",
        "indicator_engine": {
            "effective_backend": "pandas-ta-classic+talib"
        },
        "source": {"provider": "mt5"},
    }
