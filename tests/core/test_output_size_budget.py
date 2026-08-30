import json

from mtdata.core._mcp_tools import shape_public_tool_output


def _json_size(payload: dict) -> int:
    return len(json.dumps(payload, separators=(",", ":")))


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

    raw_size = _json_size(raw)
    compact_size = _json_size(compact)
    assert compact_size <= raw_size * 0.75
    assert set(compact) == {"success", "symbol", "timeframe", "data", "source"}
    assert all("bar_state" not in row and "gap_before" not in row for row in compact["data"])


def test_compact_model_inventory_keeps_ids_and_only_exception_details() -> None:
    raw = {
        "success": True,
        "detail": "compact",
        "count": 2,
        "total_models": 2,
        "count_by_method": {"theta": 2},
        "filters": {"method": None, "symbol": None, "timeframe": None},
        "expired_models_hint": "Use forecast_models_cleanup to remove expired models.",
        "show_all_hint": "Increase limit to list every stored model.",
        "models": [
            {
                "model_id": "theta/EURUSD_H1/one",
                "method": "theta",
                "adapter": "statsforecast",
                "data_scope": "EURUSD_H1",
                "created_at": "2026-08-28T10:00:00Z",
                "last_used_at": "2026-08-29T10:00:00Z",
                "age_seconds": 86400,
                "disk_size_bytes": 12345,
                "request_compatibility_status": "ready",
                "store_compatibility_status": "ready",
            },
            {
                "model_id": "theta/EURUSD_H1/two",
                "method": "theta",
                "adapter": "statsforecast",
                "data_scope": "EURUSD_H1",
                "created_at": "2026-08-27T10:00:00Z",
                "last_used_at": "2026-08-28T10:00:00Z",
                "age_seconds": 172800,
                "disk_size_bytes": 23456,
                "request_compatibility_status": "incompatible",
                "request_compatibility_reason": "horizon exceeds trained range",
                "supported_horizon": 12,
                "store_compatibility_status": "ready",
            },
        ],
        "pagination": {"total": 2, "returned": 2, "has_more": False},
        "source": {"provider": "local_model_store", "path": "C:/private/models"},
    }

    compact = shape_public_tool_output(
        raw,
        tool_name="forecast_models_list",
        detail="compact",
    )

    assert compact["models"] == [
        {"model_id": "theta/EURUSD_H1/one"},
        {
            "model_id": "theta/EURUSD_H1/two",
            "request_compatibility_status": "incompatible",
            "request_compatibility_reason": "horizon exceeds trained range",
            "supported_horizon": 12,
        },
    ]
    assert compact["source"] == {"provider": "local_model_store"}
    assert _json_size(compact) <= _json_size(raw) * 0.4


def test_compact_market_snapshot_drops_prose_but_keeps_execution_gates() -> None:
    warning = {
        "code": "data_stale",
        "scope": "market_snapshot.quote",
        "message": "Quote is too old for live execution.",
        "age_seconds": 900,
    }
    raw = {
        "success": True,
        "symbol": "EURUSD",
        "timeframe": "H1",
        "assembled_at": "2026-08-29T12:00:00Z",
        "sections_requested": ["quote", "status", "levels", "patterns"],
        "sections_summarized": ["quote", "status", "levels", "patterns"],
        "summary": "EURUSD snapshot; mid=1.1001; WARNING: quote is stale.",
        "warnings": [warning],
        "snapshot": {
            "bid": 1.1,
            "ask": 1.1002,
            "mid": 1.1001,
            "spread": 0.0002,
            "spread_points": 20,
            "spread_pips": 2,
            "spread_pct": 0.018,
            "warnings": [warning],
            "execution": {
                "usable_for_live_trading": False,
                "usable_for_live_trading_basis": "quote_age_exceeds_live_threshold",
                "status": "closed",
                "status_source": "mt5_heuristic",
                "status_confidence": "medium",
                "heuristic_note": "Long static explanation of the inference policy.",
                "is_tradable": True,
                "can_open_new_positions": False,
                "trade_mode_allows_opening": True,
                "tradability": {
                    "confidence": "high",
                    "means": "account and terminal permission only",
                },
                "reason": "market_closed",
            },
            "nearest_support": 1.09,
            "nearest_resistance": 1.11,
            "support_count": 3,
            "resistance_count": 4,
            "range_count": 1,
            "containing_range": {"low": 1.09, "high": 1.11, "score": 18.9},
            "score_basis": {
                "scale": "unbounded_nonnegative",
                "higher_is_stronger": True,
            },
            "levels_context": {
                "lookback_bars": 200,
                "structure_as_of": "2026-08-29T11:00:00Z",
                "scan_window": {"start": "2026-08-20", "end": "2026-08-29"},
                "input_bar_policy": "closed_bars_only",
                "current_price_source": "latest_quote_mid",
            },
            "latest_pattern_bias": "bullish",
            "pattern_is_signal": False,
            "pattern_usage": "information_only",
            "pattern_window_bars": 3,
            "pattern_scan_note": "Use patterns_detect for a wider historical scan.",
            "latest_match_score": 0.8,
            "latest_match_score_scale": "similarity_0_to_1",
        },
        "source": {"provider": "mt5", "server": "Demo"},
    }

    compact = shape_public_tool_output(
        raw,
        tool_name="market_snapshot",
        detail="compact",
    )

    assert compact["snapshot"]["execution"] == {
        "usable_for_live_trading": False,
        "status": "closed",
        "can_open_new_positions": False,
        "reason": "market_closed",
    }
    assert compact["snapshot"]["nearest_support"] == 1.09
    assert compact["snapshot"]["pattern_is_signal"] is False
    assert compact["warnings"] == [warning]
    assert "summary" not in compact
    assert "score_basis" not in compact["snapshot"]
    assert _json_size(compact) <= _json_size(raw) * 0.5


def test_compact_market_status_keeps_calendar_outcomes_not_clock_echoes() -> None:
    market_rows = []
    for venue, name in (("NYSE", "New York Stock Exchange"), ("LSE", "London Stock Exchange")):
        market_rows.append(
            {
                "venue": venue,
                "name": name,
                "status": "closed",
                "reason": "weekend",
                "local_time": "2026-08-29T08:00:00-04:00",
                "exchange_local_time": "2026-08-29T08:00:00-04:00",
                "display_time": "2026-08-29T12:00:00Z",
                "display_timezone": "UTC",
                "exchange_day_of_week": "Saturday",
                "next_open": "2026-08-31T09:30:00-04:00",
                "minutes_until_open": 2730,
                "message": f"{venue} is closed for the weekend.",
            }
        )
    raw = {
        "success": True,
        "mode": "equity_exchanges",
        "market_scope": "major_equity_exchanges",
        "scope_note": "This view covers major equity exchanges only.",
        "data_fetched_at": "2026-08-29T12:00:00Z",
        "timezone": "UTC",
        "timezone_display": "utc",
        "display_timezone": "UTC",
        "day_of_week": "Saturday",
        "day_of_week_basis": "exchange_local",
        "region": "all",
        "summary": "2 closed: NYSE, LSE",
        "markets_open": 0,
        "markets_closed": 2,
        "markets_pre_market": 0,
        "markets_after_hours": 0,
        "markets_lunch_break": 0,
        "closed_reason_counts": {"weekend": 2},
        "global_status": "weekend",
        "markets": market_rows,
        "source": {"provider": "mtdata_exchange_calendar"},
    }

    compact = shape_public_tool_output(
        raw,
        tool_name="market_status",
        detail="compact",
    )

    assert compact["global_status"] == "weekend"
    assert compact["markets"][0] == {
        "venue": "NYSE",
        "name": "New York Stock Exchange",
        "status": "closed",
        "next_open": "2026-08-31T09:30:00-04:00",
    }
    assert "summary" not in compact
    assert _json_size(compact) <= _json_size(raw) * 0.45


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
