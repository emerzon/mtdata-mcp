from mtdata.core._mcp_tools import shape_public_tool_output


def test_compact_analysis_omits_nominal_observation_and_request_echoes() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "result": {"statistic": 1.25, "p_value": 0.04},
        "data_window": {"start": "2026-01-01", "end": "2026-02-01"},
        "data_as_of": "2026-02-01T00:00:00Z",
        "data_age_seconds": 60,
        "data_stale": False,
        "freshness": "fresh",
        "query_applied": {"timeframe": "H1", "limit": 500},
        "processing_pipeline": ["fetch", "test"],
        "source": {"provider": "mt5", "server": "Demo"},
        "meta": {"diagnostics": {"latency_ms": 10}},
        "related_tools": ["cointegration_test"],
    }

    result = shape_public_tool_output(
        payload,
        tool_name="stationarity_test",
        detail="compact",
    )

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "result": {"statistic": 1.25, "p_value": 0.04},
        "data_window": payload["data_window"],
        "source": {"provider": "mt5"},
    }


def test_compact_analysis_keeps_only_non_nominal_freshness_warning() -> None:
    payload = {
        "success": True,
        "result": {"volatility": 0.2},
        "data_as_of": "2026-02-01T00:00:00Z",
        "data_age_seconds": 9000,
        "data_stale": True,
        "freshness_reason": "outside_policy",
    }

    result = shape_public_tool_output(
        payload,
        tool_name="forecast_volatility_estimate",
        detail="compact",
    )

    assert result["warnings"] == [
        {
            "code": "data_stale",
            "scope": "forecast_volatility_estimate",
            "message": "The latest data is outside the expected freshness window.",
            "data_as_of": "2026-02-01T00:00:00Z",
            "age_seconds": 9000,
        }
    ]


def test_catalog_compact_uses_exception_list_and_minimal_pagination() -> None:
    payload = {
        "success": True,
        "detail": "compact",
        "count": 2,
        "row_key": "methods",
        "catalog_source": "rebuilt",
        "methods": [
            {
                "method": "arima",
                "available": True,
                "category": "classical",
                "unavailable_reason": None,
            },
            {
                "method": "nbeatsx",
                "available": False,
                "category": "neural",
                "unavailable_reason": "Requires neuralforecast",
            },
        ],
        "pagination": {
            "total": 10,
            "returned": 2,
            "offset": 0,
            "limit": 2,
            "has_more": True,
            "more_available": 8,
        },
    }

    result = shape_public_tool_output(
        payload,
        tool_name="forecast_list_methods",
        detail="compact",
    )

    assert result == {
        "success": True,
        "methods": [
            {"method": "arima", "category": "classical"},
            {"method": "nbeatsx", "category": "neural"},
        ],
        "pagination": {"has_more": True, "total": 10, "next_offset": 2},
        "unavailable": [
            {"method": "nbeatsx", "reason": "Requires neuralforecast"}
        ],
    }


def test_catalog_compact_keeps_pagination_total_when_complete() -> None:
    payload = {
        "success": True,
        "tools": [{"name": "calendar", "category": "news"}],
        "pagination": {
            "total": 2,
            "returned": 2,
            "offset": 0,
            "limit": 20,
            "has_more": False,
            "more_available": 0,
        },
    }

    result = shape_public_tool_output(
        payload,
        tool_name="tools_list",
        detail="compact",
    )

    assert result["pagination"] == {"total": 2}


def test_denoise_catalog_omits_default_filters_and_derived_flags() -> None:
    payload = {
        "success": True,
        "detail": "compact",
        "available_only": False,
        "causality": None,
        "core_only": False,
        "columns": [
            "method",
            "available",
            "causality",
            "requires_causality_opt_in",
        ],
        "methods": [
            {
                "method": "kalman",
                "available": True,
                "causality": ["causal", "zero_phase"],
                "requires_causality_opt_in": False,
            }
        ],
        "describe_hint": "Use denoise_describe(method) for defaults.",
        "list_all_hint": "Pass limit=20 to list every method.",
    }

    result = shape_public_tool_output(
        payload,
        tool_name="denoise_list_methods",
        detail="compact",
    )

    assert result == {
        "success": True,
        "methods": [
            {
                "method": "kalman",
                "causality": ["causal", "zero_phase"],
            }
        ],
    }


def test_compact_warnings_deduplicate_same_message_across_scopes() -> None:
    payload = {
        "success": True,
        "items": [],
        "warnings": [
            "The sample is too small for a reliable estimate.",
            {
                "code": "low_sample",
                "scope": "seasonality_detect",
                "message": "  The sample is too small for a reliable estimate. ",
                "samples": 12,
            },
        ],
    }

    result = shape_public_tool_output(
        payload,
        tool_name="seasonality_detect",
        detail="compact",
    )

    assert result["warnings"] == [
        {
            "code": "low_sample",
            "scope": "seasonality_detect",
            "message": "The sample is too small for a reliable estimate.",
            "samples": 12,
        }
    ]


def test_compact_volatility_term_structure_keeps_replay_anchor() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "timeframe": "H1",
        "history_policy": "completed_bars_only",
        "last_bar_open": "2026-08-28T11:00:00Z",
        "data_as_of": "2026-08-28T12:00:00Z",
        "data_as_of_basis": "completed_bar_close",
        "analysis_window": {
            "requested_as_of": "2026-08-28T12:00:00Z",
            "resolved_as_of": "2026-08-28T12:00:00Z",
            "period_start": "2026-08-18T04:00:00Z",
            "period_end": "2026-08-28T11:00:00Z",
            "timezone": "UTC",
            "bars_used": 200,
        },
        "items": [{"horizon": 1, "volatility": 0.1}],
    }

    result = shape_public_tool_output(
        payload,
        tool_name="volatility_term_structure",
        detail="compact",
    )

    assert result["requested_as_of"] == "2026-08-28T12:00:00Z"
    assert result["data_as_of"] == "2026-08-28T12:00:00Z"
    assert result["data_as_of_basis"] == "completed_bar_close"
    assert result["timezone"] == "UTC"
    assert result["history_policy"] == "completed_bars_only"
    assert "analysis_window" not in result


def test_compact_error_drops_empty_success_shape_and_duplicate_symbol() -> None:
    payload = {
        "success": False,
        "error": "Unknown symbol NOTREAL.",
        "error_code": "invalid_symbol",
        "details": {
            "symbol": "NOTREAL",
            "search_hint": "Use symbols_list to find the broker symbol.",
        },
        "remediation": "Use symbols_list to find the broker symbol.",
        "items": [],
        "forming_candle_status": "none",
    }

    result = shape_public_tool_output(
        payload,
        tool_name="symbols_describe",
        detail="compact",
    )

    assert result == {
        "success": False,
        "error": "Unknown symbol NOTREAL.",
        "error_code": "invalid_symbol",
        "remediation": "Use symbols_list to find the broker symbol.",
    }


def test_error_warnings_use_the_same_structured_schema_as_success() -> None:
    payload = {
        "success": False,
        "error": "Preview blocked.",
        "error_code": "preview_blocked",
        "warnings": ["No order was sent."],
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

    expected = [{"code": "data_warning", "message": "No order was sent."}]
    assert compact["warnings"] == expected
    assert full["warnings"] == expected


def test_wait_event_budget_error_keeps_only_actionable_compact_state() -> None:
    payload = {
        "success": False,
        "error": "The next candle boundary is beyond the inferred timeframe wait budget.",
        "error_code": "wait_budget_exceeded",
        "request_id": "63c928cdb2c9",
        "operation": "wait_event",
        "buffer_seconds": 1,
        "next_candle_close_utc": "2026-08-30T19:00:00Z",
        "next_candle_close_server": "2026-08-30T22:00:00+03:00",
        "server_timezone": "Europe/Nicosia",
        "server_utc_offset": "+03:00",
        "completed": False,
        "status": "wait_budget_exceeded",
        "not_waited": True,
        "remaining_seconds": 654,
        "wait_mode": "timeframe_boundary",
        "remediation": "Retry closer to the boundary or choose a shorter timeframe.",
        "symbol": "BTCUSD",
        "bid": 79061.3,
        "ask": 79066.3,
        "quote_time": "2026-08-30T18:49:06Z",
        "data_age_seconds": 0.162,
        "data_stale": False,
        "usable_for_live_trading": True,
        "source": {"provider": "mt5", "server": "ICMarketsSC-MT5-2"},
    }

    result = shape_public_tool_output(
        payload,
        tool_name="wait_event",
        detail="compact",
    )

    assert result == {
        "success": False,
        "error": "The next candle boundary is beyond the inferred timeframe wait budget.",
        "error_code": "wait_budget_exceeded",
        "request_id": "63c928cdb2c9",
        "symbol": "BTCUSD",
        "next_candle_close_utc": "2026-08-30T19:00:00Z",
        "remaining_seconds": 654,
        "remediation": "Retry closer to the boundary or choose a shorter timeframe.",
    }


def test_wait_event_error_retains_diagnostics_at_full_detail() -> None:
    payload = {
        "success": False,
        "error": "The next candle boundary is beyond the inferred timeframe wait budget.",
        "error_code": "wait_budget_exceeded",
        "request_id": "63c928cdb2c9",
        "operation": "wait_event",
        "next_candle_close_server": "2026-08-30T22:00:00+03:00",
        "source": {"provider": "mt5", "server": "ICMarketsSC-MT5-2"},
    }

    result = shape_public_tool_output(
        payload,
        tool_name="wait_event",
        detail="full",
    )

    assert result["request_id"] == "63c928cdb2c9"
    assert result["operation"] == "wait_event"
    assert result["next_candle_close_server"] == "2026-08-30T22:00:00+03:00"
    assert result["source"]["server"] == "ICMarketsSC-MT5-2"


def test_task_list_compact_keeps_actionable_task_state() -> None:
    payload = {
        "success": True,
        "detail": "compact",
        "count": 1,
        "row_key": "tasks",
        "runtime": {"workers": {"active_futures": 0}},
        "pagination": {"has_more": False, "returned": 1, "total": 1},
        "tasks": [
            {
                "task_id": "abc",
                "method": "mlf_rf",
                "adapter_method": "mlf_rf",
                "data_scope": "EURUSD_H1",
                "status": "completed",
                "timezone": "UTC",
                "cancel_requested": False,
                "created_at": "2026-08-29T01:00:00Z",
                "heartbeat_at": "2026-08-29T01:00:03Z",
                "elapsed_seconds": 3,
                "pid": 123,
                "progress_fraction": 1.0,
                "model_id": "mlf_rf/EURUSD_H1/123",
                "model_store_status": "present",
            }
        ],
    }

    result = shape_public_tool_output(
        payload,
        tool_name="forecast_task_list",
        detail="compact",
    )

    assert result == {
        "success": True,
        "tasks": [
            {
                "task_id": "abc",
                "method": "mlf_rf",
                "data_scope": "EURUSD_H1",
                "status": "completed",
                "progress_fraction": 1.0,
                "model_id": "mlf_rf/EURUSD_H1/123",
                "model_store_status": "present",
            }
        ],
    }


def test_full_analysis_moves_processing_and_request_under_meta() -> None:
    payload = {
        "success": True,
        "result": {"score": 0.9},
        "query_applied": {"timeframe": "H1"},
        "processing_pipeline": ["fetch", "score"],
        "data_as_of": "2026-08-29T01:00:00Z",
        "timezone": "UTC",
        "data_stale": False,
        "data_age_seconds": 10,
        "source": {"provider": "mt5", "server": "Demo"},
        "units": {"score": "ratio"},
    }

    result = shape_public_tool_output(
        payload,
        tool_name="market_relative_strength",
        detail="full",
    )

    assert "query_applied" not in result
    assert "processing_pipeline" not in result
    assert "data_as_of" not in result
    assert "source" not in result
    assert "units" not in result
    assert result["meta"]["request"]["query_applied"] == {"timeframe": "H1"}
    assert result["meta"]["processing"]["processing_pipeline"] == [
        "fetch",
        "score",
    ]
    assert result["meta"]["source"]["provider"] == "mt5"
    assert result["meta"]["units"] == {"score": "ratio"}


def test_compact_seasonality_keeps_elapsed_duration_basis() -> None:
    result = shape_public_tool_output(
        {
            "success": True,
            "items": [
                {
                    "period_bars": 72,
                    "period_duration": "5 days",
                    "nominal_period_duration": "3 days",
                    "period_duration_basis": "median_observed_timestamp_lag",
                    "period_duration_observed_range": {
                        "min": "3 days",
                        "max": "5 days",
                    },
                }
            ],
        },
        tool_name="seasonality_detect",
        detail="compact",
    )

    assert result["items"][0]["nominal_period_duration"] == "3 days"
    assert result["items"][0]["period_duration_basis"] == (
        "median_observed_timestamp_lag"
    )
    assert result["items"][0]["period_duration_observed_range"] == {
        "min": "3 days",
        "max": "5 days",
    }


def test_compact_regime_keeps_calibration_confidence() -> None:
    result = shape_public_tool_output(
        {
            "success": True,
            "symbol": "EURUSD",
            "method": "hmm",
            "calibration": {
                "confidence": (
                    "model or heuristic assignment score, not historical hit rate"
                ),
                "note": "Regime labels describe observed state.",
            },
        },
        tool_name="regime_detect",
        detail="compact",
    )

    assert result["calibration"] == {
        "confidence": "model or heuristic assignment score, not historical hit rate"
    }


def test_compact_barrier_optimize_does_not_warn_quote_not_live_when_quote_is_live() -> None:
    result = shape_public_tool_output(
        {
            "success": True,
            "usable_for_live_trading": False,
            "data_stale": False,
            "freshness_state": "live",
            "data_age_seconds": 2.2,
            "execution_blockers": ["optimizer_non_viable"],
        },
        tool_name="forecast_barrier_optimize",
        detail="compact",
    )

    codes = [warning.get("code") for warning in result.get("warnings") or []]
    assert "quote_not_live" not in codes
    assert "optimizer_non_viable" in codes


def test_compact_trade_stress_test_omits_freshness_when_not_applicable() -> None:
    result = shape_public_tool_output(
        {
            "success": True,
            "empty": True,
            "status": "no_open_positions",
            "mark_freshness_status": "not_applicable",
            "data_stale": None,
        },
        tool_name="trade_stress_test",
        detail="compact",
    )

    assert "data_stale" not in result
    assert "warnings" not in result


def test_compact_patterns_keeps_confidence_scope() -> None:
    result = shape_public_tool_output(
        {
            "success": True,
            "pattern_confidence": 0.149,
            "confidence_basis": (
                "match_score is per-pattern heuristic fit; pattern_confidence "
                "is aggregate directional strength"
            ),
            "status_scope": "historical_window",
        },
        tool_name="patterns_detect",
        detail="compact",
    )

    assert result["status_scope"] == "historical_window"
    assert "aggregate directional strength" in result["confidence_basis"]
