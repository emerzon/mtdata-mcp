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
        "pagination": {"has_more": True, "next_offset": 2},
        "unavailable": [
            {"method": "nbeatsx", "reason": "Requires neuralforecast"}
        ],
    }


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
