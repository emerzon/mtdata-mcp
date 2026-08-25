from mtdata.core._mcp_tools import _select_output_fields


def test_output_fields_supports_dotted_nested_paths() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "details": {"time": "2026-07-14T15:00Z", "digits": 5, "trade_mode": "full"},
    }

    result = _select_output_fields(payload, "details.time,details.digits")

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "details": {"time": "2026-07-14T15:00Z", "digits": 5},
    }


def test_output_fields_partial_projection_is_explicit() -> None:
    payload = {"success": True, "symbol": "EURUSD", "details": {"digits": 5}}

    result = _select_output_fields(payload, "symbol,details.missing")

    assert result["success"] is True
    assert result["symbol"] == "EURUSD"
    assert result["unresolved_output_fields"] == ["details.missing"]
    assert result["valid_output_fields"] == ["details"]
    assert result["output_fields_status"] == "partial"
    assert "detail full" in result["remediation"]


def test_output_fields_total_miss_returns_structured_error() -> None:
    payload = {"success": True, "value": 1}

    result = _select_output_fields(payload, "missing")

    assert result == {
        "success": False,
        "error": "None of the requested output fields are available in this response contract.",
        "error_code": "output_fields_unresolved",
        "unresolved_output_fields": ["missing"],
        "valid_output_fields": ["value"],
        "output_fields_status": "failed",
        "remediation": (
            "Choose one or more paths from valid_output_fields and retry "
            "--output-fields. valid_output_fields lists paths present in this "
            "response; compact detail omits some diagnostics, so retry with "
            "--detail full if a declared field is absent."
        ),
    }


def test_output_fields_resolves_declared_path_through_empty_collection() -> None:
    payload = {
        "success": True,
        "count": 0,
        "items": [],
        "empty": True,
    }

    result = _select_output_fields(
        payload,
        "items.symbol",
        tool_name="trade_get_open",
    )

    assert result == {"success": True, "count": 0, "items": []}


def test_output_fields_does_not_advertise_absent_compact_row_paths() -> None:
    payload = {
        "success": True,
        "count": 1,
        "items": [{"ticket": 1, "symbol": "EURUSD", "magic": 7, "comment": "audit"}],
    }

    result = _select_output_fields(
        payload,
        "items.ticket,items.magic,items.comment,items.raw",
        tool_name="trade_history",
    )

    assert result["items"] == [
        {"ticket": 1, "magic": 7, "comment": "audit"}
    ]
    assert result["unresolved_output_fields"] == ["items.raw"]
    assert "items.magic" in result["valid_output_fields"]
    assert "items.comment" in result["valid_output_fields"]
    assert "items.raw" not in result["valid_output_fields"]
    assert result["output_fields_status"] == "partial"
    assert "detail full" in result["remediation"]


def test_output_fields_rejects_unknown_path_through_empty_collection() -> None:
    payload = {"success": True, "count": 0, "items": [], "empty": True}

    result = _select_output_fields(
        payload,
        "items.symbl",
        tool_name="trade_get_open",
    )

    assert result["success"] is False
    assert result["error_code"] == "output_fields_unresolved"
    assert result["unresolved_output_fields"] == ["items.symbl"]
    assert "items.symbol" in result["valid_output_fields"]


def test_output_fields_resolves_canonical_forecast_arrays_from_compact_rows() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "quantity": "price",
        "forecast": [
            {"time": "2026-08-18T02:00Z", "value": 1.15782},
            {"time": "2026-08-18T03:00Z", "value": 1.15801},
        ],
    }

    result = _select_output_fields(payload, "forecast_time,forecast_price")

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "forecast_time": ["2026-08-18T02:00Z", "2026-08-18T03:00Z"],
        "forecast_price": [1.15782, 1.15801],
    }


def test_output_fields_allows_error_field_on_success() -> None:
    payload = {"success": True, "symbol": "EURUSD", "data": [1, 2]}

    result = _select_output_fields(payload, "success,data,error")

    assert result == {"success": True, "symbol": "EURUSD", "data": [1, 2]}


def test_output_fields_keeps_requested_query_context_on_error() -> None:
    payload = {
        "success": False,
        "error": "No data available",
        "error_code": "data_fetch_candles_no_data",
        "query_applied": {
            "resolved_start": "2026-08-12T00:00:00Z",
            "start_bound": "inclusive_day_start",
        },
    }

    result = _select_output_fields(payload, "success,query_applied")

    assert result == {
        "success": False,
        "error": "No data available",
        "error_code": "data_fetch_candles_no_data",
        "query_applied": payload["query_applied"],
    }


def test_output_fields_preserves_complete_error_recovery_envelope() -> None:
    payload = {
        "success": False,
        "error": "Unsupported date range",
        "error_code": "unsupported_date_range",
        "request_id": "abc123",
        "operation": "data_fetch_candles",
        "remediation": "Use a date on or after 1970-01-01.",
        "related_tools": ["data_fetch_candles"],
        "valid_values": {"end": ">= 1970-01-01"},
        "example": "--end 2024-01-01",
        "documentation": "docs/CLI.md",
        "details": {"end": "1960-01-01"},
        "data": [],
    }

    result = _select_output_fields(payload, "success,data")

    assert result == payload


def test_output_fields_does_not_inject_units_for_selected_values() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "bid": 1.1,
        "ask": 1.2,
        "units": {"bid": "price", "ask": "price"},
    }

    result = _select_output_fields(payload, "bid,ask")

    assert result == {"success": True, "symbol": "EURUSD", "bid": 1.1, "ask": 1.2}


def test_output_fields_keeps_row_positions_for_sparse_nested_fields() -> None:
    payload = {
        "success": True,
        "count": 3,
        "items": [
            {
                "event": "CPI",
                "date": "2026-08-26T12:30:00Z",
                "country_code": "US",
                "country_attribution": "inferred",
            },
            {
                "event": "GDP Growth Rate QoQ 2nd Est",
                "date": "2026-08-26T12:30:00Z",
                "country_attribution": "unknown",
            },
            {
                "event": "ISM Manufacturing PMI",
                "date": "2026-09-01T14:00:00Z",
                "country_code": "US",
                "country_attribution": "inferred",
            },
        ],
    }

    result = _select_output_fields(
        payload,
        "items.event,items.date,items.country_code,items.country_attribution",
    )

    assert result["success"] is True
    assert result["count"] == 3
    assert result["items"] == [
        {
            "event": "CPI",
            "date": "2026-08-26T12:30:00Z",
            "country_code": "US",
            "country_attribution": "inferred",
        },
        {
            "event": "GDP Growth Rate QoQ 2nd Est",
            "date": "2026-08-26T12:30:00Z",
            "country_attribution": "unknown",
        },
        {
            "event": "ISM Manufacturing PMI",
            "date": "2026-09-01T14:00:00Z",
            "country_code": "US",
            "country_attribution": "inferred",
        },
    ]
    assert "unresolved_output_fields" not in result


def test_output_fields_prefers_top_level_quote_values_over_nested_diagnostics() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "bid": 1.1,
        "ask": 1.1002,
        "quote_source_conflict": {
            "symbol_info_tick": {"bid": 1.0999, "ask": 1.1001},
            "stream_tick": {"bid": 1.1, "ask": 1.1002},
        },
    }

    result = _select_output_fields(payload, "bid,ask")

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "bid": 1.1,
        "ask": 1.1002,
    }


def test_output_fields_projects_bare_fields_from_row_collections() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "row_key": "data",
        "count": 2,
        "data": [
            {"time": 1, "close": 1.1, "open": 1.0},
            {"time": 2, "close": 1.2, "open": 1.1},
        ],
    }

    result = _select_output_fields(payload, "time,close")

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "count": 2,
        "data": [{"time": 1, "close": 1.1}, {"time": 2, "close": 1.2}],
    }


def test_output_fields_uses_dotted_paths_for_row_collections() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "data": [{"time": 1, "close": 1.1}, {"time": 2, "close": 1.2}],
    }

    result = _select_output_fields(payload, "data.close")

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "data": [{"close": 1.1}, {"close": 1.2}],
    }


def test_output_fields_preserves_pagination_metadata() -> None:
    payload = {
        "success": True,
        "tools": [{"name": "forecast_generate", "description": "Forecast"}],
        "pagination": {"offset": 0, "limit": 1, "returned": 1, "total": 8},
    }

    result = _select_output_fields(payload, "tools.name")

    assert result == {
        "success": True,
        "tools": [{"name": "forecast_generate"}],
        "pagination": {"offset": 0, "limit": 1, "returned": 1, "total": 8},
    }


def test_output_fields_keeps_domain_remediation_on_partial_error() -> None:
    payload = {
        "success": False,
        "error": "No usable quote data for DE40.",
        "error_code": "market_ticker_quote_unavailable",
        "remediation": "Ensure the symbol has a recent tick in Market Watch.",
        "details": {"symbol": "DE40"},
    }

    result = _select_output_fields(payload, "success,error,bid,ask")

    assert result["success"] is False
    assert result["error"] == payload["error"]
    assert result["remediation"] == payload["remediation"]
    assert result["output_fields_status"] == "partial"
    assert result["output_fields_remediation"]
    assert result["valid_output_fields"]
    assert "details" in result["valid_output_fields"]


def test_output_fields_preserves_history_truncation_warnings() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "data": [{"time": 1, "bid": 1.1}],
        "history_window_truncated": True,
        "history_window_limit_days": 30,
        "history_window_floor": "2026-07-16T00:00Z",
        "effective_start": "2026-07-16T00:00Z",
        "warnings": ["Requested start was outside the tick-history budget."],
    }

    result = _select_output_fields(payload, "data")

    assert result == payload
